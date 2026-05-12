using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;
using WowViewer.Core.IO.Mdx;

namespace WowViewer.Core.IO.M2;

public static class MdxToM2Converter
{
    private const uint StrictM2Version = 0x108u;
    private const int HeaderSizeBytes = 0x130;
    private const int SequenceStride = 0x40;
    private const int BoneStride = 0x58;
    private const int VertexStride = 0x30;
    private const int TextureStride = 0x10;
    private const int RenderFlagStride = 0x04;
    private const int LookupStride = sizeof(ushort);
    private const int SkinHeaderSizeBytes = 60;
    private const int SkinBoneEntryStride = 0x04;
    private const int SkinSubmeshStride = 0x30;
    private const int SkinBatchStride = 0x18;

    public static void Convert(string inputPath, string outputPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        string fullInputPath = Path.GetFullPath(inputPath);
        string fullOutputPath = Path.GetFullPath(outputPath);

        MdxSummary summary = MdxSummaryReader.Read(fullInputPath);
        MdxGeometryFile geometry = MdxGeometryReader.Read(fullInputPath);
        MdxToM2ConversionResult result = Convert(summary, geometry, fullOutputPath);

        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputPath) ?? ".");
        File.WriteAllBytes(fullOutputPath, result.ModelBytes);
        File.WriteAllBytes(result.SkinPath, result.SkinBytes);
    }

    public static MdxToM2ConversionResult Convert(byte[] mdxBytes, string sourcePath = "<memory>.mdx")
    {
        ArgumentNullException.ThrowIfNull(mdxBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        using MemoryStream summaryStream = new(mdxBytes, writable: false);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, sourcePath);

        using MemoryStream geometryStream = new(mdxBytes, writable: false);
        MdxGeometryFile geometry = MdxGeometryReader.Read(geometryStream, sourcePath);

        string canonicalOutputPath = M2ModelIdentity.FromPath(sourcePath).CanonicalModelPath;
        return Convert(summary, geometry, canonicalOutputPath);
    }

    public static MdxToM2ConversionResult Convert(MdxSummary summary, MdxGeometryFile geometry, string outputPath)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        M2ModelIdentity identity = M2ModelIdentity.FromPath(outputPath);
        string modelName = ResolveModelName(summary, identity);
        ConvertedModelBounds bounds = ResolveBounds(summary, geometry);
        List<M2SequenceDefinition> sequences = BuildSequences(summary, bounds);
        List<M2BoneDefinition> bones = BuildBones(summary);
        ConvertedGeometry convertedGeometry = BuildGeometry(summary, geometry, bones.Count);

        byte[] modelBytes = BuildModelBytes(
            identity,
            modelName,
            bounds,
            sequences,
            bones,
            convertedGeometry.Vertices,
            convertedGeometry.Textures,
            convertedGeometry.RenderFlags,
            convertedGeometry.TextureLookup,
            convertedGeometry.TextureUnitLookup,
            convertedGeometry.BoneLookup);
        byte[] skinBytes = BuildSkinBytes(convertedGeometry);

        return new MdxToM2ConversionResult(identity.CanonicalModelPath, identity.BuildSkinPath(0), modelBytes, skinBytes);
    }

    private static string ResolveModelName(MdxSummary summary, M2ModelIdentity identity)
    {
        if (!string.IsNullOrWhiteSpace(summary.ModelName))
            return summary.ModelName;

        string fileName = Path.GetFileNameWithoutExtension(identity.CanonicalModelPath);
        return string.IsNullOrWhiteSpace(fileName) ? "Converted" : fileName;
    }

    private static ConvertedModelBounds ResolveBounds(MdxSummary summary, MdxGeometryFile geometry)
    {
        Vector3 min = summary.BoundsMin ?? new Vector3(float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity);
        Vector3 max = summary.BoundsMax ?? new Vector3(float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity);

        foreach (MdxGeosetGeometry geoset in geometry.Geosets)
        {
            foreach (Vector3 vertex in geoset.Vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            if (geoset.BoundsMin is Vector3 geosetMin)
                min = Vector3.Min(min, geosetMin);
            if (geoset.BoundsMax is Vector3 geosetMax)
                max = Vector3.Max(max, geosetMax);
        }

        if (!float.IsFinite(min.X) || !float.IsFinite(max.X))
        {
            min = Vector3.Zero;
            max = Vector3.Zero;
        }

        float radius = 0f;
        Vector3 center = (min + max) * 0.5f;
        foreach (MdxGeosetGeometry geoset in geometry.Geosets)
        {
            foreach (Vector3 vertex in geoset.Vertices)
                radius = Math.Max(radius, Vector3.Distance(center, vertex));

            if (geoset.BoundsRadius is float geosetRadius)
                radius = Math.Max(radius, geosetRadius);
        }

        if (radius <= 0f)
            radius = Vector3.Distance(min, max) * 0.5f;

        return new ConvertedModelBounds(min, max, radius);
    }

    private static List<M2SequenceDefinition> BuildSequences(MdxSummary summary, ConvertedModelBounds bounds)
    {
        List<M2SequenceDefinition> sequences = new(summary.Sequences.Count);
        for (int index = 0; index < summary.Sequences.Count; index++)
        {
            MdxSequenceSummary sequence = summary.Sequences[index];
            int duration = Math.Max(0, sequence.Duration);
            uint blendTime = sequence.BlendTime ?? summary.BlendTime ?? 0u;
            sequences.Add(new M2SequenceDefinition(
                index,
                ClampToUInt16(index),
                0,
                (uint)duration,
                float.IsFinite(sequence.MoveSpeed) ? sequence.MoveSpeed : 0f,
                0u,
                ClampToInt16((int)MathF.Round(sequence.Frequency)),
                (uint)Math.Max(0, sequence.ReplayStart),
                (uint)Math.Max(0, sequence.ReplayEnd),
                ClampToUInt16((int)blendTime),
                ClampToUInt16((int)blendTime),
                sequence.BoundsMin ?? bounds.Min,
                sequence.BoundsMax ?? bounds.Max,
                sequence.BoundsRadius ?? bounds.Radius,
                -1,
                ushort.MaxValue));
        }

        return sequences;
    }

    private static List<M2BoneDefinition> BuildBones(MdxSummary summary)
    {
        List<M2BoneDefinition> bones = new(summary.Bones.Count);
        for (int index = 0; index < summary.Bones.Count; index++)
        {
            MdxBoneSummary bone = summary.Bones[index];
            Vector3 pivot = index < summary.PivotPoints.Count
                ? summary.PivotPoints[index].Position
                : Vector3.Zero;
            bones.Add(new M2BoneDefinition(
                index,
                -1,
                0u,
                bone.HasParent ? ClampToInt16(bone.ParentId) : (short)-1,
                0,
                0u,
                CreateEmptyTrack<Vector3>(),
                CreateEmptyTrack<M2CompQuaternion>(),
                CreateEmptyTrack<Vector3>(),
                pivot));
        }

        return bones;
    }

    private static ConvertedGeometry BuildGeometry(MdxSummary summary, MdxGeometryFile geometry, int boneCount)
    {
        List<ConvertedTexture> textures = BuildTextures(summary, geometry.GeosetCount > 0);
        List<ConvertedVertex> vertices = [];
        List<ushort> triangleIndices = [];
        List<M2SkinBoneEntry> boneEntries = [];
        List<M2SkinSubmesh> submeshes = [];
        List<M2SkinBatch> batches = [];
        List<ConvertedRenderFlag> renderFlags = [];
        List<ushort> textureLookup = [];
        List<ushort> textureUnitLookup = [];
        List<ushort> boneLookup = [];

        int runningVertexStart = 0;
        int runningIndexStart = 0;
        for (int geosetIndex = 0; geosetIndex < geometry.Geosets.Count; geosetIndex++)
        {
            MdxGeosetGeometry geoset = geometry.Geosets[geosetIndex];
            if (geoset.VertexCount == 0)
                continue;

            if (runningVertexStart + geoset.VertexCount > ushort.MaxValue)
                throw new InvalidDataException($"MDX geoset '{geosetIndex}' would exceed the minimal converter vertex limit of {ushort.MaxValue}.");

            if (runningIndexStart + geoset.IndexCount > ushort.MaxValue)
                throw new InvalidDataException($"MDX geoset '{geosetIndex}' would exceed the minimal converter index limit of {ushort.MaxValue}.");

            List<ushort> geosetBones = [];
            for (int vertexIndex = 0; vertexIndex < geoset.VertexCount; vertexIndex++)
            {
                BoneAssignment assignment = ResolveBoneAssignment(geoset, vertexIndex, boneCount);
                vertices.Add(new ConvertedVertex(
                    geoset.Vertices[vertexIndex],
                    vertexIndex < geoset.NormalCount ? geoset.Normals[vertexIndex] : Vector3.UnitZ,
                    vertexIndex < geoset.PrimaryUvCount ? geoset.PrimaryUvSet[vertexIndex] : Vector2.Zero,
                    Vector2.Zero,
                    assignment));
                boneEntries.Add(new M2SkinBoneEntry(assignment.Bone0, assignment.Bone1, assignment.Bone2, assignment.Bone3));
                AddUniqueBone(geosetBones, assignment.Bone0, boneCount);
            }

            foreach (ushort index in geoset.Indices)
                triangleIndices.Add(checked((ushort)(runningVertexStart + index)));

            int boneComboIndex = boneLookup.Count;
            boneLookup.AddRange(geosetBones);

            submeshes.Add(new M2SkinSubmesh(
                ClampToUInt16((int)Math.Min(geoset.SelectionGroup, ushort.MaxValue)),
                0,
                ClampToUInt16(runningVertexStart),
                ClampToUInt16(geoset.VertexCount),
                ClampToUInt16(runningIndexStart),
                ClampToUInt16(geoset.IndexCount),
                ClampToUInt16(geosetBones.Count),
                ClampToUInt16(boneComboIndex),
                geosetBones.Count > 0 ? (ushort)1 : (ushort)0,
                geosetBones.Count > 0 ? geosetBones[0] : (ushort)0));

            MaterialLayerInfo layer = ResolveMaterialLayer(summary, geoset.MaterialId);
            ushort renderFlagsIndex = ClampToUInt16(renderFlags.Count);
            renderFlags.Add(new ConvertedRenderFlag(
                ClampToUInt16((int)Math.Min(layer.Flags, ushort.MaxValue)),
                ClampToUInt16((int)Math.Min(layer.BlendMode, ushort.MaxValue))));

            ushort textureId = layer.TextureId >= 0 && layer.TextureId < textures.Count
                ? (ushort)layer.TextureId
                : (ushort)0;
            ushort textureLookupIndex = ClampToUInt16(textureLookup.Count);
            textureLookup.Add(textureId);

            ushort textureCoordLookupIndex = ClampToUInt16(textureUnitLookup.Count);
            textureUnitLookup.Add(ClampToUInt16(Math.Max(0, layer.CoordId)));

            batches.Add(new M2SkinBatch(
                0,
                (byte)Math.Clamp(layer.PriorityPlane, byte.MinValue, byte.MaxValue),
                0,
                ClampToUInt16(submeshes.Count - 1),
                ClampToUInt16(geosetIndex),
                -1,
                renderFlagsIndex,
                0,
                1,
                textureLookupIndex,
                textureCoordLookupIndex,
                ushort.MaxValue,
                ushort.MaxValue));

            runningVertexStart += geoset.VertexCount;
            runningIndexStart += geoset.IndexCount;
        }

        return new ConvertedGeometry(vertices, textures, renderFlags, textureLookup, textureUnitLookup, boneLookup, triangleIndices, boneEntries, submeshes, batches);
    }

    private static List<ConvertedTexture> BuildTextures(MdxSummary summary, bool ensureDefaultTexture)
    {
        List<ConvertedTexture> textures = new(summary.Textures.Count == 0 && ensureDefaultTexture ? 1 : summary.Textures.Count);
        foreach (MdxTextureSummary texture in summary.Textures)
            textures.Add(new ConvertedTexture(texture.Path, texture.ReplaceableId, texture.Flags));

        if (textures.Count == 0 && ensureDefaultTexture)
            textures.Add(new ConvertedTexture(null, 0u, 0u));

        return textures;
    }

    private static MaterialLayerInfo ResolveMaterialLayer(MdxSummary summary, int materialId)
    {
        if (materialId >= 0 && materialId < summary.Materials.Count)
        {
            MdxMaterialSummary material = summary.Materials[materialId];
            if (material.LayerCount > 0)
            {
                MdxMaterialLayerSummary layer = material.Layers[0];
                return new MaterialLayerInfo(material.PriorityPlane, layer.BlendMode, layer.Flags, layer.TextureId, layer.CoordId);
            }
        }

        if (summary.Materials.Count > 0 && summary.Materials[0].LayerCount > 0)
        {
            MdxMaterialSummary material = summary.Materials[0];
            MdxMaterialLayerSummary layer = material.Layers[0];
            return new MaterialLayerInfo(material.PriorityPlane, layer.BlendMode, layer.Flags, layer.TextureId, layer.CoordId);
        }

        return new MaterialLayerInfo(0, 0u, 0u, 0, 0);
    }

    private static BoneAssignment ResolveBoneAssignment(MdxGeosetGeometry geoset, int vertexIndex, int boneCount)
    {
        if (boneCount <= 0)
            return default;

        int primaryBone = 0;
        if (vertexIndex < geoset.VertexGroupCount)
        {
            int vertexGroup = geoset.VertexGroups[vertexIndex];
            if (vertexGroup >= 0 && vertexGroup < geoset.BoneIndexCount)
                primaryBone = (int)Math.Min(geoset.BoneIndices[vertexGroup], (uint)(boneCount - 1));
            else if (vertexGroup >= 0)
                primaryBone = Math.Min(vertexGroup, boneCount - 1);
        }

        return new BoneAssignment((byte)primaryBone, 0, 0, 0, 255, 0, 0, 0);
    }

    private static void AddUniqueBone(List<ushort> bones, byte boneIndex, int boneCount)
    {
        if (boneCount <= 0)
            return;

        ushort value = ClampToUInt16(Math.Min(boneIndex, boneCount - 1));
        if (!bones.Contains(value))
            bones.Add(value);
    }

    private static byte[] BuildModelBytes(
        M2ModelIdentity identity,
        string modelName,
        ConvertedModelBounds bounds,
        IReadOnlyList<M2SequenceDefinition> sequences,
        IReadOnlyList<M2BoneDefinition> bones,
        IReadOnlyList<ConvertedVertex> vertices,
        IReadOnlyList<ConvertedTexture> textures,
        IReadOnlyList<ConvertedRenderFlag> renderFlags,
        IReadOnlyList<ushort> textureLookup,
        IReadOnlyList<ushort> textureUnitLookup,
        IReadOnlyList<ushort> boneLookup)
    {
        byte[] modelNameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        List<byte[]> textureNameBytes = textures
            .Select(static texture => string.IsNullOrWhiteSpace(texture.Path)
                ? Array.Empty<byte>()
                : Encoding.UTF8.GetBytes(texture.Path + "\0"))
            .ToList();

        int cursor = HeaderSizeBytes;
        int nameOffset = cursor;
        cursor += modelNameBytes.Length;

        int sequenceOffset = Align(cursor, 0x10);
        cursor = sequenceOffset + (sequences.Count * SequenceStride);

        int sequenceLookupOffset = Align(cursor, LookupStride);
        cursor = sequenceLookupOffset + (sequences.Count * LookupStride);

        int boneOffset = Align(cursor, 0x10);
        cursor = boneOffset + (bones.Count * BoneStride);

        int vertexOffset = Align(cursor, 0x10);
        cursor = vertexOffset + (vertices.Count * VertexStride);

        int textureOffset = Align(cursor, 0x10);
        cursor = textureOffset + (textures.Count * TextureStride);

        int[] textureNameOffsets = new int[textures.Count];
        int textureNameCursor = Align(cursor, 0x04);
        for (int index = 0; index < textureNameBytes.Count; index++)
        {
            if (textureNameBytes[index].Length == 0)
            {
                textureNameOffsets[index] = 0;
                continue;
            }

            textureNameOffsets[index] = textureNameCursor;
            textureNameCursor += textureNameBytes[index].Length;
        }

        int renderFlagOffset = Align(textureNameCursor, 0x04);
        cursor = renderFlagOffset + (renderFlags.Count * RenderFlagStride);

        int boneLookupOffset = Align(cursor, LookupStride);
        cursor = boneLookupOffset + (boneLookup.Count * LookupStride);

        int textureLookupOffset = Align(cursor, LookupStride);
        cursor = textureLookupOffset + (textureLookup.Count * LookupStride);

        int textureUnitLookupOffset = Align(cursor, LookupStride);
        cursor = textureUnitLookupOffset + (textureUnitLookup.Count * LookupStride);

        byte[] data = new byte[cursor];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        WriteUInt32(data, 0x04, StrictM2Version);
        WriteUInt32(data, 0x08, (uint)modelNameBytes.Length);
        WriteUInt32(data, 0x0C, (uint)nameOffset);
        WriteUInt32(data, 0x1C, (uint)sequences.Count);
        WriteUInt32(data, 0x20, sequences.Count == 0 ? 0u : (uint)sequenceOffset);
        WriteUInt32(data, 0x24, (uint)sequences.Count);
        WriteUInt32(data, 0x28, sequences.Count == 0 ? 0u : (uint)sequenceLookupOffset);
        WriteUInt32(data, 0x2C, (uint)bones.Count);
        WriteUInt32(data, 0x30, bones.Count == 0 ? 0u : (uint)boneOffset);
        WriteUInt32(data, 0x3C, (uint)vertices.Count);
        WriteUInt32(data, 0x40, vertices.Count == 0 ? 0u : (uint)vertexOffset);
        WriteUInt32(data, 0x44, 1u);
        WriteUInt32(data, 0x50, (uint)textures.Count);
        WriteUInt32(data, 0x54, textures.Count == 0 ? 0u : (uint)textureOffset);
        WriteUInt32(data, 0x70, (uint)renderFlags.Count);
        WriteUInt32(data, 0x74, renderFlags.Count == 0 ? 0u : (uint)renderFlagOffset);
        WriteUInt32(data, 0x78, (uint)boneLookup.Count);
        WriteUInt32(data, 0x7C, boneLookup.Count == 0 ? 0u : (uint)boneLookupOffset);
        WriteUInt32(data, 0x80, (uint)textureLookup.Count);
        WriteUInt32(data, 0x84, textureLookup.Count == 0 ? 0u : (uint)textureLookupOffset);
        WriteUInt32(data, 0x88, (uint)textureUnitLookup.Count);
        WriteUInt32(data, 0x8C, textureUnitLookup.Count == 0 ? 0u : (uint)textureUnitLookupOffset);
        WriteVector3(data, 0xA0, bounds.Min);
        WriteVector3(data, 0xAC, bounds.Max);
        WriteSingle(data, 0xB8, bounds.Radius);

        modelNameBytes.CopyTo(data, nameOffset);

        for (int index = 0; index < sequences.Count; index++)
            WriteSequence(data, sequenceOffset + (index * SequenceStride), sequences[index]);
        for (int index = 0; index < sequences.Count; index++)
            WriteInt16(data, sequenceLookupOffset + (index * LookupStride), (short)index);
        for (int index = 0; index < bones.Count; index++)
            WriteBone(data, boneOffset + (index * BoneStride), bones[index]);
        for (int index = 0; index < vertices.Count; index++)
            WriteVertex(data, vertexOffset + (index * VertexStride), vertices[index]);
        for (int index = 0; index < textures.Count; index++)
            WriteTexture(data, textureOffset + (index * TextureStride), textures[index], textureNameBytes[index], textureNameOffsets[index]);
        for (int index = 0; index < textureNameBytes.Count; index++)
        {
            if (textureNameBytes[index].Length > 0)
                textureNameBytes[index].CopyTo(data, textureNameOffsets[index]);
        }
        for (int index = 0; index < renderFlags.Count; index++)
        {
            int offset = renderFlagOffset + (index * RenderFlagStride);
            WriteUInt16(data, offset + 0x00, renderFlags[index].Flags);
            WriteUInt16(data, offset + 0x02, renderFlags[index].BlendMode);
        }
        for (int index = 0; index < boneLookup.Count; index++)
            WriteUInt16(data, boneLookupOffset + (index * LookupStride), boneLookup[index]);
        for (int index = 0; index < textureLookup.Count; index++)
            WriteUInt16(data, textureLookupOffset + (index * LookupStride), textureLookup[index]);
        for (int index = 0; index < textureUnitLookup.Count; index++)
            WriteUInt16(data, textureUnitLookupOffset + (index * LookupStride), textureUnitLookup[index]);

        return data;
    }

    private static byte[] BuildSkinBytes(ConvertedGeometry geometry)
    {
        int vertexLookupOffset = SkinHeaderSizeBytes;
        int triangleIndexOffset = vertexLookupOffset + (geometry.Vertices.Count * LookupStride);
        int boneEntryOffset = triangleIndexOffset + (geometry.TriangleIndices.Count * LookupStride);
        int submeshOffset = boneEntryOffset + (geometry.BoneEntries.Count * SkinBoneEntryStride);
        int batchOffset = submeshOffset + (geometry.Submeshes.Count * SkinSubmeshStride);
        byte[] data = new byte[batchOffset + (geometry.Batches.Count * SkinBatchStride)];

        Encoding.ASCII.GetBytes("SKIN").CopyTo(data, 0);
        WriteUInt32(data, 0x04, (uint)geometry.Vertices.Count);
        WriteUInt32(data, 0x08, geometry.Vertices.Count == 0 ? 0u : (uint)vertexLookupOffset);
        WriteUInt32(data, 0x0C, (uint)geometry.TriangleIndices.Count);
        WriteUInt32(data, 0x10, geometry.TriangleIndices.Count == 0 ? 0u : (uint)triangleIndexOffset);
        WriteUInt32(data, 0x14, (uint)geometry.BoneEntries.Count);
        WriteUInt32(data, 0x18, geometry.BoneEntries.Count == 0 ? 0u : (uint)boneEntryOffset);
        WriteUInt32(data, 0x1C, (uint)geometry.Submeshes.Count);
        WriteUInt32(data, 0x20, geometry.Submeshes.Count == 0 ? 0u : (uint)submeshOffset);
        WriteUInt32(data, 0x24, (uint)geometry.Batches.Count);
        WriteUInt32(data, 0x28, geometry.Batches.Count == 0 ? 0u : (uint)batchOffset);
        WriteUInt32(data, 0x2C, 0u);

        for (int index = 0; index < geometry.Vertices.Count; index++)
            WriteUInt16(data, vertexLookupOffset + (index * LookupStride), ClampToUInt16(index));
        for (int index = 0; index < geometry.TriangleIndices.Count; index++)
            WriteUInt16(data, triangleIndexOffset + (index * LookupStride), geometry.TriangleIndices[index]);
        for (int index = 0; index < geometry.BoneEntries.Count; index++)
        {
            int offset = boneEntryOffset + (index * SkinBoneEntryStride);
            M2SkinBoneEntry entry = geometry.BoneEntries[index];
            data[offset + 0x00] = entry.Bone0;
            data[offset + 0x01] = entry.Bone1;
            data[offset + 0x02] = entry.Bone2;
            data[offset + 0x03] = entry.Bone3;
        }
        for (int index = 0; index < geometry.Submeshes.Count; index++)
            WriteSubmesh(data, submeshOffset + (index * SkinSubmeshStride), geometry.Submeshes[index]);
        for (int index = 0; index < geometry.Batches.Count; index++)
            WriteBatch(data, batchOffset + (index * SkinBatchStride), geometry.Batches[index]);

        return data;
    }

    private static void WriteSequence(byte[] data, int offset, M2SequenceDefinition sequence)
    {
        WriteUInt16(data, offset + 0x00, sequence.AnimationId);
        WriteUInt16(data, offset + 0x02, sequence.VariationIndex);
        WriteUInt32(data, offset + 0x04, sequence.Duration);
        WriteSingle(data, offset + 0x08, sequence.MoveSpeed);
        WriteUInt32(data, offset + 0x0C, sequence.Flags);
        WriteInt16(data, offset + 0x10, sequence.Frequency);
        WriteUInt32(data, offset + 0x14, sequence.ReplayMinimum);
        WriteUInt32(data, offset + 0x18, sequence.ReplayMaximum);
        WriteUInt16(data, offset + 0x1C, sequence.BlendTimeIn);
        WriteUInt16(data, offset + 0x1E, sequence.BlendTimeOut);
        WriteVector3(data, offset + 0x20, sequence.BoundsMin);
        WriteVector3(data, offset + 0x2C, sequence.BoundsMax);
        WriteSingle(data, offset + 0x38, sequence.BoundsRadius);
        WriteInt16(data, offset + 0x3C, sequence.VariationNext);
        WriteUInt16(data, offset + 0x3E, sequence.AliasNext);
    }

    private static void WriteBone(byte[] data, int offset, M2BoneDefinition bone)
    {
        WriteInt32(data, offset + 0x00, bone.KeyBoneId);
        WriteUInt32(data, offset + 0x04, bone.Flags);
        WriteInt16(data, offset + 0x08, bone.ParentBone);
        WriteUInt16(data, offset + 0x0A, bone.SubmeshId);
        WriteUInt32(data, offset + 0x0C, bone.BoneNameCrc);
        WriteVector3(data, offset + 0x4C, bone.Pivot);
    }

    private static void WriteVertex(byte[] data, int offset, ConvertedVertex vertex)
    {
        WriteVector3(data, offset + 0x00, vertex.Position);
        data[offset + 0x0C] = vertex.Bones.Weight0;
        data[offset + 0x0D] = vertex.Bones.Weight1;
        data[offset + 0x0E] = vertex.Bones.Weight2;
        data[offset + 0x0F] = vertex.Bones.Weight3;
        data[offset + 0x10] = vertex.Bones.Bone0;
        data[offset + 0x11] = vertex.Bones.Bone1;
        data[offset + 0x12] = vertex.Bones.Bone2;
        data[offset + 0x13] = vertex.Bones.Bone3;
        WriteVector3(data, offset + 0x14, vertex.Normal);
        WriteVector2(data, offset + 0x20, vertex.TextureCoords0);
        WriteVector2(data, offset + 0x28, vertex.TextureCoords1);
    }

    private static void WriteTexture(byte[] data, int offset, ConvertedTexture texture, byte[] nameBytes, int nameOffset)
    {
        WriteUInt32(data, offset + 0x00, texture.ReplaceableId);
        WriteUInt32(data, offset + 0x04, texture.Flags);
        WriteUInt32(data, offset + 0x08, (uint)nameBytes.Length);
        WriteUInt32(data, offset + 0x0C, nameBytes.Length == 0 ? 0u : (uint)nameOffset);
    }

    private static void WriteSubmesh(byte[] data, int offset, M2SkinSubmesh submesh)
    {
        WriteUInt16(data, offset + 0x00, submesh.SkinSectionId);
        WriteUInt16(data, offset + 0x02, submesh.Level);
        WriteUInt16(data, offset + 0x04, submesh.VertexStart);
        WriteUInt16(data, offset + 0x06, submesh.VertexCount);
        WriteUInt16(data, offset + 0x08, submesh.IndexStart);
        WriteUInt16(data, offset + 0x0A, submesh.IndexCount);
        WriteUInt16(data, offset + 0x0C, submesh.BoneCount);
        WriteUInt16(data, offset + 0x0E, submesh.BoneComboIndex);
        WriteUInt16(data, offset + 0x10, submesh.BoneInfluences);
        WriteUInt16(data, offset + 0x12, submesh.CenterBoneIndex);
    }

    private static void WriteBatch(byte[] data, int offset, M2SkinBatch batch)
    {
        data[offset + 0x00] = batch.Flags;
        data[offset + 0x01] = batch.PriorityPlane;
        WriteUInt16(data, offset + 0x02, batch.ShaderId);
        WriteUInt16(data, offset + 0x04, batch.SkinSectionIndex);
        WriteUInt16(data, offset + 0x06, batch.GeosetIndex);
        WriteInt16(data, offset + 0x08, batch.ColorIndex);
        WriteUInt16(data, offset + 0x0A, batch.RenderFlagsIndex);
        WriteUInt16(data, offset + 0x0C, batch.MaterialLayer);
        WriteUInt16(data, offset + 0x0E, batch.TextureCount);
        WriteUInt16(data, offset + 0x10, batch.TextureComboIndex);
        WriteUInt16(data, offset + 0x12, batch.TextureCoordComboIndex);
        WriteUInt16(data, offset + 0x14, batch.TransparencyComboIndex);
        WriteUInt16(data, offset + 0x16, batch.TextureAnimationLookupIndex);
    }

    private static M2TrackDefinition<T> CreateEmptyTrack<T>()
    {
        return new M2TrackDefinition<T>(
            M2TrackInterpolation.None,
            -1,
            new M2TrackArrayReference(0u, 0u),
            new M2TrackArrayReference(0u, 0u));
    }

    private static int Align(int value, int alignment)
    {
        int remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private static ushort ClampToUInt16(int value) => (ushort)Math.Clamp(value, ushort.MinValue, ushort.MaxValue);

    private static short ClampToInt16(int value) => (short)Math.Clamp(value, short.MinValue, short.MaxValue);

    private static void WriteUInt32(byte[] data, int offset, uint value) => BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)), value);

    private static void WriteInt32(byte[] data, int offset, int value) => BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(offset, sizeof(int)), value);

    private static void WriteUInt16(byte[] data, int offset, ushort value) => BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)), value);

    private static void WriteInt16(byte[] data, int offset, short value) => BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(offset, sizeof(short)), value);

    private static void WriteSingle(byte[] data, int offset, float value) => WriteInt32(data, offset, BitConverter.SingleToInt32Bits(value));

    private static void WriteVector2(byte[] data, int offset, Vector2 value)
    {
        WriteSingle(data, offset + 0x00, value.X);
        WriteSingle(data, offset + 0x04, value.Y);
    }

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        WriteSingle(data, offset + 0x00, value.X);
        WriteSingle(data, offset + 0x04, value.Y);
        WriteSingle(data, offset + 0x08, value.Z);
    }

    private readonly record struct ConvertedModelBounds(Vector3 Min, Vector3 Max, float Radius);

    private readonly record struct MaterialLayerInfo(int PriorityPlane, uint BlendMode, uint Flags, int TextureId, int CoordId);

    private readonly record struct BoneAssignment(byte Bone0, byte Bone1, byte Bone2, byte Bone3, byte Weight0, byte Weight1, byte Weight2, byte Weight3);

    private readonly record struct ConvertedVertex(Vector3 Position, Vector3 Normal, Vector2 TextureCoords0, Vector2 TextureCoords1, BoneAssignment Bones);

    private readonly record struct ConvertedTexture(string? Path, uint ReplaceableId, uint Flags);

    private readonly record struct ConvertedRenderFlag(ushort Flags, ushort BlendMode);

    private sealed class ConvertedGeometry
    {
        public ConvertedGeometry(
            IReadOnlyList<ConvertedVertex> vertices,
            IReadOnlyList<ConvertedTexture> textures,
            IReadOnlyList<ConvertedRenderFlag> renderFlags,
            IReadOnlyList<ushort> textureLookup,
            IReadOnlyList<ushort> textureUnitLookup,
            IReadOnlyList<ushort> boneLookup,
            IReadOnlyList<ushort> triangleIndices,
            IReadOnlyList<M2SkinBoneEntry> boneEntries,
            IReadOnlyList<M2SkinSubmesh> submeshes,
            IReadOnlyList<M2SkinBatch> batches)
        {
            Vertices = vertices;
            Textures = textures;
            RenderFlags = renderFlags;
            TextureLookup = textureLookup;
            TextureUnitLookup = textureUnitLookup;
            BoneLookup = boneLookup;
            TriangleIndices = triangleIndices;
            BoneEntries = boneEntries;
            Submeshes = submeshes;
            Batches = batches;
        }

        public IReadOnlyList<ConvertedVertex> Vertices { get; }

        public IReadOnlyList<ConvertedTexture> Textures { get; }

        public IReadOnlyList<ConvertedRenderFlag> RenderFlags { get; }

        public IReadOnlyList<ushort> TextureLookup { get; }

        public IReadOnlyList<ushort> TextureUnitLookup { get; }

        public IReadOnlyList<ushort> BoneLookup { get; }

        public IReadOnlyList<ushort> TriangleIndices { get; }

        public IReadOnlyList<M2SkinBoneEntry> BoneEntries { get; }

        public IReadOnlyList<M2SkinSubmesh> Submeshes { get; }

        public IReadOnlyList<M2SkinBatch> Batches { get; }
    }
}

public sealed class MdxToM2ConversionResult
{
    public MdxToM2ConversionResult(string modelPath, string skinPath, byte[] modelBytes, byte[] skinBytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(skinPath);
        ArgumentNullException.ThrowIfNull(modelBytes);
        ArgumentNullException.ThrowIfNull(skinBytes);

        ModelPath = M2ModelIdentity.NormalizePath(modelPath);
        SkinPath = M2ModelIdentity.NormalizePath(skinPath);
        ModelBytes = modelBytes;
        SkinBytes = skinBytes;
    }

    public string ModelPath { get; }

    public string SkinPath { get; }

    public byte[] ModelBytes { get; }

    public byte[] SkinBytes { get; }
}