using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2ToMdxConverter
{
    private const uint ClassicMdxVersion = 1300u;
    private const int ModlNameSizeBytes = 0x50;
    private const int TexsPathSizeBytes = 0x104;
    private const uint NoGeosetBinding = uint.MaxValue;

    public static void Convert(string inputPath, string skinPath, string outputPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(skinPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        M2GeometryDocument geometry = M2GeometryReader.Read(inputPath);
        M2SkinDocument skin = M2SkinReader.Read(skinPath);
        byte[] converted = Convert(geometry, skin);

        string fullOutputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOutputPath) ?? ".");
        File.WriteAllBytes(fullOutputPath, converted);
    }

    public static byte[] Convert(
        byte[] m2Bytes,
        byte[] skinBytes,
        string sourcePath = "<memory>.m2",
        string skinSourcePath = "<memory>.skin")
    {
        ArgumentNullException.ThrowIfNull(m2Bytes);
        ArgumentNullException.ThrowIfNull(skinBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(skinSourcePath);

        using MemoryStream m2Stream = new(m2Bytes, writable: false);
        using MemoryStream skinStream = new(skinBytes, writable: false);
        M2GeometryDocument geometry = M2GeometryReader.Read(m2Stream, sourcePath);
        M2SkinDocument skin = M2SkinReader.Read(skinStream, skinSourcePath);
        return Convert(geometry, skin);
    }

    public static byte[] Convert(M2GeometryDocument geometry, M2SkinDocument skin)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(skin);

        ushort[] indices = BuildTriangleIndices(geometry, skin);
        MaterialLayerInfo? materialLayer = TryBuildMaterialLayer(geometry, skin);
        int geosetMaterialId = materialLayer is null ? -1 : 0;
        uint selectionGroup = skin.Submeshes.Count == 0 ? 0u : skin.Submeshes[0].SkinSectionId;
        string modelName = ResolveModelName(geometry.Model);

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.ASCII, leaveOpen: true);

        writer.Write(Encoding.ASCII.GetBytes("MDLX"));
        WriteChunk(writer, "VERS", payload => payload.Write(ClassicMdxVersion));
        WriteChunk(writer, "MODL", payload => WriteModl(payload, modelName, geometry.Model));

        if (geometry.Model.Sequences.Count > 0)
            WriteChunk(writer, "SEQS", payload => WriteSeqs(payload, geometry.Model.Sequences));

        if (geometry.Textures.Count > 0)
            WriteChunk(writer, "TEXS", payload => WriteTexs(payload, geometry.Textures));

        if (materialLayer is not null)
            WriteChunk(writer, "MTLS", payload => WriteMtls(payload, materialLayer.Value));

        if (geometry.Vertices.Count > 0 && indices.Length > 0)
        {
            WriteChunk(writer, "GEOS", payload => WriteGeos(
                payload,
                geometry,
                indices,
                geosetMaterialId,
                selectionGroup));
        }

        if (geometry.Model.Bones.Count > 0)
        {
            WriteChunk(writer, "BONE", payload => WriteBone(payload, geometry.Model.Bones));
            WriteChunk(writer, "PIVT", payload => WritePivt(payload, geometry.Model.Bones));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static ushort[] BuildTriangleIndices(M2GeometryDocument geometry, M2SkinDocument skin)
    {
        if (skin.VertexLookup.Count == 0 || skin.TriangleIndices.Count == 0)
            return [];

        List<ushort> remapped = new(skin.TriangleIndices.Count);
        foreach (ushort lookupIndex in skin.TriangleIndices)
        {
            if (lookupIndex >= skin.VertexLookup.Count)
                continue;

            ushort vertexIndex = skin.VertexLookup[lookupIndex];
            if (vertexIndex >= geometry.Vertices.Count)
                continue;

            remapped.Add(vertexIndex);
        }

        int triangleSafeCount = remapped.Count - (remapped.Count % 3);
        if (triangleSafeCount <= 0)
            return [];

        if (triangleSafeCount == remapped.Count)
            return [.. remapped];

        return [.. remapped.Take(triangleSafeCount)];
    }

    private static MaterialLayerInfo? TryBuildMaterialLayer(M2GeometryDocument geometry, M2SkinDocument skin)
    {
        if (geometry.Textures.Count == 0)
            return null;

        int textureId = 0;
        uint blendMode = 0;
        uint flags = 0;

        if (skin.Batches.Count > 0)
        {
            M2SkinBatch batch = skin.Batches[0];
            if (batch.TextureComboIndex != ushort.MaxValue)
            {
                if (batch.TextureComboIndex < geometry.TextureLookup.Count)
                {
                    ushort resolvedTextureId = geometry.TextureLookup[batch.TextureComboIndex].TextureId;
                    if (resolvedTextureId < geometry.Textures.Count)
                        textureId = resolvedTextureId;
                }
                else if (batch.TextureComboIndex < geometry.Textures.Count)
                {
                    textureId = batch.TextureComboIndex;
                }
            }

            if (batch.RenderFlagsIndex < geometry.RenderFlags.Count)
            {
                M2GeometryRenderFlag renderFlag = geometry.RenderFlags[batch.RenderFlagsIndex];
                blendMode = renderFlag.RawBlendMode;
                flags = renderFlag.Flags;
            }
        }

        return new MaterialLayerInfo(
            PriorityPlane: 0,
            BlendMode: blendMode,
            Flags: flags,
            TextureId: textureId,
            TransformId: -1,
            CoordId: 0,
            StaticAlpha: 1.0f);
    }

    private static string ResolveModelName(M2ModelDocument model)
    {
        if (!string.IsNullOrWhiteSpace(model.ModelName))
            return model.ModelName;

        string fileName = Path.GetFileNameWithoutExtension(model.Identity.CanonicalModelPath);
        return string.IsNullOrWhiteSpace(fileName) ? "Converted" : fileName;
    }

    private static void WriteModl(BinaryWriter writer, string modelName, M2ModelDocument model)
    {
        WriteFixedAscii(writer, modelName, ModlNameSizeBytes);
        WriteVector3(writer, model.BoundsMin);
        WriteVector3(writer, model.BoundsMax);
        writer.Write((uint)150);
    }

    private static void WriteSeqs(BinaryWriter writer, IReadOnlyList<M2SequenceDefinition> sequences)
    {
        writer.Write((uint)sequences.Count);
        foreach (M2SequenceDefinition sequence in sequences)
        {
            WriteFixedAscii(writer, GetAnimationSequenceName(sequence.AnimationId, sequence.VariationIndex), 0x50);
            writer.Write(0);
            writer.Write(checked((int)sequence.Duration));
            writer.Write(sequence.MoveSpeed);
            writer.Write(sequence.Flags);
            writer.Write(sequence.Frequency < 0 ? 0f : sequence.Frequency);
            writer.Write(checked((int)Math.Min(sequence.ReplayMinimum, int.MaxValue)));
            writer.Write(checked((int)Math.Min(sequence.ReplayMaximum, int.MaxValue)));
            writer.Write(sequence.BoundsRadius);
            WriteVector3(writer, sequence.BoundsMin);
            WriteVector3(writer, sequence.BoundsMax);
        }
    }

    private static void WriteTexs(BinaryWriter writer, IReadOnlyList<M2GeometryTexture> textures)
    {
        foreach (M2GeometryTexture texture in textures)
        {
            writer.Write(texture.ReplaceableId);
            WriteFixedAscii(writer, texture.Filename ?? string.Empty, TexsPathSizeBytes);
            writer.Write(texture.Flags);
        }
    }

    private static void WriteMtls(BinaryWriter writer, MaterialLayerInfo materialLayer)
    {
        writer.Write(1u);
        writer.Write(0u);
        WriteSizedBlock(writer, materialWriter =>
        {
            materialWriter.Write(materialLayer.PriorityPlane);
            materialWriter.Write(1u);
            WriteSizedBlock(materialWriter, layerWriter =>
            {
                layerWriter.Write(materialLayer.BlendMode);
                layerWriter.Write(materialLayer.Flags);
                layerWriter.Write(materialLayer.TextureId);
                layerWriter.Write(materialLayer.TransformId);
                layerWriter.Write(materialLayer.CoordId);
                layerWriter.Write(materialLayer.StaticAlpha);
            });
        });
    }

    private static void WriteGeos(
        BinaryWriter writer,
        M2GeometryDocument geometry,
        IReadOnlyList<ushort> indices,
        int materialId,
        uint selectionGroup)
    {
        writer.Write(1);
        WriteSizedBlock(writer, geosetWriter =>
        {
            WriteTagAndCount(geosetWriter, "VRTX", geometry.Vertices.Count);
            foreach (M2GeometryVertex vertex in geometry.Vertices)
                WriteVector3(geosetWriter, vertex.Position);

            WriteTagAndCount(geosetWriter, "NRMS", geometry.Vertices.Count);
            foreach (M2GeometryVertex vertex in geometry.Vertices)
                WriteVector3(geosetWriter, vertex.Normal);

            WriteTagAndCount(geosetWriter, "UVAS", 1);
            foreach (M2GeometryVertex vertex in geometry.Vertices)
                WriteVector2(geosetWriter, vertex.TextureCoords0);

            WriteTagAndCount(geosetWriter, "PTYP", 1);
            geosetWriter.Write((byte)4);

            WriteTagAndCount(geosetWriter, "PCNT", 1);
            geosetWriter.Write(indices.Count);

            WriteTagAndCount(geosetWriter, "PVTX", indices.Count);
            foreach (ushort index in indices)
                geosetWriter.Write(index);

            WriteTagAndCount(geosetWriter, "GNDX", geometry.Vertices.Count);
            for (int vertexIndex = 0; vertexIndex < geometry.Vertices.Count; vertexIndex++)
                geosetWriter.Write((byte)0);

            WriteTagAndCount(geosetWriter, "MTGC", 0);
            WriteTagAndCount(geosetWriter, "MATS", 0);
            WriteTagAndCount(geosetWriter, "BIDX", 0);
            WriteTagAndCount(geosetWriter, "BWGT", 0);

            geosetWriter.Write(materialId);
            geosetWriter.Write(unchecked((int)selectionGroup));
            geosetWriter.Write(0u);
            geosetWriter.Write(geometry.Model.BoundsRadius);
            WriteVector3(geosetWriter, geometry.Model.BoundsMin);
            WriteVector3(geosetWriter, geometry.Model.BoundsMax);
            geosetWriter.Write(0);
        });
    }

    private static void WriteBone(BinaryWriter writer, IReadOnlyList<M2BoneDefinition> bones)
    {
        writer.Write((uint)bones.Count);
        foreach (M2BoneDefinition bone in bones)
        {
            WriteSizedBlock(writer, boneWriter =>
            {
                WriteFixedAscii(boneWriter, $"Bone{bone.Index}", 0x50);
                boneWriter.Write(bone.Index);
                boneWriter.Write((int)bone.ParentBone);
                boneWriter.Write(bone.Flags);
            });
            writer.Write(NoGeosetBinding);
            writer.Write(NoGeosetBinding);
        }
    }

    private static void WritePivt(BinaryWriter writer, IReadOnlyList<M2BoneDefinition> bones)
    {
        foreach (M2BoneDefinition bone in bones)
            WriteVector3(writer, bone.Pivot);
    }

    private static void WriteChunk(BinaryWriter writer, string id, Action<BinaryWriter> writePayload)
    {
        writer.Write(Encoding.ASCII.GetBytes(id));

        using MemoryStream payloadStream = new();
        using BinaryWriter payloadWriter = new(payloadStream, Encoding.ASCII, leaveOpen: true);
        writePayload(payloadWriter);
        payloadWriter.Flush();

        byte[] payload = payloadStream.ToArray();
        writer.Write((uint)payload.Length);
        writer.Write(payload);
    }

    private static void WriteSizedBlock(BinaryWriter writer, Action<BinaryWriter> writePayload)
    {
        using MemoryStream payloadStream = new();
        using BinaryWriter payloadWriter = new(payloadStream, Encoding.ASCII, leaveOpen: true);
        writePayload(payloadWriter);
        payloadWriter.Flush();

        byte[] payload = payloadStream.ToArray();
        writer.Write((uint)(payload.Length + sizeof(uint)));
        writer.Write(payload);
    }

    private static void WriteTagAndCount(BinaryWriter writer, string tag, int count)
    {
        writer.Write(Encoding.ASCII.GetBytes(tag));
        writer.Write(count);
    }

    private static void WriteFixedAscii(BinaryWriter writer, string value, int length)
    {
        byte[] bytes = new byte[length];
        int count = Encoding.ASCII.GetBytes(value.AsSpan(0, Math.Min(value.Length, length - 1)), bytes);
        if (count < length)
            bytes[count] = 0;

        writer.Write(bytes);
    }

    private static void WriteVector3(BinaryWriter writer, Vector3 value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
        writer.Write(value.Z);
    }

    private static void WriteVector2(BinaryWriter writer, Vector2 value)
    {
        writer.Write(value.X);
        writer.Write(value.Y);
    }

    private static string GetAnimationSequenceName(ushort animationId, ushort variationIndex)
    {
        string baseName = animationId switch
        {
            0 => "Stand",
            1 => "Death",
            2 => "Spell",
            3 => "Stop",
            4 => "Walk",
            5 => "Run",
            6 => "Dead",
            7 => "Rise",
            8 => "StandWound",
            9 => "CombatWound",
            10 => "CombatCritical",
            11 => "ShuffleLeft",
            12 => "ShuffleRight",
            13 => "WalkBackwards",
            14 => "Stun",
            15 => "HandsClosed",
            16 => "AttackUnarmed",
            17 => "Attack1H",
            18 => "Attack2H",
            19 => "Attack2HL",
            20 => "ParryUnarmed",
            21 => "Parry1H",
            22 => "Parry2H",
            23 => "Parry2HL",
            24 => "ShieldBlock",
            25 => "ReadyUnarmed",
            26 => "Ready1H",
            27 => "Ready2H",
            28 => "Ready2HL",
            29 => "ReadyBow",
            30 => "Dodge",
            31 => "SpellPrecast",
            32 => "SpellCast",
            33 => "SpellCastArea",
            34 => "NPCWelcome",
            35 => "NPCGoodbye",
            36 => "Block",
            37 => "JumpStart",
            38 => "Jump",
            39 => "JumpEnd",
            40 => "Fall",
            41 => "SwimIdle",
            42 => "Swim",
            43 => "SwimLeft",
            44 => "SwimRight",
            45 => "SwimBackwards",
            _ => $"Anim{animationId}",
        };

        return variationIndex == 0 ? baseName : $"{baseName}_{variationIndex}";
    }

    private readonly record struct MaterialLayerInfo(
        int PriorityPlane,
        uint BlendMode,
        uint Flags,
        int TextureId,
        int TransformId,
        int CoordId,
        float StaticAlpha);
}