using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2ToMdxConverter
{
    private const uint ClassicMdxVersion = 1300u;
    private const int ModlNameSizeBytes = 0x50;
    private const int TexsPathSizeBytes = 0x104;
    private const int TrackArrayReferenceSizeBytes = 0x08;
    private const uint NoGeosetBinding = uint.MaxValue;

    public static void Convert(string inputPath, string skinPath, string outputPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(skinPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        M2GeometryDocument geometry = M2GeometryReader.Read(inputPath);
        M2SkinDocument skin = M2SkinReader.Read(skinPath);
        IReadOnlyDictionary<string, M2ExternalAnimationDocument> externalAnimations = LoadLocalExternalAnimations(geometry.Model);
        byte[] converted = Convert(geometry, skin, rewrittenTexturePaths: null, externalAnimations);

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
        return Convert(geometry, skin, rewrittenTexturePaths: null);
    }

    public static byte[] Convert(
        M2GeometryDocument geometry,
        M2SkinDocument skin,
        IReadOnlyDictionary<string, string>? rewrittenTexturePaths)
    {
        return Convert(geometry, skin, rewrittenTexturePaths, externalAnimations: null);
    }

    public static byte[] Convert(
        M2GeometryDocument geometry,
        M2SkinDocument skin,
        IReadOnlyDictionary<string, string>? rewrittenTexturePaths,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        ArgumentNullException.ThrowIfNull(geometry);
        ArgumentNullException.ThrowIfNull(skin);

        IReadOnlyList<M2GeometryTexture> textures = RewriteTextures(geometry.Textures, rewrittenTexturePaths);
        ushort[] indices = BuildTriangleIndices(geometry, skin);
        MaterialLayerInfo? materialLayer = TryBuildMaterialLayer(textures, geometry, skin);
        int geosetMaterialId = materialLayer is null ? -1 : 0;
        uint selectionGroup = skin.Submeshes.Count == 0 ? 0u : skin.Submeshes[0].SkinSectionId;
        string modelName = ResolveModelName(geometry.Model);
        int[] sequenceStartTimes = BuildSequenceStartTimes(geometry.Model.Sequences);

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.ASCII, leaveOpen: true);

        writer.Write(Encoding.ASCII.GetBytes("MDLX"));
        WriteChunk(writer, "VERS", payload => payload.Write(ClassicMdxVersion));
        WriteChunk(writer, "MODL", payload => WriteModl(payload, modelName, geometry.Model));

        if (geometry.Model.Sequences.Count > 0)
            WriteChunk(writer, "SEQS", payload => WriteSeqs(payload, geometry.Model.Sequences, sequenceStartTimes));

        if (geometry.Model.GlobalLoops.Count > 0)
            WriteChunk(writer, "GLBS", payload => WriteGlbs(payload, geometry.Model.GlobalLoops));

        if (textures.Count > 0)
            WriteChunk(writer, "TEXS", payload => WriteTexs(payload, textures));

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
            WriteChunk(writer, "BONE", payload => WriteBone(payload, geometry.Model, sequenceStartTimes, externalAnimations));
            WriteChunk(writer, "PIVT", payload => WritePivt(payload, geometry.Model.Bones));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static IReadOnlyList<M2GeometryTexture> RewriteTextures(
        IReadOnlyList<M2GeometryTexture> textures,
        IReadOnlyDictionary<string, string>? rewrittenTexturePaths)
    {
        if (textures.Count == 0 || rewrittenTexturePaths is null || rewrittenTexturePaths.Count == 0)
            return textures;

        List<M2GeometryTexture>? rewritten = null;
        for (int index = 0; index < textures.Count; index++)
        {
            M2GeometryTexture texture = textures[index];
            string? filename = texture.Filename;
            if (string.IsNullOrWhiteSpace(filename))
                continue;

            string normalized = NormalizeVirtualPath(filename);
            if (!rewrittenTexturePaths.TryGetValue(normalized, out string? mappedPath)
                || string.Equals(mappedPath, normalized, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            rewritten ??= [.. textures];
            rewritten[index] = new M2GeometryTexture(mappedPath, texture.ReplaceableId, texture.Flags);
        }

        return rewritten ?? textures;
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

    private static MaterialLayerInfo? TryBuildMaterialLayer(
        IReadOnlyList<M2GeometryTexture> textures,
        M2GeometryDocument geometry,
        M2SkinDocument skin)
    {
        if (textures.Count == 0)
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

    private static string NormalizeVirtualPath(string path)
    {
        return path.Replace('/', '\\').Trim().TrimStart('\\');
    }

    private static void WriteModl(BinaryWriter writer, string modelName, M2ModelDocument model)
    {
        WriteFixedAscii(writer, modelName, ModlNameSizeBytes);
        WriteVector3(writer, model.BoundsMin);
        WriteVector3(writer, model.BoundsMax);
        writer.Write((uint)150);
    }

    private static void WriteSeqs(BinaryWriter writer, IReadOnlyList<M2SequenceDefinition> sequences, IReadOnlyList<int> sequenceStartTimes)
    {
        writer.Write((uint)sequences.Count);
        for (int index = 0; index < sequences.Count; index++)
        {
            M2SequenceDefinition sequence = sequences[index];
            int startTime = index < sequenceStartTimes.Count ? sequenceStartTimes[index] : 0;
            WriteFixedAscii(writer, GetAnimationSequenceName(sequence.AnimationId, sequence.VariationIndex), 0x50);
            writer.Write(startTime);
            writer.Write(checked(startTime + (int)Math.Min(sequence.Duration, int.MaxValue - Math.Max(startTime, 0))));
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

    private static void WriteGlbs(BinaryWriter writer, IReadOnlyList<uint> globalLoops)
    {
        foreach (uint duration in globalLoops)
            writer.Write(duration);
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

    private static IReadOnlyDictionary<string, M2ExternalAnimationDocument> LoadLocalExternalAnimations(M2ModelDocument model)
    {
        Dictionary<string, M2ExternalAnimationDocument> animations = new(StringComparer.OrdinalIgnoreCase);
        foreach (string companionPath in EnumerateExternalAnimationPaths(model))
        {
            if (!File.Exists(companionPath))
                continue;

            using MemoryStream stream = new(File.ReadAllBytes(companionPath), writable: false);
            animations[companionPath] = M2AnimationReader.Read(stream, companionPath);
        }

        return animations;
    }

    internal static IReadOnlyList<string> EnumerateExternalAnimationPaths(M2ModelDocument model)
    {
        ArgumentNullException.ThrowIfNull(model);

        HashSet<string> paths = new(StringComparer.OrdinalIgnoreCase);
        for (int sequenceIndex = 0; sequenceIndex < model.Sequences.Count; sequenceIndex++)
        {
            int sourceSequenceIndex = ResolveTrackSourceSequenceIndex(model, sequenceIndex);
            if (sourceSequenceIndex < 0 || sourceSequenceIndex >= model.Sequences.Count)
                continue;

            M2SequenceDefinition sequence = model.Sequences[sourceSequenceIndex];
            if (!sequence.UsesExternalAnimationFile)
                continue;

            paths.Add(model.Identity.BuildAnimationPath(sequence.AnimationId, sequence.VariationIndex));
        }

        return [.. paths];
    }

    private static void WriteBone(
        BinaryWriter writer,
        M2ModelDocument model,
        IReadOnlyList<int> sequenceStartTimes,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        writer.Write((uint)model.Bones.Count);
        foreach (M2BoneDefinition bone in model.Bones)
        {
            WriteSizedBlock(writer, boneWriter =>
            {
                WriteFixedAscii(boneWriter, $"Bone{bone.Index}", 0x50);
                boneWriter.Write(bone.Index);
                boneWriter.Write((int)bone.ParentBone);
                boneWriter.Write(bone.Flags);

                WriteVector3Track(boneWriter, "KGTR", model, bone.TranslationTrack, sequenceStartTimes, externalAnimations);
                WriteQuaternionTrack(boneWriter, "KGRT", model, bone.RotationTrack, sequenceStartTimes, externalAnimations);
                WriteVector3Track(boneWriter, "KGSC", model, bone.ScalingTrack, sequenceStartTimes, externalAnimations);
            });
            writer.Write(NoGeosetBinding);
            writer.Write(NoGeosetBinding);
        }
    }

    private static int[] BuildSequenceStartTimes(IReadOnlyList<M2SequenceDefinition> sequences)
    {
        int[] startTimes = new int[sequences.Count];
        int currentStart = 0;
        for (int index = 0; index < sequences.Count; index++)
        {
            startTimes[index] = currentStart;
            int duration = checked((int)Math.Min(sequences[index].Duration, int.MaxValue));
            currentStart = checked(currentStart + Math.Max(duration, 0));
        }

        return startTimes;
    }

    private static void WriteVector3Track(
        BinaryWriter writer,
        string tag,
        M2ModelDocument model,
        M2TrackDefinition<Vector3> track,
        IReadOnlyList<int> sequenceStartTimes,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        List<Vector3TrackKeyframe> keys = BuildVector3TrackKeyframes(model, track, sequenceStartTimes, externalAnimations);
        if (keys.Count == 0)
            return;

        WriteTrackHeader(writer, tag, keys.Count, track.Interpolation, track.GlobalSequenceIndex);
        bool usesTangents = TrackUsesTangents(track.Interpolation);
        foreach (Vector3TrackKeyframe key in keys)
        {
            writer.Write(key.Time);
            WriteVector3(writer, key.Value);
            if (!usesTangents)
                continue;

            WriteVector3(writer, key.InTangent);
            WriteVector3(writer, key.OutTangent);
        }
    }

    private static void WriteQuaternionTrack(
        BinaryWriter writer,
        string tag,
        M2ModelDocument model,
        M2TrackDefinition<M2CompQuaternion> track,
        IReadOnlyList<int> sequenceStartTimes,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        List<QuaternionTrackKeyframe> keys = BuildQuaternionTrackKeyframes(model, track, sequenceStartTimes, externalAnimations);
        if (keys.Count == 0)
            return;

        WriteTrackHeader(writer, tag, keys.Count, track.Interpolation, track.GlobalSequenceIndex);
        bool usesTangents = TrackUsesTangents(track.Interpolation);
        foreach (QuaternionTrackKeyframe key in keys)
        {
            writer.Write(key.Time);
            WriteCompressedQuaternion(writer, key.Value);
            if (!usesTangents)
                continue;

            WriteCompressedQuaternion(writer, key.InTangent);
            WriteCompressedQuaternion(writer, key.OutTangent);
        }
    }

    private static void WriteTrackHeader(BinaryWriter writer, string tag, int keyCount, M2TrackInterpolation interpolation, int globalSequenceIndex)
    {
        writer.Write(Encoding.ASCII.GetBytes(tag));
        writer.Write((uint)keyCount);
        writer.Write((uint)interpolation);
        writer.Write(globalSequenceIndex);
    }

    private static List<Vector3TrackKeyframe> BuildVector3TrackKeyframes(
        M2ModelDocument model,
        M2TrackDefinition<Vector3> track,
        IReadOnlyList<int> sequenceStartTimes,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(track);

        byte[] payload = model.RawBytes;
        List<Vector3TrackKeyframe> keyframes = [];

        if (track.UsesGlobalSequence)
        {
            AppendVector3TrackKeyframes(payload, track, sequenceIndex: 0, baseTime: 0, keyframes);
            return keyframes;
        }

        for (int sequenceIndex = 0; sequenceIndex < model.Sequences.Count; sequenceIndex++)
        {
            int sourceSequenceIndex = ResolveTrackSourceSequenceIndex(model, sequenceIndex);
            int baseTime = sequenceIndex < sequenceStartTimes.Count ? sequenceStartTimes[sequenceIndex] : 0;
            byte[]? sequencePayload = ResolveTrackPayload(model, sourceSequenceIndex, externalAnimations);
            if (sequencePayload is null)
                continue;

            AppendVector3TrackKeyframes(sequencePayload, track, sourceSequenceIndex, baseTime, keyframes);
        }

        return keyframes;
    }

    private static List<QuaternionTrackKeyframe> BuildQuaternionTrackKeyframes(
        M2ModelDocument model,
        M2TrackDefinition<M2CompQuaternion> track,
        IReadOnlyList<int> sequenceStartTimes,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(track);

        byte[] payload = model.RawBytes;
        List<QuaternionTrackKeyframe> keyframes = [];

        if (track.UsesGlobalSequence)
        {
            AppendQuaternionTrackKeyframes(payload, track, sequenceIndex: 0, baseTime: 0, keyframes);
            return keyframes;
        }

        for (int sequenceIndex = 0; sequenceIndex < model.Sequences.Count; sequenceIndex++)
        {
            int sourceSequenceIndex = ResolveTrackSourceSequenceIndex(model, sequenceIndex);
            int baseTime = sequenceIndex < sequenceStartTimes.Count ? sequenceStartTimes[sequenceIndex] : 0;
            byte[]? sequencePayload = ResolveTrackPayload(model, sourceSequenceIndex, externalAnimations);
            if (sequencePayload is null)
                continue;

            AppendQuaternionTrackKeyframes(sequencePayload, track, sourceSequenceIndex, baseTime, keyframes);
        }

        return keyframes;
    }

    private static byte[]? ResolveTrackPayload(
        M2ModelDocument model,
        int sourceSequenceIndex,
        IReadOnlyDictionary<string, M2ExternalAnimationDocument>? externalAnimations)
    {
        if (sourceSequenceIndex < 0 || sourceSequenceIndex >= model.Sequences.Count)
            return null;

        M2SequenceDefinition sequence = model.Sequences[sourceSequenceIndex];
        if (!sequence.UsesExternalAnimationFile)
            return model.RawBytes;

        if (externalAnimations is null || externalAnimations.Count == 0)
            return null;

        string companionPath = model.Identity.BuildAnimationPath(sequence.AnimationId, sequence.VariationIndex);
        return externalAnimations.TryGetValue(companionPath, out M2ExternalAnimationDocument? animation)
            ? animation.Payload
            : null;
    }

    private static int ResolveTrackSourceSequenceIndex(M2ModelDocument model, int sequenceIndex)
    {
        int resolvedSequenceIndex = sequenceIndex;
        HashSet<int> visited = [];
        while (resolvedSequenceIndex >= 0 && resolvedSequenceIndex < model.Sequences.Count)
        {
            if (!visited.Add(resolvedSequenceIndex))
                break;

            M2SequenceDefinition sequence = model.Sequences[resolvedSequenceIndex];
            if (!sequence.IsAlias || sequence.AliasNext == ushort.MaxValue)
                break;

            if (sequence.AliasNext >= model.Sequences.Count)
                break;

            resolvedSequenceIndex = sequence.AliasNext;
        }

        return resolvedSequenceIndex;
    }

    private static void AppendVector3TrackKeyframes(
        byte[] payload,
        M2TrackDefinition<Vector3> track,
        int sequenceIndex,
        int baseTime,
        List<Vector3TrackKeyframe> destination)
    {
        if (!TryReadSequenceSlice(payload, track.TimestampArray, track.ValueArray, sequenceIndex, out M2TrackSequenceSlice slice) || !slice.HasData)
            return;

        int keyCount = checked((int)Math.Min(slice.TimestampCount, slice.ValueCount));
        if (keyCount <= 0)
            return;

        int valueStride = GetTrackValueStride(track.Interpolation, scalarSize: 12);
        if (!IsReadable(payload, slice.TimestampOffset, checked(keyCount * sizeof(uint)))
            || !IsReadable(payload, slice.ValueOffset, checked(keyCount * valueStride)))
        {
            return;
        }

        for (int keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            uint timeOffset = checked(slice.TimestampOffset + (uint)(keyIndex * sizeof(uint)));
            uint valueOffset = checked(slice.ValueOffset + (uint)(keyIndex * valueStride));
            int time = checked(baseTime + (int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan((int)timeOffset, sizeof(uint))));
            Vector3 value = ReadVector3Value(payload, valueOffset);
            Vector3 inTangent = value;
            Vector3 outTangent = value;
            if (TrackUsesTangents(track.Interpolation))
            {
                inTangent = ReadVector3Value(payload, valueOffset + 12u);
                outTangent = ReadVector3Value(payload, valueOffset + 24u);
            }

            destination.Add(new Vector3TrackKeyframe(time, value, inTangent, outTangent));
        }
    }

    private static void AppendQuaternionTrackKeyframes(
        byte[] payload,
        M2TrackDefinition<M2CompQuaternion> track,
        int sequenceIndex,
        int baseTime,
        List<QuaternionTrackKeyframe> destination)
    {
        if (!TryReadSequenceSlice(payload, track.TimestampArray, track.ValueArray, sequenceIndex, out M2TrackSequenceSlice slice) || !slice.HasData)
            return;

        int keyCount = checked((int)Math.Min(slice.TimestampCount, slice.ValueCount));
        if (keyCount <= 0)
            return;

        int valueStride = GetTrackValueStride(track.Interpolation, scalarSize: 8);
        if (!IsReadable(payload, slice.TimestampOffset, checked(keyCount * sizeof(uint)))
            || !IsReadable(payload, slice.ValueOffset, checked(keyCount * valueStride)))
        {
            return;
        }

        for (int keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            uint timeOffset = checked(slice.TimestampOffset + (uint)(keyIndex * sizeof(uint)));
            uint valueOffset = checked(slice.ValueOffset + (uint)(keyIndex * valueStride));
            int time = checked(baseTime + (int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan((int)timeOffset, sizeof(uint))));
            Quaternion value = ReadCompQuaternionValue(payload, valueOffset);
            Quaternion inTangent = value;
            Quaternion outTangent = value;
            if (TrackUsesTangents(track.Interpolation))
            {
                inTangent = ReadCompQuaternionValue(payload, valueOffset + 8u);
                outTangent = ReadCompQuaternionValue(payload, valueOffset + 16u);
            }

            destination.Add(new QuaternionTrackKeyframe(time, value, inTangent, outTangent));
        }
    }

    private static Vector3 ReadVector3Value(byte[] payload, uint offset)
    {
        return new Vector3(
            BinaryPrimitives.ReadSingleLittleEndian(payload.AsSpan((int)offset, sizeof(float))),
            BinaryPrimitives.ReadSingleLittleEndian(payload.AsSpan((int)offset + sizeof(float), sizeof(float))),
            BinaryPrimitives.ReadSingleLittleEndian(payload.AsSpan((int)offset + (sizeof(float) * 2), sizeof(float))));
    }

    private static Quaternion ReadCompQuaternionValue(byte[] payload, uint offset)
    {
        short x = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan((int)offset, sizeof(short)));
        short y = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan((int)offset + sizeof(short), sizeof(short)));
        short z = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan((int)offset + (sizeof(short) * 2), sizeof(short)));
        short w = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan((int)offset + (sizeof(short) * 3), sizeof(short)));
        return new M2CompQuaternion(x, y, z, w).ToQuaternion();
    }

    private static void WriteCompressedQuaternion(BinaryWriter writer, Quaternion value)
    {
        Quaternion normalized = Quaternion.Normalize(value);
        int xq = (int)MathF.Round(normalized.X * (1 << 21));
        int yq = (int)MathF.Round(normalized.Y * (1 << 20));
        int zq = (int)MathF.Round(normalized.Z * (1 << 20));
        xq = Math.Clamp(xq, -(1 << 21), (1 << 21) - 1);
        yq = Math.Clamp(yq, -(1 << 20), (1 << 20) - 1);
        zq = Math.Clamp(zq, -(1 << 20), (1 << 20) - 1);

        uint ux = unchecked((uint)xq) & 0x003F_FFFFu;
        uint uy = unchecked((uint)yq) & 0x001F_FFFFu;
        uint uz = unchecked((uint)zq) & 0x001F_FFFFu;

        uint data0 = (uz & 0x001F_FFFFu) | ((uy & 0x0000_07FFu) << 21);
        uint data1 = ((uy >> 11) & 0x0000_03FFu) | (ux << 10);
        writer.Write(data0);
        writer.Write(data1);
    }

    private static bool TryReadSequenceSlice(byte[] payload, M2TrackArrayReference timestampArray, M2TrackArrayReference valueArray, int sequenceIndex, out M2TrackSequenceSlice slice)
    {
        slice = default;
        if (timestampArray.Count == 0 || valueArray.Count == 0)
            return false;

        if (sequenceIndex < 0 || sequenceIndex >= timestampArray.Count || sequenceIndex >= valueArray.Count)
            return false;

        int timestampRefOffset = checked((int)timestampArray.Offset + (sequenceIndex * TrackArrayReferenceSizeBytes));
        int valueRefOffset = checked((int)valueArray.Offset + (sequenceIndex * TrackArrayReferenceSizeBytes));
        if (!IsReadable(payload, (uint)timestampRefOffset, TrackArrayReferenceSizeBytes)
            || !IsReadable(payload, (uint)valueRefOffset, TrackArrayReferenceSizeBytes))
        {
            return false;
        }

        uint timestampCount = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(timestampRefOffset, sizeof(uint)));
        uint timestampOffset = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(timestampRefOffset + 0x04, sizeof(uint)));
        uint valueCount = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(valueRefOffset, sizeof(uint)));
        uint valueOffset = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(valueRefOffset + 0x04, sizeof(uint)));
        slice = new M2TrackSequenceSlice(timestampCount, timestampOffset, valueCount, valueOffset);
        return true;
    }

    private static bool IsReadable(byte[] payload, uint offset, int size)
    {
        return offset <= payload.Length && size >= 0 && offset <= payload.Length - size;
    }

    private static int GetTrackValueStride(M2TrackInterpolation interpolation, int scalarSize)
    {
        return interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier
            ? checked(scalarSize * 3)
            : scalarSize;
    }

    private static bool TrackUsesTangents(M2TrackInterpolation interpolation)
    {
        return interpolation is M2TrackInterpolation.Hermite or M2TrackInterpolation.Bezier;
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

    private readonly record struct Vector3TrackKeyframe(int Time, Vector3 Value, Vector3 InTangent, Vector3 OutTangent);

    private readonly record struct QuaternionTrackKeyframe(int Time, Quaternion Value, Quaternion InTangent, Quaternion OutTangent);

}