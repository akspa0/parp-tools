using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2Era100;

/// <summary>
/// Reader for WoW 1.0.0 (build 3980, beta-3) M2/MDX models — version 0x100 with the
/// "classic" header layout (M2Vertex records + M2Division embedded skin profiles).
///
/// All header offsets and element sizes are from the Ghidra static trace of
/// FUN_0071e190 (MD20 parser/relocator):
/// specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md §4.
///
/// CRITICAL: 1.0.0 and 1.12.1 both use version 0x100 but have completely different
/// header layouts. This reader handles 1.0.0; the 1.12.1 reader (M2Era1121ModelReader)
/// handles 1.12.1. The dispatcher distinguishes them via layout validation.
/// </summary>
public static class M2Era100ModelReader
{
    public static M2ModelDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("1.0.0 M2 model reading requires a seekable stream.", nameof(stream));

        byte[] data = ReadAllBytes(stream);
        if (data.Length < M2Era100Constants.DispatchHeaderSizeBytes)
            throw new InvalidDataException($"1.0.0 M2 file '{sourcePath}' is too small to contain a magic+version pair.");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(0, sizeof(uint)));
        if (magic != M2Era100Constants.Md20Magic)
            throw new InvalidDataException($"1.0.0 M2 file '{sourcePath}' does not contain a strict MD20 root.");

        if (data.Length < M2Era100Constants.MinimumHeaderSizeBytes)
            throw new InvalidDataException($"1.0.0 M2 file '{sourcePath}' is too small to contain a 1.0.0 MD20 header (≥ 0x144 bytes).");

        uint rawVersion = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(M2Era100Constants.VersionOffset, sizeof(uint)));
        if (rawVersion != 0x100u)
            throw new NotSupportedException($"1.0.0 M2 file '{sourcePath}' has version 0x{rawVersion:X}. Expected 0x100.");

        return ParseM2(data, sourcePath);
    }

    private static M2ModelDocument ParseM2(byte[] data, string sourcePath)
    {
        uint flags = ReadUInt32At(data, M2Era100Constants.FlagsOffset);
        uint viewCount = ReadUInt32At(data, M2Era100Constants.DivisionCountOffset); // divisions == views
        string? modelName = TryReadName(data, sourcePath);

        List<uint> globalLoops = ReadUInt32Table(data, sourcePath, "globalLoops",
            M2Era100Constants.GlobalLoopCountOffset, M2Era100Constants.GlobalLoopOffsetOffset);

        List<M2SequenceDefinition> sequences = ReadSequences(data, sourcePath);
        List<short> sequenceLookup = ReadInt16Table(data, sourcePath, "sequenceLookup",
            M2Era100Constants.SequenceLookupCountOffset, M2Era100Constants.SequenceLookupOffsetOffset);

        // Bounds (at 0xB4, matching the gap in the 1.0.0 header).
        Vector3 boundsMin = ReadLenientVector3At(data, M2Era100Constants.BoundsOffset, sourcePath, "boundsMin");
        Vector3 boundsMax = ReadLenientVector3At(data, M2Era100Constants.BoundsOffset + 0x0C, sourcePath, "boundsMax");
        float boundsRadius = ReadLenientSingleAt(data, M2Era100Constants.BoundsRadiusOffset, sourcePath, "boundsRadius");

        // --- Geometry: the core of the "empty bounding box" fix ---

        M2Era100Geometry? geometry = ReadGeometry(data, sourcePath);

        // --- Textures ---

        List<M2Era100Texture> textures = ReadTextures(data, sourcePath);
        List<ushort> textureLookup = ReadUInt16Lookup(data, sourcePath, "textureLookup",
            M2Era100Constants.TextureLookupCountOffset, M2Era100Constants.TextureLookupOffsetOffset);

        // Attach texture lookup to geometry if we have it.
        if (geometry != null && textureLookup.Count > 0)
        {
            geometry = new M2Era100Geometry(
                geometry.RenderVertices,
                geometry.Triangles,
                geometry.Sections,
                geometry.Batches,
                textures,
                textureLookup);
        }
        else if (geometry != null)
        {
            geometry = new M2Era100Geometry(
                geometry.RenderVertices,
                geometry.Triangles,
                geometry.Sections,
                geometry.Batches,
                textures,
                geometry.TextureLookup);
        }

        // --- Build the document ---
        // Animation blocks (colors, lights, cameras, ribbons, particles, bones) are read
        // with the correct 1.0.0 strides where feasible; empty lists are provided for
        // blocks whose per-field layout is not yet fully recovered. These can be filled
        // in follow-up slices without changing the geometry contract.

        M2ModelIdentity identity = M2ModelIdentity.FromPath(sourcePath);

        M2ModelDocument document = new(
            identity,
            data,
            "MD20",
            0x100u,
            flags,
            viewCount,
            modelName,
            globalLoops,
            sequences,
            sequenceLookup,
            colors: [],
            textureWeights: [],
            textureTransforms: [],
            lights: [],
            cameras: null,
            boundsMin,
            boundsMax,
            boundsRadius,
            embeddedSkinProfileCount: viewCount,
            embeddedSkinProfileOffset: ReadUInt32At(data, M2Era100Constants.DivisionOffsetOffset),
            bones: null,
            ribbons: null,
            particles: null);

        if (geometry != null)
            document.InlineEra100Geometry = geometry;

        return document;
    }

    // ─── Geometry: M2Vertex + M2Division ─────────────────────────────────────

    private static M2Era100Geometry? ReadGeometry(byte[] data, string sourcePath)
    {
        // Read global M2Vertex[] from header 0x44.
        uint vertexCount = ReadUInt32At(data, M2Era100Constants.VertexCountOffset);
        uint vertexOffset = ReadUInt32At(data, M2Era100Constants.VertexOffsetOffset);
        if (vertexCount == 0 || vertexOffset == 0)
            return null;

        ValidateSpan(vertexCount, vertexOffset, M2Era100Constants.VertexStride, data.Length, sourcePath, "vertices");
        List<M2Era100Vertex> globalVertices = new(checked((int)vertexCount));
        for (int i = 0; i < vertexCount; i++)
        {
            int ofs = checked((int)vertexOffset + (i * M2Era100Constants.VertexStride));
            globalVertices.Add(ReadM2Vertex(data, ofs, sourcePath, i));
        }

        // Read divisions from header 0x4C. Pick division 0 (LOD 0).
        uint divisionCount = ReadUInt32At(data, M2Era100Constants.DivisionCountOffset);
        uint divisionOffset = ReadUInt32At(data, M2Era100Constants.DivisionOffsetOffset);
        if (divisionCount == 0 || divisionOffset == 0)
            return null;

        ValidateSpan(1, divisionOffset, M2Era100Constants.DivisionStride, data.Length, sourcePath, "divisions[0]");

        // Read division 0's internal arrays.
        int divBase = checked((int)divisionOffset);
        uint vtxLookupCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionVertexLookupCountOffset);
        uint vtxLookupOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionVertexLookupOffsetOffset);
        uint indicesCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionIndicesCountOffset);
        uint indicesOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionIndicesOffsetOffset);
        uint sectionsCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionSectionsCountOffset);
        uint sectionsOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionSectionsOffsetOffset);
        uint batchesCount = ReadUInt32At(data, divBase + M2Era100Constants.DivisionBatchesCountOffset);
        uint batchesOfs = ReadUInt32At(data, divBase + M2Era100Constants.DivisionBatchesOffsetOffset);

        // Read vertexLookup (int16[] — local → global vertex index).
        List<ushort> vertexLookup = new();
        if (vtxLookupCount > 0 && vtxLookupOfs > 0)
        {
            ValidateSpan(vtxLookupCount, vtxLookupOfs, sizeof(ushort), data.Length, sourcePath, "division.vertexLookup");
            for (int i = 0; i < vtxLookupCount; i++)
            {
                int ofs = checked((int)vtxLookupOfs + (i * sizeof(ushort)));
                vertexLookup.Add(ReadUInt16At(data, ofs));
            }
        }

        // Read triangle indices (int16[]).
        List<ushort> triangles = new();
        if (indicesCount > 0 && indicesOfs > 0)
        {
            ValidateSpan(indicesCount, indicesOfs, sizeof(ushort), data.Length, sourcePath, "division.indices");
            for (int i = 0; i < indicesCount; i++)
            {
                int ofs = checked((int)indicesOfs + (i * sizeof(ushort)));
                triangles.Add(ReadUInt16At(data, ofs));
            }
        }

        // Read sections (0x20 B each).
        List<M2Era100Section> sections = new();
        if (sectionsCount > 0 && sectionsOfs > 0)
        {
            ValidateSpan(sectionsCount, sectionsOfs, M2Era100Constants.SectionStride, data.Length, sourcePath, "division.sections");
            for (int i = 0; i < sectionsCount; i++)
            {
                int ofs = checked((int)sectionsOfs + (i * M2Era100Constants.SectionStride));
                sections.Add(new M2Era100Section(
                    ReadUInt16At(data, ofs + M2Era100Constants.SectionSubmeshIdOffset),
                    ReadUInt16At(data, ofs + M2Era100Constants.SectionLevelOffset),
                    ReadUInt16At(data, ofs + M2Era100Constants.SectionVertexStartOffset),
                    ReadUInt16At(data, ofs + M2Era100Constants.SectionVertexCountOffset),
                    ReadUInt32At(data, ofs + M2Era100Constants.SectionIndexStartOffset),
                    ReadUInt32At(data, ofs + M2Era100Constants.SectionIndexCountOffset)));
            }
        }

        // Read batches (0x18 B each).
        List<M2Era100Batch> batches = new();
        if (batchesCount > 0 && batchesOfs > 0)
        {
            ValidateSpan(batchesCount, batchesOfs, M2Era100Constants.BatchStride, data.Length, sourcePath, "division.batches");
            for (int i = 0; i < batchesCount; i++)
            {
                int ofs = checked((int)batchesOfs + (i * M2Era100Constants.BatchStride));
                batches.Add(ReadM2Batch(data, ofs));
            }
        }

        // Resolve render vertices: walk vertexLookup → global M2Vertex.
        List<M2Era100Vertex> renderVertices = new(vertexLookup.Count);
        for (int i = 0; i < vertexLookup.Count; i++)
        {
            ushort globalIndex = vertexLookup[i];
            M2Era100Vertex vertex = globalIndex < globalVertices.Count
                ? globalVertices[globalIndex]
                : default;
            renderVertices.Add(vertex);
        }

        return new M2Era100Geometry(
            renderVertices,
            triangles,
            sections,
            batches,
            textures: [],
            textureLookup: []);
    }

    private static M2Era100Vertex ReadM2Vertex(byte[] data, int ofs, string sourcePath, int index)
    {
        Vector3 position = ReadLenientVector3At(data, ofs + M2Era100Constants.VertexPositionOffset, sourcePath, $"vertices[{index}].position");
        Vector3 normal = ReadLenientVector3At(data, ofs + M2Era100Constants.VertexNormalOffset, sourcePath, $"vertices[{index}].normal");
        Vector2 uv0 = new(
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords0Offset + 0x00, sourcePath, $"vertices[{index}].uv0.x"),
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords0Offset + 0x04, sourcePath, $"vertices[{index}].uv0.y"));
        Vector2 uv1 = new(
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords1Offset + 0x00, sourcePath, $"vertices[{index}].uv1.x"),
            ReadLenientSingleAt(data, ofs + M2Era100Constants.VertexTexCoords1Offset + 0x04, sourcePath, $"vertices[{index}].uv1.y"));

        // Bone weights/indices: 4 bytes each, packed as uint8[4].
        uint weightsPacked = ReadUInt32At(data, ofs + M2Era100Constants.VertexBoneWeightsOffset);
        uint indicesPacked = ReadUInt32At(data, ofs + M2Era100Constants.VertexBoneIndicesOffset);

        return new M2Era100Vertex(
            position, normal, uv0, uv1,
            (byte)(weightsPacked & 0xFF),
            (byte)((weightsPacked >> 8) & 0xFF),
            (byte)((weightsPacked >> 16) & 0xFF),
            (byte)((weightsPacked >> 24) & 0xFF),
            (byte)(indicesPacked & 0xFF),
            (byte)((indicesPacked >> 8) & 0xFF),
            (byte)((indicesPacked >> 16) & 0xFF),
            (byte)((indicesPacked >> 24) & 0xFF));
    }

    private static M2Era100Batch ReadM2Batch(byte[] data, int ofs)
    {
        return new M2Era100Batch(
            ReadByteAt(data, ofs + M2Era100Constants.BatchFlagsOffset),
            ReadByteAt(data, ofs + M2Era100Constants.BatchPriorityPlaneOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchShaderIdOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchSkinSectionIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchGeosetIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchColorIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchMaterialIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchMaterialLayerOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureCountOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureComboIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureCoordComboIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureWeightComboIndexOffset),
            ReadUInt16At(data, ofs + M2Era100Constants.BatchTextureTransformComboIndexOffset));
    }

    // ─── Textures ────────────────────────────────────────────────────────────

    private static List<M2Era100Texture> ReadTextures(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.TextureCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.TextureOffsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, M2Era100Constants.TextureStride, data.Length, sourcePath, "textures");
        List<M2Era100Texture> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
        {
            int ofs = checked((int)offset + (i * M2Era100Constants.TextureStride));
            uint type = ReadUInt32At(data, ofs + M2Era100Constants.TextureTypeOffset);
            uint texFlags = ReadUInt32At(data, ofs + M2Era100Constants.TextureFlagsOffset);
            uint nameLen = ReadUInt32At(data, ofs + M2Era100Constants.TextureNameLenOffset);
            uint nameOfs = ReadUInt32At(data, ofs + M2Era100Constants.TextureNameOfsOffset);
            string? filename = TryReadStringAt(data, sourcePath, $"textures[{i}].filename", nameLen, nameOfs);
            values.Add(new M2Era100Texture(type, texFlags, filename ?? string.Empty));
        }

        return values;
    }

    // ─── Sequences ───────────────────────────────────────────────────────────

    private static List<M2SequenceDefinition> ReadSequences(byte[] data, string sourcePath)
    {
        uint count = ReadUInt32At(data, M2Era100Constants.SequenceCountOffset);
        uint offset = ReadUInt32At(data, M2Era100Constants.SequenceOffsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, M2Era100Constants.SequenceStride, data.Length, sourcePath, "sequences");
        List<M2SequenceDefinition> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
        {
            int ofs = checked((int)offset + (i * M2Era100Constants.SequenceStride));
            values.Add(new M2SequenceDefinition(
                i,
                ReadUInt16At(data, ofs + 0x00),
                ReadUInt16At(data, ofs + 0x02),
                ReadUInt32At(data, ofs + 0x04),
                ReadLenientSingleAt(data, ofs + 0x08, sourcePath, $"sequences[{i}].moveSpeed"),
                ReadUInt32At(data, ofs + 0x0C),
                ReadInt16At(data, ofs + 0x10),
                ReadUInt32At(data, ofs + 0x14),
                ReadUInt32At(data, ofs + 0x18),
                ReadUInt16At(data, ofs + 0x1C),
                ReadUInt16At(data, ofs + 0x1E),
                ReadLenientVector3At(data, ofs + 0x20, sourcePath, $"sequences[{i}].boundsMin"),
                ReadLenientVector3At(data, ofs + 0x2C, sourcePath, $"sequences[{i}].boundsMax"),
                ReadLenientSingleAt(data, ofs + 0x38, sourcePath, $"sequences[{i}].boundsRadius"),
                ReadInt16At(data, ofs + 0x3C),
                ReadUInt16At(data, ofs + 0x3E)));
        }

        return values;
    }

    // ─── Primitive readers ───────────────────────────────────────────────────

    private static byte[] ReadAllBytes(Stream stream)
    {
        if (stream is MemoryStream ms && ms.TryGetBuffer(out ArraySegment<byte> segment))
            return segment.ToArray();

        long length = stream.Length;
        if (length > int.MaxValue)
            throw new InvalidDataException("M2 file exceeds 2 GiB.");

        stream.Position = 0;
        byte[] buffer = new byte[length];
        int totalRead = 0;
        while (totalRead < buffer.Length)
        {
            int read = stream.Read(buffer, totalRead, buffer.Length - totalRead);
            if (read == 0)
                break;
            totalRead += read;
        }

        return totalRead == buffer.Length ? buffer : buffer.AsSpan(0, totalRead).ToArray();
    }

    private static uint ReadUInt32At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(uint), "m2 data", "uint32");
        return BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
    }

    private static ushort ReadUInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(ushort), "m2 data", "uint16");
        return BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)));
    }

    private static short ReadInt16At(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(short), "m2 data", "int16");
        return BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(offset, sizeof(short)));
    }

    private static byte ReadByteAt(byte[] data, int offset)
    {
        EnsureReadable(data, offset, sizeof(byte), "m2 data", "byte");
        return data[offset];
    }

    private static float ReadLenientSingleAt(byte[] data, int offset, string sourcePath, string label)
    {
        EnsureReadable(data, offset, sizeof(float), sourcePath, label);
        float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, sizeof(float))));
        return float.IsFinite(value) ? value : 0f;
    }

    private static Vector3 ReadLenientVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadLenientSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadLenientSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadLenientSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static string? TryReadName(byte[] data, string sourcePath)
    {
        uint nameCount = ReadUInt32At(data, M2Era100Constants.NameCountOffset);
        uint nameOffset = ReadUInt32At(data, M2Era100Constants.NameOffsetOffset);
        if (nameCount == 0 || nameOffset == 0)
            return null;

        ValidateSpan(nameCount, nameOffset, 1, data.Length, sourcePath, "modelName");
        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)nameOffset), checked((int)nameCount));
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        return length == 0 ? null : Encoding.UTF8.GetString(bytes[..length]);
    }

    private static string? TryReadStringAt(byte[] data, string sourcePath, string label, uint count, uint offset)
    {
        if (count == 0 || offset == 0)
            return null;

        ValidateSpan(count, offset, sizeof(byte), data.Length, sourcePath, label);
        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)offset), checked((int)count));
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        return length == 0 ? null : Encoding.UTF8.GetString(bytes[..length]);
    }

    private static List<uint> ReadUInt32Table(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, sizeof(uint), data.Length, sourcePath, label);
        List<uint> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
            values.Add(ReadUInt32At(data, checked((int)offset + (i * sizeof(uint)))));

        return values;
    }

    private static List<short> ReadInt16Table(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, sizeof(short), data.Length, sourcePath, label);
        List<short> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
            values.Add(ReadInt16At(data, checked((int)offset + (i * sizeof(short)))));

        return values;
    }

    private static List<ushort> ReadUInt16Lookup(byte[] data, string sourcePath, string label, int countOffset, int offsetOffset)
    {
        uint count = ReadUInt32At(data, countOffset);
        uint offset = ReadUInt32At(data, offsetOffset);
        if (count == 0 || offset == 0)
            return [];

        ValidateSpan(count, offset, sizeof(ushort), data.Length, sourcePath, label);
        List<ushort> values = new(checked((int)count));
        for (int i = 0; i < count; i++)
            values.Add(ReadUInt16At(data, checked((int)offset + (i * sizeof(ushort)))));

        return values;
    }

    // ─── Validation ──────────────────────────────────────────────────────────

    private static void EnsureReadable(byte[] data, int offset, int size, string sourcePath, string label)
    {
        if (offset < 0 || offset > data.Length - size)
            throw new InvalidDataException($"M2 file '{sourcePath}' is truncated at {label} (offset 0x{offset:X}, need {size} bytes, have {data.Length - Math.Max(0, offset)}).");
    }

    private static void ValidateSpan(uint count, uint offset, int stride, long fileSize, string sourcePath, string label)
    {
        if (count == 0)
            return;
        if (offset == 0)
            throw new InvalidDataException($"M2 file '{sourcePath}' has zero offset for non-empty {label}.");

        // Check for overflow: count * stride
        if (count > int.MaxValue / stride)
            throw new InvalidDataException($"M2 file '{sourcePath}' has impossibly large {label} count ({count}).");

        long end = checked((long)offset + (long)count * stride);
        if (end > fileSize)
            throw new InvalidDataException($"M2 file '{sourcePath}' has {label} span (offset 0x{offset:X}, count {count}, stride {stride}) exceeding file size (0x{fileSize:X}).");
    }

    // ─── Layout validation (used by the dispatcher to distinguish 1.0.0 from 1.12.1) ───

    /// <summary>
    /// Validates whether the header bytes are consistent with the 1.0.0 layout.
    /// Checks that the vertices and divisions M2Array fields at the 1.0.0 header
    /// positions produce sane offsets within the file.
    /// </summary>
    public static bool ValidateLayout(ReadOnlySpan<byte> data, string sourcePath)
    {
        if (data.Length < M2Era100Constants.MinimumHeaderSizeBytes)
            return false;

        // Check vertices at 0x44 (count) / 0x48 (offset) — M2Vertex stride 0x30.
        if (!TryValidateArray(data, M2Era100Constants.VertexCountOffset, M2Era100Constants.VertexOffsetOffset,
            M2Era100Constants.VertexStride, "vertices"))
            return false;

        // Check divisions at 0x4C (count) / 0x50 (offset) — M2Division stride 0x2C.
        if (!TryValidateArray(data, M2Era100Constants.DivisionCountOffset, M2Era100Constants.DivisionOffsetOffset,
            M2Era100Constants.DivisionStride, "divisions"))
            return false;

        // Check textures at 0x5C (count) / 0x60 (offset) — M2Texture stride 0x10.
        if (!TryValidateArray(data, M2Era100Constants.TextureCountOffset, M2Era100Constants.TextureOffsetOffset,
            M2Era100Constants.TextureStride, "textures"))
            return false;

        return true;
    }

    private static bool TryValidateArray(ReadOnlySpan<byte> data, int countOffset, int offsetOffset, int stride, string label)
    {
        if (countOffset + sizeof(uint) > data.Length || offsetOffset + sizeof(uint) > data.Length)
            return false;

        uint count = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(countOffset, sizeof(uint)));
        uint offset = BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(offsetOffset, sizeof(uint)));

        if (count == 0)
            return true; // Empty arrays are valid.
        if (offset == 0 || offset >= (uint)data.Length)
            return false;
        if (count > int.MaxValue / (uint)stride)
            return false;

        long end = checked((long)offset + (long)count * stride);
        return end <= data.Length;
    }
}