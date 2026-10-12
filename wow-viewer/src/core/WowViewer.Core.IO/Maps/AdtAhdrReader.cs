using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Spec 237: decodes AHDR-family terrain files (DAT v26 measured; wiki v22/v23 by the same chunk
/// vocabulary). Never throws on malformed input: problems are reported in
/// <see cref="AdtAhdrTile.Diagnostics"/>. Chunks are unpadded (measured: 0 unaccounted bytes on
/// the 699-file v26 corpus).
/// </summary>
public static class AdtAhdrReader
{
    private const uint Mver = 0x4D564552; // 'MVER'
    private const uint Ahdr = 0x41484452;
    private const uint Aloc = 0x414C4F43;
    private const uint Avtx = 0x41565458;
    private const uint Anrm = 0x414E524D;
    private const uint Atex = 0x41544558;
    private const uint Adoo = 0x41444F4F;
    private const uint Acnk = 0x41434E4B;
    private const uint Acvt = 0x41435654;
    private const uint Adst = 0x41445354;
    private const uint Alyr = 0x414C5952;
    private const uint Amap = 0x414D4150;
    private const uint Ashd = 0x41534844;
    private const uint Acdo = 0x4143444F;
    private const uint Aoch = 0x414F4348;

    private const int AcnkHeaderSize = 0x40;
    private const int AlyrFixedSize = 0x20;
    // v26 records are 0x38 or 0x3C bytes; the wiki's v22/v23 layout is unmeasured, so fields past 0x30 are optional.
    private const int AcdoMinimumSize = 0x30;

    /// <summary>True when the buffer starts with AHDR, or with MVER immediately followed by AHDR.</summary>
    public static bool IsAhdrFamily(ReadOnlySpan<byte> data)
    {
        if (data.Length < 8)
            return false;

        uint first = BinaryPrimitives.ReadUInt32LittleEndian(data);
        if (first == Ahdr)
            return true;

        if (first != Mver)
            return false;

        long next = 8L + BinaryPrimitives.ReadUInt32LittleEndian(data[4..]);
        return next + 4 <= data.Length && BinaryPrimitives.ReadUInt32LittleEndian(data[(int)next..]) == Ahdr;
    }

    /// <summary>
    /// Reads the AHDR version field without decoding the whole file. The AHDR-family DATs are loose
    /// developer files scattered through client data under arbitrary extensions, so a folder can mix
    /// revisions and the version is only knowable per file.
    /// </summary>
    public static bool TryReadVersion(ReadOnlySpan<byte> data, out uint version)
    {
        version = 0;
        foreach ((uint id, int offset, int size) in Walk(data, 0, data.Length))
        {
            if (id == Ahdr && size >= 4)
            {
                version = BinaryPrimitives.ReadUInt32LittleEndian(data[offset..]);
                return true;
            }
        }

        return false;
    }

    /// <summary>Reads only ALOC tile coordinates (X, Y) without decoding the whole file.</summary>
    public static bool TryReadTileLocation(ReadOnlySpan<byte> data, out int tileX, out int tileY)
    {
        tileX = tileY = -1;
        foreach ((uint id, int offset, int size) in Walk(data, 0, data.Length))
        {
            if (id == Aloc && size >= 12)
            {
                tileX = (int)BinaryPrimitives.ReadUInt32LittleEndian(data[(offset + 4)..]);
                tileY = (int)BinaryPrimitives.ReadUInt32LittleEndian(data[(offset + 8)..]);
                return true;
            }

            if (id == Avtx)
                return false; // ALOC precedes AVTX in every observed file
        }

        return false;
    }

    /// <summary>
    /// Parses the trailing "XX_YY" tile coordinates from a DAT file name (e.g. "area_51_31.dat"
    /// -> X=51, Y=31). v22/v23 files carry the same AHDR-family vocabulary as v26 but no ALOC
    /// chunk, so the file name is the only tile-location source. The two integers are read in
    /// ALOC order (X then Y). Returns false when the name has fewer than two integers.
    /// </summary>
    public static bool TryParseTileLocationFromName(string path, out int tileX, out int tileY)
    {
        tileX = tileY = -1;
        string stem = Path.GetFileNameWithoutExtension(path);
        var numbers = new List<int>();
        int current = 0;
        bool inNumber = false;
        foreach (char c in stem)
        {
            if (char.IsAsciiDigit(c))
            {
                current = (current * 10) + (c - '0');
                inNumber = true;
            }
            else if (inNumber)
            {
                numbers.Add(current);
                current = 0;
                inNumber = false;
            }
        }

        if (inNumber)
            numbers.Add(current);

        if (numbers.Count < 2)
            return false;

        tileX = numbers[^2];
        tileY = numbers[^1];
        return true;
    }

    public static AdtAhdrTile Read(byte[] data, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(data);
        var diagnostics = new List<string>();
        uint? mverVersion = null;
        uint version = 0;
        int verticesX = 0, verticesY = 0, chunksX = 0, chunksY = 0;
        uint[] reserved = [];
        uint[]? aloc = null;
        float[] outer = [], inner = [];
        byte[]? normals = null, shading = null, aoch = null;
        var textures = new List<string>();
        var models = new List<string>();
        var chunks = new List<AdtAhdrChunk>();
        var modelFileReferences = new List<AdtAhdrModelFileReference>();

        ReadOnlySpan<byte> span = data;
        int end = 0;
        foreach ((uint id, int offset, int size) in Walk(span, 0, span.Length))
        {
            end = offset + size;
            ReadOnlySpan<byte> payload = span.Slice(offset, size);
            switch (id)
            {
                case Mver when size >= 4:
                    mverVersion = BinaryPrimitives.ReadUInt32LittleEndian(payload);
                    break;
                case Ahdr when size >= 20:
                    version = BinaryPrimitives.ReadUInt32LittleEndian(payload);
                    verticesX = (int)BinaryPrimitives.ReadUInt32LittleEndian(payload[4..]);
                    verticesY = (int)BinaryPrimitives.ReadUInt32LittleEndian(payload[8..]);
                    chunksX = (int)BinaryPrimitives.ReadUInt32LittleEndian(payload[12..]);
                    chunksY = (int)BinaryPrimitives.ReadUInt32LittleEndian(payload[16..]);
                    reserved = ReadUInt32Array(payload[20..]);
                    break;
                case Aloc:
                    aloc = ReadUInt32Array(payload);
                    break;
                case Aoch:
                    aoch = payload.ToArray();
                    break;
                case Avtx:
                    (outer, inner) = SplitGrid(payload, verticesX, verticesY, diagnostics);
                    break;
                case Anrm:
                    normals = payload.ToArray();
                    break;
                case Atex:
                    textures.Add(ReadCString(payload));
                    break;
                case Adoo:
                    models.Add(ReadCString(payload));
                    break;
                case Acnk:
                    chunks.Add(ReadChunk(span, offset, size, chunks.Count, diagnostics));
                    break;
                case Adst when size >= 12:
                    modelFileReferences.Add(new AdtAhdrModelFileReference(
                        BinaryPrimitives.ReadUInt32LittleEndian(payload),
                        BinaryPrimitives.ReadUInt32LittleEndian(payload[4..]),
                        BinaryPrimitives.ReadUInt32LittleEndian(payload[8..])));
                    break;
                case Adst:
                    diagnostics.Add($"ADST is {size} bytes, expected 12");
                    break;
                case Acvt:
                    shading = payload.ToArray();
                    break;
            }
        }

        if (end != span.Length)
            diagnostics.Add($"top-level walk ended at {end} of {span.Length} bytes");
        if (version == 0)
            diagnostics.Add("no AHDR chunk");
        if (aloc is null)
            diagnostics.Add("no ALOC chunk (tile location unknown)");
        if (chunksX > 0 && chunks.Count != chunksX * chunksY)
            diagnostics.Add($"expected {chunksX * chunksY} ACNK, found {chunks.Count}");

        return new AdtAhdrTile
        {
            SourcePath = sourcePath,
            MverVersion = mverVersion,
            Version = version,
            VerticesX = verticesX,
            VerticesY = verticesY,
            ChunksX = chunksX,
            ChunksY = chunksY,
            HeaderReserved = reserved,
            Aloc = aloc,
            AochRaw = aoch,
            OuterHeights = outer,
            InnerHeights = inner,
            NormalsRaw = normals,
            VertexShadingRaw = shading,
            TextureNames = textures,
            ModelNames = models,
            Chunks = chunks,
            ModelFileReferences = modelFileReferences,
            Diagnostics = diagnostics,
        };
    }

    private static AdtAhdrChunk ReadChunk(ReadOnlySpan<byte> file, int offset, int size, int ordinal, List<string> diagnostics)
    {
        ReadOnlySpan<byte> payload = file.Slice(offset, size);
        if (size < AcnkHeaderSize)
        {
            diagnostics.Add($"ACNK {ordinal}: {size} bytes, shorter than the 0x40 header");
            return new AdtAhdrChunk { IndexX = ordinal % 16, IndexY = ordinal / 16 };
        }

        var layers = new List<AdtAhdrLayer>();
        var objects = new List<AdtAhdrObjectDefinition>();
        byte[]? shadow = null;
        foreach ((uint id, int subOffset, int subSize) in Walk(file, offset + AcnkHeaderSize, offset + size))
        {
            ReadOnlySpan<byte> sub = file.Slice(subOffset, subSize);
            switch (id)
            {
                case Alyr when subSize >= 8:
                    layers.Add(ReadLayer(file, subOffset, subSize));
                    break;
                case Ashd:
                    shadow = sub.ToArray();
                    break;
                case Acdo when subSize >= AcdoMinimumSize:
                    objects.Add(ReadObject(sub));
                    break;
            }
        }

        return new AdtAhdrChunk
        {
            IndexX = BinaryPrimitives.ReadInt32LittleEndian(payload),
            IndexY = BinaryPrimitives.ReadInt32LittleEndian(payload[4..]),
            HeaderRaw = payload[..AcnkHeaderSize].ToArray(),
            Layers = layers,
            ShadowRaw = shadow,
            Objects = objects,
        };
    }

    private static AdtAhdrLayer ReadLayer(ReadOnlySpan<byte> file, int offset, int size)
    {
        ReadOnlySpan<byte> payload = file.Slice(offset, size);
        int textureIndex = BinaryPrimitives.ReadInt32LittleEndian(payload);
        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload[4..]);
        byte[]? alpha = null;
        if (size > AlyrFixedSize)
        {
            foreach ((uint id, int subOffset, int subSize) in Walk(file, offset + AlyrFixedSize, offset + size))
            {
                if (id == Amap)
                    alpha = file.Slice(subOffset, subSize).ToArray();
            }
        }

        return new AdtAhdrLayer(textureIndex, flags, alpha);
    }

    private static AdtAhdrObjectDefinition ReadObject(ReadOnlySpan<byte> record)
    {
        uint trailingCount = record.Length >= 0x34 ? BinaryPrimitives.ReadUInt32LittleEndian(record[0x30..]) : 0;
        int available = Math.Max(0, (record.Length - 0x38) / 4);
        var trailing = new uint[Math.Max(0, (int)Math.Min(trailingCount, (uint)available))];
        for (int i = 0; i < trailing.Length; i++)
            trailing[i] = BinaryPrimitives.ReadUInt32LittleEndian(record[(0x38 + i * 4)..]);

        return new AdtAhdrObjectDefinition(
            BinaryPrimitives.ReadInt32LittleEndian(record),
            ReadVector3(record[0x04..]),
            ReadVector3(record[0x10..]),
            BinaryPrimitives.ReadSingleLittleEndian(record[0x1C..]),
            BinaryPrimitives.ReadSingleLittleEndian(record[0x20..]),
            BinaryPrimitives.ReadUInt32LittleEndian(record[0x24..]),
            BinaryPrimitives.ReadSingleLittleEndian(record[0x28..]),
            BinaryPrimitives.ReadUInt32LittleEndian(record[0x2C..]),
            record.Length >= 0x38 ? BinaryPrimitives.ReadUInt32LittleEndian(record[0x34..]) : 0,
            trailing,
            record.ToArray());
    }

    private static (float[] Outer, float[] Inner) SplitGrid(ReadOnlySpan<byte> payload, int verticesX, int verticesY, List<string> diagnostics)
    {
        int outerCount = verticesX * verticesY;
        int innerCount = Math.Max(0, verticesX - 1) * Math.Max(0, verticesY - 1);
        if (outerCount == 0 || payload.Length != (outerCount + innerCount) * 4)
        {
            diagnostics.Add($"AVTX is {payload.Length} bytes; header {verticesX}x{verticesY} expects {(outerCount + innerCount) * 4}");
            return ([], []);
        }

        var outer = new float[outerCount];
        var inner = new float[innerCount];
        for (int i = 0; i < outerCount; i++)
            outer[i] = BinaryPrimitives.ReadSingleLittleEndian(payload[(i * 4)..]);
        for (int i = 0; i < innerCount; i++)
            inner[i] = BinaryPrimitives.ReadSingleLittleEndian(payload[((outerCount + i) * 4)..]);
        return (outer, inner);
    }

    /// <summary>Unpadded chunk walk over [start, end). Stops at the first chunk that overruns.</summary>
    private static List<(uint Id, int PayloadOffset, int Size)> Walk(ReadOnlySpan<byte> data, int start, int end)
    {
        var result = new List<(uint, int, int)>();
        int position = start;
        while (position + 8 <= end)
        {
            uint id = BinaryPrimitives.ReadUInt32LittleEndian(data[position..]);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[(position + 4)..]);
            if (position + 8L + size > end)
                break;

            result.Add((id, position + 8, (int)size));
            position += 8 + (int)size;
        }

        return result;
    }

    private static uint[] ReadUInt32Array(ReadOnlySpan<byte> payload)
    {
        var values = new uint[payload.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadUInt32LittleEndian(payload[(i * 4)..]);
        return values;
    }

    private static Vector3 ReadVector3(ReadOnlySpan<byte> data) => new(
        BinaryPrimitives.ReadSingleLittleEndian(data),
        BinaryPrimitives.ReadSingleLittleEndian(data[4..]),
        BinaryPrimitives.ReadSingleLittleEndian(data[8..]));

    private static string ReadCString(ReadOnlySpan<byte> payload)
    {
        int terminator = payload.IndexOf((byte)0);
        return Encoding.Latin1.GetString(terminator >= 0 ? payload[..terminator] : payload);
    }
}
