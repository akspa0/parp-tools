using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Formats.Liquids;

/// <summary>
/// Unified reader for WL* files (WLW, WLM, WLQ, WLL).
/// These are loose "Water Level" files containing liquid heightmaps.
/// Each block is 360 bytes with a 4x4 vertex grid.
/// 
/// WL* files are NOT read by the WoW client but contain water plane data
/// that can be used to recover missing liquid information from map data.
/// </summary>
public class WlFile
{
    public WlHeader Header { get; set; } = new();
    public List<WlBlock> Blocks { get; } = new();

    /// <summary>
    /// Reads any WL* file (WLW, WLM, WLQ, or WLL).
    /// </summary>
    public static WlFile Read(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        return Read(br, path);
    }

    /// <summary>
    /// Reads any WL* file from a stream.
    /// </summary>
    public static WlFile Read(BinaryReader br, string? fileName = null)
    {
        var wl = new WlFile();

        var magic = br.ReadBytes(4);
        string magicStr = Encoding.ASCII.GetString(magic);

        WlFileType fileType;
        if (magicStr == "*QIL" || magicStr == "LIQ*")
        {
            // Determine WLW vs WLM from filename
            fileType = fileName?.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) == true
                ? WlFileType.WLM
                : WlFileType.WLW;
            wl.Header = ReadWlwHeader(br, magic, fileType);
        }
        else if (magicStr == "2QIL")
        {
            fileType = WlFileType.WLQ;
            wl.Header = ReadWlqHeader(br, magic);
        }
        else
        {
            throw new InvalidDataException($"Unknown WL* magic: {magicStr} (0x{BitConverter.ToUInt32(magic):X8})");
        }

        // Read blocks (identical 360-byte format for all WL* types)
        for (uint i = 0; i < wl.Header.BlockCount; i++)
        {
            wl.Blocks.Add(ReadBlock(br));
        }

        return wl;
    }

    private static WlHeader ReadWlwHeader(BinaryReader br, byte[] magic, WlFileType type)
    {
        // WLW/WLM: 16 bytes total
        // magic(4) + version(2) + unk06(2) + liquidType(2) + padding(2) + blockCount(4)
        var header = new WlHeader
        {
            Magic = magic,
            FileType = type,
            Version = br.ReadUInt16(),
            Unk06 = br.ReadUInt16(),
            RawLiquidType = br.ReadUInt16(),
            Padding = br.ReadUInt16(),
            BlockCount = br.ReadUInt32()
        };

        // WLM files always use magma (type 6) regardless of header value
        if (type == WlFileType.WLM)
            header.RawLiquidType = 6;

        header.LiquidType = MapWlwLiquidType(header.RawLiquidType);
        return header;
    }

    private static WlHeader ReadWlqHeader(BinaryReader br, byte[] magic)
    {
        // WLQ: magic(4) + version(2) + unk06(2) + unk08[4](4) + liquidType(4) + unk10[9](18) + blockCount(4)
        var header = new WlHeader
        {
            Magic = magic,
            FileType = WlFileType.WLQ,
            Version = br.ReadUInt16(),
            Unk06 = br.ReadUInt16()
        };

        br.ReadBytes(4); // unk08[4] - always 0
        header.RawLiquidType = (ushort)br.ReadUInt32();
        for (int i = 0; i < 9; i++) br.ReadUInt16(); // unk10[9]
        header.BlockCount = br.ReadUInt32();

        header.LiquidType = MapWlqLiquidType(header.RawLiquidType);
        return header;
    }

    private static WlBlock ReadBlock(BinaryReader br)
    {
        var block = new WlBlock();

        // 16 vertices (4x4 grid, 3 floats each = 192 bytes)
        // Layout: starts at lower-right corner, indices 15..0
        block.Vertices = new Vector3[16];
        for (int v = 0; v < 16; v++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle(); // Z is height (z-up)
            block.Vertices[v] = new Vector3(x, y, z);
        }

        // Coord (2 floats = 8 bytes) - internal grid position
        block.CoordX = br.ReadSingle();
        block.CoordY = br.ReadSingle();

        // Data (80 ushorts = 160 bytes) - purpose unknown
        block.Data = new ushort[80];
        for (int d = 0; d < 80; d++)
            block.Data[d] = br.ReadUInt16();

        return block;
    }

    /// <summary>
    /// Maps WLW/WLM liquid type to unified WlLiquidType.
    /// WLW types: 0=still, 1=ocean, 4=river, 6=magma, 8=fast
    /// </summary>
    private static WlLiquidType MapWlwLiquidType(ushort raw) => raw switch
    {
        0 => WlLiquidType.StillWater,
        1 => WlLiquidType.Ocean,
        4 => WlLiquidType.River,
        6 => WlLiquidType.Magma,
        8 => WlLiquidType.FastWater,
        _ => WlLiquidType.StillWater
    };

    /// <summary>
    /// Maps WLQ liquid type to unified WlLiquidType.
    /// WLQ types (WMO style): 0=river, 1=ocean, 2=magma, 3=slime
    /// </summary>
    private static WlLiquidType MapWlqLiquidType(ushort raw) => raw switch
    {
        0 => WlLiquidType.River,
        1 => WlLiquidType.Ocean,
        2 => WlLiquidType.Magma,
        3 => WlLiquidType.Slime,
        _ => WlLiquidType.River
    };
}

public enum WlFileType
{
    WLW,  // Water Level Water
    WLM,  // Water Level Magma (always magma)
    WLQ,  // Water Level (alternate format, WMO-style types)
    WLL   // Water Level (legacy?)
}

public enum WlLiquidType
{
    StillWater = 0,
    Ocean = 1,
    River = 2,
    Magma = 3,
    Slime = 4,
    FastWater = 5
}

public class WlHeader
{
    public byte[] Magic { get; set; } = new byte[4];
    public WlFileType FileType { get; set; }
    public ushort Version { get; set; }
    public ushort Unk06 { get; set; }
    public ushort RawLiquidType { get; set; }
    public ushort Padding { get; set; }
    public uint BlockCount { get; set; }
    public WlLiquidType LiquidType { get; set; }
}

public class WlBlock
{
    /// <summary>
    /// 16 vertices in 4x4 grid (z-up).
    /// Layout starts at lower-right corner: indices 15, 14, 13... 0
    /// </summary>
    public Vector3[] Vertices { get; set; } = new Vector3[16];

    /// <summary>Internal grid X coordinate.</summary>
    public float CoordX { get; set; }

    /// <summary>Internal grid Y coordinate.</summary>
    public float CoordY { get; set; }

    /// <summary>Unknown data (80 ushorts).</summary>
    public ushort[] Data { get; set; } = new ushort[80];

    /// <summary>
    /// Gets heights in standard row-major order (reversed from file layout).
    /// </summary>
    public float[] GetHeights4x4()
    {
        var heights = new float[16];
        for (int i = 0; i < 16; i++)
            heights[15 - i] = Vertices[i].Z;
        return heights;
    }

    /// <summary>
    /// Gets the world position from the first vertex.
    /// </summary>
    public Vector3 WorldPosition => Vertices[0];
}
