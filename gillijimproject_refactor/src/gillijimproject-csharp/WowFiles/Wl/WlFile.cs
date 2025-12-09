using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace GillijimProject.WowFiles.Wl;

/// <summary>
/// Unified reader for WL* files (WLW, WLM, WLQ).
/// All formats share the same block structure (360 bytes each).
/// </summary>
public class WlFile
{
    public WlHeader Header { get; set; }
    public List<WlBlock> Blocks { get; } = new();

    /// <summary>
    /// Reads any WL* file (WLW, WLM, or WLQ).
    /// </summary>
    public static WlFile Read(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var wl = new WlFile();

        // Read magic to determine format
        var magic = br.ReadBytes(4);
        string magicStr = Encoding.ASCII.GetString(magic);
        
        WlFileType fileType;
        if (magicStr == "*QIL" || magicStr == "LIQ*")
        {
            fileType = path.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) 
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
            throw new InvalidDataException($"Unknown WL* magic: {magicStr}");
        }

        // Read blocks (identical format for all WL* types)
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

        // Map WLW/WLM liquid type to unified enum
        header.LiquidType = MapWlwLiquidType(header.RawLiquidType);
        
        return header;
    }

    private static WlHeader ReadWlqHeader(BinaryReader br, byte[] magic)
    {
        // WLQ: magic(4) + version(2) + unk06(2) + unk08[4](4) + liquidType(4?) + unk10[9](18) + blockCount(4)
        // Approximate: 4+2+2+4+4+18+4 = 38 bytes header
        var header = new WlHeader
        {
            Magic = magic,
            FileType = WlFileType.WLQ,
            Version = br.ReadUInt16(),
            Unk06 = br.ReadUInt16()
        };

        // unk08[4] - always 0
        br.ReadBytes(4);
        
        // liquidType (WLQ uses WMO-style: 0=river, 1=ocean, 2=magma, 3=slime)
        header.RawLiquidType = (ushort)br.ReadUInt32();
        
        // unk10[9] ushorts
        for (int i = 0; i < 9; i++)
            br.ReadUInt16();
        
        header.BlockCount = br.ReadUInt32();
        header.LiquidType = MapWlqLiquidType(header.RawLiquidType);

        return header;
    }

    private static WlBlock ReadBlock(BinaryReader br)
    {
        var block = new WlBlock();
        
        // 16 vertices (4x4 grid, 3 floats each = 192 bytes)
        block.Vertices = new Vector3[16];
        for (int v = 0; v < 16; v++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            block.Vertices[v] = new Vector3(x, y, z);
        }

        // Coord (2 floats = 8 bytes) - internal grid position
        block.CoordX = br.ReadSingle();
        block.CoordY = br.ReadSingle();

        // Data (80 ushorts = 160 bytes)
        block.Data = new ushort[80];
        for (int d = 0; d < 80; d++)
            block.Data[d] = br.ReadUInt16();

        return block;
    }

    /// <summary>
    /// Maps WLW/WLM liquid type to unified LiquidType.
    /// WLW types: 0=still, 1=ocean, 4=river, 6=magma, 8=fast
    /// </summary>
    private static LiquidType MapWlwLiquidType(ushort raw)
    {
        return raw switch
        {
            0 => LiquidType.StillWater,
            1 => LiquidType.Ocean,
            4 => LiquidType.River,
            6 => LiquidType.Magma,
            8 => LiquidType.FastWater,
            _ => LiquidType.StillWater
        };
    }

    /// <summary>
    /// Maps WLQ liquid type to unified LiquidType.
    /// WLQ types (WMO style): 0=river, 1=ocean, 2=magma, 3=slime
    /// </summary>
    private static LiquidType MapWlqLiquidType(ushort raw)
    {
        return raw switch
        {
            0 => LiquidType.River,
            1 => LiquidType.Ocean,
            2 => LiquidType.Magma,
            3 => LiquidType.Slime,
            _ => LiquidType.River
        };
    }
}

public enum WlFileType
{
    WLW,  // Water Level Water
    WLM,  // Water Level Magma (always magma)
    WLQ   // Water Level (alternate format, WMO-style types)
}

public enum LiquidType
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
    
    /// <summary>Unified liquid type enum.</summary>
    public LiquidType LiquidType { get; set; }
}

public class WlBlock
{
    public Vector3[] Vertices { get; set; } = new Vector3[16];
    public float CoordX { get; set; }
    public float CoordY { get; set; }
    public ushort[] Data { get; set; } = new ushort[80];
}
