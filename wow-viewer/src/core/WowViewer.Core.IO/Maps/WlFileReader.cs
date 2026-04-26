using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Unified reader for WL* files (WLW, WLM, WLQ, WLL).
/// Ported from WoWMapConverter.Core.Formats.Liquids.WlFile.
/// </summary>
public static class WlFileReader
{
    public static WlFile Read(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Read(fs, path);
    }

    public static WlFile Read(Stream stream, string? fileName = null)
    {
        using BinaryReader br = new(stream, Encoding.UTF8, leaveOpen: true);
        return Read(br, fileName);
    }

    public static WlFile Read(BinaryReader br, string? fileName = null)
    {
        byte[] magic = br.ReadBytes(4);
        string magicStr = Encoding.ASCII.GetString(magic);

        WlFileType fileType;
        WlHeader header;

        if (magicStr == "*QIL" || magicStr == "LIQ*")
        {
            if (fileName?.EndsWith(".wlm", StringComparison.OrdinalIgnoreCase) == true)
                fileType = WlFileType.WLM;
            else if (fileName?.EndsWith(".wll", StringComparison.OrdinalIgnoreCase) == true)
                fileType = WlFileType.WLL;
            else
                fileType = WlFileType.WLW;

            header = ReadWlwHeader(br, magic, fileType);
        }
        else if (magicStr == "2QIL")
        {
            fileType = WlFileType.WLQ;
            header = ReadWlqHeader(br, magic);
        }
        else
        {
            throw new InvalidDataException($"Unknown WL* magic: {magicStr} (0x{BitConverter.ToUInt32(magic):X8})");
        }

        List<WlBlock> blocks = new((int)header.BlockCount);
        for (uint i = 0; i < header.BlockCount; i++)
        {
            blocks.Add(ReadBlock(br));
        }

        return new WlFile
        {
            Header = header,
            Blocks = blocks
        };
    }

    private static WlHeader ReadWlwHeader(BinaryReader br, byte[] magic, WlFileType type)
    {
        ushort version = br.ReadUInt16();
        ushort unk06 = br.ReadUInt16();
        ushort rawLiquidType = br.ReadUInt16();
        ushort padding = br.ReadUInt16();
        uint blockCount = br.ReadUInt32();

        // WLM files always use magma (type 6) regardless of header value
        if (type == WlFileType.WLM)
            rawLiquidType = 6;

        return new WlHeader
        {
            Magic = magic,
            FileType = type,
            Version = version,
            Unk06 = unk06,
            RawLiquidType = rawLiquidType,
            Padding = padding,
            BlockCount = blockCount,
            LiquidType = MapWlwLiquidType(rawLiquidType)
        };
    }

    private static WlHeader ReadWlqHeader(BinaryReader br, byte[] magic)
    {
        ushort version = br.ReadUInt16();
        ushort unk06 = br.ReadUInt16();
        br.ReadBytes(4); // unk08[4] - always 0
        ushort rawLiquidType = (ushort)br.ReadUInt32();
        for (int i = 0; i < 9; i++) br.ReadUInt16(); // unk10[9]
        uint blockCount = br.ReadUInt32();

        return new WlHeader
        {
            Magic = magic,
            FileType = WlFileType.WLQ,
            Version = version,
            Unk06 = unk06,
            RawLiquidType = rawLiquidType,
            Padding = 0,
            BlockCount = blockCount,
            LiquidType = MapWlqLiquidType(rawLiquidType)
        };
    }

    private static WlBlock ReadBlock(BinaryReader br)
    {
        Vector3[] vertices = new Vector3[16];
        for (int v = 0; v < 16; v++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            vertices[v] = new Vector3(x, y, z);
        }

        float coordX = br.ReadSingle();
        float coordY = br.ReadSingle();

        ushort[] data = new ushort[80];
        for (int d = 0; d < 80; d++)
            data[d] = br.ReadUInt16();

        return new WlBlock
        {
            Vertices = vertices,
            CoordX = coordX,
            CoordY = coordY,
            Data = data
        };
    }

    /// <summary>Maps WLW/WLM liquid type to unified WlLiquidType.</summary>
    private static WlLiquidType MapWlwLiquidType(ushort raw) => raw switch
    {
        0 => WlLiquidType.StillWater,
        1 => WlLiquidType.Ocean,
        4 => WlLiquidType.River,
        6 => WlLiquidType.Magma,
        8 => WlLiquidType.FastWater,
        _ => WlLiquidType.StillWater
    };

    /// <summary>Maps WLQ liquid type to unified WlLiquidType.</summary>
    private static WlLiquidType MapWlqLiquidType(ushort raw) => raw switch
    {
        0 => WlLiquidType.River,
        1 => WlLiquidType.Ocean,
        2 => WlLiquidType.Magma,
        3 => WlLiquidType.Slime,
        _ => WlLiquidType.River
    };
}
