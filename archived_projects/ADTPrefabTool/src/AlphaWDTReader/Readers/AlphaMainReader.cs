using System.Text;
using AlphaWDTReader.IO;
using AlphaWDTReader.Model;
using System.Collections.Generic;

namespace AlphaWDTReader.Readers;

public static class AlphaMainReader
{
    public static List<AlphaMainEntry> ReadMainTable(string filePath)
    {
        var blocks = TopLevelBlockIndexer.Index(filePath);
        var main = blocks.FirstOrDefault(b => AlphaFourCC.Matches(b.FourCC, AlphaFourCC.MAIN));
        if (main == default) return new List<AlphaMainEntry>();

        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        fs.Position = main.PayloadOffset;
        var bytes = br.ReadBytes((int)main.Size);
        int stride = (bytes.Length >= 64 * 64 * 16) ? 16 : 8; // Alpha MAIN uses 16-byte stride (MHDR offset first)
        int entryCount = Math.Min(bytes.Length / stride, 64 * 64); // clamp to 4096 entries
        var list = new List<AlphaMainEntry>(4096);
        for (int i = 0; i < entryCount; i++)
        {
            uint off = BitConverter.ToUInt32(bytes, i * stride + 0);
            uint size = (stride == 16) ? 0u : BitConverter.ToUInt32(bytes, i * stride + 4);
            int tileX = i % 64;
            int tileY = i / 64;
            list.Add(new AlphaMainEntry(off, size, tileX, tileY));
        }
        return list;
    }

    // Some Alpha WDTs store MAIN as 64x64 entries of (flags:uint32, pad:uint32) where flags==1 indicates presence.
    // This helper returns the set of present (x,y) tiles if the payload length matches 64*64*8; otherwise returns an empty set.
    public static HashSet<(int x, int y)> ReadMainPresenceFlags(string filePath)
    {
        var set = new HashSet<(int x, int y)>();
        var blocks = TopLevelBlockIndexer.Index(filePath);
        var main = blocks.FirstOrDefault(b => AlphaFourCC.Matches(b.FourCC, AlphaFourCC.MAIN));
        if (main == default) return set;

        if (main.Size < 64 * 64 * 8) return set;

        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        fs.Position = main.PayloadOffset;
        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                uint flags = br.ReadUInt32();
                br.ReadUInt32(); // pad/unused
                if (flags == 1)
                {
                    set.Add((x, y));
                }
            }
        }
        return set;
    }
}
