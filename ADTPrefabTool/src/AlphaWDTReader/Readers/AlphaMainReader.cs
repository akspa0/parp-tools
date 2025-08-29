using System.Text;
using AlphaWDTReader.IO;
using AlphaWDTReader.Model;

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
        int entryCount = bytes.Length / 8; // offset:uint32, size:uint32
        var list = new List<AlphaMainEntry>(4096);
        for (int i = 0; i < entryCount; i++)
        {
            uint off = BitConverter.ToUInt32(bytes, i * 8 + 0);
            uint size = BitConverter.ToUInt32(bytes, i * 8 + 4);
            int tileX = i % 64;
            int tileY = i / 64;
            list.Add(new AlphaMainEntry(off, size, tileX, tileY));
        }
        return list;
    }
}
