using System.Text;

namespace AlphaWDTReader.IO;

public readonly record struct BlockEntry(uint FourCC, long PayloadOffset, uint Size);

public static class TopLevelBlockIndexer
{
    public static List<BlockEntry> Index(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long fileLen = fs.Length;
        var list = new List<BlockEntry>();
        while (fs.Position + 8 <= fileLen)
        {
            uint fourcc = AlphaFourCC.ReadFourCC(br);
            uint size = br.ReadUInt32();
            long payload = fs.Position;
            long end = payload + size;
            if (end > fileLen) break;
            list.Add(new BlockEntry(fourcc, payload, size));
            fs.Position = end;
        }
        return list;
    }
}
