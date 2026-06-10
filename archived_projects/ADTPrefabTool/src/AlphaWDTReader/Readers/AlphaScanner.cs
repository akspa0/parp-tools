using System.Text;
using AlphaWDTReader.IO;
using AlphaWDTReader.Model;

namespace AlphaWDTReader.Readers;

public static class AlphaScanner
{
    public static AlphaScanResult Scan(string filePath)
    {
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);

        var result = new AlphaScanResult { FilePath = filePath };
        long fileLen = fs.Length;

        while (fs.Position + 8 <= fileLen)
        {
            long blockStart = fs.Position;
            uint fourcc = AlphaFourCC.ReadFourCC(br);
            uint size = br.ReadUInt32();
            result.ChunkCount++;

            long payloadStart = fs.Position;
            long payloadEnd = payloadStart + size;
            if (payloadEnd > fileLen || size > int.MaxValue)
            {
                // Corrupt or truncated; stop scanning to be safe
                break;
            }

            if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MVER))
            {
                result.HasMver = true;
                fs.Position = payloadEnd;
            }
            else if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MPHD))
            {
                result.HasMphd = true;
                fs.Position = payloadEnd;
            }
            else if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MAIN))
            {
                result.HasMain = true;
                try
                {
                    Span<byte> buf = stackalloc byte[(int)Math.Min(size, 4096 * 8)];
                    int read = br.Read(buf);
                    // Heuristic: 8-byte entries (offset:uint32, size:uint32). Non-zero offset indicates present tile.
                    if (read % 8 == 0)
                    {
                        int entries = read / 8;
                        int present = 0;
                        for (int i = 0; i < entries; i++)
                        {
                            uint off = BitConverter.ToUInt32(buf.Slice(i * 8, 4));
                            // uint len = BitConverter.ToUInt32(buf.Slice(i * 8 + 4, 4));
                            if (off != 0) present++;
                        }
                        result.MainDeclaredTiles = present;
                    }
                    // skip remainder if any
                    fs.Position = blockStart + 8 + size;
                }
                catch
                {
                    fs.Position = blockStart + 8 + size;
                }
            }
            else if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MDNM))
            {
                try
                {
                    var names = ReadZeroTerminatedStrings(br, size);
                    result.DoodadNameCount = names.Count;
                    foreach (var n in names.Take(5)) result.FirstDoodadNames.Add(n);
                }
                finally { fs.Position = payloadEnd; }
            }
            else if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MONM))
            {
                try
                {
                    var names = ReadZeroTerminatedStrings(br, size);
                    result.WmoNameCount = names.Count;
                    foreach (var n in names.Take(5)) result.FirstWmoNames.Add(n);
                }
                finally { fs.Position = payloadEnd; }
            }
            else
            {
                // Skip unknown block
                fs.Position = payloadEnd;
            }
        }

        return result;
    }

    private static List<string> ReadZeroTerminatedStrings(BinaryReader br, uint size)
    {
        var list = new List<string>();
        long start = br.BaseStream.Position;
        long end = start + size;
        var bytes = br.ReadBytes((int)size);
        int begin = 0;
        for (int i = 0; i < bytes.Length; i++)
        {
            if (bytes[i] == 0)
            {
                if (i > begin)
                {
                    list.Add(Encoding.UTF8.GetString(bytes, begin, i - begin));
                }
                begin = i + 1;
            }
        }
        // ignore trailing partial
        br.BaseStream.Position = end;
        return list;
    }
}
