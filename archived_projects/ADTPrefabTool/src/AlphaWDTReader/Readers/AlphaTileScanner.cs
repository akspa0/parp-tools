using System.Text;
using AlphaWDTReader.IO;
using AlphaWDTReader.Model;

namespace AlphaWDTReader.Readers;

public static class AlphaTileScanner
{
    public static int CountMcnkBlocks(string filePath, AlphaMainEntry entry)
    {
        if (entry.Offset == 0 || entry.Size == 0) return 0;
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long start = entry.Offset;
        long end = Math.Min(fs.Length, start + entry.Size);
        if (start >= fs.Length || start + 8 > end) return 0;
        fs.Position = start;
        int count = 0;
        while (fs.Position + 8 <= end)
        {
            long pos = fs.Position;
            uint fourcc;
            try { fourcc = AlphaFourCC.ReadFourCC(br); } catch { break; }
            uint size;
            try { size = br.ReadUInt32(); } catch { break; }
            long payload = fs.Position;
            long blockEnd = payload + size;
            if (blockEnd > end || size > int.MaxValue)
            {
                // If the size looks wrong, advance by 4 bytes and resync
                fs.Position = pos + 4; // shift window to try to resync on next iteration
                continue;
            }

            if (fourcc == 0x4B4E434Du) // 'MCNK'
            {
                count++;
            }
            fs.Position = blockEnd;
        }
        return count;
    }
}
