using System.Text;
using AlphaWDTReader.Model;
using AlphaWDTReader.IO;

namespace AlphaWDTReader.Readers;

public static class AlphaChunkIndexReader
{
    public static List<long> GetChunkOffsets(string filePath, AlphaMainEntry tile)
    {
        var offsets = new List<long>();
        if (tile.Offset == 0) return offsets;

        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long start = tile.Offset;
        long end = (tile.Size == 0) ? fs.Length : Math.Min(fs.Length, start + tile.Size);
        if (start >= fs.Length || start + 8 > end) return offsets;

        // Helper local to read MCIN table at a known position
        List<long> ReadMcinAt(long mcinPos)
        {
            var list = new List<long>(64);
            fs.Position = mcinPos;
            uint mcinFour = br.ReadUInt32();
            uint mcinSize = br.ReadUInt32();
            long mcinPayload = fs.Position;
            long mcinEnd = mcinPayload + mcinSize;
            if (!AlphaFourCC.Matches(mcinFour, AlphaFourCC.MCIN) || mcinEnd > end || mcinSize == 0) return list;
            int entrySize = (int)(mcinSize / 256);
            if (entrySize <= 0) return list;
            for (int i = 0; i < 256; i++)
            {
                long entryPos = mcinPayload + i * entrySize;
                if (entryPos + 8 > mcinEnd) break;
                fs.Position = entryPos;
                uint ofsMcnk = br.ReadUInt32();
                uint size = br.ReadUInt32();
                if (ofsMcnk != 0 && size != 0)
                {
                    list.Add(ofsMcnk);
                }
            }
            return list;
        }

        // First try MHDR -> OfsMCIN
        fs.Position = start;
        while (fs.Position + 8 <= end)
        {
            long blockPos = fs.Position;
            uint fourcc;
            try { fourcc = br.ReadUInt32(); } catch { break; }
            uint size;
            try { size = br.ReadUInt32(); } catch { break; }
            long payload = fs.Position;
            long blockEnd = payload + size;
            if (blockEnd > end || size > int.MaxValue)
            {
                fs.Position = blockPos + 4;
                continue;
            }
            if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MHDR))
            {
                if (payload + 8 <= blockEnd)
                {
                    fs.Position = payload + 0; // Alpha MHDR: first int is mcinOffset
                    uint ofsMcin = br.ReadUInt32();
                    long mcinPos = payload + ofsMcin; // relative to MHDR payload base
                    if (mcinPos + 8 <= end)
                    {
                        offsets = ReadMcinAt(mcinPos);
                        return offsets;
                    }
                }
                break;
            }
            fs.Position = blockEnd;
        }

        // Fallback: scan for MCIN within tile
        fs.Position = start;
        while (fs.Position + 8 <= end)
        {
            long pos = fs.Position;
            uint fourcc;
            try { fourcc = br.ReadUInt32(); } catch { break; }
            uint size;
            try { size = br.ReadUInt32(); } catch { break; }
            long payload = fs.Position;
            long blockEnd = payload + size;
            if (blockEnd > end || size > int.MaxValue)
            {
                fs.Position = pos + 4;
                continue;
            }
            if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MCIN))
            {
                offsets = ReadMcinAt(pos);
                return offsets;
            }
            fs.Position = blockEnd;
        }

        return offsets;
    }
}
