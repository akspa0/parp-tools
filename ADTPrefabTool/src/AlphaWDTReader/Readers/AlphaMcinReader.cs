using System.Text;
using AlphaWDTReader.Model;
using AlphaWDTReader.IO;

namespace AlphaWDTReader.Readers;

public static class AlphaMcinReader
{
    public static int? CountPresentChunks(string filePath, AlphaMainEntry tile)
    {
        if (tile.Offset == 0) return null;
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long start = tile.Offset;
        long end = (tile.Size == 0) ? fs.Length : Math.Min(fs.Length, start + tile.Size);
        if (start >= fs.Length || start + 8 > end) return null;
        fs.Position = start;

        // First pass: find MHDR in tile region
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
                fs.Position = blockPos + 4; // resync
                continue;
            }

            if (AlphaFourCC.Matches(fourcc, AlphaFourCC.MHDR))
            {
                // Alpha MHDR payload: first 4 bytes is mcinOffset (relative to ADT start)
                if (payload + 8 <= blockEnd)
                {
                    fs.Position = payload + 0;
                    uint ofsMcin = br.ReadUInt32();
                    long mcinPos = start + ofsMcin;
                    if (mcinPos + 8 <= end)
                    {
                        fs.Position = mcinPos;
                        uint mcinFour = br.ReadUInt32();
                        uint mcinSize = br.ReadUInt32();
                        long mcinPayload = fs.Position;
                        long mcinEnd = mcinPayload + mcinSize;
                        if (AlphaFourCC.Matches(mcinFour, AlphaFourCC.MCIN) && mcinEnd <= end && mcinSize > 0)
                        {
                            int entrySize = (int)(mcinSize / 256);
                            if (entrySize <= 0) return 0;
                            int count = 0;
                            for (int i = 0; i < 256; i++)
                            {
                                long entryPos = mcinPayload + i * entrySize;
                                if (entryPos + 4 > mcinEnd) break;
                                fs.Position = entryPos;
                                uint ofsMcnk = br.ReadUInt32();
                                if (ofsMcnk != 0) count++;
                            }
                            return count;
                        }
                    }
                }
                // If MHDR path failed, break to fallback scan
                break;
            }

            fs.Position = blockEnd;
        }

        // Fallback: scan for MCIN within tile region (previous behavior)
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
                int entrySize = (int)(size / 256);
                if (entrySize <= 0) return null;
                int count = 0;
                for (int i = 0; i < 256; i++)
                {
                    long entryPos = payload + i * entrySize;
                    if (entryPos + 4 > blockEnd) break;
                    fs.Position = entryPos;
                    uint ofsMcnk = br.ReadUInt32();
                    if (ofsMcnk != 0) count++;
                }
                return count;
            }

            fs.Position = blockEnd;
        }

        return null; // not found
    }
}
