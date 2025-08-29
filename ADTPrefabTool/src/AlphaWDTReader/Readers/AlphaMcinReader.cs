using System.Text;
using AlphaWDTReader.Model;

namespace AlphaWDTReader.Readers;

public static class AlphaMcinReader
{
    public static int? CountPresentChunks(string filePath, AlphaMainEntry tile)
    {
        if (tile.Offset == 0 || tile.Size == 0) return null;
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long start = tile.Offset;
        long end = Math.Min(fs.Length, start + tile.Size);
        if (start >= fs.Length || start + 8 > end) return null;
        fs.Position = start;

        const uint MHDR = 0x5244484Du; // 'MHDR'
        const uint MCIN = 0x4E49434Du; // 'MCIN'

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

            if (fourcc == MHDR)
            {
                // MHDR payload: first 4 bytes Flags, next 4 bytes OfsMCIN (relative to ADT start)
                if (payload + 8 <= blockEnd)
                {
                    fs.Position = payload + 4;
                    uint ofsMcin = br.ReadUInt32();
                    long mcinPos = start + ofsMcin;
                    if (mcinPos + 8 <= end)
                    {
                        fs.Position = mcinPos;
                        uint mcinFour = br.ReadUInt32();
                        uint mcinSize = br.ReadUInt32();
                        long mcinPayload = fs.Position;
                        long mcinEnd = mcinPayload + mcinSize;
                        if (mcinFour == MCIN && mcinEnd <= end && mcinSize > 0)
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

            if (fourcc == MCIN)
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
