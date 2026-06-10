using System.Text;
using AlphaWDTReader.Model;
using AlphaWDTReader.IO;

namespace AlphaWDTReader.Readers;

public static class AlphaChunkIndexer
{
    public static List<AlphaChunkIndex> BuildForTile(string filePath, AlphaMainEntry tile)
    {
        var indices = new List<AlphaChunkIndex>();
        var chunkOffsets = AlphaChunkIndexReader.GetChunkOffsets(filePath, tile);
        if (chunkOffsets.Count == 0) return indices;

        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long tileStart = tile.Offset;
        long fileEnd = fs.Length;
        long limitEnd = tile.Size > 0 ? Math.Min(fileEnd, tileStart + tile.Size) : fileEnd; // Alpha: Size may be 0

        foreach (var ofs in chunkOffsets)
        {
            if (ofs < 0 || ofs + 8 > limitEnd) continue;
            fs.Position = ofs;
            uint four = br.ReadUInt32();
            uint size = br.ReadUInt32();
            long payload = fs.Position;
            long blockEnd = payload + size;
            if (!AlphaFourCC.Matches(four, AlphaFourCC.MCNK) || size == 0 || blockEnd > limitEnd) continue;

            // Alpha MCNK header is 0x80 bytes at the start of the MCNK payload
            const int headerSize = 0x80;
            if (payload + headerSize > blockEnd) continue;
            fs.Position = payload;
            var header = br.ReadBytes(headerSize);
            int mcvtRel = BitConverter.ToInt32(header, 0x18);
            int mcnrRel = BitConverter.ToInt32(header, 0x1C);
            int mclqRel = BitConverter.ToInt32(header, 0x68);
            int chunksSize = BitConverter.ToInt32(header, 0x64);

            long foundMcvt = 0, foundMcnr = 0, foundMclq = 0;
            uint sizeMcvt = 0, sizeMcnr = 0, sizeMclq = 0;

            if (mcvtRel >= 0)
            {
                long mcvtAbs = payload + headerSize + mcvtRel;
                if (mcvtAbs + 580 <= blockEnd) { foundMcvt = mcvtAbs; sizeMcvt = 580; }
            }
            if (mcnrRel >= 0)
            {
                long mcnrAbs = payload + headerSize + mcnrRel;
                // Alpha MCNR payload commonly 448 bytes
                if (mcnrAbs + 448 <= blockEnd) { foundMcnr = mcnrAbs; sizeMcnr = 448; }
            }
            if (mclqRel > 0 && chunksSize > 0)
            {
                long mclqAbs = payload + headerSize + mclqRel;
                int mclqLen = Math.Max(0, chunksSize - mclqRel);
                if (mclqLen > 0 && mclqAbs + mclqLen <= blockEnd) { foundMclq = mclqAbs; sizeMclq = (uint)mclqLen; }
            }

            indices.Add(new AlphaChunkIndex
            {
                ChunkFileOffset = ofs,
                ChunkSize = size,
                OfsMCVT = foundMcvt,
                OfsMCNR = foundMcnr,
                OfsMCLQ = foundMclq,
                SizeMCVT = sizeMcvt,
                SizeMCNR = sizeMcnr,
                SizeMCLQ = sizeMclq,
            });
        }

        return indices;
    }
}
