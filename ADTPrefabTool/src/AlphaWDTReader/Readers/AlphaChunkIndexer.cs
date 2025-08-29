using System.Text;
using AlphaWDTReader.Model;

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
        long tileEnd = Math.Min(fs.Length, tileStart + tile.Size);

        const uint MCNK = 0x4B4E434Du; // 'MCNK'
        const uint MCVT = 0x5456434Du; // 'MCVT'
        const uint MCNR = 0x524E434Du; // 'MCNR'
        const uint MCLQ = 0x514C434Du; // 'MCLQ'

        foreach (var ofs in chunkOffsets)
        {
            if (ofs + 8 > tileEnd) continue;
            fs.Position = ofs;
            uint four = br.ReadUInt32();
            uint size = br.ReadUInt32();
            long payload = fs.Position;
            long blockEnd = payload + size;
            if (four != MCNK || blockEnd > tileEnd || size == 0) continue;

            long foundMcvt = 0, foundMcnr = 0, foundMclq = 0;
            uint sizeMcvt = 0, sizeMcnr = 0, sizeMclq = 0;

            long scanPos = payload;
            while (scanPos + 8 <= blockEnd)
            {
                fs.Position = scanPos;
                uint subFour = br.ReadUInt32();
                uint subSize = br.ReadUInt32();
                long subPayload = fs.Position;
                long subEnd = subPayload + subSize;
                if (subEnd > blockEnd) break;
                if (subFour == MCVT) { foundMcvt = subPayload; sizeMcvt = subSize; }
                else if (subFour == MCNR) { foundMcnr = subPayload; sizeMcnr = subSize; }
                else if (subFour == MCLQ) { foundMclq = subPayload; sizeMclq = subSize; }
                scanPos = subEnd;
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
