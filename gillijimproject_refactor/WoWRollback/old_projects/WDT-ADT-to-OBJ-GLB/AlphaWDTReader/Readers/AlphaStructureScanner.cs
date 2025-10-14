using System.Text;
using AlphaWDTReader.IO;
using AlphaWDTReader.Model;
using AlphaWDTReader.ReferencePort;

namespace AlphaWDTReader.Readers;

public static class AlphaStructureScanner
{
    public static StructureFile ScanFile(string filePath)
    {
        var file = new StructureFile { FilePath = filePath };
        var tiles = AlphaMainReader.ReadMainTable(filePath);
        file.Tiles = new List<StructureTile>();
        using var fs = File.OpenRead(filePath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
        long fileEnd = fs.Length;

        // Precompute per-tile limit end: if Size==0, bound to the start of the next tile by offset
        var present = tiles.Where(t => t.Offset > 0).OrderBy(t => t.Offset).ToList();
        var nextByOffset = new Dictionary<(int x, int y), long>();
        for (int i = 0; i < present.Count; i++)
        {
            var cur = present[i];
            long nextStart = (i + 1 < present.Count) ? present[i + 1].Offset : fileEnd;
            nextByOffset[(cur.TileX, cur.TileY)] = nextStart;
        }

        foreach (var t in tiles)
        {
            if (t.Offset <= 0) continue;
            long tileStart = t.Offset;
            long nominalEnd = t.Size > 0 ? (tileStart + t.Size) : fileEnd;
            long boundedEnd = nominalEnd;
            if (t.Size == 0 && nextByOffset.TryGetValue((t.TileX, t.TileY), out var nextStart))
            {
                // Bound this tile by the next tile's start to avoid scanning entire file
                boundedEnd = Math.Min(fileEnd, Math.Max(tileStart, nextStart));
            }
            long limitEnd = Math.Min(fileEnd, boundedEnd);
            var tile = new StructureTile { X = t.TileX, Y = t.TileY, Blocks = new List<StructureBlock>() };

            long pos = tileStart;
            // Safety: cap iterations to prevent runaway in corrupt data
            int iter = 0, maxIter = 2_000_000; // generous upper bound
            while (pos + 8 <= limitEnd && iter++ < maxIter)
            {
                fs.Position = pos;
                uint four;
                uint size;
                try { four = AlphaFourCC.ReadFourCC(br); } catch { break; }
                try { size = br.ReadUInt32(); } catch { break; }
                long payload = fs.Position;

                // Default blockEnd assuming well-formed size
                long blockEnd = payload + size;

                // Special handling: Alpha MCNK may report size==0 but still contain header+subchunks region
                if (size == 0 && AlphaFourCC.Matches(four, AlphaFourCC.MCNK))
                {
                    const int headerSize = 0x80;
                    if (payload + headerSize > limitEnd)
                    {
                        // Not enough bytes for MCNK header; abort tile safely
                        break;
                    }

                    // Read header to get chunksSize
                    fs.Position = payload;
                    if (!McnkHeader.TryRead(br, out var hdr))
                    {
                        break;
                    }
                    int chunksSize = hdr.ChunksSize;
                    long regionStart = payload + headerSize;
                    long regionEnd = regionStart;
                    if (chunksSize > 0)
                    {
                        regionEnd = Math.Min(limitEnd, regionStart + chunksSize);
                    }

                    // Define a conservative block end so we can advance the cursor
                    blockEnd = Math.Min(limitEnd, Math.Max(regionEnd, regionStart));

                    var block = new StructureBlock
                    {
                        FourCC = AlphaFourCC.ToDisplayString(four),
                        Offset = pos,
                        Size = size,
                        PayloadStart = payload,
                        BlockEnd = blockEnd,
                        Subchunks = ScanMcnkSubchunks(fs, br, payload, blockEnd)
                    };

                    tile.Blocks.Add(block);
                    pos = blockEnd;
                    continue;
                }

                // If size is invalid or runs past the tile, try to stop safely
                if (size == 0 || size > int.MaxValue || blockEnd > limitEnd)
                {
                    // For non-MCNK zero-sized or invalid blocks, advance minimally to avoid infinite loop
                    pos = Math.Min(limitEnd, pos + 8);
                    continue;
                }

                var block2 = new StructureBlock
                {
                    FourCC = AlphaFourCC.ToDisplayString(four),
                    Offset = pos,
                    Size = size,
                    PayloadStart = payload,
                    BlockEnd = blockEnd,
                    Subchunks = AlphaFourCC.Matches(four, AlphaFourCC.MCNK) ? ScanMcnkSubchunks(fs, br, payload, blockEnd) : new List<StructureSubchunk>()
                };

                tile.Blocks.Add(block2);
                pos = blockEnd;
            }

            file.Tiles.Add(tile);
        }

        return file;
    }

    // Reusable helper: enumerate MCNK subchunks using header.chunksSize constraints.
    // - fs/br must be open on the ADT file.
    // - mcnkPayloadStart: start of the MCNK payload (immediately after the MCNK size field)
    // - mcnkBlockEnd: end of the MCNK block region (payloadStart + size) or conservative bound
    public static List<StructureSubchunk> ScanMcnkSubchunks(Stream fs, BinaryReader br, long mcnkPayloadStart, long mcnkBlockEnd)
    {
        const int headerSize = 0x80;
        var list = new List<StructureSubchunk>();
        if (mcnkPayloadStart < 0 || mcnkBlockEnd <= mcnkPayloadStart || mcnkPayloadStart + headerSize > fs.Length)
            return list;

        long payload = mcnkPayloadStart;
        fs.Position = payload;
        if (!McnkHeader.TryRead(br, out var hdr)) return list;
        int chunksSize = hdr.ChunksSize;
        if (chunksSize <= 0) return list;

        long regionStart = payload + headerSize;
        long regionEnd = Math.Min(mcnkBlockEnd, regionStart + chunksSize);
        long cur = regionStart;
        while (cur + 8 <= regionEnd)
        {
            fs.Position = cur;
            uint four2;
            uint size2;
            try { four2 = br.ReadUInt32(); } catch { break; }
            try { size2 = br.ReadUInt32(); } catch { break; }
            long dataStart = fs.Position;
            long dataEnd = dataStart + size2;
            if (size2 == 0 || size2 > int.MaxValue || dataEnd > regionEnd) break;
            list.Add(new StructureSubchunk
            {
                Tag = AlphaFourCC.ToDisplayString(four2),
                Offset = cur,
                Size = size2,
                PayloadStart = dataStart,
                BlockEnd = dataEnd,
                IsDerived = false,
            });
            cur = dataEnd;
        }
        // Add derived MCLQ region if present in Alpha header (no FourCC tag in Alpha)
        if (hdr.MclqRel > 0)
        {
            long mclqStart = regionStart + hdr.MclqRel;
            if (mclqStart >= regionStart && mclqStart < regionEnd)
            {
                // Determine derived size up to next known subchunk after mclqStart; fallback to regionEnd
                long next = regionEnd;
                foreach (var sc in list)
                {
                    if (sc.Offset > mclqStart && sc.Offset < next) next = sc.Offset;
                }
                long derivedEnd = Math.Min(regionEnd, next);
                if (derivedEnd > mclqStart)
                {
                    list.Add(new StructureSubchunk
                    {
                        Tag = "MCLQ(derived)",
                        Offset = mclqStart,
                        Size = (uint)Math.Min(int.MaxValue, derivedEnd - mclqStart),
                        PayloadStart = mclqStart + 0, // no extra header; payload starts at offset
                        BlockEnd = derivedEnd,
                        IsDerived = true,
                    });
                }
            }
        }
        return list;
    }
}

public sealed class StructureFile
{
    public string FilePath { get; set; } = string.Empty;
    public List<StructureTile> Tiles { get; set; } = new();
}

public sealed class StructureTile
{
    public int X { get; set; }
    public int Y { get; set; }
    public List<StructureBlock> Blocks { get; set; } = new();
}

public sealed class StructureBlock
{
    public string FourCC { get; set; } = string.Empty;
    public long Offset { get; set; }
    public uint Size { get; set; }
    public long PayloadStart { get; set; }
    public long BlockEnd { get; set; }
    public List<StructureSubchunk> Subchunks { get; set; } = new();
}

public sealed class StructureSubchunk
{
    public string Tag { get; set; } = string.Empty;
    public long Offset { get; set; }
    public uint Size { get; set; }
    public long PayloadStart { get; set; }
    public long BlockEnd { get; set; }
    public bool IsDerived { get; set; }
}
