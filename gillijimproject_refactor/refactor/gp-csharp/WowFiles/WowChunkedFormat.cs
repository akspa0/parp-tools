using GillijimProject.Utilities;

namespace GillijimProject.WowFiles;

public static class WowChunkedFormat
{
    public record ChunkIndex(uint Tag, string TagDisplay, long Offset, uint Size);

    public static IEnumerable<ChunkHeader> EnumerateTop(byte[] data)
    {
        long off = 0;
        while (off + 8 <= data.Length)
        {
            var ch = Chunk.ReadHeader(data, off);
            yield return ch;
            off = ch.PayloadOffset + ch.Size;
        }
    }

    public static ChunkHeader RequireTop(byte[] data, uint tag)
    {
        foreach (var ch in EnumerateTop(data)) if (ch.Tag == tag) return ch;
        throw new InvalidDataException($"Missing top-level chunk {Util.FourCcToDisplay(tag)}");
    }

    // Build comprehensive index of all potential FOURCC chunks in file
    public static List<ChunkIndex> IndexAllChunks(Stream s)
    {
        var chunks = new List<ChunkIndex>();
        long length = s.Length;
        s.Seek(0, SeekOrigin.Begin);
        
        const int bufferSize = 1 << 20; // 1MB buffer
        byte[] buffer = new byte[bufferSize];
        long pos = 0;
        
        Console.WriteLine($"Indexing chunks in {length:N0} byte file...");
        
        while (pos < length)
        {
            int bytesToRead = (int)Math.Min(bufferSize, length - pos);
            s.Seek(pos, SeekOrigin.Begin);
            int bytesRead = s.Read(buffer, 0, bytesToRead);
            if (bytesRead <= 0) break;
            
            Span<byte> span = buffer.AsSpan(0, bytesRead);
            
            // Scan for potential FOURCC chunks (4 ASCII chars + valid size)
            for (int i = 0; i <= bytesRead - 8; i++)
            {
                uint tag = Util.ReadUInt32LE(span, i);
                uint size = Util.ReadUInt32LE(span, i + 4);
                
                // Check if tag looks like valid WoW FOURCC chunk name
                if (IsValidFourCC(tag))
                {
                    long chunkOffset = pos + i;
                    long payloadEnd = chunkOffset + 8 + size;
                    
                    // Basic sanity checks
                    if (payloadEnd <= length && size > 0 && size < length)
                    {
                        chunks.Add(new ChunkIndex(tag, Util.FourCcToDisplay(tag), chunkOffset, size));
                    }
                }
            }
            
            pos += bytesRead;
            if (pos % (10 * 1024 * 1024) == 0) // Progress every 10MB
            {
                Console.WriteLine($"  Scanned {pos:N0} / {length:N0} bytes ({100.0 * pos / length:F1}%)");
            }
        }
        
        Console.WriteLine($"Found {chunks.Count:N0} potential chunks");
        return chunks;
    }
    
    // Known WoW chunk types from Alpha.md and ADT_v18.md documentation
    private static readonly HashSet<uint> KnownChunks = new()
    {
        Tags.MVER, // Map Version
        Tags.MPHD, // Map Header
        Tags.MAIN, // Map Tile Table
        Tags.MDNM, // Doodad Names
        Tags.MONM, // Map Object Names
        Tags.MODF, // Map Object Definition
        Tags.MHDR, // Area Header
        Tags.MCIN, // Chunk Index
        0x4D544558, // MTEX - Textures
        Tags.MDDF, // Doodad Definition
        Tags.MCNK, // Map Chunk
        Tags.MCVT, // Vertices
        Tags.MCNR, // Normals
        0x4D434C59, // MCLY - Layers
        0x4D435246, // MCRF - References
        0x4D435348, // MCSH - Shadow
        0x4D43414C, // MCAL - Alpha
        Tags.MCLQ, // Liquid
        Tags.MCSE, // Sound Emitters
        Tags.MCBB, // Bounding Box
        Tags.MCCV, // Vertex Colors
        Tags.MMDX, // Model Names
        Tags.MMID, // Model IDs
        Tags.MWMO, // WMO Names
        Tags.MWID, // WMO IDs
    };

    // Check if uint32 represents a known WoW chunk
    private static bool IsValidFourCC(uint tag)
    {
        return KnownChunks.Contains(tag);
    }

    // Find MAIN chunk using comprehensive index
    public static ChunkHeader FindMain(Stream s)
    {
        var allChunks = IndexAllChunks(s);
        
        // Print summary of chunk types found
        var chunkGroups = allChunks.GroupBy(c => c.TagDisplay).OrderBy(g => g.Key);
        Console.WriteLine("\nChunk types found:");
        foreach (var group in chunkGroups)
        {
            var chunks = group.ToList();
            Console.WriteLine($"  {group.Key}: {chunks.Count} instances");
            
            // Show first few instances with offsets for key chunks
            if (group.Key == "MAIN" || group.Key == "MHDR" || group.Key == "MVER" || group.Key == "MPHD")
            {
                foreach (var chunk in chunks.Take(3))
                {
                    Console.WriteLine($"    @0x{chunk.Offset:X8} size={chunk.Size:N0}");
                }
                if (chunks.Count > 3)
                {
                    Console.WriteLine($"    ... and {chunks.Count - 3} more");
                }
            }
        }
        
        // Look for MAIN chunks
        var mainChunks = allChunks.Where(c => c.Tag == Tags.MAIN).ToList();
        if (mainChunks.Count == 0)
        {
            throw new InvalidDataException("Missing top-level chunk MAIN");
        }
        
        Console.WriteLine($"\nFound {mainChunks.Count} MAIN chunks:");
        foreach (var main in mainChunks)
        {
            Console.WriteLine($"  MAIN at offset 0x{main.Offset:X8}, size: {main.Size:N0}");
        }
        
        // Use first MAIN chunk found
        var firstMain = mainChunks[0];
        return new ChunkHeader(firstMain.Tag, firstMain.Size, firstMain.Offset + 8);
    }
}
