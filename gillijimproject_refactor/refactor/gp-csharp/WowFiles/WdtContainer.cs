using GillijimProject.Utilities;
using GillijimProject.WowFiles.Terrain;
using GillijimProject.WowFiles.Objects;

namespace GillijimProject.WowFiles;

/// <summary>
/// Complete WDT container that preserves all chunks for 1:1 recompilation
/// </summary>
public sealed class WdtContainer
{
    public List<IChunkData> AllChunks { get; }
    public Dictionary<uint, List<IChunkData>> ChunksByType { get; }
    public long OriginalFileSize { get; }
    
    private WdtContainer(List<IChunkData> allChunks, long originalFileSize)
    {
        AllChunks = allChunks;
        OriginalFileSize = originalFileSize;
        ChunksByType = allChunks.GroupBy(c => c.Tag).ToDictionary(g => g.Key, g => g.ToList());
    }
    
    public static WdtContainer LoadComplete(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return LoadComplete(fs);
    }
    
    public static WdtContainer LoadComplete(Stream stream)
    {
        var allChunks = new List<IChunkData>();
        long fileSize = stream.Length;
        
        stream.Seek(0, SeekOrigin.Begin);
        
        // Phase 1: Parse WDT header chunks (MVER, MPHD, MAIN, MDNM, MONM, MMID, MMDX, MWMO)
        var mainChunk = ParseHeaderChunks(stream, allChunks);
        
        // Phase 2: Parse tiles using MAIN chunk offsets
        if (mainChunk != null)
        {
            ParseTilesFromMain(stream, allChunks, mainChunk);
        }
        
        return new WdtContainer(allChunks, fileSize);
    }
    
    private static IChunkData? ParseHeaderChunks(Stream stream, List<IChunkData> allChunks)
    {
        var buffer = new byte[8];
        long position = 0;
        long fileSize = stream.Length;
        IChunkData? mainChunk = null;
        
        // Parse header chunks until we hit tile data
        while (position < fileSize - 8)
        {
            stream.Seek(position, SeekOrigin.Begin);
            int read = stream.Read(buffer, 0, 8);
            if (read < 8) break;
            
            uint tag = BitConverter.ToUInt32(buffer, 0);
            uint size = BitConverter.ToUInt32(buffer, 4);
            
            // Only parse header chunks
            if (!IsHeaderChunk(tag))
                break;
            
            if (!IsValidChunkTag(tag) || size > fileSize - position - 8)
            {
                position++;
                continue;
            }
            
            try
            {
                var chunk = ParseChunk(stream, tag, position, size);
                if (chunk != null)
                {
                    allChunks.Add(chunk);
                    if (tag == Tags.MAIN)
                        mainChunk = chunk;
                    
                    position += 8 + size;
                    while (position % 4 != 0)
                        position++;
                }
                else
                {
                    position++;
                }
            }
            catch
            {
                position++;
            }
        }
        
        return mainChunk;
    }
    
    private static void ParseTilesFromMain(Stream stream, List<IChunkData> allChunks, IChunkData mainChunk)
    {
        // Parse MAIN chunk to get tile offsets
        var mainData = mainChunk.RawData.Span;
        
        Console.WriteLine($"MAIN chunk size: {mainData.Length} bytes");
        int validTiles = 0;
        
        // MAIN contains 64x64 = 4096 entries of SMAreaInfo (16 bytes each)
        for (int i = 0; i < 4096; i++)
        {
            int entryOffset = i * 16;
            if (entryOffset + 16 > mainData.Length) break;
            
            uint tileOffset = BitConverter.ToUInt32(mainData[entryOffset..(entryOffset + 4)]);
            uint tileSize = BitConverter.ToUInt32(mainData[(entryOffset + 4)..(entryOffset + 8)]);
            uint flags = BitConverter.ToUInt32(mainData[(entryOffset + 8)..(entryOffset + 12)]);
            
            // Skip empty tiles
            if (tileOffset == 0 || tileSize == 0)
                continue;
            
            // Calculate XX_YY position from index (64x64 grid)
            int tileX = i % 64;
            int tileY = i / 64;
            
            validTiles++;
            if (validTiles <= 5) // Debug first 5 tiles
            {
                Console.WriteLine($"Tile {i} ({tileX:D2}_{tileY:D2}): offset={tileOffset}, size={tileSize}, flags={flags:X8}");
            }
            
            // Parse tile at offset
            ParseTileAtOffset(stream, allChunks, tileOffset, tileSize, tileX, tileY);
        }
        
        Console.WriteLine($"Found {validTiles} valid tiles in MAIN chunk");
    }
    
    private static void ParseTileAtOffset(Stream stream, List<IChunkData> allChunks, uint tileOffset, uint tileSize, int tileX, int tileY)
    {
        var buffer = new byte[8];
        long position = tileOffset;
        long tileEnd = tileOffset + tileSize;
        int chunksInTile = 0;
        
        Console.WriteLine($"Parsing tile {tileX:D2}_{tileY:D2} at offset {tileOffset}, size {tileSize}");
        
        // First, parse MHDR chunk which should be at the start of each tile
        stream.Seek(position, SeekOrigin.Begin);
        int read = stream.Read(buffer, 0, 8);
        if (read < 8) return;
        
        uint tag = BitConverter.ToUInt32(buffer, 0);
        uint size = BitConverter.ToUInt32(buffer, 4);
        
        if (tag == Tags.MHDR)
        {
            Console.WriteLine($"  Found MHDR at offset {position}, size {size}");
            var mhdrChunk = ParseChunk(stream, tag, position, size);
            if (mhdrChunk != null)
            {
                allChunks.Add(mhdrChunk);
                chunksInTile++;
                
                // MHDR contains offsets to other chunks within this tile
                // For now, skip to after MHDR and continue sequential parsing
                position += 8 + size;
                while (position % 4 != 0) position++;
            }
        }
        
        // Parse remaining chunks in tile sequentially
        while (position < tileEnd - 8)
        {
            stream.Seek(position, SeekOrigin.Begin);
            read = stream.Read(buffer, 0, 8);
            if (read < 8) break;
            
            tag = BitConverter.ToUInt32(buffer, 0);
            size = BitConverter.ToUInt32(buffer, 4);
            
            // Validate chunk tag and size
            if (!IsValidChunkTag(tag) || size > tileEnd - position - 8 || size > 100_000_000)
            {
                position++;
                continue;
            }
            
            try
            {
                var chunk = ParseChunk(stream, tag, position, size);
                if (chunk != null)
                {
                    allChunks.Add(chunk);
                    chunksInTile++;
                    position += 8 + size;
                    
                    var tagStr = System.Text.Encoding.ASCII.GetString(BitConverter.GetBytes(tag));
                    Console.WriteLine($"  Found chunk '{tagStr}' at offset {position - 8 - size}, size {size}");
                    
                    while (position % 4 != 0)
                        position++;
                }
                else
                {
                    position++;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error parsing chunk at {position}: {ex.Message}");
                position++;
            }
        }
        
        Console.WriteLine($"Tile {tileX:D2}_{tileY:D2}: Parsed {chunksInTile} chunks");
    }
    
    private static bool IsHeaderChunk(uint tag)
    {
        return tag == Tags.MVER || tag == Tags.MPHD || tag == Tags.MAIN || 
               tag == Tags.MDNM || tag == Tags.MONM || tag == Tags.MMID ||
               tag == Tags.MMDX || tag == Tags.MWMO;
    }
    
    private static bool IsValidChunkTag(uint tag)
    {
        // Only accept documented WDT chunk tags
        return tag switch
        {
            Tags.MVER => true,
            Tags.MPHD => true,
            Tags.MAIN => true,
            Tags.MWMO => true,
            Tags.MWID => true,
            Tags.MODF => true,
            Tags.MMDX => true,
            Tags.MMID => true,
            Tags.MDDF => true,
            Tags.MCNK => true,
            Tags.MCLY => true,
            Tags.MCRF => true,
            Tags.MCAL => true,
            Tags.MCSH => true,
            Tags.MCCV => true,
            Tags.MCLQ => true,
            Tags.MCSE => true,
            Tags.MTEX => true,
            Tags.MDNM => true,
            Tags.MONM => true,
            _ => false
        };
    }
    
    private static IChunkData? ParseChunk(Stream stream, uint tag, long offset, uint size)
    {
        // Read raw chunk data
        var rawData = new byte[size];
        stream.Seek(offset + 8, SeekOrigin.Begin);
        stream.Read(rawData, 0, (int)size);
        
        // Parse known chunk types with structured data + raw preservation
        return tag switch
        {
            Tags.MTEX => GillijimProject.WowFiles.Terrain.MtexAlpha.Parse(stream, offset),
            Tags.MCLY => GillijimProject.WowFiles.Terrain.MclyAlpha.Parse(stream, offset),
            Tags.MODF => GillijimProject.WowFiles.Objects.ModfAlpha.Parse(stream, offset),
            Tags.MDDF => GillijimProject.WowFiles.Objects.MddfAlpha.Parse(stream, offset),
            Tags.MCRF => GillijimProject.WowFiles.Terrain.McrfAlpha.Parse(stream, offset),
            Tags.MMID => GillijimProject.WowFiles.Objects.MmidAlpha.Parse(stream, offset),
            Tags.MDNM => GillijimProject.WowFiles.Objects.MdnmAlpha.Parse(stream, offset),
            Tags.MONM => GillijimProject.WowFiles.Objects.MonmAlpha.Parse(stream, offset),
            Tags.MMDX => GillijimProject.WowFiles.Objects.MmdxAlpha.Parse(stream, offset),
            Tags.MWMO => GillijimProject.WowFiles.Objects.MwmoAlpha.Parse(stream, offset),
            Tags.MCNK => GillijimProject.WowFiles.Terrain.McnkAlpha.Parse(stream, offset),
            
            // Unknown chunks - preserve as raw data
            _ => new UnknownChunk(tag, rawData, offset)
        };
    }
    
    /// <summary>
    /// Get all chunks of a specific type
    /// </summary>
    public List<T> GetChunks<T>() where T : class, IChunkData
    {
        return AllChunks.OfType<T>().ToList();
    }
    
    /// <summary>
    /// Get chunks by tag
    /// </summary>
    public List<IChunkData> GetChunksByTag(uint tag)
    {
        return ChunksByType.TryGetValue(tag, out var chunks) ? chunks : new List<IChunkData>();
    }
    
    /// <summary>
    /// Serialize entire WDT back to binary format
    /// </summary>
    public byte[] ToBytes()
    {
        // Calculate total size without padding
        var totalSize = AllChunks.Sum(c => c.ToBytes().Length);
        var result = new byte[totalSize];
        int position = 0;
        
        foreach (var chunk in AllChunks)
        {
            var chunkBytes = chunk.ToBytes();
            chunkBytes.CopyTo(result, position);
            position += chunkBytes.Length;
        }
        
        return result;
    }
    
    /// <summary>
    /// Save WDT to file with 1:1 binary preservation
    /// </summary>
    public void SaveToFile(string path)
    {
        File.WriteAllBytes(path, ToBytes());
    }
    
    /// <summary>
    /// Get statistics about chunk distribution
    /// </summary>
    public Dictionary<string, int> GetChunkStatistics()
    {
        return ChunksByType.ToDictionary(
            kvp => ChunkSerializer.TagToString(kvp.Key),
            kvp => kvp.Value.Count
        );
    }
}
