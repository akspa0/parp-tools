using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Extracts texture-related chunks (MTEX, MCLY, MCAL, MCSH) from monolithic ADTs
/// and formats them for use as tex0-style data in the split ADT merger.
/// </summary>
public sealed class TextureDataExtractor
{
    /// <summary>
    /// Extract texture chunks from a monolithic ADT and return them in a format
    /// compatible with the AdtPatcher's tex0 parsing (headerless MCNK subchunks).
    /// </summary>
    /// <param name="monoAdtData">Raw bytes of a monolithic ADT file</param>
    /// <returns>Combined texture data (MTEX + per-MCNK texture subchunks), or null if no texture data found</returns>
    public byte[]? ExtractTextureChunks(byte[] monoAdtData)
    {
        var chunks = ParseChunks(monoAdtData);
        
        // Get global MTEX chunk
        var mtexData = chunks.GetValueOrDefault("MTEX");
        if (mtexData == null || mtexData.Length == 0)
        {
            Console.WriteLine("  [TextureExtractor] No MTEX chunk found");
            return null;
        }
        
        // Parse texture filenames for logging
        var textures = ParseMtexFilenames(mtexData);
        Console.WriteLine($"  [TextureExtractor] Found {textures.Count} textures in MTEX");
        
        // Find all MCNK chunks and extract their texture subchunks
        var mcnkTextureData = new List<byte[]>();
        int mcnkCount = 0;
        
        int pos = 0;
        while (pos < monoAdtData.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(monoAdtData, pos, 4);
            int size = BitConverter.ToInt32(monoAdtData, pos + 4);
            
            if (size < 0 || pos + 8 + size > monoAdtData.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            if (readable == "MCNK" && size >= 128)
            {
                // Extract texture subchunks from this MCNK
                var mcnkData = new byte[size];
                Buffer.BlockCopy(monoAdtData, pos + 8, mcnkData, 0, size);
                
                var textureSubchunks = ExtractMcnkTextureSubchunks(mcnkData);
                mcnkTextureData.Add(textureSubchunks);
                mcnkCount++;
            }
            
            pos += 8 + size;
        }
        
        Console.WriteLine($"  [TextureExtractor] Extracted texture data from {mcnkCount} MCNKs");
        
        if (mcnkCount == 0)
            return null;
        
        // Build output: MTEX chunk followed by per-MCNK texture data
        // The AdtPatcher expects tex0 to have MTEX at the top, then MCNK chunks with texture subchunks
        using var ms = new MemoryStream();
        
        // Write MTEX chunk
        WriteChunk(ms, "MTEX", mtexData);
        
        // Write MCNK-style chunks containing only texture subchunks (no header)
        // Actually, tex0 format has headerless MCNKs - just the subchunks directly
        // But we need to wrap them so the parser can find them
        foreach (var texData in mcnkTextureData)
        {
            WriteChunk(ms, "MCNK", texData);
        }
        
        return ms.ToArray();
    }

    /// <summary>
    /// Extract texture-related subchunks (MCLY, MCAL, MCSH) from a single MCNK.
    /// Returns headerless subchunk data suitable for tex0 format.
    /// </summary>
    private byte[] ExtractMcnkTextureSubchunks(byte[] mcnkData)
    {
        // MCNK has 128-byte header, then subchunks
        if (mcnkData.Length < 128)
            return Array.Empty<byte>();
        
        var subchunks = ParseSubchunks(mcnkData, 128);
        
        using var ms = new MemoryStream();
        
        // Write texture-related subchunks in order
        if (subchunks.TryGetValue("MCLY", out var mcly) && mcly.Length > 0)
            WriteChunk(ms, "MCLY", mcly);
        
        if (subchunks.TryGetValue("MCAL", out var mcal) && mcal.Length > 0)
            WriteChunk(ms, "MCAL", mcal);
        
        if (subchunks.TryGetValue("MCSH", out var mcsh) && mcsh.Length > 0)
            WriteChunk(ms, "MCSH", mcsh);
        
        return ms.ToArray();
    }

    /// <summary>
    /// Parse top-level chunks from ADT data.
    /// </summary>
    private Dictionary<string, byte[]> ParseChunks(byte[] data)
    {
        var result = new Dictionary<string, byte[]>();
        int pos = 0;
        
        while (pos < data.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);
            
            if (size < 0 || pos + 8 + size > data.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            // Only store first occurrence of each chunk type (except MCNK)
            if (!result.ContainsKey(readable) && readable != "MCNK")
            {
                var chunkData = new byte[size];
                Buffer.BlockCopy(data, pos + 8, chunkData, 0, size);
                result[readable] = chunkData;
            }
            
            pos += 8 + size;
        }
        
        return result;
    }

    /// <summary>
    /// Parse subchunks from MCNK data starting at given offset.
    /// </summary>
    private Dictionary<string, byte[]> ParseSubchunks(byte[] data, int startOffset)
    {
        var result = new Dictionary<string, byte[]>();
        int pos = startOffset;
        
        while (pos < data.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);
            
            if (size < 0 || size > 10_000_000 || pos + 8 + size > data.Length) break;
            
            string readable = new string(sig.Reverse().ToArray());
            
            var chunkData = new byte[size];
            Buffer.BlockCopy(data, pos + 8, chunkData, 0, size);
            result[readable] = chunkData;
            
            pos += 8 + size;
        }
        
        return result;
    }

    /// <summary>
    /// Parse MTEX chunk into list of texture filenames.
    /// </summary>
    private List<string> ParseMtexFilenames(byte[] mtexData)
    {
        var result = new List<string>();
        int start = 0;
        
        for (int i = 0; i < mtexData.Length; i++)
        {
            if (mtexData[i] == 0)
            {
                if (i > start)
                {
                    var filename = Encoding.ASCII.GetString(mtexData, start, i - start);
                    result.Add(filename);
                }
                start = i + 1;
            }
        }
        
        return result;
    }

    /// <summary>
    /// Write a chunk with reversed signature (disk format).
    /// </summary>
    private void WriteChunk(MemoryStream ms, string sig, byte[] data)
    {
        // Reverse signature for disk format
        var reversed = new string(sig.Reverse().ToArray());
        ms.Write(Encoding.ASCII.GetBytes(reversed), 0, 4);
        ms.Write(BitConverter.GetBytes(data.Length), 0, 4);
        ms.Write(data, 0, data.Length);
    }
}
