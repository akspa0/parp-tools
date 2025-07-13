using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWToolbox.Core.WMO
{
    /// <summary>
    /// Very small self-contained chunk reader for use inside the legacy Core project.
    /// Reads all top-level chunks (id, size, data) from a WMO stream. In WoW files, FourCC bytes are stored *reversed* (big-endian),
    /// so this reader always flips the 4-byte ID to obtain the canonical chunk name (e.g., 'REVM' on disk → 'MVER' in code).
    /// </summary>
    internal static class SimpleChunkReader
    {
        internal sealed record Chunk(string Id, long Offset, uint Size, byte[] Data);

        internal static List<Chunk> ReadAllChunks(Stream stream)
        {
            if (!stream.CanSeek) throw new ArgumentException("Stream needs seek", nameof(stream));
            var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
            var list = new List<Chunk>();
            stream.Position = 0;
            
            // Log initial stream state
            Console.WriteLine($"[DEBUG][SimpleChunkReader] Starting chunk scan at position {stream.Position} of {stream.Length}");
            Console.WriteLine($"[DEBUG][SimpleChunkReader] Initial stream state - Position: 0x{stream.Position:X}, Length: 0x{stream.Length:X}");
            
            // Dump first 64 bytes of the stream for debugging
            byte[] initialBytes = new byte[Math.Min(64, stream.Length)];
            long originalPosition = stream.Position;
            stream.Read(initialBytes, 0, initialBytes.Length);
            stream.Position = originalPosition;
            
            Console.WriteLine("[DEBUG][SimpleChunkReader] First 64 bytes of stream:");
            Console.WriteLine($"Hex: {BitConverter.ToString(initialBytes).Replace("-", " ")}");
            Console.WriteLine($"ASCII: {new string(initialBytes.Select(b => b >= 32 && b < 127 ? (char)b : '.').ToArray())}");
            
            int chunkCount = 0;
            long lastGoodPosition = 0;
            
            try
            {
                while (stream.Position + 8 <= stream.Length)
                {
                    chunkCount++;
                    long offset = stream.Position;
                    lastGoodPosition = offset;
                    
                    // Log position before reading chunk header
                    Console.WriteLine($"\n[DEBUG][SimpleChunkReader] [{chunkCount}] Reading chunk header at 0x{offset:X} (0x{stream.Position:X} absolute)");
                    
                    // Read chunk ID (4 bytes, ASCII) - ALWAYS REVERSED in WoW files
                    byte[] idBytes = new byte[4];
                    int bytesRead = stream.Read(idBytes, 0, 4);
                    
                    if (bytesRead < 4) 
                    {
                        Console.WriteLine($"[WARN][SimpleChunkReader] Couldn't read 4 bytes for chunk ID at position 0x{offset:X}");
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] Only read {bytesRead} bytes: {BitConverter.ToString(idBytes, 0, bytesRead)}");
                        break;
                    }
                    
                    // Always reverse the chunk ID bytes for WoW files
                    Array.Reverse(idBytes);
                    string id = Encoding.ASCII.GetString(idBytes);
                    
                    // Log chunk ID
                    Console.WriteLine($"[DEBUG][SimpleChunkReader] Chunk ID: '{id}' (reversed for WoW format)");
                    
                    // Read chunk size (4 bytes) - ALWAYS LITTLE-ENDIAN in WoW files
                    byte[] sizeBytes = new byte[4];
                    bytesRead = stream.Read(sizeBytes, 0, 4);
                    
                    if (bytesRead < 4)
                    {
                        Console.WriteLine($"[WARN][SimpleChunkReader] Couldn't read 4 bytes for chunk size at position 0x{stream.Position - 4:X}");
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] Only read {bytesRead} bytes: {BitConverter.ToString(sizeBytes, 0, bytesRead)}");
                        break;
                    }
                    
                    // Data is always little-endian in WoW files, no need to reverse sizeBytes
                    
                    // Try little-endian first (WMO v17+)
                    uint size = BitConverter.ToUInt32(sizeBytes, 0);
                    bool usedBigEndian = false;
                    
                    // If size is suspiciously large, try big-endian (WMO v14)
                    if (size > 0x1000000) // 16MB is a reasonable max chunk size
                    {
                        Array.Reverse(sizeBytes);
                        size = BitConverter.ToUInt32(sizeBytes, 0);
                        usedBigEndian = true;
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}] Chunk size too large, trying big-endian: {size} (0x{size:X8})");
                    }
                    
                    // Log chunk header details
                    Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}] Found chunk '{id}' at 0x{offset:X8} with size {size} (0x{size:X8})");
                    Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}]   Header bytes: {BitConverter.ToString(idBytes)} {BitConverter.ToString(sizeBytes)}");
                    Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}]   Endianness: {(usedBigEndian ? "Big" : "Little")}-endian");
                    Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}]   Data range: 0x{stream.Position:X8}-0x{stream.Position + size:X8}");
                    
                    // Validate chunk size
                    if (offset + 8 + size > stream.Length) 
                    {
                        Console.WriteLine($"[WARN][SimpleChunkReader] Chunk '{id}' at 0x{offset:X8} with size {size} would exceed stream length {stream.Length}, stopping");
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] Chunk end would be at 0x{offset + 8 + size:X8} but stream ends at 0x{stream.Length:X8}");
                        break; // corrupt
                    }
                    
                    // Read chunk data
                    byte[] data = new byte[size];
                    int totalRead = 0;
                    while (totalRead < size)
                    {
                        int read = stream.Read(data, totalRead, (int)size - totalRead);
                        if (read == 0) break; // End of stream
                        totalRead += read;
                    }
                    
                    if (totalRead < size)
                    {
                        Console.WriteLine($"[WARN][SimpleChunkReader] Couldn't read full chunk data for '{id}'. Requested: {size}, Read: {totalRead}");
                        // Truncate the data array to what we actually read
                        Array.Resize(ref data, totalRead);
                    }
                    
                    // Add chunk to list
                    var chunk = new Chunk(id, offset, (uint)totalRead, data);
                    list.Add(chunk);
                    
                    // Log some data from the chunk for debugging
                    int bytesToLog = Math.Min(16, data.Length);
                    if (bytesToLog > 0)
                    {
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}]   First {bytesToLog} bytes: {BitConverter.ToString(data, 0, bytesToLog).Replace("-", " ")}");
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}]   ASCII: {new string(data.Take(bytesToLog).Select(b => b >= 32 && b < 127 ? (char)b : '.').ToArray())}");
                    }
                    
                    // Special handling for MOHD chunk
                    if (id == "MOHD" && data.Length >= 4)
                    {
                        int groupCount = BitConverter.ToInt32(data, 0);
                        Console.WriteLine($"[DEBUG][SimpleChunkReader] [{chunkCount}] MOHD chunk detected - Group count: {groupCount}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR][SimpleChunkReader] Error reading chunks at position 0x{lastGoodPosition:X}: {ex.Message}");
                Console.WriteLine($"[DEBUG][SimpleChunkReader] Stack trace: {ex.StackTrace}");
            }
            
            Console.WriteLine($"[DEBUG][SimpleChunkReader] Finished chunk scan. Found {list.Count} chunks.");
            return list;
        }
    }
}
