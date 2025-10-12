using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.Logging;
using System.Diagnostics;

namespace WCAnalyzer.TestParser
{
    public class SimpleTestParser
    {
        private static readonly ILogger logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger<SimpleTestParser>();
        
        public static void ParseFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Error: File '{filePath}' not found.");
                return;
            }
            
            Console.WriteLine($"Analyzing file: {filePath}");
            
            try
            {
                byte[] fileData = File.ReadAllBytes(filePath);
                Console.WriteLine($"File size: {FormatFileSize(fileData.Length)}");
                
                // Parse chunks
                Console.WriteLine("Parsing chunks...");
                List<ChunkInfo> chunks = new List<ChunkInfo>();
                
                using (MemoryStream ms = new MemoryStream(fileData))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    while (ms.Position < ms.Length)
                    {
                        try
                        {
                            // Read chunk header
                            if (ms.Position + 8 > ms.Length)
                            {
                                // Not enough data for a chunk header
                                Console.WriteLine($"Warning: Incomplete chunk header at position {ms.Position}. File may be truncated.");
                                break;
                            }
                            
                            long chunkStart = ms.Position;
                            string signature = new string(reader.ReadChars(4));
                            uint size = reader.ReadUInt32();
                            
                            if (ms.Position + size > ms.Length)
                            {
                                // Chunk size exceeds file bounds
                                Console.WriteLine($"Warning: Chunk {signature} at position {ms.Position - 8} claims size {size} bytes, but only {ms.Length - ms.Position} bytes remain. File may be corrupted.");
                                break;
                            }
                            
                            // Read chunk data
                            byte[] chunkData = reader.ReadBytes((int)size);
                            
                            // Add to list
                            chunks.Add(new ChunkInfo 
                            {
                                Signature = signature,
                                Size = size,
                                Data = chunkData,
                                Position = chunkStart
                            });
                            
                            Console.WriteLine($"Found chunk: {signature} at position {chunkStart}, size: {size} bytes");
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error parsing chunk at position {ms.Position}: {ex.Message}");
                            break;
                        }
                    }
                }
                
                Console.WriteLine($"\nChunk Analysis Results:");
                Console.WriteLine($"Total chunks: {chunks.Count}\n");
                
                // Display each chunk
                foreach (var chunk in chunks)
                {
                    Console.WriteLine($"Chunk: {chunk.Signature} (Position: 0x{chunk.Position:X8}, Size: {chunk.Size} bytes)");
                    
                    // Special handling for known chunks
                    if (chunk.Signature == "MVER" && chunk.Size >= 4)
                    {
                        uint version = BitConverter.ToUInt32(chunk.Data, 0);
                        Console.WriteLine($"  Version: {version}");
                    }
                    else if (chunk.Signature == "MSPV" && chunk.Size >= 12)
                    {
                        int vertexCount = (int)chunk.Size / 12; // 3 floats per vertex
                        Console.WriteLine($"  Vertex count: {vertexCount}");
                        
                        if (vertexCount > 0)
                        {
                            Console.WriteLine("  First vertex data:");
                            float x = BitConverter.ToSingle(chunk.Data, 0);
                            float y = BitConverter.ToSingle(chunk.Data, 4);
                            float z = BitConverter.ToSingle(chunk.Data, 8);
                            Console.WriteLine($"    Raw: ({x}, {y}, {z})");
                            
                            // Apply transformation (World Coordinates)
                            float worldX = 17066.666f - x;
                            float worldY = 17066.666f - y;
                            float worldZ = z / 36.0f;
                            Console.WriteLine($"    World: ({worldX:F2}, {worldY:F2}, {worldZ:F2})");
                        }
                    }
                    else if (chunk.Signature == "MSPI" && chunk.Size >= 4)
                    {
                        int indexCount = (int)chunk.Size / 4; // 4 bytes per index
                        Console.WriteLine($"  Index count: {indexCount}");
                        
                        if (indexCount > 0 && indexCount <= 10)
                        {
                            Console.WriteLine("  Indices:");
                            for (int i = 0; i < indexCount; i++)
                            {
                                uint index = BitConverter.ToUInt32(chunk.Data, i * 4);
                                Console.WriteLine($"    [{i}]: {index}");
                            }
                        }
                        else if (indexCount > 10)
                        {
                            Console.WriteLine("  First 10 indices:");
                            for (int i = 0; i < 10; i++)
                            {
                                uint index = BitConverter.ToUInt32(chunk.Data, i * 4);
                                Console.WriteLine($"    [{i}]: {index}");
                            }
                            Console.WriteLine($"    ... and {indexCount - 10} more indices");
                        }
                    }
                    else if (chunk.Signature == "MDBF")
                    {
                        // Null-terminated string
                        string filename = Encoding.ASCII.GetString(chunk.Data).TrimEnd('\0');
                        Console.WriteLine($"  Destructible Building Filename: {filename}");
                    }
                    else if (chunk.Signature == "MDBI" && chunk.Size >= 4)
                    {
                        uint index = BitConverter.ToUInt32(chunk.Data, 0);
                        Console.WriteLine($"  Destructible Building Index: {index}");
                    }
                    else
                    {
                        // Show hex dump for unhandled chunks
                        Console.WriteLine("  Data (first 32 bytes):");
                        int bytesToShow = Math.Min(32, chunk.Data.Length);
                        Console.WriteLine("  " + BitConverter.ToString(chunk.Data, 0, bytesToShow).Replace("-", " "));
                        
                        if (chunk.Data.Length > 32)
                        {
                            Console.WriteLine($"  ... and {chunk.Data.Length - 32} more bytes");
                        }
                    }
                    
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
        
        private static string FormatFileSize(long bytes)
        {
            string[] suffixes = { "B", "KB", "MB", "GB", "TB" };
            int counter = 0;
            decimal number = bytes;
            
            while (Math.Round(number / 1024) >= 1)
            {
                number /= 1024;
                counter++;
            }
            
            return $"{number:n2} {suffixes[counter]} ({bytes:N0} bytes)";
        }
        
        private class ChunkInfo
        {
            public string Signature { get; set; } = string.Empty;
            public uint Size { get; set; }
            public byte[] Data { get; set; } = Array.Empty<byte>();
            public long Position { get; set; }
        }
    }
} 