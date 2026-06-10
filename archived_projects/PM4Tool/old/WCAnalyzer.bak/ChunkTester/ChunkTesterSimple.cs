using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Text;

namespace ChunkTester
{
    /// <summary>
    /// A simplified chunk tester that doesn't rely on the WCAnalyzer.Core implementation
    /// </summary>
    public class ChunkTesterSimple
    {
        /// <summary>
        /// Basic chunk class to hold chunk data
        /// </summary>
        public class Chunk
        {
            public string Signature { get; }
            public byte[] Data { get; }
            public long FilePosition { get; }
            
            public Chunk(string signature, byte[] data, long filePosition)
            {
                Signature = signature ?? throw new ArgumentNullException(nameof(signature));
                Data = data ?? throw new ArgumentNullException(nameof(data));
                FilePosition = filePosition;
            }
            
            public override string ToString()
            {
                return $"{Signature} ({Data.Length} bytes)";
            }
        }
        
        /// <summary>
        /// Main method to run the tester
        /// </summary>
        public static void Run(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: ChunkTester <filename> [--verbose]");
                return;
            }
            
            string filename = args[0];
            bool verbose = args.Length > 1 && args[1] == "--verbose";
            
            if (!File.Exists(filename))
            {
                Console.WriteLine($"File not found: {filename}");
                return;
            }
            
            try
            {
                // Determine file type from extension
                string extension = Path.GetExtension(filename).ToLower();
                
                if (extension == ".pm4")
                {
                    AnalyzePM4File(filename, verbose);
                }
                else if (extension == ".pd4")
                {
                    AnalyzePD4File(filename, verbose);
                }
                else
                {
                    Console.WriteLine($"Unsupported file extension: {extension}. Expected .pm4 or .pd4");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine(ex.StackTrace);
                }
            }
        }
        
        /// <summary>
        /// Analyzes a PM4 file
        /// </summary>
        private static void AnalyzePM4File(string filename, bool verbose)
        {
            Console.WriteLine($"Analyzing PM4 file: {filename}");
            var fileInfo = new FileInfo(filename);
            Console.WriteLine($"File size: {fileInfo.Length:N0} bytes");
            
            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                List<Chunk> chunks = ReadChunks(filename);
                
                stopwatch.Stop();
                
                // Print summary
                Console.WriteLine($"Parse completed in {stopwatch.ElapsedMilliseconds:N0}ms");
                Console.WriteLine($"Found {chunks.Count} chunks");
                
                // Count chunk types
                var chunkTypes = chunks.GroupBy(c => c.Signature)
                                      .Select(g => new { Type = g.Key, Count = g.Count() })
                                      .OrderByDescending(x => x.Count);
                
                Console.WriteLine("Chunk type counts:");
                foreach (var type in chunkTypes)
                {
                    Console.WriteLine($"  {type.Type}: {type.Count}");
                }
                
                // Print detailed info for verbose mode
                if (verbose)
                {
                    foreach (var chunk in chunks)
                    {
                        PrintChunkDetails(chunk);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to analyze file: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine(ex.StackTrace);
                }
            }
        }
        
        /// <summary>
        /// Analyzes a PD4 file
        /// </summary>
        private static void AnalyzePD4File(string filename, bool verbose)
        {
            Console.WriteLine($"Analyzing PD4 file: {filename}");
            var fileInfo = new FileInfo(filename);
            Console.WriteLine($"File size: {fileInfo.Length:N0} bytes");
            
            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                List<Chunk> chunks = ReadChunks(filename);
                
                stopwatch.Stop();
                
                // Print summary
                Console.WriteLine($"Parse completed in {stopwatch.ElapsedMilliseconds:N0}ms");
                Console.WriteLine($"Found {chunks.Count} chunks");
                
                // Count chunk types
                var chunkTypes = chunks.GroupBy(c => c.Signature)
                                      .Select(g => new { Type = g.Key, Count = g.Count() })
                                      .OrderByDescending(x => x.Count);
                
                Console.WriteLine("Chunk type counts:");
                foreach (var type in chunkTypes)
                {
                    Console.WriteLine($"  {type.Type}: {type.Count}");
                }
                
                // Print detailed info for verbose mode
                if (verbose)
                {
                    foreach (var chunk in chunks)
                    {
                        PrintChunkDetails(chunk);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to analyze file: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine(ex.StackTrace);
                }
            }
        }
        
        /// <summary>
        /// Reads all chunks from a file
        /// </summary>
        private static List<Chunk> ReadChunks(string filename)
        {
            var chunks = new List<Chunk>();
            var errorCount = 0;
            
            using (var fileStream = new FileStream(filename, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(fileStream))
            {
                while (fileStream.Position < fileStream.Length)
                {
                    try
                    {
                        // Try to read the chunk signature
                        if (fileStream.Position + 8 > fileStream.Length)
                        {
                            Console.WriteLine("Incomplete chunk at end of file");
                            break;
                        }
                        
                        // Remember position for reporting
                        long chunkStart = fileStream.Position;
                        
                        // Read chunk signature
                        byte[] signatureBytes = reader.ReadBytes(4);
                        string signature = Encoding.ASCII.GetString(signatureBytes);
                        
                        // Read chunk size
                        uint size = reader.ReadUInt32();
                        
                        // Sanity check on chunk size
                        if (size > fileStream.Length - fileStream.Position)
                        {
                            Console.WriteLine($"Warning: Chunk {signature} claims size {size:N0} bytes, which exceeds remaining file size");
                            size = (uint)(fileStream.Length - fileStream.Position);
                        }
                        
                        // Read chunk data
                        byte[] data = reader.ReadBytes((int)size);
                        
                        // Create chunk
                        var chunk = new Chunk(signature, data, chunkStart);
                        chunks.Add(chunk);
                        
                        Console.WriteLine($"Read chunk: {chunk.ToString()}");
                    }
                    catch (Exception ex)
                    {
                        errorCount++;
                        Console.WriteLine($"Error parsing chunk at position {fileStream.Position}: {ex.Message}");
                        
                        // Try to recover - skip ahead to find next valid chunk
                        if (!TryRecoverToNextChunk(fileStream, reader))
                        {
                            Console.WriteLine("Cannot recover from error, stopping parse");
                            break;
                        }
                    }
                }
            }
            
            if (errorCount > 0)
            {
                Console.WriteLine($"Encountered {errorCount} errors during parsing");
            }
            
            return chunks;
        }
        
        /// <summary>
        /// Attempts to recover from a parsing error by finding the next valid chunk
        /// </summary>
        private static bool TryRecoverToNextChunk(FileStream fileStream, BinaryReader reader)
        {
            const int MAX_SCAN = 1024; // Maximum bytes to scan for a new chunk signature
            const int CHUNK_ALIGN = 4; // Chunks are typically aligned to 4-byte boundaries
            
            long startPos = fileStream.Position;
            int bytesScanned = 0;
            
            Console.WriteLine($"Attempting to recover by finding next valid chunk...");
            
            while (bytesScanned < MAX_SCAN && fileStream.Position < fileStream.Length - 8)
            {
                // Try to align to 4-byte boundary if not already aligned
                long currentPos = fileStream.Position;
                if (currentPos % CHUNK_ALIGN != 0)
                {
                    long offset = CHUNK_ALIGN - (currentPos % CHUNK_ALIGN);
                    fileStream.Seek(offset, SeekOrigin.Current);
                    bytesScanned += (int)offset;
                }
                
                // Check if we have a valid chunk signature (4 ASCII chars)
                long posBeforeRead = fileStream.Position;
                byte[] signatureBytes = reader.ReadBytes(4);
                
                if (signatureBytes.Length < 4)
                {
                    return false; // End of file
                }
                
                // Check if all bytes are printable ASCII characters
                bool validSignature = signatureBytes.All(b => b >= 32 && b <= 126);
                
                if (validSignature)
                {
                    string signature = Encoding.ASCII.GetString(signatureBytes);
                    
                    // Check if size is reasonable
                    uint size = reader.ReadUInt32();
                    
                    if (size > 0 && size <= fileStream.Length - fileStream.Position)
                    {
                        Console.WriteLine($"Found potential chunk {signature} with size {size:N0} at position {posBeforeRead}");
                        
                        // Go back to the start of the chunk so it can be properly read
                        fileStream.Position = posBeforeRead;
                        return true;
                    }
                }
                
                // Reset to just after the position we tried
                fileStream.Position = posBeforeRead + 1;
                bytesScanned++;
            }
            
            Console.WriteLine($"Failed to find valid chunk signature after scanning {bytesScanned} bytes");
            return false;
        }
        
        /// <summary>
        /// Prints detailed information about a chunk based on its type
        /// </summary>
        private static void PrintChunkDetails(Chunk chunk)
        {
            Console.WriteLine($"Detailed information for {chunk.Signature} chunk:");
            Console.WriteLine($"  File position: 0x{chunk.FilePosition:X8}");
            Console.WriteLine($"  Data size: {chunk.Data.Length:N0} bytes");
            
            switch (chunk.Signature)
            {
                case "MVER":
                    if (chunk.Data.Length >= 4)
                    {
                        uint version = BitConverter.ToUInt32(chunk.Data, 0);
                        Console.WriteLine($"  Version: {version}");
                    }
                    break;
                
                case "MCRC":
                    if (chunk.Data.Length >= 4)
                    {
                        uint crc = BitConverter.ToUInt32(chunk.Data, 0);
                        Console.WriteLine($"  CRC: 0x{crc:X8}");
                    }
                    break;
                
                case "MSPV":
                    int vertexCount = chunk.Data.Length / 12; // 3 floats * 4 bytes
                    Console.WriteLine($"  Vertex count: {vertexCount}");
                    
                    if (vertexCount > 0)
                    {
                        // Print first 5 vertices
                        int vertexSampleCount = Math.Min(5, vertexCount);
                        Console.WriteLine($"  Sample vertices (first {vertexSampleCount}):");
                        
                        for (int i = 0; i < vertexSampleCount; i++)
                        {
                            int offset = i * 12;
                            float x = BitConverter.ToSingle(chunk.Data, offset);
                            float y = BitConverter.ToSingle(chunk.Data, offset + 4);
                            float z = BitConverter.ToSingle(chunk.Data, offset + 8);
                            
                            // Calculate world coordinates
                            float worldX = 17066.666f - x;
                            float worldY = 17066.666f - y;
                            float worldZ = z / 36.0f;
                            
                            Console.WriteLine($"    Vertex {i}: File({x}, {y}, {z}) → World({worldX}, {worldY}, {worldZ})");
                        }
                    }
                    break;
                
                case "MSPI":
                    int indexCount = chunk.Data.Length / 4; // uint32 indices
                    Console.WriteLine($"  Index count: {indexCount}");
                    
                    if (indexCount > 0)
                    {
                        // Print first 10 indices
                        int indexSampleCount = Math.Min(10, indexCount);
                        Console.WriteLine($"  Sample indices (first {indexSampleCount}):");
                        
                        for (int i = 0; i < indexSampleCount; i++)
                        {
                            int offset = i * 4;
                            uint index = BitConverter.ToUInt32(chunk.Data, offset);
                            Console.WriteLine($"    Index {i}: {index}");
                        }
                    }
                    break;
                
                case "MSHD":
                    if (chunk.Data.Length >= 32) // Header size is 8 + 5 * 4 = 32 bytes
                    {
                        uint field1 = BitConverter.ToUInt32(chunk.Data, 0);
                        uint field2 = BitConverter.ToUInt32(chunk.Data, 4);
                        uint field3 = BitConverter.ToUInt32(chunk.Data, 8);
                        
                        Console.WriteLine($"  Header fields: {field1}, {field2}, {field3}");
                    }
                    break;
                
                case "MPRL":
                    int entryCount = chunk.Data.Length / 24; // Each entry is 24 bytes
                    Console.WriteLine($"  Position record count: {entryCount}");
                    
                    if (entryCount > 0)
                    {
                        // Print first 3 entries
                        int entrySampleCount = Math.Min(3, entryCount);
                        Console.WriteLine($"  Sample entries (first {entrySampleCount}):");
                        
                        for (int i = 0; i < entrySampleCount; i++)
                        {
                            int offset = i * 24;
                            ushort field1 = BitConverter.ToUInt16(chunk.Data, offset);
                            short field2 = BitConverter.ToInt16(chunk.Data, offset + 2);
                            ushort field3 = BitConverter.ToUInt16(chunk.Data, offset + 4);
                            ushort field4 = BitConverter.ToUInt16(chunk.Data, offset + 6);
                            
                            float posX = BitConverter.ToSingle(chunk.Data, offset + 8);
                            float posY = BitConverter.ToSingle(chunk.Data, offset + 12);
                            float posZ = BitConverter.ToSingle(chunk.Data, offset + 16);
                            
                            short field5 = BitConverter.ToInt16(chunk.Data, offset + 20);
                            ushort field6 = BitConverter.ToUInt16(chunk.Data, offset + 22);
                            
                            Console.WriteLine($"    Entry {i}: Fields({field1}, {field2}, {field3}, {field4}, {field5}, {field6})");
                            Console.WriteLine($"      Position: ({posX}, {posY}, {posZ})");
                        }
                    }
                    break;
                
                case "MDBF":
                    // This is a null-terminated string
                    string filename = ReadNullTerminatedString(chunk.Data, 0);
                    Console.WriteLine($"  Destructible Building Filename: {filename}");
                    break;
                
                case "MDBI":
                    if (chunk.Data.Length >= 4)
                    {
                        uint index = BitConverter.ToUInt32(chunk.Data, 0);
                        Console.WriteLine($"  Destructible Building Index: {index}");
                    }
                    break;
                
                default:
                    // Hex dump first 32 bytes (or less if the chunk is smaller)
                    int bytesToDump = Math.Min(32, chunk.Data.Length);
                    Console.WriteLine($"  First {bytesToDump} bytes:");
                    
                    for (int i = 0; i < bytesToDump; i += 16)
                    {
                        int lineLength = Math.Min(16, bytesToDump - i);
                        StringBuilder hexLine = new StringBuilder();
                        StringBuilder asciiLine = new StringBuilder();
                        
                        for (int j = 0; j < lineLength; j++)
                        {
                            byte b = chunk.Data[i + j];
                            hexLine.Append($"{b:X2} ");
                            
                            // Show printable ASCII chars, replace others with a dot
                            asciiLine.Append(b >= 32 && b <= 126 ? (char)b : '.');
                        }
                        
                        // Pad short lines
                        for (int j = lineLength; j < 16; j++)
                        {
                            hexLine.Append("   ");
                        }
                        
                        Console.WriteLine($"    {i:X4}: {hexLine} | {asciiLine}");
                    }
                    break;
            }
        }
        
        /// <summary>
        /// Reads a null-terminated string from a byte array
        /// </summary>
        private static string ReadNullTerminatedString(byte[] data, int offset)
        {
            int length = 0;
            while (offset + length < data.Length && data[offset + length] != 0)
            {
                length++;
            }
            
            return Encoding.ASCII.GetString(data, offset, length);
        }
    }
} 