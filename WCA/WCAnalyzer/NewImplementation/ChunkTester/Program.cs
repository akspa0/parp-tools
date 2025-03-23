using System;
using System.CommandLine;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files;
using WCAnalyzer.Core.Files.PM4;
using WCAnalyzer.Core.Files.PD4;

namespace ChunkTester
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Setup DI
            var serviceProvider = BuildServiceProvider();
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
            
            // Create root command
            var rootCommand = new RootCommand("Chunk Tester - Test PM4/PD4 file parsing with detailed output");
            
            // PM4 command
            var pm4Command = new Command("pm4", "Analyze PM4 files");
            var pm4FileOption = new Option<FileInfo>(
                aliases: new[] { "--file", "-f" },
                description: "Path to the PM4 file"
            )
            {
                IsRequired = true
            };
            pm4Command.AddOption(pm4FileOption);
            
            var pm4VerboseOption = new Option<bool>(
                aliases: new[] { "--verbose", "-v" },
                description: "Enable verbose output"
            );
            pm4Command.AddOption(pm4VerboseOption);
            
            pm4Command.SetHandler((fileInfo, verbose) =>
            {
                AnalyzePM4File(fileInfo, verbose, serviceProvider);
            }, pm4FileOption, pm4VerboseOption);
            
            rootCommand.AddCommand(pm4Command);
            
            // PD4 command
            var pd4Command = new Command("pd4", "Analyze PD4 files");
            var pd4FileOption = new Option<FileInfo>(
                aliases: new[] { "--file", "-f" },
                description: "Path to the PD4 file"
            )
            {
                IsRequired = true
            };
            pd4Command.AddOption(pd4FileOption);
            
            var pd4VerboseOption = new Option<bool>(
                aliases: new[] { "--verbose", "-v" },
                description: "Enable verbose output"
            );
            pd4Command.AddOption(pd4VerboseOption);
            
            pd4Command.SetHandler((fileInfo, verbose) =>
            {
                AnalyzePD4File(fileInfo, verbose, serviceProvider);
            }, pd4FileOption, pd4VerboseOption);
            
            rootCommand.AddCommand(pd4Command);
            
            // Parse command line and execute
            return await rootCommand.InvokeAsync(args);
        }
        
        private static ServiceProvider BuildServiceProvider()
        {
            var services = new ServiceCollection();
            
            // Add logging
            services.AddLogging(builder =>
            {
                builder
                    .AddConsole()
                    .SetMinimumLevel(LogLevel.Information);
            });
            
            return services.BuildServiceProvider();
        }
        
        private static void AnalyzePM4File(FileInfo fileInfo, bool verbose, ServiceProvider serviceProvider)
        {
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
            
            if (!fileInfo.Exists)
            {
                logger.LogError($"File not found: {fileInfo.FullName}");
                return;
            }
            
            logger.LogInformation($"Analyzing PM4 file: {fileInfo.FullName}");
            logger.LogInformation($"File size: {fileInfo.Length:N0} bytes");
            
            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                using (var fileStream = new FileStream(fileInfo.FullName, FileMode.Open, FileAccess.Read))
                using (var reader = new BinaryReader(fileStream))
                {
                    var chunks = new List<BaseChunk>();
                    var chunkFactory = new PM4.PM4ChunkFactory(logger);
                    var errorCount = 0;
                    
                    // Read chunks until end of file
                    while (fileStream.Position < fileStream.Length)
                    {
                        try
                        {
                            // Try to read the chunk signature
                            if (fileStream.Position + 8 > fileStream.Length)
                            {
                                logger.LogWarning("Incomplete chunk at end of file");
                                break;
                            }
                            
                            // Read chunk signature
                            byte[] signatureBytes = reader.ReadBytes(4);
                            string signature = System.Text.Encoding.ASCII.GetString(signatureBytes);
                            
                            // Read chunk size
                            uint size = reader.ReadUInt32();
                            
                            // Sanity check on chunk size
                            if (size > fileStream.Length - fileStream.Position)
                            {
                                logger.LogWarning($"Chunk {signature} claims size {size:N0} bytes, which exceeds remaining file size");
                                size = (uint)(fileStream.Length - fileStream.Position);
                            }
                            
                            // Read chunk data
                            byte[] data = reader.ReadBytes((int)size);
                            
                            // Create appropriate chunk
                            var chunk = chunkFactory.CreateChunk(signature, data);
                            chunk.Parse();
                            chunks.Add(chunk);
                            
                            logger.LogInformation($"Read chunk: {chunk.ToString()} ({size:N0} bytes)");
                            
                            // Print detailed info for verbose mode
                            if (verbose)
                            {
                                PrintChunkDetails(chunk, logger);
                            }
                        }
                        catch (Exception ex)
                        {
                            errorCount++;
                            logger.LogError($"Error parsing chunk at position {fileStream.Position}: {ex.Message}");
                            
                            // Try to recover - skip ahead to find next valid chunk
                            if (!TryRecoverToNextChunk(fileStream, reader, logger))
                            {
                                logger.LogError("Cannot recover from error, stopping parse");
                                break;
                            }
                        }
                    }
                    
                    stopwatch.Stop();
                    
                    // Print summary
                    logger.LogInformation($"Parse completed in {stopwatch.ElapsedMilliseconds:N0}ms");
                    logger.LogInformation($"Found {chunks.Count} chunks");
                    logger.LogInformation($"Encountered {errorCount} errors during parsing");
                    
                    // Count chunk types
                    var chunkTypes = chunks.GroupBy(c => c.Signature)
                                          .Select(g => new { Type = g.Key, Count = g.Count() })
                                          .OrderByDescending(x => x.Count);
                    
                    logger.LogInformation("Chunk type counts:");
                    foreach (var type in chunkTypes)
                    {
                        logger.LogInformation($"  {type.Type}: {type.Count}");
                    }
                }
            }
            catch (Exception ex)
            {
                logger.LogError($"Failed to analyze file: {ex.Message}");
            }
        }
        
        private static void AnalyzePD4File(FileInfo fileInfo, bool verbose, ServiceProvider serviceProvider)
        {
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
            
            if (!fileInfo.Exists)
            {
                logger.LogError($"File not found: {fileInfo.FullName}");
                return;
            }
            
            logger.LogInformation($"Analyzing PD4 file: {fileInfo.FullName}");
            logger.LogInformation($"File size: {fileInfo.Length:N0} bytes");
            
            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                using (var fileStream = new FileStream(fileInfo.FullName, FileMode.Open, FileAccess.Read))
                using (var reader = new BinaryReader(fileStream))
                {
                    var chunks = new List<BaseChunk>();
                    var chunkFactory = new PD4.PD4ChunkFactory(logger);
                    var errorCount = 0;
                    
                    // Read chunks until end of file
                    while (fileStream.Position < fileStream.Length)
                    {
                        try
                        {
                            // Try to read the chunk signature
                            if (fileStream.Position + 8 > fileStream.Length)
                            {
                                logger.LogWarning("Incomplete chunk at end of file");
                                break;
                            }
                            
                            // Read chunk signature
                            byte[] signatureBytes = reader.ReadBytes(4);
                            string signature = System.Text.Encoding.ASCII.GetString(signatureBytes);
                            
                            // Read chunk size
                            uint size = reader.ReadUInt32();
                            
                            // Sanity check on chunk size
                            if (size > fileStream.Length - fileStream.Position)
                            {
                                logger.LogWarning($"Chunk {signature} claims size {size:N0} bytes, which exceeds remaining file size");
                                size = (uint)(fileStream.Length - fileStream.Position);
                            }
                            
                            // Read chunk data
                            byte[] data = reader.ReadBytes((int)size);
                            
                            // Create appropriate chunk
                            var chunk = chunkFactory.CreateChunk(signature, data);
                            chunk.Parse();
                            chunks.Add(chunk);
                            
                            logger.LogInformation($"Read chunk: {chunk.ToString()} ({size:N0} bytes)");
                            
                            // Print detailed info for verbose mode
                            if (verbose)
                            {
                                PrintChunkDetails(chunk, logger);
                            }
                        }
                        catch (Exception ex)
                        {
                            errorCount++;
                            logger.LogError($"Error parsing chunk at position {fileStream.Position}: {ex.Message}");
                            
                            // Try to recover - skip ahead to find next valid chunk
                            if (!TryRecoverToNextChunk(fileStream, reader, logger))
                            {
                                logger.LogError("Cannot recover from error, stopping parse");
                                break;
                            }
                        }
                    }
                    
                    stopwatch.Stop();
                    
                    // Print summary
                    logger.LogInformation($"Parse completed in {stopwatch.ElapsedMilliseconds:N0}ms");
                    logger.LogInformation($"Found {chunks.Count} chunks");
                    logger.LogInformation($"Encountered {errorCount} errors during parsing");
                    
                    // Count chunk types
                    var chunkTypes = chunks.GroupBy(c => c.Signature)
                                          .Select(g => new { Type = g.Key, Count = g.Count() })
                                          .OrderByDescending(x => x.Count);
                    
                    logger.LogInformation("Chunk type counts:");
                    foreach (var type in chunkTypes)
                    {
                        logger.LogInformation($"  {type.Type}: {type.Count}");
                    }
                }
            }
            catch (Exception ex)
            {
                logger.LogError($"Failed to analyze file: {ex.Message}");
            }
        }
        
        private static bool TryRecoverToNextChunk(FileStream fileStream, BinaryReader reader, ILogger logger)
        {
            const int MAX_SCAN = 1024; // Maximum bytes to scan for a new chunk signature
            const int CHUNK_ALIGN = 4; // Chunks are typically aligned to 4-byte boundaries
            
            long startPos = fileStream.Position;
            int bytesScanned = 0;
            
            logger.LogWarning($"Attempting to recover by finding next valid chunk...");
            
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
                    string signature = System.Text.Encoding.ASCII.GetString(signatureBytes);
                    
                    // Check if size is reasonable
                    uint size = reader.ReadUInt32();
                    
                    if (size > 0 && size <= fileStream.Length - fileStream.Position)
                    {
                        logger.LogWarning($"Found potential chunk {signature} with size {size:N0} at position {posBeforeRead}");
                        
                        // Go back to the start of the chunk so it can be properly read
                        fileStream.Position = posBeforeRead;
                        return true;
                    }
                }
                
                // Reset to just after the position we tried
                fileStream.Position = posBeforeRead + 1;
                bytesScanned++;
            }
            
            logger.LogError($"Failed to find valid chunk signature after scanning {bytesScanned} bytes");
            return false;
        }
        
        private static void PrintChunkDetails(BaseChunk chunk, ILogger logger)
        {
            logger.LogInformation($"Detailed information for {chunk.Signature} chunk:");
            
            // Print different details based on chunk type
            switch (chunk)
            {
                case PM4.Chunks.MspvChunk mspvChunk:
                    logger.LogInformation($"  Vertex count: {mspvChunk.GetVertexCount()}");
                    if (mspvChunk.GetVertexCount() > 0)
                    {
                        var (min, max) = mspvChunk.GetBoundingBox();
                        logger.LogInformation($"  Bounding box: Min({min.X}, {min.Y}, {min.Z}), Max({max.X}, {max.Y}, {max.Z})");
                        
                        // Print first few vertices
                        int vertexSampleCount = Math.Min(5, mspvChunk.GetVertexCount());
                        logger.LogInformation($"  Sample vertices (first {vertexSampleCount}):");
                        for (int i = 0; i < vertexSampleCount; i++)
                        {
                            var vertex = mspvChunk.GetVertex(i);
                            var worldVertex = PM4.Chunks.MspvChunk.GetWorldCoordinates(vertex);
                            logger.LogInformation($"    Vertex {i}: File({vertex.X}, {vertex.Y}, {vertex.Z}) → World({worldVertex.X}, {worldVertex.Y}, {worldVertex.Z})");
                        }
                    }
                    break;
                
                case PD4.Chunks.MspvChunk pd4MspvChunk:
                    logger.LogInformation($"  Vertex count: {pd4MspvChunk.GetVertexCount()}");
                    if (pd4MspvChunk.GetVertexCount() > 0)
                    {
                        // Print first few vertices
                        int vertexSampleCount = Math.Min(5, pd4MspvChunk.GetVertexCount());
                        logger.LogInformation($"  Sample vertices (first {vertexSampleCount}):");
                        for (int i = 0; i < vertexSampleCount; i++)
                        {
                            var vertex = pd4MspvChunk.GetVertex(i);
                            logger.LogInformation($"    Vertex {i}: ({vertex.X}, {vertex.Y}, {vertex.Z})");
                        }
                    }
                    break;
                
                case PM4.Chunks.MspiChunk mspiChunk:
                    logger.LogInformation($"  Index count: {mspiChunk.GetIndexCount()}");
                    if (mspiChunk.GetIndexCount() > 0)
                    {
                        // Print first few indices
                        int indexSampleCount = Math.Min(10, mspiChunk.GetIndexCount());
                        logger.LogInformation($"  Sample indices (first {indexSampleCount}):");
                        for (int i = 0; i < indexSampleCount; i++)
                        {
                            logger.LogInformation($"    Index {i}: {mspiChunk.GetIndex(i)}");
                        }
                    }
                    break;
                
                // Add cases for other chunk types as needed
                
                default:
                    logger.LogInformation($"  Raw data size: {chunk.Data.Length:N0} bytes");
                    break;
            }
        }
    }
}
