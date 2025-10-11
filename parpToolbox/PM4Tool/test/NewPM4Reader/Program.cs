using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Numerics;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NewPM4Reader.PM4;
using NewPM4Reader.PM4.Chunks;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Usage: NewPM4Reader <path-to-file>");
                return;
            }

            var serviceProvider = new ServiceCollection()
                .AddLogging(configure => configure.AddConsole())
                .BuildServiceProvider();

            var loggerFactory = serviceProvider.GetRequiredService<ILoggerFactory>();
            var pm4Logger = loggerFactory.CreateLogger<PM4File>();

            var filePath = args[0];
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"File not found: {filePath}");
                return;
            }

            var fileExtension = Path.GetExtension(filePath).ToLowerInvariant();
            
            try
            {
                if (fileExtension == ".pm4")
                {
                    var pm4File = new PM4File(pm4Logger);
                    pm4File.Load(filePath);
                    
                    Console.WriteLine($"PM4 File: {Path.GetFileName(filePath)}");
                    Console.WriteLine($"Version: {pm4File.Version}");
                    Console.WriteLine($"Unique Chunk Types: {pm4File.ChunksBySignature.Count}");
                    Console.WriteLine($"Total Chunks: {pm4File.AllChunks.Count()}");
                    
                    // Display a summary of chunk counts by signature
                    Console.WriteLine("\nChunk Summary:");
                    foreach (var chunkGroup in pm4File.ChunksBySignature)
                    {
                        string reversedSignature = new string(chunkGroup.Key.Reverse().ToArray());
                        Console.WriteLine($"  {chunkGroup.Key} (reversed: {reversedSignature}): {chunkGroup.Value.Count} chunks");
                    }
                    
                    Console.WriteLine("\nDetailed Chunk Information:");
                    
                    // Now display detailed information for each chunk
                    foreach (var signatureGroup in pm4File.ChunksBySignature)
                    {
                        var signature = signatureGroup.Key;
                        var chunks = signatureGroup.Value;
                        
                        Console.WriteLine($"\nChunk Type: {signature} ({chunks.Count} chunks)");
                        
                        // Display information for the first few chunks of this type
                        int displayCount = Math.Min(chunks.Count, 3); // Show up to 3 chunks of each type
                        
                        for (int i = 0; i < displayCount; i++)
                        {
                            Console.WriteLine($"  Chunk {i + 1} of {chunks.Count}:");
                            DisplayPM4ChunkInfo(chunks[i], "    ");
                        }
                        
                        // If there are more chunks, just show a count
                        if (chunks.Count > displayCount)
                        {
                            Console.WriteLine($"  ... and {chunks.Count - displayCount} more chunks of this type");
                        }
                    }
                }
                else
                {
                    Console.WriteLine($"Unsupported file extension: {fileExtension}. Expected .pm4");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing file: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }
        
        private static void DisplayPM4ChunkInfo(IPM4Chunk chunk, string indent = "")
        {
            switch (chunk)
            {
                case REVM revm:
                    Console.WriteLine($"{indent}Version: 0x{revm.Version:X} ({revm.Version})");
                    break;
                    
                case IBDM ibdm:
                    Console.WriteLine($"{indent}Index: 0x{ibdm.Index:X8} ({ibdm.Index})");
                    break;
                    
                case FBDM fbdm:
                    Console.WriteLine($"{indent}File Path: {fbdm.FilePath}");
                    break;
                    
                case HBDM hbdm:
                    Console.WriteLine($"{indent}Header Value: 0x{hbdm.Value:X8} ({hbdm.Value})");
                    break;
                    
                case MSPL mspl:
                    Console.WriteLine($"{indent}Version: {mspl.Version}");
                    Console.WriteLine($"{indent}Number of Splits: {mspl.NumSplits}");
                    
                    // Only show the first few splits if there are many
                    int displayCount = Math.Min(mspl.Splits.Count, 3);
                    
                    for (int i = 0; i < displayCount; i++)
                    {
                        var split = mspl.Splits[i];
                        Console.WriteLine($"{indent}Split {i + 1}:");
                        Console.WriteLine($"{indent}  Center: ({split.Center.X}, {split.Center.Y}, {split.Center.Z})");
                        Console.WriteLine($"{indent}  Normal: ({split.Normal.X}, {split.Normal.Y}, {split.Normal.Z})");
                        Console.WriteLine($"{indent}  Height: {split.Height}");
                    }
                    
                    if (mspl.Splits.Count > displayCount)
                    {
                        Console.WriteLine($"{indent}... and {mspl.Splits.Count - displayCount} more splits");
                    }
                    break;
                
                case BaseMeshChunk meshChunk:
                    string[] lines = meshChunk.GetDetailedDescription().Split('\n');
                    foreach (var line in lines)
                    {
                        Console.WriteLine($"{indent}{line}");
                    }
                    break;
                
                case UnknownChunk unknown:
                    Console.WriteLine($"{indent}Unknown Chunk Type");
                    Console.WriteLine($"{indent}Data Length: {unknown.Data.Length} bytes");
                    
                    // Add a hex dump of the first few bytes
                    if (unknown.Data.Length > 0)
                    {
                        int bytesToShow = Math.Min(unknown.Data.Length, 32);
                        int bytesPerLine = 16;
                        
                        // Show hex dump
                        for (int i = 0; i < bytesToShow; i += bytesPerLine)
                        {
                            Console.Write($"{indent}");
                            
                            // Offset
                            Console.Write($"{i:X4}: ");
                            
                            // Hex values
                            int lineLength = Math.Min(bytesPerLine, bytesToShow - i);
                            for (int j = 0; j < lineLength; j++)
                            {
                                Console.Write($"{unknown.Data[i + j]:X2} ");
                            }
                            
                            // Padding for alignment if needed
                            for (int j = lineLength; j < bytesPerLine; j++)
                            {
                                Console.Write("   ");
                            }
                            
                            // ASCII representation
                            Console.Write("  ");
                            for (int j = 0; j < lineLength; j++)
                            {
                                byte b = unknown.Data[i + j];
                                char c = b >= 32 && b <= 126 ? (char)b : '.';
                                Console.Write(c);
                            }
                            
                            Console.WriteLine();
                        }
                        
                        if (unknown.Data.Length > bytesToShow)
                        {
                            Console.WriteLine($"{indent}... and {unknown.Data.Length - bytesToShow} more bytes");
                        }
                    }
                    break;
            }
        }
    }
} 