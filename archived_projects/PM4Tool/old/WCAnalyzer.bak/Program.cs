using System;
using System.CommandLine;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Services;
using WCAnalyzer.UniqueIdAnalysis;
// Add explicit alias for AnalysisSummary to avoid ambiguity
using ModelsSummary = WCAnalyzer.Core.Models.AnalysisSummary;
using ServicesSummary = WCAnalyzer.Core.Services.AnalysisSummary;
// Add explicit alias for ReportGenerator to avoid ambiguity
using CoreReportGenerator = WCAnalyzer.Core.Services.ReportGenerator;
using UniqueIdReportGenerator = WCAnalyzer.UniqueIdAnalysis.ReportGenerator;
using WCAnalyzer.Core.Models.PM4;
using System.Text;

namespace ChunkTesterStandalone
{
    class Program
    {
        private static readonly Dictionary<string, string> ChunkDescriptions = new Dictionary<string, string>
        {
            { "MVER", "" },
            { "MCRC", "" },
            { "MSHD", "" },
            { "MSPV", "" },
            { "MSPI", "" },
            { "MSCN", "" },
            { "MSLK", "" },
            { "MSVT", "" },
            { "MSVI", "" },
            { "MSUR", "" },
            { "MPRL", "" },
            { "MPRR", "" },
            { "MDBH", "" },
            { "MDBF", "" },
            { "MDBI", "" },
            { "MDOS", "" },
            { "MDSF", "" }
        };

        static void Main(string[] args)
        {
            Console.WriteLine("PM4/PD4 Chunk Tester");
            Console.WriteLine("====================");
            
            if (args.Length == 0)
            {
                Console.WriteLine("Please provide a file path to a PM4 or PD4 file.");
                return;
            }

            string filePath = args[0];
            
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"File not found: {filePath}");
                return;
            }

            try
            {
                using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
                using (BinaryReader reader = new BinaryReader(fs))
                {
                    ProcessFile(reader, filePath);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing file: {ex.Message}");
            }
        }

        private static void ProcessFile(BinaryReader reader, string filePath)
        {
            Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");
            Console.WriteLine("---------------------------------");

            var chunks = new List<ChunkInfo>();
            long fileLength = reader.BaseStream.Length;
            long position = 0;

            while (position < fileLength)
            {
                reader.BaseStream.Position = position;
                
                // Make sure we have at least 8 bytes to read (ChunkID + Size)
                if (fileLength - position < 8)
                    break;

                // Read chunk ID (4 bytes)
                byte[] chunkIdBytes = reader.ReadBytes(4);
                string chunkId = Encoding.ASCII.GetString(chunkIdBytes);
                
                // Read chunk size (4 bytes)
                uint chunkSize = reader.ReadUInt32();
                
                // Store chunk info
                chunks.Add(new ChunkInfo
                {
                    Id = chunkId,
                    Size = chunkSize,
                    Offset = position
                });
                
                // Move to the next chunk
                position += 8 + chunkSize;
            }

            // Display chunk information
            Console.WriteLine($"Found {chunks.Count} chunks:");
            Console.WriteLine();
            Console.WriteLine("ID   | Size      | Offset     ");
            Console.WriteLine("-----|-----------|------------");
            
            foreach (var chunk in chunks)
            {
                Console.WriteLine($"{chunk.Id} | {chunk.Size,10} | {chunk.Offset,10}");
            }
            
            Console.WriteLine();
            Console.WriteLine("Done!");

            // Read chunk data
            foreach (var chunk in chunks)
            {
                reader.BaseStream.Position = chunk.Offset;
                
                // Read chunk data
                byte[] chunkData;
                try
                {
                    chunkData = reader.ReadBytes((int)chunk.Size);
                    if (chunkData.Length < chunk.Size)
                    {
                        Console.WriteLine($"Warning: Could only read {chunkData.Length} bytes of {chunk.Size} for chunk {chunk.Id}");
                    }
                    
                    // Debug: Dump MTEX chunk to file
                    if (chunk.Id == "XETM")
                    {
                        File.WriteAllBytes("debug_mtex_chunk.bin", chunkData);
                        Console.WriteLine("DEBUG: Written MTEX chunk raw data to debug_mtex_chunk.bin");
                        
                        // Dump the first 100 bytes as hex
                        StringBuilder hexDump = new StringBuilder();
                        for (int i = 0; i < Math.Min(chunkData.Length, 100); i++)
                        {
                            hexDump.AppendFormat("{0:X2} ", chunkData[i]);
                            if ((i + 1) % 16 == 0) hexDump.AppendLine();
                        }
                        Console.WriteLine($"MTEX hex dump (first bytes):");
                        Console.WriteLine(hexDump.ToString());
                        
                        // Try to read strings
                        using (MemoryStream ms = new MemoryStream(chunkData))
                        using (BinaryReader br = new BinaryReader(ms))
                        {
                            int textureCount = 0;
                            try 
                            {
                                while (ms.Position < ms.Length)
                                {
                                    string texture = ReadCString(br);
                                    if (!string.IsNullOrEmpty(texture))
                                    {
                                        Console.WriteLine($"Found texture {textureCount++}: {texture}");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error reading strings from MTEX: {ex.Message}");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reading chunk {chunk.Id}: {ex.Message}");
                }
            }
        }

        private class ChunkInfo
        {
            public string Id { get; set; } = string.Empty;
            public uint Size { get; set; }
            public long Offset { get; set; }
            public byte[] Data { get; set; } = Array.Empty<byte>();
        }

        private static string ReadCString(BinaryReader reader)
        {
            List<byte> stringBytes = new List<byte>();
            
            try
            {
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    byte b = reader.ReadByte();
                    if (b == 0) break;
                    stringBytes.Add(b);
                }
                
                if (stringBytes.Count > 0)
                {
                    return Encoding.ASCII.GetString(stringBytes.ToArray());
                }
            }
            catch (IOException)
            {
                // End of stream reached
            }
            
            return string.Empty;
        }
    }
}
