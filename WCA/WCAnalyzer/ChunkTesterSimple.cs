using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace ChunkTester
{
    /// <summary>
    /// Simplified standalone chunk tester for PM4/PD4 files
    /// </summary>
    public static class ChunkTesterSimple
    {
        private static readonly Dictionary<string, string> ChunkDescriptions = new Dictionary<string, string>
        {
            { "MCRC", "CRC-32 Checksum" },
            { "MVER", "Version Information" },
            { "MSC3", "Scene Header" },
            { "MSCN", "Scene" },
            { "MSVI", "Scene Visibility Information" },
            { "MSHI", "Shadow Information" },
            { "MSUR", "Surface Information" },
            { "MSPI", "Spawn Information" },
            { "MSLK", "Skybox Link" },
            { "MSVT", "Scene Variables Table" },
            { "MSHD", "Shadow Batch Data" },
            { "MDBF", "Database File" },
            { "MDBI", "Database Information" },
            { "MDOS", "Database Object Storage" },
            { "MDSF", "Database String Formats" },
            { "MPRL", "Properties List" },
            { "MPRR", "Properties Record" }
        };

        public static void Run(string[] args)
        {
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
            Console.WriteLine("ID   | Size      | Offset     | Description");
            Console.WriteLine("-----|-----------|------------|------------");
            
            foreach (var chunk in chunks)
            {
                string description = ChunkDescriptions.ContainsKey(chunk.Id) 
                    ? ChunkDescriptions[chunk.Id] 
                    : "Unknown";
                
                Console.WriteLine($"{chunk.Id} | {chunk.Size,10} | {chunk.Offset,10} | {description}");
            }
            
            Console.WriteLine();
            Console.WriteLine("Done!");
        }

        private class ChunkInfo
        {
            public string Id { get; set; }
            public uint Size { get; set; }
            public long Offset { get; set; }
        }
    }
} 