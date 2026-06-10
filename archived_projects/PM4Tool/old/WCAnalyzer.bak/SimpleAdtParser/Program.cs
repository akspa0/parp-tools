using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleAdtParser
{
    class Program
    {
        static async Task Main(string[] args)
        {
            string testDataDir = @"..\..\test_data\development";
            
            if (!Directory.Exists(testDataDir))
            {
                Console.WriteLine($"Test data directory not found: {testDataDir}");
                return;
            }

            Console.WriteLine($"Analyzing ADT files in {testDataDir}");
            Console.WriteLine();

            var adtFiles = Directory.GetFiles(testDataDir, "*.adt");
            Console.WriteLine($"Found {adtFiles.Length} ADT files");
            Console.WriteLine();

            int successCount = 0;
            int errorCount = 0;
            int totalTerrainChunks = 0;
            int totalTextureNames = 0;
            int totalModelNames = 0;
            int totalWmoNames = 0;
            int totalModelPlacements = 0;
            int totalWmoPlacements = 0;

            // Process all files
            foreach (var adtFile in adtFiles)
            {
                try
                {
                    Console.WriteLine($"Parsing file: {Path.GetFileName(adtFile)}");
                    
                    // Read the file data
                    byte[] fileData = await File.ReadAllBytesAsync(adtFile);
                    
                    // Check if the file has reversed chunk IDs
                    if (fileData.Length >= 4)
                    {
                        string signature = Encoding.ASCII.GetString(fileData, 0, 4);
                        Console.WriteLine($"First 4 bytes: {signature}");
                        
                        if (signature == "REVM")
                        {
                            Console.WriteLine("File has reversed chunk IDs, correcting...");
                            fileData = CorrectChunkIds(fileData);
                            Console.WriteLine($"First 4 bytes after correction: {Encoding.ASCII.GetString(fileData, 0, 4)}");
                        }
                    }
                    
                    // Parse the ADT file manually
                    var adtInfo = ParseAdtFile(fileData);
                    
                    // Output information about the ADT file
                    Console.WriteLine($"Version: {adtInfo.Version}");
                    Console.WriteLine($"Flags: {adtInfo.Flags}");
                    Console.WriteLine($"Terrain chunks: {adtInfo.TerrainChunks}");
                    
                    // Output texture names
                    if (adtInfo.TextureNames.Count > 0)
                    {
                        Console.WriteLine($"Texture names: {adtInfo.TextureNames.Count}");
                        foreach (var textureName in adtInfo.TextureNames.Take(5)) // Limit to 5 for brevity
                        {
                            Console.WriteLine($"  {textureName}");
                        }
                        if (adtInfo.TextureNames.Count > 5)
                        {
                            Console.WriteLine($"  ... and {adtInfo.TextureNames.Count - 5} more");
                        }
                    }
                    
                    // Output model names
                    if (adtInfo.ModelNames.Count > 0)
                    {
                        Console.WriteLine($"Model names: {adtInfo.ModelNames.Count}");
                        foreach (var modelName in adtInfo.ModelNames.Take(5)) // Limit to 5 for brevity
                        {
                            Console.WriteLine($"  {modelName}");
                        }
                        if (adtInfo.ModelNames.Count > 5)
                        {
                            Console.WriteLine($"  ... and {adtInfo.ModelNames.Count - 5} more");
                        }
                    }
                    
                    // Output WMO names
                    if (adtInfo.WmoNames.Count > 0)
                    {
                        Console.WriteLine($"WMO names: {adtInfo.WmoNames.Count}");
                        foreach (var wmoName in adtInfo.WmoNames.Take(5)) // Limit to 5 for brevity
                        {
                            Console.WriteLine($"  {wmoName}");
                        }
                        if (adtInfo.WmoNames.Count > 5)
                        {
                            Console.WriteLine($"  ... and {adtInfo.WmoNames.Count - 5} more");
                        }
                    }
                    
                    // Output model placements
                    if (adtInfo.ModelPlacements > 0)
                    {
                        Console.WriteLine($"Model placements: {adtInfo.ModelPlacements}");
                    }
                    
                    // Output WMO placements
                    if (adtInfo.WmoPlacements > 0)
                    {
                        Console.WriteLine($"WMO placements: {adtInfo.WmoPlacements}");
                    }
                    
                    // Update totals
                    totalTerrainChunks += adtInfo.TerrainChunks;
                    totalTextureNames += adtInfo.TextureNames.Count;
                    totalModelNames += adtInfo.ModelNames.Count;
                    totalWmoNames += adtInfo.WmoNames.Count;
                    totalModelPlacements += adtInfo.ModelPlacements;
                    totalWmoPlacements += adtInfo.WmoPlacements;
                    
                    successCount++;
                    Console.WriteLine(new string('-', 80));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error parsing {Path.GetFileName(adtFile)}: {ex.Message}");
                    errorCount++;
                    Console.WriteLine(new string('-', 80));
                }
            }

            // Output summary
            Console.WriteLine("Analysis complete.");
            Console.WriteLine($"Successfully processed {successCount} of {adtFiles.Length} files. {errorCount} files had errors.");
            Console.WriteLine($"Total terrain chunks: {totalTerrainChunks}");
            Console.WriteLine($"Total texture names: {totalTextureNames}");
            Console.WriteLine($"Total model names: {totalModelNames}");
            Console.WriteLine($"Total WMO names: {totalWmoNames}");
            Console.WriteLine($"Total model placements: {totalModelPlacements}");
            Console.WriteLine($"Total WMO placements: {totalWmoPlacements}");
        }

        static byte[] CorrectChunkIds(byte[] fileData)
        {
            byte[] correctedData = new byte[fileData.Length];
            Array.Copy(fileData, correctedData, fileData.Length);
            
            int offset = 0;
            while (offset + 8 <= correctedData.Length)
            {
                // Reverse the chunk ID (4 bytes)
                Array.Reverse(correctedData, offset, 4);
                
                // Read the chunk size
                int chunkSize = BitConverter.ToInt32(correctedData, offset + 4);
                
                // Move to the next chunk
                offset += 8 + chunkSize;
                
                // Safety check to prevent infinite loops
                if (chunkSize < 0 || offset < 0)
                {
                    Console.WriteLine("Invalid chunk size or offset detected. Stopping correction.");
                    break;
                }
            }
            
            return correctedData;
        }

        static AdtInfo ParseAdtFile(byte[] fileData)
        {
            var adtInfo = new AdtInfo();
            
            using (var ms = new MemoryStream(fileData))
            using (var br = new BinaryReader(ms))
            {
                // Parse chunks
                while (ms.Position < ms.Length - 8) // Need at least 8 bytes for chunk header
                {
                    // Read chunk header
                    string chunkId = Encoding.ASCII.GetString(br.ReadBytes(4));
                    int chunkSize = br.ReadInt32();
                    
                    // Skip invalid chunks
                    if (chunkSize < 0 || chunkSize > 100 * 1024 * 1024) // 100 MB max
                    {
                        ms.Position += 4;
                        continue;
                    }
                    
                    // Process chunk based on ID
                    switch (chunkId)
                    {
                        case "MVER": // Version chunk
                            adtInfo.Version = br.ReadInt32();
                            break;
                            
                        case "MHDR": // Header chunk
                            adtInfo.Flags = br.ReadUInt32();
                            // Skip the rest of the header
                            ms.Position += chunkSize - 4;
                            break;
                            
                        case "MCNK": // Terrain chunk
                            adtInfo.TerrainChunks++;
                            // Skip the chunk data
                            ms.Position += chunkSize;
                            break;
                            
                        case "MTEX": // Texture names chunk
                            long endPos = ms.Position + chunkSize;
                            while (ms.Position < endPos)
                            {
                                string textureName = ReadNullTerminatedString(br);
                                if (!string.IsNullOrWhiteSpace(textureName))
                                {
                                    adtInfo.TextureNames.Add(textureName);
                                }
                                
                                // Break if we've reached the end of the chunk
                                if (ms.Position >= endPos)
                                    break;
                            }
                            break;
                            
                        case "MMDX": // Model names chunk
                            endPos = ms.Position + chunkSize;
                            while (ms.Position < endPos)
                            {
                                string modelName = ReadNullTerminatedString(br);
                                if (!string.IsNullOrWhiteSpace(modelName))
                                {
                                    adtInfo.ModelNames.Add(modelName);
                                }
                                
                                // Break if we've reached the end of the chunk
                                if (ms.Position >= endPos)
                                    break;
                            }
                            break;
                            
                        case "MWMO": // WMO names chunk
                            endPos = ms.Position + chunkSize;
                            while (ms.Position < endPos)
                            {
                                string wmoName = ReadNullTerminatedString(br);
                                if (!string.IsNullOrWhiteSpace(wmoName))
                                {
                                    adtInfo.WmoNames.Add(wmoName);
                                }
                                
                                // Break if we've reached the end of the chunk
                                if (ms.Position >= endPos)
                                    break;
                            }
                            break;
                            
                        case "MDDF": // Model placements chunk
                            adtInfo.ModelPlacements = chunkSize / 36; // Each entry is 36 bytes
                            // Skip the chunk data
                            ms.Position += chunkSize;
                            break;
                            
                        case "MODF": // WMO placements chunk
                            adtInfo.WmoPlacements = chunkSize / 64; // Each entry is 64 bytes
                            // Skip the chunk data
                            ms.Position += chunkSize;
                            break;
                            
                        default:
                            // Skip unknown chunks
                            ms.Position += chunkSize;
                            break;
                    }
                }
            }
            
            return adtInfo;
        }

        static string ReadNullTerminatedString(BinaryReader br)
        {
            var bytes = new List<byte>();
            byte b;
            while ((b = br.ReadByte()) != 0)
            {
                bytes.Add(b);
                
                // Safety check to prevent infinite loops
                if (bytes.Count > 1000) // Max string length
                {
                    break;
                }
            }
            
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
    }

    class AdtInfo
    {
        public int Version { get; set; }
        public uint Flags { get; set; }
        public int TerrainChunks { get; set; }
        public List<string> TextureNames { get; set; } = new List<string>();
        public List<string> ModelNames { get; set; } = new List<string>();
        public List<string> WmoNames { get; set; } = new List<string>();
        public int ModelPlacements { get; set; }
        public int WmoPlacements { get; set; }
    }
}
