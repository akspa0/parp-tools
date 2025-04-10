using System;
using System.IO;
using System.Threading.Tasks;
using Warcraft.NET.Files.ADT;

class SimpleAdtTest
{
    static async Task Main(string[] args)
    {
        // Path to test data
        string testDataPath = Path.Combine("..", "test_data", "development");
        if (!Directory.Exists(testDataPath))
        {
            Console.WriteLine($"Test data directory not found: {testDataPath}");
            Console.WriteLine("Current directory: " + Directory.GetCurrentDirectory());
            return;
        }
        
        Console.WriteLine($"Found test data directory: {testDataPath}");
        
        // Find ADT files
        var adtFiles = Directory.GetFiles(testDataPath, "*.adt", SearchOption.AllDirectories);
        Console.WriteLine($"Found {adtFiles.Length} ADT files");
        
        // Parse each ADT file
        foreach (var adtFile in adtFiles)
        {
            Console.WriteLine($"\nParsing file: {Path.GetFileName(adtFile)}");
            try
            {
                // Read the file data
                byte[] fileData = await File.ReadAllBytesAsync(adtFile);
                
                // Parse the ADT file using Warcraft.NET
                var terrain = new Terrain(fileData);
                
                // Display basic information
                Console.WriteLine($"  Version: {terrain.Version}");
                Console.WriteLine($"  Flags: {terrain.Flags}");
                
                // Display chunk information if available
                if (terrain.Chunks != null)
                {
                    Console.WriteLine($"  Terrain chunks: {terrain.Chunks.Count}");
                    
                    // Display the first chunk's information if available
                    if (terrain.Chunks.Count > 0)
                    {
                        var firstChunk = terrain.Chunks[0];
                        Console.WriteLine($"  First chunk area ID: {firstChunk.AreaId}");
                    }
                }
                else
                {
                    Console.WriteLine("  No terrain chunks found");
                }
                
                // Display texture information if available
                if (terrain.TextureNames != null)
                {
                    Console.WriteLine($"  Texture names: {terrain.TextureNames.Count}");
                    
                    // Display the first few texture names if available
                    for (int i = 0; i < Math.Min(5, terrain.TextureNames.Count); i++)
                    {
                        Console.WriteLine($"    - {terrain.TextureNames[i]}");
                    }
                }
                else
                {
                    Console.WriteLine("  No texture names found");
                }
                
                // Display model information if available
                if (terrain.ModelNames != null)
                {
                    Console.WriteLine($"  Model names: {terrain.ModelNames.Count}");
                    
                    // Display the first few model names if available
                    for (int i = 0; i < Math.Min(5, terrain.ModelNames.Count); i++)
                    {
                        Console.WriteLine($"    - {terrain.ModelNames[i]}");
                    }
                }
                else
                {
                    Console.WriteLine("  No model names found");
                }
                
                // Display WMO information if available
                if (terrain.WmoNames != null)
                {
                    Console.WriteLine($"  WMO names: {terrain.WmoNames.Count}");
                    
                    // Display the first few WMO names if available
                    for (int i = 0; i < Math.Min(5, terrain.WmoNames.Count); i++)
                    {
                        Console.WriteLine($"    - {terrain.WmoNames[i]}");
                    }
                }
                else
                {
                    Console.WriteLine("  No WMO names found");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error parsing file: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"  Inner exception: {ex.InnerException.Message}");
                }
            }
        }
    }
} 