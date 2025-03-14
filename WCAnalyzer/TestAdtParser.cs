using System;
using System.IO;
using System.Threading.Tasks;
using WCAnalyzer.Core.Services;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Test
{
    public class TestAdtParser
    {
        public static async Task Main(string[] args)
        {
            // Set up logging
            var logger = new ConsoleLogger(LogLevel.Debug);
            
            // Create parser
            var parser = new AdtParser(logger);
            
            // Path to test data
            string testDataPath = Path.Combine("..", "..", "test_data", "development");
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
                    var result = await parser.ParseAdtFileAsync(adtFile);
                    
                    Console.WriteLine($"  Coordinates: ({result.XCoord}, {result.YCoord})");
                    Console.WriteLine($"  Version: {result.Version}");
                    Console.WriteLine($"  Flags: {result.Flags}");
                    Console.WriteLine($"  Terrain chunks: {result.TerrainChunkCount}");
                    Console.WriteLine($"  Model placements: {result.ModelPlacementCount}");
                    Console.WriteLine($"  WMO placements: {result.WmoPlacementCount}");
                    Console.WriteLine($"  Texture references: {result.TextureReferences.Count}");
                    Console.WriteLine($"  Model references: {result.ModelReferences.Count}");
                    Console.WriteLine($"  WMO references: {result.WmoReferences.Count}");
                    
                    if (result.Errors.Count > 0)
                    {
                        Console.WriteLine($"  Errors: {result.Errors.Count}");
                        foreach (var error in result.Errors)
                        {
                            Console.WriteLine($"    - {error}");
                        }
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
} 