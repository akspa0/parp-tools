using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Services;

namespace TestAdtParser
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Create a simple logging service
            var logger = new ConsoleLoggingService();
            
            // Create the ADT parser
            var adtParser = new AdtParser(logger);
            
            // Path to test data
            string testDataPath = Path.Combine("..", "..", "test_data", "development");
            if (!Directory.Exists(testDataPath))
            {
                Console.WriteLine($"Test data directory not found: {testDataPath}");
                return;
            }
            
            Console.WriteLine($"Found test data directory: {testDataPath}");
            
            // Find ADT files
            var adtFiles = Directory.GetFiles(testDataPath, "*.adt", SearchOption.AllDirectories);
            Console.WriteLine($"Found {adtFiles.Length} ADT files");
            
            // Parse a few sample files
            foreach (var adtFile in adtFiles.Take(5))
            {
                try
                {
                    Console.WriteLine($"\nParsing file: {Path.GetFileName(adtFile)}");
                    var result = await adtParser.ParseAdtFileAsync(adtFile);
                    
                    Console.WriteLine($"  Coordinates: ({result.XCoord}, {result.YCoord})");
                    Console.WriteLine($"  ADT Version: {result.AdtVersion}");
                    Console.WriteLine($"  Terrain Chunks: {result.Header.TerrainChunkCount}");
                    Console.WriteLine($"  Model Placements: {result.Header.ModelPlacementCount}");
                    Console.WriteLine($"  WMO Placements: {result.Header.WmoPlacementCount}");
                    Console.WriteLine($"  Texture References: {result.TextureReferences.Count}");
                    Console.WriteLine($"  Model References: {result.ModelReferences.Count}");
                    Console.WriteLine($"  WMO References: {result.WmoReferences.Count}");
                    
                    // Show some sample textures
                    if (result.TextureReferences.Count > 0)
                    {
                        Console.WriteLine("\n  Sample Textures:");
                        foreach (var texture in result.TextureReferences.Take(5))
                        {
                            Console.WriteLine($"    {texture.OriginalPath}");
                        }
                    }
                    
                    // Show some sample models
                    if (result.ModelReferences.Count > 0)
                    {
                        Console.WriteLine("\n  Sample Models:");
                        foreach (var model in result.ModelReferences.Take(5))
                        {
                            Console.WriteLine($"    {model.OriginalPath}");
                        }
                    }
                    
                    // Show some sample WMOs
                    if (result.WmoReferences.Count > 0)
                    {
                        Console.WriteLine("\n  Sample WMOs:");
                        foreach (var wmo in result.WmoReferences.Take(5))
                        {
                            Console.WriteLine($"    {wmo.OriginalPath}");
                        }
                    }
                    
                    // Show any errors
                    if (result.Errors.Count > 0)
                    {
                        Console.WriteLine("\n  Errors:");
                        foreach (var error in result.Errors)
                        {
                            Console.WriteLine($"    {error}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Error parsing file: {ex.Message}");
                }
            }
        }
    }
    
    // Simple console logging service implementation
    class ConsoleLoggingService : ILoggingService
    {
        public void LogDebug(string message)
        {
            Console.WriteLine($"[DEBUG] {message}");
        }
        
        public void LogInfo(string message)
        {
            Console.WriteLine($"[INFO] {message}");
        }
        
        public void LogWarning(string message)
        {
            Console.WriteLine($"[WARNING] {message}");
        }
        
        public void LogError(string message)
        {
            Console.WriteLine($"[ERROR] {message}");
        }
    }
} 