using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Services;
using WCAnalyzer.Core.Models;

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
            string testDataPath = args.Length > 0 
                ? args[0] 
                : Path.Combine("..", "..", "test_data", "development");
                
            if (File.Exists(testDataPath))
            {
                // Single file mode
                await ParseAndAnalyzeFile(adtParser, testDataPath, logger);
                return;
            }
            
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
                await ParseAndAnalyzeFile(adtParser, adtFile, logger);
            }
        }
        
        static async Task ParseAndAnalyzeFile(AdtParser parser, string adtFilePath, ILoggingService logger)
        {
            try
            {
                Console.WriteLine($"\nParsing file: {Path.GetFileName(adtFilePath)}");
                
                // First parse into AdtInfo
                var adtInfo = parser.ParseAdtFile(adtFilePath);
                
                // Show FileDataID information
                Console.WriteLine($"  Model FileDataIDs found: {adtInfo.ReferencedModels.Count}");
                if (adtInfo.ReferencedModels.Count > 0)
                {
                    Console.WriteLine("  Sample model FileDataIDs:");
                    foreach (var modelId in adtInfo.ReferencedModels.Take(5))
                    {
                        Console.WriteLine($"    {modelId}");
                    }
                }
                
                Console.WriteLine($"  WMO FileDataIDs found: {adtInfo.ReferencedWmos.Count}");
                if (adtInfo.ReferencedWmos.Count > 0)
                {
                    Console.WriteLine("  Sample WMO FileDataIDs:");
                    foreach (var wmoId in adtInfo.ReferencedWmos.Take(5))
                    {
                        Console.WriteLine($"    {wmoId}");
                    }
                }
                
                // Convert to analysis result
                var result = parser.ConvertToAdtAnalysisResult(adtInfo);
                
                Console.WriteLine($"  Coordinates: ({result.XCoord}, {result.YCoord})");
                Console.WriteLine($"  ADT Version: {result.AdtVersion}");
                Console.WriteLine($"  Terrain Chunks: {result.Header.TerrainChunkCount}");
                Console.WriteLine($"  Model Placements: {result.Header.ModelPlacementCount}");
                Console.WriteLine($"  WMO Placements: {result.Header.WmoPlacementCount}");
                Console.WriteLine($"  Texture References: {result.TextureReferences.Count}");
                Console.WriteLine($"  Model References: {result.ModelReferences.Count}");
                Console.WriteLine($"  WMO References: {result.WmoReferences.Count}");
                
                // Check if there are model references from FileDataIDs
                var modelRefsFromFileDataId = result.ModelReferences
                    .Where(r => r.NameId > 0)
                    .ToList();
                    
                Console.WriteLine($"  Model references from FileDataIDs: {modelRefsFromFileDataId.Count}");
                
                // Check if there are WMO references from FileDataIDs
                var wmoRefsFromFileDataId = result.WmoReferences
                    .Where(r => r.NameId > 0)
                    .ToList();
                    
                Console.WriteLine($"  WMO references from FileDataIDs: {wmoRefsFromFileDataId.Count}");
                
                // Save result to JSON
                string outputPath = Path.ChangeExtension(adtFilePath, ".json");
                var jsonOptions = new JsonSerializerOptions
                {
                    WriteIndented = true
                };
                await File.WriteAllTextAsync(outputPath, JsonSerializer.Serialize(result, jsonOptions));
                Console.WriteLine($"  Analysis result saved to: {outputPath}");
                
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
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"  Inner exception: {ex.InnerException.Message}");
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
        
        public void LogCritical(string message)
        {
            Console.WriteLine($"[CRITICAL] {message}");
        }
    }
} 