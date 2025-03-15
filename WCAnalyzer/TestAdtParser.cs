using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Text.Json;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Services;

namespace TestAdtParser
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create logger factory
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder
                    .AddFilter("Microsoft", LogLevel.Warning)
                    .AddFilter("System", LogLevel.Warning)
                    .AddFilter("TestAdtParser", LogLevel.Debug)
                    .AddConsole();
            });
            
            var logger = loggerFactory.CreateLogger<Program>();
            logger.LogInformation("AdtParser Test Program");

            if (args.Length == 0)
            {
                logger.LogError("Please provide a path to an ADT file");
                return;
            }

            string adtPath = args[0];
            if (!File.Exists(adtPath))
            {
                logger.LogError($"File not found: {adtPath}");
                return;
            }

            try
            {
                // Create AdtParser
                var parser = new AdtParser(loggerFactory.CreateLogger<AdtParser>());
                
                // Parse ADT file
                logger.LogInformation($"Parsing ADT file: {adtPath}");
                var adtInfo = parser.ParseAdtFile(adtPath);
                
                // Log model and WMO references
                logger.LogInformation($"Found {adtInfo.ReferencedModels.Count} model FileDataIDs");
                foreach (var modelId in adtInfo.ReferencedModels)
                {
                    logger.LogInformation($"Model FileDataID: {modelId}");
                }
                
                logger.LogInformation($"Found {adtInfo.ReferencedWmos.Count} WMO FileDataIDs");
                foreach (var wmoId in adtInfo.ReferencedWmos)
                {
                    logger.LogInformation($"WMO FileDataID: {wmoId}");
                }
                
                // Convert to AdtAnalysisResult
                var result = parser.ConvertToAdtAnalysisResult(adtInfo);
                
                // Log model and WMO references from result
                logger.LogInformation($"Result contains {result.ModelReferences.Count} model references");
                logger.LogInformation($"Result contains {result.WmoReferences.Count} WMO references");
                
                // Output as JSON
                var jsonOptions = new JsonSerializerOptions
                {
                    WriteIndented = true
                };
                
                string outputPath = Path.ChangeExtension(adtPath, ".json");
                File.WriteAllText(outputPath, JsonSerializer.Serialize(result, jsonOptions));
                logger.LogInformation($"JSON output written to: {outputPath}");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error parsing ADT file");
            }
        }
    }
} 