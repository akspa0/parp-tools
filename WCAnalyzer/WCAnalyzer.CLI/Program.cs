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

namespace WCAnalyzer.CLI
{
    // Custom JSON converter for Vector3 to ensure whole number formatting
    public class Vector3Converter : JsonConverter<Vector3>
    {
        public override Vector3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            throw new NotImplementedException("Reading Vector3 from JSON is not supported.");
        }

        public override void Write(Utf8JsonWriter writer, Vector3 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", (float)Math.Round(value.X, 2));
            writer.WriteNumber("Y", (float)Math.Round(value.Y, 2));
            writer.WriteNumber("Z", (float)Math.Round(value.Z, 2));
            writer.WriteEndObject();
        }
    }

    // Add a custom float converter to control decimal places and prevent scientific notation
    public class CustomFloatConverter : JsonConverter<float>
    {
        public override float Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            return reader.GetSingle();
        }

        public override void Write(Utf8JsonWriter writer, float value, JsonSerializerOptions options)
        {
            // Round to 2 decimal places and format without scientific notation
            writer.WriteNumberValue((float)Math.Round(value, 2));
        }
    }

    // Add a custom integer converter for large values to ensure they're properly formatted
    public class CustomIntConverter : JsonConverter<int>
    {
        public override int Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            return reader.GetInt32();
        }

        public override void Write(Utf8JsonWriter writer, int value, JsonSerializerOptions options)
        {
            // Validate integer value to prevent ridiculously large values
            if (value > 100000000 || value < -100000000) // Arbitrary limit for sanity
            {
                writer.WriteNumberValue(0); // Replace with 0 for clearly invalid values
            }
            else
            {
                writer.WriteNumberValue(value);
            }
        }
    }

    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Set up command line options
            var rootCommand = new RootCommand("WCAnalyzer CLI tool for analyzing World of Warcraft files");

            // Add options
            var directoryOption = new Option<string>(
                "--directory",
                "Directory containing ADT files to analyze")
            { IsRequired = true };

            var outputOption = new Option<string>(
                "--output",
                description: "Output directory for analysis results")
            {
                IsRequired = false
            };
            outputOption.AddAlias("-o");

            var verboseOption = new Option<bool>(
                "--verbose",
                "Enable verbose logging");

            // Add options to command
            rootCommand.AddOption(directoryOption);
            rootCommand.AddOption(outputOption);
            rootCommand.AddOption(verboseOption);

            // Set up handler
            rootCommand.SetHandler(async (string directory, string output, bool verbose) =>
            {
                // Set up logging
                var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(verbose ? Microsoft.Extensions.Logging.LogLevel.Debug : Microsoft.Extensions.Logging.LogLevel.Information);
                });

                var logger = loggerFactory.CreateLogger<Program>();

                // Create services
                var adtParser = new AdtParser(loggerFactory.CreateLogger<AdtParser>());
                var csvGenerator = new TerrainDataCsvGenerator(loggerFactory.CreateLogger<TerrainDataCsvGenerator>());
                var markdownGenerator = new MarkdownReportGenerator(loggerFactory.CreateLogger<MarkdownReportGenerator>());
                
                // Check output directory if specified
                if (!string.IsNullOrEmpty(output))
                {
                    // Create a timestamped subfolder for this run
                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    output = Path.Combine(output, $"run_{timestamp}");
                    
                    Directory.CreateDirectory(output);
                    logger.LogInformation("Analysis results will be saved to {OutputDirectory}", Path.GetFullPath(output));
                }

                // Find ADT files
                var adtFiles = Directory.GetFiles(directory, "*.adt", SearchOption.AllDirectories);
                logger.LogInformation("Found {Count} ADT files in {Directory}", adtFiles.Length, directory);

                // Group related ADT files (base, _obj0, _tex0) by their coordinates
                var adtFileGroups = new Dictionary<string, List<string>>();
                foreach (var adtFile in adtFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(adtFile);
                    
                    // Extract the base name without _obj0 or _tex0 suffix
                    string baseName = fileName;
                    if (fileName.EndsWith("_obj0") || fileName.EndsWith("_tex0"))
                    {
                        baseName = fileName.Substring(0, fileName.LastIndexOf('_'));
                    }
                    
                    if (!adtFileGroups.ContainsKey(baseName))
                    {
                        adtFileGroups[baseName] = new List<string>();
                    }
                    
                    adtFileGroups[baseName].Add(adtFile);
                }
                
                logger.LogInformation("Grouped {FileCount} files into {GroupCount} logical ADT units", 
                    adtFiles.Length, adtFileGroups.Count);

                int successCount = 0;
                int errorCount = 0;
                var results = new List<AdtAnalysisResult>();
                var startTime = DateTime.Now;

                // Create custom JSON serialization options
                var jsonOptions = new JsonSerializerOptions
                {
                    WriteIndented = true,
                    NumberHandling = JsonNumberHandling.Strict,
                    Converters = 
                    {
                        new Vector3Converter(), // Custom converter for Vector3
                        new CustomFloatConverter(), // Controls float formatting
                        new CustomIntConverter(), // Validates integers
                        new JsonStringEnumConverter() // For better enum formatting
                    }
                };

                // Process ADT file groups
                foreach (var group in adtFileGroups)
                {
                    string baseName = group.Key;
                    List<string> relatedFiles = group.Value;
                    
                    try
                    {
                        // Initialize a consolidated result
                        AdtAnalysisResult consolidatedResult = null;
                        
                        // Process each file in the group
                        foreach (var adtFile in relatedFiles)
                        {
                            var result = await adtParser.ParseAsync(adtFile);
                            logger.LogDebug("Parsed component {FileName}: {TerrainChunks} terrain chunks, {TextureCount} textures, {ModelCount} models, {WmoCount} WMOs",
                                result.FileName,
                                result.TerrainChunks.Count,
                                result.TextureReferences.Count,
                                result.ModelReferences.Count,
                                result.WmoReferences.Count);
                            
                            // If this is the first file, use it as the base for consolidated result
                            if (consolidatedResult == null)
                            {
                                consolidatedResult = result;
                                // Standardize the filename to the base name
                                consolidatedResult.FileName = baseName + ".adt";
                            }
                            else
                            {
                                // Merge data from this result into the consolidated result
                                MergeResults(consolidatedResult, result);
                            }
                        }
                        
                        // Log the consolidated result
                        logger.LogInformation("Successfully processed ADT group {BaseName}: {TerrainChunks} terrain chunks, {TextureCount} textures, {ModelCount} models, {WmoCount} WMOs",
                            baseName,
                            consolidatedResult.TerrainChunks.Count,
                            consolidatedResult.TextureReferences.Count,
                            consolidatedResult.ModelReferences.Count,
                            consolidatedResult.WmoReferences.Count);
                        
                        // Add to results list
                        results.Add(consolidatedResult);
                        
                        // Save result to output directory if specified
                        if (!string.IsNullOrEmpty(output))
                        {
                            var outputFile = Path.Combine(output, baseName + ".json");
                            await File.WriteAllTextAsync(outputFile, JsonSerializer.Serialize(consolidatedResult, jsonOptions));
                            logger.LogDebug("Saved consolidated analysis result to {OutputFile}", outputFile);
                        }
                        
                        successCount++;
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error processing ADT group {BaseName}", baseName);
                        errorCount++;
                    }
                }

                // Create summary
                var endTime = DateTime.Now;
                var summary = new WCAnalyzer.Core.Models.AnalysisSummary
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    TotalFiles = adtFileGroups.Count,
                    ProcessedFiles = successCount,
                    FailedFiles = errorCount,
                    TotalTextureReferences = results.Sum(r => r.TextureReferences.Count),
                    TotalModelReferences = results.Sum(r => r.ModelReferences.Count),
                    TotalWmoReferences = results.Sum(r => r.WmoReferences.Count),
                    TotalTerrainChunks = results.Sum(r => r.TerrainChunks.Count),
                    TotalModelPlacements = results.Sum(r => r.ModelPlacements.Count),
                    TotalWmoPlacements = results.Sum(r => r.WmoPlacements.Count)
                };
                summary.Complete();

                // Write summary to file
                if (!string.IsNullOrEmpty(output))
                {
                    File.WriteAllText(Path.Combine(output, "summary.txt"),
                        $"Analysis Summary\n" +
                        $"---------------\n" +
                        $"Analysis Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}\n" +
                        $"Total ADT Units: {summary.TotalFiles}\n" +
                        $"Processed Units: {summary.ProcessedFiles}\n" +
                        $"Failed Units: {summary.FailedFiles}\n" +
                        $"Analysis Duration: {summary.Duration.TotalSeconds:F2} seconds\n\n" +
                        $"Total Texture References: {summary.TotalTextureReferences}\n" +
                        $"Total Model References: {summary.TotalModelReferences}\n" +
                        $"Total WMO References: {summary.TotalWmoReferences}\n" +
                        $"Total Terrain Chunks: {summary.TotalTerrainChunks}\n" +
                        $"Total Model Placements: {summary.TotalModelPlacements}\n" +
                        $"Total WMO Placements: {summary.TotalWmoPlacements}");
                    
                    // Generate CSV reports
                    logger.LogInformation("Generating CSV reports...");
                    await csvGenerator.GenerateAllCsvAsync(results, output);
                    
                    // Generate Markdown reports
                    logger.LogInformation("Generating Markdown reports...");
                    // Convert to the appropriate AnalysisSummary type for the markdown generator
                    var serviceSummary = new WCAnalyzer.Core.Services.AnalysisSummary
                    {
                        TotalFiles = summary.TotalFiles,
                        ProcessedFiles = summary.ProcessedFiles,
                        FailedFiles = summary.FailedFiles,
                        TotalTextureReferences = summary.TotalTextureReferences,
                        TotalModelReferences = summary.TotalModelReferences,
                        TotalWmoReferences = summary.TotalWmoReferences,
                        TotalTerrainChunks = summary.TotalTerrainChunks,
                        TotalModelPlacements = summary.TotalModelPlacements,
                        TotalWmoPlacements = summary.TotalWmoPlacements,
                        // Copy any other properties needed
                        MissingReferences = summary.MissingReferences,
                        FilesNotInListfile = summary.FilesNotInListfile,
                        DuplicateIds = summary.DuplicateIds,
                        MaxUniqueId = summary.MaxUniqueId,
                        ParsingErrors = summary.ParsingErrors
                    };
                    await markdownGenerator.GenerateReportsAsync(results, serviceSummary, output);
                }

                logger.LogInformation("Analysis complete. Successfully processed {SuccessCount} of {TotalCount} ADT units. {ErrorCount} units had errors.",
                    successCount, adtFileGroups.Count, errorCount);
            }, directoryOption, outputOption, verboseOption);

            return await rootCommand.InvokeAsync(args);
        }
        
        /// <summary>
        /// Merges data from the source result into the target result
        /// </summary>
        private static void MergeResults(AdtAnalysisResult target, AdtAnalysisResult source)
        {
            // Determine if source is a _tex0 file (contains texture information)
            bool isSourceTexFile = source.FileName.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase);
            bool isSourceObjFile = source.FileName.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase);
            
            // Special handling for terrain chunks based on file type
            foreach (var chunk in source.TerrainChunks)
            {
                // Find matching terrain chunk in target if it exists
                var existingChunk = target.TerrainChunks.FirstOrDefault(c => 
                    c.Position.X == chunk.Position.X && c.Position.Y == chunk.Position.Y);
                
                if (existingChunk == null)
                {
                    // If chunk doesn't exist in target, add it
                    target.TerrainChunks.Add(chunk);
                }
                else if (isSourceTexFile)
                {
                    // For _tex0 files, prioritize their texture layers but keep other chunk data
                    
                    // If source chunk has texture layers, replace the layers in the target chunk
                    if (chunk.TextureLayers != null && chunk.TextureLayers.Count > 0)
                    {
                        // Save any existing texture-less layers if they should be preserved
                        var existingEmptyLayers = existingChunk.TextureLayers
                            .Where(l => string.IsNullOrEmpty(l.TextureName) || l.TextureName.StartsWith("<unknown"))
                            .ToList();
                        
                        // Clear existing texture layers and add ones from tex0 file
                        existingChunk.TextureLayers.Clear();
                        existingChunk.TextureLayers.AddRange(chunk.TextureLayers);
                        
                        // If we had placeholder layers and no real textures were added, restore the placeholders
                        if (existingEmptyLayers.Count > 0 && 
                            chunk.TextureLayers.All(l => string.IsNullOrEmpty(l.TextureName) || l.TextureName.StartsWith("<unknown")))
                        {
                            existingChunk.TextureLayers.AddRange(existingEmptyLayers);
                        }
                    }
                    
                    // Merge alpha maps if available
                    if (chunk.AlphaMaps != null && chunk.AlphaMaps.Count > 0)
                    {
                        existingChunk.AlphaMaps.AddRange(chunk.AlphaMaps);
                    }
                }
                else if (isSourceObjFile)
                {
                    // For _obj0 files, focus on model/WMO references but preserve existing texture data
                    // You could add special handling here if needed
                }
            }
            
            // Add model references if they don't already exist
            foreach (var modelRef in source.ModelReferences)
            {
                if (!target.ModelReferences.Any(m => m.OriginalPath == modelRef.OriginalPath))
                {
                    target.ModelReferences.Add(modelRef);
                }
            }
            
            // Add WMO references if they don't already exist
            foreach (var wmoRef in source.WmoReferences)
            {
                if (!target.WmoReferences.Any(w => w.OriginalPath == wmoRef.OriginalPath))
                {
                    target.WmoReferences.Add(wmoRef);
                }
            }
            
            // Add texture references if they don't already exist
            foreach (var texRef in source.TextureReferences)
            {
                if (!target.TextureReferences.Any(t => t.OriginalPath == texRef.OriginalPath))
                {
                    target.TextureReferences.Add(texRef);
                }
            }
            
            // Add model placements
            target.ModelPlacements.AddRange(source.ModelPlacements);
            
            // Add WMO placements
            target.WmoPlacements.AddRange(source.WmoPlacements);
            
            // Update header information to reflect the merged data
            if (target.Header != null)
            {
                // Update counts in the header
                target.Header.ModelReferenceCount = target.ModelReferences.Count;
                target.Header.WmoReferenceCount = target.WmoReferences.Count;
                target.Header.ModelPlacementCount = target.ModelPlacements.Count;
                target.Header.WmoPlacementCount = target.WmoPlacements.Count;
                target.Header.TerrainChunkCount = target.TerrainChunks.Count;
                
                // Determine if we have various data types
                bool hasHeightData = target.TerrainChunks.Any(c => c.Heights != null && c.Heights.Length > 0);
                bool hasNormalData = target.TerrainChunks.Any(c => c.Normals != null && c.Normals.Length > 0);
                bool hasLiquidData = target.TerrainChunks.Any(c => c.LiquidLevel > 0);
                bool hasVertexShading = target.TerrainChunks.Any(c => c.VertexColors != null && c.VertexColors.Count > 0);
                
                // Update the Flags value based on the presence of data
                uint newFlags = target.Header.Flags;
                newFlags = hasHeightData ? newFlags | 0x1U : newFlags & ~0x1U;
                newFlags = hasNormalData ? newFlags | 0x2U : newFlags & ~0x2U;
                newFlags = hasLiquidData ? newFlags | 0x4U : newFlags & ~0x4U;
                newFlags = hasVertexShading ? newFlags | 0x8U : newFlags & ~0x8U;
                target.Header.Flags = newFlags;
                
                // Update texture layer count based on the maximum number of texture layers in any chunk
                int maxTextureLayers = target.TerrainChunks.Count > 0 
                    ? target.TerrainChunks.Max(c => c.TextureLayers?.Count ?? 0) 
                    : 0;
                target.Header.TextureLayerCount = maxTextureLayers;
            }
        }
    }
}
