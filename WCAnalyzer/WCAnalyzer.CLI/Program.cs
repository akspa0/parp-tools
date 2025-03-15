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

    // Custom JSON converter for ModelPlacement
    public class ModelPlacementConverter : JsonConverter<ModelPlacement>
    {
        public override ModelPlacement Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            throw new NotImplementedException("Reading ModelPlacement from JSON is not supported.");
        }

        public override void Write(Utf8JsonWriter writer, ModelPlacement value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            
            // Check if this is a FileDataID reference and resolve it if possible
            bool isFileDataIdRef = FileReferenceConverter.IsFileDataIdReference(value.Name, out uint fileDataId);
            string resolvedPath = value.Name;
            
            if (isFileDataIdRef && fileDataId > 0 && FileReferenceConverter.TryGetResolvedPath(fileDataId, out string? mappedPath))
            {
                resolvedPath = mappedPath;
            }
            else if (value.UsesFileDataId && value.FileDataId > 0 && FileReferenceConverter.TryGetResolvedPath(value.FileDataId, out mappedPath))
            {
                resolvedPath = mappedPath;
            }
            
            // Write basic properties
            writer.WriteNumber("UniqueId", value.UniqueId);
            writer.WriteNumber("NameId", value.NameId);
            
            // Write Name and ResolvedName
            writer.WriteString("Name", value.Name);
            if (resolvedPath != value.Name)
            {
                writer.WriteString("ResolvedName", resolvedPath);
            }
            
            // Write Position as an object with X, Y, Z properties
            writer.WritePropertyName("Position");
            JsonSerializer.Serialize(writer, value.Position, options);
            
            // Write Rotation as an object with X, Y, Z properties
            writer.WritePropertyName("Rotation");
            JsonSerializer.Serialize(writer, value.Rotation, options);
            
            // Write Scale
            writer.WriteNumber("Scale", value.Scale);
            
            // Write Flags
            writer.WriteNumber("Flags", value.Flags);
            
            // Write FileDataId if available
            if (value.FileDataId > 0)
            {
                writer.WriteNumber("FileDataId", value.FileDataId);
            }
            
            // Write UsesFileDataId if true
            if (value.UsesFileDataId)
            {
                writer.WriteBoolean("UsesFileDataId", value.UsesFileDataId);
            }
            
            writer.WriteEndObject();
        }
    }
    
    // Custom JSON converter for WmoPlacement
    public class WmoPlacementConverter : JsonConverter<WmoPlacement>
    {
        public override WmoPlacement Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            throw new NotImplementedException("Reading WmoPlacement from JSON is not supported.");
        }

        public override void Write(Utf8JsonWriter writer, WmoPlacement value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            
            // Check if this is a FileDataID reference and resolve it if possible
            bool isFileDataIdRef = FileReferenceConverter.IsFileDataIdReference(value.Name, out uint fileDataId);
            string resolvedPath = value.Name;
            
            if (isFileDataIdRef && fileDataId > 0 && FileReferenceConverter.TryGetResolvedPath(fileDataId, out string? mappedPath))
            {
                resolvedPath = mappedPath;
            }
            else if (value.UsesFileDataId && value.FileDataId > 0 && FileReferenceConverter.TryGetResolvedPath(value.FileDataId, out mappedPath))
            {
                resolvedPath = mappedPath;
            }
            
            // Write basic properties
            writer.WriteNumber("UniqueId", value.UniqueId);
            writer.WriteNumber("NameId", value.NameId);
            
            // Write Name and ResolvedName
            writer.WriteString("Name", value.Name);
            if (resolvedPath != value.Name)
            {
                writer.WriteString("ResolvedName", resolvedPath);
            }
            
            // Write Position as an object with X, Y, Z properties
            writer.WritePropertyName("Position");
            JsonSerializer.Serialize(writer, value.Position, options);
            
            // Write Rotation as an object with X, Y, Z properties
            writer.WritePropertyName("Rotation");
            JsonSerializer.Serialize(writer, value.Rotation, options);
            
            // Write Scale
            writer.WriteNumber("Scale", value.Scale);
            
            // Write DoodadSet
            writer.WriteNumber("DoodadSet", value.DoodadSet);
            
            // Write NameSet
            writer.WriteNumber("NameSet", value.NameSet);
            
            // Write Flags
            writer.WriteNumber("Flags", value.Flags);
            
            // Write FileDataId if available
            if (value.FileDataId > 0)
            {
                writer.WriteNumber("FileDataId", value.FileDataId);
            }
            
            // Write UsesFileDataId if true
            if (value.UsesFileDataId)
            {
                writer.WriteBoolean("UsesFileDataId", value.UsesFileDataId);
            }
            
            writer.WriteEndObject();
        }
    }

    // Custom JSON converter for FileReference to exclude NormalizedPath
    public class FileReferenceConverter : JsonConverter<FileReference>
    {
        // Add a static dictionary to store FileDataID-to-path mappings
        private static Dictionary<uint, string> FileDataIdToPathMap = new Dictionary<uint, string>();
        
        // Static method to initialize the FileDataID-to-path map from the ReferenceValidator
        public static void InitializeFileDataIdMap(ReferenceValidator referenceValidator)
        {
            FileDataIdToPathMap.Clear();
            
            // Use reflection to access the private _fileDataIdToPathMap dictionary from ReferenceValidator
            var fieldInfo = typeof(ReferenceValidator).GetField("_fileDataIdToPathMap", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (fieldInfo != null)
            {
                var map = fieldInfo.GetValue(referenceValidator) as Dictionary<uint, string>;
                if (map != null)
                {
                    // Copy all mappings to our static dictionary
                    foreach (var kvp in map)
                    {
                        FileDataIdToPathMap[kvp.Key] = kvp.Value;
                    }
                }
            }
        }
        
        // Helper method to check if a string is in the format "<FileDataID:12345>"
        public static bool IsFileDataIdReference(string path, out uint fileDataId)
        {
            fileDataId = 0;
            if (string.IsNullOrEmpty(path)) return false;
            
            // Check if the path follows the pattern <FileDataID:12345>
            var match = System.Text.RegularExpressions.Regex.Match(path, @"<FileDataID:(\d+)>");
            if (match.Success && match.Groups.Count > 1)
            {
                if (uint.TryParse(match.Groups[1].Value, out uint id))
                {
                    fileDataId = id;
                    return true;
                }
            }
            
            return false;
        }
        
        // Helper method to get the resolved path for a FileDataID
        public static bool TryGetResolvedPath(uint fileDataId, out string resolvedPath)
        {
            return FileDataIdToPathMap.TryGetValue(fileDataId, out resolvedPath);
        }
        
        public override FileReference Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            throw new NotImplementedException("Reading FileReference from JSON is not supported.");
        }

        public override void Write(Utf8JsonWriter writer, FileReference value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            
            // Check if this is a FileDataID reference and resolve it if possible
            bool isFileDataIdRef = IsFileDataIdReference(value.OriginalPath, out uint fileDataId);
            string resolvedPath = value.OriginalPath;
            
            if (isFileDataIdRef && fileDataId > 0 && TryGetResolvedPath(fileDataId, out string? mappedPath))
            {
                resolvedPath = mappedPath;
            }
            
            // Write OriginalPath
            writer.WriteString("OriginalPath", value.OriginalPath);
            
            // Write the ResolvedPath if it's different from OriginalPath
            if (isFileDataIdRef && resolvedPath != value.OriginalPath)
            {
                writer.WriteString("ResolvedPath", resolvedPath);
            }
            
            // Write Type as string
            writer.WriteString("Type", value.Type.ToString());
            
            // Write IsValid
            writer.WriteBoolean("IsValid", value.IsValid);
            
            // Write ExistsInListfile
            writer.WriteBoolean("ExistsInListfile", value.ExistsInListfile);
            
            // Write FileDataId if available
            if (value.FileDataId > 0)
            {
                writer.WriteNumber("FileDataId", value.FileDataId);
            }
            
            // Write UsesFileDataId if true
            if (value.UsesFileDataId)
            {
                writer.WriteBoolean("UsesFileDataId", value.UsesFileDataId);
            }
            
            // Write MatchedByFileDataId if true
            if (value.MatchedByFileDataId)
            {
                writer.WriteBoolean("MatchedByFileDataId", value.MatchedByFileDataId);
            }
            
            // Write RepairedPath if not null or empty
            if (!string.IsNullOrEmpty(value.RepairedPath))
            {
                writer.WriteString("RepairedPath", value.RepairedPath);
            }
            
            // Write AlternativeExtensionPath if not null or empty
            if (!string.IsNullOrEmpty(value.AlternativeExtensionPath) && value.AlternativeExtensionFound)
            {
                writer.WriteString("AlternativeExtensionPath", value.AlternativeExtensionPath);
                writer.WriteBoolean("AlternativeExtensionFound", value.AlternativeExtensionFound);
            }
            
            writer.WriteEndObject();
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
                
            var listfileOption = new Option<string>(
                "--listfile",
                description: "Path to a listfile containing known file paths and FileDataIDs in format: <FileDataID>;<asset path>")
            {
                IsRequired = false
            };
            listfileOption.AddAlias("-l");

            // Add options to command
            rootCommand.AddOption(directoryOption);
            rootCommand.AddOption(outputOption);
            rootCommand.AddOption(verboseOption);
            rootCommand.AddOption(listfileOption);

            // Set up handler
            rootCommand.SetHandler(async (string directory, string output, bool verbose, string listfile) =>
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
                var csvGenerator = new TerrainDataCsvGenerator(loggerFactory.CreateLogger<TerrainDataCsvGenerator>(), output);
                var markdownGenerator = new MarkdownReportGenerator(loggerFactory.CreateLogger<MarkdownReportGenerator>());
                var jsonGenerator = new JsonReportGenerator(loggerFactory.CreateLogger<JsonReportGenerator>());
                var referenceValidator = new ReferenceValidator(loggerFactory.CreateLogger<ReferenceValidator>());
                
                // Check output directory if specified
                if (!string.IsNullOrEmpty(output))
                {
                    // Create a timestamped subfolder for this run
                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    output = Path.Combine(output, $"run_{timestamp}");
                    
                    Directory.CreateDirectory(output);
                    logger.LogInformation("Analysis results will be saved to {OutputDirectory}", Path.GetFullPath(output));
                }
                
                // Load listfile if specified
                HashSet<string> knownGoodFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                if (!string.IsNullOrEmpty(listfile))
                {
                    if (File.Exists(listfile))
                    {
                        logger.LogInformation("Loading listfile from {ListfilePath}", Path.GetFullPath(listfile));
                        knownGoodFiles = await referenceValidator.LoadListfileAsync(listfile);
                        logger.LogInformation("Loaded {Count} entries from listfile", knownGoodFiles.Count);
                        
                        // Initialize the FileDataID-to-path map for the FileReferenceConverter
                        FileReferenceConverter.InitializeFileDataIdMap(referenceValidator);
                    }
                    else
                    {
                        logger.LogWarning("Specified listfile {ListfilePath} not found. Reference validation will be limited.", listfile);
                    }
                }
                else
                {
                    logger.LogInformation("No listfile specified. File references will not be validated against known good files.");
                }

                // Find ADT files
                var adtFiles = Directory.GetFiles(directory, "*.adt", SearchOption.AllDirectories);
                logger.LogInformation("Found {Count} ADT files in {Directory}", adtFiles.Length, directory);

                // Updated JsonOptions with new converters
                var jsonOptions = new JsonSerializerOptions
                {
                    WriteIndented = true,
                    NumberHandling = JsonNumberHandling.Strict,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                    Converters = 
                    {
                        new Vector3Converter(), // Custom converter for Vector3
                        new CustomFloatConverter(), // Controls float formatting
                        new CustomIntConverter(), // Validates integers
                        new FileReferenceConverter(), // Excludes NormalizedPath from output
                        new ModelPlacementConverter(), // Resolves FileDataID in ModelPlacements
                        new WmoPlacementConverter(), // Resolves FileDataID in WmoPlacements
                        new JsonStringEnumConverter() // For better enum formatting
                    }
                };

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
                        
                        // Get grouping key using base name
                        if (!adtFileGroups.ContainsKey(baseName))
                        {
                            adtFileGroups[baseName] = new List<string>();
                        }
                        
                        adtFileGroups[baseName].Add(adtFile);
                        
                        // Check if base ADT exists
                        string baseAdtPath = Path.Combine(Path.GetDirectoryName(adtFile) ?? string.Empty, $"{baseName}.adt");
                        if (File.Exists(baseAdtPath) && !adtFileGroups[baseName].Contains(baseAdtPath))
                        {
                            adtFileGroups[baseName].Add(baseAdtPath);
                        }
                    }
                    else
                    {
                        // This is a base ADT file
                        if (!adtFileGroups.ContainsKey(baseName))
                        {
                            adtFileGroups[baseName] = new List<string>();
                        }
                        
                        // Add base file if not already present
                        if (!adtFileGroups[baseName].Contains(adtFile))
                        {
                            adtFileGroups[baseName].Add(adtFile);
                        }
                        
                        // Check for potential _obj0 and _tex0 files
                        string objPath = Path.Combine(Path.GetDirectoryName(adtFile) ?? string.Empty, $"{baseName}_obj0.adt");
                        string texPath = Path.Combine(Path.GetDirectoryName(adtFile) ?? string.Empty, $"{baseName}_tex0.adt");
                        
                        if (File.Exists(objPath) && !adtFileGroups[baseName].Contains(objPath))
                        {
                            adtFileGroups[baseName].Add(objPath);
                        }
                        
                        if (File.Exists(texPath) && !adtFileGroups[baseName].Contains(texPath))
                        {
                            adtFileGroups[baseName].Add(texPath);
                        }
                    }
                }
                
                logger.LogInformation("Grouped {FileCount} files into {GroupCount} logical ADT units", 
                    adtFiles.Length, adtFileGroups.Count);

                int successCount = 0;
                int errorCount = 0;
                var results = new List<AdtAnalysisResult>();
                var startTime = DateTime.Now;

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
                        if (consolidatedResult != null)
                        {
                            logger.LogInformation("Successfully processed ADT group {BaseName}: {TerrainChunks} terrain chunks, {TextureCount} textures, {ModelCount} models, {WmoCount} WMOs",
                                baseName,
                                consolidatedResult.TerrainChunks.Count,
                                consolidatedResult.TextureReferences.Count,
                                consolidatedResult.ModelReferences.Count,
                                consolidatedResult.WmoReferences.Count);
                            
                            // Add to results list
                            results.Add(consolidatedResult);
                            
                            // Validate references if we have a listfile
                            if (knownGoodFiles.Count > 0)
                            {
                                int invalidCount = referenceValidator.ValidateReferences(consolidatedResult, knownGoodFiles);
                                if (invalidCount > 0)
                                {
                                    logger.LogWarning("{InvalidCount} invalid references found in {FileName}", 
                                        invalidCount, consolidatedResult.FileName);
                                }
                            }
                        }
                        else
                        {
                            logger.LogWarning("No valid result was created for ADT group {BaseName}", baseName);
                        }
                        
                        // Save result to output directory if specified
                        if (!string.IsNullOrEmpty(output))
                        {
                            var outputFile = Path.Combine(output, baseName + ".json");
                            var serializerOptions = new JsonSerializerOptions(jsonOptions)
                            {
                                // Ensure we write empty collections rather than omitting them entirely
                                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                                IgnoreReadOnlyProperties = false,
                                IncludeFields = true
                            };
                            await File.WriteAllTextAsync(outputFile, JsonSerializer.Serialize(consolidatedResult, serializerOptions));
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
                var summary = new ModelsSummary
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
                    
                    // Convert to the Services version of AnalysisSummary for the markdown generator
                    var servicesSummary = new ServicesSummary
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
                        ParsingErrors = summary.ParsingErrors,
                        // Copy collection properties
                        MissingReferenceMap = summary.MissingReferenceMap,
                        FilesNotInListfileMap = summary.FilesNotInListfileMap,
                        DuplicateIdMap = summary.DuplicateIdMap,
                        DuplicateIdSet = summary.DuplicateIdSet,
                        AreaIdMap = summary.AreaIdMap,
                        // Copy time-related properties
                        StartTime = summary.StartTime,
                        EndTime = summary.EndTime,
                        Duration = summary.Duration
                    };
                    
                    // Use the appropriate types for each generator
                    await markdownGenerator.GenerateReportsAsync(results, servicesSummary, output);

                    // Generate JSON reports
                    logger.LogInformation("Generating JSON reports...");
                    
                    // The JsonReportGenerator also uses the Services.AnalysisSummary type
                    await jsonGenerator.GenerateAllReportsAsync(results, servicesSummary, output);
                }

                logger.LogInformation("Analysis complete. Successfully processed {SuccessCount} of {TotalCount} ADT units. {ErrorCount} units had errors.",
                    successCount, adtFileGroups.Count, errorCount);
            }, directoryOption, outputOption, verboseOption, listfileOption);

            // Create a command for UniqueID analysis
            var uniqueIdCommand = new Command("uniqueid", "Analyze UniqueID data from ADT analysis results");
            
            // Add options for UniqueID analysis
            var resultsDirectoryOption = new Option<string>(
                "--results-directory",
                description: "Directory containing JSON results from ADT analysis")
            { IsRequired = true };
            resultsDirectoryOption.AddAlias("-r");
            
            var uniqueIdOutputOption = new Option<string>(
                "--output",
                description: "Directory to write UniqueID analysis results to")
            { IsRequired = true };
            uniqueIdOutputOption.AddAlias("-o");
            
            var clusterThresholdOption = new Option<int>(
                "--cluster-threshold",
                () => 10,
                "Minimum number of IDs to form a cluster");
            
            var clusterGapThresholdOption = new Option<int>(
                "--cluster-gap",
                () => 1000,
                "Maximum gap between IDs to be considered part of the same cluster");
            
            var comprehensiveReportOption = new Option<bool>(
                "--comprehensive",
                () => true,
                "Generate a comprehensive report with all assets");
            
            var exportCsvOption = new Option<bool>(
                "--export-csv",
                () => true,
                "Export data to CSV files in addition to markdown reports");
            
            // Add listfile option for UniqueID analysis
            var uniqueIdListfileOption = new Option<string?>(
                "--listfile",
                description: "Path to a listfile for resolving FileDataID references");
            uniqueIdListfileOption.AddAlias("-l");
            
            // Add options to uniqueId command
            uniqueIdCommand.AddOption(resultsDirectoryOption);
            uniqueIdCommand.AddOption(uniqueIdOutputOption);
            uniqueIdCommand.AddOption(clusterThresholdOption);
            uniqueIdCommand.AddOption(clusterGapThresholdOption);
            uniqueIdCommand.AddOption(comprehensiveReportOption);
            uniqueIdCommand.AddOption(exportCsvOption);
            uniqueIdCommand.AddOption(verboseOption);
            uniqueIdCommand.AddOption(uniqueIdListfileOption);
            
            // Set up handler for uniqueId command
            uniqueIdCommand.SetHandler(async (string resultsDirectory, string output, int clusterThreshold, int clusterGap, bool comprehensive, bool exportCsv, bool verbose, string? listfilePath) =>
            {
                // Set up logging
                var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(verbose ? Microsoft.Extensions.Logging.LogLevel.Debug : Microsoft.Extensions.Logging.LogLevel.Information);
                });
                
                var logger = loggerFactory.CreateLogger<Program>();
                
                try
                {
                    logger.LogInformation("Starting UniqueID analysis...");
                    
                    // Create output directory if it doesn't exist
                    if (!Directory.Exists(output))
                    {
                        Directory.CreateDirectory(output);
                        logger.LogInformation("Created output directory: {OutputDir}", output);
                    }
                    
                    // Create the UniqueIdAnalyzer
                    var analyzer = new UniqueIdAnalyzer(
                        resultsDirectory,
                        output,
                        loggerFactory.CreateLogger<UniqueIdAnalyzer>(),
                        clusterThreshold,
                        clusterGap,
                        comprehensive);
                    
                    // Run the analysis
                    var result = await analyzer.AnalyzeAsync(comprehensive);
                    
                    // Create the reference validator if we have a listfile
                    ReferenceValidator? referenceValidator = null;
                    if (!string.IsNullOrEmpty(listfilePath))
                    {
                        referenceValidator = new ReferenceValidator(loggerFactory.CreateLogger<ReferenceValidator>());
                    }
                    
                    // Generate reports
                    var reportGenerator = new UniqueIdReportGenerator(
                        loggerFactory.CreateLogger<UniqueIdReportGenerator>(),
                        referenceValidator);
                    
                    // Load the listfile if provided
                    if (referenceValidator != null && !string.IsNullOrEmpty(listfilePath))
                    {
                        await reportGenerator.LoadListfileAsync(listfilePath);
                    }
                    
                    // Generate markdown reports
                    logger.LogInformation("Generating markdown reports...");
                    
                    // Generate summary report
                    await reportGenerator.GenerateSummaryReportAsync(result, Path.Combine(output, "summary.md"));
                    
                    // Generate comprehensive full summary report
                    await reportGenerator.GenerateFullSummaryReportAsync(result, Path.Combine(output, "full_summary.md"));
                    
                    // Generate detailed reports for each cluster
                    foreach (var cluster in result.Clusters)
                    {
                        var clusterFileName = $"cluster_{cluster.MinId}-{cluster.MaxId}.md";
                        await reportGenerator.GenerateClusterReportAsync(cluster, Path.Combine(output, clusterFileName));
                    }
                    
                    // Generate CSV reports if enabled
                    if (exportCsv)
                    {
                        logger.LogInformation("Generating CSV reports...");
                        var csvGenerator = new CsvReportGenerator(loggerFactory.CreateLogger<CsvReportGenerator>());
                        await csvGenerator.GenerateAllCsvReportsAsync(result, output);
                    }
                    
                    logger.LogInformation("UniqueID analysis complete!");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error during UniqueID analysis");
                }
            }, resultsDirectoryOption, uniqueIdOutputOption, clusterThresholdOption, clusterGapThresholdOption, comprehensiveReportOption, exportCsvOption, verboseOption, uniqueIdListfileOption);
            
            // Add uniqueId command to root command
            rootCommand.Add(uniqueIdCommand);

            // Add PM4 analysis command
            var pm4Command = new Command("pm4", "Analyze PM4 files");

            // Add options to the PM4 command
            var pm4DirectoryOption = new Option<string>(
                "--directory",
                "Directory containing PM4 files to analyze");
            pm4DirectoryOption.AddAlias("-d");

            var pm4OutputOption = new Option<string>(
                "--output",
                "Directory to write analysis output to");
            pm4OutputOption.AddAlias("-o");

            var pm4VerboseOption = new Option<bool>(
                "--verbose",
                "Enable verbose logging");
            pm4VerboseOption.AddAlias("-v");

            var pm4QuietOption = new Option<bool>(
                "--quiet",
                "Suppress all but error messages");
            pm4QuietOption.AddAlias("-q");

            var pm4RecursiveOption = new Option<bool>(
                "--recursive",
                "Recursively search for files in subdirectories");
            pm4RecursiveOption.AddAlias("-r");

            var pm4ListfileOption = new Option<string>(
                "--listfile",
                "Path to a listfile for resolving FileDataID references");
            pm4ListfileOption.AddAlias("-l");

            var pm4ExportCsvOption = new Option<bool>(
                "--export-csv",
                () => true,
                "Export vertex data to CSV files");
            pm4ExportCsvOption.AddAlias("-c");

            var pm4ExportObjOption = new Option<bool>(
                "--export-obj",
                () => true,
                "Export geometry to Wavefront OBJ files");
            pm4ExportObjOption.AddAlias("-obj");

            var pm4ExportConsolidatedObjOption = new Option<bool>(
                "--export-consolidated-obj",
                () => true,
                "Export all geometry to a single consolidated Wavefront OBJ file");
            pm4ExportConsolidatedObjOption.AddAlias("-cobj");

            var pm4ExportClusteredObjOption = new Option<bool>(
                "--export-clustered-obj",
                () => false,
                "Export mesh data to clustered OBJ format with vertex proximity clustering");
            pm4ExportClusteredObjOption.AddAlias("-clobj");

            // Add new option for terrain extraction
            var pm4ExtractTerrainOption = new Option<bool>(
                "--extract-terrain",
                () => false,
                "Extract and export terrain data from PM4 position data");
            pm4ExtractTerrainOption.AddAlias("-terrain");

            pm4Command.AddOption(pm4DirectoryOption);
            pm4Command.AddOption(pm4OutputOption);
            pm4Command.AddOption(pm4VerboseOption);
            pm4Command.AddOption(pm4QuietOption);
            pm4Command.AddOption(pm4RecursiveOption);
            pm4Command.AddOption(pm4ListfileOption);
            pm4Command.AddOption(pm4ExportCsvOption);
            pm4Command.AddOption(pm4ExportObjOption);
            pm4Command.AddOption(pm4ExportConsolidatedObjOption);
            pm4Command.AddOption(pm4ExportClusteredObjOption);
            pm4Command.AddOption(pm4ExtractTerrainOption);

            var pm4FileOption = new Option<string>(
                "--file",
                "Path to a specific PM4 file to analyze");
            pm4FileOption.AddAlias("-f");
            pm4Command.AddOption(pm4FileOption);

            pm4Command.SetHandler(async (context) =>
            {
                string inputPath = context.ParseResult.GetValueForOption(pm4DirectoryOption) ?? string.Empty;
                string outputPath = context.ParseResult.GetValueForOption(pm4OutputOption) ?? string.Empty;
                bool verbose = context.ParseResult.GetValueForOption(pm4VerboseOption);
                bool quiet = context.ParseResult.GetValueForOption(pm4QuietOption);
                bool recursive = context.ParseResult.GetValueForOption(pm4RecursiveOption);
                string specificFile = context.ParseResult.GetValueForOption(pm4FileOption) ?? string.Empty;
                string listfilePath = context.ParseResult.GetValueForOption(pm4ListfileOption) ?? string.Empty;
                bool exportCsv = context.ParseResult.GetValueForOption(pm4ExportCsvOption);
                bool exportObj = context.ParseResult.GetValueForOption(pm4ExportObjOption);
                bool exportConsolidatedObj = context.ParseResult.GetValueForOption(pm4ExportConsolidatedObjOption);
                bool exportClusteredObj = context.ParseResult.GetValueForOption(pm4ExportClusteredObjOption);
                bool extractTerrain = context.ParseResult.GetValueForOption(pm4ExtractTerrainOption);

                // Configure logging with fully qualified type name
                Microsoft.Extensions.Logging.LogLevel minLevel = quiet 
                    ? Microsoft.Extensions.Logging.LogLevel.Error 
                    : (verbose ? Microsoft.Extensions.Logging.LogLevel.Debug : Microsoft.Extensions.Logging.LogLevel.Information);
                
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole().SetMinimumLevel(minLevel);
                });
                var logger = loggerFactory.CreateLogger<Program>();

                try
                {
                    if (string.IsNullOrEmpty(inputPath) && string.IsNullOrEmpty(specificFile))
                    {
                        logger.LogError("Either --directory or --file option must be specified");
                        context.ExitCode = 1;
                        return;
                    }

                    if (string.IsNullOrEmpty(outputPath))
                    {
                        outputPath = Path.Combine(Directory.GetCurrentDirectory(), "pm4_analysis_output");
                    }

                    Directory.CreateDirectory(outputPath);
                    logger.LogInformation("Output directory: {OutputPath}", outputPath);

                    // Use fully qualified types to avoid ambiguity
                    var pm4Parser = new WCAnalyzer.Core.Services.PM4Parser(
                        loggerFactory.CreateLogger<WCAnalyzer.Core.Services.PM4Parser>());
                    
                    // Create a properly configured ReportGenerator
                    var terrainDataCsvGenerator = new WCAnalyzer.Core.Services.TerrainDataCsvGenerator(
                        loggerFactory.CreateLogger<WCAnalyzer.Core.Services.TerrainDataCsvGenerator>(),
                        outputPath);
                    
                    // Create OBJ exporter
                    var pm4ObjExporter = new WCAnalyzer.Core.Services.PM4ObjExporter(
                        loggerFactory.CreateLogger<WCAnalyzer.Core.Services.PM4ObjExporter>(),
                        outputPath);
                    
                    // Create clustered OBJ exporter if needed
                    WCAnalyzer.Core.Services.PM4ClusteredObjExporter? pm4ClusteredObjExporter = null;
                    if (exportClusteredObj)
                    {
                        pm4ClusteredObjExporter = new WCAnalyzer.Core.Services.PM4ClusteredObjExporter(
                            loggerFactory.CreateLogger<WCAnalyzer.Core.Services.PM4ClusteredObjExporter>(),
                            outputPath);
                    }
                    
                    // Create report generator
                    var reportGenerator = new WCAnalyzer.Core.Services.ReportGenerator(
                        loggerFactory.CreateLogger<WCAnalyzer.Core.Services.ReportGenerator>(),
                        terrainDataCsvGenerator,
                        null, // jsonReportGenerator
                        null, // markdownReportGenerator
                        pm4ObjExporter);
                    
                    List<WCAnalyzer.Core.Models.PM4.PM4AnalysisResult> results;

                    if (!string.IsNullOrEmpty(specificFile))
                    {
                        // Analyze a specific file
                        logger.LogInformation("Analyzing specific PM4 file: {File}", specificFile);
                        if (!File.Exists(specificFile))
                        {
                            logger.LogError("File not found: {File}", specificFile);
                            context.ExitCode = 1;
                            return;
                        }

                        var result = pm4Parser.ParseFile(specificFile);
                        
                        // If a listfile was provided, resolve references
                        if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
                        {
                            logger.LogInformation("Resolving references using listfile: {Listfile}", listfilePath);
                            var listfile = LoadListfile(listfilePath, logger);
                            pm4Parser.ResolveReferences(result, listfile);
                        }
                        
                results = new List<WCAnalyzer.Core.Models.PM4.PM4AnalysisResult> { result };
            }
            else
            {
                // Analyze files in directory
                logger.LogInformation("Analyzing PM4 files in directory: {InputPath}", inputPath);
                if (!Directory.Exists(inputPath))
                {
                    logger.LogError("Directory not found: {InputPath}", inputPath);
                    context.ExitCode = 1;
                    return;
                }

                var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
                
                // Use the listfile if provided
                if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
                {
                    logger.LogInformation("Parsing with reference resolution using listfile: {Listfile}", listfilePath);
                    results = await pm4Parser.ParseDirectoryWithReferencesAsync(inputPath, listfilePath, "*.pm4", searchOption);
                }
                else
                {
                    results = await pm4Parser.ParseDirectoryAsync(inputPath, "*.pm4", searchOption);
                }
            }

            logger.LogInformation("Analyzed {Count} PM4 files", results.Count);

            // Generate reports with CSV option
            await reportGenerator.GeneratePM4ReportsAsync(results, outputPath, exportCsv, exportObj, exportConsolidatedObj);
            
            // Export to clustered OBJ if requested
            if (exportClusteredObj && pm4ClusteredObjExporter != null)
            {
                logger.LogInformation("Exporting to clustered OBJ format");
                await pm4ClusteredObjExporter.ExportAllToClusteredObjAsync(results);
                logger.LogInformation("Clustered OBJ export completed");
            }

            // Export terrain data if requested
            if (extractTerrain)
            {
                logger.LogInformation("Extracting terrain data from PM4 position data");
                string terrainObjPath = Path.Combine(outputPath, "terrain_reconstruction.obj");
                if (pm4ObjExporter.ExportTerrainFromPositionData(results, terrainObjPath))
                {
                    logger.LogInformation("Terrain data exported to: {Path}", terrainObjPath);
                }
                else
                {
                    logger.LogWarning("Failed to export terrain data");
                }
            }
            
            logger.LogInformation("PM4 analysis completed. Reports written to {OutputPath}", outputPath);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error during PM4 analysis");
            context.ExitCode = 1;
        }
    });

    rootCommand.AddCommand(pm4Command);

    // Parse the command line
    return await rootCommand.InvokeAsync(args);
}

/// <summary>
/// Merges data from the source result into the target result
/// </summary>
private static void MergeResults(AdtAnalysisResult target, AdtAnalysisResult source)
{
    if (target == null || source == null)
    {
        Console.WriteLine("Cannot merge null results");
        return;
    }

    // Log the contents of both results before merging
    Console.WriteLine($"Merging {source.FileName} into {target.FileName}");
    Console.WriteLine($"  Before merge - Target: {target.ModelPlacements?.Count ?? 0} models, {target.WmoPlacements?.Count ?? 0} WMOs, {target.TextureReferences?.Count ?? 0} textures");
    Console.WriteLine($"  Before merge - Source: {source.ModelPlacements?.Count ?? 0} models, {source.WmoPlacements?.Count ?? 0} WMOs, {source.TextureReferences?.Count ?? 0} textures");
    
    // Determine if source is a _tex0 file (contains texture information)
    bool isSourceTexFile = source.FileName.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase);
    bool isSourceObjFile = source.FileName.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase);
    
    // Ensure target has all references and placements from source
    int initialTextureCount = target.TextureReferences.Count;
    int initialModelCount = target.ModelReferences.Count;
    int initialWmoCount = target.WmoReferences.Count;
    int initialModelPlacementCount = target.ModelPlacements.Count;
    int initialWmoPlacementCount = target.WmoPlacements.Count;
    
    // Special handling for terrain chunks based on file type
    foreach (var chunk in source.TerrainChunks ?? Enumerable.Empty<TerrainChunk>())
    {
        // Find matching terrain chunk in target if it exists
        var existingChunk = target.TerrainChunks?.FirstOrDefault(c => 
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
    
    // Add model placements with deduplication
    foreach (var modelPlacement in source.ModelPlacements)
    {
        // Check if this placement already exists in the target based on UniqueId and position
        bool isDuplicate = target.ModelPlacements.Any(mp => 
            // First check by UniqueId if it's not zero (UniqueId == 0 is often used for invalid/default entries)
            (modelPlacement.UniqueId != 0 && mp.UniqueId == modelPlacement.UniqueId) ||
            // If UniqueIds don't match or are zero, check by position, rotation, scale, and name/nameid
            (Vector3.Distance(mp.Position, modelPlacement.Position) < 0.001f &&
             Vector3.Distance(mp.Rotation, modelPlacement.Rotation) < 0.001f &&
             Math.Abs(mp.Scale - modelPlacement.Scale) < 0.001f &&
             (mp.NameId == modelPlacement.NameId || mp.Name == modelPlacement.Name))
        );
        
        if (!isDuplicate)
        {
            target.ModelPlacements.Add(modelPlacement);
            
            // For logging/debugging
            if (isSourceObjFile || isSourceTexFile)
            {
                Console.WriteLine($"Added model placement from {(isSourceObjFile ? "_obj0" : "_tex0")} file: {modelPlacement.Name}, Position: {modelPlacement.Position}");
            }
        }
        else
        {
            Console.WriteLine($"Skipped duplicate model placement: {modelPlacement.Name}, UniqueId: {modelPlacement.UniqueId}, Position: {modelPlacement.Position}");
        }
    }
    
    // Add WMO placements with deduplication
    foreach (var wmoPlacement in source.WmoPlacements)
    {
        // Check if this placement already exists in the target based on UniqueId and position
        bool isDuplicate = target.WmoPlacements.Any(wp => 
            // First check by UniqueId if it's not zero
            (wmoPlacement.UniqueId != 0 && wp.UniqueId == wmoPlacement.UniqueId) ||
            // If UniqueIds don't match or are zero, check by position, rotation, and name/nameid
            (Vector3.Distance(wp.Position, wmoPlacement.Position) < 0.001f &&
             Vector3.Distance(wp.Rotation, wmoPlacement.Rotation) < 0.001f &&
             (wp.NameId == wmoPlacement.NameId || wp.Name == wmoPlacement.Name))
        );
        
        if (!isDuplicate)
        {
            target.WmoPlacements.Add(wmoPlacement);
        }
    }
    
    // Log the counts after merging to verify data is being merged correctly
    Console.WriteLine($"  After merge - Target: {target.ModelPlacements?.Count ?? 0} models, {target.WmoPlacements?.Count ?? 0} WMOs, {target.TextureReferences?.Count ?? 0} textures");
    Console.WriteLine($"  Added: {(target.ModelReferences?.Count ?? 0) - initialModelCount} model refs, {(target.WmoReferences?.Count ?? 0) - initialWmoCount} WMO refs, {(target.TextureReferences?.Count ?? 0) - initialTextureCount} texture refs");
    Console.WriteLine($"  Added: {(target.ModelPlacements?.Count ?? 0) - initialModelPlacementCount} model placements, {(target.WmoPlacements?.Count ?? 0) - initialWmoPlacementCount} WMO placements");
    
    // Update header information to reflect the merged data
    if (target.Header != null)
    {
        // Update counts in the header to reflect deduplicated data
        target.Header.ModelReferenceCount = target.ModelReferences?.Count ?? 0;
        target.Header.WmoReferenceCount = target.WmoReferences?.Count ?? 0;
        target.Header.ModelPlacementCount = target.ModelPlacements?.Count ?? 0;
        target.Header.WmoPlacementCount = target.WmoPlacements?.Count ?? 0;
        target.Header.TerrainChunkCount = target.TerrainChunks?.Count ?? 0;
        
        // Determine if we have various data types
        bool hasHeightData = target.TerrainChunks?.Any(c => c.Heights != null && c.Heights.Length > 0) ?? false;
        bool hasNormalData = target.TerrainChunks?.Any(c => c.Normals != null && c.Normals.Length > 0) ?? false;
        bool hasLiquidData = target.TerrainChunks?.Any(c => c.LiquidLevel > 0) ?? false;
        bool hasVertexShading = target.TerrainChunks?.Any(c => c.VertexColors != null && c.VertexColors.Count > 0) ?? false;
        
        // Update the Flags value based on the presence of data
        uint newFlags = target.Header.Flags;
        newFlags = hasHeightData ? newFlags | 0x1U : newFlags & ~0x1U;
        newFlags = hasNormalData ? newFlags | 0x2U : newFlags & ~0x2U;
        newFlags = hasLiquidData ? newFlags | 0x4U : newFlags & ~0x4U;
        newFlags = hasVertexShading ? newFlags | 0x8U : newFlags & ~0x8U;
        target.Header.Flags = newFlags;
        
        // Update texture layer count based on the maximum number of texture layers in any chunk
        int maxTextureLayers = target.TerrainChunks?.Count > 0 
            ? target.TerrainChunks.Max(c => c.TextureLayers?.Count ?? 0) 
            : 0;
        target.Header.TextureLayerCount = maxTextureLayers;
    }
    
    // Double-check that all relevant collections have values before saving
    if (target.ModelPlacements?.Count > 0 && target.Header?.ModelPlacementCount == 0)
    {
        target.Header.ModelPlacementCount = target.ModelPlacements.Count;
    }
    
    if (target.WmoPlacements?.Count > 0 && target.Header?.WmoPlacementCount == 0)
    {
        target.Header.WmoPlacementCount = target.WmoPlacements.Count;
    }
    
    if (target.ModelReferences?.Count > 0 && target.Header?.ModelReferenceCount == 0)
    {
        target.Header.ModelReferenceCount = target.ModelReferences.Count;
    }
    
    if (target.WmoReferences?.Count > 0 && target.Header?.WmoReferenceCount == 0)
    {
        target.Header.WmoReferenceCount = target.WmoReferences.Count;
    }
}

// Add a helper method to load a listfile
private static Dictionary<uint, string> LoadListfile(string listfilePath, ILogger logger)
{
    var listfile = new Dictionary<uint, string>();
    try
    {
        logger.LogInformation("Loading listfile from {ListfilePath}", listfilePath);
        var lines = File.ReadAllLines(listfilePath);
        foreach (var line in lines)
        {
            var parts = line.Split(';');
            if (parts.Length >= 2 && uint.TryParse(parts[0], out uint fileId))
            {
                listfile[fileId] = parts[1];
            }
        }
        logger.LogInformation("Loaded {Count} entries from listfile", listfile.Count);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Error loading listfile from {ListfilePath}", listfilePath);
    }
    return listfile;
}
    }
}
