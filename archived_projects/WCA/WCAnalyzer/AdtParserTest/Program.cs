using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Services;
using WCAnalyzer.Core.Utilities;

namespace AdtParserTest
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Create a root command
            var rootCommand = new RootCommand("ADT Parser Test Application");

            // Add options
            var fileOption = new Option<string>(
                aliases: new[] { "-f", "--file" },
                description: "The path to a single ADT file to parse");

            var directoryOption = new Option<string>(
                aliases: new[] { "-d", "--directory" },
                description: "Directory containing ADT files to parse");

            var listfileOption = new Option<string>(
                aliases: new[] { "-l", "--listfile" },
                description: "Optional path to a listfile for reference validation");

            var outputOption = new Option<string>(
                aliases: new[] { "-o", "--output" },
                description: "Output directory for reports")
            {
                IsRequired = true
            };

            var verboseOption = new Option<bool>(
                aliases: new[] { "-v", "--verbose" },
                description: "Enable verbose logging");

            // Add options to the command
            rootCommand.AddOption(fileOption);
            rootCommand.AddOption(directoryOption);
            rootCommand.AddOption(listfileOption);
            rootCommand.AddOption(outputOption);
            rootCommand.AddOption(verboseOption);

            // Setup the handler
            rootCommand.SetHandler(async (file, directory, listfile, output, verbose) =>
            {
                await RunAnalysis(file, directory, listfile, output, verbose);
            }, fileOption, directoryOption, listfileOption, outputOption, verboseOption);

            return await rootCommand.InvokeAsync(args);
        }

        private static async Task RunAnalysis(string filePath, string directoryPath, string listfilePath, string outputPath, bool verbose)
        {
            Console.WriteLine("ADT Parser Test Application");
            Console.WriteLine("---------------------------");

            // Validate input
            if (string.IsNullOrEmpty(filePath) && string.IsNullOrEmpty(directoryPath))
            {
                Console.WriteLine("Error: You must specify either a file or directory to parse.");
                return;
            }

            if (!string.IsNullOrEmpty(filePath) && !File.Exists(filePath))
            {
                Console.WriteLine($"Error: File not found: {filePath}");
                return;
            }

            if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
            {
                Console.WriteLine($"Error: Directory not found: {directoryPath}");
                return;
            }

            // Create output directory if it doesn't exist
            if (!Directory.Exists(outputPath))
            {
                Directory.CreateDirectory(outputPath);
            }

            // Setup DI
            var serviceProvider = ConfigureServices(verbose);

            try
            {
                // Get services
                var parser = serviceProvider.GetRequiredService<AdtParser>();
                var validator = serviceProvider.GetRequiredService<ReferenceValidator>();
                var reportGenerator = serviceProvider.GetRequiredService<ReportGenerator>();
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();

                // Load listfile if provided
                HashSet<string> knownGoodFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                if (!string.IsNullOrEmpty(listfilePath))
                {
                    logger.LogInformation("Loading listfile: {ListfilePath}", listfilePath);
                    knownGoodFiles = await validator.LoadListfileAsync(listfilePath);
                }

                // Initialize summary
                var modelsSummary = new WCAnalyzer.Core.Models.AnalysisSummary
                {
                    StartTime = DateTime.Now
                };

                // Create a Services.AnalysisSummary for report generation
                var servicesSummary = new WCAnalyzer.Core.Services.AnalysisSummary
                {
                    StartTime = modelsSummary.StartTime
                };

                // Parse files
                var results = new List<AdtAnalysisResult>();

                if (!string.IsNullOrEmpty(filePath))
                {
                    // Parse a single file
                    logger.LogInformation("Parsing file: {FilePath}", filePath);
                    modelsSummary.TotalFiles = 1;
                    servicesSummary.TotalFiles = 1;

                    try
                    {
                        var result = await parser.ParseAsync(filePath);
                        if (result != null)
                        {
                            if (!string.IsNullOrEmpty(listfilePath))
                            {
                                validator.ValidateReferences(result, knownGoodFiles);
                            }
                            
                            // Log model and WMO placement counts for diagnostic purposes
                            logger.LogInformation("File {FileName} contains: {ModelCount} model references, {WmoCount} WMO references, {ModelPlacementCount} model placements, {WmoPlacementCount} WMO placements, {UniqueIdCount} unique IDs",
                                result.FileName,
                                result.ModelReferences.Count,
                                result.WmoReferences.Count,
                                result.ModelPlacements.Count,
                                result.WmoPlacements.Count,
                                result.UniqueIds.Count);
                            
                            // Display some sample model placements if available
                            if (result.ModelPlacements.Count > 0)
                            {
                                var sample = result.ModelPlacements.First();
                                logger.LogInformation("Sample model placement - UniqueId: {UniqueId}, Name: {Name}, Position: ({X}, {Y}, {Z})",
                                    sample.UniqueId, sample.Name, sample.Position.X, sample.Position.Y, sample.Position.Z);
                            }
                            
                            // Display some sample WMO placements if available
                            if (result.WmoPlacements.Count > 0)
                            {
                                var sample = result.WmoPlacements.First();
                                logger.LogInformation("Sample WMO placement - UniqueId: {UniqueId}, Name: {Name}, Position: ({X}, {Y}, {Z})",
                                    sample.UniqueId, sample.Name, sample.Position.X, sample.Position.Y, sample.Position.Z);
                            }
                            
                            results.Add(result);
                            modelsSummary.ProcessedFiles++;
                            servicesSummary.ProcessedFiles++;
                        }
                    }
                    catch (Exception ex)
                    {
                        logger.LogError(ex, "Error parsing file: {FilePath}", filePath);
                        modelsSummary.FailedFiles++;
                        servicesSummary.FailedFiles++;
                    }
                }
                else
                {
                    // Parse all ADT files in the directory
                    var adtFiles = Directory.GetFiles(directoryPath, "*.adt", SearchOption.AllDirectories);
                    modelsSummary.TotalFiles = adtFiles.Length;
                    servicesSummary.TotalFiles = adtFiles.Length;
                    logger.LogInformation("Found {AdtFileCount} ADT files in directory: {DirectoryPath}", adtFiles.Length, directoryPath);

                    foreach (var file in adtFiles)
                    {
                        logger.LogInformation("Parsing file: {FilePath}", file);
                        try
                        {
                            var result = await parser.ParseAsync(file);
                            if (result != null)
                            {
                                if (!string.IsNullOrEmpty(listfilePath))
                                {
                                    validator.ValidateReferences(result, knownGoodFiles);
                                }
                                
                                // Log model and WMO placement counts for diagnostic purposes
                                logger.LogInformation("File {FileName} contains: {ModelCount} model references, {WmoCount} WMO references, {ModelPlacementCount} model placements, {WmoPlacementCount} WMO placements, {UniqueIdCount} unique IDs",
                                    result.FileName,
                                    result.ModelReferences.Count,
                                    result.WmoReferences.Count,
                                    result.ModelPlacements.Count,
                                    result.WmoPlacements.Count,
                                    result.UniqueIds.Count);
                                
                                // Display some sample model placements if available
                                if (result.ModelPlacements.Count > 0)
                                {
                                    var sample = result.ModelPlacements.First();
                                    logger.LogInformation("Sample model placement - UniqueId: {UniqueId}, Name: {Name}, Position: ({X}, {Y}, {Z})",
                                        sample.UniqueId, sample.Name, sample.Position.X, sample.Position.Y, sample.Position.Z);
                                }
                                
                                // Display some sample WMO placements if available
                                if (result.WmoPlacements.Count > 0)
                                {
                                    var sample = result.WmoPlacements.First();
                                    logger.LogInformation("Sample WMO placement - UniqueId: {UniqueId}, Name: {Name}, Position: ({X}, {Y}, {Z})",
                                        sample.UniqueId, sample.Name, sample.Position.X, sample.Position.Y, sample.Position.Z);
                                }
                                
                                results.Add(result);
                                modelsSummary.ProcessedFiles++;
                                servicesSummary.ProcessedFiles++;
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.LogError(ex, "Error parsing file: {FilePath}", file);
                            modelsSummary.FailedFiles++;
                            servicesSummary.FailedFiles++;
                        }
                    }
                }

                // Finalize summary
                var endTime = DateTime.Now;
                modelsSummary.EndTime = endTime;
                servicesSummary.EndTime = endTime;

                // Update summary statistics
                foreach (var result in results)
                {
                    modelsSummary.TotalTextureReferences += result.TextureReferences.Count;
                    modelsSummary.TotalModelReferences += result.ModelReferences.Count;
                    modelsSummary.TotalWmoReferences += result.WmoReferences.Count;
                    modelsSummary.TotalModelPlacements += result.ModelPlacements.Count;
                    modelsSummary.TotalWmoPlacements += result.WmoPlacements.Count;
                    modelsSummary.TotalTerrainChunks += result.TerrainChunks.Count;
                    
                    servicesSummary.TotalTextureReferences += result.TextureReferences.Count;
                    servicesSummary.TotalModelReferences += result.ModelReferences.Count;
                    servicesSummary.TotalWmoReferences += result.WmoReferences.Count;
                    servicesSummary.TotalModelPlacements += result.ModelPlacements.Count;
                    servicesSummary.TotalWmoPlacements += result.WmoPlacements.Count;
                    servicesSummary.TotalTerrainChunks += result.TerrainChunks.Count;
                    
                    // Track unique IDs
                    foreach (var id in result.UniqueIds)
                    {
                        if (id > modelsSummary.MaxUniqueId)
                        {
                            modelsSummary.MaxUniqueId = id;
                        }
                        
                        if (id > servicesSummary.MaxUniqueId)
                        {
                            servicesSummary.MaxUniqueId = id;
                        }
                    }
                }

                // Generate reports
                logger.LogInformation("Generating reports in directory: {OutputPath}", outputPath);
                await reportGenerator.GenerateAllReportsAsync(results, servicesSummary, outputPath);

                logger.LogInformation("Analysis complete. Processed {ProcessedFiles} files with {FailedFiles} failures in {Duration:F2} seconds",
                    modelsSummary.ProcessedFiles, modelsSummary.FailedFiles, modelsSummary.Duration.TotalSeconds);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during analysis: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine(ex.StackTrace);
                }
            }
            finally
            {
                // Dispose services
                if (serviceProvider is IDisposable disposable)
                {
                    disposable.Dispose();
                }
            }
        }

        private static ServiceProvider ConfigureServices(bool verbose)
        {
            var services = new ServiceCollection();

            // Add logging
            services.AddLogging(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(verbose ? Microsoft.Extensions.Logging.LogLevel.Debug : Microsoft.Extensions.Logging.LogLevel.Information);
            });

            // Register services
            services.AddSingleton<ILoggingService, LoggingService>();
            services.AddSingleton<AdtParser>();
            services.AddSingleton<ReferenceValidator>();
            services.AddSingleton<TerrainDataCsvGenerator>();
            services.AddSingleton<JsonReportGenerator>();
            services.AddSingleton<MarkdownReportGenerator>();
            services.AddSingleton<ReportGenerator>();

            return services.BuildServiceProvider();
        }
    }
} 