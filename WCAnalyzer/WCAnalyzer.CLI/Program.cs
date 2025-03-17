using System;
using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Help;
using System.CommandLine.Parsing;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Services;
using WCAnalyzer.UniqueIdAnalysis;

namespace WCAnalyzer.CLI
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Load configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .Build();

            // Configure services
            var serviceProvider = ConfigureServices(configuration);
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();

            try
            {
                logger.LogInformation("Starting WCAnalyzer CLI - Version {Version}", 
                    typeof(Program).Assembly.GetName().Version);

                // Parse command line arguments and execute commands
                var parser = BuildCommandLine(serviceProvider);
                return await parser.InvokeAsync(args);
            }
            catch (Exception ex)
            {
                logger.LogCritical(ex, "An unhandled exception occurred");
                Console.Error.WriteLine($"Error: {ex.Message}");
                return 1;
            }
            finally
            {
                // Ensure all logs are flushed
                (serviceProvider as ServiceProvider)?.Dispose();
            }
        }

        // Configure the dependency injection container
        private static IServiceProvider ConfigureServices(IConfiguration configuration)
        {
            var services = new ServiceCollection();

            // Add logging
            services.AddLogging(builder =>
            {
                builder.AddConfiguration(configuration.GetSection("Logging"));
                builder.AddConsole();
                builder.AddDebug();
                builder.AddTraceSource(new System.Diagnostics.SourceSwitch("WCAnalyzer", "Verbose"));
            });

            // Add services
            services.AddSingleton(configuration);
            services.AddTransient<UniqueIdAnalyzer>();
            
            // Add PM4 services
            services.AddTransient<PM4Parser>();
            services.AddTransient<PM4CsvGenerator>();
            services.AddTransient<PM4ObjExporter>();
            
            // Add PD4 services
            services.AddTransient<PD4Parser>();
            services.AddTransient<PD4CsvGenerator>();
            
            // Add ADT services
            services.AddTransient<AdtParser>();
            
            return services.BuildServiceProvider();
        }

        // Configure the command line interface
        private static Parser BuildCommandLine(IServiceProvider serviceProvider)
        {
            // Create root command
            var rootCommand = new RootCommand("WCAnalyzer CLI tool for analyzing World of Warcraft game data files");
            
            // Add verbose option
            var verboseOption = new Option<bool>(new[] { "-v", "--verbose" }, "Enable verbose output");
            rootCommand.AddGlobalOption(verboseOption);
            
            // Add uniqueid command
            var uniqueIdCommand = BuildUniqueIdCommand(serviceProvider);
            rootCommand.AddCommand(uniqueIdCommand);
            
            // Add pm4 command
            var pm4Command = BuildPM4Command(serviceProvider);
            rootCommand.AddCommand(pm4Command);
            
            // Add pd4 command
            var pd4Command = BuildPD4Command(serviceProvider);
            rootCommand.AddCommand(pd4Command);
            
            // Add adt command
            var adtCommand = BuildAdtCommand(serviceProvider);
            rootCommand.AddCommand(adtCommand);
            
            // Add more commands here
            
            // Handle the default case
            rootCommand.SetHandler(() =>
            {
                // Show help if no commands were specified
                Console.WriteLine(rootCommand.Description);
            });

            // Build the parser
            return new CommandLineBuilder(rootCommand)
                .UseDefaults()
                .Build();
        }

        // Configure the uniqueid command
        private static Command BuildUniqueIdCommand(IServiceProvider serviceProvider)
        {
            var uniqueIdCommand = new Command("uniqueid", "Analyze unique IDs in game data files");
            
            var inputOption = new Option<FileInfo>(
                new[] { "-i", "--input" },
                "Input file to analyze"
            )
            { 
                IsRequired = true 
            };
            uniqueIdCommand.AddOption(inputOption);
            
            // Add output directory option
            var outputOption = new Option<DirectoryInfo>(
                new[] { "-o", "--output" },
                () => new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "output")),
                "Output directory for results"
            );
            uniqueIdCommand.AddOption(outputOption);
            
            // Add verbose option
            var verboseOption = new Option<bool>(
                new[] { "-v", "--verbose" },
                "Enable verbose output"
            );
            uniqueIdCommand.AddOption(verboseOption);
            
            uniqueIdCommand.SetHandler(async (FileInfo input, DirectoryInfo output, bool verbose) =>
            {
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                
                if (verbose)
                {
                    logger.LogInformation("Analyzing file: {InputFile}", input.FullName);
                    logger.LogInformation("Output directory: {OutputDir}", output.FullName);
                }
                
                try
                {
                    // Create a new UniqueIdAnalyzer with the correct parameters
                    var uniqueIdAnalyzer = new UniqueIdAnalyzer(
                        input.DirectoryName ?? Environment.CurrentDirectory,
                        output.FullName,
                        serviceProvider.GetRequiredService<ILogger<UniqueIdAnalyzer>>()
                    );
                    
                    // Run the analysis
                    var result = await uniqueIdAnalyzer.AnalyzeAsync();
                    
                    // Generate reports
                    await uniqueIdAnalyzer.GenerateReportsAsync(result, output.FullName);
                    
                    logger.LogInformation("Analysis completed successfully");
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error analyzing file {InputFile}", input.FullName);
                    throw;
                }
            }, inputOption, outputOption, verboseOption);
            
            return uniqueIdCommand;
        }

        // Configure the pm4 command
        private static Command BuildPM4Command(IServiceProvider serviceProvider)
        {
            var pm4Command = new Command("pm4", "Work with PM4 files");
            
            // Add analyze subcommand
            var analyzeCommand = new Command("analyze", "Analyze PM4 files");
            var analyzeInputOption = new Option<FileInfo[]>(
                new[] { "-i", "--input" },
                "Input PM4 files to analyze"
            )
            { 
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };
            analyzeCommand.AddOption(analyzeInputOption);
            
            // Add output directory option
            var outputOption = new Option<DirectoryInfo>(
                new[] { "-o", "--output" },
                () => new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "output")),
                "Output directory for results"
            );
            analyzeCommand.AddOption(outputOption);
            
            // Add verbose option
            var verboseOption = new Option<bool>(
                new[] { "-v", "--verbose" },
                "Enable verbose output"
            );
            analyzeCommand.AddOption(verboseOption);
            
            analyzeCommand.SetHandler(async (FileInfo[] inputs, DirectoryInfo output, bool verbose) =>
            {
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                
                if (verbose)
                {
                    logger.LogInformation("Analyzing {Count} PM4 files", inputs.Length);
                    logger.LogInformation("Output directory: {OutputDir}", output.FullName);
                }
                
                try
                {
                    // Get the services
                    var pm4Parser = serviceProvider.GetRequiredService<PM4Parser>();
                    var csvGenerator = serviceProvider.GetRequiredService<PM4CsvGenerator>();
                    
                    // Ensure output directory exists
                    output.Create();
                    
                    // Process each input file
                    int successCount = 0;
                    int failCount = 0;
                    
                    foreach (var input in inputs)
                    {
                        try
                        {
                            logger.LogInformation("Processing file: {InputFile}", input.FullName);
                            
                            // Parse the PM4 file
                            var result = pm4Parser.ParseFile(input.FullName);
                            
                            // Generate CSV reports with the specified output directory
                            await csvGenerator.GenerateReportsAsync(result, output.FullName);
                            
                            if (result.HasErrors)
                            {
                                logger.LogWarning("Processed file with errors: {InputFile}", input.FullName);
                                failCount++;
                            }
                            else
                            {
                                logger.LogInformation("Successfully processed: {InputFile}", input.FullName);
                                successCount++;
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.LogError(ex, "Error analyzing PM4 file {InputFile}", input.FullName);
                            failCount++;
                        }
                    }
                    
                    logger.LogInformation("PM4 analysis completed. Success: {SuccessCount}, Failed: {FailCount}", 
                        successCount, failCount);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error in PM4 analyze command");
                    throw;
                }
            }, analyzeInputOption, outputOption, verboseOption);
            
            pm4Command.AddCommand(analyzeCommand);
            
            // Add extract-terrain subcommand
            var extractTerrainCommand = new Command("extract-terrain", "Extract terrain data from PM4 files");
            var terrainInputOption = new Option<FileInfo[]>(
                new[] { "-i", "--input" },
                "Input PM4 files to extract terrain from"
            )
            { 
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };
            extractTerrainCommand.AddOption(terrainInputOption);
            
            extractTerrainCommand.SetHandler(async (FileInfo[] inputs, DirectoryInfo output, bool verbose) =>
            {
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                
                if (verbose)
                {
                    logger.LogInformation("Extracting terrain from {Count} PM4 files", inputs.Length);
                    logger.LogInformation("Output directory: {OutputDir}", output.FullName);
                }
                
                try
                {
                    // Log that terrain extraction is not implemented
                    logger.LogInformation("Terrain extraction not implemented. Would process {Count} files", 
                        inputs.Length);
                    
                    foreach (var input in inputs)
                    {
                        logger.LogInformation("Would extract terrain from: {InputFile}", input.FullName);
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error extracting terrain data");
                    throw;
                }
            }, terrainInputOption, outputOption, verboseOption);
            
            pm4Command.AddCommand(extractTerrainCommand);
            
            // Add export-obj subcommand
            var exportObjCommand = new Command("export-obj", "Export PM4 files to OBJ 3D model format");
            var objInputOption = new Option<FileInfo[]>(
                new[] { "-i", "--input" },
                "Input PM4 files to export to OBJ"
            )
            { 
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };
            exportObjCommand.AddOption(objInputOption);
            
            // Add output directory option
            exportObjCommand.AddOption(outputOption);
            
            // Add verbose option
            exportObjCommand.AddOption(verboseOption);
            
            exportObjCommand.SetHandler(async (FileInfo[] inputs, DirectoryInfo output, bool verbose) =>
            {
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                
                if (verbose)
                {
                    logger.LogInformation("Exporting {Count} PM4 files to OBJ format", inputs.Length);
                    logger.LogInformation("Output directory: {OutputDir}", output.FullName);
                }
                
                try
                {
                    // Create output directory if it doesn't exist
                    output.Create();
                    
                    // Get the PM4Parser service
                    var pm4Parser = serviceProvider.GetRequiredService<PM4Parser>();
                    
                    // Create an OBJ exporter
                    var objExporter = new PM4ObjExporter();
                    
                    // Process each input file
                    int successCount = 0;
                    int failCount = 0;
                    
                    foreach (var input in inputs)
                    {
                        try
                        {
                            logger.LogInformation("Processing file: {InputFile}", input.FullName);
                            
                            // Parse the PM4 file
                            var result = pm4Parser.ParseFile(input.FullName);
                            
                            if (result.HasVertexPositions || result.HasVertexData)
                            {
                                // Export to OBJ format using the new method that creates separate files
                                await objExporter.ExportToDirectoryAsync(result, output.FullName);
                                
                                logger.LogInformation("Exported to OBJ files in: {OutputDirectory}", output.FullName);
                                successCount++;
                            }
                            else
                            {
                                logger.LogWarning("No vertex data in file {InputFile}, skipping", input.FullName);
                                failCount++;
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.LogError(ex, "Error exporting PM4 file to OBJ: {InputFile}", input.FullName);
                            failCount++;
                        }
                    }
                    
                    logger.LogInformation("OBJ export completed. Success: {SuccessCount}, Failed: {FailCount}", 
                        successCount, failCount);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error in PM4 export-obj command");
                    throw;
                }
            }, objInputOption, outputOption, verboseOption);
            
            pm4Command.AddCommand(exportObjCommand);
            
            return pm4Command;
        }

        // Configure the pd4 command
        private static Command BuildPD4Command(IServiceProvider serviceProvider)
        {
            var pd4Command = new Command("pd4", "Work with PD4 files");
            
            // Add analyze subcommand
            var analyzeCommand = new Command("analyze", "Analyze PD4 files");
            var analyzeInputOption = new Option<FileInfo[]>(
                new[] { "-i", "--input" },
                "Input PD4 files to analyze"
            )
            { 
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };
            analyzeCommand.AddOption(analyzeInputOption);
            
            // Add output directory option
            var outputOption = new Option<DirectoryInfo>(
                new[] { "-o", "--output" },
                () => new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "output")),
                "Output directory for results"
            );
            analyzeCommand.AddOption(outputOption);
            
            // Add verbose option
            var verboseOption = new Option<bool>(
                new[] { "-v", "--verbose" },
                "Enable verbose output"
            );
            analyzeCommand.AddOption(verboseOption);
            
            analyzeCommand.SetHandler(async (FileInfo[] inputs, DirectoryInfo output, bool verbose) =>
            {
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                
                if (verbose)
                {
                    logger.LogInformation("Analyzing {Count} PD4 files", inputs.Length);
                    logger.LogInformation("Output directory: {OutputDir}", output.FullName);
                }
                
                try
                {
                    // Get the services
                    var pd4Parser = serviceProvider.GetRequiredService<PD4Parser>();
                    var csvGenerator = serviceProvider.GetRequiredService<PD4CsvGenerator>();
                    
                    // Ensure output directory exists
                    output.Create();
                    
                    // Process each input file
                    int successCount = 0;
                    int failCount = 0;
                    
                    foreach (var input in inputs)
                    {
                        try
                        {
                            logger.LogInformation("Processing file: {InputFile}", input.FullName);
                            
                            // Parse the PD4 file
                            var result = pd4Parser.ParseFile(input.FullName);
                            
                            // Generate CSV reports with the specified output directory
                            await csvGenerator.GenerateAllCsvReportsAsync(result, output.FullName);
                            
                            if (result.HasErrors)
                            {
                                logger.LogWarning("Processed file with errors: {InputFile}", input.FullName);
                                failCount++;
                            }
                            else
                            {
                                logger.LogInformation("Successfully processed: {InputFile}", input.FullName);
                                successCount++;
                            }
                        }
                        catch (Exception ex)
                        {
                            logger.LogError(ex, "Error analyzing PD4 file {InputFile}", input.FullName);
                            failCount++;
                        }
                    }
                    
                    logger.LogInformation("PD4 analysis completed. Success: {SuccessCount}, Failed: {FailCount}", 
                        successCount, failCount);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error in PD4 analyze command");
                    throw;
                }
            }, analyzeInputOption, outputOption, verboseOption);
            
            pd4Command.AddCommand(analyzeCommand);
            
            return pd4Command;
        }

        // Configure the adt command
        private static Command BuildAdtCommand(IServiceProvider serviceProvider)
        {
            var adtCommand = new Command("adt", "Work with ADT (terrain) files");
            
            // Add analyze subcommand
            var analyzeCommand = new Command("analyze", "Analyze ADT files");
            var analyzeInputOption = new Option<FileInfo[]>(
                new[] { "-i", "--input" },
                "Input ADT files to analyze"
            )
            { 
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };
            analyzeCommand.AddOption(analyzeInputOption);
            
            // Add output directory option
            var outputOption = new Option<DirectoryInfo>(
                new[] { "-o", "--output" },
                () => new DirectoryInfo(Path.Combine(Environment.CurrentDirectory, "output")),
                "Output directory for results"
            );
            analyzeCommand.AddOption(outputOption);
            
            // Add verbose option
            var verboseOption = new Option<bool>(
                new[] { "-v", "--verbose" },
                "Enable verbose output"
            );
            analyzeCommand.AddOption(verboseOption);
            
            analyzeCommand.SetHandler(async (FileInfo[] inputs, DirectoryInfo output, bool verbose) =>
            {
                var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
                
                if (verbose)
                {
                    logger.LogInformation("Analyzing {Count} ADT files", inputs.Length);
                    logger.LogInformation("Output directory: {OutputDir}", output.FullName);
                }
                
                try
                {
                    // Get the AdtParser service from the dependency injection container
                    var adtParser = serviceProvider.GetRequiredService<AdtParser>();
                    
                    // Process each input file
                    int successCount = 0;
                    int failCount = 0;
                    
                    foreach (var input in inputs)
                    {
                        try
                        {
                            logger.LogInformation("Processing file: {InputFile}", input.FullName);
                            
                            // Parse the ADT file
                            var result = await adtParser.ParseAsync(input.FullName);
                            
                            // Write output as JSON
                            var jsonOptions = new System.Text.Json.JsonSerializerOptions
                            {
                                WriteIndented = true
                            };
                            
                            // Create output directory if it doesn't exist
                            output.Create();
                            
                            // Write JSON file
                            string outputFile = Path.Combine(output.FullName, Path.GetFileNameWithoutExtension(input.Name) + ".json");
                            await File.WriteAllTextAsync(outputFile, System.Text.Json.JsonSerializer.Serialize(result, jsonOptions));
                            
                            logger.LogInformation("Successfully processed: {InputFile}", input.FullName);
                            logger.LogInformation("Output written to: {OutputFile}", outputFile);
                            successCount++;
                        }
                        catch (Exception ex)
                        {
                            logger.LogError(ex, "Error analyzing ADT file {InputFile}", input.FullName);
                            failCount++;
                        }
                    }
                    
                    logger.LogInformation("ADT analysis completed. Success: {SuccessCount}, Failed: {FailCount}", 
                        successCount, failCount);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error in ADT analyze command");
                    throw;
                }
            }, analyzeInputOption, outputOption, verboseOption);
            
            adtCommand.AddCommand(analyzeCommand);
            
            return adtCommand;
        }
    }
}
