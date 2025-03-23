using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.ADT;
using WCAnalyzer.Core.Services;

namespace WCAnalyzer.CLI
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Setup DI
            var serviceProvider = BuildServiceProvider();
            
            // Create root command
            var rootCommand = new RootCommand("WCAnalyzer - World of Warcraft file analyzer");
            
            // ADT command
            var adtCommand = new Command("adt", "Analyze ADT (terrain) files");
            var fileOption = new Option<FileInfo>(
                aliases: new[] { "--file", "-f" },
                description: "Path to the ADT file"
            );
            fileOption.IsRequired = true;
            
            adtCommand.AddOption(fileOption);
            adtCommand.SetHandler(async (fileInfo) =>
            {
                await AnalyzeADTFile(fileInfo, serviceProvider);
            }, fileOption);
            
            rootCommand.AddCommand(adtCommand);
            
            // Parse command line and execute
            return await rootCommand.InvokeAsync(args);
        }
        
        private static ServiceProvider BuildServiceProvider()
        {
            var services = new ServiceCollection();
            
            // Add logging
            services.AddLogging(builder =>
            {
                builder
                    .AddConsole()
                    .SetMinimumLevel(LogLevel.Information);
            });
            
            // Add services
            services.AddSingleton<FileAnalyzerService>();
            
            return services.BuildServiceProvider();
        }
        
        private static async Task AnalyzeADTFile(FileInfo fileInfo, ServiceProvider serviceProvider)
        {
            var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
            var analyzer = serviceProvider.GetRequiredService<FileAnalyzerService>();
            
            if (!fileInfo.Exists)
            {
                logger.LogError($"File not found: {fileInfo.FullName}");
                return;
            }
            
            logger.LogInformation($"Analyzing ADT file: {fileInfo.FullName}");
            
            ADTFile adtFile = analyzer.AnalyzeADTFile(fileInfo.FullName);
            
            if (adtFile.GetErrors().Count > 0)
            {
                logger.LogWarning($"Found {adtFile.GetErrors().Count} errors while parsing ADT file");
                foreach (var error in adtFile.GetErrors())
                {
                    logger.LogError($"  {error}");
                }
            }
            
            logger.LogInformation($"ADT File Analysis:");
            logger.LogInformation($"  Version: {adtFile.VersionChunk?.Version ?? 0}");
            logger.LogInformation($"  Total Chunks: {adtFile.Chunks.Count}");
            
            // Print chunk list
            logger.LogInformation("Chunks:");
            foreach (var chunk in adtFile.Chunks)
            {
                logger.LogInformation($"  {chunk}");
            }
        }
    }
} 