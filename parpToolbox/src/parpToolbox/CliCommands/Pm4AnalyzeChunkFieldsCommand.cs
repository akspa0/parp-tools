using System;
using System.IO;
using Microsoft.Extensions.Logging;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    public static class Pm4AnalyzeChunkFieldsCommand
    {
        public static int Run(string inputFile, string outputDirectory = "")
        {
            try
            {
                // Validate input file
                if (!File.Exists(inputFile))
                {
                    ConsoleLogger.WriteLine($"Error: File '{inputFile}' does not exist");
                    return 1;
                }

                // Set up output directory
                if (string.IsNullOrEmpty(outputDirectory))
                {
                    var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                    outputDirectory = Path.Combine("project_output", $"pm4_field_analysis_{timestamp}");
                }

                Directory.CreateDirectory(outputDirectory);
                ConsoleLogger.WriteLine($"Output directory: {outputDirectory}");

                // Create services - use console logger directly since analyzer doesn't need ILogger
                ConsoleLogger.WriteLine($"Starting PM4 chunk field analysis for: {inputFile}");

                // Load PM4 scene
                ConsoleLogger.WriteLine("Loading PM4 scene data...");
                var adapter = new Pm4Adapter();
                var scene = adapter.Load(inputFile);
                
                // Create analyzer with basic logger
                var analyzer = new Pm4ChunkFieldAnalyzer();

                ConsoleLogger.WriteLine("Analyzing chunk fields...");
                analyzer.AnalyzeChunkFields(scene, outputDirectory);

                ConsoleLogger.WriteLine("✓ PM4 chunk field analysis complete!");
                ConsoleLogger.WriteLine($"Analysis results written to: {outputDirectory}");
                
                // List generated files
                var files = Directory.GetFiles(outputDirectory, "*.csv");
                var reports = Directory.GetFiles(outputDirectory, "*.txt");
                
                ConsoleLogger.WriteLine("Generated files:");
                foreach (var file in files)
                {
                    ConsoleLogger.WriteLine($"  📊 {Path.GetFileName(file)}");
                }
                foreach (var report in reports)
                {
                    ConsoleLogger.WriteLine($"  📋 {Path.GetFileName(report)}");
                }

                return 0;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error: {ex.Message}");
                ConsoleLogger.WriteLine(ex.StackTrace ?? "No stack trace available");
                return 1;
            }
        }
    }
}
