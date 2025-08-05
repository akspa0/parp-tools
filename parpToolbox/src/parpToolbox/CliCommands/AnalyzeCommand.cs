using System;
using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;
using System.Text.Json;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Unified PM4 analysis CLI command handler.
    /// Extracted from Program.cs to reduce main entry complexity.
    /// </summary>
    public static class AnalyzeCommand
    {
        /// <summary>
        /// Executes analysis logic. Returns process exit code (0 = OK, 1 = error).
        /// </summary>
        /// <param name="args">Full command-line args array.</param>
        /// <param name="inputPath">Resolved absolute input file path.</param>
        public static int Run(string[] args, string inputPath)
        {
            try
            {
                bool useSingleTile = args != null && Array.Exists(args, a => a.Equals("--single-tile", StringComparison.OrdinalIgnoreCase));
                bool generateJsonReport = args != null && (Array.Exists(args, a => a.Equals("--json-report", StringComparison.OrdinalIgnoreCase)) || Array.Exists(args, a => a.Equals("--report-json", StringComparison.OrdinalIgnoreCase)) || Array.Exists(args, a => a.Equals("--report", StringComparison.OrdinalIgnoreCase)));

                // Load scene – default region loader gives more complete data.
                Pm4Scene scene;
                if (useSingleTile)
                {
                    ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
                    var loader = new Pm4Adapter();
                    scene = loader.Load(inputPath, new Pm4LoadOptions
                    {
                        VerboseLogging = true,
                        ValidateData = true,
                        AnalyzeIndexPatterns = true
                    });
                }
                else
                {
                    ConsoleLogger.WriteLine("Region mode active (default) – loading cross-tile references…");
                    var adapter = new Pm4Adapter();
                    scene = adapter.LoadRegion(inputPath);
                }

                var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputPath) + "_analysis");
                var analysisAdapter = new Pm4Adapter();
                var analysisOpts = new Pm4AnalysisOptions
                {
                    AnalyzeDataStructure = true,
                    AnalyzeIndexPatterns = true,
                    AnalyzeUnknownFields = true,
                    AnalyzeChunkRelationships = true,
                    GenerateCsvReports = false, // CSV reports deprecated
                    OutputDirectory = outputDir,
                    VerboseLogging = true
                };

                var report = analysisAdapter.Analyze(scene, analysisOpts);
                ConsoleLogger.WriteLine(report.Summary);

                 if (generateJsonReport)
                 {
                     var jsonPath = Path.Combine(outputDir, "analysis_report.json");
                     File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
                     ConsoleLogger.WriteLine($"JSON report written to {jsonPath}");
                 }
                ConsoleLogger.WriteLine("PM4 comprehensive analysis complete!");
                return 0;
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error during analysis: {ex.Message}\n{ex.StackTrace}");
                return 1;
            }
        }
    }
}
