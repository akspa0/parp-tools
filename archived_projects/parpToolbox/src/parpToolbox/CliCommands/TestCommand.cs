using System;
using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Unified test command for PM4/PD4 validation and regression testing.
    /// </summary>
    public static class TestCommand
    {
        /// <summary>
        /// Executes test validation logic. Returns process exit code (0 = OK, 1 = error).
        /// </summary>
        /// <param name="args">Full command-line args array.</param>
        /// <param name="inputPath">Resolved absolute input file path.</param>
        public static int Run(string[] args, string inputPath)
        {
            // Note: The 'inputPath' argument is for generic tests that might need it.
            // Specific tests like regression or analysis define their own paths internally.
            if (args.Length < 2)
            {
                PrintUsage();
                return 1;
            }

            var testName = args[1];

            try
            {
                switch (testName)
                {
                    case "--pm4-per-object-exporter":
                        Pm4PerObjectExporterTests.RunRegressionTest();
                        break;
                    case "--analyze-chunks":
                        Pm4ChunkAnalysisTests.DumpChunkDataForAnalysis();
                        break;
                    default:
                        ConsoleLogger.WriteLine($"Unknown test: {testName}");
                        PrintUsage();
                        return 1;
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: Test execution failed: {ex.Message}");
                ConsoleLogger.WriteLine(ex.ToString());
                return 1;
            }

            return 0; // Success
        }

        private static void PrintUsage()
        {
            Console.WriteLine("Usage: parpToolbox test <test_name>");
            Console.WriteLine("Available tests:");
            Console.WriteLine("  --pm4-per-object-exporter  - Runs the PM4 per-object exporter regression test.");
            Console.WriteLine("  --analyze-chunks           - Dumps PM4 chunk data to CSV files for analysis.");
        }
    }
}
