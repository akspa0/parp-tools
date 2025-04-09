// Program.cs in WoWToolbox.AnalysisTool
using System;
using System.IO;
using WoWToolbox.Core; // Use the core project namespace
using WoWToolbox.AnalysisTool; // Add namespace for MslkAnalyzer

public class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("WoWToolbox MSLK Analysis Tool");
        Console.WriteLine("=============================");

        string? skippedLog = null;
        string? debugLog = null;

        // Basic argument parsing
        if (args.Length >= 2)
        {
            skippedLog = args[0];
            debugLog = args[1];
            Console.WriteLine("Using paths from command line arguments.");
        }
        else if (args.Length == 1 && (args[0] == "-h" || args[0] == "--help"))
        {
            Console.WriteLine("Usage: dotnet run --project <csproj_path> -- <skipped_log_path> <debug_log_path>");
            Console.WriteLine("Example: dotnet run --project src/WoWToolbox.AnalysisTool/WoWToolbox.AnalysisTool.csproj -- test_data/pm4/dev_00_00_skipped.log test_data/pm4/dev_00_00.debug.log");
             Console.WriteLine("If no arguments are provided, it defaults to hardcoded PM4 paths.");
            return;
        }
        else
        {
             Console.WriteLine("No paths provided via arguments, using default hardcoded paths...");
            // --- Default Hardcoded Paths (Fallback) ---
            string defaultTestOutputBasePath = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\test_data\development"; 
            skippedLog = Path.Combine(defaultTestOutputBasePath, "development_00_00_skipped_mslk.log");
            debugLog = Path.Combine(defaultTestOutputBasePath, "development_00_00.debug.log");
             // --- End Default Paths ---
        }

        Console.WriteLine($"Using Skipped Log: {skippedLog}");
        Console.WriteLine($"Using Debug Log:   {debugLog}");

        // Check if log files exist before running analysis
        if (string.IsNullOrEmpty(skippedLog) || !System.IO.File.Exists(skippedLog)) 
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\nError: Skipped log not found or path invalid!");
            Console.WriteLine($"Path: {skippedLog ?? "(null)"}");
            Console.WriteLine("Please ensure the path is correct and the file exists.");
            Console.ResetColor();
            Console.WriteLine("\nPress Enter to exit.");
            Console.ReadLine();
            return;
        }
         if (string.IsNullOrEmpty(debugLog) || !System.IO.File.Exists(debugLog)) 
         {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\nError: Debug log not found or path invalid!");
             Console.WriteLine($"Path: {debugLog ?? "(null)"}");
            Console.WriteLine("Please ensure the path is correct and the file exists.");
            Console.ResetColor();
            Console.WriteLine("\nPress Enter to exit.");
            Console.ReadLine();
            return;
        }

        Console.WriteLine("\nStarting analysis...");
        // Call the analyzer from the (now local) MslkAnalyzer class
        try
        {
            // Define output log path based on input base name
            string baseName = Path.GetFileNameWithoutExtension(skippedLog).Replace("_skipped_mslk", ""); // e.g., development_00_00
            string outputLogPath = Path.Combine(Path.GetDirectoryName(skippedLog) ?? ".", $"{baseName}_mslk_analysis.log");

            MslkAnalyzer.AnalyzeMslkData(skippedLog, debugLog, outputLogPath); // Pass output path
        }
        catch (Exception ex)
        {
             Console.ForegroundColor = ConsoleColor.Red;
             Console.WriteLine($"\nAn error occurred during analysis: {ex.Message}");
             Console.WriteLine(ex.StackTrace); // Optional: include stack trace for debugging
             Console.ResetColor();
        }
       

        Console.WriteLine("\nAnalysis attempt complete. Press Enter to exit.");
        Console.ReadLine();
    }
}