// Program.cs in WoWToolbox.AnalysisTool
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core; // Use the core project namespace
using WoWToolbox.Core.ADT; // Added for AdtService, Placement
using WoWToolbox.Core.Navigation.PM4; // Added for PM4File
using WoWToolbox.Core.Navigation.PM4.Chunks; // Added for MDSFChunk
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using WoWToolbox.AnalysisTool; // Added for MslkAnalyzer
using Warcraft.NET.Files.ADT.TerrainObject.Zero; // Added for TerrainObjectZero
using Warcraft.NET.Files.ADT.Terrain.BfA; // ADDED: For modern base ADT
using System.Text; // For StringBuilder in summary report

// ADDED: Data structure for consolidated output
public class Pm4AnalysisResult
{
    public string Pm4Name { get; set; } = ""; // PM4 base name (e.g., development_XX_YY)
    public List<uint> Pm4UniqueIds { get; set; } = new List<uint>(); // IDs from PM4
    public List<uint>? AdtUniqueIds { get; set; } = null; // IDs from _obj0.adt (null if N/A)
    public List<Placement>? CorrelatedPlacements { get; set; } = null; // Correlated ADT Placements (null if N/A or none found)
    public string? ProcessingError { get; set; } = null; // Record specific file errors
}

public class Program
{
    // Dictionary to hold listfile data (FileDataId -> FilePath)
    private static Dictionary<uint, string> _listfileData = new Dictionary<uint, string>();

    public static void Main(string[] args)
    {
        Console.WriteLine("WoWToolbox Analysis Tool - Directory Mode"); // Updated Title
        Console.WriteLine("===========================================");

        // --- Argument Parsing ---
        string inputDirectory = @"I:\parp-scripts\WoWToolbox_v3\original_development\development"; // Default input
        string outputDirectory = Path.Combine(Directory.GetCurrentDirectory(), "analysis_output"); // Default output
        string? mslkDebugLogPath = null;
        string? mslkSkippedLogPath = null;
        bool runMslkAnalysis = false;
        bool runHypothesisAnalysis = false; // ADDED: Flag for new analysis mode

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-d":
                case "--directory":
                    if (i + 1 < args.Length)
                    {
                        inputDirectory = args[++i];
                    }
                    else
                    {
                        WriteError("Missing value for input directory argument (-d/--directory).");
                        return;
                    }
                    break;
                case "-o":
                case "--output":
                    if (i + 1 < args.Length)
                    {
                        outputDirectory = args[++i];
                    }
                    else
                    {
                        WriteError("Missing value for output directory argument (-o/--output).");
                        return;
                    }
                    break;
                case "--analyze-mslk":
                    runMslkAnalysis = true;
                    if (i + 1 < args.Length)
                    {
                        mslkDebugLogPath = args[++i];
                        if (i + 1 < args.Length && !args[i + 1].StartsWith("-")) // Check if next arg is not another option
                        {
                            mslkSkippedLogPath = args[++i];
                        }
                    }
                    else
                    {
                        WriteError("Missing debug log path for --analyze-mslk.");
                        return;
                    }
                    break;
                case "--analyze-hypotheses": // ADDED: New argument
                    runHypothesisAnalysis = true;
                    break;
                case "-h":
                case "--help":
                    PrintHelp();
                    return;
                default:
                    WriteError($"Unknown argument: {args[i]}");
                    PrintHelp();
                    return;
            }
        }

        // --- Mode Execution ---
        if (runMslkAnalysis)
        {
            if (string.IsNullOrEmpty(mslkDebugLogPath))
            {
                 WriteError("Debug log path must be provided for MSLK analysis.");
                 return;
            }
            RunMslkAnalysis(mslkDebugLogPath, mslkSkippedLogPath);
        }
        else if (runHypothesisAnalysis) // ADDED: Check for new mode
        {
            // Instantiate and run the new analyzer
            var hypothesisAnalyzer = new MprrHypothesisAnalyzer();
            hypothesisAnalyzer.Analyze(inputDirectory, outputDirectory);
        }
        else
        {
            RunDirectoryCorrelation(inputDirectory, outputDirectory);
        }

        Console.WriteLine("\nAnalysis complete. Press Enter to exit.");
        Console.ReadLine();
    }

    private static void RunDirectoryCorrelation(string inputDirectory, string outputDirectory)
    {
        Console.WriteLine($"\nMode: ADT/PM4 Directory Correlation");
        Console.WriteLine("----------------------------------");
        Console.WriteLine($"Input Directory:  {inputDirectory}");
        Console.WriteLine($"Output Directory: {outputDirectory}");

        // Validate input directory
        if (!Directory.Exists(inputDirectory))
        {
            WriteError($"Input directory not found: {inputDirectory}");
            return;
        }

        // Create output directory
        try
        {
            Directory.CreateDirectory(outputDirectory);
        }
        catch (Exception ex)
        {
            WriteError($"Failed to create output directory '{outputDirectory}': {ex.Message}");
            return;
        }

        // --- Load Listfile (once) ---
        // Path relative from AnalysisTool output (bin/Debug/net8.0) up 5 levels to workspace root/
        string listfilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "src", "community-listfile-withcapitals.csv");
        Console.WriteLine($"Attempting to load listfile from: {Path.GetFullPath(listfilePath)}");
        try
        {
            _listfileData = LoadListfile(listfilePath);
            Console.WriteLine($"Successfully loaded {_listfileData.Count} entries from listfile.");
        }
        catch (FileNotFoundException)
        {
            Console.WriteLine($"Warning: Listfile not found at expected path: {Path.GetFullPath(listfilePath)}. FileDataId lookups will fail.");
            _listfileData = new Dictionary<uint, string>(); // Ensure it's initialized
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Error loading listfile: {ex.Message}. FileDataId lookups will fail.");
             _listfileData = new Dictionary<uint, string>(); // Ensure it's initialized
        }
        // --- End Load Listfile ---

        Console.WriteLine("\nScanning input directory for ADT and PM4 files...");

        // Only need PM4 files for the main loop now
        var pm4Files = Directory.EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly).ToList();

        Console.WriteLine($"Found {pm4Files.Count} PM4 files.");

        int skippedZeroByteFiles = 0; // Counter for skipped empty files
        var adtService = new AdtService(); // Instantiate service once
        var allResults = new List<Pm4AnalysisResult>(); // List to hold all results

        // --- Main Processing Loop (PM4-centric) ---
        Console.WriteLine("\nProcessing PM4 files and checking for corresponding _obj0.adt...");
        foreach (var pm4FilePath in pm4Files)
        {
            string pm4FileName = Path.GetFileName(pm4FilePath);

            // Restore skipping 00_00 - it has known issues unrelated to this investigation
            if (pm4FileName.Equals("development_00_00.pm4", StringComparison.OrdinalIgnoreCase))
            {
                 Console.WriteLine($"  Skipping known problematic file: {pm4FileName}"); // Keep minimal skip message
                 continue;
            }

            // --- Start of new try block ---
            try
            {
                // --- Basic File Handling & PM4 Loading ---
                string pm4BaseName = Path.GetFileNameWithoutExtension(pm4FilePath);
                var currentResult = new Pm4AnalysisResult { Pm4Name = pm4BaseName };

                // Skip 0-byte PM4
                if (new FileInfo(pm4FilePath).Length == 0)
                {
                    Console.WriteLine($"  Skipping 0-byte PM4 file: {pm4FileName}");
                    skippedZeroByteFiles++;
                    continue; // Skip this iteration within the try block
                }
                
                HashSet<uint> currentPm4UniqueIds = new HashSet<uint>(); // Initialize here
                bool pm4LoadSuccess = false;
                PM4File? pm4File = null; 
                try
                {
                    byte[] pm4Bytes = File.ReadAllBytes(pm4FilePath);
                    pm4File = new PM4File(pm4Bytes); 

                    // Check and Log MDSF Chunk Status - Keep this minimal check?
                    bool hasMdsfData = pm4File.MDSF != null && pm4File.MDSF.Entries.Count > 0;

                    currentPm4UniqueIds = ExtractPm4UniqueIds(pm4File);
                    currentResult.Pm4UniqueIds = currentPm4UniqueIds.OrderBy(id => id).ToList();
                    pm4LoadSuccess = true; // Mark PM4 load as successful
                }
                catch (Exception ex)
                {
                    currentResult.ProcessingError = $"PM4 Loading/Parsing Error: {ex.Message}";
                }

                // --- Check for and Process _obj0.adt (only if PM4 loaded successfully) ---
                if (pm4LoadSuccess)
                {
                    string obj0FilePath = Path.Combine(inputDirectory, pm4BaseName + "_obj0.adt");

                    if (File.Exists(obj0FilePath))
                    {
                        if (new FileInfo(obj0FilePath).Length == 0)
                        {
                            Console.WriteLine($"    Found 0-byte _obj0.adt for {pm4FileName}. No correlation possible.");
                            skippedZeroByteFiles++;
                        }
                        else
                        {
                            try
                            {
                                // Uncomment ADT loading and ID extraction
                                byte[] obj0Bytes = File.ReadAllBytes(obj0FilePath);
                                var adtObj0Data = new TerrainObjectZero(obj0Bytes);

                                currentResult.AdtUniqueIds = ExtractAdtUniqueIds(adtObj0Data).OrderBy(id => id).ToList();

                                // Uncomment placement extraction
                                var placements = adtService.ExtractPlacements(adtObj0Data, _listfileData); // Extract to variable first

                                // Restore original correlation logic
                                currentResult.CorrelatedPlacements = placements
                                    .Where(p => currentPm4UniqueIds.Contains(p.UniqueId)) 
                                    .OrderBy(p => p.UniqueId)
                                    .ToList(); 
                                
                                // Restore commented out WriteLine
                                Console.WriteLine($"      -> Found {currentResult.CorrelatedPlacements?.Count ?? 0} correlated placements."); // Restore logging (with null check)
                            }
                            catch (Exception ex)
                            {
                                currentResult.ProcessingError = (currentResult.ProcessingError ?? "") + $"; ADT Error: {ex.Message}";
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                // Catch errors in the outer block (e.g., file access issues before inner try)
                Console.WriteLine($"  !!! Outer loop error processing {pm4FileName}: {ex.GetType().Name} - {ex.Message}");
                // Optionally add an error result here if needed, though most errors should be caught inside
                 var errorResult = new Pm4AnalysisResult { Pm4Name = Path.GetFileNameWithoutExtension(pm4FilePath), ProcessingError = $"Outer Loop Error: {ex.Message}" };
                 allResults.Add(errorResult);
            }
        } // End of foreach loop

        // --- Serialize Final Results to YAML ---
        string finalYamlPath = Path.Combine(outputDirectory, "analysis_results.yaml");
        Console.WriteLine($"\nSerializing {allResults.Count} results to {finalYamlPath}...");
        try
        {
            var serializer = new SerializerBuilder()
                .WithNamingConvention(PascalCaseNamingConvention.Instance)
                .ConfigureDefaultValuesHandling(DefaultValuesHandling.OmitNull) // Omit null lists/errors
                .Build();
            string finalYamlOutput = serializer.Serialize(allResults);
            File.WriteAllText(finalYamlPath, finalYamlOutput);
            Console.WriteLine("Final YAML report written successfully.");
        }
         catch (Exception ex)
        {
            // Log error writing final YAML
            WriteError($"FATAL: Error writing final analysis results YAML: {ex.Message}");
        }
    }

    // --- Helper to Extract PM4 UniqueIDs ---
    private static HashSet<uint> ExtractPm4UniqueIds(PM4File pm4File)
    {
        var pm4UniqueIds = new HashSet<uint>();
        if (pm4File.MDSF != null && pm4File.MDSF.Entries != null && pm4File.MDOS != null && pm4File.MDOS.Entries != null)
        {
            Console.WriteLine($"    DEBUG: Extracting IDs from MDSF ({pm4File.MDSF.Entries.Count} entries) and MDOS ({pm4File.MDOS.Entries.Count} entries)."); // DEBUG
            foreach (var mdsfEntry in pm4File.MDSF.Entries)
            {
                uint mdosIndex = mdsfEntry.mdos_index;
                if (mdosIndex < pm4File.MDOS.Entries.Count)
                {
                    uint uniqueId = pm4File.MDOS.Entries[(int)mdosIndex].m_destructible_building_index;
                    pm4UniqueIds.Add(uniqueId);
                }
                // else { Log warning if needed }
            }
        }
        else 
        {
            Console.WriteLine("    DEBUG: MDSF or MDOS chunk (or their Entries) missing/null. Cannot extract PM4 UniqueIDs."); // DEBUG
        }
        // else { Log warning if chunks missing }
        return pm4UniqueIds;
    }

    // --- Helper to Extract ADT UniqueIDs ---
    private static HashSet<uint> ExtractAdtUniqueIds(TerrainObjectZero adtData)
    {
        var adtUniqueIds = new HashSet<uint>();
        // From MDDF
        if (adtData.ModelPlacementInfo?.MDDFEntries != null)
        {
            foreach (var entry in adtData.ModelPlacementInfo.MDDFEntries)
            {
                adtUniqueIds.Add(entry.UniqueID);
            }
        }
        // From MODF
        if (adtData.WorldModelObjectPlacementInfo?.MODFEntries != null)
        {
            foreach (var entry in adtData.WorldModelObjectPlacementInfo.MODFEntries)
            {
                adtUniqueIds.Add((uint)entry.UniqueId); // Cast needed
            }
        }
        return adtUniqueIds;
    }

    // --- Helper Method for Printing Usage ---
    private static void PrintHelp()
    {
         Console.WriteLine("Usage: dotnet run --project <csproj_path> -- [options]");
         Console.WriteLine("\nOptions:");
         // Corrected help text for directory mode
         Console.WriteLine("  Directory Correlation (Default Mode):");
         Console.WriteLine("    -d, --directory <input_path>    Directory containing PM4/ADT files (default: see code).");
         Console.WriteLine("    -o, --output <output_path>      Output directory for consolidated YAML report (default: ./analysis_output).");
         Console.WriteLine("\n  MSLK Log Analysis:");
         Console.WriteLine("    --analyze-mslk <debug_log> [skip_log] Run analysis on existing PM4 test debug/skipped logs.");
         Console.WriteLine("                                            Outputs summary to <debug_log_name>.debug_mslk_summary.txt.");
         Console.WriteLine("\n  MPRR/MSLK/MPRL Hypothesis Analysis:");
         Console.WriteLine("    --analyze-hypotheses               Run MPRR/MSLK/MPRL hypothesis validation analysis.");
         Console.WriteLine("\n  General:");
         Console.WriteLine("    -h, --help                          Show this help message.");
         Console.WriteLine("\nExamples:");
         string defaultInputDir = @"I:\parp-scripts\WoWToolbox_v3\original_development\development"; // Example input dir
         string defaultOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "analysis_output"); // Example output dir
         string defaultDebugLog = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\output\development\development_00_00.debug.log"; // Example log path
         Console.WriteLine($"  Correlation:     dotnet run --project src/WoWToolbox.AnalysisTool/WoWToolbox.AnalysisTool.csproj -- -d \"{defaultInputDir}\" -o \"{defaultOutputDir}\"");
         Console.WriteLine($"  MSLK Analysis:   dotnet run --project src/WoWToolbox.AnalysisTool/WoWToolbox.AnalysisTool.csproj -- --analyze-mslk \"{defaultDebugLog}\"");
    }

    // --- LoadListfile (Restore) ---
    private static Dictionary<uint, string> LoadListfile(string filePath)
    {
        var data = new Dictionary<uint, string>();
        if (!File.Exists(filePath))
        {
            // Changed to warning instead of throwing exception
            Console.WriteLine($"Warning: Listfile not found at '{filePath}'. Proceeding without listfile data.");
            return data; // Return empty dictionary
        }

        // Using ReadLines for potentially large file
        try 
        { 
            foreach (var line in File.ReadLines(filePath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                var parts = line.Split(';');
                if (parts.Length >= 2 && uint.TryParse(parts[0], out uint fileDataId) && !string.IsNullOrWhiteSpace(parts[1]))
                {
                    // Use TryAdd to handle potential duplicate FileDataIds gracefully (keep first encountered)
                    data.TryAdd(fileDataId, parts[1].Trim()); 
                }
                // else { // Optional: Log malformed lines }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Error reading listfile '{filePath}': {ex.Message}. Proceeding without listfile data.");
            data.Clear(); // Clear any partially read data on error
        }
        return data;
    }

    // --- Error Writer (Restore) ---
    private static void WriteError(string message)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\nError: {message}");
        Console.ResetColor();
        // No automatic exit here, let the calling method decide
    }

    // --- New Method for MSLK Analysis ---
    private static void RunMslkAnalysis(string debugLogPath, string? skippedLogPath)
    {
        Console.WriteLine("\nMode: MSLK Log Analysis");
        Console.WriteLine("-----------------------");
        Console.WriteLine($"Debug Log: {debugLogPath}");
        if (!string.IsNullOrEmpty(skippedLogPath))
        {
            Console.WriteLine($"Skipped Log: {skippedLogPath}");
        }

        // Check if files exist
        if (!File.Exists(debugLogPath))
        {
            WriteError($"Debug log file not found: {debugLogPath}");
            return;
        }
        if (!string.IsNullOrEmpty(skippedLogPath) && !File.Exists(skippedLogPath))
        {
            WriteError($"Skipped log file not found: {skippedLogPath}");
            return;
        }

        try
        {
            Console.WriteLine("Starting MSLK analysis...");
            string outputSummaryPath = Path.Combine(Path.GetDirectoryName(debugLogPath) ?? ".", Path.GetFileNameWithoutExtension(debugLogPath) + ".debug_mslk_summary.txt");
            Console.WriteLine($"Outputting summary to: {outputSummaryPath}");
            // Call the new summary method that writes to a file
            MslkAnalyzer.AnalyzeMslkAndSummarizeToFile(debugLogPath, skippedLogPath, outputSummaryPath); 
            Console.WriteLine("MSLK analysis complete.");
        }
        catch (Exception ex)
        {
            WriteError($"An error occurred during MSLK analysis: {ex.Message}\n{ex.StackTrace}");
        }

        // Console.WriteLine("\nMSLK analysis finished. Press Enter to exit.");
        // Console.ReadLine(); // Removed - Wait is now in AnalyzeMslkAndSummarizeToFile
    }
} // End of Program class