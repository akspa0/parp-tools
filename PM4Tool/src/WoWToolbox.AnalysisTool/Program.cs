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
using System.Text; // For StringBuilder in summary report

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

        Console.WriteLine($"DEBUG: Attempting to enumerate files in: {inputDirectory}"); // DEBUG
        var allFiles = Directory.EnumerateFiles(inputDirectory, "*.*", SearchOption.TopDirectoryOnly)
                                .Where(f => f.EndsWith(".adt", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase))
                                .ToList();
        Console.WriteLine($"DEBUG: Found {allFiles.Count} total .adt/.pm4 files."); // DEBUG
        if (allFiles.Any())
        {
            Console.WriteLine($"DEBUG: First few files found: {string.Join(", ", allFiles.Take(5).Select(Path.GetFileName))}"); // DEBUG
        }

        var pm4Files = allFiles.Where(f => f.EndsWith(".pm4", StringComparison.OrdinalIgnoreCase)).ToList();
        // Identify base ADT files (e.g., map_xx_yy.adt, NOT map_xx_yy_obj0.adt)
        var baseAdtFiles = allFiles.Where(f => f.EndsWith(".adt", StringComparison.OrdinalIgnoreCase) && 
                                               !Path.GetFileNameWithoutExtension(f).EndsWith("_obj0", StringComparison.OrdinalIgnoreCase) &&
                                               !Path.GetFileNameWithoutExtension(f).EndsWith("_tex0", StringComparison.OrdinalIgnoreCase) && // Add other split types if needed
                                               !Path.GetFileNameWithoutExtension(f).EndsWith("_lod", StringComparison.OrdinalIgnoreCase)) 
                                     .ToList();
        Console.WriteLine($"DEBUG: Filtered {baseAdtFiles.Count} base ADT files and {pm4Files.Count} PM4 files."); // DEBUG

        // Console.WriteLine($"Found {baseAdtFiles.Count} base ADT files and {pm4Files.Count} PM4 files."); // Replaced by DEBUG line

        var adtPm4Pairs = new Dictionary<string, string>(); // Key: base ADT path, Value: PM4 path
        var pm4OnlyFiles = new List<string>(pm4Files);
        var adtOnlyFiles = new List<string>(baseAdtFiles);
        var pm4UniqueIdCache = new Dictionary<string, HashSet<uint>>(); // Cache extracted IDs
        var errorsEncountered = new List<string>();
        int skippedZeroByteFiles = 0; // Counter for skipped empty files
        var globalUniqueIds = new HashSet<uint>(); // Set to store all unique IDs globally

        // --- Match Pairs ---
        foreach (var adtPath in baseAdtFiles)
        {
            string adtFileName = Path.GetFileName(adtPath);
            string adtBaseNameNoExt = Path.GetFileNameWithoutExtension(adtPath);
            string[] nameParts = adtBaseNameNoExt.Split('_');

            // Expecting format like "development_X_Y"
            if (nameParts.Length >= 3 && 
                int.TryParse(nameParts[nameParts.Length - 2], out int coordX) && 
                int.TryParse(nameParts[nameParts.Length - 1], out int coordY))
            {
                // Format expected PM4 name with zero-padding
                string expectedPm4BaseName = $"{nameParts[0]}_{coordX:D2}_{coordY:D2}"; // e.g., development_00_00
                string expectedPm4FileName = expectedPm4BaseName + ".pm4";

                // Find matching PM4 file
                string? matchingPm4 = pm4Files.FirstOrDefault(p => Path.GetFileName(p).Equals(expectedPm4FileName, StringComparison.OrdinalIgnoreCase));

                if (matchingPm4 != null)
                {
                    adtPm4Pairs.Add(adtPath, matchingPm4);
                    pm4OnlyFiles.Remove(matchingPm4);
                    adtOnlyFiles.Remove(adtPath);
                }
            }
            else
            {
                // Log if ADT filename format is unexpected
                Console.WriteLine($"Warning: Could not parse coordinates from ADT filename: {adtFileName}. Skipping pairing for this file.");
                // This file will remain in adtOnlyFiles list
            }
        }
        Console.WriteLine($"DEBUG: Matched {adtPm4Pairs.Count} ADT/PM4 pairs."); // DEBUG

        // --- Process All PM4 Files ---
        Console.WriteLine("\nProcessing all PM4 files to extract UniqueIDs...");
        foreach (var pm4FilePath in pm4Files)
        {
            // Check for 0-byte file
            if (new FileInfo(pm4FilePath).Length == 0)
            {
                Console.WriteLine($"  Skipping 0-byte PM4 file: {Path.GetFileName(pm4FilePath)}");
                skippedZeroByteFiles++;
                continue;
            }

            string pm4FileName = Path.GetFileNameWithoutExtension(pm4FilePath);
            string fileOutputDir = Path.Combine(outputDirectory, pm4FileName);
            Directory.CreateDirectory(fileOutputDir); // Ensure subdirectory exists
            // string uniqueIdOutputPath = Path.Combine(fileOutputDir, "pm4_unique_ids.txt"); // REMOVED Per-file output path

            try
            {
                Console.WriteLine($"  Processing {Path.GetFileName(pm4FilePath)}...");
                byte[] pm4Bytes = File.ReadAllBytes(pm4FilePath);
                var pm4File = new PM4File(pm4Bytes);

                var uniqueIds = ExtractPm4UniqueIds(pm4File);
                pm4UniqueIdCache[pm4FilePath] = uniqueIds; // Cache for correlation step
                globalUniqueIds.UnionWith(uniqueIds); // Add to global set

                // Write UniqueIDs to file // REMOVED Per-file write
                // File.WriteAllLines(uniqueIdOutputPath, uniqueIds.Select(id => id.ToString()));
                // Console.WriteLine($"    -> Extracted {uniqueIds.Count} UniqueIDs. Saved to {Path.GetFileName(uniqueIdOutputPath)}");
                Console.WriteLine($"    -> Extracted {uniqueIds.Count} UniqueIDs."); // Updated log message
            }
            catch (Exception ex)
            {
                string errorMsg = $"Error processing PM4 file '{Path.GetFileName(pm4FilePath)}': {ex.Message}";
                Console.WriteLine($"    -> Error: {ex.Message}");
                errorsEncountered.Add(errorMsg);
                // Optionally write an error file in the subdirectory
                File.WriteAllText(Path.Combine(fileOutputDir, "error.log"), errorMsg + "\n" + ex.StackTrace);
            }
        }

        // --- Write Global Unique ID File ---
        string globalUniqueIdPath = Path.Combine(outputDirectory, "global_pm4_unique_ids.txt");
        Console.WriteLine($"\nWriting {globalUniqueIds.Count} unique PM4 IDs globally to {globalUniqueIdPath}...");
        try
        {
            // Sort IDs before writing for consistency
            File.WriteAllLines(globalUniqueIdPath, globalUniqueIds.OrderBy(id => id).Select(id => id.ToString()));
            Console.WriteLine("Global unique ID file written successfully.");
        }
        catch (Exception ex)
        {
            string errorMsg = $"Error writing global unique ID file: {ex.Message}";
            Console.WriteLine($"    -> Error: {ex.Message}");
            errorsEncountered.Add(errorMsg);
        }

        // --- Process ADT/PM4 Pairs ---
        Console.WriteLine("\nProcessing ADT/PM4 pairs for correlation...");
        var adtService = new AdtService();
        foreach (var pair in adtPm4Pairs)
        {
            string baseAdtFilePath = pair.Key;
            string pm4FilePath = pair.Value;
            string pm4BaseFileName = Path.GetFileNameWithoutExtension(pm4FilePath); // Guaranteed XX_YY format
            string fileOutputDir = Path.Combine(outputDirectory, pm4BaseFileName); // Use PM4 name for dir
            // Subdirectory should already exist from PM4 processing
            string correlationOutputPath = Path.Combine(fileOutputDir, "correlated_placements.yaml");

            Console.WriteLine($"  Processing pair {pm4BaseFileName}..."); // Log consistent name
            try
            {
                // Load Obj0 ADT (Required for placements and filenames)
                string obj0FilePath = Path.ChangeExtension(baseAdtFilePath, null) + "_obj0.adt";
                TerrainObjectZero? adtObj0Data = null;
                if (File.Exists(obj0FilePath))
                {
                    // Check for 0-byte obj0 ADT
                    if (new FileInfo(obj0FilePath).Length == 0)
                    {
                        Console.WriteLine($"    Skipping pair: _obj0 ADT is 0 bytes ({Path.GetFileName(obj0FilePath)})");
                        errorsEncountered.Add($"Skipped pair {pm4BaseFileName}: _obj0 ADT is 0 bytes");
                        skippedZeroByteFiles++;
                        continue;
                    }
                    byte[] obj0Bytes = File.ReadAllBytes(obj0FilePath);
                    adtObj0Data = new TerrainObjectZero(obj0Bytes);
                }
                else
                {
                    Console.WriteLine($"    Warning: _obj0 file not found for {pm4BaseFileName}. Cannot extract placements or filenames. Skipping pair.");
                    errorsEncountered.Add($"Skipped pair {pm4BaseFileName}: _obj0 ADT not found");
                    continue; 
                }

                // Get cached PM4 UniqueIDs
                if (!pm4UniqueIdCache.TryGetValue(pm4FilePath, out var pm4UniqueIds))
                {
                    // Should not happen if PM4 processing was successful, but handle defensively
                    Console.WriteLine($"    Error: Could not find cached UniqueIDs for {Path.GetFileName(pm4FilePath)}. Skipping correlation.");
                    errorsEncountered.Add($"Missing cached UniqueIDs for {Path.GetFileName(pm4FilePath)}");
                    continue;
                }

                // Extract ADT Placements (Requires _obj0 data)
                var adtPlacements = adtService.ExtractPlacements(adtObj0Data, _listfileData).ToList(); // Pass only obj0Data

                // Filter ADT Placements
                var correlatedPlacements = adtPlacements.Where(p => pm4UniqueIds.Contains(p.UniqueId)).ToList();

                Console.WriteLine($"    -> Found {correlatedPlacements.Count} correlated placements.");

                // Serialize to YAML
                if (correlatedPlacements.Any())
                {
                    var serializer = new SerializerBuilder()
                        .WithNamingConvention(PascalCaseNamingConvention.Instance)
                        .Build();
                    string yamlOutput = serializer.Serialize(correlatedPlacements);
                    File.WriteAllText(correlationOutputPath, yamlOutput);
                    Console.WriteLine($"    -> Saved correlation to {Path.GetFileName(correlationOutputPath)}");
                }
            }
            catch (Exception ex)
            {
                 string errorMsg = $"Error processing ADT/PM4 pair '{pm4BaseFileName}': {ex.Message}"; // Use consistent name
                Console.WriteLine($"    -> Error: {ex.Message}");
                errorsEncountered.Add(errorMsg);
                // Ensure writing error log uses the correct (PM4-based) directory path
                File.WriteAllText(Path.Combine(fileOutputDir, "error.log"), errorMsg + "\n" + ex.StackTrace);
            }
        }

        // --- Generate Summary Report ---
        Console.WriteLine("\nGenerating summary report...");
        GenerateSummaryReport(outputDirectory, baseAdtFiles.Count, pm4Files.Count, adtPm4Pairs, pm4OnlyFiles, adtOnlyFiles, errorsEncountered, skippedZeroByteFiles);
        Console.WriteLine("Summary report generated.");
    }

    // --- Helper to Extract PM4 UniqueIDs ---
    private static HashSet<uint> ExtractPm4UniqueIds(PM4File pm4File)
    {
        var pm4UniqueIds = new HashSet<uint>();
        if (pm4File.MDSF != null && pm4File.MDSF.Entries != null && pm4File.MDOS != null && pm4File.MDOS.Entries != null)
        {
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
        // else { Log warning if chunks missing }
        return pm4UniqueIds;
    }

    // --- Helper to Generate Summary Report ---
    private static void GenerateSummaryReport(string outputDir, int totalAdt, int totalPm4, 
                                            Dictionary<string, string> pairs, List<string> pm4Only, List<string> adtOnly, 
                                            List<string> errors, int skippedZeroByte)
    {
        string reportPath = Path.Combine(outputDir, "summary_report.txt");
        var sb = new StringBuilder();

        sb.AppendLine("WoWToolbox Analysis Tool - Summary Report");
        sb.AppendLine("=========================================");
        sb.AppendLine($"Timestamp: {DateTime.Now}");
        sb.AppendLine($"Output Directory: {outputDir}");
        sb.AppendLine();
        sb.AppendLine("--- Counts ---");
        sb.AppendLine($"Total Base ADT Files Found: {totalAdt}");
        sb.AppendLine($"Total PM4 Files Found:    {totalPm4}");
        sb.AppendLine($"Processed ADT/PM4 Pairs:  {pairs.Count}");
        sb.AppendLine($"Processed PM4-Only Files: {pm4Only.Count}");
        sb.AppendLine($"Found ADT-Only Files:     {adtOnly.Count}");
        sb.AppendLine($"Errors Encountered:       {errors.Count}");
        sb.AppendLine($"Skipped 0-Byte Files:   {skippedZeroByte}");
        sb.AppendLine();

        sb.AppendLine("--- Processed ADT/PM4 Pairs ---");
        if (pairs.Any())
        {
            foreach (var pair in pairs)
            {
                sb.AppendLine($"  {Path.GetFileNameWithoutExtension(pair.Key)}");
            }
        }
        else { sb.AppendLine("  None"); }
        sb.AppendLine();

        sb.AppendLine("--- Processed PM4-Only Files ---");
         if (pm4Only.Any())
        {
            foreach (var file in pm4Only)
            {
                sb.AppendLine($"  {Path.GetFileName(file)}");
            }
        }
        else { sb.AppendLine("  None"); }
        sb.AppendLine();

        sb.AppendLine("--- Found ADT-Only Files (No PM4 Match) ---");
         if (adtOnly.Any())
        {
            foreach (var file in adtOnly)
            {
                sb.AppendLine($"  {Path.GetFileName(file)}");
            }
        }
        else { sb.AppendLine("  None"); }
        sb.AppendLine();

        sb.AppendLine("--- Errors Encountered During Processing ---");
        if (errors.Any())
        {
            foreach (var error in errors)
            {
                sb.AppendLine($"  - {error}");
            }
        }
        else { sb.AppendLine("  None"); }
        sb.AppendLine();

        File.WriteAllText(reportPath, sb.ToString());
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

    // --- Helper Method for Printing Usage ---
    private static void PrintHelp()
    {
         Console.WriteLine("Usage: dotnet run --project <csproj_path> -- [options]");
         Console.WriteLine("\nOptions:");
         Console.WriteLine("  <adt_file_path> <pm4_file_path>   Run ADT/PM4 correlation (Default mode).");
         Console.WriteLine("                                      Uses hardcoded paths if none provided.");
         Console.WriteLine("  --analyze-mslk <debug_log> [skip_log] Run MSLK log analysis.");
         Console.WriteLine("  -h, --help                          Show this help message.");
         Console.WriteLine("\nExamples:");
         string defaultAdtPath = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\test_data\development\development_0_0.adt"; // Re-declare for example
         string defaultPm4Path = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\test_data\development\development_00_00.pm4"; // Re-declare for example
         string defaultDebugLog = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\output\development\development_00_00.debug.log"; // Example log path
         Console.WriteLine($"  Correlation (Default): dotnet run --project src/WoWToolbox.AnalysisTool/WoWToolbox.AnalysisTool.csproj -- \"{defaultAdtPath}\" \"{defaultPm4Path}\"");
         Console.WriteLine($"  MSLK Analysis:       dotnet run --project src/WoWToolbox.AnalysisTool/WoWToolbox.AnalysisTool.csproj -- --analyze-mslk \"{defaultDebugLog}\"");
    }

    private static Dictionary<uint, string> LoadListfile(string filePath)
    {
        var data = new Dictionary<uint, string>();
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Listfile not found", filePath);
        }

        // Using ReadLines for potentially large file
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
        return data;
    }

    // --- Error Writer (Restored) ---
    private static void WriteError(string message)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\nError: {message}");
        Console.ResetColor();
        // No automatic exit here, let the calling method decide
        // Console.WriteLine("\nPress Enter to exit.");
        // Console.ReadLine();
    }
} // End of Program class