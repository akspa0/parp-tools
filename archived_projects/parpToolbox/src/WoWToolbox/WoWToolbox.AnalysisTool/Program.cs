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
using WoWToolbox.Core.WMO; // Ensure WMO namespace is present
using WoWToolbox.Core.Models; // Ensure Models namespace is present
using System.Globalization; // For CultureInfo in helper

// +++ ADDED: MeshDataExtensions class for IsValid() +++
public static class MeshDataExtensions
{
    public static bool IsValid(this MeshData? meshData) // Made parameter nullable
    {
        return meshData != null 
            && meshData.Vertices != null 
            && meshData.Indices != null 
            && meshData.Vertices.Count > 0 
            && meshData.Indices.Count > 0 
            && meshData.Indices.Count % 3 == 0;
    }
} 
// +++ END: MeshDataExtensions +++

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
                case "convert-v14-ironforge":
                    ConvertV14Ironforge(args);
                    return;
                case "dump-wmo-meshes":
                    if (i + 1 < args.Length)
                    {
                        DumpWmoMeshes(args[++i]);
                        return;
                    }
                    else
                    {
                        WriteError("Missing value for root WMO path argument (dump-wmo-meshes).\nUsage: dump-wmo-meshes <rootWmoPath>");
                        return;
                    }
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

    // --- v14 WMO to v17 conversion (hardcoded for Ironforge_053.wmo) ---
    private static void ConvertV14Ironforge(string[] args)
    {
        string inputPath = "test_data/053_wmo/Ironforge_053.wmo";
        string assetName = "Ironforge_053";
        string outputDir = Path.Combine("outputs", "wmo", assetName);
        Directory.CreateDirectory(outputDir);
        string outputRootPath = Path.Combine(outputDir, assetName + ".wmo");

        Console.WriteLine($"[WMOv14->v17] Loading v14 WMO: {inputPath}");
        // REVERTED: Assuming LoadGroupInfo primarily returns count. Handle groupNames if available/needed.
        (int groupCount, List<string> groupNames) = WoWToolbox.Core.WMO.WmoRootLoader.LoadGroupInfo(inputPath);
        if (groupCount <= 0)
        {
            Console.WriteLine($"[WMOv14->v17] Failed to load group info from {inputPath}");
            return;
        }
        Console.WriteLine($"[WMOv14->v17] Found {groupCount} groups");

        // Load and convert each group file (assume group files are named Ironforge_053_000.wmo, etc.)
        // FIX CS1503: Change list type
        var groupMeshes = new List<WmoGroupMesh>(); // CHANGED from MeshData
        for (int i = 0; i < groupCount; i++)
        {
            // Use groupName if available, otherwise default naming
             string groupFile = (groupNames != null && i < groupNames.Count && !string.IsNullOrEmpty(groupNames[i]))
                 ? Path.Combine(Path.GetDirectoryName(inputPath) ?? ".", groupNames[i])
                 : inputPath.Replace(".wmo", $"_{i:D3}.wmo");
                 
            if (!File.Exists(groupFile))
            {
                Console.WriteLine($"[WMOv14->v17] Group file missing: {groupFile}");
                continue;
            }
            using var groupStream = File.OpenRead(groupFile);
            // Ensure type matches return type (WmoGroupMesh?)
            var mesh = WoWToolbox.Core.WMO.WmoGroupMesh.LoadFromStream(groupStream, groupFile);
            // Add null check before adding to list
            if (mesh != null) 
            {
                 groupMeshes.Add(mesh);
            }
            else
            {
                 Console.WriteLine($"[WMOv14->v17] Failed to load mesh data from group file: {groupFile}");
            }
        }

        // TODO: Implement v14->v17 conversion logic here
        // For now, just re-save the meshes as a proof of concept
        for (int i = 0; i < groupMeshes.Count; i++)
        {
            string outGroupFile = Path.Combine(outputDir, $"{assetName}_{i:D3}.wmo");
            // Optionally, implement a SaveToWmoGroupFile method for v17 format
            // For now, just save as OBJ for visual validation
            string outObjFile = outGroupFile + ".obj";
            // FIX: Convert WmoGroupMesh to MeshData before saving
            SaveMeshDataToObjHelper(WmoGroupMeshToMeshData(groupMeshes[i]), outObjFile);
            Console.WriteLine($"[WMOv14->v17] Saved OBJ: {outObjFile}");
        }
        Console.WriteLine($"[WMOv14->v17] Conversion complete. Output in: {outputDir}");
        return;
    }

    private static void DumpWmoMeshes(string rootWmoPath)
    {
        Console.WriteLine($"\nMode: Dump WMO Meshes");
        Console.WriteLine($"----------------------");
        Console.WriteLine($"Root WMO Path: {rootWmoPath}");

        if (!File.Exists(rootWmoPath))
        {
            WriteError($"Root WMO file not found: {rootWmoPath}");
            return;
        }

        string baseOutputPath = Path.Combine(Directory.GetCurrentDirectory(), "output", "wmo", Path.GetFileNameWithoutExtension(rootWmoPath));
        Directory.CreateDirectory(baseOutputPath);
        string logFilePath = Path.Combine(baseOutputPath, $"{Path.GetFileNameWithoutExtension(rootWmoPath)}_mesh_log.txt");

        using (var logWriter = new StreamWriter(logFilePath, false)) // Overwrite log file
        {
            logWriter.WriteLine($"[WMO Mesh Dump] Loading root WMO: {rootWmoPath}");
            Console.WriteLine($"[WMO Mesh Dump] Loading root WMO: {rootWmoPath}");

            try
            {
                // REVERTED: Load group info first, assuming method primarily returns count.
                // Remove dependency on WmoRootLoadResult (CS0246)
                (int groupCount, _) = WmoRootLoader.LoadGroupInfo(rootWmoPath);
                logWriter.WriteLine($"[WMO Mesh Dump] Found {groupCount} group(s) referenced.");
                Console.WriteLine($"[WMO Mesh Dump] Found {groupCount} group(s) referenced.");

                // Removed check for WmoRootLoadResult (CS0103)
                
                if (groupCount == 0)
                {
                     logWriter.WriteLine($"[WMO Mesh Dump] WMO has no groups (groupCount = 0). No meshes to dump.");
                     Console.WriteLine($"[WMO Mesh Dump] WMO has no groups (groupCount = 0). No meshes to dump.");
                     return; // Exit if no groups
                }

                List<MeshData> allGroupMeshes = new List<MeshData>();
                string wmoDir = Path.GetDirectoryName(rootWmoPath) ?? ".";
                string wmoBaseName = Path.GetFileNameWithoutExtension(rootWmoPath);

                for (int i = 0; i < groupCount; i++)
                {
                    string groupPath = Path.Combine(wmoDir, $"{wmoBaseName}_{i:000}.wmo");
                    logWriter.WriteLine($"[WMO Mesh Dump] Attempting to load group {i}: {groupPath}");
                    Console.WriteLine($"[WMO Mesh Dump] Attempting to load group {i}: {groupPath}");

                    if (!File.Exists(groupPath))
                    {
                        logWriter.WriteLine($"  -> Group file not found. Skipping.");
                        Console.WriteLine($"  -> Group file not found. Skipping.");
                        continue;
                    }

                    try
                    {
                        using var stream = File.OpenRead(groupPath);
                        var groupMesh = WmoGroupMesh.LoadFromStream(stream, groupPath);
                        var meshData = WmoGroupMeshToMeshData(groupMesh);
                        if (meshData.IsValid())
                        {
                            logWriter.WriteLine($"  -> Successfully loaded group {i}. Vertices: {meshData.Vertices.Count}, Triangles: {meshData.Indices.Count / 3}");
                            Console.WriteLine($"  -> Successfully loaded group {i}. Vertices: {meshData.Vertices.Count}, Triangles: {meshData.Indices.Count / 3}");
                            allGroupMeshes.Add(meshData);

                            // Save individual group mesh
                            string objOutputPath = Path.Combine(baseOutputPath, $"{wmoBaseName}_{i:000}.obj");
                            logWriter.WriteLine($"  -> Saving group {i} mesh to: {objOutputPath}");
                            Console.WriteLine($"  -> Saving group {i} mesh to: {objOutputPath}");
                            SaveMeshDataToObjHelper(meshData, objOutputPath);
                        }
                        else
                        {
                            logWriter.WriteLine($"  -> Failed to load or mesh data invalid for group {i}.");
                            Console.WriteLine($"  -> Failed to load or mesh data invalid for group {i}.");
                        }
                    }
                    catch (Exception groupEx)
                    {
                        logWriter.WriteLine($"  -> ERROR loading group {i}: {groupEx.Message}");
                        logWriter.WriteLine($"     StackTrace: {groupEx.StackTrace}");
                        Console.WriteLine($"  -> ERROR loading group {i}: {groupEx.Message}");
                        // Continue to next group
                    }
                }

                // Merge and save combined mesh if any groups were loaded
                if (allGroupMeshes.Any())
                {
                    logWriter.WriteLine($"[WMO Mesh Dump] Merging {allGroupMeshes.Count} loaded group meshes...");
                    Console.WriteLine($"[WMO Mesh Dump] Merging {allGroupMeshes.Count} loaded group meshes...");
                    // FIX: Use custom merge utility for MeshData
                    MeshData? combinedMesh = MergeMeshDataList(allGroupMeshes);

                    if (combinedMesh.IsValid())
                    {
                        string combinedObjPath = Path.Combine(baseOutputPath, $"{wmoBaseName}_combined.obj");
                        logWriter.WriteLine($"[WMO Mesh Dump] Saving combined mesh (Vertices: {combinedMesh.Vertices.Count}, Triangles: {combinedMesh.Indices.Count / 3}) to: {combinedObjPath}");
                        Console.WriteLine($"[WMO Mesh Dump] Saving combined mesh (Vertices: {combinedMesh.Vertices.Count}, Triangles: {combinedMesh.Indices.Count / 3}) to: {combinedObjPath}");
                        SaveMeshDataToObjHelper(combinedMesh, combinedObjPath);
                    }
                    else
                    {
                        logWriter.WriteLine($"[WMO Mesh Dump] Combined mesh is invalid or null after merging.");
                        Console.WriteLine($"[WMO Mesh Dump] Combined mesh is invalid or null after merging.");
                    }
                }
                 else
                {
                     logWriter.WriteLine($"[WMO Mesh Dump] No valid group meshes were loaded. Cannot create combined mesh.");
                      Console.WriteLine($"[WMO Mesh Dump] No valid group meshes were loaded. Cannot create combined mesh.");
                }

            }
            catch (Exception ex)
            {
                logWriter.WriteLine($"[WMO Mesh Dump] UNHANDLED ERROR processing {rootWmoPath}: {ex.Message}");
                logWriter.WriteLine($"   StackTrace: {ex.StackTrace}");
                Console.WriteLine($"[WMO Mesh Dump] UNHANDLED ERROR processing {rootWmoPath}: {ex.Message}");
            }
            finally
            {
                 logWriter.WriteLine($"[WMO Mesh Dump] Finished processing {rootWmoPath}. Log saved to: {logFilePath}");
                  Console.WriteLine($"[WMO Mesh Dump] Finished processing {rootWmoPath}. Log saved to: {logFilePath}");
            }
        } // using logWriter
    }

    // +++ ADDED HELPER METHOD (Copied from ComparisonTests.cs) +++
    private static void SaveMeshDataToObjHelper(MeshData meshData, string outputPath)
    {
         if (meshData == null)
        {
            Console.WriteLine($"[WARN] MeshData is null, cannot save OBJ.");
            return;
        }
        try
        {
             string? directoryPath = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }
            using (var writer = new StreamWriter(outputPath, false))
            {
                CultureInfo culture = CultureInfo.InvariantCulture;
                writer.WriteLine($"# Mesh saved by WoWToolbox.AnalysisTool"); // Updated comment source
                writer.WriteLine($"# Vertices: {meshData.Vertices.Count}");
                // FIX CS1061 reference (originally here too)
                writer.WriteLine($"# Triangles: {meshData.Indices.Count / 3}"); 
                writer.WriteLine($"# Generated: {DateTime.Now}");
                writer.WriteLine();
                if (meshData.Vertices.Count > 0)
                {
                    writer.WriteLine("# Vertex Definitions");
                    foreach (var vertex in meshData.Vertices)
                    {
                        writer.WriteLine(string.Format(culture, "v {0} {1} {2}", vertex.X, vertex.Y, vertex.Z));
                    }
                    writer.WriteLine();
                }
                if (meshData.Indices.Count > 0)
                {
                    writer.WriteLine("# Face Definitions");
                    for (int i = 0; i < meshData.Indices.Count; i += 3)
                    {
                        // OBJ indices are 1-based
                        int idx0 = meshData.Indices[i + 0] + 1;
                        int idx1 = meshData.Indices[i + 1] + 1;
                        int idx2 = meshData.Indices[i + 2] + 1;
                        writer.WriteLine($"f {idx0} {idx1} {idx2}");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERR] Failed to save MeshData to OBJ file '{outputPath}': {ex.Message}");
            // Optionally re-throw or handle differently
            // throw; 
        }
    }
    // +++ END HELPER METHOD +++

    // Add a helper to convert WmoGroupMesh to MeshData
    private static MeshData WmoGroupMeshToMeshData(WmoGroupMesh mesh)
    {
        var md = new MeshData();
        if (mesh == null) return md;
        foreach (var v in mesh.Vertices)
            md.Vertices.Add(v.Position);
        foreach (var tri in mesh.Triangles)
        {
            md.Indices.Add(tri.Index0);
            md.Indices.Add(tri.Index1);
            md.Indices.Add(tri.Index2);
        }
        return md;
    }

    // +++ ADDED: MeshData merge utility +++
    private static MeshData MergeMeshDataList(List<MeshData> meshes)
    {
        var merged = new MeshData();
        int vertexOffset = 0;
        foreach (var mesh in meshes)
        {
            // Copy vertices
            merged.Vertices.AddRange(mesh.Vertices);
            // Copy indices with offset
            foreach (var idx in mesh.Indices)
            {
                merged.Indices.Add(idx + vertexOffset);
            }
            vertexOffset += mesh.Vertices.Count;
        }
        return merged;
    }
} // End of Program class