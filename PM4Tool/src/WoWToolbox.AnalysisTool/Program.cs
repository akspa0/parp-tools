// Program.cs in WoWToolbox.AnalysisTool
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core; // Use the core project namespace
using WoWToolbox.Core.ADT; // Added for AdtService, ADTFile, Placement
using WoWToolbox.Core.Navigation.PM4; // Added for PM4File
using WoWToolbox.Core.Navigation.PM4.Chunks; // Added for MDSFChunk
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

public class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("WoWToolbox ADT/PM4 Correlation Tool");
        Console.WriteLine("===================================");

        string? adtFilePath = null;
        string? pm4FilePath = null;

        // --- Default Hardcoded Paths --- 
        // Use specific paths provided by user as defaults
        string defaultAdtPath = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\test_data\development\development_0_0.adt";
        string defaultPm4Path = @"I:\parp-scripts\WoWToolbox_v3\test\WoWToolbox.Tests\bin\Debug\net8.0\test_data\development\development_00_00.pm4";
        // --- End Default Paths ---

        // Basic argument parsing
        if (args.Length >= 2)
        {
            adtFilePath = args[0];
            pm4FilePath = args[1];
            Console.WriteLine("Using paths from command line arguments.");
        }
        else if (args.Length == 1 && (args[0] == "-h" || args[0] == "--help"))
        {
            Console.WriteLine("Usage: dotnet run --project <csproj_path> -- <adt_file_path> <pm4_file_path>");
            Console.WriteLine($"Example: dotnet run --project src/WoWToolbox.AnalysisTool/WoWToolbox.AnalysisTool.csproj -- \"{defaultAdtPath}\" \"{defaultPm4Path}\"");
            Console.WriteLine("If no arguments are provided, it defaults to hardcoded ADT/PM4 paths.");
            return;
        }
        else
        {
            Console.WriteLine("No paths provided via arguments, using default hardcoded paths...");
            adtFilePath = defaultAdtPath;
            pm4FilePath = defaultPm4Path;
        }

        Console.WriteLine($"Using ADT File: {adtFilePath}");
        Console.WriteLine($"Using PM4 File: {pm4FilePath}");

        // Check if files exist before running analysis
        if (string.IsNullOrEmpty(adtFilePath) || !System.IO.File.Exists(adtFilePath))
        {
            WriteError($"ADT file not found or path invalid! Path: {adtFilePath ?? "(null)"}");
            return;
        }
        if (string.IsNullOrEmpty(pm4FilePath) || !System.IO.File.Exists(pm4FilePath))
        {
            WriteError($"PM4 file not found or path invalid! Path: {pm4FilePath ?? "(null)"}");
            return;
        }

        Console.WriteLine("\nStarting correlation analysis...");
        try
        {
            // 1. Load Files
            Console.WriteLine("Loading ADT file...");
            byte[] adtBytes = File.ReadAllBytes(adtFilePath);
            var adtFile = new ADTFile(adtBytes);
            Console.WriteLine("ADT file loaded.");

            Console.WriteLine("Loading PM4 file...");
            byte[] pm4Bytes = File.ReadAllBytes(pm4FilePath);
            var pm4File = new PM4File(pm4Bytes);
            Console.WriteLine("PM4 file loaded.");

            // 2. Extract ADT Placements
            var adtService = new AdtService();
            var adtPlacements = adtService.ExtractPlacements(adtFile).ToList();
            Console.WriteLine($"Extracted {adtPlacements.Count} placements from ADT.");

            // 3. Extract PM4 UniqueIDs
            var pm4UniqueIds = new HashSet<uint>();
            if (pm4File.MDSF != null && pm4File.MDSF.Entries != null && pm4File.MDOS != null && pm4File.MDOS.Entries != null)
            {
                Console.WriteLine("Extracting UniqueIDs from PM4 MDSF/MDOS...");
                foreach (var mdsfEntry in pm4File.MDSF.Entries)
                {
                    uint mdosIndex = mdsfEntry.mdos_index; // Correct field name from MDSFChunk definition
                    // Check index bounds
                    if (mdosIndex < pm4File.MDOS.Entries.Count)
                    {
                        uint uniqueId = pm4File.MDOS.Entries[(int)mdosIndex].m_destructible_building_index; // Cast mdosIndex to int
                        pm4UniqueIds.Add(uniqueId);
                    }
                    else
                    {
                        Console.WriteLine($"Warning: MDSF entry references out-of-bounds MDOS index ({mdosIndex}). Skipping.");
                    }
                }
                Console.WriteLine($"Extracted {pm4UniqueIds.Count} unique IDs from PM4.");
            }
            else
            {
                Console.WriteLine("Warning: PM4 file missing MDSF or MDOS chunk, cannot extract UniqueIDs.");
            }

            // 4. Filter ADT Placements
            var correlatedPlacements = adtPlacements.Where(p => pm4UniqueIds.Contains(p.UniqueId)).ToList();

            // 5. Output Results
            Console.WriteLine("\n--- Correlation Results ---");
            Console.WriteLine($"Found {correlatedPlacements.Count} placements in ADT that match UniqueIDs in PM4.");

            // Implement YAML output
            if (correlatedPlacements.Any())
            {
                string outputYamlPath = Path.Combine(Path.GetDirectoryName(adtFilePath) ?? ".", "correlated_placements.yaml");
                Console.WriteLine($"Serializing correlated placements to: {outputYamlPath}");

                var serializer = new SerializerBuilder()
                    .WithNamingConvention(PascalCaseNamingConvention.Instance) // Match C# property names
                    .Build();

                string yamlOutput = serializer.Serialize(correlatedPlacements);
                File.WriteAllText(outputYamlPath, yamlOutput);
                Console.WriteLine("YAML file written successfully.");
            }
            else
            {
                Console.WriteLine("No correlated placements found to serialize.");
            }

            // Optional console output for debugging can be kept or removed
            /*
            if (correlatedPlacements.Any())
            {
                Console.WriteLine("\nCorrelated Placement Details:");
                foreach (var placement in correlatedPlacements)
                {
                    Console.WriteLine($"  - UniqueID: {placement.UniqueId}, NameID: {placement.NameId}, Position: {placement.Position}");
                }
            }
            */

        }
        catch (Exception ex)
        {
            WriteError($"An error occurred during analysis: {ex.Message}\n{ex.StackTrace}");
        }

        Console.WriteLine("\nCorrelation analysis complete. Press Enter to exit.");
        Console.ReadLine();
    }

    private static void WriteError(string message)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\nError: {message}");
        Console.ResetColor();
        Console.WriteLine("\nPress Enter to exit.");
        Console.ReadLine();
    }
}