using System;
using System.IO;
using System.Linq;
using WoWFormatLib.FileReaders;
using WoWFormatLib.FileProviders;
using ParpToolbox;
using ParpToolbox.Services.WMO;
using ParpToolbox.Utils;

// Initialize logging to timestamped file
var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
var logOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{timestamp}");
ConsoleLogger.Initialize(logOutputDir);

if (args.Length == 0)
{
    ConsoleLogger.WriteLine("Usage: parpToolbox <command> --input <file> [flags]\n" +
                      "       or parpToolbox <command> <file> [flags] (positional)\n" +
                      "Commands: wmo | pm4 | pd4 | pm4-region | pm4-test-chunks | pm4-analyze-indices\n" +
                      "Common flags:\n" +
                      "   --include-collision   Include collision geometry (WMO only)\n" +
                      "   --split-groups        Export each WMO group separately\n" +
                      "   --include-facades     Keep facade/no-draw geometry (WMO)\n" +
                      "   --exportfaces        Write OBJ faces (default: point cloud)\n" +
                      "   --exportchunks      Export each MSUR group to separate OBJ\n" +
                      "   --bulk-dump        Dump OBJs & CSVs for all groupings\n" +
                      "   --csv-dump         Export all chunk data to CSV files");
    ConsoleLogger.Close();
    return 1;
}

var command = args[0].ToLowerInvariant();

switch (command)
{
    case "wmo":
        // handled below
        break;
    case "pm4":
    case "pd4":
    case "pm4-region":
    case "pm4-test-chunks":
        // handled below
        break;
    default:
        ConsoleLogger.WriteLine($"Error: Unknown command '{command}'");
        return 1;
}

string inputFile = null;
bool includeCollision = false;
bool splitGroups = false;
bool includeFacades = false;
bool exportFaces = false;
bool exportChunks = false;
bool bulkDump = false;
bool csvDump = false;
// Detect optional flags
if (args.Contains("--include-collision"))
    includeCollision = true;
if (args.Contains("--split-groups"))
    splitGroups = true;
if (args.Contains("--include-facades") || args.Contains("--include-no-draw"))
    includeFacades = true;
if (args.Contains("--exportfaces"))
    exportFaces = true;
if (args.Contains("--exportchunks"))
    exportChunks = true;
if (args.Contains("--bulk-dump"))
    bulkDump = true;
if (args.Contains("--csv-dump"))
    csvDump = true;

var localProvider = new LocalFileProvider(".");
FileProvider.SetProvider(localProvider, "local");
FileProvider.SetDefaultBuild("local");

var inputIndex = Array.IndexOf(args, "--input");
if (inputIndex == -1) inputIndex = Array.IndexOf(args, "-i");

if (inputIndex != -1 && inputIndex + 1 < args.Length)
{
    inputFile = args[inputIndex + 1];
}

// Fallback: treat first argument after command that doesn't start with '-' as the input file
if (string.IsNullOrEmpty(inputFile))
{
    foreach (var candidate in args.Skip(1))
    {
        if (!candidate.StartsWith("-"))
        {
            inputFile = candidate;
            break;
        }
    }
}

if (string.IsNullOrEmpty(inputFile))
{
    Console.WriteLine("Error: --input <file> is required for the wmo command.");
    return 1;
}

var fileInfo = new FileInfo(inputFile);
if (!fileInfo.Exists)
{
    Console.WriteLine($"Error: Input file not found at '{fileInfo.FullName}'");
    return 1;
}

if (command == "wmo")
{
    Console.WriteLine($"Processing WMO file: {fileInfo.FullName}");
}
else
{
    Console.WriteLine($"Processing {command.ToUpper()} file: {fileInfo.FullName}");
}
try
{
    if (command == "wmo")
    {
        var wmoLoader = new WowToolsLocalWmoLoader();
    var (textures, groups) = wmoLoader.Load(inputFile, includeFacades);
    Console.WriteLine($"Successfully loaded WMO with {textures.Count} textures and {groups.Count} groups.");

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile));
    if (splitGroups)
    {
        Console.WriteLine($"Exporting each group to {outputDir}...");
        ObjExporter.ExportPerGroup(groups, outputDir, includeCollision);
    }
    else
    {
        var outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputFile) + ".obj");
        Console.WriteLine($"Exporting to {outputFile}...");
        ObjExporter.Export(groups, outputFile, includeCollision);
    }
    ConsoleLogger.WriteLine("Export complete!");
}
}
catch (Exception e)
{
    Console.WriteLine($"An error occurred: {e.Message}");
    Console.WriteLine(e.StackTrace);
    return 1;
}


if (command == "pm4")
{
    ConsoleLogger.WriteLine($"Parsing PM4 file: {fileInfo.FullName}");
    var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
    var scene = loader.Load(fileInfo.FullName);

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile));

    // Bulk dump path – CSV + per-group OBJs and early exit
    if (bulkDump)
    {
        var bulkDir = Path.Combine(outputDir, "bulk_dump");
        ConsoleLogger.WriteLine($"Running bulk dump to {bulkDir} ...");
        ParpToolbox.Services.PM4.Pm4BulkDumper.Dump(scene, bulkDir, exportFaces, ParpToolbox.Services.PM4.Pm4Adapter.LastRawMsvtData);
        ConsoleLogger.WriteLine("Bulk dump complete!");
        return 0;
    }
    
    // CSV dump path – export all chunk data to CSV files and early exit
    if (csvDump)
    {
        var csvDir = Path.Combine(outputDir, "csv_dump");
        ConsoleLogger.WriteLine($"Running CSV dump to {csvDir} ...");
        ParpToolbox.Services.PM4.Pm4CsvDumper.DumpAllChunks(scene, csvDir);
        return 0;
    }

    var outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputFile) + ".obj");
    if (exportChunks && scene.Groups.Count > 0)
    {
        ConsoleLogger.WriteLine($"Exporting {scene.Groups.Count} groups to {outputDir}...");
        ParpToolbox.Services.PM4.Pm4GroupObjExporter.Export(scene, outputDir, exportFaces);
    }
    else
    {
        ConsoleLogger.WriteLine($"Exporting OBJ to {outputFile}...");
        ParpToolbox.Services.PM4.Pm4ObjExporter.Export(scene, outputFile, exportFaces);
    }
    ConsoleLogger.WriteLine("Export complete!");
}
else if (command == "pd4")
{
    ConsoleLogger.WriteLine($"Parsing PD4 file: {fileInfo.FullName}");
    var loader = new ParpToolbox.Services.PD4.Pd4Adapter();
    var scene = loader.Load(fileInfo.FullName);

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile));
    var outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputFile) + ".obj");
    if (exportChunks && scene.Groups.Count > 0)
    {
        ConsoleLogger.WriteLine($"Exporting {scene.Groups.Count} groups to {outputDir}...");
        ParpToolbox.Services.PM4.Pm4GroupObjExporter.Export(scene, outputDir, exportFaces);
    }
    else
    {
        ConsoleLogger.WriteLine($"Exporting OBJ to {outputFile}...");
        ParpToolbox.Services.PM4.Pm4ObjExporter.Export(scene, outputFile, exportFaces);
    }
    ConsoleLogger.WriteLine("Export complete!");
}
else if (command == "pm4-region")
{
    // PM4 Region command - load entire directory as unified global scene
    if (string.IsNullOrEmpty(inputFile))
    {
        ConsoleLogger.WriteLine("Error: Input directory required for pm4-region command");
        ConsoleLogger.Close();
        return 1;
    }
    
    if (!Directory.Exists(inputFile))
    {
        ConsoleLogger.WriteLine($"Error: Directory not found: {inputFile}");
        ConsoleLogger.Close();
        return 1;
    }
    
    ConsoleLogger.WriteLine($"Loading PM4 region from directory: {inputFile}");
    
    // Load all PM4 files in directory as unified global scene
    var globalScene = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.LoadRegion(inputFile, "*.pm4");
    
    // Convert to standard scene for compatibility with existing exporters
    var unifiedScene = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.ToStandardScene(globalScene);
    
    var outputDir = ProjectOutput.CreateOutputDirectory("pm4_region_" + Path.GetFileName(inputFile.TrimEnd(Path.DirectorySeparatorChar)));
    
    // Test the unified scene with our object assembler
    ConsoleLogger.WriteLine("Testing unified scene with MSUR object assembler...");
    var assembledObjects = ParpToolbox.Services.PM4.Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(unifiedScene);
    
    // Export the assembled objects
    if (assembledObjects.Count > 0)
    {
        ParpToolbox.Services.PM4.Pm4MsurObjectAssembler.ExportMsurObjects(assembledObjects, unifiedScene, outputDir);
        ConsoleLogger.WriteLine($"Exported {assembledObjects.Count} unified objects from {globalScene.TotalLoadedTiles} tiles");
    }
    else
    {
        ConsoleLogger.WriteLine("No objects assembled from unified scene");
    }
    
    ConsoleLogger.WriteLine("PM4 region processing complete!");
}
else if (command == "pm4-test-chunks")
{
    // PM4 Chunk Testing command - run exhaustive tests to discover correct grouping logic
    if (string.IsNullOrEmpty(inputFile))
    {
        ConsoleLogger.WriteLine("Error: Input PM4 file required for pm4-test-chunks command");
        ConsoleLogger.Close();
        return 1;
    }
    
    var testFileInfo = new FileInfo(inputFile);
    if (!testFileInfo.Exists)
    {
        ConsoleLogger.WriteLine($"Error: File not found: {inputFile}");
        ConsoleLogger.Close();
        return 1;
    }
    
    ConsoleLogger.WriteLine($"Running exhaustive chunk combination tests on: {testFileInfo.FullName}");
    
    // Load the PM4 file
    var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
    var scene = loader.Load(fileInfo.FullName);
    
    var outputDir = ProjectOutput.CreateOutputDirectory("chunk_tests_" + Path.GetFileNameWithoutExtension(inputFile));
    
    // Run exhaustive chunk combination tests
    ConsoleLogger.WriteLine("Starting exhaustive chunk combination testing...");
    var testResults = ParpToolbox.Services.PM4.Pm4ChunkCombinationTester.RunExhaustiveChunkTests(scene, outputDir);
    
    // Report results
    ConsoleLogger.WriteLine($"Completed {testResults.Count} tests");
    if (testResults.Any())
    {
        var bestResult = testResults.First();
        ConsoleLogger.WriteLine($"Best result: {bestResult.TestName}");
        ConsoleLogger.WriteLine($"  Quality Score: {bestResult.GeometryQualityScore:F3}");
        ConsoleLogger.WriteLine($"  Objects: {bestResult.ObjectCount}, Triangles: {bestResult.TotalTriangles}");
        ConsoleLogger.WriteLine($"  Details: {bestResult.Details}");
    }
    
    ConsoleLogger.WriteLine("Chunk combination testing complete!");
}
else if (command == "pm4-index-patterns")
{
    // PM4 Index Pattern Analysis command - analyze index patterns and missing data in PM4 files
    if (string.IsNullOrEmpty(inputFile))
    {
        ConsoleLogger.WriteLine("Error: Input PM4 file required for pm4-index-patterns command");
        ConsoleLogger.Close();
        return 1;
    }
    
    var testFileInfo = new FileInfo(inputFile);
    if (!testFileInfo.Exists)
    {
        ConsoleLogger.WriteLine($"Error: File not found: {inputFile}");
        ConsoleLogger.Close();
        return 1;
    }
    
    AnalyzePm4IndexPatterns(inputFile);
}

ConsoleLogger.Close();
return 0;
