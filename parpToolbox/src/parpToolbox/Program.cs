using System;
using System.IO;
using System.Linq;
using WoWFormatLib.FileReaders;
using WoWFormatLib.FileProviders;
using ParpToolbox;
using ParpToolbox.Services.WMO;
using ParpToolbox.Utils;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

// Initialize logging to timestamped file
var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
var logOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{timestamp}");
ConsoleLogger.Initialize(logOutputDir);

if (args.Length == 0)
{
    ConsoleLogger.WriteLine("Usage: parpToolbox <command> --input <file> [flags]\n" +
                      "       or parpToolbox <command> <file> [flags] (positional)\n" +
                      "Commands: wmo | pm4 | pd4 | pm4-region | pm4-export-scene | pm4-test-chunks | pm4-analyze | pm4-analyze-indices | pm4-analyze-unknowns | pm4-test-grouping | pm4-mprl-objects | pm4-analyze-data | pm4-mprr-objects | pm4-mprr-objects-fast | pm4-tile-objects | pm4-raw-geometry | pm4-buildings\n" +
                      "Common flags:\n" +
                      "   --include-collision   Include collision geometry (WMO only)\n" +
                      "   --split-groups        Export each WMO group separately\n" +
                      "   --include-facades     Keep facade/no-draw geometry (WMO)\n" +
                      "   --exportfaces        Write OBJ faces (default: point cloud)\n" +
                      "   --exportchunks      Export each MSUR group to separate OBJ\n" +
                      "   --objects           Export assembled objects by MSUR.IndexCount\n" +
                      "   --bulk-dump        Dump OBJs & CSVs for all groupings\n" +
                      "   --csv-dump         Export all chunk data to CSV files\n" +
                      "   --single-tile       Load only single PM4 tile (default: load region with cross-tile refs)\n" +
                      "PM4 Grouping test flags:\n" +
                      "   --composite         Use composite key (ParentIndex+SurfaceGroupKey+IndexCount)\n" +
                      "   --multi-strategy    Run all experimental grouping strategies with comparison report\n" +
                      "   --mingroup <byte>   Only export groups with byte value >= mingroup");
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
    // 'pm4-region' deprecated: use 'pm4' (region loading is default)
    case "pm4-region":
    case "pm4-export-scene":
    case "pm4-test-chunks":
    case "pm4-analyze":
    case "pm4-mprl-objects":
    case "pm4-analyze-data":
    case "pm4-mprr-objects":
    case "pm4-mprr-objects-fast":
    case "pm4-tile-objects":
    case "pm4-raw-geometry":
    case "pm4-buildings":
    case "pm4-analyze-unknowns":
        case "pm4-test-grouping":
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
bool exportObjects = false;


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
if (args.Contains("--objects") || args.Contains("--indexcount"))
    exportObjects = true;





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


if (command == "pm4-region")
{
    ConsoleLogger.WriteLine($"Parsing PM4 region starting at: {fileInfo.FullName}");
    var adapter = new Pm4Adapter();
    var scene = adapter.LoadRegion(fileInfo.FullName);
    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_region");
    var assembled = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(scene);
    Pm4MsurObjectAssembler.ExportMsurObjects(assembled, scene, outputDir);
    return 0;
}

if (command == "pm4")
{
    // Delegated to dedicated command handler for simplicity
        return ParpToolbox.CliCommands.ExportCommand.Run(args, fileInfo.FullName);
    // Use region loader by default to resolve cross-tile vertex references
    bool useSingleTile = args.Contains("--single-tile");
    Pm4Scene scene;
    if (useSingleTile)
    {
        ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    else
    {
        ConsoleLogger.WriteLine("Region mode active (default) - loading cross-tile references...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }

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
        // Auto-clean previous timestamped dumps to avoid clutter
        foreach (var dir in Directory.GetDirectories(outputDir, "csv_dump_*"))
        {
            try { Directory.Delete(dir, recursive: true); }
            catch (Exception ex) { ConsoleLogger.WriteLine($"Warning: Failed to delete old dump {dir}: {ex.Message}"); }
        }

        var csvDir = Path.Combine(outputDir, "csv_dump");
        ConsoleLogger.WriteLine($"Running CSV dump to {csvDir} ...");
        ParpToolbox.Services.PM4.Pm4CsvDumper.DumpAllChunks(scene, csvDir);
        return 0;
    }




    // Export assembled objects grouped by MSUR.IndexCount if requested
    if (exportObjects)
    {
        ConsoleLogger.WriteLine("Exporting assembled objects by MSUR.IndexCount ...");
        var assembled = ParpToolbox.Services.PM4.Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(scene);
        ParpToolbox.Services.PM4.Pm4MsurObjectAssembler.ExportMsurObjects(assembled, scene, outputDir);
        ConsoleLogger.WriteLine("Assembled object export complete!");
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
else if (command == "pm4-test-grouping")
{
    ConsoleLogger.WriteLine($"Loading PM4 for grouping test: {fileInfo.FullName}");
    // If --region flag is present, load whole region across tiles
    bool loadRegion = args.Contains("--region");
    Pm4Scene scene;
    if (loadRegion)
    {
        ConsoleLogger.WriteLine("Region mode active – loading all tiles in region ...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }
    else
    {
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    // Create an output directory for the results
    string outputDirSuffix = "_msur_grouping";
    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + outputDirSuffix);

    ConsoleLogger.WriteLine("Running optimized MSUR raw fields grouping...");
    ParpToolbox.Services.PM4.Pm4GroupingTester.RunMultipleGroupingStrategies(scene, outputDir, writeFaces: exportFaces);
    ConsoleLogger.WriteLine("MSUR grouping export complete!");
}
else if (command == "pm4-export-scene")
{
    ConsoleLogger.WriteLine($"Parsing PM4 file for complete scene export: {fileInfo.FullName}");
    // Use region loader by default to resolve cross-tile vertex references
    bool useSingleTile = args.Contains("--single-tile");
    Pm4Scene scene;
    if (useSingleTile)
    {
        ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    else
    {
        ConsoleLogger.WriteLine("Region mode active (default) - loading cross-tile references...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_scene");
    ParpToolbox.Services.PM4.Pm4SceneExporter.ExportCompleteScene(scene, outputDir);
    ConsoleLogger.WriteLine("Scene export complete!");
}
else if (command == "pm4-mprl-objects")
{
    ConsoleLogger.WriteLine($"Parsing PM4 file for MPRL-based object grouping: {fileInfo.FullName}");
    // Use region loader by default to resolve cross-tile vertex references
    bool useSingleTile = args.Contains("--single-tile");
    Pm4Scene scene;
    if (useSingleTile)
    {
        ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    else
    {
        ConsoleLogger.WriteLine("Region mode active (default) - loading cross-tile references...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_mprl_objects");
    
    // Group geometry by MPRL placements (actual building objects)
    var buildingObjects = ParpToolbox.Services.PM4.Pm4MprlObjectGrouper.GroupByMprlPlacements(scene);
    
    // Export each building object as separate OBJ file
    ParpToolbox.Services.PM4.Pm4MprlObjectGrouper.ExportBuildingObjects(buildingObjects, scene, outputDir);
    
    ConsoleLogger.WriteLine($"MPRL object export complete! Exported {buildingObjects.Count} building objects");
}
else if (command == "pm4-analyze")
{
    // Delegated to dedicated command handler for simplicity
    return ParpToolbox.CliCommands.AnalyzeCommand.Run(args, fileInfo.FullName);
}


else if (command == "pm4-analyze-data")
{
    ConsoleLogger.WriteLine($"Analyzing PM4 data structure: {fileInfo.FullName}");
    // Use region loader by default to get complete data
    bool useSingleTile = args.Contains("--single-tile");
    Pm4Scene scene;
    if (useSingleTile)
    {
        ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    else
    {
        ConsoleLogger.WriteLine("Region mode active (default) - loading cross-tile references...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_data_analysis");
    
    // Analyze PM4 data structure and relationships
    ParpToolbox.Services.PM4.Pm4DataAnalyzer.AnalyzeDataStructure(scene, outputDir);
    
    ConsoleLogger.WriteLine("PM4 data analysis complete!");
}
else if (command == "pm4-mprr-objects")
{
    ConsoleLogger.WriteLine($"Parsing PM4 file for MPRR-based hierarchical object grouping: {fileInfo.FullName}");
    // Use region loader by default to resolve cross-tile vertex references
    bool useSingleTile = args.Contains("--single-tile");
    Pm4Scene scene;
    if (useSingleTile)
    {
        ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    else
    {
        ConsoleLogger.WriteLine("Region mode active (default) - loading cross-tile references...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_mprr_objects");
    
    // Assemble objects using MPRR hierarchical grouping (sentinel-based)
    var hierarchicalObjects = ParpToolbox.Services.PM4.Pm4HierarchicalObjectAssembler.AssembleHierarchicalObjects(scene);
    
    // Export each hierarchical object as separate OBJ file
    ParpToolbox.Services.PM4.Pm4HierarchicalObjectAssembler.ExportHierarchicalObjects(hierarchicalObjects, scene, outputDir);
    
    ConsoleLogger.WriteLine($"MPRR hierarchical object export complete! Exported {hierarchicalObjects.Count} building objects");
}
else if (command == "pm4-mprr-objects-fast")
{
    ConsoleLogger.WriteLine($"Parsing PM4 file for OPTIMIZED MPRR-based hierarchical object grouping: {fileInfo.FullName}");
    // Use region loader by default to resolve cross-tile vertex references
    bool useSingleTile = args.Contains("--single-tile");
    Pm4Scene scene;
    if (useSingleTile)
    {
        ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
        var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = loader.Load(fileInfo.FullName);
    }
    else
    {
        ConsoleLogger.WriteLine("Region mode active (default) - loading cross-tile references...");
        var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
        scene = adapter.LoadRegion(fileInfo.FullName);
    }

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_mprr_objects_fast");
    
    // Assemble objects using MPRR hierarchical grouping (sentinel-based)
    var hierarchicalObjects = ParpToolbox.Services.PM4.Pm4HierarchicalObjectAssembler.AssembleHierarchicalObjects(scene);
    
    // Parse command line arguments for optimization settings
    int maxObjects = 100;
    int maxTriangles = 100000;
    bool useParallel = true;
    
    foreach (var arg in args)
    {
        if (arg.StartsWith("--max-objects="))
            int.TryParse(arg.Substring("--max-objects=".Length), out maxObjects);
        else if (arg.StartsWith("--max-triangles="))
            int.TryParse(arg.Substring("--max-triangles=".Length), out maxTriangles);
        else if (arg == "--no-parallel")
            useParallel = false;
    }
    
    // Export using optimized exporter with limits
    ParpToolbox.Services.PM4.Pm4OptimizedObjectExporter.ExportOptimized(
        hierarchicalObjects, scene, outputDir, maxObjects, maxTriangles, useParallel);
    
    ConsoleLogger.WriteLine($"Optimized MPRR hierarchical object export complete!");
}
else if (command == "pm4-tile-objects")
{
    ConsoleLogger.WriteLine($"Parsing PM4 file for TILE-BASED object export: {fileInfo.FullName}");
    
    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_tile_objects");
    
    // Parse command line arguments for tile-based export settings
    int maxObjectsPerTile = 50;
    
    foreach (var arg in args)
    {
        if (arg.StartsWith("--max-objects="))
            int.TryParse(arg.Substring("--max-objects=".Length), out maxObjectsPerTile);
    }
    
    // Export using tile-based approach (much faster)
    ParpToolbox.Services.PM4.Pm4TileBasedExporter.ExportTileBased(
        inputFile, outputDir, maxObjectsPerTile);
    
    ConsoleLogger.WriteLine($"Tile-based PM4 object export complete!");
}
else if (command == "pm4-raw-geometry")
{
    ConsoleLogger.WriteLine($"Exporting RAW PM4 geometry (no grouping): {fileInfo.FullName}");
    
    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_raw_geometry");
    
    // Export raw geometry to understand what PM4 actually contains
    ParpToolbox.Services.PM4.Pm4RawGeometryExporter.ExportRawGeometry(
        inputFile, outputDir);
    
    ConsoleLogger.WriteLine($"Raw PM4 geometry export complete!");
}
else if (command == "pm4-buildings")
{
    ConsoleLogger.WriteLine($"Exporting PM4 buildings (surface group objects): {fileInfo.FullName}");
    
    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_buildings");
    
    // Export buildings using correct surface group logic
    ParpToolbox.Services.PM4.Pm4SurfaceGroupExporter.ExportSurfaceGroups(
        inputFile, outputDir);
    
    ConsoleLogger.WriteLine($"PM4 building export complete!");
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
else if (command == "pm4-analyze-unknowns")
{
    // PM4 Unknown Field Analysis command - correlate unknown fields
    if (string.IsNullOrEmpty(inputFile))
    {
        ConsoleLogger.WriteLine("Error: Input PM4 file required for pm4-analyze-unknowns command");
        ConsoleLogger.Close();
        return 1;
    }

    if (!fileInfo.Exists)
    {
        ConsoleLogger.WriteLine($"Error: File not found: {inputFile}");
        ConsoleLogger.Close();
        return 1;
    }

    ConsoleLogger.WriteLine($"Running unknown field analysis on: {fileInfo.FullName}");
    var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
    var scene = loader.Load(fileInfo.FullName);
    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile) + "_unknowns");
    ParpToolbox.Services.PM4.Pm4UnknownFieldAnalyzer.AnalyzeUnknownFields(scene, outputDir);
    ConsoleLogger.WriteLine("Unknown field analysis complete!");
    ConsoleLogger.Close();
    return 0;
}
#if DEBUG_ANALYZER
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
#endif

ConsoleLogger.Close();
return 0;
