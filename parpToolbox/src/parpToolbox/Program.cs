using System;
using System.IO;
using System.Linq;
using System.CommandLine;
using WoWFormatLib.FileReaders;
using WoWFormatLib.FileProviders;
using ParpToolbox;
using ParpToolbox.CliCommands;
using ParpToolbox.Utils;
using ParpToolbox.Services.WMO;
using ParpToolbox.Formats.PM4;

// Initialize logging to timestamped file
var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
var logOutputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", $"session_{timestamp}");
ConsoleLogger.Initialize(logOutputDir);

if (args.Length == 0)
{
    ConsoleLogger.WriteLine("Usage: parpToolbox <command> --input <file> [flags]\n" +
                      "       or parpToolbox <command> <file> [flags] (positional)\n" +
                      "Commands:\n" +
                      "  analyze    Analyze PM4/PD4 files and generate reports\n" +
                      "  export     Export PM4/PD4/WMO files to OBJ/MTL\n" +
                      "  test       Run validation tests against real data\n" +
                      "  wmo        Export WMO files to OBJ/MTL\n" +
                      "  chunk-test Run PM4 chunk combination tests to understand object grouping\n" +
                      "  refined-hierarchical-export Export PM4 using refined hierarchical object grouping\n" +
                      "  pm4-analyze-fields    Analyze PM4 chunk field distributions and correlations\n" +
                      "  pm4-export-wmo-inspired Export PM4 objects using WMO organizational logic\n" +
                      "  pm4-export-spatial-clustering Export PM4 objects using spatial clustering (with region loading)\n" +
                      "  pm4-export-scene-graph Export PM4 using scene graph traversal (BREAKTHROUGH APPROACH)\n" +
                      "  pm4-wmo-match         Perform PM4-to-WMO spatial correlation and matching analysis\n" +
                      "  test       Run regression tests\n" +
                      "\nCommon flags:\n" +
                       "   --input <file>      Input file path\n" +
                      "   --single-tile       Load only single PM4 tile (default: load region)\n" +
                      "   --csv-dump          Export chunk data to CSV for analysis");
    ConsoleLogger.Close();
    return 1;
}

var command = args.Length > 0 ? args[0] : string.Empty;

// Handle self-contained commands first
if (command == "test")
{
    var exitCode = ParpToolbox.CliCommands.TestCommand.Run(args, string.Empty); // input path not used
    ConsoleLogger.Close();
    return exitCode;
}

// Global help handler
if (command == "help" || command == "--help" || command == "-h")
{
    ConsoleLogger.WriteLine("Usage: parpToolbox <command> --input <file> [flags]" +
                      "       or parpToolbox <command> <file> [flags] (positional)\n" +
                      "Primary commands: analyze | export | test | wmo\n" +
                      "Use '<command> --help' for specific flags.\n\n" +
                      "Examples:\n" +
                      "  parpToolbox export development_00_00.pm4 --faces\n" +
                      "  parpToolbox analyze development_00_00.pm4 --report\n" +
                      "  parpToolbox test --analyze-chunks\n" +
                      "  parpToolbox wmo dalaran.wmo --split-groups\n" +
                      "\nLegacy commands are deprecated but supported with warnings.");
    ConsoleLogger.Close();
    return 1;
}

// Parse common arguments for commands that require an input file
string? inputFile = null;
for (int i = 1; i < args.Length; i++)
{
    if (args[i] == "--input" && i + 1 < args.Length)
    {
        inputFile = args[i + 1];
        i++;
    }
    else if (!args[i].StartsWith("--") && inputFile == null)
    {
        inputFile = args[i];
    }
}

var fileInfo = new FileInfo(inputFile ?? string.Empty);

// Command-specific argument parsing
var includeFacades = args.Contains("--facades");
var splitGroups = args.Contains("--split-groups");
var includeCollision = args.Contains("--collision");

try
{
    switch (command)
    {
        case "wmo":
            if (string.IsNullOrEmpty(inputFile))
            {
                Console.WriteLine("Error: --input <file> is required for the wmo command.");
                return 1;
            }
            
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
            break;
            
        case "analyze":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input file required for analyze command");
                return 1;
            }
            return ParpToolbox.CliCommands.AnalyzeCommand.Run(args, fileInfo.FullName);
            
        case "export":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input file required for export command");
                return 1;
            }
            return ParpToolbox.CliCommands.ExportCommand.Run(args, fileInfo.FullName).GetAwaiter().GetResult();
            
        case "test":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input file required for test command");
                return 1;
            }
            return ParpToolbox.CliCommands.TestCommand.Run(args, fileInfo.FullName);
            
        // Legacy command support - route to unified handlers with deprecation warnings
        case "pm4-export":
        case "pm4":
        case "pm4-region":
        case "pm4-export-scene":
        case "pm4-mprl-objects":
        case "pm4-mprr-objects":
        case "pm4-mprr-objects-fast":
        case "pm4-tile-objects":
        case "pm4-raw-geometry":
        case "pm4-buildings":
            ConsoleLogger.WriteLine($"Note: '{command}' is deprecated. Use 'export' instead.");
            return ParpToolbox.CliCommands.ExportCommand.Run(args, fileInfo.FullName).GetAwaiter().GetResult();
            
        case "pm4-analyze":
        case "pm4-analyze-data":
        case "pm4-analyze-unknowns":
        case "pm4-test-grouping":
            ConsoleLogger.WriteLine($"Note: '{command}' is deprecated. Use 'analyze' instead.");
            return ParpToolbox.CliCommands.AnalyzeCommand.Run(args, fileInfo.FullName);
            
        case "pm4-test":
        case "pm4-test-chunks":
            ConsoleLogger.WriteLine($"Note: '{command}' is deprecated. Use 'test' instead.");
            return ParpToolbox.CliCommands.TestCommand.Run(args, fileInfo.FullName);
            
        case "mprl-pattern-analysis":
        case "mpa":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for MPRL pattern analysis");
                return 1;
            }
            new ParpToolbox.CliCommands.MprlPatternAnalysisCommand().RunAsync(fileInfo.FullName).GetAwaiter().GetResult();
            return 0;
            
        case "mslk-pattern-analysis":
        case "mla":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for MSLK pattern analysis");
                return 1;
            }
            new ParpToolbox.CliCommands.MslkPatternAnalysisCommand().RunAsync(fileInfo.FullName).GetAwaiter().GetResult();
            return 0;
            
        case "quality-analysis":
        case "qa":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for quality analysis");
                return 1;
            }
            return ParpToolbox.CliCommands.QualityAnalysisCommand.Run(args, fileInfo.FullName).GetAwaiter().GetResult();
            
        case "global-mesh-analysis":
        case "mprl-analysis":
        case "mfa":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for MPRL field analysis");
                return 1;
            }
            var mprlAnalyzer = new ParpToolbox.Services.PM4.Database.Pm4MprlFieldAnalyzer(
                fileInfo.FullName, 
                ProjectOutput.CreateOutputDirectory("mprl_analysis"));
            await mprlAnalyzer.AnalyzeAsync();
            break;
            
        case "gma":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for global mesh analysis");
                return 1;
            }
            ParpToolbox.Services.PM4.Database.Pm4GlobalMeshAnalyzer.AnalyzeGlobalMeshLinkage(fileInfo.FullName).GetAwaiter().GetResult();
            return 0;
            
        case "chunk-validation":
        case "cv":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: PM4 file required for chunk validation");
                return 1;
            }
            
            // Load PM4 scene for chunk validation
            var chunkAdapter = new ParpToolbox.Services.PM4.Pm4Adapter();
            var chunkLoadOptions = new ParpToolbox.Services.PM4.Pm4LoadOptions 
            { 
                CaptureRawData = true,
                ValidateData = true,
                VerboseLogging = true
            };
            
            ParpToolbox.Formats.PM4.Pm4Scene validationScene;
            if (inputFile.Contains("_00_00") || inputFile.Contains("_000"))
            {
                validationScene = chunkAdapter.LoadRegion(inputFile, chunkLoadOptions);
            }
            else
            {
                validationScene = chunkAdapter.Load(inputFile, chunkLoadOptions);
            }
            
            var validator = new ParpToolbox.Services.PM4.Pm4ChunkValidationTool();
            var chunkOutputDir = ProjectOutput.CreateOutputDirectory("chunk_validation");
            var validationReport = validator.ValidateMsurChunk(validationScene, chunkOutputDir);
            
            ConsoleLogger.WriteLine($"Chunk validation complete. Report saved to: {chunkOutputDir}");
            return 0;
            
        case "surface-encoding-analysis":
        case "sea":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for surface encoding analysis");
                return 1;
            }
            ParpToolbox.Services.PM4.Database.Pm4SurfaceEncodingAnalyzer.AnalyzeSurfaceEncodingPatterns(fileInfo.FullName).GetAwaiter().GetResult();
            return 0;
            
        case "bounds-decoder":
        case "bd":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for bounds decoder analysis");
                return 1;
            }
            var boundsDecoder = new ParpToolbox.Services.PM4.Database.Pm4SurfaceBoundsDecoder(
                fileInfo.FullName, 
                ProjectOutput.CreateOutputDirectory("bounds_decoder"));
            await boundsDecoder.AnalyzeAsync();
            return 0;
            
        case "hierarchical-decoder":
        case "hd":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Database file required for hierarchical container decoder");
                return 1;
            }
            var hierarchicalDecoder = new ParpToolbox.Services.PM4.Database.Pm4HierarchicalContainerDecoder(
                fileInfo.FullName, 
                ProjectOutput.CreateOutputDirectory("hierarchical_decoder"));
            await hierarchicalDecoder.AnalyzeAsync();
            return 0;
            
        case "chunk-test":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: PM4 file required for chunk combination testing");
                return 1;
            }
            return await ParpToolbox.CliCommands.ChunkCombinationTestCommand.Run(args, fileInfo.FullName);
            
        case "refined-hierarchical-export":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: PM4 file required for refined hierarchical export");
                return 1;
            }
            // For now, we'll pass an empty string for outputPath since the command handles its own output directory
            await ParpToolbox.CliCommands.RefinedHierarchicalExportCommand.Run(fileInfo.FullName, "");
            return 0;

        case "pm4-export-json":
            var outputArg = args.FirstOrDefault(a => a.StartsWith("--output"))?.Split('=')[1];
            if (string.IsNullOrEmpty(inputFile) || string.IsNullOrEmpty(outputArg))
            {
                ConsoleLogger.WriteLine("Error: --input and --output are required for pm4-export-json command");
                return 1;
            }
            var jsonCommand = new ParpToolbox.CliCommands.Pm4ExportJsonCommand();
            await jsonCommand.InvokeAsync(new string[] { "--input", inputFile, "--output", outputArg });
            return 0;

        case "analyze-pm4-keys":
            var analyzeKeysCommand = ParpToolbox.CliCommands.AnalyzePm4KeysCommand.CreateCommand();
            return await analyzeKeysCommand.InvokeAsync(args);

        case "export-pm4-dataweb":
            var dataWebExportCommand = ParpToolbox.CliCommands.ExportPm4DataWebCommand.CreateCommand();
            return await dataWebExportCommand.InvokeAsync(args);

        case "diagnose-linkage":
            var diagnoseLinkageCommand = ParpToolbox.CliCommands.DiagnoseLinkageCommand.CreateCommand();
            return await diagnoseLinkageCommand.InvokeAsync(args);

        case "analyze-surfacerefindex":
            var surfaceRefAnalysisCommand = ParpToolbox.CliCommands.AnalyzeSurfaceRefIndexCommand.CreateCommand();
            return await surfaceRefAnalysisCommand.InvokeAsync(args);

        case "analyze-mprl-fields":
            var mprlFieldsCommand = ParpToolbox.CliCommands.AnalyzeMprlFieldsCommand.CreateCommand();
            return await mprlFieldsCommand.InvokeAsync(args);

        case "analyze-mslk-fields":
            var mslkFieldsCommand = ParpToolbox.CliCommands.AnalyzeMslkFieldsCommand.CreateCommand();
            return await mslkFieldsCommand.InvokeAsync(args);

        case "test-surfaceref-building-grouping":
            var surfaceRefGroupingCommand = ParpToolbox.CliCommands.TestSurfaceRefBuildingGroupingCommand.CreateCommand();
            return await surfaceRefGroupingCommand.InvokeAsync(args);

        case "spatial-clustering":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input PM4 file required for spatial clustering export");
                return 1;
            }
            
            var spatialOutputDir = args.Length > 2 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", "spatial_clustering_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            
            var spatialClusteringCommand = new ParpToolbox.CliCommands.Pm4ExportSpatialClusteringCommand();
            spatialClusteringCommand.Execute(inputFile, spatialOutputDir);
            return 0;

        case "diagnose-tile-contamination":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input PM4 file required for tile contamination diagnostic");
                return 1;
            }
            
            var contaminationOutputDir = args.Length > 2 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", "tile_contamination_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            
            var tileContaminationCommand = new ParpToolbox.CliCommands.DiagnoseTileContaminationCommand();
            tileContaminationCommand.Execute(inputFile, contaminationOutputDir);
            return 0;

        case "diagnostic-spatial-clustering":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input PM4 file required for diagnostic spatial clustering export");
                return 1;
            }
            
            var diagnosticOutputDir = args.Length > 2 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", "diagnostic_spatial_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            
            var diagnosticSpatialCommand = new ParpToolbox.CliCommands.DiagnosticSpatialClusteringCommand();
            diagnosticSpatialCommand.Execute(inputFile, diagnosticOutputDir);
            return 0;

        case "enhanced-diagnostic-spatial":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input PM4 file required for enhanced diagnostic spatial clustering export");
                return 1;
            }
            
            var enhancedDiagnosticOutputDir = args.Length > 2 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", "enhanced_diagnostic_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            
            var enhancedDiagnosticCommand = new ParpToolbox.CliCommands.EnhancedDiagnosticSpatialCommand();
            enhancedDiagnosticCommand.Execute(inputFile, enhancedDiagnosticOutputDir);
            return 0;

        case "fixed-spatial-clustering":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: Input PM4 file required for fixed spatial clustering export");
                return 1;
            }
            
            var fixedSpatialOutputDir = args.Length > 2 ? args[2] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", "fixed_spatial_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            
            var fixedSpatialCommand = new ParpToolbox.CliCommands.FixedSpatialClusteringCommand();
            fixedSpatialCommand.Execute(inputFile, fixedSpatialOutputDir);
            return 0;

        case "pm4-wmo-match":
            if (args.Length < 3)
            {
                ConsoleLogger.WriteLine("Error: PM4 file and WMO file required for PM4-WMO matching");
                ConsoleLogger.WriteLine("Usage: parpToolbox pm4-wmo-match <pm4-file> <wmo-file> [output-dir]");
                return 1;
            }

            var pm4File = args[1];
            var wmoFile = args[2];
            var matchOutputDir = args.Length > 3 ? args[3] : Path.Combine(Directory.GetCurrentDirectory(), "project_output", "pm4_wmo_match_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));

            if (!File.Exists(pm4File))
            {
                ConsoleLogger.WriteLine($"Error: PM4 file not found: {pm4File}");
                return 1;
            }

            if (!File.Exists(wmoFile))
            {
                ConsoleLogger.WriteLine($"Error: WMO file not found: {wmoFile}");
                return 1;
            }

            var pm4WmoMatchCommand = new ParpToolbox.CliCommands.Pm4WmoMatchCommand();
            pm4WmoMatchCommand.Execute(pm4File, wmoFile, matchOutputDir);
            return 0;

        case "pm4-analyze-fields":
            if (string.IsNullOrEmpty(inputFile))
            {
                ConsoleLogger.WriteLine("Error: PM4 file required for field analysis");
                return 1;
            }
            // Create analyzer and run analysis
            var analyzer = new ParpToolbox.Services.PM4.Pm4ChunkFieldAnalyzer();
            var adapter = new ParpToolbox.Services.PM4.Pm4Adapter();
            var scene = adapter.Load(inputFile);
            
            // Create timestamped output directory
            var analysisTimestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var analysisOutputDir = Path.Combine("project_output", $"pm4_field_analysis_{analysisTimestamp}");
            Directory.CreateDirectory(analysisOutputDir);
            
            ConsoleLogger.WriteLine($"Running PM4 chunk field analysis on: {inputFile}");
            ConsoleLogger.WriteLine($"Output directory: {analysisOutputDir}");
            // TODO: Execute analysis command here
            return 0;
            
        case "pm4-export-spatial-clustering":
            // Handle both --output=value and --output value formats
            string spatialOutputArg = null;
            var outputFlagIndex = Array.FindIndex(args, a => a == "--output" || a.StartsWith("--output="));
            if (outputFlagIndex >= 0)
            {
                if (args[outputFlagIndex].StartsWith("--output="))
                {
                    spatialOutputArg = args[outputFlagIndex].Split('=')[1];
                }
                else if (outputFlagIndex + 1 < args.Length)
                {
                    spatialOutputArg = args[outputFlagIndex + 1];
                }
            }
            
            if (string.IsNullOrEmpty(inputFile) || string.IsNullOrEmpty(spatialOutputArg))
            {
                ConsoleLogger.WriteLine("Error: --input and --output are required for pm4-export-spatial-clustering command");
                return 1;
            }
            var spatialCommand = new ParpToolbox.CliCommands.Pm4ExportSpatialClusteringCommand();
            new Pm4ExportSpatialClusteringCommand().Execute(inputFile, spatialOutputArg);
            break;
            
        case "pm4-export-scene-graph":
            string sceneGraphInputPath = null;
            string sceneGraphOutputPath = null;
            
            // Parse arguments for scene graph export command
            for (int i = 1; i < args.Length; i++) // Start from 1 to skip command name
            {
                if (args[i] == "--input" && i + 1 < args.Length)
                {
                    sceneGraphInputPath = args[i + 1];
                    i++; // Skip next argument as it's the value
                }
                else if (args[i].StartsWith("--input="))
                {
                    sceneGraphInputPath = args[i].Substring("--input=".Length);
                }
                else if (args[i] == "--output" && i + 1 < args.Length)
                {
                    sceneGraphOutputPath = args[i + 1];
                    i++; // Skip next argument as it's the value
                }
                else if (args[i].StartsWith("--output="))
                {
                    sceneGraphOutputPath = args[i].Substring("--output=".Length);
                }
                else if (!args[i].StartsWith("--"))
                {
                    // If no flags, assume first non-flag argument is input
                    if (sceneGraphInputPath == null)
                        sceneGraphInputPath = args[i];
                }
            }
            
            if (string.IsNullOrEmpty(sceneGraphInputPath))
            {
                ConsoleLogger.WriteLine("Error: --input is required for pm4-export-scene-graph command");
                return 1;
            }
            
            // Use project output if no output specified
            if (string.IsNullOrEmpty(sceneGraphOutputPath))
            {
                sceneGraphOutputPath = Path.Combine(Directory.GetCurrentDirectory(), "output", DateTime.Now.ToString("yyyyMMdd_HHmmss"));
                ConsoleLogger.WriteLine($"Using timestamped project output: {sceneGraphOutputPath}");
            }
            
            // Execute scene graph export
            var sceneGraphAdapter = new ParpToolbox.Services.PM4.Pm4Adapter();
            var pm4Scene = sceneGraphAdapter.LoadRegion(sceneGraphInputPath);
            
            if (pm4Scene == null)
            {
                ConsoleLogger.WriteLine("Failed to load PM4 scene");
                return 1;
            }
            
            var sceneGraphExporter = new ParpToolbox.Services.PM4.Pm4SceneGraphExporter();
            sceneGraphExporter.ExportPm4SceneGraph(pm4Scene, sceneGraphOutputPath);
            break;

        case "pm4-analyze-data-banding":
            string bandingInputPath = null;
            string bandingOutputPath = null;
            
            // Parse arguments for data banding command
            for (int i = 1; i < args.Length; i++) // Start from 1 to skip command name
            {
                if (args[i] == "--input" && i + 1 < args.Length)
                {
                    bandingInputPath = args[i + 1];
                    i++; // Skip next argument as it's the value
                }
                else if (args[i].StartsWith("--input="))
                {
                    bandingInputPath = args[i].Substring("--input=".Length);
                }
                else if (args[i] == "--output" && i + 1 < args.Length)
                {
                    bandingOutputPath = args[i + 1];
                    i++; // Skip next argument as it's the value
                }
                else if (args[i].StartsWith("--output="))
                {
                    bandingOutputPath = args[i].Substring("--output=".Length);
                }
                else if (!args[i].StartsWith("--"))
                {
                    // If no flags, assume first non-flag argument is input
                    if (bandingInputPath == null)
                        bandingInputPath = args[i];
                }
            }
            
            if (string.IsNullOrEmpty(bandingInputPath))
            {
                ConsoleLogger.WriteLine("Error: --input is required for pm4-analyze-data-banding command");
                return 1;
            }
            
            if (string.IsNullOrEmpty(bandingOutputPath))
            {
                ConsoleLogger.WriteLine("Error: --output is required for pm4-analyze-data-banding command");
                return 1;
            }
            
            await new Pm4AnalyzeDataBandingCommand().RunAsync(bandingInputPath, bandingOutputPath);
            break;

        case "pm4-export-4d-objects":
            string export4DInputPath = null;
            string export4DOutputPath = null;
            
            // Parse arguments for 4D export command
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "--input" && i + 1 < args.Length)
                {
                    export4DInputPath = args[i + 1];
                    i++;
                }
                else if (args[i].StartsWith("--input="))
                {
                    export4DInputPath = args[i].Substring("--input=".Length);
                }
                else if (args[i] == "--output" && i + 1 < args.Length)
                {
                    export4DOutputPath = args[i + 1];
                    i++;
                }
                else if (args[i].StartsWith("--output="))
                {
                    export4DOutputPath = args[i].Substring("--output=".Length);
                }
                else if (!args[i].StartsWith("--"))
                {
                    if (export4DInputPath == null)
                        export4DInputPath = args[i];
                }
            }
            
            if (string.IsNullOrEmpty(export4DInputPath))
            {
                ConsoleLogger.WriteLine("Error: --input is required for pm4-export-4d-objects command");
                return 1;
            }
            
            if (string.IsNullOrEmpty(export4DOutputPath))
            {
                ConsoleLogger.WriteLine("Error: --output is required for pm4-export-4d-objects command");
                return 1;
            }
            
            await new Pm4Export4DObjectsCommand().RunAsync(export4DInputPath, export4DOutputPath);
            break;

        default:
            Console.WriteLine($"Unknown command: {command}");
            return 1;
    }
}
catch (Exception e)
{
    Console.WriteLine($"An error occurred: {e.Message}");
    Console.WriteLine(e.StackTrace);
    return 1;
}
finally
{
    ConsoleLogger.Close();
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
