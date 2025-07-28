using System;
using System.IO;
using System.Linq;
using System.CommandLine;
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
                      "Commands:\n" +
                      "  analyze    Analyze PM4/PD4 files and generate reports\n" +
                      "  export     Export PM4/PD4/WMO files to OBJ/MTL\n" +
                      "  test       Run validation tests against real data\n" +
                      "  wmo        Export WMO files to OBJ/MTL\n" +
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
            
        default:
            ConsoleLogger.WriteLine($"Error: Unknown command '{command}'");
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
