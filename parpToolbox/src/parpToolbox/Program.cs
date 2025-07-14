using System;
using System.IO;
using System.Linq;
using WoWFormatLib.FileReaders;
using WoWFormatLib.FileProviders;
using ParpToolbox;
using ParpToolbox.Services.WMO;

if (args.Length == 0)
{
    Console.WriteLine("Usage: parpToolbox <command> --input <file> [flags]\n" +
                      "       or parpToolbox <command> <file> [flags] (positional)\n" +
                      "Commands: wmo | pm4 | pd4\n" +
                      "Common flags:\n" +
                      "   --include-collision   Include collision geometry (WMO only)\n" +
                      "   --split-groups        Export each WMO group separately\n" +
                      "   --include-facades     Keep facade/no-draw geometry (WMO)");
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
        // handled below
        break;
    default:
        Console.WriteLine($"Error: Unknown command '{command}'");
        return 1;
}

string inputFile = null;
bool includeCollision = false;
bool splitGroups = false;
bool includeFacades = false;
// Detect optional flags
if (args.Contains("--include-collision"))
    includeCollision = true;
if (args.Contains("--split-groups"))
    splitGroups = true;
if (args.Contains("--include-facades") || args.Contains("--include-no-draw"))
    includeFacades = true;

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
    Console.WriteLine("Export complete!");
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
    Console.WriteLine($"Parsing PM4 file: {fileInfo.FullName}");
    var loader = new ParpToolbox.Services.PM4.Pm4Adapter();
    var scene = loader.Load(fileInfo.FullName);

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile));
    var outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputFile) + ".obj");
    Console.WriteLine($"Exporting OBJ to {outputFile}...");
    ParpToolbox.Services.PM4.Pm4ObjExporter.Export(scene, outputFile);
    Console.WriteLine("Export complete!");
}

return 0;
