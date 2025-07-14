using System;

Console.WriteLine($"Received {args.Length} arguments:");
for(int i = 0; i < args.Length; i++)
{
    Console.WriteLine($"  arg[{i}] = {args[i]}");
}
using System.IO;
using WoWFormatLib.FileReaders;
using WoWFormatLib.FileProviders;

if (args.Length == 0)
{
    Console.WriteLine("Usage: parpToolbox <command> [options]");
    return 1;
}

var command = args[0];
if (command.ToLower() != "wmo")
{
    Console.WriteLine($"Error: Unknown command '{command}'");
    return 1;
}

string inputFile = null;
var localProvider = new LocalFileProvider(".");
FileProvider.SetProvider(localProvider, "local");
FileProvider.SetDefaultBuild("local");

var inputIndex = Array.IndexOf(args, "--input");
if (inputIndex == -1) inputIndex = Array.IndexOf(args, "-i");

if (inputIndex != -1 && inputIndex + 1 < args.Length)
{
    inputFile = args[inputIndex + 1];
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

Console.WriteLine($"Processing WMO file: {fileInfo.FullName}");

try
{
    var reader = new WMOReader();
    var wmo = reader.LoadWMO(fileInfo.FullName);
    
    Console.WriteLine($"Successfully loaded WMO with {wmo.materials.Length} materials and {wmo.group.Length} groups.");

    var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputFile));
    var outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputFile) + ".obj");

    Console.WriteLine($"Exporting to {outputFile}...");
    ObjExporter.Export(wmo, outputFile);
    Console.WriteLine("Export complete!");
}
catch (Exception e)
{
    Console.WriteLine($"An error occurred: {e.Message}");
    Console.WriteLine(e.StackTrace);
    return 1;
}

return 0;
