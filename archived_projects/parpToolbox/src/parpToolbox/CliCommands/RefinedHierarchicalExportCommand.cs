using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands;

/// <summary>
/// CLI command for exporting PM4 data using the scene graph traversal exporter.
/// </summary>
internal class RefinedHierarchicalExportCommand
{
    public static Command CreateCommand()
    {
        var command = new Command("refined-hierarchical-export", "Export PM4 data using scene graph traversal object grouping");
        
        var inputOption = new Option<string>(
            "--input",
            "Path to the input PM4 file or directory")
        { IsRequired = true };
        command.AddOption(inputOption);
        
        var outputOption = new Option<string>(
            "--output",
            "Output directory for OBJ files")
        { IsRequired = true };
        command.AddOption(outputOption);
        
        command.SetHandler(async (inputPath, outputPath) =>
        {
            await Run(inputPath, outputPath);
        }, inputOption, outputOption);
        
        return command;
    }
    
    public static Task Run(string inputPath, string outputPath)
    {
        return Task.Run(() =>
        {
            try
            {
                var pm4Files = new List<string>();
                if (File.Exists(inputPath))
                {
                    pm4Files.Add(inputPath);
                }
                else if (Directory.Exists(inputPath))
                {
                    pm4Files.AddRange(Directory.GetFiles(inputPath, "*.pm4", SearchOption.AllDirectories));
                }

                if (!pm4Files.Any())
                {
                    Console.WriteLine("No PM4 files found at the specified path.");
                    return;
                }

                var exportPath = ProjectOutput.CreateOutputDirectory(outputPath);
                Console.WriteLine($"Found {pm4Files.Count} PM4 files. Exporting to: {exportPath}");

                var exporter = new Pm4SceneGraphExporter();
                var adapter = new Pm4Adapter();
                var options = new Pm4LoadOptions { VerboseLogging = false }; // Reduce noise for per-tile loading

                foreach (var file in pm4Files)
                {
                    try
                    {
                        Console.WriteLine($"--- Processing tile: {Path.GetFileName(file)} ---");
                        var scene = adapter.LoadRegion(file, options);
                        exporter.ExportPm4SceneGraph(scene, exportPath);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"ERROR: Failed to process tile {Path.GetFileName(file)}: {ex.Message}");
                    }
                }

                Console.WriteLine($"\nScene graph export completed. Objects exported to: {exportPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during scene graph export: {ex.Message}");
                Console.WriteLine(ex.StackTrace ?? "No stack trace available");
            }
        });
    }
}
