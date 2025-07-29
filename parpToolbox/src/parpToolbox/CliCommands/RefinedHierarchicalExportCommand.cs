using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.PM4;

namespace ParpToolbox.CliCommands;

/// <summary>
/// CLI command for exporting PM4 data using the refined hierarchical object assembler.
/// </summary>
internal class RefinedHierarchicalExportCommand
{
    public static Command CreateCommand()
    {
        var command = new Command("refined-hierarchical-export", "Export PM4 data using refined hierarchical object grouping");
        
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
    
    public static async Task Run(string inputPath, string outputPath)
    {
        try
        {
            Console.WriteLine($"Loading PM4 scene from: {inputPath}");
            
            // Load the PM4 scene
            var options = new Pm4LoadOptions { VerboseLogging = true };
            var adapter = new Pm4Adapter();
            var scene = adapter.LoadRegion(inputPath, options);
            
            Console.WriteLine($"Scene loaded successfully:");
            Console.WriteLine($"  Vertices: {scene.Vertices.Count}");
            Console.WriteLine($"  Indices: {scene.Indices.Count}");
            Console.WriteLine($"  Surfaces: {scene.Surfaces.Count}");
            Console.WriteLine($"  Links: {scene.Links.Count}");
            Console.WriteLine($"  Placements: {scene.Placements.Count}");
            Console.WriteLine($"  Properties: {scene.Properties.Count}");
            
            // Export objects using the streaming method to avoid building all objects in memory
            var exportPath = Path.Combine(ProjectOutput.CreateOutputDirectory("refined_hierarchical_export"), "objects");
            Pm4RefinedHierarchicalObjectAssembler.StreamRefinedHierarchicalObjects(scene, exportPath);
            
            Console.WriteLine($"Refined hierarchical export completed. Objects exported to: {exportPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error during refined hierarchical export: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
