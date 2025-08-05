using System.CommandLine;
using System.CommandLine.Invocation;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;
using System.IO;
using System.Reflection;

namespace ParpToolbox.CliCommands;

public class AnalyzePm4KeysCommand
{
    private static readonly Argument<string> InputPathArgument = new("input-path", "Path to the input PM4 file or directory.");
    private static readonly Argument<string> OutputPathArgument = new("output-path", "Path to the output directory for analysis results.");
    private static readonly Option<string> ChunkTypeOption = new("--chunk-type", () => "MSLK", "The 4CC of the chunk to analyze (e.g., MSLK, MPRL).");
    private static readonly Option<string> KeyFieldOption = new("--key-field", () => "ParentIndex", "The name of the key field to analyze.");

    public static Command CreateCommand()
    {
        var command = new Command("analyze-pm4-keys", "Analyzes the structure and relationships of key fields in PM4 files.");
        command.AddArgument(InputPathArgument);
        command.AddArgument(OutputPathArgument);
        command.AddOption(ChunkTypeOption);
        command.AddOption(KeyFieldOption);

        command.SetHandler(Run);
        return command;
    }

    public static async Task<int> Run(InvocationContext invocationContext)
    {
        var inputPath = invocationContext.ParseResult.GetValueForArgument(InputPathArgument);
        var outputPath = invocationContext.ParseResult.GetValueForArgument(OutputPathArgument);
        var chunkType = invocationContext.ParseResult.GetValueForOption(ChunkTypeOption);
        var keyField = invocationContext.ParseResult.GetValueForOption(KeyFieldOption);

        // Validate required arguments
        if (string.IsNullOrEmpty(inputPath))
        {
            Console.WriteLine("Error: Input path is required.");
            return 1;
        }
        if (string.IsNullOrEmpty(outputPath))
        {
            Console.WriteLine("Error: Output path is required.");
            return 1;
        }
        if (string.IsNullOrEmpty(chunkType))
        {
            Console.WriteLine("Error: Chunk type is required.");
            return 1;
        }
        if (string.IsNullOrEmpty(keyField))
        {
            Console.WriteLine("Error: Key field is required.");
            return 1;
        }

        Console.WriteLine("=== PM4 Data Web Analysis ===");
        Console.WriteLine($"Input: {inputPath}");
        Console.WriteLine($"Output: {outputPath}");
        Console.WriteLine($"Target Chunk: {chunkType}");
        Console.WriteLine($"Target Field: {keyField}");
        Console.WriteLine();

        try
        {
            // Load PM4 file
            if (!File.Exists(inputPath))
            {
                Console.WriteLine($"Error: Input file '{inputPath}' does not exist.");
                return 1;
            }

            var adapter = new Pm4Adapter();
            var scene = adapter.Load(inputPath);
            
            Console.WriteLine($"Loaded PM4 file: {Path.GetFileName(inputPath)}");
            Console.WriteLine($"Chunks found: MPRL={scene.Placements?.Count ?? 0}, MSLK={scene.Links?.Count ?? 0}, MSUR={scene.Surfaces?.Count ?? 0}");
            Console.WriteLine();

            // Analyze the specified chunk type
            switch (chunkType.ToUpper())
            {
                case "MSLK":
                    await AnalyzeMslkKeys(scene, keyField, outputPath);
                    break;
                case "MPRL":
                    await AnalyzeMprlKeys(scene, keyField, outputPath);
                    break;
                case "MSUR":
                    await AnalyzeMsurKeys(scene, keyField, outputPath);
                    break;
                default:
                    Console.WriteLine($"Error: Unsupported chunk type '{chunkType}'. Supported types: MSLK, MPRL, MSUR");
                    return 1;
            }

            Console.WriteLine("Analysis complete!");
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error during analysis: {ex.Message}");
            return 1;
        }
    }

    private static async Task AnalyzeMslkKeys(Pm4Scene scene, string keyField, string outputPath)
    {
        Console.WriteLine("=== MSLK Key Analysis ===");
        
        if (scene.Links == null || !scene.Links.Any())
        {
            Console.WriteLine("No MSLK entries found.");
            return;
        }

        Console.WriteLine($"Analyzing {scene.Links.Count} MSLK entries for field '{keyField}'...");
        
        // Use reflection to get the specified field values
        var fieldValues = new List<object>();
        var entryType = scene.Links.First().GetType();
        var property = entryType.GetProperty(keyField, BindingFlags.Public | BindingFlags.Instance);
        
        if (property == null)
        {
            Console.WriteLine($"Error: Field '{keyField}' not found in MSLK entries.");
            Console.WriteLine($"Available fields: {string.Join(", ", entryType.GetProperties().Select(p => p.Name))}");
            return;
        }

        foreach (var entry in scene.Links)
        {
            var value = property.GetValue(entry);
            if (value != null)
                fieldValues.Add(value);
        }

        // Analyze patterns
        Console.WriteLine($"Field '{keyField}' analysis:");
        Console.WriteLine($"  Total values: {fieldValues.Count}");
        Console.WriteLine($"  Unique values: {fieldValues.Distinct().Count()}");
        Console.WriteLine($"  Value range: {fieldValues.Min()} - {fieldValues.Max()}");
        
        // Check for packed hierarchical patterns (Data Web hypothesis)
        if (property.PropertyType == typeof(uint) || property.PropertyType == typeof(int))
        {
            Console.WriteLine("\n=== Packed Key Analysis (Data Web Hypothesis) ===");
            AnalyzePackedKeys(fieldValues.Cast<uint>().ToList());
        }
        
        await Task.CompletedTask;
    }

    private static async Task AnalyzeMprlKeys(Pm4Scene scene, string keyField, string outputPath)
    {
        Console.WriteLine("=== MPRL Key Analysis ===");
        Console.WriteLine("MPRL analysis not yet implemented.");
        await Task.CompletedTask;
    }

    private static async Task AnalyzeMsurKeys(Pm4Scene scene, string keyField, string outputPath)
    {
        Console.WriteLine("=== MSUR Key Analysis ===");
        Console.WriteLine("MSUR analysis not yet implemented.");
        await Task.CompletedTask;
    }

    private static void AnalyzePackedKeys(List<uint> keys)
    {
        Console.WriteLine("Analyzing for packed hierarchical structure...");
        
        // Break each 32-bit key into 4 bytes
        var byteAnalysis = new Dictionary<int, Dictionary<byte, int>>();
        
        for (int bytePos = 0; bytePos < 4; bytePos++)
        {
            byteAnalysis[bytePos] = new Dictionary<byte, int>();
            
            foreach (var key in keys)
            {
                byte byteValue = (byte)((key >> (bytePos * 8)) & 0xFF);
                if (!byteAnalysis[bytePos].ContainsKey(byteValue))
                    byteAnalysis[bytePos][byteValue] = 0;
                byteAnalysis[bytePos][byteValue]++;
            }
        }
        
        // Report byte-level patterns
        for (int bytePos = 0; bytePos < 4; bytePos++)
        {
            var byteStats = byteAnalysis[bytePos];
            Console.WriteLine($"  Byte {bytePos}: {byteStats.Count} unique values, range {byteStats.Keys.Min()}-{byteStats.Keys.Max()}");
            
            // Show most common values for this byte position
            var topValues = byteStats.OrderByDescending(kvp => kvp.Value).Take(5);
            Console.WriteLine($"    Most common: {string.Join(", ", topValues.Select(kvp => $"{kvp.Key}({kvp.Value}x)"))}");
        }
    }
}
