using System.CommandLine;
using System.CommandLine.Invocation;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.CliCommands;

public class AnalyzeSurfaceRefIndexCommand
{
    private static readonly Argument<string> InputPathArgument = new("input-path", "Path to the input PM4 file.");

    public static Command CreateCommand()
    {
        var command = new Command("analyze-surfacerefindex", "Analyze SurfaceRefIndex packed structure to find correct MSUR linkage.");
        command.AddArgument(InputPathArgument);
        command.SetHandler(Run);
        return command;
    }

    public static async Task<int> Run(InvocationContext invocationContext)
    {
        var inputPath = invocationContext.ParseResult.GetValueForArgument(InputPathArgument);

        Console.WriteLine("=== SurfaceRefIndex Packed Structure Analysis ===");
        Console.WriteLine($"Input: {inputPath}");
        Console.WriteLine();

        try
        {
            var adapter = new Pm4Adapter();
            var scene = adapter.Load(inputPath);
            
            Console.WriteLine($"Data: MSLK={scene.Links.Count}, MSUR={scene.Surfaces.Count}");
            Console.WriteLine();

            // Analyze SurfaceRefIndex packed structure
            var surfaceRefIndexValues = scene.Links.Select(l => l.SurfaceRefIndex).ToList();
            var surfaceKeyValues = scene.Surfaces.Select(s => s.SurfaceKey).ToList();
            
            Console.WriteLine($"SurfaceRefIndex type: {surfaceRefIndexValues.First().GetType()}");
            Console.WriteLine($"SurfaceKey type: {surfaceKeyValues.First().GetType()}");
            Console.WriteLine();
            
            Console.WriteLine("=== SurfaceRefIndex Packed Analysis (16-bit) ===");
            AnalyzePackedStructure16Bit("SurfaceRefIndex", surfaceRefIndexValues.Select(v => (ushort)v).ToList());
            
            Console.WriteLine();
            Console.WriteLine("=== MSUR SurfaceKey Analysis ===");
            AnalyzePackedStructure("SurfaceKey", surfaceKeyValues.Cast<uint>().ToList());
            
            Console.WriteLine();
            Console.WriteLine("=== Linkage Testing ===");
            
            // Test different component combinations (SurfaceRefIndex is 16-bit, SurfaceKey is 32-bit)
            TestLinkage("SurfRef (16-bit) vs SurfKey (32-bit)", 
                surfaceRefIndexValues.Select(v => (uint)(ushort)v), 
                surfaceKeyValues.Cast<uint>());
            
            TestLinkage("SurfRef vs SurfKey Lower 16-bit", 
                surfaceRefIndexValues.Select(v => (uint)(ushort)v), 
                surfaceKeyValues.Select(v => (uint)(v & 0xFFFF)));
            
            TestLinkage("SurfRef vs SurfKey Upper 16-bit", 
                surfaceRefIndexValues.Select(v => (uint)(ushort)v), 
                surfaceKeyValues.Select(v => (uint)((v >> 16) & 0xFFFF)));
                
            TestLinkage("SurfRef Lower 8-bit vs SurfKey Lower 8-bit", 
                surfaceRefIndexValues.Select(v => (uint)((ushort)v & 0xFF)), 
                surfaceKeyValues.Select(v => (uint)(v & 0xFF)));
                
            TestLinkage("SurfRef Upper 8-bit vs SurfKey Upper 8-bit", 
                surfaceRefIndexValues.Select(v => (uint)(((ushort)v >> 8) & 0xFF)), 
                surfaceKeyValues.Select(v => (uint)((v >> 24) & 0xFF)));
            
            await Task.CompletedTask;
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void AnalyzePackedStructure16Bit(string fieldName, List<ushort> values)
    {
        Console.WriteLine($"{fieldName} packed analysis ({values.Count} values):");
        
        // Break each 16-bit value into 2 bytes
        var byteAnalysis = new Dictionary<int, Dictionary<byte, int>>();
        
        for (int bytePos = 0; bytePos < 2; bytePos++)
        {
            byteAnalysis[bytePos] = new Dictionary<byte, int>();
        }
        
        foreach (var value in values)
        {
            // Analyze bytes
            for (int bytePos = 0; bytePos < 2; bytePos++)
            {
                byte byteValue = (byte)((value >> (bytePos * 8)) & 0xFF);
                if (!byteAnalysis[bytePos].ContainsKey(byteValue))
                    byteAnalysis[bytePos][byteValue] = 0;
                byteAnalysis[bytePos][byteValue]++;
            }
        }
        
        // Report patterns
        Console.WriteLine($"  Range: {values.Min()}-{values.Max()}, Unique values: {values.Distinct().Count()}");
        
        for (int bytePos = 0; bytePos < 2; bytePos++)
        {
            var byteStats = byteAnalysis[bytePos];
            Console.WriteLine($"  Byte {bytePos}: {byteStats.Count} unique, range {byteStats.Keys.Min()}-{byteStats.Keys.Max()}");
            var topValues = byteStats.OrderByDescending(kvp => kvp.Value).Take(5);
            Console.WriteLine($"    Most common: {string.Join(", ", topValues.Select(kvp => $"{kvp.Key}({kvp.Value}x)"))}");
        }
    }

    private static void AnalyzePackedStructure(string fieldName, List<uint> values)
    {
        Console.WriteLine($"{fieldName} packed analysis ({values.Count} values):");
        
        // Break each 32-bit value into 4 bytes and 2 16-bit words
        var byteAnalysis = new Dictionary<int, Dictionary<byte, int>>();
        var wordAnalysis = new Dictionary<int, Dictionary<ushort, int>>();
        
        for (int bytePos = 0; bytePos < 4; bytePos++)
        {
            byteAnalysis[bytePos] = new Dictionary<byte, int>();
        }
        
        for (int wordPos = 0; wordPos < 2; wordPos++)
        {
            wordAnalysis[wordPos] = new Dictionary<ushort, int>();
        }
        
        foreach (var value in values)
        {
            // Analyze bytes
            for (int bytePos = 0; bytePos < 4; bytePos++)
            {
                byte byteValue = (byte)((value >> (bytePos * 8)) & 0xFF);
                if (!byteAnalysis[bytePos].ContainsKey(byteValue))
                    byteAnalysis[bytePos][byteValue] = 0;
                byteAnalysis[bytePos][byteValue]++;
            }
            
            // Analyze 16-bit words
            for (int wordPos = 0; wordPos < 2; wordPos++)
            {
                ushort wordValue = (ushort)((value >> (wordPos * 16)) & 0xFFFF);
                if (!wordAnalysis[wordPos].ContainsKey(wordValue))
                    wordAnalysis[wordPos][wordValue] = 0;
                wordAnalysis[wordPos][wordValue]++;
            }
        }
        
        // Report byte-level patterns
        for (int bytePos = 0; bytePos < 4; bytePos++)
        {
            var byteStats = byteAnalysis[bytePos];
            Console.WriteLine($"  Byte {bytePos}: {byteStats.Count} unique, range {byteStats.Keys.Min()}-{byteStats.Keys.Max()}");
        }
        
        // Report word-level patterns (more important for linkage)
        for (int wordPos = 0; wordPos < 2; wordPos++)
        {
            var wordStats = wordAnalysis[wordPos];
            Console.WriteLine($"  Word {wordPos}: {wordStats.Count} unique, range {wordStats.Keys.Min()}-{wordStats.Keys.Max()}");
            var topValues = wordStats.OrderByDescending(kvp => kvp.Value).Take(5);
            Console.WriteLine($"    Most common: {string.Join(", ", topValues.Select(kvp => $"{kvp.Key}({kvp.Value}x)"))}");
        }
    }

    private static void TestLinkage(string testName, IEnumerable<uint> refValues, IEnumerable<uint> keyValues)
    {
        var refSet = refValues.ToHashSet();
        var keySet = keyValues.ToHashSet();
        
        var matches = refSet.Intersect(keySet).Count();
        var totalRefs = refSet.Count;
        var totalKeys = keySet.Count;
        
        double matchPercent = totalRefs > 0 ? (matches * 100.0 / totalRefs) : 0;
        
        Console.WriteLine($"{testName}: {matches} matches out of {totalRefs} refs ({matchPercent:F1}%) and {totalKeys} keys");
    }
}
