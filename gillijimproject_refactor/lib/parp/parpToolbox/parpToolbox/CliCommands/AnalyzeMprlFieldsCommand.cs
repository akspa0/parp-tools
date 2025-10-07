using System.CommandLine;
using System.CommandLine.Invocation;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.CliCommands;

public class AnalyzeMprlFieldsCommand
{
    private static readonly Argument<string> InputPathArgument = new("input-path", "Path to the input PM4 file.");

    public static Command CreateCommand()
    {
        var command = new Command("analyze-mprl-fields", "Analyze MPRL unknown fields for hierarchical grouping and MSLK/MSUR linkage clues.");
        command.AddArgument(InputPathArgument);
        command.SetHandler(Run);
        return command;
    }

    public static async Task<int> Run(InvocationContext invocationContext)
    {
        var inputPath = invocationContext.ParseResult.GetValueForArgument(InputPathArgument);

        Console.WriteLine("=== MPRL Unknown Fields Analysis ===");
        Console.WriteLine($"Input: {inputPath}");
        Console.WriteLine();

        try
        {
            var adapter = new Pm4Adapter();
            var scene = adapter.Load(inputPath);
            
            Console.WriteLine($"Data: MPRL={scene.Placements.Count}, MSLK={scene.Links.Count}, MSUR={scene.Surfaces.Count}");
            Console.WriteLine();

            if (!scene.Placements.Any())
            {
                Console.WriteLine("No MPRL entries found.");
                return 0;
            }

            // Analyze MPRL unknown fields structure
            Console.WriteLine("=== MPRL Field Analysis ===");
            AnalyzeMprlFields(scene.Placements);
            
            Console.WriteLine();
            Console.WriteLine("=== MPRL → MSLK Linkage Investigation ===");
            InvestigateMprlMslkLinkage(scene);
            
            Console.WriteLine();
            Console.WriteLine("=== MPRL → MSUR Linkage Investigation ===");
            InvestigateMprlMsurLinkage(scene);
            
            await Task.CompletedTask;
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void AnalyzeMprlFields(List<ParpToolbox.Formats.P4.Chunks.Common.MprlChunk.Entry> placements)
    {
        Console.WriteLine($"Analyzing {placements.Count} MPRL entries...");
        
        // Get all unknown field values (convert UInt16/Int16 to UInt32 for analysis)
        var unknown4Values = placements.Select(p => (uint)p.Unknown4).ToList();
        var unknown6Values = placements.Select(p => (uint)p.Unknown6).ToList();
        var unknown14Values = placements.Select(p => p.Unknown14 >= 0 ? (uint)p.Unknown14 : (uint)(0x10000 + p.Unknown14)).ToList();
        var unknown16Values = placements.Select(p => (uint)p.Unknown16).ToList();

        Console.WriteLine();
        Console.WriteLine("Field Analysis:");
        AnalyzeField("Unknown4", unknown4Values);
        AnalyzeField("Unknown6", unknown6Values);
        AnalyzeField("Unknown14", unknown14Values);
        AnalyzeField("Unknown16", unknown16Values);
        
        Console.WriteLine();
        Console.WriteLine("Cross-field correlations:");
        
        // Check for correlations between fields
        var uniquePairs46 = placements.Select(p => ((uint)p.Unknown4, (uint)p.Unknown6)).Distinct().Count();
        var uniquePairs416 = placements.Select(p => ((uint)p.Unknown4, (uint)p.Unknown16)).Distinct().Count();
        var uniquePairs614 = placements.Select(p => ((uint)p.Unknown6, p.Unknown14)).Distinct().Count();
        
        Console.WriteLine($"Unknown4+Unknown6 combinations: {uniquePairs46}");
        Console.WriteLine($"Unknown4+Unknown16 combinations: {uniquePairs416}");
        Console.WriteLine($"Unknown6+Unknown14 combinations: {uniquePairs614}");
    }

    private static void AnalyzeField(string fieldName, List<uint> values)
    {
        var uniqueCount = values.Distinct().Count();
        var min = values.Any() ? values.Min() : 0;
        var max = values.Any() ? values.Max() : 0;
        var mostCommon = values.GroupBy(v => v).OrderByDescending(g => g.Count()).Take(5);
        
        Console.WriteLine($"  {fieldName}: {uniqueCount} unique values, range {min}-{max}");
        Console.WriteLine($"    Most common: {string.Join(", ", mostCommon.Select(g => $"{g.Key}({g.Count()}x)"))}");
        
        // Check if values look like packed fields
        if (max > 0xFFFF)
        {
            Console.WriteLine($"    32-bit analysis: Upper 16-bit has {values.Select(v => v >> 16).Distinct().Count()} unique values");
            Console.WriteLine($"    32-bit analysis: Lower 16-bit has {values.Select(v => v & 0xFFFF).Distinct().Count()} unique values");
        }
    }

    private static void InvestigateMprlMslkLinkage(Pm4Scene scene)
    {
        Console.WriteLine("Testing MPRL fields as potential MSLK linkage keys...");
        
        var mslkParentIndices = scene.Links.Select(l => l.ParentIndex).ToHashSet();
        var mslkSurfaceRefs = scene.Links.Select(l => (uint)l.SurfaceRefIndex).ToHashSet();
        
        foreach (var placement in scene.Placements.Take(10)) // Sample first 10
        {
            var u4Matches = scene.Links.Count(l => l.ParentIndex == (uint)placement.Unknown4);
            var u6Matches = scene.Links.Count(l => l.ParentIndex == (uint)placement.Unknown6);
            var u14Matches = scene.Links.Count(l => (uint)l.SurfaceRefIndex == (placement.Unknown14 >= 0 ? (uint)placement.Unknown14 : 0));
            var u16Matches = scene.Links.Count(l => (uint)l.SurfaceRefIndex == (uint)placement.Unknown16);
            
            if (u4Matches > 0 || u6Matches > 0 || u14Matches > 0 || u16Matches > 0)
            {
                Console.WriteLine($"MPRL[{Array.IndexOf(scene.Placements.ToArray(), placement)}]: U4→{u4Matches}MSLK, U6→{u6Matches}MSLK, U14→{u14Matches}MSLK, U16→{u16Matches}MSLK");
            }
        }
        
        // Test correlation with container IDs (from ParentIndex analysis)
        var containerCounts = new Dictionary<byte, int>();
        foreach (var link in scene.Links)
        {
            var containerID = (byte)((link.ParentIndex >> 8) & 0xFF);
            containerCounts[containerID] = containerCounts.GetValueOrDefault(containerID, 0) + 1;
        }
        
        Console.WriteLine();
        Console.WriteLine("Container correlation test:");
        foreach (var placement in scene.Placements.Take(5))
        {
            var u4Container = (byte)(((uint)placement.Unknown4 >> 8) & 0xFF);
            var u6Container = (byte)(((uint)placement.Unknown6 >> 8) & 0xFF);
            var mslkInU4Container = containerCounts.GetValueOrDefault(u4Container, 0);
            var mslkInU6Container = containerCounts.GetValueOrDefault(u6Container, 0);
            
            Console.WriteLine($"MPRL U4 container {u4Container} has {mslkInU4Container} MSLK entries");
            Console.WriteLine($"MPRL U6 container {u6Container} has {mslkInU6Container} MSLK entries");
        }
    }

    private static void InvestigateMprlMsurLinkage(Pm4Scene scene)
    {
        Console.WriteLine("Testing MPRL fields as potential MSUR linkage keys...");
        
        var msurSurfaceKeys = scene.Surfaces.Select(s => s.SurfaceKey).ToHashSet();
        
        foreach (var placement in scene.Placements.Take(10)) // Sample first 10
        {
            var u4MatchesRaw = msurSurfaceKeys.Contains((uint)placement.Unknown4) ? 1 : 0;
            var u16MatchesRaw = msurSurfaceKeys.Contains((uint)placement.Unknown16) ? 1 : 0;
            
            // Test lower 8-bit matching (based on our SurfaceRefIndex breakthrough)
            var u4Matches8bit = scene.Surfaces.Count(s => (s.SurfaceKey & 0xFF) == ((uint)placement.Unknown4 & 0xFF));
            var u16Matches8bit = scene.Surfaces.Count(s => (s.SurfaceKey & 0xFF) == ((uint)placement.Unknown16 & 0xFF));
            
            if (u4MatchesRaw > 0 || u16MatchesRaw > 0 || u4Matches8bit > 0 || u16Matches8bit > 0)
            {
                Console.WriteLine($"MPRL[{Array.IndexOf(scene.Placements.ToArray(), placement)}]: U4→{u4MatchesRaw}raw+{u4Matches8bit}8bit MSUR, U16→{u16MatchesRaw}raw+{u16Matches8bit}8bit MSUR");
            }
        }
    }
}
