using System.CommandLine;
using System.CommandLine.Invocation;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.CliCommands;

public class AnalyzeMslkFieldsCommand
{
    private static readonly Argument<string> InputPathArgument = new("input-path", "Path to the input PM4 file.");

    public static Command CreateCommand()
    {
        var command = new Command("analyze-mslk-fields", "Analyze MSLK unknown fields for building-scale grouping keys and packed field structures.");
        command.AddArgument(InputPathArgument);
        command.SetHandler(Run);
        return command;
    }

    public static async Task<int> Run(InvocationContext invocationContext)
    {
        var inputPath = invocationContext.ParseResult.GetValueForArgument(InputPathArgument);

        Console.WriteLine("=== MSLK Unknown Fields Analysis ===");
        Console.WriteLine($"Input: {inputPath}");
        Console.WriteLine();

        try
        {
            var adapter = new Pm4Adapter();
            var scene = adapter.Load(inputPath);
            
            Console.WriteLine($"Data: MSLK={scene.Links.Count}, MPRL={scene.Placements.Count}, MSUR={scene.Surfaces.Count}");
            Console.WriteLine();

            if (!scene.Links.Any())
            {
                Console.WriteLine("No MSLK entries found.");
                return 0;
            }

            // Analyze MSLK unknown fields for building-scale grouping
            Console.WriteLine("=== MSLK Field Analysis ===");
            AnalyzeMslkFields(scene);
            
            Console.WriteLine();
            Console.WriteLine("=== Packed Field Structure Analysis ===");
            AnalyzePackedFields(scene);
            
            Console.WriteLine();
            Console.WriteLine("=== Building-Scale Grouping Tests ===");
            TestBuildingScaleGrouping(scene);
            
            await Task.CompletedTask;
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            return 1;
        }
    }

    private static void AnalyzeMslkFields(Pm4Scene scene)
    {
        Console.WriteLine($"Analyzing {scene.Links.Count} MSLK entries...");
        
        // Analyze all unknown fields in MSLK
        var linksList = scene.Links.ToList();
        var parentIndices = linksList.Select(l => l.ParentIndex).ToList();
        var surfaceRefIndices = linksList.Select(l => (uint)l.SurfaceRefIndex).ToList();
        var mspiFirstIndices = linksList.Select(l => (uint)l.MspiFirstIndex).ToList();
        var mspiIndexCounts = linksList.Select(l => (uint)l.MspiIndexCount).ToList();

        Console.WriteLine();
        Console.WriteLine("Field Analysis:");
        AnalyzeField("ParentIndex", parentIndices);
        AnalyzeField("SurfaceRefIndex", surfaceRefIndices);
        AnalyzeField("MspiFirstIndex", mspiFirstIndices);
        AnalyzeField("MspiIndexCount", mspiIndexCounts);
    }

    private static void AnalyzeField(string fieldName, List<uint> values)
    {
        var uniqueCount = values.Distinct().Count();
        var min = values.Any() ? values.Min() : 0;
        var max = values.Any() ? values.Max() : 0;
        var mostCommon = values.GroupBy(v => v).OrderByDescending(g => g.Count()).Take(5);
        
        Console.WriteLine($"  {fieldName}: {uniqueCount} unique values, range {min}-{max}");
        Console.WriteLine($"    Most common: {string.Join(", ", mostCommon.Select(g => $"{g.Key}({g.Count()}x)"))}");
        
        // Analyze as packed field (dual 16-bit)
        if (max > 0xFFFF)
        {
            var upperBytes = values.Select(v => v >> 16).Distinct().Count();
            var lowerBytes = values.Select(v => v & 0xFFFF).Distinct().Count();
            Console.WriteLine($"    Packed analysis: Upper 16-bit = {upperBytes} unique, Lower 16-bit = {lowerBytes} unique");
            
            // Check if upper bytes could be building IDs
            if (upperBytes < 1000 && upperBytes > 10) // Reasonable building count range
            {
                Console.WriteLine($"    *** POTENTIAL BUILDING ID: Upper 16-bit ({upperBytes} unique values) ***");
            }
        }
    }

    private static void AnalyzePackedFields(Pm4Scene scene)
    {
        Console.WriteLine("Testing packed field combinations for building-scale grouping...");
        
        // Test different packed field interpretations
        var linksList = scene.Links.ToList();
        var parentIndexHighBytes = linksList.Select(l => l.ParentIndex >> 16).Distinct().ToList();
        var parentIndexLowBytes = linksList.Select(l => l.ParentIndex & 0xFFFF).Distinct().ToList();
        
        Console.WriteLine($"ParentIndex upper 16-bit: {parentIndexHighBytes.Count} unique values");
        Console.WriteLine($"ParentIndex lower 16-bit: {parentIndexLowBytes.Count} unique values");
        
        // Test grouping by upper 16-bit of ParentIndex
        var groupsByUpperParent = linksList.GroupBy(l => l.ParentIndex >> 16).ToList();
        Console.WriteLine($"Grouping by ParentIndex[upper 16-bit]: {groupsByUpperParent.Count} groups");
        
        var topGroups = groupsByUpperParent.OrderByDescending(g => g.Count()).Take(10);
        Console.WriteLine("Top groups by size:");
        foreach (var group in topGroups)
        {
            Console.WriteLine($"  Group {group.Key}: {group.Count()} MSLK entries");
        }
        
        // Test other packed combinations
        Console.WriteLine();
        Console.WriteLine("Testing SurfaceRefIndex packed structure:");
        var surfaceRefGroups = linksList.GroupBy(l => (uint)l.SurfaceRefIndex >> 8).ToList();
        Console.WriteLine($"Grouping by SurfaceRefIndex[upper 8-bit]: {surfaceRefGroups.Count} groups");
    }

    private static void TestBuildingScaleGrouping(Pm4Scene scene)
    {
        Console.WriteLine("Testing historical insights about building-scale grouping...");
        
        // Test grouping by upper bytes of ParentIndex (historical "first two bytes" insight)
        var buildingGroups = scene.Links.GroupBy(l => l.ParentIndex >> 16).ToList();
        
        Console.WriteLine($"Building-scale grouping test: {buildingGroups.Count} potential buildings");
        
        var significantBuildings = buildingGroups.Where(g => g.Count() >= 10).OrderByDescending(g => g.Count()).Take(5);
        
        Console.WriteLine("Top 5 significant buildings (≥10 MSLK entries):");
        foreach (var building in significantBuildings)
        {
            var links = building.ToList();
            var surfaceMatches = 0;
            
            // Count how many surfaces this building can access
            foreach (var link in links)
            {
                var surface = scene.Surfaces.FirstOrDefault(s => (s.SurfaceKey & 0xFF) == (link.SurfaceRefIndex & 0xFF));
                if (surface != null) surfaceMatches++;
            }
            
            Console.WriteLine($"  Building {building.Key}: {links.Count} MSLK entries, {surfaceMatches} surface matches");
            
            // Test MPRL correlation
            var mprlMatches = scene.Placements.Count(p => (p.Unknown4 >> 16) == building.Key);
            Console.WriteLine($"    MPRL correlation: {mprlMatches} placements");
        }
    }
}
