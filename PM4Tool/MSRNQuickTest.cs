using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

class Program
{
    static void Main()
    {
        Console.WriteLine("--- MSRN INVESTIGATION + FALLBACK STRATEGY VALIDATION ---");
        
        var testDataRoot = Path.GetFullPath(Path.Combine("..", "test_data"));
        var testFiles = new[]
        {
            "development_14_38.pm4",   // 66KB 
            "development_15_39.pm4",   // 40KB
            "development_21_38.pm4",   // 288KB
            "development_00_00.pm4"    // 1.2MB - Our known MDOS case
        };
        
        foreach (string filename in testFiles)
        {
            string inputFilePath = Path.Combine(testDataRoot, "original_development", filename);
            if (!File.Exists(inputFilePath))
            {
                Console.WriteLine($"❌ File not found: {filename}");
                continue;
            }
            
            Console.WriteLine($"\n📁 Processing: {filename}");
            InvestigateSingleFile(inputFilePath);
        }
    }
    
    static void InvestigateSingleFile(string inputFilePath)
    {
        try
        {
            var pm4File = PM4File.FromFile(inputFilePath);
            
            // Basic chunk analysis
            bool hasMSRN = pm4File.MSRN?.Normals?.Count > 0;
            bool hasMDOS = pm4File.MDOS?.Entries?.Count > 0;
            
            Console.WriteLine($"  MSLK: {pm4File.MSLK?.Entries?.Count ?? 0}");
            Console.WriteLine($"  MSUR: {pm4File.MSUR?.Entries?.Count ?? 0}");
            Console.WriteLine($"  MSRN: {pm4File.MSRN?.Normals?.Count ?? 0}");
            Console.WriteLine($"  MDOS: {pm4File.MDOS?.Entries?.Count ?? 0}");
            
            // MSRN investigation
            if (hasMSRN)
            {
                var normals = pm4File.MSRN.Normals;
                var uniqueNormals = normals.Distinct().Count();
                bool matchesMSUR = normals.Count == (pm4File.MSUR?.Entries?.Count ?? 0);
                
                Console.WriteLine($"  🔍 MSRN: Unique={uniqueNormals}/{normals.Count}, Matches MSUR={matchesMSUR}");
                
                if (normals.Count > 0)
                {
                    var sample = normals.Take(3);
                    Console.WriteLine($"    Sample: {string.Join(", ", sample.Select(n => $"({n.X},{n.Y},{n.Z})"))}");
                }
            }
            
            // Fallback strategy test
            if (!hasMDOS)
            {
                Console.WriteLine($"  🔄 Testing fallback strategy (no MDOS)...");
                TestFallbackStrategy(pm4File);
            }
            else
            {
                Console.WriteLine($"  ✅ Has MDOS - proven strategy");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ❌ Error: {ex.Message}");
        }
    }
    
    static void TestFallbackStrategy(PM4File pm4File)
    {
        if (pm4File.MSLK?.Entries == null)
        {
            Console.WriteLine($"    ❌ No MSLK data");
            return;
        }
        
        // Find self-referencing MSLK entries (building roots)
        var rootNodes = new System.Collections.Generic.List<int>();
        for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
        {
            if (pm4File.MSLK.Entries[i].Unknown_0x04 == i)
            {
                rootNodes.Add(i);
            }
        }
        
        Console.WriteLine($"    🏗️ Building roots: {rootNodes.Count}");
        
        if (rootNodes.Count == 0)
        {
            Console.WriteLine($"    ❌ No building roots found");
            return;
        }
        
        // Test if we can extract geometry for each root
        int viableBuildings = 0;
        foreach (int rootIndex in rootNodes)
        {
            uint groupId = pm4File.MSLK.Entries[rootIndex].Unknown_0x04;
            
            var structuralElements = pm4File.MSLK.Entries
                .Where(entry => entry.Unknown_0x04 == groupId && entry.MspiIndexCount > 0)
                .Count();
            
            if (structuralElements > 0)
            {
                viableBuildings++;
            }
            
            Console.WriteLine($"      Root {rootIndex}: {structuralElements} structural elements");
        }
        
        Console.WriteLine($"    {(viableBuildings > 0 ? "✅" : "❌")} Viable buildings: {viableBuildings}/{rootNodes.Count}");
    }
} 