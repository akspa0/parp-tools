using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

class MSRNInvestigator
{
    static void Main()
    {
        Console.WriteLine("=== MSRN CHUNK INVESTIGATION ===");
        
        string testDataPath = Path.Combine("..", "..", "test_data", "original_development", "development");
        
        if (!Directory.Exists(testDataPath))
        {
            Console.WriteLine($"❌ Test data directory not found: {testDataPath}");
            return;
        }
        
        var pm4Files = Directory.GetFiles(testDataPath, "*.pm4")
            .Where(f => new FileInfo(f).Length > 5120) // Skip 0-byte and tiny files
            .Take(10) // Just test first 10 files
            .ToArray();
            
        Console.WriteLine($"Found {pm4Files.Length} PM4 files to analyze");
        
        foreach (string filePath in pm4Files)
        {
            AnalyzePM4File(filePath);
        }
    }
    
    static void AnalyzePM4File(string filePath)
    {
        try
        {
            string fileName = Path.GetFileName(filePath);
            long fileSize = new FileInfo(filePath).Length;
            
            Console.WriteLine($"\n📁 {fileName} ({fileSize / 1024}KB)");
            
            var pm4File = PM4File.FromFile(filePath);
            
            // Basic chunk counts
            Console.WriteLine($"  MSLK: {pm4File.MSLK?.Entries?.Count ?? 0}");
            Console.WriteLine($"  MSUR: {pm4File.MSUR?.Entries?.Count ?? 0}");
            Console.WriteLine($"  MSRN: {pm4File.MSRN?.Normals?.Count ?? 0}");
            Console.WriteLine($"  MDOS: {pm4File.MDOS?.Entries?.Count ?? 0}");
            Console.WriteLine($"  MDSF: {pm4File.MDSF?.Entries?.Count ?? 0}");
            
            // MSRN Analysis
            if (pm4File.MSRN?.Normals?.Count > 0)
            {
                var normals = pm4File.MSRN.Normals;
                var uniqueNormals = normals.Distinct().Count();
                bool matchesMSUR = normals.Count == (pm4File.MSUR?.Entries?.Count ?? 0);
                
                Console.WriteLine($"  🔍 MSRN Analysis:");
                Console.WriteLine($"    Total: {normals.Count}, Unique: {uniqueNormals}");
                Console.WriteLine($"    Matches MSUR: {matchesMSUR}");
                
                // Sample normals
                if (normals.Count > 0)
                {
                    var sample = normals.Take(3);
                    foreach (var normal in sample)
                    {
                        float length = normal.Length();
                        Console.WriteLine($"    Sample: ({normal.X:F3},{normal.Y:F3},{normal.Z:F3}) |len|={length:F3}");
                    }
                }
                
                // Check normalization
                var normalizedCount = normals.Count(n => Math.Abs(n.Length() - 1.0f) < 0.01f);
                Console.WriteLine($"    Normalized: {normalizedCount}/{normals.Count}");
            }
            else
            {
                Console.WriteLine($"  ❌ No MSRN data");
            }
            
            // Fallback strategy viability
            bool hasStructuralData = pm4File.MDOS?.Entries?.Count > 0 || pm4File.MDSF?.Entries?.Count > 0;
            if (!hasStructuralData)
            {
                Console.WriteLine($"  🔄 Fallback strategy needed (no MDOS/MDSF)");
                TestFallbackViability(pm4File);
            }
            else
            {
                Console.WriteLine($"  ✅ FlexibleMethod proven (has MDOS/MDSF)");
            }
            
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ❌ Error: {ex.Message}");
        }
    }
    
    static void TestFallbackViability(PM4File pm4File)
    {
        if (pm4File.MSLK?.Entries == null)
        {
            Console.WriteLine($"    ❌ No MSLK for fallback");
            return;
        }
        
        // Find root nodes
        var rootNodes = pm4File.MSLK.Entries
            .Select((entry, index) => new { entry, index })
            .Where(x => x.entry.Unknown_0x04 == x.index)
            .Select(x => x.index)
            .ToList();
            
        Console.WriteLine($"    🏗️  Root nodes: {rootNodes.Count}");
        
        if (rootNodes.Count == 0)
        {
            Console.WriteLine($"    ❌ No root nodes for spatial clustering");
            return;
        }
        
        // Check structural elements
        int totalStructural = rootNodes.Sum(rootIndex =>
        {
            uint groupId = pm4File.MSLK.Entries[rootIndex].Unknown_0x04;
            return pm4File.MSLK.Entries
                .Count(entry => entry.Unknown_0x04 == groupId && entry.MspiIndexCount > 0);
        });
        
        Console.WriteLine($"    📈 Structural elements: {totalStructural}");
        Console.WriteLine($"    🎯 MSUR surfaces: {pm4File.MSUR?.Entries?.Count ?? 0}");
        
        bool fallbackViable = rootNodes.Count > 0 && (pm4File.MSUR?.Entries?.Count ?? 0) > 0;
        Console.WriteLine($"    {(fallbackViable ? "✅" : "❌")} Fallback strategy viable");
    }
} 