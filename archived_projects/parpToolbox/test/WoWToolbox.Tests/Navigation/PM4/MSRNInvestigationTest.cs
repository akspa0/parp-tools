using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Xunit;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class MSRNInvestigationTest
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        
        [Fact]
        public void InvestigateMSRNChunkData_AndFallbackStrategy()
        {
            Console.WriteLine("--- MSRN INVESTIGATION + FALLBACK STRATEGY VALIDATION ---");
            
            var testFiles = new[]
            {
                "development_00_00.pm4",   // 1.2MB - Known MDOS case
                "development_00_01.pm4",   // 269KB
                "development_00_02.pm4",   // 763KB
                "development_01_00.pm4",   // 63KB
                "development_01_01.pm4",   // 145KB
                "development_14_38.pm4",   // From original_development listing
                "development_15_39.pm4",   // From original_development listing
                "development_21_38.pm4",   // From original_development listing
                "development_22_18.pm4"    // Large file from PM4FileTests.cs
            };
            
            foreach (string filename in testFiles)
            {
                string inputFilePath = Path.Combine(TestDataRoot, "original_development", "development", filename);
                if (!File.Exists(inputFilePath))
                {
                    Console.WriteLine($"❌ File not found: {filename}");
                    continue;
                }
                
                Console.WriteLine($"\n📁 Processing: {filename}");
                InvestigateSingleFile(inputFilePath);
            }
        }
        
        private void InvestigateSingleFile(string inputFilePath)
        {
            try
            {
                var pm4File = PM4File.FromFile(inputFilePath);
                
                // Basic chunk analysis
                bool hasMSRN = pm4File.MSRN?.Normals?.Count > 0;
                bool hasMDOS = pm4File.MDOS?.Entries?.Count > 0;
                bool hasMDSF = pm4File.MDSF?.Entries?.Count > 0;
                
                Console.WriteLine($"  MSLK: {pm4File.MSLK?.Entries?.Count ?? 0}");
                Console.WriteLine($"  MSUR: {pm4File.MSUR?.Entries?.Count ?? 0}");
                Console.WriteLine($"  MSRN: {pm4File.MSRN?.Normals?.Count ?? 0}");
                Console.WriteLine($"  MDOS: {pm4File.MDOS?.Entries?.Count ?? 0}");
                Console.WriteLine($"  MDSF: {pm4File.MDSF?.Entries?.Count ?? 0}");
                
                // MSRN investigation
                if (hasMSRN)
                {
                    var normals = pm4File.MSRN.Normals;
                    var uniqueNormals = normals.Distinct().Count();
                    bool matchesMSUR = normals.Count == (pm4File.MSUR?.Entries?.Count ?? 0);
                    
                    Console.WriteLine($"  🔍 MSRN Analysis:");
                    Console.WriteLine($"    Total normals: {normals.Count}");
                    Console.WriteLine($"    Unique normals: {uniqueNormals}");
                    Console.WriteLine($"    Matches MSUR count: {matchesMSUR}");
                    
                    if (normals.Count > 0)
                    {
                        var sample = normals.Take(3);
                        Console.WriteLine($"    Sample normals: {string.Join(", ", sample.Select(n => $"({n.X:F2},{n.Y:F2},{n.Z:F2})"))}");
                        
                        // Check if normals are normalized
                        var normalizedCount = normals.Count(n => Math.Abs(n.Length() - 1.0f) < 0.01f);
                        Console.WriteLine($"    Normalized normals: {normalizedCount}/{normals.Count}");
                    }
                }
                else
                {
                    Console.WriteLine($"  ❌ No MSRN data");
                }
                
                // Fallback strategy test
                if (!hasMDOS && !hasMDSF)
                {
                    Console.WriteLine($"  🔄 Testing fallback strategy (no MDOS/MDSF)...");
                    TestFallbackStrategy(pm4File);
                }
                else
                {
                    Console.WriteLine($"  ✅ Has MDOS/MDSF - proven FlexibleMethod works");
                }
                
                Console.WriteLine($"  📊 File size: {new FileInfo(inputFilePath).Length / 1024}KB");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ❌ Error: {ex.Message}");
            }
        }
        
        private void TestFallbackStrategy(PM4File pm4File)
        {
            if (pm4File.MSLK?.Entries == null)
            {
                Console.WriteLine($"    ❌ No MSLK data for fallback");
                return;
            }
            
            // Find self-referencing MSLK entries (building roots)
            var rootNodes = new List<int>();
            for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
            {
                if (pm4File.MSLK.Entries[i].Unknown_0x04 == i)
                {
                    rootNodes.Add(i);
                }
            }
            
            Console.WriteLine($"    🏗️  Building root nodes: {rootNodes.Count}");
            
            if (rootNodes.Count == 0)
            {
                Console.WriteLine($"    ❌ No self-referencing MSLK nodes found");
                return;
            }
            
            // Test if we can extract geometry for each root
            int viableBuildings = 0;
            int totalStructuralElements = 0;
            
            foreach (int rootIndex in rootNodes)
            {
                uint groupId = pm4File.MSLK.Entries[rootIndex].Unknown_0x04;
                
                var structuralElements = pm4File.MSLK.Entries
                    .Where(entry => entry.Unknown_0x04 == groupId && entry.MspiIndexCount > 0)
                    .Count();
                
                totalStructuralElements += structuralElements;
                
                if (structuralElements > 0)
                {
                    viableBuildings++;
                }
                
                Console.WriteLine($"      Root {rootIndex}: {structuralElements} structural elements");
            }
            
            Console.WriteLine($"    {(viableBuildings > 0 ? "✅" : "❌")} Viable buildings: {viableBuildings}/{rootNodes.Count}");
            Console.WriteLine($"    📈 Total structural elements: {totalStructuralElements}");
            
            // Test spatial clustering possibility
            if (pm4File.MSUR?.Entries?.Count > 0)
            {
                Console.WriteLine($"    🎯 Spatial clustering data available: {pm4File.MSUR.Entries.Count} surfaces");
            }
        }
    }
} 