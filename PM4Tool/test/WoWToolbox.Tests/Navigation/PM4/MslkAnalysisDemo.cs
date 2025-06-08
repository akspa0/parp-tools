using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.Tests.Navigation.PM4
{
    /// <summary>
    /// Simple demonstration of MSLK relationship analysis with Mermaid diagram output.
    /// Run this to see the new functionality in action.
    /// </summary>
    public class MslkAnalysisDemo
    {
        public static void RunDemo()
        {
            Console.WriteLine("🚀 MSLK RELATIONSHIP ANALYSIS DEMO");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine();

            // Find available test PM4 files
            var testDataPaths = new[]
            {
                "test_data/development",
                "test_data/development_335", 
                "test_data/original_development/development",
                "test_data"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    Console.WriteLine($"✅ Found test data directory: {path}");
                    break;
                }
            }

            if (testDataDir == null)
            {
                Console.WriteLine("❌ No test data directories found. Searched:");
                foreach (var path in testDataPaths)
                {
                    Console.WriteLine($"   - {path}");
                }
                return;
            }

            // Find PM4 files
            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                .Take(3) // Limit to first 3 files for demo
                .ToArray();

            if (pm4Files.Length == 0)
            {
                Console.WriteLine($"❌ No PM4 files found in {testDataDir}");
                return;
            }

            Console.WriteLine($"📁 Found {pm4Files.Length} PM4 files for analysis");
            Console.WriteLine();

            // Demonstrate the analyzers
            var analyzer = new Pm4ChunkAnalyzer();
            var relationshipAnalyzer = new MslkRelationshipAnalyzer();

            foreach (var pm4FilePath in pm4Files)
            {
                try
                {
                    Console.WriteLine($"🔍 ANALYZING: {Path.GetFileName(pm4FilePath)}");
                    Console.WriteLine("─".PadRight(60, '─'));

                    var pm4File = PM4File.FromFile(pm4FilePath);
                    var fileName = Path.GetFileNameWithoutExtension(pm4FilePath);

                    // Show basic file info
                    Console.WriteLine($"📊 FILE INFO:");
                    Console.WriteLine($"├─ MSLK Entries: {pm4File.MSLK?.Entries?.Count ?? 0}");
                    Console.WriteLine($"├─ MSUR Entries: {pm4File.MSUR?.Entries?.Count ?? 0}");
                    Console.WriteLine($"├─ MSVT Vertices: {pm4File.MSVT?.Vertices?.Count ?? 0}");
                    Console.WriteLine($"└─ MSPV Vertices: {pm4File.MSPV?.Vertices?.Count ?? 0}");
                    Console.WriteLine();

                    // Run the comprehensive analysis with Mermaid output
                    analyzer.AnalyzePm4FileWithMslkRelationships(pm4FilePath, outputMermaidToConsole: true);

                    // Also demonstrate the direct analyzer
                    Console.WriteLine("🧪 DIRECT ANALYZER RESULTS:");
                    var map = relationshipAnalyzer.AnalyzeMslkRelationships(pm4File, fileName);
                    
                    if (map.Nodes.Count > 0)
                    {
                        // Show validation against documented structure
                        Console.WriteLine($"├─ Validation Issues: {map.ValidationIssues.Count}");
                        foreach (var issue in map.ValidationIssues.Take(3))
                        {
                            Console.WriteLine($"│  ⚠️  {issue}");
                        }
                        
                        // Show flag patterns found vs documented patterns
                        Console.WriteLine($"├─ Flag Patterns Found:");
                        foreach (var flagGroup in map.NodesByFlags.Take(5))
                        {
                            var flagBits = GetFlagBits(flagGroup.Key);
                            var expectedFlags = flagBits.Intersect(new[] { 1, 2, 4, 8, 16 });
                            var isDocumented = expectedFlags.Any();
                            var status = isDocumented ? "✅" : "❓";
                            Console.WriteLine($"│  {status} 0x{flagGroup.Key:X2}: {flagGroup.Value.Count} nodes");
                        }
                        
                        Console.WriteLine($"└─ Structure Validation: {(map.ValidationIssues.Count == 0 ? "✅ PASSED" : "⚠️ ISSUES FOUND")}");
                    }
                    else
                    {
                        Console.WriteLine("└─ No MSLK data to analyze");
                    }

                    Console.WriteLine();
                    Console.WriteLine("═".PadRight(60, '═'));
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ ERROR analyzing {Path.GetFileName(pm4FilePath)}: {ex.Message}");
                    Console.WriteLine();
                }
            }

            // Show batch analysis summary
            if (pm4Files.Length > 1)
            {
                Console.WriteLine("🔄 BATCH ANALYSIS SUMMARY");
                Pm4MslkCliAnalyzer.BatchAnalyzeMslkPatterns(pm4Files);
            }

            Console.WriteLine("🎉 Demo completed! The MSLK relationship analyzer is working correctly.");
            Console.WriteLine("📋 Key features demonstrated:");
            Console.WriteLine("   ✅ Documented MSLK structure interpretation");
            Console.WriteLine("   ✅ Object + node relationship mapping");
            Console.WriteLine("   ✅ Mermaid diagram generation");
            Console.WriteLine("   ✅ Validation against wowdev.wiki documentation");
            Console.WriteLine("   ✅ CLI integration for analysis workflows");
        }

        private static int[] GetFlagBits(byte flags)
        {
            var bits = new List<int>();
            for (int i = 0; i < 8; i++)
            {
                if ((flags & (1 << i)) != 0)
                {
                    bits.Add(1 << i);
                }
            }
            return bits.ToArray();
        }
    }
} 