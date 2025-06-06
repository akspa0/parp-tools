using System;
using System.IO;
using System.Linq;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Simple demo program to analyze MSLK hierarchy and output results to console.
    /// Can be called directly without running complex test frameworks.
    /// </summary>
    public static class MslkHierarchyDemo
    {
        public static void RunHierarchyAnalysis()
        {
            Console.WriteLine("🔗 MSLK HIERARCHY ANALYSIS DEMO");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine("Analyzing MSLK as a node-link system to discover object hierarchies");
            Console.WriteLine();

            // Find test PM4 files
            var testDataPaths = new[]
            {
                "test_data/original_development",
                "test_data/development",
                "test_data/development_335"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    Console.WriteLine($"✅ Found test data: {path}");
                    break;
                }
            }

            if (testDataDir == null)
            {
                Console.WriteLine("❌ No test data directories found. Expected:");
                foreach (var path in testDataPaths)
                {
                    Console.WriteLine($"   - {path}");
                }
                return;
            }

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                                   .Take(2) // Analyze 2 files for demo
                                   .ToArray();

            if (!pm4Files.Any())
            {
                Console.WriteLine($"❌ No PM4 files found in {testDataDir}");
                return;
            }

            Console.WriteLine($"🎯 Found {pm4Files.Length} PM4 files to analyze");
            Console.WriteLine();

            var hierarchyAnalyzer = new MslkHierarchyAnalyzer();

            foreach (var filePath in pm4Files)
            {
                var fileName = Path.GetFileName(filePath);
                Console.WriteLine($"📄 ANALYZING: {fileName}");
                Console.WriteLine(new string('─', 60));

                try
                {
                    var pm4File = PM4File.FromFile(filePath);

                    if (pm4File.MSLK?.Entries == null || !pm4File.MSLK.Entries.Any())
                    {
                        Console.WriteLine("   ⚠️  No MSLK chunk or entries found");
                        Console.WriteLine();
                        continue;
                    }

                    Console.WriteLine($"   📊 MSLK Entries: {pm4File.MSLK.Entries.Count}");

                    // Perform hierarchy analysis
                    var hierarchyResult = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);

                    // Generate and display detailed report
                    var report = hierarchyAnalyzer.GenerateHierarchyReport(hierarchyResult, fileName);
                    Console.WriteLine(report);

                    // Export TXT
                    var outputDir = Path.Combine("output");
                    Directory.CreateDirectory(outputDir);
                    var txtPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}.mslk.txt");
                    File.WriteAllText(txtPath, report);
                    Console.WriteLine($"[TXT] Analysis written to: {txtPath}");

                    // Export YAML
                    var yamlPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(fileName)}.mslk.yaml");
                    var serializer = new SerializerBuilder()
                        .WithNamingConvention(CamelCaseNamingConvention.Instance)
                        .Build();
                    var yaml = serializer.Serialize(hierarchyResult);
                    File.WriteAllText(yamlPath, yaml);
                    Console.WriteLine($"[YAML] Analysis written to: {yamlPath}");

                    // Generate and display Mermaid diagram
                    Console.WriteLine("=== MERMAID HIERARCHY DIAGRAM ===");
                    Console.WriteLine("```mermaid");
                    var mermaid = hierarchyAnalyzer.GenerateHierarchyMermaid(hierarchyResult, fileName, 15);
                    Console.WriteLine(mermaid);
                    Console.WriteLine("```");
                    Console.WriteLine();

                    // Show raw data samples for manual inspection
                    Console.WriteLine("=== RAW DATA SAMPLES FOR MANUAL INSPECTION ===");
                    var samples = pm4File.MSLK.Entries.Take(10);
                    foreach (var (entry, index) in samples.Select((e, i) => (e, i)))
                    {
                        Console.WriteLine($"Entry {index}:");
                        Console.WriteLine($"  Flags_0x00: 0x{entry.Unknown_0x00:X2} ({entry.Unknown_0x00})");
                        Console.WriteLine($"  Sequence_0x01: {entry.Unknown_0x01}");
                        Console.WriteLine($"  Unknown_0x02: 0x{entry.Unknown_0x02:X4}");
                        Console.WriteLine($"  ParentIndex_0x04: {entry.Unknown_0x04}");
                        Console.WriteLine($"  MSPI_first_index: {entry.MspiFirstIndex}");
                        Console.WriteLine($"  MSPI_index_count: {entry.MspiIndexCount}");
                        Console.WriteLine($"  Unknown_0x0C: 0x{entry.Unknown_0x0C:X8}");
                        Console.WriteLine($"  CrossRef_0x10: {entry.Unknown_0x10}");
                        Console.WriteLine($"  Constant_0x12: 0x{entry.Unknown_0x12:X4}");
                        Console.WriteLine($"  Type: {(entry.MspiFirstIndex == -1 ? "Doodad Node" : "Geometry Node")}");
                        Console.WriteLine();
                    }

                    Console.WriteLine(new string('═', 80));
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error analyzing {fileName}: {ex.Message}");
                    Console.WriteLine();
                }
            }

            Console.WriteLine("✅ MSLK hierarchy analysis complete!");
            Console.WriteLine();
            Console.WriteLine("📋 NEXT STEPS:");
            Console.WriteLine("   1. Examine the Mermaid diagrams to see discovered hierarchies");
            Console.WriteLine("   2. Look for patterns in the raw data samples");
            Console.WriteLine("   3. Check if Parent Index relationships make logical sense");
            Console.WriteLine("   4. Validate hierarchy depth vs. object types");
        }

        /// <summary>
        /// Quick analysis of a single PM4 file - useful for focused testing
        /// </summary>
        public static void AnalyzeSingleFile(string pm4FilePath)
        {
            if (!File.Exists(pm4FilePath))
            {
                Console.WriteLine($"❌ File not found: {pm4FilePath}");
                return;
            }

            var fileName = Path.GetFileName(pm4FilePath);
            Console.WriteLine($"🎯 SINGLE FILE ANALYSIS: {fileName}");
            Console.WriteLine(new string('═', 60));

            try
            {
                var pm4File = PM4File.FromFile(pm4FilePath);

                if (pm4File.MSLK?.Entries == null)
                {
                    Console.WriteLine("❌ No MSLK chunk found");
                    return;
                }

                var hierarchyAnalyzer = new MslkHierarchyAnalyzer();
                var result = hierarchyAnalyzer.AnalyzeHierarchy(pm4File.MSLK);

                // Quick summary
                Console.WriteLine($"📊 QUICK SUMMARY:");
                Console.WriteLine($"   Total Nodes: {result.AllNodes.Count}");
                Console.WriteLine($"   Root Nodes: {result.RootNodes.Count}");
                Console.WriteLine($"   Max Depth: {result.MaxDepth}");
                Console.WriteLine($"   Geometry Nodes: {result.GeometryNodeCount}");
                Console.WriteLine($"   Doodad Nodes: {result.DoodadNodeCount}");
                Console.WriteLine();

                // Mermaid only
                Console.WriteLine("```mermaid");
                var mermaid = hierarchyAnalyzer.GenerateHierarchyMermaid(result, fileName, 20);
                Console.WriteLine(mermaid);
                Console.WriteLine("```");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
        }
    }
} 