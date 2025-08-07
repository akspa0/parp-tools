using System;
using System.IO;
using System.Linq;
using Xunit;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class MslkRelationshipAnalysisTest
    {
        [Fact]
        public void RunMslkAnalysisDemo()
        {
            // Run the comprehensive demo
            MslkAnalysisDemo.RunDemo();
        }

        [Fact]
        public void TestMslkRelationshipAnalyzerDirectly()
        {
            // Test the analyzer directly with validation
            var testDataPaths = new[]
            {
                "test_data/development",
                "test_data/development_335",
                "test_data/original_development/development"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    break;
                }
            }

            if (testDataDir == null) return;

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                .Take(1) // Test just one file for unit testing
                .ToArray();

            if (pm4Files.Length == 0) return;

            var pm4File = PM4File.FromFile(pm4Files[0]);
            var analyzer = new MslkRelationshipAnalyzer();
            
            var relationshipMap = analyzer.AnalyzeMslkRelationships(pm4File, Path.GetFileName(pm4Files[0]));
            
            // Verify the analysis produces valid results
            Assert.NotNull(relationshipMap);
            Assert.NotNull(relationshipMap.Nodes);
            Assert.NotNull(relationshipMap.GroupsByIndexReference);
            Assert.NotNull(relationshipMap.NodesByFlags);
            Assert.NotNull(relationshipMap.NodesByMsurIndex);
            Assert.NotNull(relationshipMap.ValidationIssues);

            // Generate Mermaid diagram
            var mermaidDiagram = analyzer.GenerateMermaidDiagram(relationshipMap);
            Assert.NotNull(mermaidDiagram);
            Assert.Contains("graph TD", mermaidDiagram);

            Console.WriteLine($"✅ Direct analyzer test completed");
            Console.WriteLine($"📊 Found {relationshipMap.Nodes.Count} MSLK nodes");
            Console.WriteLine($"🏷️ Found {relationshipMap.NodesByFlags.Count} unique flag patterns");
            Console.WriteLine($"👥 Found {relationshipMap.GroupsByIndexReference.Count} index reference groups");
            
            if (relationshipMap.ValidationIssues.Any())
            {
                Console.WriteLine($"⚠️ Found {relationshipMap.ValidationIssues.Count} validation issues");
            }
        }

        [Fact]
        public void TestBatchMslkAnalysis()
        {
            // Test batch analysis functionality
            var testDataPaths = new[]
            {
                "test_data/development",
                "test_data/development_335",  
                "test_data/original_development/development"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    break;
                }
            }

            if (testDataDir == null) return;

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                .Take(5) // Test first 5 files for batch analysis
                .ToArray();

            if (pm4Files.Length == 0) return;

            Console.WriteLine($"🔄 TESTING BATCH MSLK ANALYSIS");
            
            // Use the CLI analyzer for batch processing
            Pm4MslkCliAnalyzer.BatchAnalyzeMslkPatterns(pm4Files);
            
            Console.WriteLine("✅ Batch analysis completed");
        }
    }
} 