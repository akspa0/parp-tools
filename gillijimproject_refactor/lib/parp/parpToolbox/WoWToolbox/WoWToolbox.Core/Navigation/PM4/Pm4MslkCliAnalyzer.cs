using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// CLI analyzer for MSLK relationships that outputs Mermaid diagrams to console logs.
    /// Integrates with existing PM4 analysis pipeline for enhanced debugging.
    /// </summary>
    public static class Pm4MslkCliAnalyzer
    {
        private static readonly MslkRelationshipAnalyzer _analyzer = new();

        /// <summary>
        /// Analyzes a PM4 file and outputs MSLK relationship information with Mermaid diagrams to console.
        /// </summary>
        /// <param name="pm4File">The PM4 file to analyze</param>
        /// <param name="fileName">The filename for identification</param>
        /// <param name="outputMermaid">Whether to output Mermaid diagrams to console</param>
        public static void AnalyzeAndOutputMslkRelationships(PM4File pm4File, string fileName, bool outputMermaid = true)
        {
            Console.WriteLine();
            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine($"🔍 MSLK RELATIONSHIP ANALYSIS: {fileName}");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");

            var map = _analyzer.AnalyzeMslkRelationships(pm4File, fileName);

            if (map.Nodes.Count == 0)
            {
                Console.WriteLine("❌ No MSLK entries found in this PM4 file");
                return;
            }

            // Output summary statistics
            OutputSummaryStatistics(map);

            // Output detailed relationship analysis
            OutputDetailedAnalysis(map);

            // Output validation issues
            if (map.ValidationIssues.Any())
            {
                OutputValidationIssues(map);
            }

            // Output Mermaid diagram
            if (outputMermaid)
            {
                OutputMermaidDiagram(map);
            }

            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine();
        }

        /// <summary>
        /// Batch analyzes multiple PM4 files and outputs a summary comparison.
        /// </summary>
        /// <param name="pm4FilePaths">List of PM4 file paths to analyze</param>
        public static void BatchAnalyzeMslkPatterns(IEnumerable<string> pm4FilePaths)
        {
            Console.WriteLine();
            Console.WriteLine("🔍 BATCH MSLK PATTERN ANALYSIS");
            Console.WriteLine("═══════════════════════════════════════════════════════════════");

            var allMaps = new List<MslkRelationshipAnalyzer.MslkRelationshipMap>();

            foreach (var filePath in pm4FilePaths.Take(10)) // Limit to first 10 for readability
            {
                try
                {
                    var pm4File = PM4File.FromFile(filePath);
                    var fileName = Path.GetFileNameWithoutExtension(filePath);
                    var map = _analyzer.AnalyzeMslkRelationships(pm4File, fileName);
                    
                    if (map.Nodes.Count > 0)
                    {
                        allMaps.Add(map);
                        Console.WriteLine($"✅ {fileName}: {map.Nodes.Count} nodes, {map.GroupsByIndexReference.Count} groups");
                    }
                    else
                    {
                        Console.WriteLine($"⚠️  {fileName}: No MSLK data");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ {Path.GetFileName(filePath)}: Error - {ex.Message}");
                }
            }

            if (allMaps.Count > 0)
            {
                OutputBatchSummary(allMaps);
            }

            Console.WriteLine("═══════════════════════════════════════════════════════════════");
            Console.WriteLine();
        }

        private static void OutputSummaryStatistics(MslkRelationshipAnalyzer.MslkRelationshipMap map)
        {
            Console.WriteLine("📊 SUMMARY STATISTICS");
            Console.WriteLine($"├─ Total Nodes: {map.Nodes.Count}");
            Console.WriteLine($"├─ Doodad Nodes (MSPI = -1): {map.Nodes.Count(n => n.IsDoodadNode)}");
            Console.WriteLine($"├─ Geometry Nodes (MSPI >= 0): {map.Nodes.Count(n => n.IsGeometryNode)}");
            Console.WriteLine($"├─ Unique Groups (Index Ref): {map.GroupsByIndexReference.Count}");
            Console.WriteLine($"├─ Unique Flag Patterns: {map.NodesByFlags.Count}");
            Console.WriteLine($"├─ Unique MSUR References: {map.NodesByMsurIndex.Count(g => g.Key != 0)}");
            Console.WriteLine($"└─ Sequence Positions: {map.NodesBySequencePosition.Count}");
            Console.WriteLine();
        }

        private static void OutputDetailedAnalysis(MslkRelationshipAnalyzer.MslkRelationshipMap map)
        {
            // Flag patterns analysis
            Console.WriteLine("🏷️  FLAG PATTERNS (Object Types)");
            foreach (var flagGroup in map.NodesByFlags.OrderBy(g => g.Key))
            {
                var sampleNode = flagGroup.Value[0];
                var flagDesc = GetFlagDescription(flagGroup.Key);
                Console.WriteLine($"├─ 0x{flagGroup.Key:X2} ({sampleNode.FlagString}) - {flagDesc}: {flagGroup.Value.Count} nodes");
            }
            Console.WriteLine();

            // Group analysis
            Console.WriteLine("👥 GROUP ANALYSIS (Index Reference)");
            var significantGroups = map.GroupsByIndexReference.Where(g => g.Value.Count > 1).OrderBy(g => g.Key);
            if (significantGroups.Any())
            {
                foreach (var group in significantGroups.Take(5)) // Show top 5 groups
                {
                    var doodads = group.Value.Count(n => n.IsDoodadNode);
                    var geometry = group.Value.Count(n => n.IsGeometryNode);
                    Console.WriteLine($"├─ Group {group.Key}: {group.Value.Count} nodes (Doodads: {doodads}, Geometry: {geometry})");
                }
            }
            else
            {
                Console.WriteLine("├─ No multi-node groups found");
            }
            Console.WriteLine();

            // MSUR connections
            Console.WriteLine("🔗 MSUR SURFACE CONNECTIONS");
            var surfaceConnections = map.NodesByMsurIndex.Where(g => g.Key != 0).OrderBy(g => g.Key);
            if (surfaceConnections.Any())
            {
                foreach (var surface in surfaceConnections.Take(5)) // Show top 5 surfaces
                {
                    Console.WriteLine($"├─ MSUR {surface.Key}: {surface.Value.Count} nodes");
                }
            }
            else
            {
                Console.WriteLine("├─ No MSUR surface connections found");
            }
            Console.WriteLine();
        }

        private static void OutputValidationIssues(MslkRelationshipAnalyzer.MslkRelationshipMap map)
        {
            Console.WriteLine("⚠️  VALIDATION ISSUES");
            foreach (var issue in map.ValidationIssues.Take(10)) // Limit to first 10 issues
            {
                Console.WriteLine($"├─ {issue}");
            }
            if (map.ValidationIssues.Count > 10)
            {
                Console.WriteLine($"└─ ... and {map.ValidationIssues.Count - 10} more issues");
            }
            Console.WriteLine();
        }

        private static void OutputMermaidDiagram(MslkRelationshipAnalyzer.MslkRelationshipMap map)
        {
            Console.WriteLine("🎨 MERMAID RELATIONSHIP DIAGRAM");
            Console.WriteLine("├─ Copy the following diagram to view in Mermaid Live Editor:");
            Console.WriteLine("├─ https://mermaid.live/");
            Console.WriteLine("└─ Diagram:");
            Console.WriteLine();
            
            var mermaidCode = _analyzer.GenerateMermaidDiagram(map);
            
            // Output with console-friendly formatting
            Console.WriteLine("```mermaid");
            Console.WriteLine(mermaidCode);
            Console.WriteLine("```");
            Console.WriteLine();
        }

        private static void OutputBatchSummary(List<MslkRelationshipAnalyzer.MslkRelationshipMap> maps)
        {
            Console.WriteLine();
            Console.WriteLine("📈 BATCH ANALYSIS SUMMARY");
            
            var totalNodes = maps.Sum(m => m.Nodes.Count);
            var totalDoodads = maps.Sum(m => m.Nodes.Count(n => n.IsDoodadNode));
            var totalGeometry = maps.Sum(m => m.Nodes.Count(n => n.IsGeometryNode));
            
            Console.WriteLine($"├─ Files Analyzed: {maps.Count}");
            Console.WriteLine($"├─ Total Nodes: {totalNodes}");
            Console.WriteLine($"├─ Total Doodads: {totalDoodads}");
            Console.WriteLine($"├─ Total Geometry: {totalGeometry}");
            
            // Flag pattern frequency across all files
            var allFlags = maps.SelectMany(m => m.NodesByFlags.Keys).Distinct().OrderBy(f => f);
            Console.WriteLine($"├─ Unique Flag Patterns: {allFlags.Count()}");
            
            Console.WriteLine("└─ Common Flag Patterns:");
            foreach (var flag in allFlags.Take(5))
            {
                var count = maps.Sum(m => m.NodesByFlags.ContainsKey(flag) ? m.NodesByFlags[flag].Count : 0);
                var flagDesc = GetFlagDescription(flag);
                Console.WriteLine($"   ├─ 0x{flag:X2} ({flagDesc}): {count} total nodes");
            }
        }

        private static string GetFlagDescription(byte flags)
        {
            var descriptions = new List<string>();
            
            if ((flags & 0x01) != 0) descriptions.Add("Type1");
            if ((flags & 0x02) != 0) descriptions.Add("Type2");
            if ((flags & 0x04) != 0) descriptions.Add("Type4");
            if ((flags & 0x08) != 0) descriptions.Add("Type8");
            if ((flags & 0x10) != 0) descriptions.Add("Type16");
            
            return descriptions.Any() ? string.Join("|", descriptions) : "None";
        }

        /// <summary>
        /// Creates a simple Mermaid diagram for direct console output without complex formatting.
        /// </summary>
        public static string CreateSimpleMermaidDiagram(MslkRelationshipAnalyzer.MslkRelationshipMap map)
        {
            var sb = new StringBuilder();
            sb.AppendLine("graph LR");
            
            // Simple node representation
            foreach (var node in map.Nodes.Take(20)) // Limit for readability
            {
                var nodeType = node.IsDoodadNode ? "D" : "G";
                var nodeId = $"{nodeType}{node.Index}";
                sb.AppendLine($"    {nodeId}[\"{nodeType}{node.Index}<br/>F:{node.Flags:X2}\"]");
            }
            
            // Simple group connections
            foreach (var group in map.GroupsByIndexReference.Where(g => g.Value.Count > 1).Take(3))
            {
                var nodes = group.Value.Take(5);
                for (int i = 0; i < nodes.Count() - 1; i++)
                {
                    var current = nodes.ElementAt(i);
                    var next = nodes.ElementAt(i + 1);
                    var currentId = (current.IsDoodadNode ? "D" : "G") + current.Index;
                    var nextId = (next.IsDoodadNode ? "D" : "G") + next.Index;
                    sb.AppendLine($"    {currentId} --- {nextId}");
                }
            }
            
            return sb.ToString();
        }
    }
} 