using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Analyzes MSLK chunk as a hierarchical node-link system.
    /// Focuses on discovering object hierarchies and relationships between entries.
    /// 
    /// Hypothesis: MSLK represents a scene graph or object hierarchy where:
    /// - Entries are nodes in a tree/graph structure
    /// - Unknown fields represent various types of links/relationships
    /// - Some entries are parents, others are children or siblings
    /// - MSPI links connect to geometry, other fields connect to hierarchy
    /// </summary>
    public class MslkHierarchyAnalyzer
    {
        public class HierarchyNode
        {
            public int Index { get; set; }
            public byte Flags_0x00 { get; set; }           // Object type/flags
            public byte Sequence_0x01 { get; set; }        // Position in sequence
            public uint ParentIndex_0x04 { get; set; }     // Potential parent node index
            public int MspiFirstIndex { get; set; }        // Geometry link (-1 = no geometry)
            public byte MspiIndexCount { get; set; }       // Geometry extent
            public uint Unknown_0x0C { get; set; }         // Additional data/flags
            public ushort CrossReference_0x10 { get; set; } // Cross-reference to other structures
            
            // Derived properties
            public bool HasGeometry => MspiFirstIndex >= 0;
            public bool IsDoodadNode => MspiFirstIndex == -1;
            
            // Hierarchy relationships
            public List<HierarchyNode> Children { get; set; } = new();
            public HierarchyNode? Parent { get; set; }
            public int Depth { get; set; } = 0;
        }

        public class HierarchyAnalysisResult
        {
            public List<HierarchyNode> AllNodes { get; set; } = new();
            public List<HierarchyNode> RootNodes { get; set; } = new();
            public Dictionary<uint, List<HierarchyNode>> NodesByParentIndex { get; set; } = new();
            public Dictionary<byte, List<HierarchyNode>> NodesByFlags { get; set; } = new();
            public Dictionary<byte, List<HierarchyNode>> NodesBySequence { get; set; } = new();
            
            public int MaxDepth { get; set; }
            public int GeometryNodeCount { get; set; }
            public int DoodadNodeCount { get; set; }
            
            public List<string> DiscoveredPatterns { get; set; } = new();
        }

        /// <summary>
        /// Analyzes MSLK entries to discover hierarchical relationships
        /// </summary>
        public HierarchyAnalysisResult AnalyzeHierarchy(MSLK mslkChunk)
        {
            var result = new HierarchyAnalysisResult();
            
            if (mslkChunk?.Entries == null || !mslkChunk.Entries.Any())
            {
                result.DiscoveredPatterns.Add("No MSLK entries to analyze");
                return result;
            }

            // Convert entries to hierarchy nodes
            ConvertToHierarchyNodes(mslkChunk, result);
            
            // Analyze potential parent-child relationships
            AnalyzeParentChildRelationships(result);
            
            // Group nodes by various criteria to find patterns
            GroupNodesByPatterns(result);
            
            // Detect hierarchy patterns
            DetectHierarchyPatterns(result);
            
            // Calculate hierarchy statistics
            CalculateHierarchyStats(result);
            
            return result;
        }

        private void ConvertToHierarchyNodes(MSLK mslkChunk, HierarchyAnalysisResult result)
        {
            for (int i = 0; i < mslkChunk.Entries.Count; i++)
            {
                var entry = mslkChunk.Entries[i];
                var node = new HierarchyNode
                {
                    Index = i,
                    Flags_0x00 = entry.Unknown_0x00,
                    Sequence_0x01 = entry.Unknown_0x01,
                    ParentIndex_0x04 = entry.Unknown_0x04,
                    MspiFirstIndex = entry.MspiFirstIndex,
                    MspiIndexCount = entry.MspiIndexCount,
                    Unknown_0x0C = entry.Unknown_0x0C,
                    CrossReference_0x10 = entry.Unknown_0x10
                };
                
                result.AllNodes.Add(node);
                
                if (node.HasGeometry)
                    result.GeometryNodeCount++;
                else
                    result.DoodadNodeCount++;
            }
        }

        private void AnalyzeParentChildRelationships(HierarchyAnalysisResult result)
        {
            // Group nodes by potential parent index (0x04 field)
            result.NodesByParentIndex = result.AllNodes
                .GroupBy(n => n.ParentIndex_0x04)
                .ToDictionary(g => g.Key, g => g.ToList());

            // Hypothesis 1: 0x04 field represents parent index
            foreach (var node in result.AllNodes)
            {
                // Check if ParentIndex_0x04 could be an index into the MSLK array
                if (node.ParentIndex_0x04 < result.AllNodes.Count)
                {
                    var potentialParent = result.AllNodes[(int)node.ParentIndex_0x04];
                    
                    // Only create parent-child if it makes logical sense
                    if (potentialParent.Index != node.Index && // Not self-referencing
                        !WouldCreateCycle(node, potentialParent, result)) // No cycles
                    {
                        node.Parent = potentialParent;
                        potentialParent.Children.Add(node);
                    }
                }
            }

            // Find root nodes (nodes with no parent)
            result.RootNodes = result.AllNodes.Where(n => n.Parent == null).ToList();
            
            // Calculate depths
            CalculateDepths(result.RootNodes);
        }

        private bool WouldCreateCycle(HierarchyNode child, HierarchyNode potentialParent, HierarchyAnalysisResult result)
        {
            // Simple cycle detection: check if potentialParent is already a descendant of child
            var visited = new HashSet<int>();
            return IsDescendant(potentialParent, child.Index, visited);
        }

        private bool IsDescendant(HierarchyNode node, int ancestorIndex, HashSet<int> visited)
        {
            if (visited.Contains(node.Index))
                return false; // Prevent infinite loops
                
            visited.Add(node.Index);
            
            if (node.Index == ancestorIndex)
                return true;
                
            return node.Children.Any(c => IsDescendant(c, ancestorIndex, visited));
        }

        private void CalculateDepths(List<HierarchyNode> rootNodes)
        {
            foreach (var root in rootNodes)
            {
                CalculateDepthRecursive(root, 0);
            }
        }

        private void CalculateDepthRecursive(HierarchyNode node, int depth)
        {
            node.Depth = depth;
            foreach (var child in node.Children)
            {
                CalculateDepthRecursive(child, depth + 1);
            }
        }

        private void GroupNodesByPatterns(HierarchyAnalysisResult result)
        {
            // Group by flags (0x00) - likely object types
            result.NodesByFlags = result.AllNodes
                .GroupBy(n => n.Flags_0x00)
                .ToDictionary(g => g.Key, g => g.ToList());

            // Group by sequence (0x01) - likely position in sequences
            result.NodesBySequence = result.AllNodes
                .GroupBy(n => n.Sequence_0x01)
                .ToDictionary(g => g.Key, g => g.ToList());
        }

        private void DetectHierarchyPatterns(HierarchyAnalysisResult result)
        {
            // Pattern 1: Check if flags correlate with hierarchy levels
            var flagsByDepth = result.AllNodes
                .GroupBy(n => n.Depth)
                .ToDictionary(g => g.Key, g => g.Select(n => n.Flags_0x00).Distinct().ToList());

            if (flagsByDepth.Any())
            {
                result.DiscoveredPatterns.Add($"Hierarchy levels found: {flagsByDepth.Count}");
                foreach (var kvp in flagsByDepth.OrderBy(x => x.Key))
                {
                    result.DiscoveredPatterns.Add($"  Depth {kvp.Key}: Flags [{string.Join(", ", kvp.Value.Select(f => $"0x{f:X2}"))}]");
                }
            }

            // Pattern 2: Check sequence ordering within flag groups
            foreach (var flagGroup in result.NodesByFlags)
            {
                var sequences = flagGroup.Value.Select(n => n.Sequence_0x01).Distinct().OrderBy(x => x).ToList();
                if (sequences.Count > 1)
                {
                    result.DiscoveredPatterns.Add($"Flag 0x{flagGroup.Key:X2} has sequence ordering: {string.Join(", ", sequences)}");
                }
            }

            // Pattern 3: Geometry vs Doodad distribution in hierarchy
            var geometryAtDepth = result.AllNodes.Where(n => n.HasGeometry).GroupBy(n => n.Depth).ToDictionary(g => g.Key, g => g.Count());
            var doodadAtDepth = result.AllNodes.Where(n => n.IsDoodadNode).GroupBy(n => n.Depth).ToDictionary(g => g.Key, g => g.Count());
            
            result.DiscoveredPatterns.Add("Geometry/Doodad distribution by depth:");
            var allDepths = geometryAtDepth.Keys.Union(doodadAtDepth.Keys).OrderBy(x => x);
            foreach (var depth in allDepths)
            {
                var geomCount = geometryAtDepth.GetValueOrDefault(depth, 0);
                var doodCount = doodadAtDepth.GetValueOrDefault(depth, 0);
                result.DiscoveredPatterns.Add($"  Depth {depth}: {geomCount} geometry, {doodCount} doodad nodes");
            }
        }

        private void CalculateHierarchyStats(HierarchyAnalysisResult result)
        {
            result.MaxDepth = result.AllNodes.Any() ? result.AllNodes.Max(n => n.Depth) : 0;
        }

        /// <summary>
        /// Generates a Mermaid diagram showing the discovered hierarchy
        /// </summary>
        public string GenerateHierarchyMermaid(HierarchyAnalysisResult analysis, string pm4FileName, int maxNodesToShow = 20)
        {
            var sb = new StringBuilder();
            
            sb.AppendLine("graph TD");
            sb.AppendLine($"    subgraph PM4[\"{pm4FileName} - Object Hierarchy\"]");
            
            // Show root nodes and their immediate children
            var nodesToShow = new HashSet<HierarchyNode>();
            var nodeQueue = new Queue<HierarchyNode>();
            
            // Start with root nodes
            foreach (var root in analysis.RootNodes.Take(5)) // Limit root nodes shown
            {
                nodeQueue.Enqueue(root);
            }
            
            // Breadth-first traversal to collect nodes to show
            while (nodeQueue.Count > 0 && nodesToShow.Count < maxNodesToShow)
            {
                var node = nodeQueue.Dequeue();
                if (nodesToShow.Contains(node)) continue;
                
                nodesToShow.Add(node);
                
                // Add children to queue
                foreach (var child in node.Children.Take(3)) // Limit children per node
                {
                    if (!nodesToShow.Contains(child))
                        nodeQueue.Enqueue(child);
                }
            }
            
            // Generate node definitions
            foreach (var node in nodesToShow)
            {
                var shape = node.HasGeometry ? "rect" : "circle";
                var nodeType = node.HasGeometry ? "GEOM" : "DOOD";
                var label = $"{nodeType}_{node.Index}<br/>Flag:0x{node.Flags_0x00:X2}<br/>Seq:{node.Sequence_0x01}";
                
                if (node.HasGeometry)
                {
                    sb.AppendLine($"    N{node.Index}[\"{label}\"]");
                }
                else
                {
                    sb.AppendLine($"    N{node.Index}((\"{label}\"))");
                }
            }
            
            // Generate relationships
            foreach (var node in nodesToShow)
            {
                foreach (var child in node.Children.Where(c => nodesToShow.Contains(c)))
                {
                    sb.AppendLine($"    N{node.Index} --> N{child.Index}");
                }
            }
            
            sb.AppendLine("    end");
            
            // Add legend
            sb.AppendLine();
            sb.AppendLine("    subgraph Legend[\"Legend\"]");
            sb.AppendLine("        GEOM[\"Geometry Node\"]");
            sb.AppendLine("        DOOD((\"Doodad Node\"))");
            sb.AppendLine("    end");
            
            return sb.ToString();
        }

        /// <summary>
        /// Generates a detailed hierarchy analysis report
        /// </summary>
        public string GenerateHierarchyReport(HierarchyAnalysisResult analysis, string pm4FileName)
        {
            var sb = new StringBuilder();
            
            sb.AppendLine("=== MSLK HIERARCHY ANALYSIS ===");
            sb.AppendLine($"File: {pm4FileName}");
            sb.AppendLine($"Total Nodes: {analysis.AllNodes.Count}");
            sb.AppendLine($"Root Nodes: {analysis.RootNodes.Count}");
            sb.AppendLine($"Max Depth: {analysis.MaxDepth}");
            sb.AppendLine($"Geometry Nodes: {analysis.GeometryNodeCount}");
            sb.AppendLine($"Doodad Nodes: {analysis.DoodadNodeCount}");
            sb.AppendLine();

            // Show discovered patterns
            sb.AppendLine("=== DISCOVERED PATTERNS ===");
            foreach (var pattern in analysis.DiscoveredPatterns)
            {
                sb.AppendLine($"• {pattern}");
            }
            sb.AppendLine();

            // Show parent index distribution
            sb.AppendLine("=== PARENT INDEX DISTRIBUTION ===");
            sb.AppendLine("Top parent indices (potential hierarchy roots):");
            var topParentIndices = analysis.NodesByParentIndex
                .OrderByDescending(kvp => kvp.Value.Count)
                .Take(10);
            
            foreach (var kvp in topParentIndices)
            {
                sb.AppendLine($"  Parent Index {kvp.Key}: {kvp.Value.Count} children");
            }
            sb.AppendLine();

            // Show flag distribution
            sb.AppendLine("=== FLAG DISTRIBUTION ===");
            foreach (var kvp in analysis.NodesByFlags.OrderBy(x => x.Key))
            {
                var geomCount = kvp.Value.Count(n => n.HasGeometry);
                var doodCount = kvp.Value.Count(n => n.IsDoodadNode);
                sb.AppendLine($"  Flag 0x{kvp.Key:X2}: {kvp.Value.Count} total ({geomCount} geom, {doodCount} doodad)");
            }
            sb.AppendLine();

            // Show sample hierarchy trees
            sb.AppendLine("=== SAMPLE HIERARCHY TREES ===");
            foreach (var root in analysis.RootNodes.Take(3))
            {
                sb.AppendLine($"Root Node {root.Index} (Flag: 0x{root.Flags_0x00:X2}):");
                PrintHierarchyTree(sb, root, "  ", 3); // Max depth of 3 for readability
                sb.AppendLine();
            }

            return sb.ToString();
        }

        private void PrintHierarchyTree(StringBuilder sb, HierarchyNode node, string indent, int maxDepth)
        {
            if (maxDepth <= 0) return;
            
            var nodeType = node.HasGeometry ? "GEOM" : "DOOD";
            sb.AppendLine($"{indent}{nodeType}_{node.Index} [Flag:0x{node.Flags_0x00:X2}, Seq:{node.Sequence_0x01}]");
            
            foreach (var child in node.Children.Take(5)) // Limit children shown
            {
                PrintHierarchyTree(sb, child, indent + "  ", maxDepth - 1);
            }
            
            if (node.Children.Count > 5)
            {
                sb.AppendLine($"{indent}  ... and {node.Children.Count - 5} more children");
            }
        }
    }
} 