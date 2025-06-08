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
            [System.Text.Json.Serialization.JsonIgnore]
            public List<HierarchyNode> Children { get; set; } = new();
            [System.Text.Json.Serialization.JsonIgnore]
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
            
            // Investigate alternative hierarchy indicators
            InvestigateAlternativeHierarchyFields(result);
            
            // Calculate hierarchy statistics
            CalculateHierarchyStats(result);
            
            // Final validation to catch any remaining cycles
            ValidateHierarchyIntegrity(result);
            
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

            // ✨ NEW APPROACH: Investigate what ParentIndex_0x04 actually represents
            AnalyzeParentIndexPatterns(result);

            // ✨ SIMPLIFIED: NO HIERARCHY BUILDING - Treat each node as independent
            // This eliminates ALL circular reference issues
            
            var selfReferencingNodes = 0;
            var nonSelfReferencingNodes = 0;
            
            foreach (var node in result.AllNodes)
            {
                // Analyze patterns but DON'T build parent-child relationships
                if (node.ParentIndex_0x04 < result.AllNodes.Count)
                {
                    if (node.ParentIndex_0x04 == node.Index)
                    {
                        selfReferencingNodes++;
                        result.DiscoveredPatterns.Add($"Node {node.Index} self-references (primary object candidate)");
                    }
                    else
                    {
                        nonSelfReferencingNodes++;
                        result.DiscoveredPatterns.Add($"Node {node.Index} references {node.ParentIndex_0x04} (potential grouping)");
                    }
                }
                
                // Leave all Parent/Children relationships empty to prevent cycles
                // node.Parent = null; (already null)
                // node.Children = new(); (already empty)
            }

            result.DiscoveredPatterns.Add($"SUMMARY: {selfReferencingNodes} self-refs, {nonSelfReferencingNodes} other refs");
            
            // All nodes are "roots" since we don't build hierarchy
            result.RootNodes = result.AllNodes.ToList();
            
            // No depth calculation needed since no hierarchy
        }

        private void AnalyzeParentIndexPatterns(HierarchyAnalysisResult result)
        {
            var selfReferences = result.AllNodes.Count(n => n.ParentIndex_0x04 == n.Index);
            var validParentRefs = result.AllNodes.Count(n => n.ParentIndex_0x04 < result.AllNodes.Count && n.ParentIndex_0x04 != n.Index);
            var invalidRefs = result.AllNodes.Count(n => n.ParentIndex_0x04 >= result.AllNodes.Count);
            
            result.DiscoveredPatterns.Add($"PARENT INDEX ANALYSIS:");
            result.DiscoveredPatterns.Add($"  Self-references: {selfReferences} ({selfReferences * 100.0 / result.AllNodes.Count:F1}%)");
            result.DiscoveredPatterns.Add($"  Valid parent refs: {validParentRefs} ({validParentRefs * 100.0 / result.AllNodes.Count:F1}%)");
            result.DiscoveredPatterns.Add($"  Invalid refs: {invalidRefs} ({invalidRefs * 100.0 / result.AllNodes.Count:F1}%)");
            
            if (selfReferences > result.AllNodes.Count * 0.5)
            {
                result.DiscoveredPatterns.Add("HYPOTHESIS: ParentIndex_0x04 may represent GROUP MEMBERSHIP, not hierarchy");
                result.DiscoveredPatterns.Add("  - Self-references likely indicate 'group leaders' or 'primary objects'");
                result.DiscoveredPatterns.Add("  - This is NOT a traditional parent-child tree structure");
            }
        }

        private bool WouldCreateCycle(HierarchyNode child, HierarchyNode potentialParent, HierarchyAnalysisResult result)
        {
            // Enhanced cycle detection with multiple checks
            
            // Check 1: Direct self-reference
            if (child.Index == potentialParent.Index)
                return true;
            
            // Check 2: Would potentialParent become a descendant of child?
            var visited = new HashSet<int>();
            if (IsDescendant(potentialParent, child.Index, visited))
                return true;
            
            // Check 3: Would child become an ancestor of potentialParent?
            visited.Clear();
            if (IsAncestor(child, potentialParent.Index, visited))
                return true;
                
            return false;
        }
        
        private bool IsAncestor(HierarchyNode node, int descendantIndex, HashSet<int> visited)
        {
            if (visited.Contains(node.Index))
                return false; // Prevent infinite loops
                
            visited.Add(node.Index);
            
            if (node.Index == descendantIndex)
                return true;
                
            // Check if any ancestor of this node is the descendant
            return node.Parent != null && IsAncestor(node.Parent, descendantIndex, visited);
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

        private bool IsAlreadyParentOf(HierarchyNode potentialParent, HierarchyNode child)
        {
            return child.Parent?.Index == potentialParent.Index;
        }

        private bool IsAlreadyChildOf(HierarchyNode potentialChild, HierarchyNode parent)
        {
            return parent.Children.Any(c => c.Index == potentialChild.Index);
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
            var visited = new HashSet<int>();
            CalculateDepthRecursiveInternal(node, depth, visited);
        }

        private void CalculateDepthRecursiveInternal(HierarchyNode node, int depth, HashSet<int> visited)
        {
            // Prevent infinite loops with cycle detection
            if (visited.Contains(node.Index))
                return;
                
            // Safety limit to prevent extremely deep recursion
            if (depth > 100)
            {
                Console.WriteLine($"Warning: Maximum depth limit (100) reached at node {node.Index}");
                return;
            }
                
            visited.Add(node.Index);
            
            node.Depth = depth;
            foreach (var child in node.Children)
            {
                CalculateDepthRecursiveInternal(child, depth + 1, visited);
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

        private void InvestigateAlternativeHierarchyFields(HierarchyAnalysisResult result)
        {
            result.DiscoveredPatterns.Add("INVESTIGATING ALTERNATIVE HIERARCHY INDICATORS:");
            
            // Look at Sequence_0x01 as potential hierarchy indicator
            var sequenceGroups = result.AllNodes.GroupBy(n => n.Sequence_0x01).Where(g => g.Count() > 1).ToList();
            result.DiscoveredPatterns.Add($"  Sequence field groups: {sequenceGroups.Count}");
            
            // Look at CrossReference_0x10 as potential parent/sibling indicator
            var crossRefGroups = result.AllNodes.GroupBy(n => n.CrossReference_0x10).Where(g => g.Count() > 1).ToList();
            result.DiscoveredPatterns.Add($"  Cross-reference groups: {crossRefGroups.Count}");
            
            // Look at Unknown_0x0C as potential grouping field
            var unknownGroups = result.AllNodes.GroupBy(n => n.Unknown_0x0C).Where(g => g.Count() > 1).ToList();
            result.DiscoveredPatterns.Add($"  Unknown_0x0C groups: {unknownGroups.Count}");
            
            // Analyze MSPI groupings (geometry nodes)
            var geometryNodes = result.AllNodes.Where(n => n.HasGeometry).ToList();
            if (geometryNodes.Any())
            {
                var mspiRanges = geometryNodes.GroupBy(n => n.MspiFirstIndex / 100).ToList(); // Group by ranges
                result.DiscoveredPatterns.Add($"  MSPI index ranges: {mspiRanges.Count}");
            }
            
            result.DiscoveredPatterns.Add("RECOMMENDATION: Use self-referencing nodes as primary objects for export");
        }

        private void ValidateHierarchyIntegrity(HierarchyAnalysisResult result)
        {
            var issuesFound = 0;
            
            foreach (var node in result.AllNodes)
            {
                // Check for self-reference in children (should be rare now)
                if (node.Children.Any(c => c.Index == node.Index))
                {
                    Console.WriteLine($"ERROR: Node {node.Index} has itself as a child! Removing self-reference.");
                    node.Children.RemoveAll(c => c.Index == node.Index);
                    issuesFound++;
                }
                
                // Check for parent-child inconsistency
                foreach (var child in node.Children.ToList())
                {
                    if (child.Parent?.Index != node.Index)
                    {
                        Console.WriteLine($"WARNING: Parent-child inconsistency detected. Node {child.Index} is child of {node.Index} but parent is {child.Parent?.Index}");
                        // Fix the inconsistency
                        child.Parent = node;
                    }
                }
            }
            
            if (issuesFound > 0)
            {
                Console.WriteLine($"Hierarchy validation found and fixed {issuesFound} circular reference issues.");
            }
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
            var visited = new HashSet<int>();
            PrintHierarchyTreeRecursive(sb, node, indent, maxDepth, visited);
        }

        private void PrintHierarchyTreeRecursive(StringBuilder sb, HierarchyNode node, string indent, int maxDepth, HashSet<int> visited)
        {
            if (maxDepth <= 0) return;
            
            // Prevent infinite loops with cycle detection
            if (visited.Contains(node.Index))
            {
                sb.AppendLine($"{indent}[CYCLE DETECTED - Node {node.Index} already visited]");
                return;
            }
                
            visited.Add(node.Index);
            
            var nodeType = node.HasGeometry ? "GEOM" : "DOOD";
            sb.AppendLine($"{indent}{nodeType}_{node.Index} [Flag:0x{node.Flags_0x00:X2}, Seq:{node.Sequence_0x01}]");
            
            foreach (var child in node.Children.Take(5)) // Limit children shown
            {
                PrintHierarchyTreeRecursive(sb, child, indent + "  ", maxDepth - 1, visited);
            }
            
            if (node.Children.Count > 5)
            {
                sb.AppendLine($"{indent}  ... and {node.Children.Count - 5} more children");
            }
        }

        /// <summary>
        /// ✨ REVISED: Segments based on self-referencing "group leaders" instead of traditional hierarchy
        /// Each self-referencing node represents a primary object
        /// </summary>
        public List<ObjectSegmentationResult> SegmentObjectsByHierarchy(HierarchyAnalysisResult analysis)
        {
            var results = new List<ObjectSegmentationResult>();
            
            // Find self-referencing nodes (group leaders/primary objects)
            var groupLeaders = analysis.AllNodes.Where(n => n.ParentIndex_0x04 == n.Index).ToList();
            
            if (groupLeaders.Any())
            {
                // Use group leaders as primary objects
                foreach (var leader in groupLeaders)
                {
                    var geom = new List<int>();
                    var dood = new List<int>();
                    
                    // Include the leader itself
                    if (leader.HasGeometry)
                        geom.Add(leader.Index);
                    else if (leader.IsDoodadNode)
                        dood.Add(leader.Index);
                    
                    // Include any children (if traditional hierarchy exists)
                    TraverseAndCollect(leader, geom, dood);
                    
                    results.Add(new ObjectSegmentationResult
                    {
                        RootIndex = leader.Index,
                        GeometryNodeIndices = geom,
                        DoodadNodeIndices = dood,
                        SegmentationType = "group_leader"
                    });
                }
            }
            else
            {
                // Fallback to traditional root-based approach
                foreach (var root in analysis.RootNodes)
                {
                    var geom = new List<int>();
                    var dood = new List<int>();
                    TraverseAndCollect(root, geom, dood);
                    results.Add(new ObjectSegmentationResult
                    {
                        RootIndex = root.Index,
                        GeometryNodeIndices = geom,
                        DoodadNodeIndices = dood,
                        SegmentationType = "root_hierarchy"
                    });
                }
            }
            
            return results;
        }

        /// <summary>
        /// ✨ NEW: Segments each individual geometry node as a separate object
        /// This provides the finest granularity - one OBJ file per geometry node
        /// </summary>
        public List<ObjectSegmentationResult> SegmentByIndividualGeometry(HierarchyAnalysisResult analysis)
        {
            var results = new List<ObjectSegmentationResult>();
            
            foreach (var node in analysis.AllNodes.Where(n => n.HasGeometry))
            {
                results.Add(new ObjectSegmentationResult
                {
                    RootIndex = node.Index,
                    GeometryNodeIndices = new List<int> { node.Index },
                    DoodadNodeIndices = new List<int>(), // Individual geometry nodes don't include doodads
                    SegmentationType = "individual_geometry"
                });
            }
            
            return results;
        }

        /// <summary>
        /// ✨ NEW: Segments logical sub-hierarchies as separate objects
        /// Creates one object per parent node that has children, including the parent + immediate children
        /// </summary>
        public List<ObjectSegmentationResult> SegmentBySubHierarchies(HierarchyAnalysisResult analysis)
        {
            var results = new List<ObjectSegmentationResult>();
            var processedNodes = new HashSet<int>();
            
            // Find nodes that have children and could represent logical sub-objects
            var parentNodes = analysis.AllNodes.Where(n => n.Children.Any()).ToList();
            
            foreach (var parentNode in parentNodes)
            {
                // Skip if this node was already included in another sub-hierarchy
                if (processedNodes.Contains(parentNode.Index))
                    continue;
                
                var geom = new List<int>();
                var dood = new List<int>();
                
                // Include the parent if it has geometry
                if (parentNode.HasGeometry)
                    geom.Add(parentNode.Index);
                else if (parentNode.IsDoodadNode)
                    dood.Add(parentNode.Index);
                
                // Include immediate children only (not deep traversal like root hierarchy)
                foreach (var child in parentNode.Children)
                {
                    if (child.HasGeometry)
                        geom.Add(child.Index);
                    else if (child.IsDoodadNode)
                        dood.Add(child.Index);
                    
                    processedNodes.Add(child.Index);
                }
                
                // Only create an object if it has meaningful content
                if (geom.Any() || dood.Any())
                {
                    results.Add(new ObjectSegmentationResult
                    {
                        RootIndex = parentNode.Index,
                        GeometryNodeIndices = geom,
                        DoodadNodeIndices = dood,
                        SegmentationType = "sub_hierarchy"
                    });
                    
                    processedNodes.Add(parentNode.Index);
                }
            }
            
            // Also add standalone leaf geometry nodes that weren't included in any sub-hierarchy
            foreach (var node in analysis.AllNodes.Where(n => n.HasGeometry && !processedNodes.Contains(n.Index)))
            {
                results.Add(new ObjectSegmentationResult
                {
                    RootIndex = node.Index,
                    GeometryNodeIndices = new List<int> { node.Index },
                    DoodadNodeIndices = new List<int>(),
                    SegmentationType = "leaf_geometry"
                });
            }
            
            return results;
        }

        /// <summary>
        /// ✨ NEW: Comprehensive segmentation that provides all export strategies
        /// </summary>
        public SegmentationStrategiesResult SegmentByAllStrategies(HierarchyAnalysisResult analysis)
        {
            return new SegmentationStrategiesResult
            {
                ByRootHierarchy = SegmentObjectsByHierarchy(analysis),
                ByIndividualGeometry = SegmentByIndividualGeometry(analysis),
                BySubHierarchies = SegmentBySubHierarchies(analysis)
            };
        }

        private void TraverseAndCollect(HierarchyNode node, List<int> geom, List<int> dood)
        {
            var visited = new HashSet<int>();
            TraverseAndCollectRecursive(node, geom, dood, visited);
        }

        private void TraverseAndCollectRecursive(HierarchyNode node, List<int> geom, List<int> dood, HashSet<int> visited)
        {
            // Prevent infinite loops with cycle detection
            if (visited.Contains(node.Index))
                return;
                
            visited.Add(node.Index);
            
            if (node.HasGeometry)
                geom.Add(node.Index);
            else if (node.IsDoodadNode)
                dood.Add(node.Index);
                
            foreach (var child in node.Children)
                TraverseAndCollectRecursive(child, geom, dood, visited);
        }

        public class ObjectSegmentationResult
        {
            public int RootIndex { get; set; }
            public List<int> GeometryNodeIndices { get; set; } = new();
            public List<int> DoodadNodeIndices { get; set; } = new();
            public string SegmentationType { get; set; } = "unknown"; // ✨ NEW: Track segmentation strategy
        }

        /// <summary>
        /// ✨ NEW: Container for all segmentation strategies
        /// </summary>
        public class SegmentationStrategiesResult
        {
            public List<ObjectSegmentationResult> ByRootHierarchy { get; set; } = new();
            public List<ObjectSegmentationResult> ByIndividualGeometry { get; set; } = new();
            public List<ObjectSegmentationResult> BySubHierarchies { get; set; } = new();
            
            public int TotalObjects => ByRootHierarchy.Count + ByIndividualGeometry.Count + BySubHierarchies.Count;
        }
    }
} 