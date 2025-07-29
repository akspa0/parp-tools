using System;
using System.Collections.Generic;
using System.Linq;
using parpToolbox.Models.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Analyzes MSLK chunk to identify proper object hierarchies using group leaders.
    /// Ported from working reference implementation in WoWToolbox.Core.
    /// </summary>
    public class MslkHierarchyAnalyzer
    {
        public class HierarchyNode
        {
            public int Index { get; set; }
            public uint ParentIndex { get; set; }
            public int MspiFirstIndex { get; set; }
            public byte MspiIndexCount { get; set; }
            public byte Flags { get; set; }
            public MslkEntry? OriginalEntry { get; set; }
            
            // Derived properties
            public bool HasGeometry => MspiFirstIndex >= 0;
            public bool IsGroupLeader => ParentIndex == Index;
            public bool IsDoodadNode => MspiFirstIndex == -1;
            
            // Hierarchy relationships
            public List<HierarchyNode> Children { get; set; } = new();
            public HierarchyNode? Parent { get; set; }
            public int Depth { get; set; } = 0;
        }

        public class ObjectSegment
        {
            public int RootIndex { get; set; }
            public List<int> GeometryNodeIndices { get; set; } = new();
            public List<int> DoodadNodeIndices { get; set; } = new();
            public string SegmentationType { get; set; } = "";
            
            public int TotalGeometryNodes => GeometryNodeIndices.Count;
            public int TotalDoodadNodes => DoodadNodeIndices.Count;
            public int TotalNodes => TotalGeometryNodes + TotalDoodadNodes;
        }

        public class HierarchyAnalysisResult
        {
            public List<HierarchyNode> AllNodes { get; set; } = new();
            public List<HierarchyNode> GroupLeaders { get; set; } = new();
            public Dictionary<uint, List<HierarchyNode>> NodesByParentIndex { get; set; } = new();
            
            public int SelfReferencingNodes { get; set; }
            public int GeometryNodeCount { get; set; }
            public int DoodadNodeCount { get; set; }
            public List<string> DiscoveredPatterns { get; set; } = new();
        }

        /// <summary>
        /// Analyzes MSLK entries to discover group leaders and hierarchical relationships
        /// </summary>
        public HierarchyAnalysisResult AnalyzeHierarchy(List<MslkEntry> mslkEntries)
        {
            var result = new HierarchyAnalysisResult();
            
            if (mslkEntries == null || !mslkEntries.Any())
            {
                result.DiscoveredPatterns.Add("No MSLK entries to analyze");
                return result;
            }

            // Convert entries to hierarchy nodes
            ConvertToHierarchyNodes(mslkEntries, result);
            
            // Identify group leaders (self-referencing nodes)
            IdentifyGroupLeaders(result);
            
            // Analyze parent-child relationships
            AnalyzeParentChildRelationships(result);
            
            // Calculate statistics
            CalculateStatistics(result);
            
            return result;
        }

        /// <summary>
        /// Segments objects based on group leaders (self-referencing nodes)
        /// </summary>
        public List<ObjectSegment> SegmentObjectsByHierarchy(HierarchyAnalysisResult analysis)
        {
            var results = new List<ObjectSegment>();
            
            // Use group leaders as primary objects
            foreach (var leader in analysis.GroupLeaders)
            {
                var geometryNodes = new List<int>();
                var doodadNodes = new List<int>();
                
                // Include the leader itself
                if (leader.HasGeometry)
                    geometryNodes.Add(leader.Index);
                else if (leader.IsDoodadNode)
                    doodadNodes.Add(leader.Index);
                
                // Collect all children through hierarchy traversal
                TraverseAndCollect(leader, geometryNodes, doodadNodes, analysis);
                
                results.Add(new ObjectSegment
                {
                    RootIndex = leader.Index,
                    GeometryNodeIndices = geometryNodes,
                    DoodadNodeIndices = doodadNodes,
                    SegmentationType = "group_leader"
                });
            }
            
            return results;
        }

        private void ConvertToHierarchyNodes(List<MslkEntry> mslkEntries, HierarchyAnalysisResult result)
        {
            for (int i = 0; i < mslkEntries.Count; i++)
            {
                var entry = mslkEntries[i];
                
                var node = new HierarchyNode
                {
                    Index = i,
                    ParentIndex = entry.ParentIndex,
                    MspiFirstIndex = entry.MspiFirstIndex,
                    MspiIndexCount = entry.MspiIndexCount,
                    Flags = entry.Flags_0x00,
                    OriginalEntry = entry
                };
                
                result.AllNodes.Add(node);
                
                // Group by parent index for relationship building
                if (!result.NodesByParentIndex.TryGetValue(entry.ParentIndex, out var parentGroup))
                {
                    parentGroup = new List<HierarchyNode>();
                    result.NodesByParentIndex[entry.ParentIndex] = parentGroup;
                }
                parentGroup.Add(node);
            }
        }

        private void IdentifyGroupLeaders(HierarchyAnalysisResult result)
        {
            // Find self-referencing nodes (group leaders/primary objects)
            result.GroupLeaders = result.AllNodes
                .Where(n => n.ParentIndex == n.Index)
                .ToList();
            
            result.SelfReferencingNodes = result.GroupLeaders.Count;
            
            result.DiscoveredPatterns.Add($"Found {result.GroupLeaders.Count} group leaders (self-referencing nodes)");
            
            foreach (var leader in result.GroupLeaders)
            {
                result.DiscoveredPatterns.Add($"Group Leader {leader.Index}: ParentIndex={leader.ParentIndex}, HasGeometry={leader.HasGeometry}");
            }
        }

        private void AnalyzeParentChildRelationships(HierarchyAnalysisResult result)
        {
            // Build parent-child relationships (but avoid cycles by not building traditional hierarchy)
            var validParentRefs = 0;
            var invalidRefs = 0;
            
            foreach (var node in result.AllNodes)
            {
                if (node.ParentIndex < result.AllNodes.Count && node.ParentIndex != node.Index)
                {
                    validParentRefs++;
                }
                else if (node.ParentIndex >= result.AllNodes.Count)
                {
                    invalidRefs++;
                }
            }
            
            result.DiscoveredPatterns.Add($"Parent Index Analysis:");
            result.DiscoveredPatterns.Add($"  Self-references: {result.SelfReferencingNodes}");
            result.DiscoveredPatterns.Add($"  Valid parent refs: {validParentRefs}");
            result.DiscoveredPatterns.Add($"  Invalid refs: {invalidRefs}");
        }

        private void CalculateStatistics(HierarchyAnalysisResult result)
        {
            result.GeometryNodeCount = result.AllNodes.Count(n => n.HasGeometry);
            result.DoodadNodeCount = result.AllNodes.Count(n => n.IsDoodadNode);
            
            result.DiscoveredPatterns.Add($"Statistics:");
            result.DiscoveredPatterns.Add($"  Total nodes: {result.AllNodes.Count}");
            result.DiscoveredPatterns.Add($"  Geometry nodes: {result.GeometryNodeCount}");
            result.DiscoveredPatterns.Add($"  Doodad nodes: {result.DoodadNodeCount}");
            result.DiscoveredPatterns.Add($"  Group leaders: {result.GroupLeaders.Count}");
        }

        private void TraverseAndCollect(HierarchyNode rootNode, List<int> geometryNodes, List<int> doodadNodes, HierarchyAnalysisResult analysis)
        {
            // Find all nodes that reference this root as their parent
            if (analysis.NodesByParentIndex.TryGetValue((uint)rootNode.Index, out var children))
            {
                foreach (var child in children)
                {
                    // Skip the node itself to avoid cycles
                    if (child.Index == rootNode.Index)
                        continue;
                    
                    // Collect geometry and doodad nodes
                    if (child.HasGeometry && !geometryNodes.Contains(child.Index))
                    {
                        geometryNodes.Add(child.Index);
                    }
                    else if (child.IsDoodadNode && !doodadNodes.Contains(child.Index))
                    {
                        doodadNodes.Add(child.Index);
                    }
                    
                    // Recursively traverse children (with depth limit to prevent infinite loops)
                    if (child.Depth < 10)
                    {
                        child.Depth = rootNode.Depth + 1;
                        TraverseAndCollect(child, geometryNodes, doodadNodes, analysis);
                    }
                }
            }
        }
    }
}
