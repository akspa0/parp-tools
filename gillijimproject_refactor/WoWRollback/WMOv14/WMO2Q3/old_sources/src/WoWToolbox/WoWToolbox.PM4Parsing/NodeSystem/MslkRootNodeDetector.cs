using WoWToolbox.Core;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.PM4Parsing.NodeSystem
{
    /// <summary>
    /// Detects MSLK root nodes which serve as building separators in PM4 navigation hierarchy.
    /// Root nodes are identified by self-referencing Unknown_0x04 field (entry.Unknown_0x04 == index).
    /// </summary>
    public class MslkRootNodeDetector
    {
        /// <summary>
        /// Represents a detected MSLK root node with its properties.
        /// </summary>
        public class RootNodeInfo
        {
            public int NodeIndex { get; set; }
            public MSLKEntry Entry { get; set; }
            public uint GroupKey { get; set; }
            public List<int> ChildNodes { get; set; } = new();
            public bool HasGeometry { get; set; }
        }

        /// <summary>
        /// Finds all root nodes in the MSLK hierarchy.
        /// Root nodes are identified by self-referencing Unknown_0x04 field.
        /// </summary>
        /// <param name="pm4File">The PM4 file to analyze</param>
        /// <returns>List of detected root node information</returns>
        public List<RootNodeInfo> DetectRootNodes(PM4File pm4File)
        {
            if (pm4File.MSLK?.Entries == null)
                return new List<RootNodeInfo>();

            var rootNodes = new List<RootNodeInfo>();

            // Find all self-referencing entries (root nodes)
            for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
            {
                var entry = pm4File.MSLK.Entries[i];
                if (entry.Unknown_0x04 == i) // Self-referencing = root node
                {
                    var rootInfo = new RootNodeInfo
                    {
                        NodeIndex = i,
                        Entry = entry,
                        GroupKey = entry.Unknown_0x04,
                        HasGeometry = entry.MspiFirstIndex >= 0 && entry.MspiIndexCount > 0
                    };

                    rootNodes.Add(rootInfo);
                }
            }

            // Find child nodes for each root
            foreach (var rootNode in rootNodes)
            {
                rootNode.ChildNodes = FindChildNodes(pm4File, rootNode.GroupKey);
            }

            return rootNodes;
        }

        /// <summary>
        /// Finds all child nodes that belong to a specific root node group.
        /// </summary>
        /// <param name="pm4File">The PM4 file</param>
        /// <param name="rootGroupKey">The group key of the root node</param>
        /// <returns>List of child node indices</returns>
        public List<int> FindChildNodes(PM4File pm4File, uint rootGroupKey)
        {
            var childNodes = new List<int>();

            if (pm4File.MSLK?.Entries == null)
                return childNodes;

            for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
            {
                var entry = pm4File.MSLK.Entries[i];
                
                // Child nodes point to the root group key but are not self-referencing
                if (entry.Unknown_0x04 == rootGroupKey && entry.Unknown_0x04 != i)
                {
                    childNodes.Add(i);
                }
            }

            return childNodes;
        }

        /// <summary>
        /// Traverses up the hierarchy to find the root node for a given node index.
        /// </summary>
        /// <param name="pm4File">The PM4 file</param>
        /// <param name="nodeIndex">Starting node index</param>
        /// <returns>Root node index, or -1 if no valid root found</returns>
        public int TraverseToRoot(PM4File pm4File, int nodeIndex)
        {
            if (pm4File.MSLK?.Entries == null)
                return -1;

            var visited = new HashSet<int>();
            int currentIndex = nodeIndex;
            
            // Follow the Unknown_0x04 parent chain until we reach a root (self-referencing) or hit a cycle
            while (currentIndex < pm4File.MSLK.Entries.Count && !visited.Contains(currentIndex))
            {
                visited.Add(currentIndex);
                var entry = pm4File.MSLK.Entries[currentIndex];
                
                // If this node is a root (self-referencing), return it
                if (entry.Unknown_0x04 == currentIndex)
                    return currentIndex;
                    
                // Move up the hierarchy
                currentIndex = (int)entry.Unknown_0x04;
                
                // Safety check: if parent is out of bounds, this node doesn't have a valid root
                if (currentIndex >= pm4File.MSLK.Entries.Count)
                    break;
            }
            
            // If we hit a cycle or invalid reference, return -1 (no valid root found)
            return -1;
        }

        /// <summary>
        /// Checks if a node belongs to a specific root object.
        /// </summary>
        /// <param name="pm4File">The PM4 file</param>
        /// <param name="nodeIndex">Node to check</param>
        /// <param name="rootIndex">Root node index</param>
        /// <returns>True if the node belongs to the root object</returns>
        public bool BelongsToRootObject(PM4File pm4File, int nodeIndex, int rootIndex)
        {
            if (pm4File.MSLK?.Entries == null || nodeIndex >= pm4File.MSLK.Entries.Count)
                return false;

            var entry = pm4File.MSLK.Entries[nodeIndex];
            
            // Direct parent relationship: node points directly to root
            if (entry.Unknown_0x04 == rootIndex)
                return true;
                
            // Hierarchical chain: traverse up the parent chain to see if it leads to root
            return TraverseToRoot(pm4File, nodeIndex) == rootIndex;
        }

        /// <summary>
        /// Gets statistics about the root node hierarchy structure.
        /// </summary>
        /// <param name="pm4File">The PM4 file to analyze</param>
        /// <returns>Hierarchy statistics</returns>
        public HierarchyStatistics GetHierarchyStatistics(PM4File pm4File)
        {
            var stats = new HierarchyStatistics();
            var rootNodes = DetectRootNodes(pm4File);

            stats.TotalNodes = pm4File.MSLK?.Entries?.Count ?? 0;
            stats.RootNodeCount = rootNodes.Count;
            stats.RootNodesWithGeometry = rootNodes.Count(r => r.HasGeometry);
            stats.TotalChildNodes = rootNodes.Sum(r => r.ChildNodes.Count);
            stats.OrphanedNodes = stats.TotalNodes - stats.RootNodeCount - stats.TotalChildNodes;

            // Calculate distribution of child nodes per root
            if (rootNodes.Count > 0)
            {
                stats.MinChildNodesPerRoot = rootNodes.Min(r => r.ChildNodes.Count);
                stats.MaxChildNodesPerRoot = rootNodes.Max(r => r.ChildNodes.Count);
                stats.AverageChildNodesPerRoot = (double)stats.TotalChildNodes / rootNodes.Count;
            }

            return stats;
        }

        /// <summary>
        /// Statistical information about MSLK hierarchy structure.
        /// </summary>
        public class HierarchyStatistics
        {
            public int TotalNodes { get; set; }
            public int RootNodeCount { get; set; }
            public int RootNodesWithGeometry { get; set; }
            public int TotalChildNodes { get; set; }
            public int OrphanedNodes { get; set; }
            public int MinChildNodesPerRoot { get; set; }
            public int MaxChildNodesPerRoot { get; set; }
            public double AverageChildNodesPerRoot { get; set; }
        }
    }
} 