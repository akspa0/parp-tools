using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using System.Text;

namespace PM4Rebuilder
{
    /// <summary>
    /// Diagnostic tool for visualizing MSLK hierarchy and object grouping structure.
    /// Analyzes and visualizes the parent-child relationships between MSLK entries
    /// and generates diagnostic reports to better understand the object structure.
    /// </summary>
    public static class Pm4MslkHierarchyVisualizer
    {
        /// <summary>
        /// A node in the MSLK hierarchy representing an entry and its descendants
        /// </summary>
        public class MslkNode
        {
            public uint NodeId { get; set; }
            public required MslkEntry Entry { get; set; }
            public List<MslkNode> Children { get; set; } = new();
            public bool IsContainer => Entry.MspiFirstIndex == -1;
            public int Depth { get; set; }
            public uint? ParentId { get; set; }
            public int SurfaceRefIndex => Entry.SurfaceRefIndex;
        }

        /// <summary>
        /// Hierarchical grouping result containing the full node tree
        /// </summary>
        public class HierarchyResult
        {
            public List<MslkNode> RootNodes { get; set; } = new();
            public Dictionary<uint, MslkNode> AllNodes { get; set; } = new();
            public int MaxDepth { get; set; }
            public int TotalNodes { get; set; }
            public int ContainerNodes { get; set; }
            public int GeometryNodes { get; set; }
            public HashSet<uint> UniqueNodeIds { get; } = new HashSet<uint>();
        }

        /// <summary>
        /// Segment of objects within the hierarchy representing a complete building/object
        /// </summary>
        public class ObjectSegment
        {
            public uint RootIndex { get; set; }
            public required MslkNode RootNode { get; set; }
            public List<uint> GeometryNodeIndices { get; set; } = new();
            public List<uint> ContainerNodeIndices { get; set; } = new();
            public int MaxDepth { get; set; }
            public int TotalNodes => GeometryNodeIndices.Count + ContainerNodeIndices.Count;
        }

        /// <summary>
        /// Generate a hierarchical visualization and analysis of the MSLK structure
        /// from the specified PM4 scene file.
        /// </summary>
        /// <param name="pm4Path">Path to PM4 file or directory containing PM4 files</param>
        /// <param name="outputDir">Output directory for visualization and reports</param>
        /// <param name="summaryOnly">If true, only generate summary reports without detailed object content (default: false)</param>
        /// <param name="maxDepth">Maximum depth to report in the hierarchy (default: 3)</param>
        /// <param name="maxNodes">Maximum number of nodes to include in reports (default: 100)</param>
        /// <returns>0 on success, 1 on error</returns>
        public static int VisualizeHierarchy(string pm4Path, string outputDir, bool summaryOnly = false, int maxDepth = 3, int maxNodes = 100)
        {
            try
            {
                Console.WriteLine($"[MSLK HIERARCHY] Starting MSLK hierarchy visualization from: {pm4Path}");
                Console.WriteLine($"[MSLK HIERARCHY] Output directory: {outputDir}");
                
                Directory.CreateDirectory(outputDir);
                
                // Load PM4 scene
                var scene = LoadPm4Scene(pm4Path);
                if (scene == null)
                {
                    Console.WriteLine("[MSLK HIERARCHY ERROR] Failed to load PM4 scene");
                    return 1;
                }
                
                // Analyze MSLK hierarchy
                var hierarchy = AnalyzeMslkHierarchy(scene);
                Console.WriteLine($"[MSLK HIERARCHY] Analysis complete: {hierarchy.RootNodes.Count} root nodes, {hierarchy.TotalNodes} total nodes, max depth {hierarchy.MaxDepth}");
                
                // Segment objects
                var objectSegments = SegmentObjectsByHierarchy(hierarchy);
                Console.WriteLine($"[MSLK HIERARCHY] Found {objectSegments.Count} distinct objects/buildings");
                
                // Generate reports
                GenerateHierarchyReports(hierarchy, objectSegments, scene, outputDir, maxDepth, maxNodes, summaryOnly);
                
                Console.WriteLine($"[MSLK HIERARCHY] Visualization complete. Check output directory: {outputDir}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MSLK HIERARCHY ERROR] {ex}");
                return 1;
            }
        }
        
        /// <summary>
        /// Load the PM4 scene from the specified path
        /// </summary>
        private static Pm4Scene? LoadPm4Scene(string pm4Path)
        {
            try
            {
                var task = SceneLoaderHelper.LoadSceneAsync(
                    pm4Path, 
                    includeAdjacent: false, 
                    applyTransform: false, 
                    altTransform: false
                );
                
                return task.GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MSLK HIERARCHY ERROR] Failed to load PM4 scene: {ex.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Analyze the MSLK entries in the scene to build a hierarchical structure
        /// </summary>
        private static HierarchyResult AnalyzeMslkHierarchy(Pm4Scene scene)
        {
            Console.WriteLine("[MSLK HIERARCHY] Analyzing MSLK hierarchy...");
            
            var result = new HierarchyResult();
            var nodeMap = new Dictionary<uint, MslkNode>();
            var parentChildMap = new Dictionary<uint, List<uint>>();
            
            // Step 1: Create nodes for all MSLK entries
            uint nodeId = 0;
            foreach (var link in scene.Links)
            {
                var node = new MslkNode
                {
                    NodeId = nodeId,
                    Entry = link,
                    ParentId = link.ParentId
                };
                
                nodeMap[nodeId] = node;
                
                // Track parent-child relationships
                if (link.ParentId != 0 && link.ParentId != uint.MaxValue)
                {
                    if (!parentChildMap.ContainsKey(link.ParentId))
                    {
                        parentChildMap[link.ParentId] = new List<uint>();
                    }
                    parentChildMap[link.ParentId].Add(nodeId);
                }
                
                nodeId++;
            }
            
            // Step 2: Build the hierarchy
            // Find root nodes (no parents or parents outside the scene)
            var parentIds = new HashSet<uint>(nodeMap.Values.Select(n => n.ParentId ?? 0).Where(id => id != 0 && id != uint.MaxValue));
            var rootNodeIds = new HashSet<uint>();
            
            // Consider MPRL entries as potential root identifiers
            if (scene.Placements != null && scene.Placements.Count > 0)
            {
                Console.WriteLine("[MSLK HIERARCHY] Using MPRL.Unknown4 values as root identifiers");
                
                // Collect all Unknown4 values as potential root identifiers
                // Convert from ushort to uint during collection
                var mprlUnknown4Values = new HashSet<uint>(scene.Placements.Select(p => (uint)p.Unknown4).Where(id => id != 0));
                Console.WriteLine($"[MSLK HIERARCHY] Found {mprlUnknown4Values.Count} unique MPRL.Unknown4 values");
                
                // Find all nodes that match these Unknown4 values in their ParentId
                foreach (var node in nodeMap.Values)
                {
                    if (node.ParentId.HasValue)
                    {
                        // Convert node.ParentId.Value to uint for comparison
                        uint parentIdValue = node.ParentId.Value;
                        if (mprlUnknown4Values.Contains(parentIdValue))
                        {
                            // This node has a parent that matches an MPRL.Unknown4 value
                            // Add the parent as a root identifier
                            rootNodeIds.Add(parentIdValue);
                        }
                    }
                }
                
                Console.WriteLine($"[MSLK HIERARCHY] Found {rootNodeIds.Count} root node identifiers from MPRL references");
            }
            
            // If no root nodes identified from MPRL, fall back to nodes without parents
            if (rootNodeIds.Count == 0)
            {
                Console.WriteLine("[MSLK HIERARCHY] No MPRL-linked roots found, using orphaned nodes as roots");
                foreach (var node in nodeMap.Values)
                {
                    if (!node.ParentId.HasValue || node.ParentId == 0 || node.ParentId == uint.MaxValue || !parentIds.Contains(node.ParentId.Value))
                    {
                        rootNodeIds.Add(node.NodeId);
                    }
                }
            }
            
            // Step 3: Build the tree structure
            foreach (var rootId in rootNodeIds)
            {
                // Check if this is a direct root node or a referenced parent ID
                if (nodeMap.ContainsKey(rootId))
                {
                    // This is a direct root node (node exists in the scene)
                    var rootNode = nodeMap[rootId];
                    result.RootNodes.Add(rootNode);
                    BuildNodeTree(rootNode, nodeMap, parentChildMap, 0, new HashSet<uint>());
                }
                else
                {
                    // This is a referenced parent ID that doesn't exist as a node
                    // Find all nodes that reference this parent and make them roots
                    foreach (var node in nodeMap.Values.Where(n => n.ParentId == rootId))
                    {
                        result.RootNodes.Add(node);
                        BuildNodeTree(node, nodeMap, parentChildMap, 0, new HashSet<uint>());
                    }
                }
            }
            
            // Step 4: Calculate statistics
            result.AllNodes = nodeMap;
            result.MaxDepth = nodeMap.Values.Max(n => n.Depth);
            result.TotalNodes = nodeMap.Count;
            result.ContainerNodes = nodeMap.Values.Count(n => n.IsContainer);
            result.GeometryNodes = result.TotalNodes - result.ContainerNodes;
            
            // Add all nodes to the unique node tracking set
            foreach (var nodeKey in nodeMap.Keys)
            {
                result.UniqueNodeIds.Add(nodeKey);
            }
            
            return result;
        }
        
        /// <summary>
        /// Recursively build the node tree structure
        /// </summary>
        private static void BuildNodeTree(
            MslkNode node, 
            Dictionary<uint, MslkNode> nodeMap, 
            Dictionary<uint, List<uint>> parentChildMap, 
            int depth,
            HashSet<uint>? visitedNodes = null)
        {
            // Initialize visited nodes set if this is the first call
            visitedNodes ??= new HashSet<uint>();
            
            // Prevent cycles in the graph (crucial to avoid stack overflow)
            if (visitedNodes.Contains(node.NodeId))
            {
                Console.WriteLine($"[MSLK HIERARCHY WARNING] Cycle detected! Node {node.NodeId} has already been visited.");
                return;
            }
            
            // Mark this node as visited
            visitedNodes.Add(node.NodeId);
            
            // Set node depth
            node.Depth = depth;
            
            // Find child nodes
            if (parentChildMap.TryGetValue(node.NodeId, out var childIds))
            {
                foreach (var childId in childIds)
                {
                    // Skip self-reference (another source of cycles)
                    if (childId == node.NodeId)
                    {
                        Console.WriteLine($"[MSLK HIERARCHY WARNING] Self-reference detected! Node {node.NodeId} references itself as a child.");
                        continue;
                    }
                    
                    if (nodeMap.TryGetValue(childId, out var childNode) && childNode != null)
                    {
                        node.Children.Add(childNode);
                        BuildNodeTree(childNode, nodeMap, parentChildMap, depth + 1, visitedNodes);
                    }
                }
            }
        }
        
        /// <summary>
        /// Segment objects within the hierarchy based on root nodes and their descendants
        /// </summary>
        private static List<ObjectSegment> SegmentObjectsByHierarchy(HierarchyResult hierarchy)
        {
            var objects = new List<ObjectSegment>();
            
            Console.WriteLine("[MSLK HIERARCHY] Segmenting objects from hierarchy...");
            
            // Each root node and its descendants form a separate object
            foreach (var rootNode in hierarchy.RootNodes)
            {
                var segment = new ObjectSegment
                {
                    RootIndex = rootNode.NodeId,
                    RootNode = rootNode,
                    MaxDepth = 0
                };
                
                // Collect all nodes in this segment
                CollectNodesInSegment(rootNode, segment);
                
                // Only add segments that have at least one geometry node
                if (segment.GeometryNodeIndices.Count > 0)
                {
                    objects.Add(segment);
                    // Count unique nodes for accurate reporting
                    var uniqueGeometryNodes = new HashSet<uint>(segment.GeometryNodeIndices);
                    var uniqueContainerNodes = new HashSet<uint>(segment.ContainerNodeIndices);
                    Console.WriteLine($"[MSLK HIERARCHY] Object {segment.RootIndex}: {uniqueGeometryNodes.Count} unique geometry nodes, {uniqueContainerNodes.Count} unique container nodes, depth {segment.MaxDepth}");
                }
            }
            
            return objects;
        }
        
        /// <summary>
        /// Recursively collect all nodes in a segment
        /// </summary>
        private static void CollectNodesInSegment(MslkNode node, ObjectSegment segment)
        {
            // Use a stack-based approach to avoid recursion and handle cycles
            var nodesToProcess = new Stack<MslkNode>();
            var processedNodes = new HashSet<uint>();
            
            nodesToProcess.Push(node);
            
            while (nodesToProcess.Count > 0)
            {
                var currentNode = nodesToProcess.Pop();
                
                // Skip already processed nodes (prevents cycles)
                if (!processedNodes.Add(currentNode.NodeId))
                {
                    continue;
                }
                
                // Update max depth if needed
                segment.MaxDepth = Math.Max(segment.MaxDepth, currentNode.Depth);
                
                // Add node to appropriate list
                if (currentNode.IsContainer)
                {
                    segment.ContainerNodeIndices.Add(currentNode.NodeId);
                }
                else
                {
                    segment.GeometryNodeIndices.Add(currentNode.NodeId);
                }
                
                // Add children to processing stack
                foreach (var child in currentNode.Children)
                {
                    nodesToProcess.Push(child);
                }
            }
        }
        
        /// <summary>
        /// Generate hierarchy visualization reports
        /// </summary>
        private static void GenerateHierarchyReports(HierarchyResult hierarchy, List<ObjectSegment> objects, Pm4Scene scene, string outputDir, int maxDepth, int maxNodes, bool summaryOnly)
        {
            Console.WriteLine("[MSLK HIERARCHY] Generating hierarchy reports...");
            
            // Always generate the summary report
            GenerateSummaryReport(hierarchy, objects, outputDir);
            
            // Only generate detailed reports if not in summary-only mode
            if (!summaryOnly)
            {
                // Generate a simplified hierarchy diagram with limited nodes
                GenerateHierarchyDiagram(hierarchy, outputDir, maxNodes);
                
                // Generate object reports with depth limits
                GenerateObjectReports(hierarchy, objects, outputDir, maxDepth, maxNodes);
                
                // Generate a simplified relationship diagram
                GenerateRelationshipDiagram(hierarchy, scene, outputDir, maxNodes);
            }
            
            // Report on output limitations
            Console.WriteLine($"[MSLK HIERARCHY] Applied output limits: maxDepth={maxDepth}, maxNodes={maxNodes}, summaryOnly={summaryOnly}");
        }
        
        /// <summary>
        /// Generate a summary report of the MSLK hierarchy
        /// </summary>
        private static void GenerateSummaryReport(HierarchyResult hierarchy, List<ObjectSegment> objects, string outputDir)
        {
            var reportPath = Path.Combine(outputDir, "mslk_hierarchy_summary.md");
            
            using var writer = new StreamWriter(reportPath);
            
            writer.WriteLine("# MSLK Hierarchy Analysis Summary");
            writer.WriteLine();
            writer.WriteLine("## Hierarchy Statistics");
            writer.WriteLine($"- **Root Nodes**: {hierarchy.RootNodes.Count}");
            writer.WriteLine($"- **Total Nodes**: {hierarchy.TotalNodes}");
            writer.WriteLine($"- **Container Nodes**: {hierarchy.ContainerNodes}");
            writer.WriteLine($"- **Geometry Nodes**: {hierarchy.GeometryNodes}");
            writer.WriteLine($"- **Maximum Depth**: {hierarchy.MaxDepth}");
            writer.WriteLine();
            
            writer.WriteLine("## Object Segmentation");
            writer.WriteLine($"- **Total Objects**: {objects.Count}");
            writer.WriteLine();
            
            writer.WriteLine("### Top Objects by Complexity");
            // Top objects by complexity
            writer.WriteLine("### Top Objects by Complexity");
            writer.WriteLine("| Object ID | Unique Nodes | Geometry Nodes | Container Nodes | Max Depth |");
            writer.WriteLine("|-----------|-------------|----------------|-----------------|-----------|" );
            
            // Count unique nodes for each object to handle cycles properly
            var objectsWithCounts = objects.Select(o => {
                // Convert lists to HashSets to get unique counts
                var uniqueGeometryNodes = new HashSet<uint>(o.GeometryNodeIndices);
                var uniqueContainerNodes = new HashSet<uint>(o.ContainerNodeIndices);
                var uniqueTotal = new HashSet<uint>(uniqueGeometryNodes.Concat(uniqueContainerNodes));
                
                return new {
                    Object = o,
                    UniqueTotal = uniqueTotal.Count,
                    UniqueGeometry = uniqueGeometryNodes.Count,
                    UniqueContainer = uniqueContainerNodes.Count
                };
            }).OrderByDescending(x => x.UniqueTotal).Take(10);
            
            foreach (var obj in objectsWithCounts)
            {
                writer.WriteLine($"| {obj.Object.RootIndex} | {obj.UniqueTotal} | {obj.UniqueGeometry} | {obj.UniqueContainer} | {obj.Object.MaxDepth} |");
            }
            
            Console.WriteLine($"[MSLK HIERARCHY] Generated summary report: {reportPath}");
        }
        
        /// <summary>
        /// Generate a visual hierarchy diagram in DOT format (for GraphViz)
        /// </summary>
        private static void GenerateHierarchyDiagram(HierarchyResult hierarchy, string outputDir, int maxNodes)
        {
            var diagramPath = Path.Combine(outputDir, "mslk_hierarchy.dot");
            
            using var writer = new StreamWriter(diagramPath);
            
            writer.WriteLine("digraph MslkHierarchy {");
            writer.WriteLine("  rankdir=TB;");
            writer.WriteLine("  node [shape=box, style=\"filled,rounded\", fontname=Arial];");
            writer.WriteLine("  edge [arrowhead=normal, arrowtail=none, fontname=Arial];");
            
            // Only include a limited number of nodes in the diagram
            var nodesToInclude = hierarchy.AllNodes.Values
                .Where(n => n.IsContainer || n.SurfaceRefIndex >= 0)
                .OrderBy(n => n.Depth) // Prioritize nodes at lower depths
                .ThenBy(n => n.IsContainer ? 0 : 1) // Prioritize container nodes
                .Take(Math.Min(maxNodes / 2, 20)) // Limit to 20 or fewer nodes
                .ToList();
                
            Console.WriteLine($"[MSLK HIERARCHY] Including {nodesToInclude.Count} of {hierarchy.AllNodes.Count} nodes in hierarchy diagram");
            
            // Write node definitions
            foreach (var node in nodesToInclude)
            {
                var label = node.IsContainer
                    ? $"Container {node.NodeId}\\nParentId: {node.ParentId}\\nSurfaceRef: {node.SurfaceRefIndex}"
                    : $"Geometry {node.NodeId}\\nParentId: {node.ParentId}\\nSurfaceRef: {node.SurfaceRefIndex}";
                
                var color = node.IsContainer ? "lightskyblue" : "lightgreen";
                
                writer.WriteLine($"  node{node.NodeId} [label=\"{label}\", fillcolor=\"{color}\"];");
            }
            
            // Create a HashSet of included node IDs for faster lookups
            var includedNodeIds = new HashSet<uint>(nodesToInclude.Select(n => n.NodeId));
            
            // Write edge definitions - only for nodes we've included
            foreach (var node in nodesToInclude)
            {
                // Only include edges to nodes that are also in our limited set
                foreach (var child in node.Children.Where(c => includedNodeIds.Contains(c.NodeId)))
                {
                    writer.WriteLine($"  node{node.NodeId} -> node{child.NodeId};");
                }
            }
            
            writer.WriteLine("}");
            
            Console.WriteLine($"[MSLK HIERARCHY] Generated hierarchy diagram: {diagramPath}");
            Console.WriteLine($"[MSLK HIERARCHY] To visualize: Use GraphViz or an online DOT viewer");
        }
        
        /// <summary>
        /// Generate detailed reports for each object
        /// </summary>
        private static void GenerateObjectReports(HierarchyResult hierarchy, List<ObjectSegment> objects, string outputDir, int maxDepth, int maxNodes)
        {
            var objectsDir = Path.Combine(outputDir, "objects");
            Directory.CreateDirectory(objectsDir);
            
            // Limit the number of objects to report to avoid overwhelming output
            var objectsToReport = objects.Take(Math.Min(maxNodes / 5, 10)).ToList();
            
            foreach (var obj in objectsToReport)
            {
                var reportPath = Path.Combine(objectsDir, $"object_{obj.RootIndex}.md");
                
                using var writer = new StreamWriter(reportPath);
                
                writer.WriteLine($"# Object {obj.RootIndex} Analysis");
                writer.WriteLine();
                writer.WriteLine("## Object Overview");
                writer.WriteLine($"- **Total Nodes**: {obj.TotalNodes}");
                writer.WriteLine($"- **Geometry Nodes**: {obj.GeometryNodeIndices.Count}");
                writer.WriteLine($"- **Container Nodes**: {obj.ContainerNodeIndices.Count}");
                writer.WriteLine($"- **Maximum Depth**: {obj.MaxDepth}");
                writer.WriteLine();
                
                // Write the node hierarchy with depth limit
                WriteNodeHierarchy(writer, obj.RootNode, 0, maxDepth);
                
                writer.WriteLine();
                writer.WriteLine("## Surface References");
                writer.WriteLine("| Node ID | Type | Surface Index | Surface Name | Triangle Count |");
                writer.WriteLine("|---------|------|---------------|--------------|----------------|");
                
                // List all surface references
                var allNodeIds = new List<uint>();
                allNodeIds.AddRange(obj.GeometryNodeIndices);
                allNodeIds.AddRange(obj.ContainerNodeIndices);
                
                foreach (var nodeId in allNodeIds)
                {
                    if (hierarchy.AllNodes.TryGetValue(nodeId, out var node))
                    {
                        var surfaceRefIndex = node.SurfaceRefIndex;
                        string surfaceName = "N/A";
                        int triangleCount = 0;
                        
                        if (surfaceRefIndex >= 0 && surfaceRefIndex < hierarchy.AllNodes.Count)
                        {
                            var surfaceGroup = hierarchy.AllNodes[(uint)surfaceRefIndex];
                            surfaceName = $"MslkEntry-{surfaceGroup.NodeId}"; // No Name property in MslkEntry, using NodeId instead
                            triangleCount = surfaceGroup.Entry.MspiIndexCount; // Using MspiIndexCount as an approximation for triangle count
                        }
                        
                        var nodeType = node.IsContainer ? "Container" : "Geometry";
                        writer.WriteLine($"| {nodeId} | {nodeType} | {surfaceRefIndex} | {surfaceName} | {triangleCount} |");
                    }
                }
            }
            
            Console.WriteLine($"[MSLK HIERARCHY] Generated {objectsToReport.Count} object reports in: {objectsDir}");
        }
        
        /// <summary>
        /// Write a hierarchical representation of a node and its children
        /// </summary>
        private static void WriteNodeHierarchy(TextWriter writer, MslkNode node, int indent, int maxDepth)
        {
            var indentStr = new string(' ', indent * 2);
            var nodeType = node.IsContainer ? "Container" : "Geometry";
            
            writer.WriteLine($"{indentStr}- **{nodeType} Node {node.NodeId}** (SurfaceRef: {node.SurfaceRefIndex})");
            
            // Check if we've reached the maximum depth
            if (indent >= maxDepth)
            {
                // If we have children but reached max depth, just note how many there are
                if (node.Children.Count > 0)
                {
                    writer.WriteLine($"{new string(' ', (indent + 1) * 4)}... {node.Children.Count} more child nodes (depth limit reached)");
                }
                return;
            }
            
            // Recursively write child nodes
            foreach (var child in node.Children.Take(10)) // Limit children per node to 10
            {
                WriteNodeHierarchy(writer, child, indent + 1, maxDepth);
            }
            
            // If there are more children than we showed, indicate that
            if (node.Children.Count > 10)
            {
                writer.WriteLine($"{new string(' ', (indent + 1) * 4)}... {node.Children.Count - 10} more children not shown");
            }
        }
        
        /// <summary>
        /// Generate a relationship diagram showing connections between MSLK, MPRL, and MSUR
        /// </summary>
        private static void GenerateRelationshipDiagram(HierarchyResult hierarchy, Pm4Scene scene, string outputDir, int maxNodes)
        {
            var diagramPath = Path.Combine(outputDir, "relationship_diagram.md");
            
            using var writer = new StreamWriter(diagramPath);
            
            writer.WriteLine("# PM4 Chunk Relationship Diagram");
            writer.WriteLine();
            writer.WriteLine("This diagram shows the relationships between MPRL, MSLK, and MSUR chunks based on the analyzed PM4 scene.");
            writer.WriteLine();
            
            writer.WriteLine("```mermaid");
            writer.WriteLine("graph TD");
            writer.WriteLine("    classDef mprl fill:#f9d5e5,stroke:#333,stroke-width:1px");
            writer.WriteLine("    classDef mslk fill:#eeeeee,stroke:#333,stroke-width:1px");
            writer.WriteLine("    classDef msurContainer fill:#b5ead7,stroke:#333,stroke-width:1px");
            writer.WriteLine("    classDef msurGeometry fill:#c7ceea,stroke:#333,stroke-width:1px");
            
            // Add MPRL nodes
            if (scene.Placements != null && scene.Placements.Count > 0)
            {
                // Only include a sample of placements
                var samplePlacements = scene.Placements.Take(Math.Min(10, scene.Placements.Count)).ToList();
                
                foreach (var placement in samplePlacements)
                {
                    writer.WriteLine($"    MPRL{placement.Unknown4}[\"MPRL {placement.Unknown4}<br/>Unknown4: {placement.Unknown4}\"]");
                    writer.WriteLine($"    MPRL{placement.Unknown4}:::mprl");
                }
                
                // Add connections from MPRL to MSLK
                foreach (var node in hierarchy.AllNodes.Values.Where(n => n.ParentId.HasValue))
                {
                    var parentId = node.ParentId.Value;
                    
                    // Only include connections for the sample placements
                    if (parentId != 0 && samplePlacements.Any(p => p.Unknown4 == parentId))
                    {
                        writer.WriteLine($"    MPRL{parentId} -->|\"Unknown4 = ParentId\"| MSLK{node.NodeId}");
                    }
                }
            }
            
            // Add MSLK nodes (limit to a reasonable number)
            var sampleNodes = hierarchy.AllNodes.Values
                .OrderBy(n => n.IsContainer ? 0 : 1) // Show containers first
                .ThenBy(n => n.NodeId)
                .Take(Math.Min(20, hierarchy.AllNodes.Count))
                .ToList();
            
            foreach (var node in sampleNodes)
            {
                var nodeType = node.IsContainer ? "Container" : "Geometry";
                var surfaceRefStr = node.SurfaceRefIndex >= 0 ? node.SurfaceRefIndex.ToString() : "Invalid";
                
                writer.WriteLine($"    MSLK{node.NodeId}[\"MSLK {node.NodeId}<br/>Type: {nodeType}<br/>SurfaceRef: {surfaceRefStr}\"]");
                writer.WriteLine($"    MSLK{node.NodeId}:::mslk");
            }
            
            // Add MSLK parent-child connections
            foreach (var node in sampleNodes)
            {
                foreach (var child in node.Children)
                {
                    // Only show connections between sample nodes
                    if (sampleNodes.Contains(child))
                    {
                        writer.WriteLine($"    MSLK{node.NodeId} -->|\"parent-child\"| MSLK{child.NodeId}");
                    }
                }
            }
            
            // Add MSUR nodes and connections
            var surfaceIndices = new HashSet<int>();
            foreach (var node in sampleNodes)
            {
                if (node.SurfaceRefIndex >= 0 && node.SurfaceRefIndex < scene.Groups.Count)
                {
                    surfaceIndices.Add(node.SurfaceRefIndex);
                }
            }
            
            foreach (var surfaceIndex in surfaceIndices)
            {
                var surfaceGroup = scene.Groups[surfaceIndex];
                var triangleCount = surfaceGroup.Faces?.Count ?? 0;
                var surfaceName = surfaceGroup.Name ?? "Unnamed";
                
                writer.WriteLine($"    MSUR{surfaceIndex}[\"MSUR {surfaceIndex}<br/>Name: {surfaceName}<br/>Triangles: {triangleCount}\"]");
                writer.WriteLine($"    MSUR{surfaceIndex}:::msurGeometry");
                
                // Connect MSLK nodes to this surface
                foreach (var node in sampleNodes.Where(n => n.SurfaceRefIndex == surfaceIndex))
                {
                    writer.WriteLine($"    MSLK{node.NodeId} -->|\"SurfaceRefIndex\"| MSUR{surfaceIndex}");
                }
            }
            
            writer.WriteLine("```");
            
            Console.WriteLine($"[MSLK HIERARCHY] Generated relationship diagram: {diagramPath}");
        }
    }
}
