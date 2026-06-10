using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using System.Text.Json;

namespace PM4Rebuilder
{
    /// <summary>
    /// Diagnostic tool for analyzing PM4 chunk relationships and hierarchical structure.
    /// Helps visualize how MPRL, MSLK, and other chunks link together to form building objects.
    /// </summary>
    public static class Pm4DiagnosticTool
    {
        /// <summary>
        /// Run comprehensive diagnostics on a PM4 file and output visualizations of object hierarchies.
        /// </summary>
        /// <param name="pm4Path">Path to PM4 file</param>
        /// <param name="outputDir">Output directory for diagnostic files</param>
        /// <returns>0 on success, 1 on error</returns>
        public static int RunDiagnostics(string pm4Path, string outputDir)
        {
            try
            {
                Console.WriteLine($"[PM4 DIAGNOSTICS] Starting comprehensive diagnostics on: {pm4Path}");
                Directory.CreateDirectory(outputDir);
                
                // Load PM4 scene
                var scene = LoadPm4Scene(pm4Path);
                if (scene == null)
                {
                    Console.WriteLine("[PM4 DIAGNOSTICS ERROR] Failed to load PM4 scene");
                    return 1;
                }
                
                // Run comprehensive diagnostics
                AnalyzeChunkCounts(scene, outputDir);
                AnalyzeMprlUnknownFields(scene, outputDir);
                AnalyzeMslkStructure(scene, outputDir);
                AnalyzeBuildingHierarchy(scene, outputDir);
                AnalyzeVertexPoolUsage(scene, outputDir);
                
                // Generate relationship graph
                GenerateRelationshipDiagram(scene, outputDir);
                
                Console.WriteLine($"[PM4 DIAGNOSTICS] Diagnostics complete. Results written to {outputDir}");
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PM4 DIAGNOSTICS ERROR] {ex.Message}");
                return 1;
            }
        }
        
        private static Pm4Scene? LoadPm4Scene(string pm4Path)
        {
            try
            {
                // Use LoadSceneAsync from SceneLoaderHelper class
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
                Console.WriteLine($"[PM4 DIAGNOSTICS ERROR] Failed to load PM4 scene: {ex.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Analyze and report on basic chunk counts and statistics
        /// </summary>
        private static void AnalyzeChunkCounts(Pm4Scene scene, string outputDir)
        {
            Console.WriteLine("[PM4 DIAGNOSTICS] Analyzing chunk counts and statistics...");
            
            var stats = new Dictionary<string, int>
            {
                { "MPRL_Count", scene.Placements?.Count ?? 0 },
                { "MSLK_Count", scene.Links?.Count ?? 0 },
                { "MSUR_Count", scene.Groups?.Count ?? 0 },
                { "MSVT_Count", scene.Vertices?.Count ?? 0 },
                { "MSCN_Count", scene.MscnVertices?.Count ?? 0 },
                { "Triangle_Count", scene.Triangles?.Count ?? 0 },
            };
            
            // Count container nodes (MspiFirstIndex = -1)
            int containerCount = scene.Links?.Count(link => link.MspiFirstIndex == -1) ?? 0;
            stats["MSLK_Container_Nodes"] = containerCount;
            stats["MSLK_Geometry_Nodes"] = stats["MSLK_Count"] - containerCount;
            
            // Count unique parent IDs in MSLK
            var uniqueParentIds = scene.Links?
                .Select(link => link.ParentId)
                .Distinct()
                .Count() ?? 0;
            stats["Unique_MSLK_ParentIds"] = uniqueParentIds;
            
            // Count unique Unknown4 values in MPRL
            var uniqueMprlIds = scene.Placements?
                .Select(p => p.Unknown4)
                .Distinct()
                .Count() ?? 0;
            stats["Unique_MPRL_Unknown4"] = uniqueMprlIds;
            
            // Find intersection between MPRL.Unknown4 and MSLK.ParentId
            var mprlIds = new HashSet<uint>();
            if (scene.Placements != null)
            {
                foreach (var p in scene.Placements)
                {
                    mprlIds.Add(p.Unknown4);
                }
            }
            
            var mslkParentIds = new HashSet<uint>();
            if (scene.Links != null)
            {
                foreach (var link in scene.Links)
                {
                    mslkParentIds.Add(link.ParentId);
                }
            }
            
            int matchingIds = mprlIds.Intersect(mslkParentIds).Count();
            stats["Matching_MPRL_MSLK_Ids"] = matchingIds;
            
            // Output statistics to JSON
            string statsPath = Path.Combine(outputDir, "chunk_statistics.json");
            File.WriteAllText(statsPath, JsonSerializer.Serialize(stats, new JsonSerializerOptions { WriteIndented = true }));
            
            // Output basic statistics to console
            Console.WriteLine($"[PM4 DIAGNOSTICS] Scene Statistics:");
            Console.WriteLine($"  - MPRL Placements: {stats["MPRL_Count"]} (Unique Unknown4: {stats["Unique_MPRL_Unknown4"]})");
            Console.WriteLine($"  - MSLK Links: {stats["MSLK_Count"]} (Unique ParentIds: {stats["Unique_MSLK_ParentIds"]})");
            Console.WriteLine($"  - Container Nodes: {stats["MSLK_Container_Nodes"]}, Geometry Nodes: {stats["MSLK_Geometry_Nodes"]}");
            Console.WriteLine($"  - MPRL.Unknown4 to MSLK.ParentId matches: {stats["Matching_MPRL_MSLK_Ids"]} / {stats["Unique_MPRL_Unknown4"]}");
        }
        
        /// <summary>
        /// Analyze MPRL Unknown fields to identify patterns and possible usage
        /// </summary>
        private static void AnalyzeMprlUnknownFields(Pm4Scene scene, string outputDir)
        {
            if (scene.Placements == null || scene.Placements.Count == 0)
            {
                Console.WriteLine("[PM4 DIAGNOSTICS] No MPRL placements found for analysis");
                return;
            }
            
            Console.WriteLine("[PM4 DIAGNOSTICS] Analyzing MPRL Unknown fields for patterns...");
            
            var unknown4Values = new Dictionary<uint, int>();
            var unknown6Values = new Dictionary<uint, int>();
            
            foreach (var placement in scene.Placements)
            {
                // Track Unknown4 values (building IDs)
                if (!unknown4Values.ContainsKey(placement.Unknown4))
                {
                    unknown4Values[placement.Unknown4] = 0;
                }
                unknown4Values[placement.Unknown4]++;
                
                // Track Unknown6 values (likely type flags)
                if (!unknown6Values.ContainsKey(placement.Unknown6))
                {
                    unknown6Values[placement.Unknown6] = 0;
                }
                unknown6Values[placement.Unknown6]++;
            }
            
            // Output detailed MPRL data for inspection
            using (var writer = new StreamWriter(Path.Combine(outputDir, "mprl_fields.csv")))
            {
                writer.WriteLine("Index,Unknown4,Unknown6,ChildCount");
                
                for (int i = 0; i < scene.Placements.Count; i++)
                {
                    var placement = scene.Placements[i];
                    int childCount = scene.Links?.Count(link => link.ParentId == placement.Unknown4) ?? 0;
                    
                    writer.WriteLine($"{i},{placement.Unknown4},{placement.Unknown6},{childCount}");
                }
            }
            
            Console.WriteLine($"[PM4 DIAGNOSTICS] MPRL Analysis:");
            Console.WriteLine($"  - Found {unknown4Values.Count} unique Unknown4 values (building IDs)");
            Console.WriteLine($"  - Found {unknown6Values.Count} unique Unknown6 values");
            
            // Report top Unknown6 values (likely type indicators)
            var topUnknown6 = unknown6Values.OrderByDescending(kv => kv.Value).Take(5);
            Console.WriteLine($"  - Top Unknown6 values: {string.Join(", ", topUnknown6.Select(kv => $"{kv.Key}={kv.Value}"))}");
        }
        
        /// <summary>
        /// Analyze MSLK structure to identify hierarchy patterns
        /// </summary>
        private static void AnalyzeMslkStructure(Pm4Scene scene, string outputDir)
        {
            if (scene.Links == null || scene.Links.Count == 0)
            {
                Console.WriteLine("[PM4 DIAGNOSTICS] No MSLK links found for analysis");
                return;
            }
            
            Console.WriteLine("[PM4 DIAGNOSTICS] Analyzing MSLK structure and hierarchy...");
            
            // Group MSLK entries by ParentId
            var linksByParent = scene.Links
                .GroupBy(link => link.ParentId)
                .ToDictionary(g => g.Key, g => g.ToList());
            
            // Output hierarchy statistics
            int containerGroups = 0;
            int mixedGroups = 0;
            int geometryOnlyGroups = 0;
            
            using (var writer = new StreamWriter(Path.Combine(outputDir, "mslk_structure.csv")))
            {
                writer.WriteLine("ParentId,TotalEntries,ContainerEntries,GeometryEntries,HasMprlMatch,AvgSurfaceCount");
                
                foreach (var group in linksByParent)
                {
                    uint parentId = group.Key;
                    var entries = group.Value;
                    
                    int containerEntries = entries.Count(link => link.MspiFirstIndex == -1);
                    int geometryEntries = entries.Count - containerEntries;
                    bool hasMprlMatch = scene.Placements?.Any(p => p.Unknown4 == parentId) ?? false;
                    
                    // Calculate average surface count for this parent group
                    double avgSurfaceCount = 0;
                    int validSurfaceRefs = 0;
                    
                    foreach (var link in entries)
                    {
                        if (link.SurfaceRefIndex >= 0 && link.SurfaceRefIndex < scene.Groups.Count)
                        {
                            int faceCount = scene.Groups[link.SurfaceRefIndex].Faces?.Count ?? 0;
                            if (faceCount > 0)
                            {
                                avgSurfaceCount += faceCount;
                                validSurfaceRefs++;
                            }
                        }
                    }
                    
                    if (validSurfaceRefs > 0)
                    {
                        avgSurfaceCount /= validSurfaceRefs;
                    }
                    
                    writer.WriteLine($"{parentId},{entries.Count},{containerEntries},{geometryEntries},{hasMprlMatch},{avgSurfaceCount:F2}");
                    
                    // Categorize group types
                    if (containerEntries > 0 && geometryEntries == 0)
                    {
                        containerGroups++;
                    }
                    else if (containerEntries > 0 && geometryEntries > 0)
                    {
                        mixedGroups++;
                    }
                    else if (containerEntries == 0 && geometryEntries > 0)
                    {
                        geometryOnlyGroups++;
                    }
                }
            }
            
            Console.WriteLine($"[PM4 DIAGNOSTICS] MSLK Hierarchy Analysis:");
            Console.WriteLine($"  - Found {linksByParent.Count} unique parent groups");
            Console.WriteLine($"  - Container-only groups: {containerGroups}");
            Console.WriteLine($"  - Mixed container/geometry groups: {mixedGroups}");
            Console.WriteLine($"  - Geometry-only groups: {geometryOnlyGroups}");
        }
        
        /// <summary>
        /// Analyze building hierarchy and container nesting
        /// </summary>
        private static void AnalyzeBuildingHierarchy(Pm4Scene scene, string outputDir)
        {
            if (scene.Placements == null || scene.Placements.Count == 0 || scene.Links == null || scene.Links.Count == 0)
            {
                Console.WriteLine("[PM4 DIAGNOSTICS] Insufficient data for building hierarchy analysis");
                return;
            }
            
            Console.WriteLine("[PM4 DIAGNOSTICS] Analyzing building hierarchy and container nesting...");
            
            // Create lookup of MPRL placements by Unknown4 (building ID)
            var placementsByUnknown4 = scene.Placements
                .GroupBy(p => p.Unknown4)
                .ToDictionary(g => g.Key, g => g.ToList());
            
            using (var writer = new StreamWriter(Path.Combine(outputDir, "building_hierarchy.csv")))
            {
                writer.WriteLine("BuildingId,MprlCount,TotalLinks,ContainerLinks,GeometryLinks,TotalTriangles,UsedSurfaceGroups");
                
                // For each unique MPRL.Unknown4 value (building ID)
                foreach (var buildingGroup in placementsByUnknown4)
                {
                    uint buildingId = buildingGroup.Key;
                    var placements = buildingGroup.Value;
                    
                    // Find all MSLK entries with matching ParentId
                    var linkEntries = scene.Links
                        .Where(link => link.ParentId == buildingId)
                        .ToList();
                    
                    int containerLinks = linkEntries.Count(link => link.MspiFirstIndex == -1);
                    int geometryLinks = linkEntries.Count - containerLinks;
                    
                    // Calculate total triangles and used surface groups
                    var usedSurfaceGroups = new HashSet<int>();
                    int totalTriangles = 0;
                    
                    foreach (var link in linkEntries)
                    {
                        if (link.SurfaceRefIndex >= 0 && link.SurfaceRefIndex < scene.Groups.Count)
                        {
                            usedSurfaceGroups.Add(link.SurfaceRefIndex);
                            totalTriangles += scene.Groups[link.SurfaceRefIndex].Faces?.Count ?? 0;
                        }
                    }
                    
                    writer.WriteLine($"{buildingId},{placements.Count},{linkEntries.Count},{containerLinks},{geometryLinks},{totalTriangles},{usedSurfaceGroups.Count}");
                }
            }
            
            // Identify potential lost/orphaned links
            var allMprlIds = new HashSet<uint>();
            foreach (var p in scene.Placements)
            {
                allMprlIds.Add(p.Unknown4);
            }
            
            var allMslkParentIds = new List<uint>();
            var seen = new HashSet<uint>();
            foreach (var link in scene.Links)
            {
                if (!seen.Contains(link.ParentId))
                {
                    allMslkParentIds.Add(link.ParentId);
                    seen.Add(link.ParentId);
                }
            }
            
            var orphanedParentIds = new List<uint>();
            foreach (var id in allMslkParentIds)
            {
                if (!allMprlIds.Contains(id))
                {
                    orphanedParentIds.Add(id);
                }
            }
            
            Console.WriteLine($"[PM4 DIAGNOSTICS] Building Hierarchy Analysis:");
            Console.WriteLine($"  - Found {placementsByUnknown4.Count} unique buildings from MPRL.Unknown4");
            Console.WriteLine($"  - Found {allMslkParentIds.Count} unique ParentIds in MSLK");
            Console.WriteLine($"  - Orphaned MSLK parent IDs (no matching MPRL): {orphanedParentIds.Count}");
            
            if (orphanedParentIds.Count > 0)
            {
                Console.WriteLine($"  - Sample orphaned IDs: {string.Join(", ", orphanedParentIds.Take(5))}");
                
                using (var writer = new StreamWriter(Path.Combine(outputDir, "orphaned_links.csv")))
                {
                    writer.WriteLine("ParentId,EntryCount,ContainerCount,GeometryCount");
                    
                    foreach (var parentId in orphanedParentIds)
                    {
                        var entries = scene.Links.Where(link => link.ParentId == parentId).ToList();
                        int containers = entries.Count(link => link.MspiFirstIndex == -1);
                        int geometry = entries.Count - containers;
                        
                        writer.WriteLine($"{parentId},{entries.Count},{containers},{geometry}");
                    }
                }
            }
        }
        
        /// <summary>
        /// Analyze vertex pool usage across surface groups
        /// </summary>
        private static void AnalyzeVertexPoolUsage(Pm4Scene scene, string outputDir)
        {
            if (scene.Groups == null || scene.Groups.Count == 0)
            {
                Console.WriteLine("[PM4 DIAGNOSTICS] No surface groups found for vertex pool analysis");
                return;
            }
            
            Console.WriteLine("[PM4 DIAGNOSTICS] Analyzing vertex pool usage...");
            
            int totalVertexCount = scene.Vertices?.Count ?? 0;
            int totalMscnCount = scene.MscnVertices?.Count ?? 0;
            
            Console.WriteLine($"[PM4 DIAGNOSTICS] Vertex Pools:");
            Console.WriteLine($"  - MSVT Vertices: {totalVertexCount}");
            Console.WriteLine($"  - MSCN Vertices: {totalMscnCount}");
            Console.WriteLine($"  - Total Available: {totalVertexCount + totalMscnCount}");
            
            // Track vertex index ranges used by each surface group
            using (var writer = new StreamWriter(Path.Combine(outputDir, "vertex_pool_usage.csv")))
            {
                writer.WriteLine("SurfaceIndex,FaceCount,MinVertexIndex,MaxVertexIndex,MsvtCount,MscnCount,OutOfRangeCount");
                
                for (int groupIndex = 0; groupIndex < scene.Groups.Count; groupIndex++)
                {
                    var group = scene.Groups[groupIndex];
                    var faces = group.Faces;
                    
                    if (faces != null && faces.Count > 0)
                    {
                        // Extract and analyze all vertex indices used
                        var vertexIndices = new HashSet<int>();
                        foreach (var face in faces)
                        {
                            vertexIndices.Add(face.A);
                            vertexIndices.Add(face.B);
                            vertexIndices.Add(face.C);
                        }
                        
                        int minIndex = vertexIndices.Min();
                        int maxIndex = vertexIndices.Max();
                        
                        int msvtCount = vertexIndices.Count(i => i >= 0 && i < totalVertexCount);
                        int mscnCount = vertexIndices.Count(i => i >= totalVertexCount && i < totalVertexCount + totalMscnCount);
                        int outOfRangeCount = vertexIndices.Count(i => i < 0 || i >= totalVertexCount + totalMscnCount);
                        
                        writer.WriteLine($"{groupIndex},{faces.Count},{minIndex},{maxIndex},{msvtCount},{mscnCount},{outOfRangeCount}");
                        
                        // Report any surface group with out-of-range vertex indices
                        if (outOfRangeCount > 0)
                        {
                            Console.WriteLine($"  - WARNING: Surface group {groupIndex} has {outOfRangeCount} out-of-range vertex indices!");
                        }
                    }
                }
            }
        }
        
        /// <summary>
        /// Generate a relationship diagram (Mermaid format) showing the key chunk linkages
        /// </summary>
        private static void GenerateRelationshipDiagram(Pm4Scene scene, string outputDir)
        {
            if (scene.Placements == null || scene.Links == null || scene.Groups == null)
            {
                Console.WriteLine("[PM4 DIAGNOSTICS] Insufficient data for relationship diagram");
                return;
            }
            
            Console.WriteLine("[PM4 DIAGNOSTICS] Generating relationship diagram...");
            
            using (var writer = new StreamWriter(Path.Combine(outputDir, "relationship_diagram.md")))
            {
                writer.WriteLine("# PM4 Chunk Relationship Diagram");
                writer.WriteLine("");
                writer.WriteLine("```mermaid");
                writer.WriteLine("flowchart TD");
                writer.WriteLine("    %% Define node styles");
                writer.WriteLine("    classDef mprl fill:#f96,stroke:#333,stroke-width:2px");
                writer.WriteLine("    classDef mslk fill:#9cf,stroke:#333,stroke-width:2px");
                writer.WriteLine("    classDef msur fill:#9f9,stroke:#333,stroke-width:2px");
                writer.WriteLine("    classDef vertex fill:#f9f,stroke:#333,stroke-width:2px");
                writer.WriteLine("");
                
                // Create sample nodes for visualization (limit to avoid overwhelming diagram)
                int mprlSampleSize = Math.Min(5, scene.Placements.Count);
                var sampleMprl = scene.Placements.Take(mprlSampleSize).ToList();
                
                // Create MPRL nodes
                foreach (var placement in sampleMprl)
                {
                    writer.WriteLine($"    MPRL{placement.Unknown4}[\"MPRL {placement.Unknown4}\"]:::mprl");
                }
                
                writer.WriteLine("");
                
                // For each sample MPRL, add its child MSLK nodes
                foreach (var placement in sampleMprl)
                {
                    var childLinks = scene.Links
                        .Where(link => link.ParentId == placement.Unknown4)
                        .Take(3) // Limit to 3 child links per MPRL for clarity
                        .ToList();
                    
                    foreach (var link in childLinks)
                    {
                        string nodeType = link.MspiFirstIndex == -1 ? "Container" : "Geometry";
                        writer.WriteLine($"    MSLK{link.GetHashCode()}[\"MSLK {nodeType}\"]:::mslk");
                        writer.WriteLine($"    MPRL{placement.Unknown4} --> MSLK{link.GetHashCode()}");
                        
                        // Add MSUR node if valid
                        if (link.SurfaceRefIndex >= 0 && link.SurfaceRefIndex < scene.Groups.Count)
                        {
                            var surfaceGroup = scene.Groups[link.SurfaceRefIndex];
                            int faceCount = surfaceGroup.Faces?.Count ?? 0;
                            
                            writer.WriteLine($"    MSUR{link.SurfaceRefIndex}[\"MSUR {link.SurfaceRefIndex} ({faceCount} faces)\"]:::msur");
                            writer.WriteLine($"    MSLK{link.GetHashCode()} --> MSUR{link.SurfaceRefIndex}");
                        }
                    }
                }
                
                writer.WriteLine("```");
                writer.WriteLine("");
                writer.WriteLine("## Chunk Relationship Summary");
                writer.WriteLine("");
                writer.WriteLine("- **MPRL** (Placement) entries define object positions and use Unknown4 as building identifier");
                writer.WriteLine("- **MSLK** (Link) entries connect placements to geometry via ParentId → MPRL.Unknown4");
                writer.WriteLine("- **MSUR** (Surface) entries contain actual geometry referenced by MSLK.SurfaceRefIndex");
                writer.WriteLine("- Container nodes have MspiFirstIndex = -1 and serve as grouping nodes");
            }
            
            Console.WriteLine($"[PM4 DIAGNOSTICS] Generated relationship diagram at {Path.Combine(outputDir, "relationship_diagram.md")}");
        }
    }
}
