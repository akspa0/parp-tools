using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;
using WmoBspConverter.Bsp;
using WmoBspConverter.Export;
using System.Linq;
namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Generates Quake 3 .map file format from WMO and BSP data.
    /// .map files are the human-readable source format used in Quake 3 mapping.
    /// </summary>
    public class WmoMapGenerator
    {
        // Applies the single WMO -> Quake 3 coordinate transform used for map emission
        private static Vector3 TransformToQ3(Vector3 v)
        {
            // WMO stores coordinates as (X, Y, Z) with Z up. Quake 3 also uses Z up.
            // Align axes directly so forward (Y) stays forward and vertical (Z) stays vertical.
            return new Vector3(v.X, v.Y, v.Z);
        }

        private sealed record GeometryBounds(Vector3 Min, Vector3 Max)
        {
            public Vector3 Center => (Min + Max) * 0.5f;
            public Vector3 Size => Max - Min;
        }

        private sealed class MapContext
        {
            public GeometryBounds Bounds { get; init; } = new GeometryBounds(Vector3.Zero, Vector3.Zero);
            public Vector3 GeometryOffset { get; init; } = Vector3.Zero;
        }

        private static GeometryBounds ComputeGeometryBounds(BspFile bspFile)
        {
            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            foreach (var vertex in bspFile.Vertices)
            {
                var p = TransformToQ3(vertex.Position);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }

            if (bspFile.Vertices.Count == 0)
            {
                min = Vector3.Zero;
                max = Vector3.Zero;
            }

            return new GeometryBounds(min, max);
        }

        private static Vector3 ComputeGeometryOffset(GeometryBounds bounds)
        {
            var center = bounds.Center;
            return new Vector3(center.X, center.Y, bounds.Min.Z);
        }

        private (MapContext context, GeometryBounds paddedBounds) PrepareContext(BspFile bspFile, Vector3? forcedOffset = null)
        {
            var bounds = ComputeGeometryBounds(bspFile);
            var offset = forcedOffset ?? ComputeGeometryOffset(bounds);
            var padding = new Vector3(128f, 128f, 128f);
            
            // Note: If using forcedOffset, the bounds relative to that offset might be far away.
            // But we still need paddedBounds to enclose the geometry in map coordinates.
            // Map Space Point = Q3 Point - Offset
            // So Map Bounds Min = Q3 Bounds Min - Offset
            var mapMin = bounds.Min - offset;
            var mapMax = bounds.Max - offset;
            
            var paddedBounds = new GeometryBounds(mapMin - padding, mapMax + padding);

            Console.WriteLine($"[DEBUG] Geometry bounds min={bounds.Min}, max={bounds.Max}, offset={offset}");

            return (new MapContext { Bounds = bounds, GeometryOffset = offset }, paddedBounds);
        }

        public void GenerateMapFilePerGroup(string baseOutputPath, WmoV14Parser.WmoV14Data wmoData, BspFile fullBspFile)
        {
            // Compute global offset from the FULL geometry so all groups stay aligned
            var globalBounds = ComputeGeometryBounds(fullBspFile);
            var globalOffset = ComputeGeometryOffset(globalBounds);
            Console.WriteLine($"[INFO] Computed global offset: {globalOffset} (from {fullBspFile.Vertices.Count} vertices)");

            // Export each WMO group as a separate .map file
            Console.WriteLine($"[INFO] Exporting {wmoData.Groups.Count} groups as separate .map files...");
            
            int groupIndex = 0;
            foreach (var group in wmoData.Groups)
            {
                if (group.Vertices.Count == 0)
                {
                    Console.WriteLine($"[SKIP] Group {groupIndex} has no vertices");
                    groupIndex++;
                    continue;
                }
                
                var groupPath = Path.Combine(
                    Path.GetDirectoryName(baseOutputPath) ?? "",
                    $"{Path.GetFileNameWithoutExtension(baseOutputPath)}_group{groupIndex:D3}.map"
                );
                
                // Create a mini BSP with just this group's data
                var groupBsp = new BspFile();
                
                // Add vertices
                foreach (var v in group.Vertices)
                {
                    groupBsp.Vertices.Add(new BspVertex { Position = v });
                }
                
                // Add faces (triangles from indices) - must populate Vertex0/1/2 with actual indices
                for (int i = 0; i < group.Indices.Count; i += 3)
                {
                    if (i + 2 < group.Indices.Count)
                    {
                        var i0 = group.Indices[i];
                        var i1 = group.Indices[i + 1];
                        var i2 = group.Indices[i + 2];
                        groupBsp.Faces.Add(new BspFace 
                        { 
                            FirstVertex = i0,
                            NumVertices = 3,
                            Texture = i / 3 < group.FaceMaterials.Count ? group.FaceMaterials[i / 3] : 0,
                            // Explicit vertex indices for the triangle
                            Vertex0 = i0,
                            Vertex1 = i1,
                            Vertex2 = i2
                        });
                    }
                }
                
                // Copy textures
                groupBsp.Textures.AddRange(fullBspFile.Textures);
                
                // Pass the GLOBAL offset to ensure this group aligns with others
                GenerateMapFile(groupPath, wmoData, groupBsp, globalOffset);
                Console.WriteLine($"[INFO] Exported group {groupIndex}: {groupPath} ({group.Vertices.Count} verts, {group.Indices.Count / 3} faces)");
                
                groupIndex++;
            }
        }

        public void GenerateMapFileClustered(string baseOutputPath, WmoV14Parser.WmoV14Data wmoData, BspFile fullBspFile)
        {
            // Compute global offset so all clusters align
            var globalBounds = ComputeGeometryBounds(fullBspFile);
            var globalOffset = ComputeGeometryOffset(globalBounds);
            Console.WriteLine($"[INFO] Computed global offset: {globalOffset}");

            var clusters = ComputeClusters(wmoData);
            Console.WriteLine($"[INFO] Identified {clusters.Count} logical clusters from {wmoData.Groups.Count} groups.");
            
            for (int i = 0; i < clusters.Count; i++)
            {
                var cluster = clusters[i];
                if (cluster.Count == 0) continue;

                // Determine name: Cluster 0 is usually Exterior
                string suffix = (i == 0) ? "_Exterior" : $"_Interior_C{i:D2}";
                var dir = Path.GetDirectoryName(baseOutputPath);
                var name = Path.GetFileNameWithoutExtension(baseOutputPath);
                // Handle potential double extension or pre-existing suffixes
                if (name.EndsWith(".map", StringComparison.OrdinalIgnoreCase)) 
                    name = Path.GetFileNameWithoutExtension(name);
                
                var path = Path.Combine(string.IsNullOrEmpty(dir) ? "" : dir, $"{name}{suffix}.map");

                Console.WriteLine($"[INFO] Exporting Cluster {i} to {path} ({cluster.Count} groups)...");
                
                // Build sub-BSP
                var clusterBsp = new BspFile();
                clusterBsp.Textures.AddRange(fullBspFile.Textures);
                
                int totalVerts = 0;
                foreach (var groupIdx in cluster)
                {
                    var group = wmoData.Groups[groupIdx];
                    int vertexBase = clusterBsp.Vertices.Count;
                    
                    // Add vertices
                    foreach (var v in group.Vertices)
                        clusterBsp.Vertices.Add(new BspVertex { Position = v });
                        
                    // Add faces
                    for (int f = 0; f < group.Indices.Count; f += 3)
                    {
                        if (f + 2 < group.Indices.Count)
                        {
                            var i0 = group.Indices[f];
                            var i1 = group.Indices[f + 1];
                            var i2 = group.Indices[f + 2];
                            int matId = f / 3 < group.FaceMaterials.Count ? group.FaceMaterials[f / 3] : 0;
                            
                            clusterBsp.Faces.Add(new BspFace 
                            { 
                                FirstVertex = vertexBase + i0,
                                NumVertices = 3,
                                Texture = matId,
                                Vertex0 = vertexBase + i0,
                                Vertex1 = vertexBase + i1,
                                Vertex2 = vertexBase + i2
                            });
                        }
                    }
                    totalVerts += group.Vertices.Count;
                }
                
                GenerateMapFile(path, wmoData, clusterBsp, globalOffset);
            }
        }

        private List<List<int>> ComputeClusters(WmoV14Parser.WmoV14Data wmoData, int maxFacesPerCluster = 20000, float spatialTolerance = 50.0f)
        {
            var clusters = new List<List<int>>();
            var exteriorIndices = new List<int>();
            var interiorIndices = new List<int>();
            
            // 1. Separate Exterior/Outdoor from Interior
            for (int i = 0; i < wmoData.Groups.Count; i++)
            {
                // Flags: 0x8=Exterior, 0x40=Interior, 0x2000=Outdoor
                // If Exterior or Outdoor, treat as Shell
                if ((wmoData.Groups[i].Flags & (0x8 | 0x2000)) != 0)
                {
                    exteriorIndices.Add(i);
                }
                else
                {
                    interiorIndices.Add(i);
                }
            }
            
            // Cluster 0 is always Exterior (even if empty, to keep indices consistent)
            clusters.Add(exteriorIndices);
            
            // OPTIMIZATION: If maxFaces is huge (ASE Mode), skip graph building and return single interior cluster
            if (maxFacesPerCluster >= 10000000)
            {
                Console.WriteLine("[INFO] Merging all interior groups into single cluster (High Limit Mode)");
                if (interiorIndices.Count > 0)
                    clusters.Add(interiorIndices);
                return clusters;
            }

            // 2. Build Adjacency Graph
            // Prefer Portals (Logical), fallback to Spatial (Touching AABBs)
            var adjacency = new Dictionary<int, HashSet<int>>();
            bool usePortals = wmoData.Portals.Count > 0;
            
            if (usePortals)
            {
                Console.WriteLine($"[INFO] Building cluster graph from {wmoData.Portals.Count} portals...");
                foreach (var portal in wmoData.Portals)
                {
                    int s = portal.StartGroup;
                    int e = portal.EndGroup;
                    
                    if (!adjacency.ContainsKey(s)) adjacency[s] = new HashSet<int>();
                    if (!adjacency.ContainsKey(e)) adjacency[e] = new HashSet<int>();
                    
                    adjacency[s].Add(e);
                    adjacency[e].Add(s);
                }
            }
            else
            {
                Console.WriteLine("[WARN] No portals found. Falling back to Spatial Clustering (Touching AABBs).");
                // N^2 check for touching AABBs
                // Tolerance 50.0f units (generous gap to account for loose wmo group placement)
                float tol = spatialTolerance;
                
                for (int i = 0; i < interiorIndices.Count; i++)
                {
                    int idxA = interiorIndices[i];
                    var groupA = wmoData.Groups[idxA];
                    var boundsA = GetGroupBounds(groupA);
                    
                    for (int j = i + 1; j < interiorIndices.Count; j++)
                    {
                        int idxB = interiorIndices[j];
                        var groupB = wmoData.Groups[idxB];
                        var boundsB = GetGroupBounds(groupB);
                        
                        if (BoundsIntersect(boundsA, boundsB, tol))
                        {
                            if (!adjacency.ContainsKey(idxA)) adjacency[idxA] = new HashSet<int>();
                            if (!adjacency.ContainsKey(idxB)) adjacency[idxB] = new HashSet<int>();
                            
                            adjacency[idxA].Add(idxB);
                            adjacency[idxB].Add(idxA);
                        }
                    }
                }
            }
            
            // 3. Cluster Interiors via BFS/Connected Components
            // Enforce MAX SIZE check (~20k faces per cluster ~ 6-8MB map)
            // Limit controlled by argument
            
            var visited = new HashSet<int>();
            var interiorSet = new HashSet<int>(interiorIndices);
            
            foreach (var startNode in interiorIndices)
            {
                if (visited.Contains(startNode)) continue;
                
                // Start new cluster
                var cluster = new List<int>();
                var queue = new Queue<int>();
                queue.Enqueue(startNode);
                visited.Add(startNode);
                
                int clusterFaceCount = wmoData.Groups[startNode].Indices.Count / 3;
                
                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    cluster.Add(current);
                    
                    if (adjacency.ContainsKey(current))
                    {
                        foreach (var neighbor in adjacency[current])
                        {
                            if (interiorSet.Contains(neighbor) && !visited.Contains(neighbor))
                            {
                                int neighborFaces = wmoData.Groups[neighbor].Indices.Count / 3;
                                
                                // Check size limit logic
                                // If adding this neighbor exceeds limit, DO NOT add it yet?
                                // If we don't add it, it will start its own cluster later.
                                // But if it's strongly connected, this might split a room.
                                // Heuristic: Only split if cluster is already "big enough".
                                
                                if (clusterFaceCount + neighborFaces <= maxFacesPerCluster)
                                {
                                    visited.Add(neighbor);
                                    queue.Enqueue(neighbor);
                                    clusterFaceCount += neighborFaces;
                                }
                                else
                                {
                                    // Soft limit hit: Stop expanding this branch.
                                    // The neighbor remains unvisited and will spawn a new cluster loop.
                                }
                            }
                        }
                    }
                }
                clusters.Add(cluster);
            }
            
            return clusters;
        }

        private (Vector3 Min, Vector3 Max) GetGroupBounds(WmoV14Parser.WmoGroupData group)
        {
            if (group.Vertices.Count == 0) return (Vector3.Zero, Vector3.Zero);
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var v in group.Vertices)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            return (min, max);
        }
        
        private bool BoundsIntersect((Vector3 Min, Vector3 Max) a, (Vector3 Min, Vector3 Max) b, float tol)
        {
            if (a.Max.X + tol < b.Min.X || a.Min.X - tol > b.Max.X) return false;
            if (a.Max.Y + tol < b.Min.Y || a.Min.Y - tol > b.Max.Y) return false;
            if (a.Max.Z + tol < b.Min.Z || a.Min.Z - tol > b.Max.Z) return false;
            return true;
        }

        
        public void GenerateMapFile(string outputPath, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile, Vector3? forcedOffset = null)
        {
            var mapContent = new StringBuilder();
            
            // Add header info
            mapContent.AppendLine("// Auto-generated from WMO v14 file");
            mapContent.AppendLine($"// Original: WMO v{wmoData.Version}");
            mapContent.AppendLine($"// Groups: {wmoData.Groups.Count}");
            mapContent.AppendLine($"// Textures: {wmoData.Textures.Count}");
            mapContent.AppendLine();
            
            var (context, paddedBounds) = PrepareContext(bspFile, forcedOffset);
            var combinedMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var combinedMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            // Create sealed worldspawn box to contain the WMO (leaves worldspawn OPEN)
            CreateSealedWorldspawn(mapContent, wmoData, paddedBounds);
            Console.WriteLine($"[DEBUG] After worldspawn box, length: {mapContent.Length:N0}");
            
            // Add WMO geometry brushes directly inside worldspawn (NOT in separate entity!)
            // Q3Map2 only compiles brushes that are inside the worldspawn entity
            mapContent.AppendLine("// WMO geometry brushes");
            var defaultTex = bspFile.Textures.Count > 0 ? bspFile.Textures[0].Name : "textures/common/caulk";
            GenerateBrushesFromGeometry(mapContent, bspFile, defaultTex, context);
            
            // NOW close the worldspawn entity
            mapContent.AppendLine("}");
            mapContent.AppendLine();
            Console.WriteLine($"[DEBUG] After worldspawn closed, length: {mapContent.Length:N0}");
            
            // Add player spawn entity (OUTSIDE worldspawn)
            AddSpawnEntity(mapContent, context, paddedBounds.Min.Z);
            Console.WriteLine($"[DEBUG] After AddSpawn, length: {mapContent.Length:N0}");
            
            // Generate texture info
            GenerateTextureInfo(mapContent, wmoData);
            Console.WriteLine($"[DEBUG] After GenerateTextureInfo, length: {mapContent.Length:N0}");
            
            // Generate Portals (Hint brushes)
            GeneratePortalBrushes(mapContent, wmoData, context);
            Console.WriteLine($"[DEBUG] After GeneratePortalBrushes, length: {mapContent.Length:N0}");
            
            // Write to file
            Console.WriteLine($"[DEBUG] StringBuilder length: {mapContent.Length:N0} characters");
            var finalContent = mapContent.ToString();
            Console.WriteLine($"[DEBUG] Final string length: {finalContent.Length:N0} characters");
            File.WriteAllText(outputPath, finalContent);
            
            Console.WriteLine($"[INFO] Generated .map file: {outputPath}");
        }

        /// <summary>
        /// Generate .map files directly from WMO groups, using BSP reconstruction for interiors.
        /// Outputs ONE .map file per group (WMO groups are map-sized).
        /// </summary>
        public void GenerateMapFileFromGroups(
            string outputPath, 
            WmoV14Parser.WmoV14Data wmoData, 
            bool interiorsOnly = true)
        {
            var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            var baseName = Path.GetFileNameWithoutExtension(outputPath);
            
            int processedGroups = 0;
            int skippedExterior = 0;
            int skippedNoBsp = 0;
            
            for (int groupIdx = 0; groupIdx < wmoData.Groups.Count; groupIdx++)
            {
                var group = wmoData.Groups[groupIdx];
                bool isExterior = (group.Flags & (0x8 | 0x2000)) != 0;
                
                if (interiorsOnly && isExterior)
                {
                    skippedExterior++;
                    continue;
                }
                
                if (group.BspNodes.Count == 0)
                {
                    skippedNoBsp++;
                    Console.WriteLine($"[SKIP] Group {groupIdx} '{group.Name}' has no BSP nodes");
                    continue;
                }
                
                if (group.Vertices.Count == 0)
                {
                    continue;
                }
                
                // Compute bounds for THIS group only
                var groupMin = new Vector3(float.MaxValue);
                var groupMax = new Vector3(float.MinValue);
                foreach (var v in group.Vertices)
                {
                    groupMin = Vector3.Min(groupMin, v);
                    groupMax = Vector3.Max(groupMax, v);
                }
                
                var padding = new Vector3(128f);
                var paddedMin = groupMin - padding;
                var paddedMax = groupMax + padding;
                // Center offset for this group
                var offset = new Vector3(
                    (groupMin.X + groupMax.X) / 2, 
                    (groupMin.Y + groupMax.Y) / 2, 
                    groupMin.Z);
                
                var context = new MapContext 
                { 
                    Bounds = new GeometryBounds(groupMin, groupMax),
                    GeometryOffset = offset
                };
                var paddedBounds = new GeometryBounds(paddedMin - offset, paddedMax - offset);
                
                // Build map content for this group
                var mapContent = new StringBuilder();
                mapContent.AppendLine($"// Auto-generated from WMO v14 (BSP Mode) - Group {groupIdx}");
                mapContent.AppendLine($"// Group: {group.Name}");
                mapContent.AppendLine($"// Flags: 0x{group.Flags:X8}");
                mapContent.AppendLine($"// BSP Nodes: {group.BspNodes.Count}");
                mapContent.AppendLine();
                
                // Create sealed worldspawn for this group
                CreateSealedWorldspawn(mapContent, wmoData, paddedBounds);
                
                // Generate brushes from BSP tree
                GenerateBrushesFromWmoGroup(mapContent, group, wmoData, context);
                
                // Close worldspawn
                mapContent.AppendLine("}");
                mapContent.AppendLine();
                
                // Add spawn point
                AddSpawnEntity(mapContent, context, paddedBounds.Min.Z);
                
                // Write to per-group file
                var groupPath = Path.Combine(outputDir, $"{baseName}_group{groupIdx:D3}.map");
                File.WriteAllText(groupPath, mapContent.ToString());
                
                Console.WriteLine($"[OK] Group {groupIdx}: {groupPath} ({group.BspNodes.Count} BSP nodes)");
                processedGroups++;
            }
            
            Console.WriteLine();
            Console.WriteLine($"[SUMMARY] Generated {processedGroups} .map files");
            Console.WriteLine($"          Skipped {skippedExterior} exterior, {skippedNoBsp} no-BSP groups");
        }

        /// <summary>
        /// Generate .map files using BSP reconstruction with portal-based clustering.
        /// Connected interior groups are combined into single maps.
        /// </summary>
        public void GenerateClusteredBspMaps(
            string outputPath, 
            WmoV14Parser.WmoV14Data wmoData)
        {
            var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            var baseName = Path.GetFileNameWithoutExtension(outputPath);
            
            // Use portal-based clustering to find connected interior groups
            var clusters = ComputeClusters(wmoData, maxFacesPerCluster: 50000);
            
            Console.WriteLine($"[INFO] Found {clusters.Count} clusters (Cluster 0 = Exterior)");
            
            int mapCount = 0;
            
            for (int clusterIdx = 0; clusterIdx < clusters.Count; clusterIdx++)
            {
                var cluster = clusters[clusterIdx];
                if (cluster.Count == 0) continue;
                
                // Skip cluster 0 (Exterior)
                if (clusterIdx == 0)
                {
                    Console.WriteLine($"[SKIP] Cluster 0 (Exterior): {cluster.Count} groups");
                    continue;
                }
                
                // Filter to groups with BSP data
                var bspGroups = cluster.Where(idx => wmoData.Groups[idx].BspNodes.Count > 0).ToList();
                if (bspGroups.Count == 0)
                {
                    Console.WriteLine($"[SKIP] Cluster {clusterIdx}: No BSP nodes in {cluster.Count} groups");
                    continue;
                }
                
                // Compute combined bounds for this cluster
                var clusterMin = new Vector3(float.MaxValue);
                var clusterMax = new Vector3(float.MinValue);
                foreach (var groupIdx in bspGroups)
                {
                    var group = wmoData.Groups[groupIdx];
                    foreach (var v in group.Vertices)
                    {
                        clusterMin = Vector3.Min(clusterMin, v);
                        clusterMax = Vector3.Max(clusterMax, v);
                    }
                }
                
                var padding = new Vector3(128f);
                var paddedMin = clusterMin - padding;
                var paddedMax = clusterMax + padding;
                var offset = new Vector3(
                    (clusterMin.X + clusterMax.X) / 2, 
                    (clusterMin.Y + clusterMax.Y) / 2, 
                    clusterMin.Z);
                
                var context = new MapContext 
                { 
                    Bounds = new GeometryBounds(clusterMin, clusterMax),
                    GeometryOffset = offset
                };
                var paddedBounds = new GeometryBounds(paddedMin - offset, paddedMax - offset);
                
                // Build map content
                var mapContent = new StringBuilder();
                mapContent.AppendLine($"// Auto-generated from WMO v14 (BSP Mode) - Cluster {clusterIdx}");
                mapContent.AppendLine($"// Groups: {string.Join(", ", bspGroups)}");
                mapContent.AppendLine($"// Total BSP Groups: {bspGroups.Count}");
                mapContent.AppendLine();
                
                // Create sealed worldspawn
                CreateSealedWorldspawn(mapContent, wmoData, paddedBounds);
                
                // Generate brushes for each group in cluster
                int totalBrushes = 0;
                foreach (var groupIdx in bspGroups)
                {
                    var group = wmoData.Groups[groupIdx];
                    GenerateBrushesFromWmoGroup(mapContent, group, wmoData, context);
                }
                
                // Close worldspawn
                mapContent.AppendLine("}");
                mapContent.AppendLine();
                
                // Add spawn point
                AddSpawnEntity(mapContent, context, paddedBounds.Min.Z);
                
                // Write file
                var clusterPath = Path.Combine(outputDir, $"{baseName}_cluster{clusterIdx:D2}.map");
                File.WriteAllText(clusterPath, mapContent.ToString());
                
                Console.WriteLine($"[OK] Cluster {clusterIdx}: {clusterPath} ({bspGroups.Count} groups)");
                mapCount++;
            }
            
            Console.WriteLine();
            Console.WriteLine($"[SUMMARY] Generated {mapCount} clustered .map files");
        }


        public void GenerateMapFileWithModels(
            string outputPath,
            string outputRootDir,
            string wmoName,
            WmoV14Parser.WmoV14Data wmoData,
            BspFile bspFile,
            AseWriter aseWriter)
        {
            var (context, paddedBounds) = PrepareContext(bspFile);
            var materialShaderNames = BuildMaterialShaderNames(wmoData);
            var placements = new List<(int GroupIndex, AseWriter.ExportResult Result)>();
            var modelMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var modelMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

            int g = 0;
            foreach (var group in wmoData.Groups)
            {
                var intIndices = group.Indices.ConvertAll(i => (int)i);
                var faceMats = group.FaceMaterials.ConvertAll(f => (int)f);
                var result = aseWriter.ExportGroup(
                    outputRootDir,
                    wmoName,
                    g,
                    group.Vertices,
                    intIndices,
                    faceMats,
                    materialShaderNames,
                    context.GeometryOffset);

                placements.Add((g, result));
                modelMin = Vector3.Min(modelMin, result.RoomMin);
                modelMax = Vector3.Max(modelMax, result.RoomMax);
                g++;
            }

            var zShift = placements.Count == 0 ? 0f : paddedBounds.Min.Z - modelMin.Z;
            const float modelLift = 8.0f; // raise models slightly above floor to avoid z-fighting
            zShift += modelLift;

            var mapContent = new StringBuilder();
            mapContent.AppendLine("// Auto-generated from WMO v14 file (ASE model placement)");
            mapContent.AppendLine($"// Original: WMO v{wmoData.Version}");
            mapContent.AppendLine($"// Groups: {wmoData.Groups.Count}");
            mapContent.AppendLine($"// Textures: {wmoData.Textures.Count}");
            mapContent.AppendLine();

            // Sealed room and spawn
            CreateSealedWorldspawn(mapContent, wmoData, paddedBounds);
            AddSpawnEntity(mapContent, context, paddedBounds.Min.Z);

            foreach (var placement in placements)
            {
                var result = placement.Result;
                var adjustedOrigin = new Vector3(
                    result.RoomCenter.X,
                    result.RoomCenter.Y,
                    result.RoomCenter.Z + zShift);

                mapContent.AppendLine("// WMO group model");
                mapContent.AppendLine("{");
                mapContent.AppendLine("\"classname\" \"misc_model\"");
                mapContent.AppendLine($"\"model\" \"{result.RelativeModelPath.Replace("\\", "/")}\"");
                mapContent.AppendLine($"\"origin\" \"{adjustedOrigin.X:F3} {adjustedOrigin.Y:F3} {adjustedOrigin.Z:F3}\"");
                mapContent.AppendLine($"\"_wmo_group\" \"{placement.GroupIndex}\"");
                mapContent.AppendLine("}");
                mapContent.AppendLine();
            }

            Console.WriteLine($"[DEBUG] Combined model bounds (room): min={modelMin}, max={modelMax}, zShift={zShift}");

            File.WriteAllText(outputPath, mapContent.ToString());
            Console.WriteLine($"[INFO] Generated .map (models): {outputPath}");
        }

        private void CreateSealedWorldspawn(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, GeometryBounds bounds)
        {
            var min = bounds.Min;
            var max = bounds.Max;

            mapContent.AppendLine("// Sealed worldspawn room");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine($"\"message\" \"WMO v{wmoData.Version} in sealed room\"");
            
            // Create 6-sided sealed box (hollow room)
            // Each face is a brush with caulk texture
            
            // Floor
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X+1} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z-16} ) ( {min.X} {min.Y+1} {min.Z-16} ) ( {min.X+1} {min.Y} {min.Z-16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y} {min.Z-16} ) ( {min.X+1} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z-16} ) ( {max.X} {max.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) ( {min.X} {min.Y} {min.Z-16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X+1} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z-16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // Ceiling
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z+16} ) ( {min.X} {min.Y+1} {max.Z+16} ) ( {min.X+1} {min.Y} {max.Z+16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X+1} {min.Y} {max.Z} ) ( {min.X} {min.Y+1} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y} {max.Z+16} ) ( {min.X+1} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {max.Z} ) ( {max.X} {max.Y} {max.Z+16} ) ( {max.X} {max.Y+1} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y+1} {max.Z} ) ( {min.X} {min.Y} {max.Z+16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {max.Z} ) ( {max.X+1} {max.Y} {max.Z} ) ( {max.X} {max.Y} {max.Z+16} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // -X wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X-16} {min.Y} {min.Z} ) ( {min.X-16} {min.Y} {max.Z} ) ( {min.X-16} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) ( {min.X-16} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {min.X-16} {max.Y} {min.Z} ) ( {min.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X-16} {min.Y} {min.Z} ) ( {min.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y+1} {max.Z} ) ( {min.X-16} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // +X wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {max.X+16} {min.Y} {min.Z} ) ( {max.X+16} {min.Y+1} {min.Z} ) ( {max.X+16} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {max.Z} ) ( {max.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {max.Z} ) ( {max.X+16} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X+16} {max.Y} {min.Z} ) ( {max.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X+16} {min.Y} {min.Z} ) ( {max.X} {min.Y+1} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {max.Z} ) ( {max.X} {min.Y+1} {max.Z} ) ( {max.X+16} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // -Y wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) ( {max.X} {min.Y} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y-16} {min.Z} ) ( {max.X} {min.Y-16} {min.Z} ) ( {min.X} {min.Y-16} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {min.X} {min.Y-16} {min.Z} ) ( {min.X} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {max.Z} ) ( {max.X} {min.Y-16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {min.Z} ) ( {max.X} {min.Y} {min.Z} ) ( {min.X} {min.Y-16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {min.Y} {max.Z} ) ( {min.X} {min.Y-16} {max.Z} ) ( {max.X} {min.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // +Y wall
            mapContent.AppendLine("{");
            mapContent.AppendLine($"( {min.X} {max.Y+16} {min.Z} ) ( {min.X} {max.Y+16} {max.Z} ) ( {max.X} {max.Y+16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z} ) ( {min.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {min.X} {max.Y+16} {min.Z} ) ( {min.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {max.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {max.Z} ) ( {max.X} {max.Y+16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {min.Z} ) ( {max.X} {max.Y} {min.Z} ) ( {min.X} {max.Y+16} {min.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine($"( {min.X} {max.Y} {max.Z} ) ( {min.X} {max.Y+16} {max.Z} ) ( {max.X} {max.Y} {max.Z} ) common/caulk 0 0 0 0.5 0.5 0 0 0");
            mapContent.AppendLine("}");
            
            // NOTE: worldspawn entity is left OPEN here for geometry to be added
            // The closing brace will be added by the caller after GenerateBrushesFromGeometry
        }
        
        private void StartWorldspawnEntity(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            mapContent.AppendLine("// Worldspawn entity");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine($"\"message\" \"WMO v{wmoData.Version} converted to Quake 3\"");
            mapContent.AppendLine($"\"wmo_groups\" \"{wmoData.Groups.Count}\"");
            mapContent.AppendLine($"\"wmo_textures\" \"{wmoData.Textures.Count}\"");
            mapContent.AppendLine($"\"numpolygons\" \"{bspFile.Faces.Count}\"");
            mapContent.AppendLine($"\"numvertices\" \"{bspFile.Vertices.Count}\"");
            // worldspawn stays open; brushes will follow
        }

        private void EndWorldspawnEntity(StringBuilder mapContent)
        {
            Console.WriteLine("[DEBUG] Closing worldspawn entity");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }
        
        private void StartFuncGroupEntity(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData)
        {
            mapContent.AppendLine("// WMO geometry as func_group");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"func_group\"");
            mapContent.AppendLine($"\"_wmo_name\" \"{Path.GetFileNameWithoutExtension(wmoData.FileBytes.Length > 0 ? "wmo" : "unknown")}\"");
            mapContent.AppendLine($"\"_wmo_groups\" \"{wmoData.Groups.Count}\"");
            // func_group stays open; brushes will follow
        }
        
        private void EndFuncGroupEntity(StringBuilder mapContent)
        {
            Console.WriteLine("[DEBUG] Closing func_group entity");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }

        private void GenerateBrushesFromData(StringBuilder mapContent, WmoV14Parser.WmoGroupData group, BspFile bspFile, string defaultTexture, MapContext context, WmoV14Parser.WmoV14Data fullWmoData)
        {
            // Determine generation strategy
            // 0x8 = Exterior, 0x2000 = Outdoor (e.g. city streets). 
            // Interior is usually lack of these, or 0x40 (Indoor).
            bool isExterior = (group.Flags & (0x8 | 0x2000)) != 0;
            bool hasBsp = group.BspNodes.Count > 0;
            
            bool useBsp = !isExterior && hasBsp;
            
            Console.WriteLine($"[DEBUG] Group '{group.Name}' Flags=0x{group.Flags:X} Ext={isExterior} BspNodes={group.BspNodes.Count} -> Strategy: {(useBsp ? "BSP RECONSTRUCTION" : "TRIANGLE MESH")}");

            if (useBsp)
            {
                try 
                {
                    var walker = new BspTreeWalker();
                    // Need full material list for texture lookup? 
                    // Current walker creates "caulk" but we can improve.
                    // Passing simplified texture list for now.
                    var brushes = walker.GenerateBrushes(group, context.GeometryOffset, fullWmoData.Materials, fullWmoData.Textures);
                    
                    mapContent.AppendLine($"// BSP Brushes: {brushes.Count}");
                    foreach (var b in brushes)
                    {
                        mapContent.Append(b);
                    }
                    Console.WriteLine($"[INFO] Generated {brushes.Count} brushes from BSP tree for {group.Name}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] BSP Walker failed for {group.Name}: {ex.Message}. Falling back to mesh.");
                }
            }

            // Fallback / Exterior Logic: Triangle Mesh
            mapContent.AppendLine("// Meshed geometry (Exterior or Fallback)");
            GenerateBrushesFromGeometry(mapContent, bspFile, defaultTexture, context);
        }

        private void GenerateBrushesFromGeometry(StringBuilder mapContent, BspFile bspFile, string defaultTexture, MapContext context)
        {
            mapContent.AppendLine($"// Total faces: {bspFile.Faces.Count}, Total vertices: {bspFile.Vertices.Count}");
            
            int brushCount = 0;
            int skippedCount = 0;
            foreach (var face in bspFile.Faces)
            {
                // Use explicit vertex indices (Vertex0/1/2) for the triangle
                if (face.Vertex0 >= bspFile.Vertices.Count ||
                    face.Vertex1 >= bspFile.Vertices.Count ||
                    face.Vertex2 >= bspFile.Vertices.Count)
                {
                    skippedCount++;
                    continue;
                }
                
                var v0 = bspFile.Vertices[face.Vertex0].Position;
                var v1 = bspFile.Vertices[face.Vertex1].Position;
                var v2 = bspFile.Vertices[face.Vertex2].Position;
                
                // Cull degenerate triangles
                var e1 = v1 - v0;
                var e2 = v2 - v0;
                var n = Vector3.Cross(e1, e2);
                if (n.Length() < 1e-6f)
                {
                    skippedCount++;
                    continue;
                }

                // Create brush from actual triangle geometry
                var faceTex = (face.Texture >= 0 && face.Texture < bspFile.Textures.Count)
                    ? bspFile.Textures[face.Texture].Name
                    : defaultTexture;
                var brush = GenerateTriangleBrushFromGeometry(v0, v1, v2, faceTex, context.GeometryOffset);
                if (string.IsNullOrEmpty(brush))
                    continue;

                mapContent.Append(brush);
                
                brushCount++;
                
                if (brushCount % 10 == 0)
                {
                    mapContent.AppendLine();
                }
            }
            
            mapContent.AppendLine();
            Console.WriteLine($"[DEBUG] Generated {brushCount} geometry brushes from Mesh data");
            if (skippedCount > 0)
                Console.WriteLine($"[DEBUG] Skipped {skippedCount} degenerate triangles");
        }

        /// <summary>
        /// Generate brushes from a WMO group using BSP tree reconstruction for interiors
        /// or triangle mesh fallback for exteriors.
        /// </summary>
        private void GenerateBrushesFromWmoGroup(
            StringBuilder mapContent, 
            WmoV14Parser.WmoGroupData group, 
            WmoV14Parser.WmoV14Data wmoData,
            MapContext context)
        {
            // Check if this is an interior group with BSP data
            bool isExterior = (group.Flags & (0x8 | 0x2000)) != 0;
            bool hasBsp = group.BspNodes.Count > 0;
            bool useBsp = !isExterior && hasBsp;
            
            Console.WriteLine($"[BSP] Group '{group.Name}' Flags=0x{group.Flags:X} Exterior={isExterior} BspNodes={group.BspNodes.Count} -> {(useBsp ? "BSP RECONSTRUCTION" : "MESH FALLBACK")}");

            if (useBsp)
            {
                try
                {
                    var walker = new BspTreeWalker();
                    var brushes = walker.GenerateBrushes(group, context.GeometryOffset, wmoData.Materials, wmoData.Textures);
                    
                    mapContent.AppendLine($"// BSP Reconstructed Brushes: {brushes.Count} from {group.Name}");
                    foreach (var brush in brushes)
                    {
                        mapContent.Append(brush);
                    }
                    Console.WriteLine($"[BSP] Generated {brushes.Count} brushes via BSP tree for '{group.Name}'");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] BSP reconstruction failed for '{group.Name}': {ex.Message}. Falling back to mesh.");
                }
            }

            // Fallback: Triangle mesh (for exteriors or BSP failure)
            mapContent.AppendLine($"// Mesh Fallback Brushes for {group.Name}");
            int brushCount = 0;
            for (int i = 0; i + 2 < group.Indices.Count; i += 3)
            {
                var i0 = group.Indices[i];
                var i1 = group.Indices[i + 1];
                var i2 = group.Indices[i + 2];
                
                if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                    continue;
                
                var v0 = group.Vertices[i0];
                var v1 = group.Vertices[i1];
                var v2 = group.Vertices[i2];
                
                // Cull degenerates
                var e1 = v1 - v0;
                var e2 = v2 - v0;
                if (Vector3.Cross(e1, e2).Length() < 1e-6f) continue;
                
                string tex = CaulkTexture;
                int faceIdx = i / 3;
                if (faceIdx < group.FaceMaterials.Count)
                {
                    int matId = group.FaceMaterials[faceIdx];
                    if (matId < wmoData.Textures.Count)
                        tex = wmoData.Textures[matId];
                }
                
                var brush = GenerateTriangleBrushFromGeometry(v0, v1, v2, tex, context.GeometryOffset);
                if (!string.IsNullOrEmpty(brush))
                {
                    mapContent.Append(brush);
                    brushCount++;
                }
            }
            Console.WriteLine($"[MESH] Generated {brushCount} triangle brushes for '{group.Name}'");
        }


        private const string CaulkTexture = "common/caulk";
        private const float TriangleThickness = 2.0f; // Reduced thickness to preserve detail and prevent spikes
        private static readonly string BR = ""; 

        private string GenerateTriangleBrushFromGeometry(Vector3 v0, Vector3 v1, Vector3 v2, string textureName, Vector3 geometryOffset)
        {
            var tex = string.IsNullOrWhiteSpace(textureName) ? CaulkTexture : textureName;

            // Transform to Q3 coordinates and translate into sealed room space
            // Snap to 3 decimal places to avoid micro-precision errors in Q3Map2
            var q0 = SnapVector(TransformToQ3(v0) - geometryOffset);
            var q1 = SnapVector(TransformToQ3(v1) - geometryOffset);
            var q2 = SnapVector(TransformToQ3(v2) - geometryOffset);

            // Skip degenerate triangles (area too small)
            var edge1 = q1 - q0;
            var edge2 = q2 - q0;
            var normal = Vector3.Cross(edge1, edge2);
            var normalLength = normal.Length();
            
            // Q3Map2 epsilon is around 0.1, let's be safe with 0.5 area check
            if (normalLength < 0.5f) 
            {
                return string.Empty;
            }
            normal /= normalLength;

            var halfThickness = TriangleThickness * 0.5f;
            var offset = normal * halfThickness;

            // Create a prism by expanding along normal
            var top0 = q0 + offset;
            var top1 = q1 + offset;
            var top2 = q2 + offset;
            var bottom0 = q0 - offset;
            var bottom1 = q1 - offset;
            var bottom2 = q2 - offset;

            var interiorPoint = (top0 + top1 + top2 + bottom0 + bottom1 + bottom2) / 6f;
            var caulk = CaulkTexture;

            // Pre-validate all 5 planes before writing anything
            var planes = new (Vector3 p0, Vector3 p1, Vector3 p2, string texture)[]
            {
                (top0, top1, top2, tex),                    // Top face (textured)
                (bottom0, bottom2, bottom1, caulk),         // Bottom face
                (top1, top0, bottom0, caulk),               // Edge 0-1
                (top2, top1, bottom1, caulk),               // Edge 1-2
                (top0, top2, bottom2, caulk),               // Edge 2-0
            };

            var brush = new StringBuilder();
            brush.AppendLine("{");

            foreach (var (p0, p1, p2, planeTex) in planes)
            {
                // Verify plane validity - Stricter check for "no axis found" errors
                var pNormal = Vector3.Cross(p1 - p0, p2 - p0);
                if (pNormal.LengthSquared() < 0.1f)
                {
                    // Degenerate plane found - abort this brush.
                    // This happens when points are collinear or too close.
                    return string.Empty;
                }
                
                // Ensure proper winding (pointing away from interior)
                // If dot(normal, (interior - p0)) > 0, normal points IN. Flip it.
                // We re-calculate correct point order
                var finalP1 = p1;
                var finalP2 = p2;
                if (Vector3.Dot(pNormal, interiorPoint - p0) > 0)
                {
                    finalP1 = p2;
                    finalP2 = p1;
                }

                WritePlaneLine(brush, p0, finalP1, finalP2, planeTex);
            }

            brush.AppendLine("}");
            return brush.ToString();
        }

        private Vector3 SnapVector(Vector3 v)
        {
            return new Vector3(
                (float)Math.Round(v.X, 3),
                (float)Math.Round(v.Y, 3),
                (float)Math.Round(v.Z, 3)
            );
        }



        private void WritePlaneLine(StringBuilder brush, Vector3 p0, Vector3 p1, Vector3 p2, string texture)
        {
            // Quake 3 plane format: ( x y z ) ( x y z ) ( x y z ) TEXTURE offsetX offsetY rotation scaleX scaleY contentFlags surfaceFlags value
            // Must have 8 parameters after texture name (not 5!)
            // FORCE InvariantCulture to ensure '.' is used for decimals
            string line = FormattableString.Invariant($"  ( {p0.X:F6} {p0.Y:F6} {p0.Z:F6} ) ( {p1.X:F6} {p1.Y:F6} {p1.Z:F6} ) ( {p2.X:F6} {p2.Y:F6} {p2.Z:F6} ) {texture} 0 0 0 0.5 0.5 0 0 0");
            brush.AppendLine(line);
        }

        private void WriteBrushPlane(StringBuilder brush, Vector3 p0, Vector3 p1, Vector3 p2, string texture, Vector3 interiorPoint)
        {
            var normal = Vector3.Cross(p1 - p0, p2 - p0);
            if (normal.LengthSquared() < 1e-6f)
            {
                return;
            }

            if (Vector3.Dot(normal, interiorPoint - p0) > 0f)
            {
                (p1, p2) = (p2, p1);
            }

            WritePlaneLine(brush, p0, p1, p2, texture);
        }

        private void AddSpawnEntity(StringBuilder mapContent, MapContext context, float floorZ)
        {
            var bounds = context.Bounds;
            var center = bounds.Center - context.GeometryOffset;
            center.Z = floorZ + 16.0f; // place spawn slightly above sealed floor

            mapContent.AppendLine("// Default spawn");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"info_player_deathmatch\"");
            mapContent.AppendLine($"\"origin\" \"{center.X:F1} {center.Y:F1} {center.Z:F1}\"");
            mapContent.AppendLine("\"angle\" \"0\"");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
        }

        private static IReadOnlyList<string> BuildMaterialShaderNames(WmoV14Parser.WmoV14Data wmoData)
        {
            var names = new List<string>();
            if (wmoData.Materials.Count > 0 && wmoData.MaterialTextureIndices.Count > 0)
            {
                for (int i = 0; i < wmoData.Materials.Count; i++)
                {
                    var texIdx = i < wmoData.MaterialTextureIndices.Count ? (int)wmoData.MaterialTextureIndices[i] : -1;
                    names.Add(BuildShaderNameForTexture(wmoData, texIdx));
                }
            }
            else if (wmoData.Textures.Count > 0)
            {
                for (int i = 0; i < wmoData.Textures.Count; i++)
                {
                    names.Add(BuildShaderNameForTexture(wmoData, i));
                }
            }

            if (names.Count == 0)
            {
                names.Add("textures/wmo/wmo_default");
            }

            return names;
        }

        private static string BuildShaderNameForTexture(WmoV14Parser.WmoV14Data wmoData, int textureIndex)
        {
            if (textureIndex >= 0 && textureIndex < wmoData.Textures.Count)
            {
                var tex = wmoData.Textures[textureIndex];
                var baseName = Path.GetFileNameWithoutExtension(tex).ToLowerInvariant();
                return $"textures/wmo/{baseName}.tga";
            }

            return "textures/wmo/wmo_default.tga";
        }

        private void GenerateTextureInfo(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData)
        {
            mapContent.AppendLine("// Texture information");
            mapContent.AppendLine("// Available textures from WMO file:");
            
            for (int i = 0; i < wmoData.Textures.Count && i < 10; i++)
            {
                mapContent.AppendLine($"// {i}: {wmoData.Textures[i]}");
            }
            
            if (wmoData.Textures.Count > 10)
            {
                mapContent.AppendLine($"// ... and {wmoData.Textures.Count - 10} more textures");
            }
            
            mapContent.AppendLine();
        }

        /// <summary>
        /// Alternative: Generate a simple cube as a test map
        /// </summary>
        public void GenerateSimpleTestMap(string outputPath)
        {
            var mapContent = new StringBuilder();
            
            mapContent.AppendLine("// Simple test cube map");
            mapContent.AppendLine("// Generated by WMO v14 to Quake 3 converter");
            mapContent.AppendLine();
            
            // Worldspawn
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"worldspawn\"");
            mapContent.AppendLine("\"message\" \"Test map from WMO v14 converter\"");
            mapContent.AppendLine("}");
            mapContent.AppendLine();
            
            // Simple cube brush
            mapContent.AppendLine("// Test cube brush");
            mapContent.AppendLine("{");
            mapContent.AppendLine("  ( -64 0 0 ) ( 0 -64 0 ) ( 0 0 -64 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 0 -64 0 ) ( 0 0 128 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 0 0 128 ) ( 64 0 0 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 64 0 0 ) ( 0 64 0 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 0 64 0 ) ( 0 0 -64 ) NULL 0 0 0");
            mapContent.AppendLine("  ( 0 0 0 ) ( 64 0 0 ) ( 0 0 128 ) NULL 0 0 0");
            mapContent.AppendLine("}");
            
            // Add a light entity
            mapContent.AppendLine();
            mapContent.AppendLine("// Test light");
            mapContent.AppendLine("{");
            mapContent.AppendLine("\"classname\" \"light\"");
            mapContent.AppendLine("\"origin\" \"0 0 32\"");
            mapContent.AppendLine("\"light\" \"300\"");
            mapContent.AppendLine("}");
            
            File.WriteAllText(outputPath, mapContent.ToString());
            
            Console.WriteLine($"[INFO] Generated simple test .map file: {outputPath}");
        }

        private void GeneratePortalBrushes(StringBuilder mapContent, WmoV14Parser.WmoV14Data wmoData, MapContext context)
        {
            if (wmoData.Portals.Count == 0 || wmoData.PortalVertices.Count == 0)
                return;

            mapContent.AppendLine("// Portal Hint Brushes");
            mapContent.AppendLine($"// Generated {wmoData.Portals.Count} portals from {wmoData.PortalVertices.Count} vertices");

            int pIdx = 0;
            foreach (var portal in wmoData.Portals)
            {
                if (portal.VertexCount < 3) continue;

                var poly = new List<Vector3>();
                for (int i = 0; i < portal.VertexCount; i++)
                {
                    int vIdx = portal.FirstVertex + i;
                    if (vIdx < wmoData.PortalVertices.Count)
                    {
                         poly.Add(wmoData.PortalVertices[vIdx]);
                    }
                }

                if (poly.Count < 3) continue;

                // Transform to Q3 and offset
                for(int i=0; i<poly.Count; i++) 
                {
                    poly[i] = TransformToQ3(poly[i]) - context.GeometryOffset;
                }

                // Generate Brush
                // Use common/hint for the portal face and common/skip for the rest
                string brush = GenerateBrushFromPolygon(poly, "common/hint", "common/skip");
                mapContent.Append(brush);
                pIdx++;
            }
        }

        private string GenerateBrushFromPolygon(List<Vector3> points, string frontTex, string otherTex)
        {
            // Compute normal from first 3 points (assuming planar)
            var v0 = points[0];
            var v1 = points[1];
            var v2 = points[2];
            var normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
            
            // Thickness
            float thickness = 4.0f; // Thin brush
            var offset = normal * (thickness * 0.5f);
            
            var brush = new StringBuilder();
            brush.AppendLine("{");
            brush.AppendLine($"// Hint Brush (Poly: {points.Count} verts)");
            
            // Front Face (Original polygon + offset)
            // Winding: v0, v1, v2 is CCW looking from front.
            // Pushed forward by offset -> Still v0, v1, v2
            var f0 = v0 + offset;
            var f1 = v1 + offset;
            var f2 = v2 + offset;
            WritePlaneLine(brush, f0, f1, f2, frontTex);

            // Back Face (Original polygon - offset)
            // Pushed back. To face away, we reverse winding: v0, v2, v1
            var b0 = v0 - offset;
            var b1 = v1 - offset;
            var b2 = v2 - offset;
            WritePlaneLine(brush, b0, b2, b1, otherTex);

            // Side Faces
            for (int i = 0; i < points.Count; i++)
            {
                var pA = points[i];
                var pB = points[(i + 1) % points.Count];
                
                var topA = pA + offset;
                var topB = pB + offset;
                var botA = pA - offset;
                var botB = pB - offset;
                
                // Side plane: topB, topA, botA (CCW looking from outside)
                WritePlaneLine(brush, topB, topA, botA, otherTex);
            }

            brush.AppendLine("}");
            return brush.ToString();
        }
            public void GenerateClusteredMapWithASE(
            string baseOutputPath,
            WmoV14Parser.WmoV14Data wmoData,
            BspFile fullBspFile,
            string outputRootDir,
            string wmoName)
        {
            // 1. Export ASE models for ALL groups
            Console.WriteLine("[INFO] Exporting ASE models for all groups...");
            var aseExporter = new WmoAseExporter(); // Assumes WmoAseExporter is in same namespace
            // Deduce source dir from somewhere? Or use current.
            // Texture conversion happens here.
            aseExporter.ExportGroupsToAse(outputRootDir, wmoName, wmoData, ".", false, false);

            // 2. Compute Clusters
            // Combine all interiors into large clusters (High Limits)
            var clusters = ComputeClusters(wmoData, 20000000, 500.0f);
            
            // 3. Generate Map Files
            int nonEmptyClusters = clusters.Count(c => c.Count > 0);


            for (int i = 0; i < clusters.Count; i++)
            {
                var cluster = clusters[i];
                if (cluster.Count == 0) continue;

                string suffix;
                if (nonEmptyClusters == 1)
                {
                    suffix = ""; // Single file, no suffix
                }
                else
                {
                    suffix = (i == 0) ? "_Exterior" : $"_Interior_C{i:D2}";
                }
                
                var dir = Path.GetDirectoryName(baseOutputPath);
                var name = Path.GetFileNameWithoutExtension(baseOutputPath);
                if (name.EndsWith(".map", StringComparison.OrdinalIgnoreCase)) 
                    name = Path.GetFileNameWithoutExtension(name);
                
                var mapPath = Path.Combine(string.IsNullOrEmpty(dir) ? "" : dir, $"{name}{suffix}.map");
                
                Console.WriteLine($"[INFO] Generatign Clustered Map (ASE): {mapPath} ({cluster.Count} groups)");

                using var writer = new StreamWriter(mapPath, false, Encoding.ASCII);
                writer.WriteLine($"// ASE-based Clustered Map: {suffix}");
                writer.WriteLine("{");
                writer.WriteLine("\"classname\" \"worldspawn\"");
                writer.WriteLine("\"message\" \"Converted by WmoBspConverter (ASE Mode)\"");

                // Calculate bounds for this cluster (using ASE Transform)
                var min = new Vector3(float.MaxValue);
                var max = new Vector3(float.MinValue);
                
                foreach (var gIdx in cluster)
                {
                    var g = wmoData.Groups[gIdx];
                    foreach (var v in g.Vertices)
                    {
                        // ASE Transform: Y, -X, Z
                        // (Matches WmoAseExporter.WriteGeomObject logic)
                        var tv = new Vector3(v.Y, -v.X, v.Z);
                        min = Vector3.Min(min, tv);
                        max = Vector3.Max(max, tv);
                    }
                }
                
                // Add padding for outer walls/ceiling
                var geoMin = min; // Save actual geometry bounds
                var geoMax = max;
                
                Console.WriteLine($"[DEBUG] Geometry bounds: min={geoMin}, max={geoMax}");
                Console.WriteLine($"[DEBUG] Floor will be at Z={geoMin.Z - 16}");
                
                min -= new Vector3(512);
                max += new Vector3(512);
                
                // Write Sealed Room with floor at geometry level
                string caulk = "common/caulk";
                // Floor sits just below geometry, not 512 units down
                WritePlaneBrushWithFloorAt(writer, min, max, geoMin.Z - 16, caulk);
                
                writer.WriteLine("}"); // Close worldspawn
                
                // Add misc_models (Using Relative Paths to ensure VFS compatibility)
                foreach (var gIdx in cluster)
                {
                    // Relative path works best if the user sets Engine Path or if Radiant is flexible
                    // Absolute paths (j:/...) failed to load in GtkRadiant
                    string modelRelPath = $"models/wmo/{wmoName}_group{gIdx:D3}.ase";
                    
                    writer.WriteLine("{");
                    writer.WriteLine("\"classname\" \"misc_model\"");
                    writer.WriteLine($"\"model\" \"{modelRelPath}\"");
                    writer.WriteLine("\"origin\" \"0 0 0\"");
                    writer.WriteLine("\"spawnflags\" \"2\""); // 2 = solid, bake into BSP
                    writer.WriteLine("}");
                }
                
                // Add Player Spawn (use geometry bounds, not padded)
                writer.WriteLine("{");
                writer.WriteLine("\"classname\" \"info_player_deathmatch\"");
                // Spawn slightly above the floor (geoMin.Z)
                writer.WriteLine($"\"origin\" \"{geoMin.X} {geoMin.Y} {geoMin.Z + 48}\"");
                writer.WriteLine("}");
            }
        }

        private void WritePlaneBrush(StreamWriter w, Vector3 min, Vector3 max, string texture)
        {
             // Simple Box Helper
             // 6 Brushes... or 1 Brush? 
             // "Sealed Room" usually means 1 Hollow Box (6 brushes).
             float t = 16f; // thickness
             
             // Floor
             WriteSimpleBrushContent(w, min.X, min.Y, min.Z, max.X, max.Y, min.Z+t, texture); 
             // Ceiling
             WriteSimpleBrushContent(w, min.X, min.Y, max.Z-t, max.X, max.Y, max.Z, texture);
             // Walls
             WriteSimpleBrushContent(w, min.X, min.Y, min.Z, min.X+t, max.Y, max.Z, texture); // -X
             WriteSimpleBrushContent(w, max.X-t, min.Y, min.Z, max.X, max.Y, max.Z, texture); // +X
             WriteSimpleBrushContent(w, min.X, min.Y, min.Z, max.X, min.Y+t, max.Z, texture); // -Y
             WriteSimpleBrushContent(w, min.X, max.Y-t, min.Z, max.X, max.Y, max.Z, texture); // +Y
        }

        private void WritePlaneBrushWithFloorAt(StreamWriter w, Vector3 min, Vector3 max, float floorZ, string texture)
        {
             // Sealed room with floor at specific Z level
             float t = 16f; // thickness
             
             // Floor at geometry level (not padded min)
             WriteSimpleBrushContent(w, min.X, min.Y, floorZ, max.X, max.Y, floorZ+t, texture); 
             // Ceiling
             WriteSimpleBrushContent(w, min.X, min.Y, max.Z-t, max.X, max.Y, max.Z, texture);
             // Walls (extend from floor to ceiling)
             WriteSimpleBrushContent(w, min.X, min.Y, floorZ, min.X+t, max.Y, max.Z, texture); // -X
             WriteSimpleBrushContent(w, max.X-t, min.Y, floorZ, max.X, max.Y, max.Z, texture); // +X
             WriteSimpleBrushContent(w, min.X, min.Y, floorZ, max.X, min.Y+t, max.Z, texture); // -Y
             WriteSimpleBrushContent(w, min.X, max.Y-t, floorZ, max.X, max.Y, max.Z, texture); // +Y
        }
        
        private void WriteSimpleBrushContent(StreamWriter w, float x1, float y1, float z1, float x2, float y2, float z2, string texture)
        {
            // Round to integers - q3map2 works best with integer coordinates
            int ix1 = (int)Math.Round(x1), iy1 = (int)Math.Round(y1), iz1 = (int)Math.Round(z1);
            int ix2 = (int)Math.Round(x2), iy2 = (int)Math.Round(y2), iz2 = (int)Math.Round(z2);
            
            w.WriteLine("{");
            w.WriteLine($"( {ix2} {iy1} {iz1} ) ( {ix2} {iy2} {iz1} ) ( {ix2} {iy1} {iz2} ) {texture} 0 0 0 1 1"); // +X
            w.WriteLine($"( {ix1} {iy1} {iz1} ) ( {ix1} {iy1} {iz2} ) ( {ix1} {iy2} {iz1} ) {texture} 0 0 0 1 1"); // -X
            w.WriteLine($"( {ix1} {iy2} {iz1} ) ( {ix1} {iy2} {iz2} ) ( {ix2} {iy2} {iz1} ) {texture} 0 0 0 1 1"); // +Y
            w.WriteLine($"( {ix1} {iy1} {iz1} ) ( {ix2} {iy1} {iz1} ) ( {ix1} {iy1} {iz2} ) {texture} 0 0 0 1 1"); // -Y
            w.WriteLine($"( {ix1} {iy1} {iz2} ) ( {ix2} {iy1} {iz2} ) ( {ix1} {iy2} {iz2} ) {texture} 0 0 0 1 1"); // +Z
            w.WriteLine($"( {ix1} {iy1} {iz1} ) ( {ix1} {iy2} {iz1} ) ( {ix2} {iy1} {iz1} ) {texture} 0 0 0 1 1"); // -Z
            w.WriteLine("}");
        }
    }
}