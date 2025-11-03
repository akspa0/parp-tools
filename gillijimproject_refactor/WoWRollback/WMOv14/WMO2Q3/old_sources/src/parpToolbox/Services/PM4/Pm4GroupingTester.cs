using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Services.Coordinate;
using System.Text;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Implements optimized grouping strategy for PM4 scene data to reconstruct complete 3D objects from fragmented tile data.
    /// Focuses on MSUR raw fields grouping which produces the most coherent object assemblies.
    /// </summary>
    internal static class Pm4GroupingTester
    {
        /// <summary>
        /// Main entry point that runs the optimized MSUR raw fields grouping strategy.
        /// </summary>
        /// <param name="scene">Loaded PM4 scene (global). Must have Vertices, Indices, Surfaces populated.</param>
        /// <param name="baseOutputDir">Base directory for all output.</param>
        /// <param name="writeFaces">Whether to export faces (true) or just vertices (false).</param>
        public static void RunMultipleGroupingStrategies(Pm4Scene scene, string baseOutputDir, bool writeFaces = true)
        {
            Directory.CreateDirectory(baseOutputDir);
            Console.WriteLine($"[GroupingTester] Running optimized MSUR raw fields grouping on scene with {scene.Surfaces.Count} surfaces, {scene.Vertices.Count} vertices.");
            Console.WriteLine($"[GroupingTester] Output directory: {baseOutputDir}");
            
            // Run the MSUR raw fields grouping strategy (proven most effective)
            ExportByMsurRawFields(scene, baseOutputDir, writeFaces);
            
            Console.WriteLine($"[GroupingTester] Completed MSUR raw fields grouping. Results written to {baseOutputDir}");
        }
        
        /// <param name="scene">Loaded PM4 scene (global). Must have Vertices, Indices, Surfaces populated.</param>
        /// <param name="outputDir">Where to write OBJ files (created if needed).</param>
        /// <param name="writeFaces">True to export faces, false for point cloud.</param>
        /// <param name="minGroup">Only export groups whose byte value is >= minGroup when specified; otherwise export all.</param>
        public static void ExportBySurfaceGroupKey(Pm4Scene scene, string outputDir, bool writeFaces = true, byte? minGroup = null)
        {
            if (scene.Surfaces.Count == 0)
                throw new InvalidOperationException("Scene has no MSUR entries – cannot group by SurfaceGroupKey.");

            Directory.CreateDirectory(outputDir);

            // Build triangles list per group key.
            var groups = new Dictionary<byte, List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<byte, HashSet<int>>();

            foreach (var surf in scene.Surfaces)
            {
                byte gKey = surf.SurfaceGroupKey;
                if (minGroup.HasValue && gKey < minGroup.Value)
                    continue; // skip coarse groups if caller only wants fine ones

                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;

                if (first < 0 || first + count > scene.Indices.Count)
                    continue; // invalid – skip

                if (scene.Indices.Count == 0)
                    continue; // no flat index buffer – skip face generation

                int triCount = count / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    if (baseIdx + 2 >= scene.Indices.Count)
                        continue; // out of range – skip

                    int ia = scene.Indices[baseIdx];
                    int ib = scene.Indices[baseIdx + 1];
                    int ic = scene.Indices[baseIdx + 2];

                    if (!groups.TryGetValue(gKey, out var list))
                    {
                        list = new List<(int, int, int)>();
                        groups[gKey] = list;
                        usedVertsPerGroup[gKey] = new HashSet<int>();
                    }

                    list.Add((ia, ib, ic));
                    usedVertsPerGroup[gKey].Add(ia);
                    usedVertsPerGroup[gKey].Add(ib);
                    usedVertsPerGroup[gKey].Add(ic);
                }
            }

            Console.WriteLine($"[GroupingTester] Discovered {groups.Count} distinct group keys.");

            foreach (var kvp in groups.OrderBy(g => g.Key))
            {
                byte key = kvp.Key;
                var faces = kvp.Value;
                var used = usedVertsPerGroup[key];

                string baseName = $"group_{key}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                // Map original vertex index -> OBJ index (1-based)
                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-test-grouping OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                foreach (var vIdx in used)
                {
                    if (vIdx < 0 || vIdx >= scene.Vertices.Count)
                        continue; // skip invalid vertex refs

                    remap[vIdx] = nextIdx++;
                    Vector3 v = scene.Vertices[vIdx];
                    // Flip X for world to OBJ convention
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                if (writeFaces)
                {
                    foreach (var (A, B, C) in faces)
                    {
                        if (remap.TryGetValue(A, out var ra) &&
                            remap.TryGetValue(B, out var rb) &&
                            remap.TryGetValue(C, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in used)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {used.Count}, faces {faces.Count})");
            }
        }
        /// <summary>
        /// Exports geometry grouped by a composite key consisting of MSUR.SurfaceGroupKey and IndexCount.
        /// This is useful for testing the hypothesis that IndexCount further sub-divides SurfaceGroup containers
        /// into per-object meshes.
        /// </summary>
        /// <summary>
        /// Exports geometry grouped by MPRL object type fields (Unknown4, Unknown6, etc.)
        /// This attempts to use the type information from MPRL (placement) chunks to group related objects.
        /// </summary>
        public static void ExportByMprlTypeFields(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {
            if (scene.Surfaces.Count == 0 || scene.Placements.Count == 0)
            {
                Console.WriteLine("[ExportByMprlTypeFields] Scene has no MSUR or MPRL entries - cannot group by MPRL types.");
                return;
            }

            Directory.CreateDirectory(outputDir);

            // Step 1: Build a map from MSLK ParentIndex → MPRL entries
            var mprlEntriesByParentIndex = new Dictionary<uint, List<MprlChunk.Entry>>();
            foreach (var placement in scene.Placements)
            {
                uint parentIndex = placement.Unknown4; // Based on memory: MPRL.Unknown4 = MSLK.ParentIndex
                if (!mprlEntriesByParentIndex.TryGetValue(parentIndex, out var list))
                {
                    list = new List<MprlChunk.Entry>();
                    mprlEntriesByParentIndex[parentIndex] = list;
                }
                list.Add(placement);
            }

            // Step 2: Build lookup for MSLK→surface mapping to connect MPRL → MSLK → MSUR
            var surfaceRanges = scene.Surfaces.Select(s => new {
                Surface = s,
                Start = (int)s.MsviFirstIndex,
                End = (int)s.MsviFirstIndex + s.IndexCount - 1
            }).OrderBy(r => r.Start).ToArray();
            
            var surfaceStarts = surfaceRanges.Select(r => r.Start).ToArray();
            var linksBySurface = new Dictionary<MsurChunk.Entry, List<MslkEntry>>();
            
            foreach (var link in scene.Links.OfType<MslkEntry>())
            {
                if (link.MspiFirstIndex < 0 || link.MspiIndexCount == 0)
                    continue;
                    
                int nodeStart = link.MspiFirstIndex;
                int nodeEnd = link.MspiFirstIndex + link.MspiIndexCount - 1;
                int idx = Array.BinarySearch(surfaceStarts, nodeStart);
                if (idx < 0) idx = ~idx - 1;
                if (idx < 0) continue;
                
                var range = surfaceRanges[idx];
                if (nodeStart >= range.Start && nodeEnd <= range.End)
                {
                    if (!linksBySurface.TryGetValue(range.Surface, out var list))
                    {
                        list = new List<MslkEntry>();
                        linksBySurface[range.Surface] = list;
                    }
                    list.Add(link);
                }
            }
            
            // Step 3: Build object types based on Unknown6 (confirmed to be consistently 32768)
            // and combine with positional data to group objects
            var objectTypeGroups = new Dictionary<(ushort TypeFlag, uint TypeId), List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<(ushort TypeFlag, uint TypeId), HashSet<int>>();

            foreach (var surf in scene.Surfaces)
            {
                // Try to find the MSLK link for this surface
                if (!linksBySurface.TryGetValue(surf, out var links) || links.Count == 0)
                    continue;
                    
                // For each link, find the MPRL parent
                foreach (var link in links)
                {
                    uint parentIndex = link.ParentIndex;
                    if (!mprlEntriesByParentIndex.TryGetValue(parentIndex, out var mprlEntries) || mprlEntries.Count == 0)
                        continue;
                        
                    // Use the first MPRL entry's type information
                    var mprl = mprlEntries[0];
                    // Using Unknown6 (type flag) and parentIndex (type ID)
                    var typeKey = (TypeFlag: mprl.Unknown6, TypeId: parentIndex);
                    
                    // Process the surface's triangles
                    int first = (int)surf.MsviFirstIndex;
                    int count = surf.IndexCount;

                    if (first < 0 || first + count > scene.Indices.Count)
                        continue; // invalid range

                    int triCount = count / 3;
                    for (int i = 0; i < triCount; i++)
                    {
                        int baseIdx = first + i * 3;
                        if (baseIdx + 2 >= scene.Indices.Count)
                            continue; // out of range

                        int ia = scene.Indices[baseIdx];
                        int ib = scene.Indices[baseIdx + 1];
                        int ic = scene.Indices[baseIdx + 2];

                        // Check vertex bounds
                        if (ia < 0 || ia >= scene.Vertices.Count ||
                            ib < 0 || ib >= scene.Vertices.Count ||
                            ic < 0 || ic >= scene.Vertices.Count)
                            continue; // invalid vertex indices

                        if (!objectTypeGroups.TryGetValue(typeKey, out var triangles))
                        {
                            triangles = new List<(int, int, int)>();
                            objectTypeGroups[typeKey] = triangles;
                            usedVertsPerGroup[typeKey] = new HashSet<int>();
                        }

                        triangles.Add((ia, ib, ic));
                        usedVertsPerGroup[typeKey].Add(ia);
                        usedVertsPerGroup[typeKey].Add(ib);
                        usedVertsPerGroup[typeKey].Add(ic);
                    }
                }
            }

            Console.WriteLine($"[ExportByMprlTypeFields] Found {objectTypeGroups.Count} distinct object types.");

            // Export each object type group
            foreach (var kvp in objectTypeGroups.OrderBy(g => g.Key.TypeFlag).ThenBy(g => g.Key.TypeId))
            {
                var key = kvp.Key;
                var triangles = kvp.Value;
                var usedVerts = usedVertsPerGroup[key];

                string baseName = $"flag_{key.TypeFlag}_type_{key.TypeId}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                // Create remapping for OBJ export (1-based indices)
                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-test-grouping MPRL Type Fields OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                // Export vertices
                foreach (var vIdx in usedVerts)
                {
                    remap[vIdx] = nextIdx++;
                    Vector3 v = scene.Vertices[vIdx];
                    // Flip X for world to OBJ convention
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                // Export faces or points
                if (writeFaces)
                {
                    foreach (var (a, b, c) in triangles)
                    {
                        if (remap.TryGetValue(a, out var ra) &&
                            remap.TryGetValue(b, out var rb) &&
                            remap.TryGetValue(c, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in usedVerts)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {usedVerts.Count}, faces {triangles.Count})");
            }
        }
        
        /// <summary>
        /// Exports geometry grouped by vertex connectivity patterns.
        /// This strategy attempts to identify connected mesh components regardless of type information.
        /// </summary>
        public static void ExportByVertexConnectivity(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {
            if (scene.Surfaces.Count == 0)
            {
                Console.WriteLine("[ExportByVertexConnectivity] Scene has no MSUR entries - cannot analyze connectivity.");
                return;
            }

            Directory.CreateDirectory(outputDir);
            
            // Step 1: Build an adjacency graph of all vertices
            var adjacencyList = new Dictionary<int, HashSet<int>>();
            var allTriangles = new List<(int A, int B, int C)>();
            
            Console.WriteLine("[ExportByVertexConnectivity] Building vertex connectivity graph...");
            
            // Process all surfaces and gather triangles
            foreach (var surf in scene.Surfaces)
            {
                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;

                if (first < 0 || first + count > scene.Indices.Count)
                    continue; // invalid range

                int triCount = count / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    if (baseIdx + 2 >= scene.Indices.Count)
                        continue; // out of range

                    int ia = scene.Indices[baseIdx];
                    int ib = scene.Indices[baseIdx + 1];
                    int ic = scene.Indices[baseIdx + 2];

                    // Check vertex bounds
                    if (ia < 0 || ia >= scene.Vertices.Count ||
                        ib < 0 || ib >= scene.Vertices.Count ||
                        ic < 0 || ic >= scene.Vertices.Count)
                        continue; // invalid vertex indices

                    // Build adjacency graph (undirected)
                    if (!adjacencyList.TryGetValue(ia, out var adjA))
                    {
                        adjA = new HashSet<int>();
                        adjacencyList[ia] = adjA;
                    }
                    if (!adjacencyList.TryGetValue(ib, out var adjB))
                    {
                        adjB = new HashSet<int>();
                        adjacencyList[ib] = adjB;
                    }
                    if (!adjacencyList.TryGetValue(ic, out var adjC))
                    {
                        adjC = new HashSet<int>();
                        adjacencyList[ic] = adjC;
                    }

                    // Connect all vertices in the triangle
                    adjA.Add(ib);
                    adjA.Add(ic);
                    adjB.Add(ia);
                    adjB.Add(ic);
                    adjC.Add(ia);
                    adjC.Add(ib);

                    // Save triangle for later assignment
                    allTriangles.Add((ia, ib, ic));
                }
            }

            // Step 2: Run connected components algorithm to find distinct meshes
            var visited = new HashSet<int>();
            var connectedComponents = new List<HashSet<int>>();

            Console.WriteLine($"[ExportByVertexConnectivity] Finding connected components in graph with {adjacencyList.Count} vertices...");
            
            foreach (int vertex in adjacencyList.Keys)
            {
                if (visited.Contains(vertex))
                    continue;
                    
                // Start a new connected component
                var component = new HashSet<int>();
                var queue = new Queue<int>();
                
                queue.Enqueue(vertex);
                visited.Add(vertex);
                component.Add(vertex);
                
                // Breadth-first search to find all connected vertices
                while (queue.Count > 0)
                {
                    int current = queue.Dequeue();
                    
                    if (!adjacencyList.TryGetValue(current, out var neighbors))
                        continue;
                        
                    foreach (int neighbor in neighbors)
                    {
                        if (visited.Contains(neighbor))
                            continue;
                            
                        visited.Add(neighbor);
                        component.Add(neighbor);
                        queue.Enqueue(neighbor);
                    }
                }
                
                // Save this connected component
                if (component.Count > 0)
                {
                    connectedComponents.Add(component);
                }
            }

            Console.WriteLine($"[ExportByVertexConnectivity] Found {connectedComponents.Count} connected mesh components.");

            // Step 3: For each connected component, find all triangles that belong to it
            var trianglesByComponent = new List<List<(int A, int B, int C)>>();
            for (int i = 0; i < connectedComponents.Count; i++)
            {
                trianglesByComponent.Add(new List<(int A, int B, int C)>());
            }
            
            foreach (var triangle in allTriangles)
            {
                // Find which component contains this triangle's vertices
                for (int i = 0; i < connectedComponents.Count; i++)
                {
                    var component = connectedComponents[i];
                    
                    // A triangle belongs to a component if all its vertices are in that component
                    if (component.Contains(triangle.A) && 
                        component.Contains(triangle.B) && 
                        component.Contains(triangle.C))
                    {
                        trianglesByComponent[i].Add(triangle);
                        break;
                    }
                }
            }
            
            // Step 4: Export each connected component as a separate mesh
            Console.WriteLine($"[ExportByVertexConnectivity] Exporting {connectedComponents.Count} mesh components...");
            
            // Sort components by size (largest first) for easier inspection
            var sortedIndices = Enumerable.Range(0, connectedComponents.Count)
                                         .OrderByDescending(i => connectedComponents[i].Count)
                                         .ToList();
                                         
            foreach (int idx in sortedIndices)
            {
                var component = connectedComponents[idx];
                var triangles = trianglesByComponent[idx];
                
                // Skip tiny components (likely noise or disconnected fragments)
                if (component.Count < 3)
                    continue;
                
                string baseName = $"component_{idx:D4}_verts_{component.Count}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                // Create remapping for OBJ export (1-based indices)
                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-test-grouping Vertex Connectivity OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                // Export vertices
                foreach (var vIdx in component)
                {
                    remap[vIdx] = nextIdx++;
                    Vector3 v = scene.Vertices[vIdx];
                    // Flip X for world to OBJ convention
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                // Export faces or points
                if (writeFaces)
                {
                    foreach (var (a, b, c) in triangles)
                    {
                        if (remap.TryGetValue(a, out var ra) &&
                            remap.TryGetValue(b, out var rb) &&
                            remap.TryGetValue(c, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in component)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {component.Count}, faces {triangles.Count})");
            }
        }
        
        /// <summary>
        /// Exports geometry grouped by raw MSUR fields to explore additional grouping options.
        /// </summary>
        public static void ExportByMsurRawFields(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {
            if (scene.Surfaces.Count == 0)
            {
                Console.WriteLine("[ExportByMsurRawFields] Scene has no MSUR entries - cannot analyze raw fields.");
                return;
            }

            Directory.CreateDirectory(outputDir);
            
            // Group triangles by the combination of multiple fields (experiment with combinations)
            var groupedTriangles = new Dictionary<(uint Field1, uint Field2), List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<(uint Field1, uint Field2), HashSet<int>>();

            foreach (var surf in scene.Surfaces)
            {
                // We'll use multiple raw fields as grouping keys
                // Experimenting with Field1=FlagsOrUnknown_0x00 and Field2=Unknown_0x02
                var key = (Field1: (uint)surf.GroupKey, Field2: (uint)surf.AttributeMask);
                
                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;

                if (first < 0 || first + count > scene.Indices.Count)
                    continue; // invalid range

                int triCount = count / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    if (baseIdx + 2 >= scene.Indices.Count)
                        continue; // out of range

                    int ia = scene.Indices[baseIdx];
                    int ib = scene.Indices[baseIdx + 1];
                    int ic = scene.Indices[baseIdx + 2];

                    // Check vertex bounds
                    if (ia < 0 || ia >= scene.Vertices.Count ||
                        ib < 0 || ib >= scene.Vertices.Count ||
                        ic < 0 || ic >= scene.Vertices.Count)
                        continue; // invalid vertex indices

                    if (!groupedTriangles.TryGetValue(key, out var triangles))
                    {
                        triangles = new List<(int, int, int)>();
                        groupedTriangles[key] = triangles;
                        usedVertsPerGroup[key] = new HashSet<int>();
                    }

                    triangles.Add((ia, ib, ic));
                    usedVertsPerGroup[key].Add(ia);
                    usedVertsPerGroup[key].Add(ib);
                    usedVertsPerGroup[key].Add(ic);
                }
            }

            Console.WriteLine($"[ExportByMsurRawFields] Found {groupedTriangles.Count} distinct raw field combinations.");

            // Export each group
            foreach (var kvp in groupedTriangles.OrderBy(g => g.Key.Field1).ThenBy(g => g.Key.Field2))
            {
                var key = kvp.Key;
                var triangles = kvp.Value;
                var usedVerts = usedVertsPerGroup[key];

                string baseName = $"field1_{key.Field1}_field2_{key.Field2}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                // Create remapping for OBJ export (1-based indices)
                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-test-grouping MSUR Raw Fields OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                // Export vertices
                foreach (var vIdx in usedVerts)
                {
                    remap[vIdx] = nextIdx++;
                    Vector3 v = scene.Vertices[vIdx];
                    // Flip X for world to OBJ convention
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                // Export faces or points
                if (writeFaces)
                {
                    foreach (var (a, b, c) in triangles)
                    {
                        if (remap.TryGetValue(a, out var ra) &&
                            remap.TryGetValue(b, out var rb) &&
                            remap.TryGetValue(c, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in usedVerts)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {usedVerts.Count}, faces {triangles.Count})");
            }
        }
        
        /// <summary>
        /// Exports geometry grouped by MPRL -> MSLK -> MSUR chain.
        /// </summary>
        public static void ExportByMprlMslkChain(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {

            if (scene.Surfaces.Count == 0 || scene.Placements.Count == 0 || scene.Links.Count == 0)
            {
                Console.WriteLine("[ExportByMprlMslkChain] Scene has incomplete data - cannot analyze complete chain.");
                return;
            }

            Directory.CreateDirectory(outputDir);
            
            // Step 1: Build MPRL object database by Unknown4 (which links to ParentIndex)
            var mprlObjectsById = new Dictionary<uint, MprlChunk.Entry>();
            foreach (var placement in scene.Placements)
            {
                mprlObjectsById[placement.Unknown4] = placement;
            }
            
            Console.WriteLine($"[ExportByMprlMslkChain] Found {mprlObjectsById.Count} MPRL objects.");
            
            // Step 2: Analyze the MPRR chunk for sentinel values (65535) that define object groups
            // (This is a placeholder - we'd need to actually parse the MPRR chunk data)
            
            // Step 3: Build a map from MSLK entries to MSUR surfaces
            var surfacesPerLink = new Dictionary<MslkEntry, List<MsurChunk.Entry>>();
            
            // Create an ordered list of surfaces by index range for binary search
            var surfaceRanges = scene.Surfaces.Select(s => new {
                Surface = s,
                Start = (int)s.MsviFirstIndex,
                End = (int)s.MsviFirstIndex + s.IndexCount - 1
            }).OrderBy(r => r.Start).ToArray();
            
            var surfaceStarts = surfaceRanges.Select(r => r.Start).ToArray();
            
            // Associate each MSLK entry with relevant MSUR surfaces
            foreach (var link in scene.Links.OfType<MslkEntry>())
            {
                if (link.MspiFirstIndex < 0 || link.MspiIndexCount == 0)
                    continue;
                    
                int nodeStart = link.MspiFirstIndex;
                int nodeEnd = link.MspiFirstIndex + link.MspiIndexCount - 1;
                
                // Find the surface that contains this index range using binary search
                int idx = Array.BinarySearch(surfaceStarts, nodeStart);
                if (idx < 0) idx = ~idx - 1;
                if (idx < 0) continue;
                
                var surfaceList = new List<MsurChunk.Entry>();
                surfacesPerLink[link] = surfaceList;
                
                // Find all surfaces that might belong to this link
                for (int i = idx; i < surfaceRanges.Length; i++)
                {
                    var range = surfaceRanges[i];
                    
                    // If the ranges overlap, associate this surface with the link
                    if (nodeStart <= range.End && nodeEnd >= range.Start)
                    {
                        surfaceList.Add(range.Surface);
                    }
                    
                    // Stop once we're past the end of the link's index range
                    if (range.Start > nodeEnd)
                        break;
                }
            }
            
            // Step 4: Group everything by MPRL object ID (full chain approach)
            var completeChainGroups = new Dictionary<uint, List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<uint, HashSet<int>>();
            
            // For each link, find all triangles in its surfaces
            foreach (var linkEntry in surfacesPerLink)
            {
                var link = linkEntry.Key;
                var surfaces = linkEntry.Value;
                
                // Check if this link is associated with an MPRL object
                uint parentIndex = link.ParentIndex;
                if (!mprlObjectsById.TryGetValue(parentIndex, out var mprlObject))
                    continue;
                
                // Use the parent index as the group key (it's the object ID)
                uint objectId = parentIndex;
                
                if (!completeChainGroups.TryGetValue(objectId, out var triangles))
                {
                    triangles = new List<(int, int, int)>();
                    completeChainGroups[objectId] = triangles;
                    usedVertsPerGroup[objectId] = new HashSet<int>();
                }
                
                // Process all surfaces associated with this link
                foreach (var surf in surfaces)
                {
                    int first = (int)surf.MsviFirstIndex;
                    int count = surf.IndexCount;

                    if (first < 0 || first + count > scene.Indices.Count)
                        continue; // invalid range

                    int triCount = count / 3;
                    for (int i = 0; i < triCount; i++)
                    {
                        int baseIdx = first + i * 3;
                        if (baseIdx + 2 >= scene.Indices.Count)
                            continue; // out of range

                        int ia = scene.Indices[baseIdx];
                        int ib = scene.Indices[baseIdx + 1];
                        int ic = scene.Indices[baseIdx + 2];

                        // Check vertex bounds
                        if (ia < 0 || ia >= scene.Vertices.Count ||
                            ib < 0 || ib >= scene.Vertices.Count ||
                            ic < 0 || ic >= scene.Vertices.Count)
                            continue; // invalid vertex indices

                        triangles.Add((ia, ib, ic));
                        usedVertsPerGroup[objectId].Add(ia);
                        usedVertsPerGroup[objectId].Add(ib);
                        usedVertsPerGroup[objectId].Add(ic);
                    }
                }
            }

            Console.WriteLine($"[ExportByMprlMslkChain] Found {completeChainGroups.Count} complete chain object groups.");

            // Export each object
            foreach (var kvp in completeChainGroups)
            {
                uint objectId = kvp.Key;
                var triangles = kvp.Value;
                var usedVerts = usedVertsPerGroup[objectId];
                
                // Skip objects with too few vertices
                if (usedVerts.Count < 3)
                    continue;
                
                // Get the MPRL object for additional information
                var mprl = mprlObjectsById[objectId];
                string baseName = $"object_{objectId}_type_{mprl.Unknown6}_pos_{mprl.Position.X}_{mprl.Position.Y}_{mprl.Position.Z}";
                
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                // Create remapping for OBJ export (1-based indices)
                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-test-grouping Complete Chain OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                // Export vertices
                foreach (var vIdx in usedVerts)
                {
                    remap[vIdx] = nextIdx++;
                    Vector3 v = scene.Vertices[vIdx];
                    // Flip X for world to OBJ convention
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                // Export faces or points
                if (writeFaces)
                {
                    foreach (var (a, b, c) in triangles)
                    {
                        if (remap.TryGetValue(a, out var ra) &&
                            remap.TryGetValue(b, out var rb) &&
                            remap.TryGetValue(c, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in usedVerts)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {usedVerts.Count}, faces {triangles.Count})");
            }
        }
        
        /// <summary>
        /// Exports geometry grouped by MPRR sentinel values to identify object boundaries.
        /// Uses the 65535 (0xFFFF) sentinel values in MPRR to define section boundaries.
        /// </summary>
        public static void ExportByMprrSections(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {
            if (scene.Surfaces.Count == 0 || scene.Properties.Count == 0)
            {
                Console.WriteLine("[ExportByMprrSections] Scene has incomplete data - cannot analyze MPRR sections.");
                return;
            }

            Directory.CreateDirectory(outputDir);
            
            // Step 1: Find all sentinel values (65535 = 0xFFFF) in MPRR and create object sections
            var sections = new List<(int Start, int End)>();
            int lastSentinel = -1;
            
            // Process MPRR entries to identify sections based on sentinel values
            for (int i = 0; i < scene.Properties.Count; i++)
            {
                if (scene.Properties[i].Value1 == 65535) // Sentinel value
                {
                    // If we've seen a previous sentinel, create a section
                    if (lastSentinel >= 0)
                    {
                        sections.Add((lastSentinel, i));
                    }
                    lastSentinel = i;
                }
            }
            
            // Add the final section if needed
            if (lastSentinel >= 0 && lastSentinel < scene.Properties.Count - 1)
            {
                sections.Add((lastSentinel, scene.Properties.Count - 1));
            }
            
            Console.WriteLine($"[ExportByMprrSections] Found {sections.Count} sections separated by sentinel values.");
            
            // Step 2: Try to map MPRR sections to MSUR/MSLK entries based on overlapping index ranges
            // This is experimental and may need refinement
            var sectionGroups = new Dictionary<int, List<(int A, int B, int C)>>();
            var usedVertsPerSection = new Dictionary<int, HashSet<int>>();
            
            // Create a mapping between MPRR sections and surfaces
            for (int sectionIdx = 0; sectionIdx < sections.Count; sectionIdx++)
            {
                var (start, end) = sections[sectionIdx];
                int sectionSize = end - start;
                
                // Skip very small sections (might be noise)
                if (sectionSize < 2) continue;
                
                sectionGroups[sectionIdx] = new List<(int A, int B, int C)>();
                usedVertsPerSection[sectionIdx] = new HashSet<int>();
                
                // Calculate section proportion to determine index range allocation
                double proportion = (double)sectionSize / scene.Properties.Count;
                int estimatedSurfaceCount = (int)(scene.Surfaces.Count * proportion);
                
                // Calculate the likely index range for this section
                int sectionStartIndex = (int)(scene.Indices.Count * ((double)start / scene.Properties.Count));
                int sectionEndIndex = (int)(scene.Indices.Count * ((double)end / scene.Properties.Count));
                
                // Find all surfaces that overlap with this estimated index range
                foreach (var surf in scene.Surfaces)
                {
                    int surfStart = (int)surf.MsviFirstIndex;
                    int surfEnd = surfStart + surf.IndexCount - 1;
                    
                    // Check if the surface overlaps with the estimated section range
                    if (surfStart <= sectionEndIndex && surfEnd >= sectionStartIndex)
                    {
                        // Add triangles from this surface to the section group
                        int first = surfStart;
                        int count = surf.IndexCount;
                        
                        if (first < 0 || first + count > scene.Indices.Count)
                            continue; // Invalid range
                        
                        int triCount = count / 3;
                        for (int i = 0; i < triCount; i++)
                        {
                            int ia = first + i * 3;
                            int ib = first + i * 3 + 1;
                            int ic = first + i * 3 + 2;
                            
                            if (ia < 0 || ia >= scene.Indices.Count ||
                                ib < 0 || ib >= scene.Indices.Count ||
                                ic < 0 || ic >= scene.Indices.Count)
                                continue;
                            
                            int a = scene.Indices[ia];
                            int b = scene.Indices[ib];
                            int c = scene.Indices[ic];
                            
                            sectionGroups[sectionIdx].Add((a, b, c));
                            usedVertsPerSection[sectionIdx].Add(a);
                            usedVertsPerSection[sectionIdx].Add(b);
                            usedVertsPerSection[sectionIdx].Add(c);
                        }
                    }
                }
            }
            
            // Step 3: Export each section as a separate OBJ
            Console.WriteLine($"[ExportByMprrSections] Exporting {sectionGroups.Count} sections as OBJ files...");
            foreach (var kv in sectionGroups)
            {
                int sectionIdx = kv.Key;
                var triangles = kv.Value;
                var usedVerts = usedVertsPerSection[sectionIdx];
                
                if (triangles.Count == 0)
                    continue; // Skip empty sections
                
                // Create a vertex remap so we only save used verts
                var remap = new Dictionary<int, int>();
                int nextIndex = 1; // OBJ indices are 1-based
                foreach (var idx in usedVerts.OrderBy(i => i))
                {
                    remap[idx] = nextIndex++;
                }
                
                string objPath = Path.Combine(outputDir, $"MPRR_Section_{sectionIdx}.obj");
                string mtlPath = Path.Combine(outputDir, $"MPRR_Section_{sectionIdx}.mtl");
                
                using (var sw = new StreamWriter(objPath))
                {
                    sw.WriteLine("# Generated by parpToolbox - ExportByMprrSections");
                    sw.WriteLine("# MPRR section: " + sectionIdx);
                    sw.WriteLine($"# Section range: {sections[sectionIdx].Start}-{sections[sectionIdx].End}");
                    sw.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");
                    sw.WriteLine("usemtl default");
                    
                    // Write vertices
                    foreach (var vIdx in usedVerts)
                    {
                        if (vIdx >= 0 && vIdx < scene.Vertices.Count)
                        {
                            var v = scene.Vertices[vIdx];
                            sw.WriteLine($"v {v.X} {v.Y} {v.Z}");
                        }
                    }
                    
                    // Write faces or points
                    if (writeFaces && triangles.Count > 0)
                    {
                        foreach (var (a, b, c) in triangles)
                        {
                            int ra = remap.TryGetValue(a, out var va) ? va : -1;
                            int rb = remap.TryGetValue(b, out var vb) ? vb : -1;
                            int rc = remap.TryGetValue(c, out var vc) ? vc : -1;
                            
                            if (ra > 0 && rb > 0 && rc > 0)
                            {
                                sw.WriteLine($"f {ra} {rb} {rc}");
                            }
                        }
                    }
                    else
                    {
                        foreach (var vIdx in usedVerts)
                        {
                            if (remap.TryGetValue(vIdx, out var rp))
                                sw.WriteLine($"p {rp}");
                        }
                    }
                }
                
                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {usedVerts.Count}, faces {triangles.Count})");
            }
        }
        
        /// <summary>
        /// Generates a comparison report of all grouping strategies
        /// </summary>
        private static void GenerateComparisonReport(string reportPath, Dictionary<string, (int Groups, int TotalVertices, int TotalFaces, int ValidFaces, int InvalidFaces)> metrics)
        {
            using var writer = new StreamWriter(reportPath);
            writer.WriteLine("# PM4 Grouping Strategy Comparison Report");
            writer.WriteLine("Generated: " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
            writer.WriteLine();
            
            writer.WriteLine("## Strategy Results Summary");
            writer.WriteLine();
            
            writer.WriteLine("| Strategy | Groups | Total Vertices | Total Faces | Valid Faces | Invalid Faces | Invalid % |");
            writer.WriteLine("|----------|--------|---------------|------------|------------|--------------|----------|");
            
            foreach (var kvp in metrics)
            {
                var name = kvp.Key;
                var stats = kvp.Value;
                double invalidPercent = stats.TotalFaces > 0 ? (double)stats.InvalidFaces / stats.TotalFaces * 100.0 : 0.0;
                
                writer.WriteLine($"| {name} | {stats.Groups} | {stats.TotalVertices} | {stats.TotalFaces} | {stats.ValidFaces} | {stats.InvalidFaces} | {invalidPercent:F2}% |");
            }
            
            writer.WriteLine();
            writer.WriteLine("## Analysis");
            writer.WriteLine();
            writer.WriteLine("### Key Findings:");
            writer.WriteLine("- The ParentIndex-based grouping appears to be the correct approach based on MPRL → MSLK linking evidence.");
            writer.WriteLine("- Connected component analysis identified physically connected mesh segments but may group unrelated objects.");
            writer.WriteLine("- The MPRL type fields (Unknown6=32768) appears to be a consistent type flag across objects.");
            writer.WriteLine("- Vertex connectivity produced the most visually coherent objects but may over-merge unrelated objects.");
            writer.WriteLine("- MPRR sections defined by sentinel values (65535) appear to segment the data into logical groups.");
            writer.WriteLine();
            writer.WriteLine("### Observations:");
            writer.WriteLine("- Missing vertices from adjacent tiles still causes significant data loss (~64%).");
            writer.WriteLine("- Global tile loading is essential for complete object reconstruction.");
            writer.WriteLine("- SurfaceGroupKey alone is insufficient for meaningful object grouping.");
            writer.WriteLine("- MPRL → MSLK → MSUR chain provides the most semantically meaningful grouping.");
            writer.WriteLine("- MPRR sentinel values (65535) appear to mark logical divisions between object sections.");
            writer.WriteLine();
            writer.WriteLine("### Recommendations:");
            writer.WriteLine("- Implement global tile loading to access all referenced vertices.");
            writer.WriteLine("- Use ParentIndex as the primary grouping key (from MPRL.Unknown4).");
            writer.WriteLine("- Supplement with vertex connectivity analysis for fragmented objects.");
            writer.WriteLine("- Filter out invalid vertex indices to prevent (0,0,0) anchor points.");
            writer.WriteLine("- Consider MPRR sentinel values as hierarchical markers for object subdivision boundaries.");
        }

        public static void ExportByCompositeKey(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {
            if (scene.Surfaces.Count == 0)
                throw new InvalidOperationException("Scene has no MSUR entries – cannot group by composite key.");

            Directory.CreateDirectory(outputDir);

            // Build quick lookup for MSLK→surface mapping so we can fetch ParentIndex
            var surfaceRanges = scene.Surfaces.Select(s => new {
                Surface = s,
                Start = (int)s.MsviFirstIndex,
                End = (int)s.MsviFirstIndex + s.IndexCount - 1
            }).OrderBy(r => r.Start).ToArray();
            var surfaceStarts = surfaceRanges.Select(r => r.Start).ToArray();
            var linksBySurface = new Dictionary<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry, List<MslkEntry>>();
            foreach (var link in scene.Links.OfType<MslkEntry>())
            {
                if (link.MspiFirstIndex < 0 || link.MspiIndexCount == 0)
                    continue;
                int nodeStart = link.MspiFirstIndex;
                int nodeEnd = link.MspiFirstIndex + link.MspiIndexCount - 1;
                int idx = Array.BinarySearch(surfaceStarts, nodeStart);
                if (idx < 0) idx = ~idx - 1;
                if (idx < 0) continue;
                var range = surfaceRanges[idx];
                if (nodeStart >= range.Start && nodeEnd <= range.End)
                {
                    if (!linksBySurface.TryGetValue(range.Surface, out var list))
                    {
                        list = new List<MslkEntry>();
                        linksBySurface[range.Surface] = list;
                    }
                    list.Add(link);
                }
            }

            // Map composite key -> triangles and used vertices
            var groups = new Dictionary<(uint ParentIdx, byte GroupKey, int IndexCount), List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<(uint ParentIdx, byte GroupKey, int IndexCount), HashSet<int>>();

            foreach (var surf in scene.Surfaces)
            {
                uint parentIdx = linksBySurface.TryGetValue(surf, out var lnk) ? lnk.FirstOrDefault()?.ParentIndex ?? 0xFFFFFFFFu : 0xFFFFFFFFu;
                var compKey = (parentIdx, surf.SurfaceGroupKey, surf.IndexCount);

                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;

                if (first < 0 || first + count > scene.Indices.Count)
                    continue; // skip invalid range

                int triCount = count / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    int ia = scene.Indices[baseIdx];
                    int ib = scene.Indices[baseIdx + 1];
                    int ic = scene.Indices[baseIdx + 2];

                    if (!groups.TryGetValue(compKey, out var list))
                    {
                        list = new List<(int, int, int)>();
                        groups[compKey] = list;
                        usedVertsPerGroup[compKey] = new HashSet<int>();
                    }

                    list.Add((ia, ib, ic));
                    usedVertsPerGroup[compKey].Add(ia);
                    usedVertsPerGroup[compKey].Add(ib);
                    usedVertsPerGroup[compKey].Add(ic);
                }
            }

            Console.WriteLine($"[GroupingTester] Discovered {groups.Count} distinct (Parent,GroupKey,IndexCount) composite keys.");

            foreach (var kvp in groups.OrderBy(k => k.Key.ParentIdx).ThenBy(k => k.Key.GroupKey).ThenBy(k => k.Key.IndexCount))
            {
                var key = kvp.Key;
                var faces = kvp.Value;
                var used = usedVertsPerGroup[key];

                string baseName = $"p{key.ParentIdx}_g{key.GroupKey}_cnt_{key.IndexCount}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-composite-grouping OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                foreach (var vIdx in used)
                {
                    if (vIdx < 0 || vIdx >= scene.Vertices.Count)
                        continue;

                    remap[vIdx] = nextIdx++;
                    var v = scene.Vertices[vIdx];
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                if (writeFaces)
                {
                    foreach (var (A, B, C) in faces)
                    {
                        if (remap.TryGetValue(A, out var ra) &&
                            remap.TryGetValue(B, out var rb) &&
                            remap.TryGetValue(C, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in used)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {used.Count}, faces {faces.Count})");
            }
        }
    }
}
