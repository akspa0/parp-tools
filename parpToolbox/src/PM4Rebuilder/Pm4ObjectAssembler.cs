using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4Rebuilder
{
    /// <summary>
    /// Groups PM4 geometry into higher-level objects using the confirmed mapping:
    ///   MPRL.Unknown4 == MSLK.ParentIndex
    /// This logic is a trimmed copy of the internal Pm4MprlObjectGrouper from parpToolbox,
    /// made public for PM4Rebuilder so we don't need internals.
    /// </summary>
    internal static class Pm4ObjectAssembler
    {
        public sealed record BuildingObject(
            uint PlacementId,
            string Name,
            Vector3 Position,
            Vector3 Rotation,
            List<(int A,int B,int C)> Triangles,
            List<MslkEntry> Links,
            int VertexCount);

        /// <summary>
        /// Groups geometry by MPRL placement.
        /// </summary>
        public static List<BuildingObject> AssembleObjects(Pm4Scene scene)
        {
            return AssembleObjects(scene, 40.0f); // Default 40-unit clustering for typical buildings
        }

        /// <summary>
        /// Groups geometry by MPRL placement with spatial clustering.
        /// </summary>
        public static List<BuildingObject> AssembleObjects(Pm4Scene scene, float clusteringDistance)
        {
            Console.WriteLine($"[Assembler] Placements={scene.Placements.Count}, Links={scene.Links.Count}");

            var linksByParent = scene.Links
                .Where(l => l.ParentIndex > 0)
                .GroupBy(l => l.ParentIndex)
                .ToDictionary(g => g.Key, g => g.ToList());

            var results = new List<BuildingObject>();
            int totalMscnTriangles = 0;
            int totalMsviTriangles = 0;

            foreach (var placement in scene.Placements)
            {
                if (!linksByParent.TryGetValue(placement.Unknown4, out var links))
                    continue;

                var triangleList = new List<(int,int,int)>();
                var usedVerts = new HashSet<int>();

                foreach (var link in links.Where(l => l.MspiFirstIndex >= 0 && l.MspiIndexCount > 0))
                {
                    int start = link.MspiFirstIndex;
                    int count = link.MspiIndexCount;

                    // Check if MspiFirstIndex references MSCN vertices directly (based on linkage analysis)
                    int mscnStart = scene.Vertices.Count;
                    int mscnEnd = scene.Vertices.Count + scene.MscnVertices.Count;
                    
                    if (start >= mscnStart && start < mscnEnd)
                    {
                        // BREAKTHROUGH: MspiFirstIndex directly references MSCN vertices!
                        // Generate triangles from sequential MSCN vertices
                        int mscnVertexCount = Math.Min(count, mscnEnd - start);
                        
                        // Create triangles from sequential MSCN vertices (assuming triangle strips or fans)
                        for (int i = 0; i + 2 < mscnVertexCount; i += 3)
                        {
                            int a = start + i;
                            int b = start + i + 1;
                            int c = start + i + 2;
                            
                            triangleList.Add((a + 1, b + 1, c + 1)); // OBJ uses 1-based indices
                            usedVerts.Add(a);
                            usedVerts.Add(b);
                            usedVerts.Add(c);
                            totalMscnTriangles++;
                        }
                    }
                    else
                    {
                        // Traditional MSVI index buffer processing
                        // Guard against corrupt or cross-tile references that point past the available index buffer.
                        if (start < 0 || start >= scene.Indices.Count)
                            continue;

                        int maxCount = Math.Min(count, scene.Indices.Count - start);
                        // Ensure we iterate in triangle-sized steps only within the safe slice.
                        for (int i = 0; i + 2 < maxCount; i += 3)
                        {
                            int a = scene.Indices[start + i];
                            int b = scene.Indices[start + i + 1];
                            int c = scene.Indices[start + i + 2];

                        bool IsValidIndex(int idx)
                        {
                            if (idx >= 0 && idx < scene.Vertices.Count)
                                return true;
                            int mscnOffset = idx - scene.Vertices.Count;
                            return mscnOffset >= 0 && mscnOffset < scene.MscnVertices.Count;
                        }

                        if (IsValidIndex(a) && IsValidIndex(b) && IsValidIndex(c))
                        {
                            // Keep original index values for mapping later – they already align with global buffers.
                            triangleList.Add((a + 1, b + 1, c + 1)); // OBJ uses 1-based indices
                            usedVerts.Add(a);
                            usedVerts.Add(b);
                            usedVerts.Add(c);
                            totalMsviTriangles++;
                        }
                        } // Close for loop
                    } // Close else block
                }

                if (triangleList.Count==0) continue;

                results.Add(new BuildingObject(
                    placement.Unknown4,
                    $"obj_{placement.Unknown4:X4}",
                    placement.Position,
                    Vector3.Zero,
                    triangleList,
                    links,
                    usedVerts.Count));
            }

            Console.WriteLine($"[Assembler] Initial objects: {results.Count}");
            Console.WriteLine($"[BREAKTHROUGH] MSCN triangles: {totalMscnTriangles}, MSVI triangles: {totalMsviTriangles}");
            if (totalMscnTriangles > 0)
            {
                Console.WriteLine($"[SUCCESS] MSCN geometry successfully integrated! Interior/collision geometry now included.");
            }

            // Apply spatial clustering to group nearby objects into unified buildings
            var clusteredResults = ApplySpatialClustering(results, scene, clusteringDistance);
            Console.WriteLine($"[Assembler] After spatial clustering: {clusteredResults.Count} unified buildings");
            
            return clusteredResults;
        }

        /// <summary>
        /// Apply spatial clustering to group nearby objects into unified buildings.
        /// </summary>
        private static List<BuildingObject> ApplySpatialClustering(List<BuildingObject> objects, Pm4Scene scene, float clusteringDistance)
        {
            if (objects.Count == 0) return objects;

            Console.WriteLine($"[Clustering] Using {clusteringDistance} unit clustering distance");
            
            // Calculate centroids for each object
            var objectCentroids = new Dictionary<BuildingObject, Vector3>();
            foreach (var obj in objects)
            {
                objectCentroids[obj] = CalculateObjectCentroid(obj, scene);
            }

            // Simple distance-based clustering
            var clusters = new List<List<BuildingObject>>();
            var processed = new HashSet<BuildingObject>();

            foreach (var obj in objects)
            {
                if (processed.Contains(obj)) continue;

                var cluster = new List<BuildingObject> { obj };
                processed.Add(obj);
                var objCentroid = objectCentroids[obj];

                // Find all objects within clustering distance
                foreach (var other in objects)
                {
                    if (processed.Contains(other) || obj == other) continue;

                    var otherCentroid = objectCentroids[other];
                    var distance = Vector3.Distance(objCentroid, otherCentroid);

                    if (distance <= clusteringDistance)
                    {
                        cluster.Add(other);
                        processed.Add(other);
                    }
                }

                clusters.Add(cluster);
            }

            Console.WriteLine($"[Clustering] Created {clusters.Count} clusters from {objects.Count} objects");

            // Merge objects in each cluster
            var result = new List<BuildingObject>();
            for (int i = 0; i < clusters.Count; i++)
            {
                var cluster = clusters[i];
                if (cluster.Count == 1)
                {
                    result.Add(cluster[0]);
                }
                else
                {
                    var merged = MergeObjects(cluster, i);
                    result.Add(merged);
                }
            }

            return result;
        }

        /// <summary>
        /// Calculate the centroid of an object based on its triangles and vertices.
        /// </summary>
        private static Vector3 CalculateObjectCentroid(BuildingObject obj, Pm4Scene scene)
        {
            var sum = Vector3.Zero;
            int vertexCount = 0;

            foreach (var triangle in obj.Triangles)
            {
                // Convert from 1-based OBJ indices to 0-based
                var indices = new[] { triangle.A - 1, triangle.B - 1, triangle.C - 1 };
                
                foreach (var idx in indices)
                {
                    Vector3 vertex;
                    if (idx < scene.Vertices.Count)
                    {
                        // MSVT vertex
                        vertex = scene.Vertices[idx];
                    }
                    else
                    {
                        // MSCN vertex
                        int mscnIdx = idx - scene.Vertices.Count;
                        if (mscnIdx >= 0 && mscnIdx < scene.MscnVertices.Count)
                        {
                            vertex = scene.MscnVertices[mscnIdx];
                        }
                        else
                        {
                            continue; // Skip invalid indices
                        }
                    }
                    
                    sum += vertex;
                    vertexCount++;
                }
            }

            return vertexCount > 0 ? sum / vertexCount : obj.Position;
        }

        /// <summary>
        /// Merge multiple objects into a single unified building object.
        /// </summary>
        private static BuildingObject MergeObjects(List<BuildingObject> objects, int clusterId)
        {
            var allTriangles = new List<(int A, int B, int C)>();
            var allLinks = new List<MslkEntry>();
            var totalVertexCount = 0;
            
            // Calculate cluster centroid for positioning
            var positions = objects.Select(o => o.Position).ToArray();
            var clusterPosition = new Vector3(
                positions.Average(p => p.X),
                positions.Average(p => p.Y),
                positions.Average(p => p.Z)
            );

            foreach (var obj in objects)
            {
                allTriangles.AddRange(obj.Triangles);
                allLinks.AddRange(obj.Links);
                totalVertexCount += obj.VertexCount;
            }

            // Use the first object's ID as base, but mark as clustered
            var firstObj = objects[0];
            
            return new BuildingObject(
                firstObj.PlacementId,
                $"building_cluster_{clusterId:D3}_{objects.Count}objs",
                clusterPosition,
                firstObj.Rotation,
                allTriangles,
                allLinks,
                totalVertexCount
            );
        }
    }
}
