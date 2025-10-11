using WoWToolbox.Core.Models; // For MeshData
using System.Numerics; // For Vector3 if needed
using System.Collections.Generic;
using System.Linq;

namespace WoWToolbox.Common.Analysis
{
    public static class MeshAnalysisUtils
    {
        /// <summary>
        /// Finds all connected components within a given mesh.
        /// A component is a set of triangles where each triangle is reachable
        /// from any other triangle in the set by traversing shared vertices/edges.
        /// </summary>
        /// <param name="inputMesh">The input mesh data.</param>
        /// <returns>A list of MeshData objects, each representing a connected component.</returns>
        public static List<MeshData> FindConnectedComponents(MeshData inputMesh)
        {
            var components = new List<MeshData>();
            if (inputMesh == null || inputMesh.Indices == null || inputMesh.Indices.Count == 0)
            {
                return components; // Return empty list for empty mesh
            }

            int numTriangles = inputMesh.Indices.Count / 3;
            if (numTriangles == 0)
            {
                return components;
            }

            // 1. Build adjacency list: Map triangle index to list of adjacent triangle indices
            // Adjacency defined by sharing at least ONE vertex index.
            // Optimization: Sharing an edge (two vertices) is more robust but complex.
            // Let's start with shared vertex for simplicity.
            var adj = BuildTriangleAdjacency(inputMesh, numTriangles);

            // 2. Perform BFS/DFS to find connected components
            var visited = new bool[numTriangles];
            for (int i = 0; i < numTriangles; i++)
            {
                if (!visited[i])
                {
                    var currentComponentTriIndices = new List<int>();
                    var queue = new Queue<int>();

                    queue.Enqueue(i);
                    visited[i] = true;

                    while (queue.Count > 0)
                    {
                        int currentTriangle = queue.Dequeue();
                        currentComponentTriIndices.Add(currentTriangle);

                        if (adj.ContainsKey(currentTriangle))
                        {
                            foreach (int neighbor in adj[currentTriangle])
                            {
                                if (!visited[neighbor])
                                {
                                    visited[neighbor] = true;
                                    queue.Enqueue(neighbor);
                                }
                            }
                        }
                    }

                    // 3. Create MeshData for the found component
                    var componentMesh = CreateComponentMeshData(inputMesh, currentComponentTriIndices);
                    if (componentMesh != null)
                    {
                        components.Add(componentMesh);
                    }
                }
            }

            return components;
        }

        private static Dictionary<int, List<int>> BuildTriangleAdjacency(MeshData mesh, int numTriangles)
        {
            // Map vertex index to list of triangles using that vertex
            var vertexToTriangles = new Dictionary<int, List<int>>();
            for (int triIndex = 0; triIndex < numTriangles; triIndex++)
            {
                int i0 = mesh.Indices[triIndex * 3 + 0];
                int i1 = mesh.Indices[triIndex * 3 + 1];
                int i2 = mesh.Indices[triIndex * 3 + 2];

                if (!vertexToTriangles.ContainsKey(i0)) vertexToTriangles[i0] = new List<int>();
                vertexToTriangles[i0].Add(triIndex);

                if (!vertexToTriangles.ContainsKey(i1)) vertexToTriangles[i1] = new List<int>();
                vertexToTriangles[i1].Add(triIndex);

                if (!vertexToTriangles.ContainsKey(i2)) vertexToTriangles[i2] = new List<int>();
                vertexToTriangles[i2].Add(triIndex);
            }

            // Build adjacency list based on shared vertices
            var adj = new Dictionary<int, List<int>>();
            for (int triIndex = 0; triIndex < numTriangles; triIndex++)
            {
                var neighbors = new HashSet<int>();
                int i0 = mesh.Indices[triIndex * 3 + 0];
                int i1 = mesh.Indices[triIndex * 3 + 1];
                int i2 = mesh.Indices[triIndex * 3 + 2];

                // Find triangles sharing vertices with the current triangle
                if (vertexToTriangles.ContainsKey(i0)) neighbors.UnionWith(vertexToTriangles[i0]);
                if (vertexToTriangles.ContainsKey(i1)) neighbors.UnionWith(vertexToTriangles[i1]);
                if (vertexToTriangles.ContainsKey(i2)) neighbors.UnionWith(vertexToTriangles[i2]);

                // Remove self from neighbors
                neighbors.Remove(triIndex);

                if (neighbors.Count > 0)
                {
                    if (!adj.ContainsKey(triIndex)) adj[triIndex] = new List<int>();
                    adj[triIndex].AddRange(neighbors);
                }
            }
            return adj;
        }

        private static MeshData? CreateComponentMeshData(MeshData originalMesh, List<int> componentTriangleIndices)
        {
            if (componentTriangleIndices == null || componentTriangleIndices.Count == 0)
                return null;

            var componentMesh = new MeshData();
            var originalToLocalVertexIndexMap = new Dictionary<int, int>(); // Map original vertex index to new local index

            foreach (int triIndex in componentTriangleIndices)
            {
                int originalI0 = originalMesh.Indices[triIndex * 3 + 0];
                int originalI1 = originalMesh.Indices[triIndex * 3 + 1];
                int originalI2 = originalMesh.Indices[triIndex * 3 + 2];

                // Add vertices to the component mesh if not already added, and map indices
                int localI0 = AddVertexAndMap(originalMesh, componentMesh, originalToLocalVertexIndexMap, originalI0);
                int localI1 = AddVertexAndMap(originalMesh, componentMesh, originalToLocalVertexIndexMap, originalI1);
                int localI2 = AddVertexAndMap(originalMesh, componentMesh, originalToLocalVertexIndexMap, originalI2);

                // Add the triangle with local indices
                componentMesh.Indices.Add(localI0);
                componentMesh.Indices.Add(localI1);
                componentMesh.Indices.Add(localI2);
            }

            return componentMesh;
        }

        private static int AddVertexAndMap(MeshData originalMesh, MeshData componentMesh, Dictionary<int, int> map, int originalIndex)
        {
            if (!map.TryGetValue(originalIndex, out int localIndex))
            {
                // Vertex not yet in the component mesh, add it
                localIndex = componentMesh.Vertices.Count;
                if (originalIndex >= 0 && originalIndex < originalMesh.Vertices.Count)
                {
                    componentMesh.Vertices.Add(originalMesh.Vertices[originalIndex]);
                    map[originalIndex] = localIndex;
                }
                else
                {
                    // Handle potential invalid original index - this shouldn't happen if input is valid
                    System.Console.WriteLine($"[ERR][MeshAnalysisUtils] Invalid original vertex index {originalIndex} encountered.");
                    // Decide on error handling: throw, return -1, or add a placeholder vertex?
                    // For now, let's return -1, which will likely cause issues downstream but prevents crashing here.
                    return -1; 
                }
            }
            return localIndex;
        }


        /// <summary>
        /// Finds the largest connected component by triangle count.
        /// </summary>
        /// <param name="inputMesh">The input mesh data.</param>
        /// <returns>The MeshData for the largest component, or null if the input mesh is empty.</returns>
        public static MeshData? GetLargestComponent(MeshData inputMesh)
        {
            var components = FindConnectedComponents(inputMesh);
            // Ensure we handle the case where components might be empty or contain nulls if errors occurred
            return components?.Where(c => c != null).OrderByDescending(c => c.Indices.Count).FirstOrDefault();
        }

        /// <summary>
        /// Loads a simple OBJ file (vertices and faces only) into a MeshData object.
        /// Supports only 'v' and 'f' lines, ignores normals/UVs/materials.
        /// </summary>
        public static MeshData LoadObjToMeshData(string objFilePath)
        {
            var mesh = new MeshData();
            var lines = System.IO.File.ReadAllLines(objFilePath);
            foreach (var line in lines)
            {
                if (line.StartsWith("v "))
                {
                    var parts = line.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 4 &&
                        float.TryParse(parts[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float x) &&
                        float.TryParse(parts[2], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float y) &&
                        float.TryParse(parts[3], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float z))
                    {
                        mesh.Vertices.Add(new System.Numerics.Vector3(x, y, z));
                    }
                }
                else if (line.StartsWith("f "))
                {
                    var parts = line.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 4) // Only triangles supported
                    {
                        // OBJ indices are 1-based, so subtract 1
                        for (int i = 1; i + 2 <= parts.Length - 1; i++)
                        {
                            int i0 = ParseObjIndex(parts[1], mesh.Vertices.Count);
                            int i1 = ParseObjIndex(parts[i], mesh.Vertices.Count);
                            int i2 = ParseObjIndex(parts[i + 1], mesh.Vertices.Count);
                            if (i0 >= 0 && i1 >= 0 && i2 >= 0)
                            {
                                mesh.Indices.Add(i0);
                                mesh.Indices.Add(i1);
                                mesh.Indices.Add(i2);
                            }
                        }
                    }
                }
            }
            return mesh;
        }

        private static int ParseObjIndex(string part, int vertexCount)
        {
            // Handles 'f v', 'f v/vt', 'f v/vt/vn', 'f v//vn'
            var idxStr = part.Split('/')[0];
            if (int.TryParse(idxStr, out int idx))
            {
                if (idx > 0) return idx - 1;
                if (idx < 0) return vertexCount + idx; // Negative indices
            }
            return -1;
        }
    }
} 