using System.Numerics;
using System.Runtime.CompilerServices;
using WoWToolbox.Core.v2.Foundation.Data;

namespace WoWToolbox.Core.v2.Foundation.Utilities
{
    /// <summary>
    /// High-performance utilities for CompleteWMOModel operations.
    /// Provides optimized geometric calculations, normal generation, and model analysis.
    /// </summary>
    public static class CompleteWMOModelUtilities
    {
        #region Normal Generation

        /// <summary>
        /// Generates smooth vertex normals for a CompleteWMOModel using optimized algorithms.
        /// </summary>
        /// <param name="model">The model to generate normals for</param>
        /// <param name="smoothingAngle">Maximum angle in radians for normal smoothing (default: 60 degrees)</param>
        public static void GenerateNormals(CompleteWMOModel model, float smoothingAngle = 1.047f) // 60 degrees
        {
            if (!model.HasGeometry)
                return;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;
            var normals = model.Normals;
            
            // Clear existing normals and pre-allocate
            normals.Clear();
            normals.Capacity = Math.Max(normals.Capacity, vertices.Count);
            
            // Initialize normals with zero vectors
            for (int i = 0; i < vertices.Count; i++)
            {
                normals.Add(Vector3.Zero);
            }

            var cosThreshold = MathF.Cos(smoothingAngle);

            // Calculate face normals and accumulate vertex normals
            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                    continue;

                var v1 = vertices[idx1];
                var v2 = vertices[idx2];
                var v3 = vertices[idx3];

                // Calculate face normal
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var faceNormal = Vector3.Normalize(Vector3.Cross(edge1, edge2));

                // Accumulate normal for each vertex of the triangle
                normals[idx1] += faceNormal;
                normals[idx2] += faceNormal;
                normals[idx3] += faceNormal;
            }

            // Normalize all vertex normals
            for (int i = 0; i < normals.Count; i++)
            {
                var normal = normals[i];
                if (normal.LengthSquared() > 0.001f) // Avoid normalizing zero vectors
                {
                    normals[i] = Vector3.Normalize(normal);
                }
                else
                {
                    normals[i] = Vector3.UnitY; // Default upward normal
                }
            }
        }

        /// <summary>
        /// Generates flat normals (one normal per triangle) for hard-edge appearance.
        /// </summary>
        /// <param name="model">The model to generate flat normals for</param>
        public static void GenerateFlatNormals(CompleteWMOModel model)
        {
            if (!model.HasGeometry)
                return;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;
            var normals = model.Normals;
            
            // Clear existing normals
            normals.Clear();
            normals.Capacity = Math.Max(normals.Capacity, vertices.Count);
            
            // Initialize with zero vectors
            for (int i = 0; i < vertices.Count; i++)
            {
                normals.Add(Vector3.Zero);
            }

            // Calculate flat normals per triangle
            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                    continue;

                var v1 = vertices[idx1];
                var v2 = vertices[idx2];
                var v3 = vertices[idx3];

                // Calculate face normal
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var faceNormal = Vector3.Normalize(Vector3.Cross(edge1, edge2));

                // Assign same normal to all vertices of the triangle
                normals[idx1] = faceNormal;
                normals[idx2] = faceNormal;
                normals[idx3] = faceNormal;
            }
        }

        #endregion

        #region Geometric Analysis

        /// <summary>
        /// Calculates detailed geometric statistics for a model.
        /// </summary>
        /// <param name="model">The model to analyze</param>
        /// <returns>Comprehensive geometric statistics</returns>
        public static GeometricStats CalculateGeometricStats(CompleteWMOModel model)
        {
            var stats = new GeometricStats();
            
            if (!model.HasGeometry)
                return stats;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;

            stats.VertexCount = vertices.Count;
            stats.TriangleCount = model.FaceCount;

            // Calculate bounding box
            var bounds = model.CalculateBoundingBox();
            if (bounds.HasValue)
            {
                stats.BoundingBox = bounds.Value;
                stats.Volume = bounds.Value.Volume;
                stats.SurfaceArea = EstimateSurfaceArea(model);
            }

            // Calculate edge lengths and triangle areas
            var edgeLengths = new List<float>();
            var triangleAreas = new List<float>();

            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                    continue;

                var v1 = vertices[idx1];
                var v2 = vertices[idx2];
                var v3 = vertices[idx3];

                // Edge lengths
                var edge1Length = Vector3.Distance(v1, v2);
                var edge2Length = Vector3.Distance(v2, v3);
                var edge3Length = Vector3.Distance(v3, v1);

                edgeLengths.Add(edge1Length);
                edgeLengths.Add(edge2Length);
                edgeLengths.Add(edge3Length);

                // Triangle area using cross product
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var area = Vector3.Cross(edge1, edge2).Length() * 0.5f;
                triangleAreas.Add(area);
            }

            // Statistical analysis
            if (edgeLengths.Count > 0)
            {
                stats.MinEdgeLength = edgeLengths.Min();
                stats.MaxEdgeLength = edgeLengths.Max();
                stats.AverageEdgeLength = edgeLengths.Average();
            }

            if (triangleAreas.Count > 0)
            {
                stats.MinTriangleArea = triangleAreas.Min();
                stats.MaxTriangleArea = triangleAreas.Max();
                stats.AverageTriangleArea = triangleAreas.Average();
            }

            return stats;
        }

        /// <summary>
        /// Estimates the surface area of a model by summing triangle areas.
        /// </summary>
        /// <param name="model">The model to calculate surface area for</param>
        /// <returns>Estimated surface area</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float EstimateSurfaceArea(CompleteWMOModel model)
        {
            if (!model.HasGeometry)
                return 0f;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;
            float totalArea = 0f;

            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                    continue;

                var v1 = vertices[idx1];
                var v2 = vertices[idx2];
                var v3 = vertices[idx3];

                // Triangle area using cross product
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var area = Vector3.Cross(edge1, edge2).Length() * 0.5f;
                totalArea += area;
            }

            return totalArea;
        }

        #endregion

        #region Optimization Operations

        /// <summary>
        /// Removes duplicate vertices and updates indices accordingly.
        /// </summary>
        /// <param name="model">The model to optimize</param>
        /// <param name="tolerance">Distance tolerance for considering vertices duplicates</param>
        /// <returns>Number of vertices removed</returns>
        public static int RemoveDuplicateVertices(CompleteWMOModel model, float tolerance = 0.001f)
        {
            if (!model.HasGeometry)
                return 0;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;
            var normals = model.Normals;
            var texCoords = model.TexCoords;

            var uniqueVertices = new List<Vector3>();
            var vertexMap = new Dictionary<int, int>(); // old index -> new index
            var toleranceSquared = tolerance * tolerance;

            // Find unique vertices
            for (int i = 0; i < vertices.Count; i++)
            {
                var vertex = vertices[i];
                int uniqueIndex = -1;

                // Check if this vertex is close to any existing unique vertex
                for (int j = 0; j < uniqueVertices.Count; j++)
                {
                    if (Vector3.DistanceSquared(vertex, uniqueVertices[j]) <= toleranceSquared)
                    {
                        uniqueIndex = j;
                        break;
                    }
                }

                if (uniqueIndex == -1)
                {
                    // Add new unique vertex
                    uniqueIndex = uniqueVertices.Count;
                    uniqueVertices.Add(vertex);
                }

                vertexMap[i] = uniqueIndex;
            }

            var removedCount = vertices.Count - uniqueVertices.Count;

            if (removedCount > 0)
            {
                // Update vertices
                vertices.Clear();
                vertices.AddRange(uniqueVertices);

                // Update triangle indices
                for (int i = 0; i < triangles.Count; i++)
                {
                    triangles[i] = vertexMap[triangles[i]];
                }

                // Update normals if they exist
                if (normals.Count == vertices.Count + removedCount)
                {
                    var newNormals = new List<Vector3>();
                    for (int i = 0; i < uniqueVertices.Count; i++)
                    {
                        newNormals.Add(Vector3.UnitY); // Default normal
                    }
                    normals.Clear();
                    normals.AddRange(newNormals);
                }

                // Update texture coordinates if they exist
                if (texCoords.Count == vertices.Count + removedCount)
                {
                    var newTexCoords = new List<Vector2>();
                    for (int i = 0; i < uniqueVertices.Count; i++)
                    {
                        newTexCoords.Add(Vector2.Zero); // Default UV
                    }
                    texCoords.Clear();
                    texCoords.AddRange(newTexCoords);
                }
            }

            return removedCount;
        }

        /// <summary>
        /// Removes degenerate triangles (triangles with duplicate or collinear vertices).
        /// </summary>
        /// <param name="model">The model to clean</param>
        /// <returns>Number of triangles removed</returns>
        public static int RemoveDegenerateTriangles(CompleteWMOModel model)
        {
            if (!model.HasGeometry)
                return 0;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;
            var validTriangles = new List<int>();

            int removedCount = 0;

            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                // Check for duplicate indices
                if (idx1 == idx2 || idx2 == idx3 || idx1 == idx3)
                {
                    removedCount++;
                    continue;
                }

                // Check bounds
                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                {
                    removedCount++;
                    continue;
                }

                var v1 = vertices[(int)idx1];
                var v2 = vertices[(int)idx2];
                var v3 = vertices[(int)idx3];

                // Check for collinear vertices (zero area triangle)
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var crossProduct = Vector3.Cross(edge1, edge2);
                
                if (crossProduct.LengthSquared() < 0.000001f) // Very small area threshold
                {
                    removedCount++;
                    continue;
                }

                // Triangle is valid
                validTriangles.Add((int)idx1);
                validTriangles.Add((int)idx2);
                validTriangles.Add((int)idx3);
            }

            // Update triangle indices
            triangles.Clear();
            triangles.AddRange(validTriangles);

            return removedCount;
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Calculates the centroid (geometric center) of the model.
        /// </summary>
        /// <param name="model">The model to calculate centroid for</param>
        /// <returns>Centroid position</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 CalculateCentroid(CompleteWMOModel model)
        {
            if (!model.HasGeometry)
                return Vector3.Zero;

            var vertices = model.Vertices;
            var sum = Vector3.Zero;

            foreach (var vertex in vertices)
            {
                sum += vertex;
            }

            return sum / vertices.Count;
        }

        /// <summary>
        /// Checks if the model has consistent winding order.
        /// </summary>
        /// <param name="model">The model to check</param>
        /// <returns>Winding analysis result</returns>
        public static WindingOrderResult AnalyzeWindingOrder(CompleteWMOModel model)
        {
            var result = new WindingOrderResult();
            
            if (!model.HasGeometry)
                return result;

            var vertices = model.Vertices;
            var triangles = model.TriangleIndices;
            var centroid = CalculateCentroid(model);

            for (int i = 0; i < triangles.Count; i += 3)
            {
                if (i + 2 >= triangles.Count) break;

                var idx1 = triangles[i];
                var idx2 = triangles[i + 1];
                var idx3 = triangles[i + 2];

                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                    continue;

                var v1 = vertices[idx1];
                var v2 = vertices[idx2];
                var v3 = vertices[idx3];

                // Calculate normal using cross product
                var edge1 = v2 - v1;
                var edge2 = v3 - v1;
                var normal = Vector3.Cross(edge1, edge2);

                // Check if normal points outward (away from centroid)
                var triangleCenter = (v1 + v2 + v3) / 3f;
                var outwardDirection = triangleCenter - centroid;
                var dotProduct = Vector3.Dot(normal, outwardDirection);

                if (dotProduct > 0)
                    result.CounterClockwiseTriangles++;
                else
                    result.ClockwiseTriangles++;
            }

            result.TotalTriangles = result.CounterClockwiseTriangles + result.ClockwiseTriangles;
            return result;
        }

        #endregion
    }

    #region Supporting Data Structures

    /// <summary>
    /// Comprehensive geometric statistics for a CompleteWMOModel.
    /// </summary>
    public struct GeometricStats
    {
        public int VertexCount;
        public int TriangleCount;
        public BoundingBox3D BoundingBox;
        public float Volume;
        public float SurfaceArea;
        public float MinEdgeLength;
        public float MaxEdgeLength;
        public float AverageEdgeLength;
        public float MinTriangleArea;
        public float MaxTriangleArea;
        public float AverageTriangleArea;

        public bool IsValid => VertexCount > 0 && TriangleCount > 0;
        public float AspectRatio => BoundingBox.Size.X / Math.Max(BoundingBox.Size.Y, 0.001f);
    }

    /// <summary>
    /// Result of winding order analysis.
    /// </summary>
    public struct WindingOrderResult
    {
        public int ClockwiseTriangles;
        public int CounterClockwiseTriangles;
        public int TotalTriangles;

        public float ClockwiseRatio => TotalTriangles > 0 ? (float)ClockwiseTriangles / TotalTriangles : 0f;
        public bool IsConsistent => ClockwiseTriangles == 0 || CounterClockwiseTriangles == 0;
        public bool IsMajorityClockwise => ClockwiseTriangles > CounterClockwiseTriangles;
    }

    #endregion
} 