using WoWToolbox.Core.Models; // For MeshData
using System.Numerics;
using System; // For Math
using System.Collections.Generic;
using System.Linq;

namespace WoWToolbox.Common.Analysis
{
    public struct BoundingBox
    {
        public Vector3 Min { get; set; }
        public Vector3 Max { get; set; }
        public Vector3 Center => Min + (Size / 2.0f);
        public Vector3 Size => Max - Min;

        public static BoundingBox Empty => new BoundingBox { Min = new Vector3(float.MaxValue), Max = new Vector3(float.MinValue) };

        public bool IsValid => Min.X <= Max.X && Min.Y <= Max.Y && Min.Z <= Max.Z;
    }

    public enum ComparisonResult
    {
        Error = 0,       // Could not perform comparison (e.g., empty mesh)
        Mismatch = 1,    // Clearly different based on simple metrics
        PotentialMatch = 2, // Simple metrics are close, further analysis needed
        Match = 3          // Simple metrics are very close (within tolerance)
    }

    public static class MeshComparisonUtils
    {
        /// <summary>
        /// Calculates the Axis-Aligned Bounding Box (AABB) for the given mesh.
        /// </summary>
        public static BoundingBox CalculateAABB(MeshData mesh)
        {
            if (mesh == null || mesh.Vertices == null || mesh.Vertices.Count == 0)
            {
                return BoundingBox.Empty;
            }

            Vector3 min = new Vector3(float.MaxValue);
            Vector3 max = new Vector3(float.MinValue);

            foreach (var vertex in mesh.Vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            return new BoundingBox { Min = min, Max = max };
        }

        /// <summary>
        /// Performs a basic comparison between two meshes based on vertex/triangle counts and AABB.
        /// Assumes both meshes are in the SAME coordinate system.
        /// </summary>
        /// <param name="meshA">The first mesh.</param>
        /// <param name="meshB">The second mesh.</param>
        /// <param name="countTolerance">Relative tolerance for vertex/triangle counts (e.g., 0.1 for 10%).</param>
        /// <param name="sizeTolerance">Relative tolerance for AABB size dimensions (e.g., 0.1 for 10%).</param>
        /// <returns>A ComparisonResult indicating similarity.</returns>
        public static ComparisonResult CompareMeshesBasic(MeshData meshA, MeshData meshB, float countTolerance = 0.05f, float sizeTolerance = 0.1f)
        {
            if (meshA == null || meshB == null || !meshA.IsValid() || !meshB.IsValid())
            {
                return ComparisonResult.Error;
            }

            // Vertex Count Comparison
            int vertCountA = meshA.Vertices.Count;
            int vertCountB = meshB.Vertices.Count;
            if (!AreCountsClose(vertCountA, vertCountB, countTolerance))
            {
                Console.WriteLine($"[Compare] Mismatch: Vertex counts differ significantly ({vertCountA} vs {vertCountB}).");
                return ComparisonResult.Mismatch;
            }

            // Triangle Count Comparison
            int triCountA = meshA.Indices.Count / 3;
            int triCountB = meshB.Indices.Count / 3;
             if (!AreCountsClose(triCountA, triCountB, countTolerance))
            {
                Console.WriteLine($"[Compare] Mismatch: Triangle counts differ significantly ({triCountA} vs {triCountB}).");
                return ComparisonResult.Mismatch;
            }

            // Bounding Box Comparison
            var aabbA = CalculateAABB(meshA);
            var aabbB = CalculateAABB(meshB);

            if (!aabbA.IsValid || !aabbB.IsValid)
                 return ComparisonResult.Error; // Should not happen if meshes are valid

            // Compare AABB sizes (relative difference)
            if (!AreSizesClose(aabbA.Size, aabbB.Size, sizeTolerance))
            {
                 Console.WriteLine($"[Compare] Mismatch: AABB sizes differ significantly ({aabbA.Size} vs {aabbB.Size}).");
                return ComparisonResult.Mismatch;
            }

            // If counts and sizes are close, consider it a potential match for now.
            // More sophisticated checks (center point proximity, shape analysis) would be needed for higher confidence.
             Console.WriteLine($"[Compare] Potential Match: Counts and AABB sizes are within tolerance.");
            return ComparisonResult.PotentialMatch; // Or Match if tolerances are very strict
        }

        private static bool AreCountsClose(int countA, int countB, float tolerance)
        {
            if (countA == 0 && countB == 0) return true;
            if (countA == 0 || countB == 0) return false; // One is zero, the other isn't
            float diff = Math.Abs(countA - countB);
            float avg = (countA + countB) / 2.0f;
            return (diff / avg) <= tolerance;
        }

         private static bool AreSizesClose(Vector3 sizeA, Vector3 sizeB, float tolerance)
        {
            // Avoid division by zero if a dimension is zero
            if (!IsClose(sizeA.X, sizeB.X, tolerance, 0.001f)) return false;
            if (!IsClose(sizeA.Y, sizeB.Y, tolerance, 0.001f)) return false;
            if (!IsClose(sizeA.Z, sizeB.Z, tolerance, 0.001f)) return false;
            return true;
        }

        private static bool IsClose(float valA, float valB, float tolerance, float minAbsForRelative = 0.001f)
        {
             float diff = Math.Abs(valA - valB);
             // Use absolute tolerance for very small values
             if (Math.Abs(valA) < minAbsForRelative && Math.Abs(valB) < minAbsForRelative)
             {
                 return diff < minAbsForRelative * tolerance; // Scale absolute tolerance
             }
             // Use relative tolerance otherwise
            float avg = (Math.Abs(valA) + Math.Abs(valB)) / 2.0f;
            if (avg == 0) return true; // Both are zero
            return (diff / avg) <= tolerance;
        }

        /// <summary>
        /// Creates a new MeshData object with vertices transformed by the given position, rotation, and scale.
        /// </summary>
        public static MeshData TransformMeshData(MeshData mesh, Vector3 position, Quaternion rotation, float scale)
        {
            if (mesh == null || !mesh.IsValid())
            {
                return new MeshData(); // Return new empty mesh data
            }

            var transformedVertices = new List<Vector3>(mesh.Vertices.Count);

            // Create the transformation matrix
            // Scale -> Rotate -> Translate
            Matrix4x4 scaleMatrix = Matrix4x4.CreateScale(scale);
            Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(rotation);
            Matrix4x4 translationMatrix = Matrix4x4.CreateTranslation(position);

            // Combine transformations
            Matrix4x4 transformMatrix = scaleMatrix * rotationMatrix * translationMatrix;
            // Note: The order might depend on specific coordinate system conventions.
            // If M2/WMO local coords are different, adjust multiplication order.
            // Example: Matrix4x4.CreateScale(scale) * Matrix4x4.CreateFromQuaternion(rotation) * Matrix4x4.CreateTranslation(position);

            foreach (var vertex in mesh.Vertices)
            {
                transformedVertices.Add(Vector3.Transform(vertex, transformMatrix));
            }

            // Return new MeshData with transformed vertices and original indices
            return new MeshData
            {
                Vertices = transformedVertices,
                Indices = new List<int>(mesh.Indices) // Copy indices
            };
        }
    }

     // Extension method for MeshData validity check
    public static class MeshDataExtensions
    {
        public static bool IsValid(this MeshData meshData)
        {
            return meshData != null 
                && meshData.Vertices != null 
                && meshData.Indices != null 
                && meshData.Vertices.Count > 0 
                && meshData.Indices.Count > 0 
                && meshData.Indices.Count % 3 == 0;
        }
    }
} 