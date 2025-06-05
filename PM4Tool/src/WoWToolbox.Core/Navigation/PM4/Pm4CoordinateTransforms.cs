using System.Numerics;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using Warcraft.NET.Files.Structures;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Centralizes all coordinate mapping logic for PM4-related chunks.
    /// All transforms produce PM4-relative coordinates, not world coordinates.
    /// Use these methods to convert chunk-local coordinates to consistent PM4-relative coordinate systems.
    /// </summary>
    public static class Pm4CoordinateTransforms
    {
        // DEPRECATED: These constants were used for world transforms but are no longer needed for PM4-relative coordinates
        // Keeping for backward compatibility but should not be used in new code
        [System.Obsolete("CoordinateOffset is deprecated. All transforms should be PM4-relative, not world-relative.")]
        public const float CoordinateOffset = 17066.666f;
        
        [System.Obsolete("ScaleFactor is deprecated. All transforms should be PM4-relative, not world-relative.")]
        public const float ScaleFactor = 36.0f;

        /// <summary>
        /// Converts an MSVT chunk vertex to PM4-relative coordinates.
        /// </summary>
        /// <param name="v">The MSVT vertex.</param>
        /// <returns>PM4-relative coordinates as Vector3 (Y, Z, -X).</returns>
        public static Vector3 FromMsvtVertex(MsvtVertex v)
        {
            // Canonical PM4-relative: (Y, Z, -X) based on test and doc analysis
            // See MSVTChunk.cs for rationale
            return new Vector3(v.Y, v.Z, -v.X);
        }

        /// <summary>
        /// Converts an MSVT chunk vertex to simple PM4-relative coordinates.
        /// Used for anchor points and direct coordinate transformations.
        /// </summary>
        /// <param name="v">The MSVT vertex.</param>
        /// <returns>PM4-relative coordinates as Vector3 (Y, X, Z).</returns>
        public static Vector3 FromMsvtVertexSimple(MsvtVertex v)
        {
            // Simple Y,X,Z transform used for anchor points and node processing
            return new Vector3(v.Y, v.X, v.Z);
        }

        /// <summary>
        /// Converts an MPRL chunk entry to PM4-relative coordinates.
        /// </summary>
        /// <param name="e">The MPRL entry.</param>
        /// <returns>PM4-relative coordinates as Vector3 (-Z, Y, X).</returns>
        public static Vector3 FromMprlEntry(MprlEntry e)
        {
            // PM4-relative transform: Rotate 90 degrees counter-clockwise to align with other chunks
            // Previous transform (X, -Z, Y) was rotated 90 degrees clockwise
            // New transform (-Z, Y, X) should align with MSVT/MSCN/MSPV data
            return new Vector3(-e.Position.Z, e.Position.Y, e.Position.X);
        }

        /// <summary>
        /// Converts an MSCN chunk vertex to PM4-relative coordinates.
        /// </summary>
        /// <param name="v">The MSCN vertex (file order: X, Y, Z).</param>
        /// <returns>PM4-relative coordinates with 180° rotation around X-axis applied to corrected coordinates.</returns>
        public static Vector3 FromMscnVertex(Vector3 v)
        {
            // First apply Y-axis correction that we know works for X/Y positioning
            var corrected = new Vector3(v.X, -v.Y, v.Z);
            
            // Then apply 180-degree rotation around X-axis to "flip up and over"
            // Modified rotation: X unchanged, Y negated, Z preserved (not negated)
            var transformed = new Vector3(
                corrected.X,           // X unchanged
                -corrected.Y,          // Y negated (becomes -(-Y) = Y)
                corrected.Z            // Z preserved (not negated)
            );
            
            return transformed;
        }

        /// <summary>
        /// Converts an MSPV chunk vertex to PM4-relative coordinates.
        /// </summary>
        /// <param name="v">The MSPV vertex (C3Vector).</param>
        /// <returns>PM4-relative coordinates as Vector3 (X, Y, Z).</returns>
        public static Vector3 FromMspvVertex(C3Vector v)
        {
            // PM4-relative: (X, Y, Z) (no transform unless otherwise discovered)
            return new Vector3(v.X, v.Y, v.Z);
        }

        // DEPRECATED METHODS - Kept for backward compatibility but should not be used
        
        [System.Obsolete("FromMsvtVertexToMapProjection is deprecated. Use FromMsvtVertexSimple for PM4-relative coordinates.")]
        public static Vector3 FromMsvtVertexToMapProjection(MsvtVertex v)
        {
            // Deprecated: This applied world offset transforms
            // Use FromMsvtVertexSimple instead for PM4-relative coordinates
            return FromMsvtVertexSimple(v);
        }

        [System.Obsolete("FromMprlEntrySimple is renamed to FromMprlEntry for consistency. Use FromMprlEntry instead.")]
        public static Vector3 FromMprlEntrySimple(MprlEntry e)
        {
            // Deprecated: Renamed for consistency
            return FromMprlEntry(e);
        }

        [System.Obsolete("FromMscnVertexToMapProjection is deprecated. Use FromMscnVertex for PM4-relative coordinates.")]
        public static Vector3 FromMscnVertexToMapProjection(Vector3 v)
        {
            // Deprecated: This applied world offset transforms
            // Use FromMscnVertex instead for PM4-relative coordinates
            return FromMscnVertex(v);
        }

        /// <summary>
        /// Creates a unified point cloud from all aligned PM4 chunks.
        /// All chunks are now properly spatially aligned in the same coordinate system.
        /// </summary>
        /// <param name="pm4File">The PM4 file containing chunk data.</param>
        /// <returns>A dictionary mapping point types to their aligned coordinates.</returns>
        public static Dictionary<string, List<Vector3>> CreateUnifiedPointCloud(PM4File pm4File)
        {
            var pointCloud = new Dictionary<string, List<Vector3>>();

            // MSVT - Render mesh vertices
            if (pm4File.MSVT?.Vertices != null)
            {
                pointCloud["MSVT_Render"] = pm4File.MSVT.Vertices
                    .Select(v => FromMsvtVertexSimple(v))
                    .ToList();
            }

            // MSCN - Collision boundaries
            if (pm4File.MSCN?.ExteriorVertices != null)
            {
                pointCloud["MSCN_Collision"] = pm4File.MSCN.ExteriorVertices
                    .Select(v => FromMscnVertex(v))
                    .ToList();
            }

            // MSPV - Geometric structure
            if (pm4File.MSPV?.Vertices != null)
            {
                pointCloud["MSPV_Geometry"] = pm4File.MSPV.Vertices
                    .Select(v => FromMspvVertex(v))
                    .ToList();
            }

            // MPRL - Reference points (now aligned)
            if (pm4File.MPRL?.Entries != null)
            {
                pointCloud["MPRL_Reference"] = pm4File.MPRL.Entries
                    .Select(e => FromMprlEntry(e))
                    .ToList();
            }

            return pointCloud;
        }

        /// <summary>
        /// Exports a unified point cloud to OBJ format with all chunk types properly aligned.
        /// </summary>
        /// <param name="pm4File">The PM4 file containing chunk data.</param>
        /// <param name="outputPath">The output OBJ file path.</param>
        public static void ExportUnifiedPointCloudToObj(PM4File pm4File, string outputPath)
        {
            var pointCloud = CreateUnifiedPointCloud(pm4File);
            
            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# Unified PM4 Point Cloud - All Chunks Aligned");
            writer.WriteLine($"# Generated: {DateTime.Now}");
            writer.WriteLine($"# File: {pm4File.GetType().Name}");
            writer.WriteLine();

            foreach (var chunkType in pointCloud.Keys)
            {
                writer.WriteLine($"# {chunkType} Points: {pointCloud[chunkType].Count}");
                writer.WriteLine($"o {chunkType}");
                
                foreach (var point in pointCloud[chunkType])
                {
                    writer.WriteLine($"v {point.X:F6} {point.Y:F6} {point.Z:F6}");
                }
                writer.WriteLine();
            }
        }

        /// <summary>
        /// Computes a normal vector from three vertices of a triangle using the cross product.
        /// </summary>
        /// <param name="v1">First vertex of the triangle.</param>
        /// <param name="v2">Second vertex of the triangle.</param>
        /// <param name="v3">Third vertex of the triangle.</param>
        /// <returns>Normalized normal vector pointing outward from the triangle.</returns>
        public static Vector3 ComputeTriangleNormal(Vector3 v1, Vector3 v2, Vector3 v3)
        {
            // Compute two edge vectors
            var edge1 = v2 - v1;
            var edge2 = v3 - v1;
            
            // Compute cross product (right-hand rule for outward-facing normal)
            var normal = Vector3.Cross(edge1, edge2);
            
            // Normalize the result
            var length = normal.Length();
            if (length > 0.0001f) // Avoid division by zero for degenerate triangles
            {
                return normal / length;
            }
            
            // Return up vector for degenerate triangles
            return Vector3.UnitY;
        }

        /// <summary>
        /// Computes per-vertex normals from triangle data using face normal averaging.
        /// </summary>
        /// <param name="vertices">List of vertices.</param>
        /// <param name="triangleIndices">List of triangle indices (groups of 3).</param>
        /// <returns>List of normalized vertex normals.</returns>
        public static List<Vector3> ComputeVertexNormals(List<Vector3> vertices, List<int> triangleIndices)
        {
            var vertexNormals = new Vector3[vertices.Count];
            
            // Accumulate face normals for each vertex
            for (int i = 0; i < triangleIndices.Count; i += 3)
            {
                int idx1 = triangleIndices[i];
                int idx2 = triangleIndices[i + 1];
                int idx3 = triangleIndices[i + 2];
                
                // Validate indices
                if (idx1 >= vertices.Count || idx2 >= vertices.Count || idx3 >= vertices.Count)
                    continue;
                
                var v1 = vertices[idx1];
                var v2 = vertices[idx2];
                var v3 = vertices[idx3];
                
                var faceNormal = ComputeTriangleNormal(v1, v2, v3);
                
                // Add face normal to each vertex of the triangle
                vertexNormals[idx1] += faceNormal;
                vertexNormals[idx2] += faceNormal;
                vertexNormals[idx3] += faceNormal;
            }
            
            // Normalize all vertex normals
            var normalizedNormals = new List<Vector3>();
            for (int i = 0; i < vertexNormals.Length; i++)
            {
                var normal = vertexNormals[i];
                var length = normal.Length();
                if (length > 0.0001f)
                {
                    normalizedNormals.Add(normal / length);
                }
                else
                {
                    normalizedNormals.Add(Vector3.UnitY); // Default up vector
                }
            }
            
            return normalizedNormals;
        }
    }
} 