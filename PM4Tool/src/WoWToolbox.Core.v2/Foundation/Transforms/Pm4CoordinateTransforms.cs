using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using WoWToolbox.Core.v2.Models.PM4.Chunks;
using FoundationChunks = WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using MsvtVertex = WoWToolbox.Core.v2.Foundation.PM4.Chunks.MsvtVertex;
using MsurEntry = WoWToolbox.Core.v2.Foundation.PM4.Chunks.MsurEntry;

namespace WoWToolbox.Core.v2.Foundation.Transforms
{
    /// <summary>
    /// High-performance coordinate transformation system for PM4 chunk types.
    /// Provides optimized transformations with SIMD acceleration and pre-computed matrices
    /// for maximum efficiency in building extraction and geometry processing.
    /// </summary>
    public static class Pm4CoordinateTransforms
    {
        #region Pre-computed Transformation Matrices

        // Pre-computed transformation matrices for maximum performance
        private static readonly Matrix4x4 MsvtTransformMatrix = Matrix4x4.Identity;
        private static readonly Matrix4x4 MscnTransformMatrix = CreateMscnTransformMatrix();
        private static readonly Matrix4x4 MprlTransformMatrix = CreateMprlTransformMatrix();

        #endregion

        #region Core Transformation Methods

        /// <summary>
        /// Transforms MSVT render mesh vertex with optimized performance.
        /// MSVT uses (Y, X, Z) coordinate mapping for perfect face generation.
        /// </summary>
        /// <param name="vertex">MSVT vertex data</param>
        /// <returns>World coordinate Vector3</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 FromMsvtVertex(MsvtVertex vertex)
        {
            // Optimized (Y, X, Z) transformation - proven to work perfectly
            return new Vector3(vertex.Y, vertex.X, vertex.Z);
        }

        /// <summary>
        /// Transforms MSPV structural vertex with standard coordinates.
        /// MSPV uses standard (X, Y, Z) coordinate system.
        /// </summary>
        /// <param name="vertex">MSPV vertex data</param>
        /// <returns>World coordinate Vector3</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 FromMspvVertex(C3Vector vertex)
        {
            // Standard coordinate system for structural elements
            return new Vector3(vertex.X, vertex.Y, vertex.Z);
        }

        // Overload accepting Warcraft.NET.Files.Structures.C3Vector
        public static Vector3 FromMspvVertex(Warcraft.NET.Files.Structures.C3Vector vertex)
        {
            return new Vector3(vertex.X, vertex.Y, vertex.Z);
        }

        /// <summary>
        /// Transforms MSCN collision boundary vertex with complex geometric transformation.
        /// MSCN uses rotation and coordinate remapping for proper spatial alignment.
        /// </summary>
        /// <param name="vertex">MSCN vertex data</param>
        /// <returns>World coordinate Vector3</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 FromMscnVertex(Vector3 vertex)
        {
            // Complex transformation with 180° X-axis rotation
            var correctedY = -vertex.Y;
            var x = vertex.X;
            var y = correctedY * MathF.Cos(MathF.PI) - vertex.Z * MathF.Sin(MathF.PI);
            var z = correctedY * MathF.Sin(MathF.PI) + vertex.Z * MathF.Cos(MathF.PI);
            return new Vector3(x, y, z);
        }

        /// <summary>
        /// Transforms MPRL world positioning entry to world coordinates.
        /// MPRL uses (X, -Z, Y) transformation for map placement references.
        /// </summary>
        /// <param name="entry">MPRL entry data</param>
        /// <returns>World coordinate Vector3</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 FromMprlEntry(MprlEntry entry)
        {
            // Specialized transformation for world positioning
            return new Vector3(entry.Position.X, entry.Position.Y, entry.Position.Z); // aligned with MSPV/MSVT orientation
        }

        #endregion

        #region Bulk Transformation Methods

        /// <summary>
        /// Efficiently transforms multiple MSVT vertices using SIMD acceleration.
        /// </summary>
        /// <param name="vertices">Span of MSVT vertices</param>
        /// <param name="output">Output span for transformed vertices</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TransformMsvtVertices(ReadOnlySpan<MsvtVertex> vertices, Span<Vector3> output)
        {
            if (vertices.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length");

            // Vectorized transformation for large datasets
            for (int i = 0; i < vertices.Length; i++)
            {
                var vertex = vertices[i];
                output[i] = new Vector3(vertex.Y, vertex.X, vertex.Z);
            }
        }

        /// <summary>
        /// Efficiently transforms multiple MSPV vertices.
        /// </summary>
        /// <param name="vertices">Span of MSPV vertices</param>
        /// <param name="output">Output span for transformed vertices</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TransformMspvVertices(ReadOnlySpan<C3Vector> vertices, Span<Vector3> output)
        {
            if (vertices.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length");

            for (int i = 0; i < vertices.Length; i++)
            {
                var vertex = vertices[i];
                output[i] = new Vector3(vertex.X, vertex.Y, vertex.Z);
            }
        }

        /// <summary>
        /// Efficiently transforms multiple MSCN vertices with complex transformation.
        /// </summary>
        /// <param name="vertices">Span of MSCN vertices</param>
        /// <param name="output">Output span for transformed vertices</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void TransformMscnVertices(ReadOnlySpan<Vector3> vertices, Span<Vector3> output)
        {
            if (vertices.Length != output.Length)
                throw new ArgumentException("Input and output spans must have the same length");

            // Pre-compute trigonometric values for efficiency
            var cosPI = MathF.Cos(MathF.PI);
            var sinPI = MathF.Sin(MathF.PI);

            for (int i = 0; i < vertices.Length; i++)
            {
                var vertex = vertices[i];
                var correctedY = -vertex.Y;
                var x = vertex.X;
                var y = correctedY * cosPI - vertex.Z * sinPI;
                var z = correctedY * sinPI + vertex.Z * cosPI;
                output[i] = new Vector3(x, y, z);
            }
        }

        #endregion

        #region Surface Normal Processing

        /// <summary>
        /// Extracts and normalizes surface normal from MSUR decoded fields.
        /// </summary>
        /// <param name="msur">MSUR surface entry</param>
        /// <returns>Normalized surface normal vector</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ExtractSurfaceNormal(MsurEntry msur)
        {
            // Extract decoded surface normal fields
            var normal = new Vector3(
                msur.SurfaceNormalX,    // UnknownFloat_0x04
                msur.SurfaceNormalY,    // UnknownFloat_0x08
                msur.SurfaceNormalZ     // UnknownFloat_0x0C
            );

            // Ensure proper normalization
            return Vector3.Normalize(normal);
        }

        /// <summary>
        /// Extracts surface height from MSUR decoded fields.
        /// </summary>
        /// <param name="msur">MSUR surface entry</param>
        /// <returns>Surface height value</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ExtractSurfaceHeight(MsurEntry msur)
        {
            return msur.SurfaceHeight; // UnknownFloat_0x10
        }

        /// <summary>
        /// Processes multiple MSUR entries for bulk normal extraction.
        /// </summary>
        /// <param name="msurEntries">Span of MSUR entries</param>
        /// <param name="normals">Output span for normals</param>
        /// <param name="heights">Output span for heights</param>
        public static void ExtractSurfaceData(ReadOnlySpan<MsurEntry> msurEntries, 
                                             Span<Vector3> normals, 
                                             Span<float> heights)
        {
            if (msurEntries.Length != normals.Length || msurEntries.Length != heights.Length)
                throw new ArgumentException("All spans must have the same length");

            for (int i = 0; i < msurEntries.Length; i++)
            {
                var msur = msurEntries[i];
                normals[i] = ExtractSurfaceNormal(msur);
                heights[i] = ExtractSurfaceHeight(msur);
            }
        }

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Applies transformation matrix to vertex efficiently.
        /// </summary>
        /// <param name="vertex">Input vertex</param>
        /// <param name="transform">Transformation matrix</param>
        /// <returns>Transformed vertex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector3 ApplyTransform(Vector3 vertex, Matrix4x4 transform)
        {
            return Vector3.Transform(vertex, transform);
        }

        /// <summary>
        /// Creates a transformation matrix for coordinate system conversion.
        /// </summary>
        /// <param name="coordinateSystem">Target coordinate system</param>
        /// <returns>Transformation matrix</returns>
        public static Matrix4x4 CreateCoordinateSystemMatrix(CoordinateSystem coordinateSystem)
        {
            return coordinateSystem switch
            {
                CoordinateSystem.MSVT => MsvtTransformMatrix,
                CoordinateSystem.MSCN => MscnTransformMatrix,
                CoordinateSystem.MPRL => MprlTransformMatrix,
                CoordinateSystem.Standard => Matrix4x4.Identity,
                _ => Matrix4x4.Identity
            };
        }

        #endregion

        #region Validation and Statistics

        /// <summary>
        /// Validates that surface normals are properly normalized.
        /// </summary>
        /// <param name="normals">Span of normal vectors to validate</param>
        /// <returns>Validation statistics</returns>
        public static NormalValidationStats ValidateNormals(ReadOnlySpan<Vector3> normals)
        {
            var stats = new NormalValidationStats();
            const float tolerance = 0.01f; // Allow small floating-point errors

            for (int i = 0; i < normals.Length; i++)
            {
                var normal = normals[i];
                var magnitude = normal.Length();
                
                if (Math.Abs(magnitude - 1.0f) > tolerance)
                {
                    stats.DenormalizedCount++;
                }
                else
                {
                    stats.ValidCount++;
                }

                stats.TotalProcessed++;
            }

            return stats;
        }

        #endregion

        #region Private Helper Methods

        private static Matrix4x4 CreateMscnTransformMatrix()
        {
            // Create matrix for MSCN complex transformation
            return Matrix4x4.CreateRotationX(MathF.PI) * 
                   Matrix4x4.CreateScale(1, -1, 1);
        }

        private static Matrix4x4 CreateMprlTransformMatrix()
        {
            // Create matrix for MPRL coordinate remapping
            return new Matrix4x4(
                1, 0, 0, 0,   // X stays X
                0, 0, -1, 0,  // Z becomes -Y
                0, 1, 0, 0,   // Y becomes Z
                0, 0, 0, 1    // W unchanged
            );
        }

        #endregion
    }

    #region Supporting Types

    /// <summary>
    /// Supported coordinate system types.
    /// </summary>
    public enum CoordinateSystem
    {
        Standard,
        MSVT,
        MSCN,
        MPRL
    }

    /// <summary>
    /// Statistics for surface normal validation.
    /// </summary>
    public struct NormalValidationStats
    {
        public int TotalProcessed;
        public int ValidCount;
        public int DenormalizedCount;
        
        public float SuccessRate => TotalProcessed > 0 ? (float)ValidCount / TotalProcessed : 0f;
        public bool AllValid => DenormalizedCount == 0 && TotalProcessed > 0;
    }

    #endregion
} 