using System.Numerics;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using Warcraft.NET.Files.Structures;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Centralizes all coordinate mapping logic for PM4-related chunks.
    /// Use these methods to convert chunk-local coordinates to canonical world coordinates.
    /// </summary>
    public static class Pm4CoordinateTransforms
    {
        /// <summary>
        /// The canonical coordinate offset for PM4 world transforms.
        /// </summary>
        public const float CoordinateOffset = 17066.666f;
        /// <summary>
        /// The canonical scale factor for PM4 height transforms.
        /// </summary>
        public const float ScaleFactor = 36.0f;

        /// <summary>
        /// Converts an MSVT chunk vertex to canonical world coordinates.
        /// </summary>
        /// <param name="v">The MSVT vertex.</param>
        /// <returns>World coordinates as Vector3.</returns>
        public static Vector3 FromMsvtVertex(MsvtVertex v)
        {
            // Canonical: (Y, Z, -X) based on test and doc analysis
            // See MSVTChunk.cs for rationale
            return new Vector3(v.Y, v.Z, -v.X);
        }

        /// <summary>
        /// Converts an MSVT chunk vertex to map projection coordinates (used for OBJ export).
        /// </summary>
        /// <param name="v">The MSVT vertex.</param>
        /// <returns>Map projection coordinates as Vector3.</returns>
        public static Vector3 FromMsvtVertexToMapProjection(MsvtVertex v)
        {
            // Map projection: (CoordinateOffset - X, CoordinateOffset - Y, Z)
            // Used in PM4FileTests and PD4FileTests for OBJ export
            return new Vector3(CoordinateOffset - v.X, CoordinateOffset - v.Y, v.Z);
        }

        /// <summary>
        /// Converts an MPRL chunk entry to canonical world coordinates.
        /// </summary>
        /// <param name="e">The MPRL entry.</param>
        /// <returns>World coordinates as Vector3.</returns>
        public static Vector3 FromMprlEntry(MprlEntry e)
        {
            // Canonical: (X, -Z, Y) with scale/offset, as used in tests and exporters
            float x = e.Position.X * ScaleFactor - CoordinateOffset;
            float y = -e.Position.Z * ScaleFactor + CoordinateOffset;
            float z = e.Position.Y * ScaleFactor;
            return new Vector3(x, y, z);
        }

        /// <summary>
        /// Converts an MPRL chunk entry to simple transformed coordinates (used for OBJ export).
        /// This applies the (X, -Z, Y) mapping without scale/offset for raw coordinate output.
        /// </summary>
        /// <param name="e">The MPRL entry.</param>
        /// <returns>Simple transformed coordinates as Vector3.</returns>
        public static Vector3 FromMprlEntrySimple(MprlEntry e)
        {
            // Simple transform: (X, -Z, Y) without scale/offset
            // Used in PM4FileTests for direct OBJ coordinate export
            return new Vector3(e.Position.X, e.Position.Z, e.Position.Y);
        }

        /// <summary>
        /// Converts an MSCN chunk vertex to canonical world coordinates.
        /// </summary>
        /// <param name="v">The MSCN vertex (file order: X, Y, Z).</param>
        /// <returns>World coordinates as Vector3.</returns>
        public static Vector3 FromMscnVertex(Vector3 v)
        {
            // Canonical: (Y, -X, Z) as per MSCNChunk.ToCanonicalWorldCoordinates
            return new Vector3(v.Y, -v.X, v.Z);
        }

        /// <summary>
        /// Converts an MSCN chunk vertex to map projection coordinates (used for OBJ export).
        /// </summary>
        /// <param name="v">The MSCN vertex (file order: X, Y, Z).</param>
        /// <returns>Map projection coordinates as Vector3.</returns>
        public static Vector3 FromMscnVertexToMapProjection(Vector3 v)
        {
            // Map projection: (CoordinateOffset - X, CoordinateOffset - Y, Z)
            // Used in PM4FileTests for MSCN OBJ export
            return new Vector3(CoordinateOffset - v.X, CoordinateOffset - v.Y, v.Z);
        }

        /// <summary>
        /// Converts an MSPV chunk vertex to canonical world coordinates.
        /// </summary>
        /// <param name="v">The MSPV vertex (C3Vector).</param>
        /// <returns>World coordinates as Vector3.</returns>
        public static Vector3 FromMspvVertex(C3Vector v)
        {
            // Canonical: (X, Y, Z) (no transform unless otherwise discovered)
            return new Vector3(v.X, v.Y, v.Z);
        }

        /// <summary>
        /// Converts an MSVT vertex to standard Y,X,Z transformation (without scale/offset).
        /// Used for anchor points and simple coordinate transformations.
        /// </summary>
        /// <param name="v">The MSVT vertex.</param>
        /// <returns>Simple transformed coordinates as Vector3.</returns>
        public static Vector3 FromMsvtVertexSimple(MsvtVertex v)
        {
            // Simple Y,X,Z transform used for anchor points and node processing
            return new Vector3(v.Y, v.X, v.Z);
        }
    }
} 