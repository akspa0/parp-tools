using System.Numerics;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using Warcraft.NET.Files.Structures;

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
        /// <returns>PM4-relative coordinates as Vector3 (X, -Z, Y).</returns>
        public static Vector3 FromMprlEntry(MprlEntry e)
        {
            // PM4-relative transform: (X, -Z, Y) without scale/offset
            // Used for PM4-relative coordinate export
            return new Vector3(e.Position.X, -e.Position.Z, e.Position.Y);
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
    }
} 