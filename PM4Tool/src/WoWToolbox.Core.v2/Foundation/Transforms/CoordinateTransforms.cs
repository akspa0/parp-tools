using System.Numerics;
using Warcraft.NET.Files.Structures;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Foundation.Transforms
{
    /// <summary>
    /// Centralised coordinate conversion helpers for PM4 / PD4 related chunks.
    /// These methods produce coordinates in a consistent <c>Vector3</c> space that the
    /// rest of Core.v2 services can rely upon.  They intentionally avoid any game-world
    /// offset math – callers are expected to apply world transforms if necessary.
    /// </summary>
    public static class CoordinateTransforms
    {
        /// <summary>
        /// Converts an <see cref="MSVT_Vertex"/> to the library-wide coordinate space.
        /// The mapping is (X, Y, Z) –&gt; (X, Z, -Y) which flips WoW Z-up to a more conventional Y-up.
        /// </summary>
        // Canonical PM4-relative transform (Y, Z, -X)
        // New canonical mapping: MSVT file stores vertices already in correct XYZ order.
        // Adjusted: MSVT file stores floats (Y, X, Z). Swap to (X, Y, Z) requires (v.Position.Y, v.Position.X, v.Position.Z)
        // Legacy-parity mapping: (X,Y,Z) -> (Y,Z,-X)
        /// <summary>
        /// Spec-defined transform for MSVT (file order Y,X,Z → world X,Y,Z).
        /// Maps (Y, X, Z) with NO sign flips.
        /// </summary>
        public static Vector3 FromMsvtVertex(MSVT_Vertex v) => new(v.Position.Y, v.Position.X, v.Position.Z);

        /// <summary>
        /// Helper for legacy <see cref="MsvtVertex"/> used by the parsing layer.
        /// </summary>
        // Legacy-parity mapping for placeholder struct
        public static Vector3 FromMsvtVertexSimple(MsvtVertex v) => new(v.Y, v.X, v.Z);

        /// <summary>
        /// Converts an MSVT vertex maintaining XYZ order (legacy OBJ parity):
        /// Raw layout appears to be (X, Y, Z) stored as X,Y,Z → needs swap (Z,Y,-X).
        /// </summary>
        public static Vector3 FromMsvtVertexXYZ(MsvtVertex v) => new(v.Y, v.X, v.Z);

        /// <summary>
        /// Correct legacy mapping for MSVT vertex: (X, Y, Z) -> (X, Y, -Z).
        /// Matches the last known-good commit used for OBJ parity.
        /// </summary>
        public static Vector3 FromMsvtVertexLegacy(MsvtVertex v) => new(v.Y, v.Z, -v.X);

        /// <summary>
        /// Converts an MSCN exterior vertex (already XYZ) – just perform the Z-up ‑&gt; Y-up flip.
        /// </summary>
        public static Vector3 FromMscnVertex(Vector3 v) => new(v.Y, -v.X, v.Z);

        /// <summary>
        /// Converts an MSPV geometry vertex.
        /// No re-arrangement needed, only the axis flip.
        /// </summary>
        public static Vector3 FromMspvVertex(C3Vector v) => new(v.X, v.Y, v.Z);

        /// <summary>
        /// Overload accepting <see cref="Vector3"/> directly (parse layer already converted).
        /// </summary>
        public static Vector3 FromMspvVertex(Vector3 v) => new(v.X, v.Y, v.Z);

        /// <summary>
        /// Overload for placeholder MSPV_Vertex wrapper.
        /// </summary>
        public static Vector3 FromMspvVertex(MSPV_Vertex v) => new(v.Position.X, v.Position.Y, v.Position.Z);

        /// <summary>
        /// Converts an MPRL entry position.
        /// </summary>
        public static Vector3 FromMprlEntry(MprlEntry e) => new(-e.Position.Z, e.Position.Y, e.Position.X); // unchanged

        /// <summary>
        /// Overload for legacy placeholder struct.
        /// </summary>
        public static Vector3 FromMprlEntry(MPRL_Entry e) => new(-e.Position.Z, e.Position.Y, e.Position.X);
    }
}
