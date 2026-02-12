using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PM4
{
    /// <summary>
    /// Represents a single navigation mesh object ("negative mold" building fragment) extracted from a PM4 file.
    /// </summary>
    public class IndividualNavigationObject
    {
        /// <summary>
        /// Human-readable identifier, e.g. "OBJ_000".
        /// </summary>
        public string ObjectId { get; set; } = string.Empty;

        /// <summary>
        /// Vertices in world-space coordinates.
        /// </summary>
        public List<Vector3> Vertices { get; } = new();

        /// <summary>
        /// Triangle indices (counter-clockwise order).
        /// </summary>
        public List<int> Indices { get; } = new();

        /// <summary>
        /// Bounding box derived from the <see cref="Vertices"/> collection.
        /// </summary>
        public BoundingBox3D BoundingBox => BoundingBox3D.FromVertices(Vertices);

        /// <summary>
        /// Optional link back to the MDOS building id when known.
        /// </summary>
        public uint BuildingId { get; set; }

        /// <summary>
        /// Raw 32-bit LinkId value from MSLK (often 0xFFFFYYXX).
        /// </summary>
        public uint LinkIdRaw { get; set; }

        /// <summary>
        /// Parsed tile X coordinate when <see cref="LinkIdRaw"/> follows the 0xFFFFYYXX pattern.
        /// </summary>
        public int? TileX { get; set; }

        /// <summary>
        /// Parsed tile Y coordinate when <see cref="LinkIdRaw"/> follows the 0xFFFFYYXX pattern.
        /// </summary>
        public int? TileY { get; set; }
    }
}
