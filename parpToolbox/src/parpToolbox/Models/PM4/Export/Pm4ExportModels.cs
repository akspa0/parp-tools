using System.Numerics;

namespace parpToolbox.Models.PM4.Export
{
    /// <summary>
    /// Represents structured surface data mapped directly from an MSUR chunk.
    /// </summary>
    public class Pm4SurfaceData
    {
        public uint SurfaceId { get; set; }
        public int MsviFirstIndex { get; set; }
        public int IndexCount { get; set; }
        public uint SurfaceGroupKey { get; set; }
        // Add other relevant fields from MSUR that are needed for export.
    }

    /// <summary>
    /// Represents structured vertex data mapped directly from an MSVT chunk.
    /// </summary>
    public class Pm4VertexData
    {
        public Vector3 Position { get; set; }
        // Add other relevant fields from MSVT like normals and tangents when needed.
    }
}
