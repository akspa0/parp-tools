using System.Numerics;

namespace WoWToolbox.Core.Navigation.PM4.Models
{
    /// <summary>
    /// Represents a 3D bounding box for spatial calculations.
    /// Used in building geometry analysis and WMO matching.
    /// </summary>
    public struct BoundingBox3D
    {
        public Vector3 Min { get; set; }
        public Vector3 Max { get; set; }
        public Vector3 Center { get; set; }
        public Vector3 Size { get; set; }
    }
} 