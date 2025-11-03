using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PM4
{
    public class WmoGeometryData
    {
        public string FilePath { get; set; } = string.Empty;
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public BoundingBox3D BoundingBox => BoundingBox3D.FromVertices(Vertices);
    }
}
