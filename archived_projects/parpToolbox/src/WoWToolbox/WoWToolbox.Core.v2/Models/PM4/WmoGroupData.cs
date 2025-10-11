using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PM4;

public class WmoGroupData
{
    public int GroupIndex { get; set; }
    public List<Vector3> Vertices { get; set; } = new();
    public List<int> Faces { get; set; } = new();
    public BoundingBox3D BoundingBox { get; set; }
}
