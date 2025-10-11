using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PM4
{
    public class CompleteWmoModel
    {
        public uint BuildingId { get; set; }
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<int> Indices { get; set; } = new List<int>();
        public bool HasGeometry => Vertices.Count > 0 && Indices.Count > 0;
    }
}
