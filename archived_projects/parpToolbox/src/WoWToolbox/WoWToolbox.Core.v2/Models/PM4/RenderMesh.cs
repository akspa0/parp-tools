using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Models.PM4
{
    /// <summary>
    /// Represents a simple, render-ready mesh with vertices, normals, and faces.
    /// </summary>
    public class RenderMesh
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Vector3> Normals { get; set; } = new List<Vector3>();
        public List<int> Faces { get; set; } = new List<int>();
    }
}
