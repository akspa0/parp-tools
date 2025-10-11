using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.Models
{
    /// <summary>
    /// Represents basic mesh geometry with vertices and indices.
    /// Uses integer indices to support larger index ranges.
    /// </summary>
    public class MeshData
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<int> Indices { get; set; } = new List<int>();

        // Optional: Constructor for convenience
        public MeshData() { }

        public MeshData(List<Vector3> vertices, List<int> indices)
        {
            Vertices = vertices;
            Indices = indices;
        }
    }
} 