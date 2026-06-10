using System.Numerics;

namespace WoWToolbox.Core.Navigation.PM4.Models
{
    /// <summary>
    /// Represents a complete WMO (World Model Object) extracted from PM4 navigation data.
    /// Contains full geometry, normals, texture coordinates, and metadata for 3D rendering.
    /// </summary>
    public class CompleteWMOModel
    {
        public string FileName { get; set; } = "";
        public string Category { get; set; } = "";
        public List<Vector3> Vertices { get; set; } = new();
        public List<int> TriangleIndices { get; set; } = new();
        public List<Vector3> Normals { get; set; } = new();
        public List<Vector2> TexCoords { get; set; } = new();
        public string MaterialName { get; set; } = "WMO_Material";
        public Dictionary<string, object> Metadata { get; set; } = new();
        
        public int VertexCount => Vertices.Count;
        public int FaceCount => TriangleIndices.Count / 3;
    }
} 