using System.Collections.Generic;
using System.Numerics;

namespace PM4NextExporter.Model
{
    public sealed record AssembledObject(string Name, List<Vector3> Vertices, List<(int A, int B, int C)> Triangles)
    {
        public int VertexCount => Vertices?.Count ?? 0;
        public int TriangleCount => Triangles?.Count ?? 0;
        public Dictionary<string, string> Meta { get; } = new();
    }
}
