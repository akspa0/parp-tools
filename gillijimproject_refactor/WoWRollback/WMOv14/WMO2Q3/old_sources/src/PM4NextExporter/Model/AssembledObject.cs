using System.Collections.Generic;
using System.Numerics;

namespace PM4NextExporter.Model
{
    public sealed record AssembledObject(string Name, List<Vector3> Vertices, List<(int A, int B, int C)> Triangles)
    {
        public int VertexCount => Vertices?.Count ?? 0;
        public int TriangleCount => Triangles?.Count ?? 0;
        // Arbitrary metadata for diagnostics and exporters that can ignore unknown fields
        public Dictionary<string, string> Meta { get; } = new();
        // Optional: set of source global vertex indices used to build this object (for MSCN attribution)
        public HashSet<int> SourceGlobalIndices { get; } = new();
    }
}
