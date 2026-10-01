using System.Numerics;

namespace WoWViewer.Terrain;

public readonly record struct MdxCollisionMeshSummary(
    int VertexCount,
    int TriangleIndexCount,
    int TriangleCount,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3[] FootprintSampleVertices)
{
    public int FootprintSampleCount => FootprintSampleVertices?.Length ?? 0;
}
