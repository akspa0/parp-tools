using System.Numerics;

namespace WoWViewer.Terrain;

public readonly record struct WmoMeshSummary(
    int Version,
    int GroupCount,
    int VertexCount,
    int IndexCount,
    int TriangleCount,
    int BatchCount,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3[] FootprintSampleVertices,
    WmoGroupMeshSummary[] GroupSummaries)
{
    public int FootprintSampleCount => FootprintSampleVertices?.Length ?? 0;
}

public readonly record struct WmoGroupMeshSummary(
    int GroupIndex,
    int VertexCount,
    int IndexCount,
    int TriangleCount,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3[] FootprintSampleVertices)
{
    public int FootprintSampleCount => FootprintSampleVertices?.Length ?? 0;
}
