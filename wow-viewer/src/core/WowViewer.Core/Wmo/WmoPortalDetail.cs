using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoPortalDetail
{
    public WmoPortalDetail(int portalIndex, int startVertexIndex, int vertexCount, IReadOnlyList<Vector3> vertices, Vector3 normal, float planeDistance)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(portalIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(startVertexIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);
        ArgumentNullException.ThrowIfNull(vertices);

        PortalIndex = portalIndex;
        StartVertexIndex = startVertexIndex;
        VertexCount = vertexCount;
        Vertices = vertices;
        Normal = normal;
        PlaneDistance = planeDistance;
    }

    public int PortalIndex { get; }

    public int StartVertexIndex { get; }

    public int VertexCount { get; }

    public IReadOnlyList<Vector3> Vertices { get; }

    public Vector3 Normal { get; }

    public float PlaneDistance { get; }
}