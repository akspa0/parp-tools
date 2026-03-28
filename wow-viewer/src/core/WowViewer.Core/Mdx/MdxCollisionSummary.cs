using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxCollisionSummary
{
    public MdxCollisionSummary(
        int vertexCount,
        int triangleIndexCount,
        int triangleCount,
        int facetNormalCount,
        int maxTriangleIndex,
        Vector3? boundsMin,
        Vector3? boundsMax)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(triangleIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(triangleCount);
        ArgumentOutOfRangeException.ThrowIfNegative(facetNormalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxTriangleIndex);

        if (vertexCount == 0 && (boundsMin is not null || boundsMax is not null))
            throw new ArgumentException("Collision bounds must be null when there are no vertices.", nameof(boundsMin));

        VertexCount = vertexCount;
        TriangleIndexCount = triangleIndexCount;
        TriangleCount = triangleCount;
        FacetNormalCount = facetNormalCount;
        MaxTriangleIndex = maxTriangleIndex;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public int VertexCount { get; }

    public int TriangleIndexCount { get; }

    public int TriangleCount { get; }

    public int FacetNormalCount { get; }

    public int MaxTriangleIndex { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }
}