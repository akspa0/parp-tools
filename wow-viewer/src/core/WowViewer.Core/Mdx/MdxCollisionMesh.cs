using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxCollisionMesh
{
    public MdxCollisionMesh(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<ushort> triangleIndices,
        IReadOnlyList<Vector3> facetNormals,
        int maxTriangleIndex,
        Vector3? boundsMin,
        Vector3? boundsMax)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(triangleIndices);
        ArgumentNullException.ThrowIfNull(facetNormals);
        ArgumentOutOfRangeException.ThrowIfNegative(maxTriangleIndex);

        if (triangleIndices.Count % 3 != 0)
            throw new ArgumentException("Collision triangle index count must be divisible by 3.", nameof(triangleIndices));

        if (vertices.Count == 0 && triangleIndices.Count > 0)
            throw new ArgumentException("Collision triangle indices require at least one vertex.", nameof(triangleIndices));

        if (triangleIndices.Count > 0 && maxTriangleIndex >= vertices.Count)
            throw new ArgumentOutOfRangeException(nameof(maxTriangleIndex), "Collision max triangle index exceeded the vertex count.");

        if (vertices.Count == 0 && (boundsMin is not null || boundsMax is not null))
            throw new ArgumentException("Collision bounds must be null when there are no vertices.", nameof(boundsMin));

        Vertices = vertices;
        TriangleIndices = triangleIndices;
        FacetNormals = facetNormals;
        MaxTriangleIndex = maxTriangleIndex;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public IReadOnlyList<Vector3> Vertices { get; }

    public int VertexCount => Vertices.Count;

    public IReadOnlyList<ushort> TriangleIndices { get; }

    public int TriangleIndexCount => TriangleIndices.Count;

    public int TriangleCount => TriangleIndices.Count / 3;

    public IReadOnlyList<Vector3> FacetNormals { get; }

    public int FacetNormalCount => FacetNormals.Count;

    public int MaxTriangleIndex { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }
}