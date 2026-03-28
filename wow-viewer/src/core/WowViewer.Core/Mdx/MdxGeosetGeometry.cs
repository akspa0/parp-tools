using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxGeosetGeometry
{
    public MdxGeosetGeometry(
        int index,
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<Vector3> normals,
        IReadOnlyList<IReadOnlyList<Vector2>> uvSets,
        IReadOnlyList<byte> primitiveTypes,
        IReadOnlyList<int> faceGroups,
        IReadOnlyList<ushort> indices,
        IReadOnlyList<byte> vertexGroups,
        IReadOnlyList<uint> matrixGroups,
        IReadOnlyList<uint> matrixIndices,
        IReadOnlyList<uint> boneIndices,
        IReadOnlyList<uint> boneWeights,
        int materialId,
        uint selectionGroup,
        uint flags,
        float? boundsRadius,
        Vector3? boundsMin,
        Vector3? boundsMax,
        int animationExtentCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(normals);
        ArgumentNullException.ThrowIfNull(uvSets);
        ArgumentNullException.ThrowIfNull(primitiveTypes);
        ArgumentNullException.ThrowIfNull(faceGroups);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(vertexGroups);
        ArgumentNullException.ThrowIfNull(matrixGroups);
        ArgumentNullException.ThrowIfNull(matrixIndices);
        ArgumentNullException.ThrowIfNull(boneIndices);
        ArgumentNullException.ThrowIfNull(boneWeights);
        ArgumentOutOfRangeException.ThrowIfNegative(animationExtentCount);

        Index = index;
        Vertices = vertices;
        Normals = normals;
        UvSets = uvSets;
        PrimitiveTypes = primitiveTypes;
        FaceGroups = faceGroups;
        Indices = indices;
        VertexGroups = vertexGroups;
        MatrixGroups = matrixGroups;
        MatrixIndices = matrixIndices;
        BoneIndices = boneIndices;
        BoneWeights = boneWeights;
        MaterialId = materialId;
        SelectionGroup = selectionGroup;
        Flags = flags;
        BoundsRadius = boundsRadius;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        AnimationExtentCount = animationExtentCount;
    }

    public int Index { get; }

    public IReadOnlyList<Vector3> Vertices { get; }

    public int VertexCount => Vertices.Count;

    public IReadOnlyList<Vector3> Normals { get; }

    public int NormalCount => Normals.Count;

    public IReadOnlyList<IReadOnlyList<Vector2>> UvSets { get; }

    public int UvSetCount => UvSets.Count;

    public IReadOnlyList<Vector2> PrimaryUvSet => UvSets.Count == 0 ? [] : UvSets[0];

    public int PrimaryUvCount => PrimaryUvSet.Count;

    public IReadOnlyList<byte> PrimitiveTypes { get; }

    public int PrimitiveTypeCount => PrimitiveTypes.Count;

    public IReadOnlyList<int> FaceGroups { get; }

    public int FaceGroupCount => FaceGroups.Count;

    public IReadOnlyList<ushort> Indices { get; }

    public int IndexCount => Indices.Count;

    public int TriangleCount => Indices.Count / 3;

    public IReadOnlyList<byte> VertexGroups { get; }

    public int VertexGroupCount => VertexGroups.Count;

    public IReadOnlyList<uint> MatrixGroups { get; }

    public int MatrixGroupCount => MatrixGroups.Count;

    public IReadOnlyList<uint> MatrixIndices { get; }

    public int MatrixIndexCount => MatrixIndices.Count;

    public IReadOnlyList<uint> BoneIndices { get; }

    public int BoneIndexCount => BoneIndices.Count;

    public IReadOnlyList<uint> BoneWeights { get; }

    public int BoneWeightCount => BoneWeights.Count;

    public int MaterialId { get; }

    public uint SelectionGroup { get; }

    public uint Flags { get; }

    public float? BoundsRadius { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }

    public int AnimationExtentCount { get; }
}