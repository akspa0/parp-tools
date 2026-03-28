using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxGeosetSummary
{
    public MdxGeosetSummary(
        int index,
        int vertexCount,
        int normalCount,
        int uvSetCount,
        int primaryUvCount,
        int primitiveTypeCount,
        int faceGroupCount,
        int indexCount,
        int vertexGroupCount,
        int matrixGroupCount,
        int matrixIndexCount,
        int boneIndexCount,
        int boneWeightCount,
        int materialId,
        uint selectionGroup,
        uint flags,
        float? boundsRadius,
        Vector3? boundsMin,
        Vector3? boundsMax,
        int animationExtentCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(normalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(uvSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(primaryUvCount);
        ArgumentOutOfRangeException.ThrowIfNegative(primitiveTypeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(faceGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(indexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(vertexGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(matrixGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(matrixIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(boneIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(boneWeightCount);
        ArgumentOutOfRangeException.ThrowIfNegative(animationExtentCount);

        Index = index;
        VertexCount = vertexCount;
        NormalCount = normalCount;
        UvSetCount = uvSetCount;
        PrimaryUvCount = primaryUvCount;
        PrimitiveTypeCount = primitiveTypeCount;
        FaceGroupCount = faceGroupCount;
        IndexCount = indexCount;
        VertexGroupCount = vertexGroupCount;
        MatrixGroupCount = matrixGroupCount;
        MatrixIndexCount = matrixIndexCount;
        BoneIndexCount = boneIndexCount;
        BoneWeightCount = boneWeightCount;
        MaterialId = materialId;
        SelectionGroup = selectionGroup;
        Flags = flags;
        BoundsRadius = boundsRadius;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        AnimationExtentCount = animationExtentCount;
    }

    public int Index { get; }

    public int VertexCount { get; }

    public int NormalCount { get; }

    public int UvSetCount { get; }

    public int PrimaryUvCount { get; }

    public int PrimitiveTypeCount { get; }

    public int FaceGroupCount { get; }

    public int IndexCount { get; }

    public int TriangleCount => IndexCount / 3;

    public int VertexGroupCount { get; }

    public int MatrixGroupCount { get; }

    public int MatrixIndexCount { get; }

    public int BoneIndexCount { get; }

    public int BoneWeightCount { get; }

    public int MaterialId { get; }

    public uint SelectionGroup { get; }

    public uint Flags { get; }

    public float? BoundsRadius { get; }

    public Vector3? BoundsMin { get; }

    public Vector3? BoundsMax { get; }

    public int AnimationExtentCount { get; }
}