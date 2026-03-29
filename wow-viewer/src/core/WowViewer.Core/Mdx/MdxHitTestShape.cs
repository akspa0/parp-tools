using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxHitTestShape
{
    public MdxHitTestShape(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        MdxVector3NodeTrack? translationTrack,
        MdxQuaternionNodeTrack? rotationTrack,
        MdxVector3NodeTrack? scalingTrack,
        MdxGeometryShapeType shapeType,
        Vector3? minimum,
        Vector3? maximum,
        Vector3? basePoint,
        float? height,
        float? radius,
        Vector3? center,
        float? length,
        float? width)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        ShapeType = shapeType;
        Minimum = minimum;
        Maximum = maximum;
        BasePoint = basePoint;
        Height = height;
        Radius = radius;
        Center = center;
        Length = length;
        Width = width;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public bool HasParent => ParentId >= 0;

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }

    public MdxGeometryShapeType ShapeType { get; }

    public Vector3? Minimum { get; }

    public Vector3? Maximum { get; }

    public Vector3? BasePoint { get; }

    public float? Height { get; }

    public float? Radius { get; }

    public Vector3? Center { get; }

    public float? Length { get; }

    public float? Width { get; }
}