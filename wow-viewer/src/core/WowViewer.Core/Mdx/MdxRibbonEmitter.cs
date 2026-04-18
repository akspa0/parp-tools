using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxRibbonEmitter
{
    public MdxRibbonEmitter(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        Vector3 pivotPoint,
        float staticHeightAbove,
        float staticHeightBelow,
        float staticAlpha,
        Vector3 staticColor,
        float edgeLifetime,
        uint staticTextureSlot,
        uint edgesPerSecond,
        uint textureRows,
        uint textureColumns,
        uint materialId,
        float gravity,
        MdxVector3NodeTrack? translationTrack,
        MdxQuaternionNodeTrack? rotationTrack,
        MdxVector3NodeTrack? scalingTrack,
        MdxScalarTrack? heightAboveTrack,
        MdxScalarTrack? heightBelowTrack,
        MdxScalarTrack? alphaTrack,
        MdxColorTrack? colorTrack,
        MdxIntTrack? textureSlotTrack,
        MdxScalarTrack? visibilityTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        PivotPoint = pivotPoint;
        StaticHeightAbove = staticHeightAbove;
        StaticHeightBelow = staticHeightBelow;
        StaticAlpha = staticAlpha;
        StaticColor = staticColor;
        EdgeLifetime = edgeLifetime;
        StaticTextureSlot = staticTextureSlot;
        EdgesPerSecond = edgesPerSecond;
        TextureRows = textureRows;
        TextureColumns = textureColumns;
        MaterialId = materialId;
        Gravity = gravity;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        HeightAboveTrack = heightAboveTrack;
        HeightBelowTrack = heightBelowTrack;
        AlphaTrack = alphaTrack;
        ColorTrack = colorTrack;
        TextureSlotTrack = textureSlotTrack;
        VisibilityTrack = visibilityTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public Vector3 PivotPoint { get; }

    public float StaticHeightAbove { get; }

    public float StaticHeightBelow { get; }

    public float StaticAlpha { get; }

    public Vector3 StaticColor { get; }

    public float EdgeLifetime { get; }

    public uint StaticTextureSlot { get; }

    public uint EdgesPerSecond { get; }

    public uint TextureRows { get; }

    public uint TextureColumns { get; }

    public uint MaterialId { get; }

    public float Gravity { get; }

    public bool HasParent => ParentId >= 0;

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }

    public MdxScalarTrack? HeightAboveTrack { get; }

    public MdxScalarTrack? HeightBelowTrack { get; }

    public MdxScalarTrack? AlphaTrack { get; }

    public MdxColorTrack? ColorTrack { get; }

    public MdxIntTrack? TextureSlotTrack { get; }

    public MdxScalarTrack? VisibilityTrack { get; }
}
