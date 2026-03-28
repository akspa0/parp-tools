using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxRibbonEmitterSummary
{
    public MdxRibbonEmitterSummary(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
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
        MdxNodeTrackSummary? translationTrack,
        MdxNodeTrackSummary? rotationTrack,
        MdxNodeTrackSummary? scalingTrack,
        MdxTrackSummary? heightAboveTrack,
        MdxTrackSummary? heightBelowTrack,
        MdxTrackSummary? alphaTrack,
        MdxTrackSummary? colorTrack,
        MdxTrackSummary? textureSlotTrack,
        MdxVisibilityTrackSummary? visibilityTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(objectId);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
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

    public MdxNodeTrackSummary? TranslationTrack { get; }

    public MdxNodeTrackSummary? RotationTrack { get; }

    public MdxNodeTrackSummary? ScalingTrack { get; }

    public MdxTrackSummary? HeightAboveTrack { get; }

    public MdxTrackSummary? HeightBelowTrack { get; }

    public MdxTrackSummary? AlphaTrack { get; }

    public MdxTrackSummary? ColorTrack { get; }

    public MdxTrackSummary? TextureSlotTrack { get; }

    public MdxVisibilityTrackSummary? VisibilityTrack { get; }
}