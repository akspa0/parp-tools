namespace WowViewer.Core.Mdx;

public sealed class MdxBoneSummary
{
    public MdxBoneSummary(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        uint geosetId,
        uint geosetAnimationId,
        MdxNodeTrackSummary? translationTrack,
        MdxNodeTrackSummary? rotationTrack,
        MdxNodeTrackSummary? scalingTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(objectId);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        GeosetId = geosetId;
        GeosetAnimationId = geosetAnimationId;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public uint GeosetId { get; }

    public uint GeosetAnimationId { get; }

    public bool HasParent => ParentId >= 0;

    public bool UsesGeoset => GeosetId != uint.MaxValue;

    public bool UsesGeosetAnimation => GeosetAnimationId != uint.MaxValue;

    public MdxNodeTrackSummary? TranslationTrack { get; }

    public MdxNodeTrackSummary? RotationTrack { get; }

    public MdxNodeTrackSummary? ScalingTrack { get; }
}