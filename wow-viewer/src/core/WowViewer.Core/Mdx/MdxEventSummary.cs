namespace WowViewer.Core.Mdx;

public sealed class MdxEventSummary
{
    public MdxEventSummary(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        MdxNodeTrackSummary? translationTrack,
        MdxNodeTrackSummary? rotationTrack,
        MdxNodeTrackSummary? scalingTrack,
        MdxEventTrackSummary? eventTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(objectId);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        EventTrack = eventTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public bool HasParent => ParentId >= 0;

    public MdxNodeTrackSummary? TranslationTrack { get; }

    public MdxNodeTrackSummary? RotationTrack { get; }

    public MdxNodeTrackSummary? ScalingTrack { get; }

    public MdxEventTrackSummary? EventTrack { get; }
}