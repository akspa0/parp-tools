namespace WowViewer.Core.Mdx;

public sealed class MdxAttachmentSummary
{
    public MdxAttachmentSummary(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        uint attachmentId,
        string? path,
        MdxNodeTrackSummary? translationTrack,
        MdxNodeTrackSummary? rotationTrack,
        MdxNodeTrackSummary? scalingTrack,
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
        AttachmentId = attachmentId;
        Path = string.IsNullOrWhiteSpace(path) ? null : path;
        TranslationTrack = translationTrack;
        RotationTrack = rotationTrack;
        ScalingTrack = scalingTrack;
        VisibilityTrack = visibilityTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public int ObjectId { get; }

    public int ParentId { get; }

    public uint Flags { get; }

    public uint AttachmentId { get; }

    public string? Path { get; }

    public bool HasParent => ParentId >= 0;

    public bool HasPath => !string.IsNullOrWhiteSpace(Path);

    public MdxNodeTrackSummary? TranslationTrack { get; }

    public MdxNodeTrackSummary? RotationTrack { get; }

    public MdxNodeTrackSummary? ScalingTrack { get; }

    public MdxVisibilityTrackSummary? VisibilityTrack { get; }
}