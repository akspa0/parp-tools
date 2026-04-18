using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxAttachment
{
    public MdxAttachment(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        uint attachmentId,
        string? path,
        Vector3 pivotPoint,
        MdxVector3NodeTrack? translationTrack,
        MdxQuaternionNodeTrack? rotationTrack,
        MdxVector3NodeTrack? scalingTrack,
        MdxScalarTrack? visibilityTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        AttachmentId = attachmentId;
        Path = string.IsNullOrWhiteSpace(path) ? null : path;
        PivotPoint = pivotPoint;
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

    public Vector3 PivotPoint { get; }

    public bool HasParent => ParentId >= 0;

    public bool HasPath => !string.IsNullOrWhiteSpace(Path);

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }

    public MdxScalarTrack? VisibilityTrack { get; }
}
