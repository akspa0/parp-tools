using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxEvent
{
    public MdxEvent(
        int index,
        string name,
        int objectId,
        int parentId,
        uint flags,
        Vector3 pivotPoint,
        MdxVector3NodeTrack? translationTrack,
        MdxQuaternionNodeTrack? rotationTrack,
        MdxVector3NodeTrack? scalingTrack,
        MdxEventTrack? eventTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Index = index;
        Name = name;
        ObjectId = objectId;
        ParentId = parentId;
        Flags = flags;
        PivotPoint = pivotPoint;
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

    public Vector3 PivotPoint { get; }

    public bool HasParent => ParentId >= 0;

    public MdxVector3NodeTrack? TranslationTrack { get; }

    public MdxQuaternionNodeTrack? RotationTrack { get; }

    public MdxVector3NodeTrack? ScalingTrack { get; }

    public MdxEventTrack? EventTrack { get; }
}
