using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxCamera
{
    public MdxCamera(
        int index,
        string name,
        Vector3 pivotPoint,
        float fieldOfView,
        float farClip,
        float nearClip,
        Vector3 targetPivotPoint,
        MdxVector3NodeTrack? positionTrack,
        MdxScalarTrack? rollTrack,
        MdxScalarTrack? visibilityTrack,
        MdxVector3NodeTrack? targetPositionTrack)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        Index = index;
        Name = name;
        PivotPoint = pivotPoint;
        FieldOfView = fieldOfView;
        FarClip = farClip;
        NearClip = nearClip;
        TargetPivotPoint = targetPivotPoint;
        PositionTrack = positionTrack;
        RollTrack = rollTrack;
        VisibilityTrack = visibilityTrack;
        TargetPositionTrack = targetPositionTrack;
    }

    public int Index { get; }

    public string Name { get; }

    public Vector3 PivotPoint { get; }

    public float FieldOfView { get; }

    public float FarClip { get; }

    public float NearClip { get; }

    public Vector3 TargetPivotPoint { get; }

    public MdxVector3NodeTrack? PositionTrack { get; }

    public MdxScalarTrack? RollTrack { get; }

    public MdxScalarTrack? VisibilityTrack { get; }

    public MdxVector3NodeTrack? TargetPositionTrack { get; }
}
