using System.Numerics;

namespace WowViewer.Core.Mdx;

public sealed class MdxCameraSummary
{
    public MdxCameraSummary(
        int index,
        string name,
        Vector3 pivotPoint,
        float fieldOfView,
        float farClip,
        float nearClip,
        Vector3 targetPivotPoint,
        MdxTrackSummary? positionTrack,
        MdxTrackSummary? rollTrack,
        MdxVisibilityTrackSummary? visibilityTrack,
        MdxTrackSummary? targetPositionTrack)
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

    public MdxTrackSummary? PositionTrack { get; }

    public MdxTrackSummary? RollTrack { get; }

    public MdxVisibilityTrackSummary? VisibilityTrack { get; }

    public MdxTrackSummary? TargetPositionTrack { get; }
}