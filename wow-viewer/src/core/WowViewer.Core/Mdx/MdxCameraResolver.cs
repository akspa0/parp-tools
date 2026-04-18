using System.Numerics;

namespace WowViewer.Core.Mdx;

public readonly record struct MdxResolvedCameraState(
    Vector3 Position,
    Vector3 Target,
    Vector3 Up,
    float FieldOfView,
    float NearClip,
    float FarClip,
    bool Visible);

public static class MdxCameraResolver
{
    public static MdxResolvedCameraState Resolve(MdxSummary summary, MdxCamera camera, int sequenceIndex, int timeMs)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentNullException.ThrowIfNull(camera);

        Vector3 position = camera.PivotPoint
            + MdxAnimationSampler.SampleVector3Track(camera.PositionTrack, summary, sequenceIndex, timeMs, Vector3.Zero);
        Vector3 target = camera.TargetPivotPoint
            + MdxAnimationSampler.SampleVector3Track(camera.TargetPositionTrack, summary, sequenceIndex, timeMs, Vector3.Zero);
        float roll = MdxAnimationSampler.SampleScalarTrack(camera.RollTrack, summary, sequenceIndex, timeMs, 0.0f);
        float visibility = MdxAnimationSampler.SampleScalarTrack(camera.VisibilityTrack, summary, sequenceIndex, timeMs, 1.0f);

        return new MdxResolvedCameraState(
            position,
            target,
            ResolveUpVector(position, target, roll),
            camera.FieldOfView,
            camera.NearClip,
            camera.FarClip,
            visibility > 0.001f);
    }

    private static Vector3 ResolveUpVector(Vector3 position, Vector3 target, float rollRadians)
    {
        Vector3 forward = target - position;
        if (forward.LengthSquared() <= 0.000001f)
            return Vector3.UnitZ;

        forward = Vector3.Normalize(forward);
        Vector3 baseUp = MathF.Abs(Vector3.Dot(forward, Vector3.UnitZ)) > 0.99f ? Vector3.UnitX : Vector3.UnitZ;
        if (MathF.Abs(rollRadians) <= 0.000001f)
            return baseUp;

        Quaternion roll = Quaternion.CreateFromAxisAngle(forward, rollRadians);
        Vector3 rotatedUp = Vector3.Transform(baseUp, roll);
        return rotatedUp.LengthSquared() > 0.000001f ? Vector3.Normalize(rotatedUp) : baseUp;
    }
}
