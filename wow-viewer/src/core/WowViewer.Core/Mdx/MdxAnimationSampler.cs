using System.Numerics;

namespace WowViewer.Core.Mdx;

public static class MdxAnimationSampler
{
    public static int ResolveSequenceFrame(MdxSummary summary, int sequenceIndex, int timeMs)
    {
        ArgumentNullException.ThrowIfNull(summary);

        if (summary.SequenceCount == 0)
            return Math.Max(0, timeMs);

        int resolvedSequenceIndex = Math.Clamp(sequenceIndex, 0, summary.SequenceCount - 1);
        MdxSequenceSummary sequence = summary.Sequences[resolvedSequenceIndex];
        int duration = Math.Max(0, sequence.Duration);
        if (duration == 0)
            return sequence.StartTime;

        int wrappedTime = Math.Max(0, timeMs) % duration;
        return sequence.StartTime + wrappedTime;
    }

    public static float SampleScalarTrack(MdxScalarTrack? track, MdxSummary summary, int sequenceIndex, int timeMs, float defaultValue)
    {
        if (track is null || track.KeyCount == 0)
            return defaultValue;

        int frameTime = ResolveTrackFrame(summary, sequenceIndex, timeMs, track.GlobalSequenceId);
        return SampleScalarTrack(track, frameTime, defaultValue);
    }

    public static Vector3 SampleColorTrack(MdxColorTrack? track, MdxSummary summary, int sequenceIndex, int timeMs, Vector3 defaultValue)
    {
        if (track is null || track.KeyCount == 0)
            return defaultValue;

        int frameTime = ResolveTrackFrame(summary, sequenceIndex, timeMs, track.GlobalSequenceId);
        return SampleColorTrack(track, frameTime, defaultValue);
    }

    public static Vector3 SampleVector3Track(MdxVector3NodeTrack? track, MdxSummary summary, int sequenceIndex, int timeMs, Vector3 defaultValue)
    {
        if (track is null || track.KeyCount == 0)
            return defaultValue;

        int frameTime = ResolveTrackFrame(summary, sequenceIndex, timeMs, track.GlobalSequenceId);
        return SampleVector3Track(track, frameTime, defaultValue);
    }

    public static int SampleIntTrack(MdxIntTrack? track, MdxSummary summary, int sequenceIndex, int timeMs, int defaultValue)
    {
        if (track is null || track.KeyCount == 0)
            return defaultValue;

        int frameTime = ResolveTrackFrame(summary, sequenceIndex, timeMs, track.GlobalSequenceId);
        return SampleIntTrack(track, frameTime, defaultValue);
    }

    public static Quaternion SampleQuaternionTrack(MdxQuaternionNodeTrack? track, MdxSummary summary, int sequenceIndex, int timeMs, Quaternion defaultValue)
    {
        if (track is null || track.KeyCount == 0)
            return defaultValue;

        int frameTime = ResolveTrackFrame(summary, sequenceIndex, timeMs, track.GlobalSequenceId);
        return SampleQuaternionTrack(track, frameTime, defaultValue);
    }

    public static float SampleScalarTrack(MdxScalarTrack track, int frameTime, float defaultValue)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (!TryFindKeyframePair(track.Keys, frameTime, out MdxScalarKeyframe? left, out MdxScalarKeyframe? right) || left is null)
            return defaultValue;

        if (right is null || left.Time == right.Time)
            return left.Value;

        float t = ComputeInterpolationFactor(left.Time, right.Time, frameTime);
        return track.InterpolationType switch
        {
            MdxTrackInterpolationType.None => left.Value,
            MdxTrackInterpolationType.Hermite => Hermite(
                left.Value,
                left.OutTangent ?? left.Value,
                right.InTangent ?? right.Value,
                right.Value,
                t),
            MdxTrackInterpolationType.Bezier => Bezier(
                left.Value,
                left.OutTangent ?? left.Value,
                right.InTangent ?? right.Value,
                right.Value,
                t),
            _ => Lerp(left.Value, right.Value, t),
        };
    }

    public static Vector3 SampleColorTrack(MdxColorTrack track, int frameTime, Vector3 defaultValue)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (!TryFindKeyframePair(track.Keys, frameTime, out MdxColorKeyframe? left, out MdxColorKeyframe? right) || left is null)
            return defaultValue;

        if (right is null || left.Time == right.Time)
            return left.Value;

        float t = ComputeInterpolationFactor(left.Time, right.Time, frameTime);
        return track.InterpolationType switch
        {
            MdxTrackInterpolationType.None => left.Value,
            MdxTrackInterpolationType.Hermite => new Vector3(
                Hermite(left.Value.X, left.OutTangent?.X ?? left.Value.X, right.InTangent?.X ?? right.Value.X, right.Value.X, t),
                Hermite(left.Value.Y, left.OutTangent?.Y ?? left.Value.Y, right.InTangent?.Y ?? right.Value.Y, right.Value.Y, t),
                Hermite(left.Value.Z, left.OutTangent?.Z ?? left.Value.Z, right.InTangent?.Z ?? right.Value.Z, right.Value.Z, t)),
            MdxTrackInterpolationType.Bezier => new Vector3(
                Bezier(left.Value.X, left.OutTangent?.X ?? left.Value.X, right.InTangent?.X ?? right.Value.X, right.Value.X, t),
                Bezier(left.Value.Y, left.OutTangent?.Y ?? left.Value.Y, right.InTangent?.Y ?? right.Value.Y, right.Value.Y, t),
                Bezier(left.Value.Z, left.OutTangent?.Z ?? left.Value.Z, right.InTangent?.Z ?? right.Value.Z, right.Value.Z, t)),
            _ => Vector3.Lerp(left.Value, right.Value, t),
        };
    }

    public static Vector3 SampleVector3Track(MdxVector3NodeTrack track, int frameTime, Vector3 defaultValue)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (!TryFindKeyframePair(track.Keys, frameTime, out MdxVector3Keyframe? left, out MdxVector3Keyframe? right) || left is null)
            return defaultValue;

        if (right is null || left.Time == right.Time)
            return left.Value;

        float t = ComputeInterpolationFactor(left.Time, right.Time, frameTime);
        return track.InterpolationType switch
        {
            MdxTrackInterpolationType.None => left.Value,
            MdxTrackInterpolationType.Hermite => new Vector3(
                Hermite(left.Value.X, left.OutTangent?.X ?? left.Value.X, right.InTangent?.X ?? right.Value.X, right.Value.X, t),
                Hermite(left.Value.Y, left.OutTangent?.Y ?? left.Value.Y, right.InTangent?.Y ?? right.Value.Y, right.Value.Y, t),
                Hermite(left.Value.Z, left.OutTangent?.Z ?? left.Value.Z, right.InTangent?.Z ?? right.Value.Z, right.Value.Z, t)),
            MdxTrackInterpolationType.Bezier => new Vector3(
                Bezier(left.Value.X, left.OutTangent?.X ?? left.Value.X, right.InTangent?.X ?? right.Value.X, right.Value.X, t),
                Bezier(left.Value.Y, left.OutTangent?.Y ?? left.Value.Y, right.InTangent?.Y ?? right.Value.Y, right.Value.Y, t),
                Bezier(left.Value.Z, left.OutTangent?.Z ?? left.Value.Z, right.InTangent?.Z ?? right.Value.Z, right.Value.Z, t)),
            _ => Vector3.Lerp(left.Value, right.Value, t),
        };
    }

    public static Quaternion SampleQuaternionTrack(MdxQuaternionNodeTrack track, int frameTime, Quaternion defaultValue)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (!TryFindKeyframePair(track.Keys, frameTime, out MdxQuaternionKeyframe? left, out MdxQuaternionKeyframe? right) || left is null)
            return defaultValue;

        if (right is null || left.Time == right.Time)
            return Quaternion.Normalize(left.Value);

        float t = ComputeInterpolationFactor(left.Time, right.Time, frameTime);
        Quaternion interpolated = Quaternion.Slerp(left.Value, right.Value, t);
        return Quaternion.Normalize(interpolated);
    }

    public static int SampleIntTrack(MdxIntTrack track, int frameTime, int defaultValue)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (!TryFindKeyframePair(track.Keys, frameTime, out MdxIntKeyframe? left, out MdxIntKeyframe? right) || left is null)
            return defaultValue;

        if (right is null || left.Time == right.Time)
            return left.Value;

        float t = ComputeInterpolationFactor(left.Time, right.Time, frameTime);
        return track.InterpolationType switch
        {
            MdxTrackInterpolationType.None => left.Value,
            MdxTrackInterpolationType.Hermite => (int)MathF.Round(Hermite(
                left.Value,
                left.OutTangent ?? left.Value,
                right.InTangent ?? right.Value,
                right.Value,
                t)),
            MdxTrackInterpolationType.Bezier => (int)MathF.Round(Bezier(
                left.Value,
                left.OutTangent ?? left.Value,
                right.InTangent ?? right.Value,
                right.Value,
                t)),
            _ => (int)MathF.Round(Lerp(left.Value, right.Value, t)),
        };
    }

    private static int ResolveTrackFrame(MdxSummary summary, int sequenceIndex, int timeMs, int globalSequenceId)
    {
        ArgumentNullException.ThrowIfNull(summary);

        if (globalSequenceId >= 0 && globalSequenceId < summary.GlobalSequenceCount)
        {
            uint duration = summary.GlobalSequences[globalSequenceId].Duration;
            if (duration == 0)
                return 0;

            return (int)(Math.Max(0, timeMs) % duration);
        }

        return ResolveSequenceFrame(summary, sequenceIndex, timeMs);
    }

    private static bool TryFindKeyframePair<T>(IReadOnlyList<T> keys, int frameTime, out T? left, out T? right)
        where T : class
    {
        left = null;
        right = null;
        if (keys.Count == 0)
            return false;

        int firstGreaterIndex = keys.Count;
        for (int index = 0; index < keys.Count; index++)
        {
            int time = GetKeyframeTime(keys[index]);
            if (time > frameTime)
            {
                firstGreaterIndex = index;
                break;
            }
        }

        if (firstGreaterIndex == 0)
        {
            left = keys[0];
            right = keys[0];
            return true;
        }

        if (firstGreaterIndex >= keys.Count)
        {
            left = keys[^1];
            right = keys[^1];
            return true;
        }

        left = keys[firstGreaterIndex - 1];
        right = keys[firstGreaterIndex];
        return true;
    }

    private static int GetKeyframeTime<T>(T keyframe)
        where T : class
    {
        return keyframe switch
        {
            MdxScalarKeyframe scalar => scalar.Time,
            MdxColorKeyframe color => color.Time,
            MdxVector3Keyframe vector => vector.Time,
            MdxQuaternionKeyframe quaternion => quaternion.Time,
            MdxIntKeyframe integer => integer.Time,
            _ => throw new InvalidOperationException($"Unsupported MDX keyframe type '{typeof(T).FullName}'."),
        };
    }

    private static float ComputeInterpolationFactor(int leftTime, int rightTime, int frameTime)
    {
        if (rightTime <= leftTime)
            return 0.0f;

        return Math.Clamp((frameTime - leftTime) / (float)(rightTime - leftTime), 0.0f, 1.0f);
    }

    private static float Lerp(float left, float right, float t) => left + ((right - left) * t);

    private static float Hermite(float a, float aOutTan, float bInTan, float b, float t)
    {
        float t2 = t * t;
        float f1 = t2 * (2 * t - 3) + 1;
        float f2 = t2 * (t - 2) + t;
        float f3 = t2 * (t - 1);
        float f4 = t2 * (3 - 2 * t);
        return (a * f1) + (aOutTan * f2) + (bInTan * f3) + (b * f4);
    }

    private static float Bezier(float a, float aOutTan, float bInTan, float b, float t)
    {
        float inv = 1 - t;
        float inv2 = inv * inv;
        float t2 = t * t;
        return (a * (inv2 * inv)) + (aOutTan * (3 * t * inv2)) + (bInTan * (3 * t2 * inv)) + (b * (t2 * t));
    }
}
