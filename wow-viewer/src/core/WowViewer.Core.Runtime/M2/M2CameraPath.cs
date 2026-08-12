using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public enum M2CameraPathInterpolation
{
    Linear = 0,
    CatmullRom = 1,
}

public sealed class M2CameraPathDocument
{
    public string Format { get; set; } = "wowviewer-m2-camera-path-v1";
    public string Name { get; set; } = "camera_path";
    public string MapName { get; set; } = "unknown";
    public string BuildVersion { get; set; } = "unknown";
    public M2CameraPathInterpolation Interpolation { get; set; } = M2CameraPathInterpolation.CatmullRom;
    public List<M2CameraPathKeyframe> Keyframes { get; set; } = new();

    public int DurationMs => Keyframes.Count == 0 ? 0 : Math.Max(0, Keyframes[^1].TimeMs);
}

public sealed class M2CameraPathKeyframe
{
    public int TimeMs { get; set; }
    public Vector3 Position { get; set; }
    public Vector3 Target { get; set; }
    public float FovDegrees { get; set; } = 45f;
    public float RollDegrees { get; set; }
}

public readonly record struct M2CameraPathSample(
    Vector3 Position,
    Vector3 Target,
    float FovDegrees,
    float RollDegrees);

public static class M2CameraPathEvaluator
{
    public static M2CameraPathSample Sample(M2CameraPathDocument path, int timeMs, bool loop = false)
    {
        ArgumentNullException.ThrowIfNull(path);
        IReadOnlyList<M2CameraPathKeyframe> keys = path.Keyframes;
        if (keys.Count == 0)
            return new M2CameraPathSample(Vector3.Zero, Vector3.UnitY, 45f, 0f);

        if (keys.Count == 1)
            return ToSample(keys[0]);

        int duration = path.DurationMs;
        int sampleTime = loop && duration > 0 ? PositiveModulo(timeMs, duration) : Math.Clamp(timeMs, 0, duration);
        int right = FindRightKey(keys, sampleTime);
        if (right <= 0)
            return ToSample(keys[0]);
        if (right >= keys.Count)
            return ToSample(keys[^1]);

        M2CameraPathKeyframe leftKey = keys[right - 1];
        M2CameraPathKeyframe rightKey = keys[right];
        int span = Math.Max(1, rightKey.TimeMs - leftKey.TimeMs);
        float factor = Math.Clamp((sampleTime - leftKey.TimeMs) / (float)span, 0f, 1f);

        if (path.Interpolation == M2CameraPathInterpolation.CatmullRom)
        {
            M2CameraPathKeyframe before = right >= 2
                ? keys[right - 2]
                : loop ? keys[^1] : leftKey;
            M2CameraPathKeyframe after = right + 1 < keys.Count
                ? keys[right + 1]
                : loop ? keys[0] : rightKey;
            return new M2CameraPathSample(
                CatmullRom(before.Position, leftKey.Position, rightKey.Position, after.Position, factor),
                CatmullRom(before.Target, leftKey.Target, rightKey.Target, after.Target, factor),
                CatmullRom(before.FovDegrees, leftKey.FovDegrees, rightKey.FovDegrees, after.FovDegrees, factor),
                CatmullRom(before.RollDegrees, leftKey.RollDegrees, rightKey.RollDegrees, after.RollDegrees, factor));
        }

        return new M2CameraPathSample(
            Vector3.Lerp(leftKey.Position, rightKey.Position, factor),
            Vector3.Lerp(leftKey.Target, rightKey.Target, factor),
            Lerp(leftKey.FovDegrees, rightKey.FovDegrees, factor),
            Lerp(leftKey.RollDegrees, rightKey.RollDegrees, factor));
    }

    public static void NormalizeAndValidate(M2CameraPathDocument path)
    {
        ArgumentNullException.ThrowIfNull(path);
        if (path.Keyframes.Count == 0)
            return;

        path.Keyframes.Sort(static (left, right) => left.TimeMs.CompareTo(right.TimeMs));
        for (int index = 0; index < path.Keyframes.Count; index++)
        {
            M2CameraPathKeyframe key = path.Keyframes[index];
            key.TimeMs = Math.Max(0, key.TimeMs);
            key.FovDegrees = float.IsFinite(key.FovDegrees) ? Math.Clamp(key.FovDegrees, 1f, 179f) : 45f;
            key.RollDegrees = float.IsFinite(key.RollDegrees) ? key.RollDegrees : 0f;
            if (!IsFinite(key.Position) || !IsFinite(key.Target))
                throw new InvalidDataException($"Camera path key {index} contains a non-finite position or target.");
        }
    }

    private static int FindRightKey(IReadOnlyList<M2CameraPathKeyframe> keys, int timeMs)
    {
        for (int index = 0; index < keys.Count; index++)
        {
            if (timeMs < keys[index].TimeMs)
                return index;
        }

        return keys.Count;
    }

    private static M2CameraPathSample ToSample(M2CameraPathKeyframe key)
        => new(key.Position, key.Target, key.FovDegrees, key.RollDegrees);

    private static Vector3 CatmullRom(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
        => new(
            CatmullRom(p0.X, p1.X, p2.X, p3.X, t),
            CatmullRom(p0.Y, p1.Y, p2.Y, p3.Y, t),
            CatmullRom(p0.Z, p1.Z, p2.Z, p3.Z, t));

    private static float CatmullRom(float p0, float p1, float p2, float p3, float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return 0.5f * ((2f * p1)
            + (-p0 + p2) * t
            + (2f * p0 - 5f * p1 + 4f * p2 - p3) * t2
            + (-p0 + 3f * p1 - 3f * p2 + p3) * t3);
    }

    private static int PositiveModulo(int value, int modulo)
        => (int)(((long)value % modulo + modulo) % modulo);

    private static float Lerp(float left, float right, float factor)
        => left + ((right - left) * factor);

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}

public static class M2CameraPathImporter
{
    public static M2CameraPathDocument Import(M2ModelDocument model, int cameraIndex = 0, int sequenceIndex = 0, int sampleIntervalMs = 125)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (cameraIndex < 0 || cameraIndex >= model.Cameras.Count)
            throw new ArgumentOutOfRangeException(nameof(cameraIndex));

        M2CameraDefinition camera = model.Cameras[cameraIndex];
        int duration = ResolveDuration(model, camera, sequenceIndex);
        int interval = Math.Clamp(sampleIntervalMs, 16, 1000);
        // M2 sequence durations are exclusive: sampling exactly at Duration wraps
        // back to the first key in the native track sampler.
        int sampleDuration = Math.Max(0, duration - 1);
        int sampleCount = sampleDuration <= 0 ? 1 : Math.Clamp((sampleDuration / interval) + 1, 2, 512);
        List<M2CameraPathKeyframe> keys = new(sampleCount);

        for (int index = 0; index < sampleCount; index++)
        {
            int time = sampleCount == 1 ? 0 : Math.Min(sampleDuration, (int)MathF.Round(sampleDuration * (index / (float)(sampleCount - 1))));
            Vector3 position = SampleVector(model, camera.PositionBase, camera.PositionTrack, sequenceIndex, time);
            Vector3 target = SampleVector(model, camera.TargetPositionBase, camera.TargetPositionTrack, sequenceIndex, time);
            float fov = camera.StaticFieldOfView is float staticFov
                ? staticFov * (180f / MathF.PI)
                : 45f;
            if (camera.FieldOfViewTrack is { } fovTrack && CanSample(model, fovTrack, sequenceIndex))
                fov = M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, time, fovTrack, MathF.PI / 4f) * (180f / MathF.PI);

            float roll = CanSample(model, camera.RollTrack, sequenceIndex)
                ? M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, time, camera.RollTrack, 0f) * (180f / MathF.PI)
                : 0f;
            keys.Add(new M2CameraPathKeyframe { TimeMs = time, Position = position, Target = target, FovDegrees = fov, RollDegrees = roll });
        }

        M2CameraPathDocument result = new()
        {
            Name = string.IsNullOrWhiteSpace(model.ModelName) ? Path.GetFileNameWithoutExtension(model.Identity.CanonicalModelPath) : model.ModelName,
            Keyframes = keys,
        };
        M2CameraPathEvaluator.NormalizeAndValidate(result);
        return result;
    }

    private static int ResolveDuration(M2ModelDocument model, M2CameraDefinition camera, int sequenceIndex)
    {
        if (camera.PositionTrack.UsesGlobalSequence && camera.PositionTrack.GlobalSequenceIndex >= 0 && camera.PositionTrack.GlobalSequenceIndex < model.GlobalLoops.Count)
            return Math.Max(0, (int)model.GlobalLoops[camera.PositionTrack.GlobalSequenceIndex]);
        if (camera.TargetPositionTrack.UsesGlobalSequence && camera.TargetPositionTrack.GlobalSequenceIndex >= 0 && camera.TargetPositionTrack.GlobalSequenceIndex < model.GlobalLoops.Count)
            return Math.Max(0, (int)model.GlobalLoops[camera.TargetPositionTrack.GlobalSequenceIndex]);
        if (sequenceIndex >= 0 && sequenceIndex < model.Sequences.Count)
            return Math.Max(0, (int)model.Sequences[sequenceIndex].Duration);
        return 0;
    }

    private static Vector3 SampleVector(M2ModelDocument model, Vector3 baseValue, M2TrackDefinition<Vector3> track, int sequenceIndex, int time)
        => CanSample(model, track, sequenceIndex)
            ? baseValue + M2TrackSampler.SampleVector3(model.RawBytes, model, sequenceIndex, time, track, Vector3.Zero)
            : baseValue;

    private static bool CanSample<T>(M2ModelDocument model, M2TrackDefinition<T> track, int sequenceIndex)
        => track.TimestampArray.Count > 0 && track.ValueArray.Count > 0
            && (track.UsesGlobalSequence || (sequenceIndex >= 0 && sequenceIndex < model.Sequences.Count));
}
