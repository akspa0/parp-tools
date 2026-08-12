using System.Numerics;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Runtime.M2;

/// <summary>
/// Converts a classic MDX CAMS track into the viewer's map-bound camera-path document.
/// The MDX camera reader and resolver remain the format authorities; this class only
/// samples their decoded tracks into the reusable path representation.
/// </summary>
public static class MdxCameraPathImporter
{
    public static M2CameraPathDocument Import(
        Stream stream,
        string sourcePath = "<memory>",
        int cameraIndex = 0,
        int sequenceIndex = 0,
        int sampleIntervalMs = 125)
    {
        ArgumentNullException.ThrowIfNull(stream);
        MdxCameraFile cameraFile = MdxCameraReader.Read(stream, sourcePath);
        stream.Position = 0;
        MdxSummary summary = MdxSummaryReader.Read(stream, sourcePath);
        return Import(cameraFile, summary, cameraIndex, sequenceIndex, sampleIntervalMs);
    }

    public static M2CameraPathDocument Import(
        MdxCameraFile cameraFile,
        MdxSummary summary,
        int cameraIndex = 0,
        int sequenceIndex = 0,
        int sampleIntervalMs = 125)
    {
        ArgumentNullException.ThrowIfNull(cameraFile);
        ArgumentNullException.ThrowIfNull(summary);
        if (cameraIndex < 0 || cameraIndex >= cameraFile.Cameras.Count)
            throw new ArgumentOutOfRangeException(nameof(cameraIndex));

        MdxCamera camera = cameraFile.Cameras[cameraIndex];
        int duration = ResolveDuration(summary, camera, sequenceIndex);
        int interval = Math.Clamp(sampleIntervalMs, 16, 1000);
        int sampleDuration = Math.Max(0, duration - 1);
        int sampleCount = sampleDuration <= 0
            ? 1
            : Math.Clamp((sampleDuration / interval) + 1, 2, 512);
        List<M2CameraPathKeyframe> keys = new(sampleCount);

        for (int index = 0; index < sampleCount; index++)
        {
            int time = sampleCount == 1
                ? 0
                : Math.Min(sampleDuration, (int)MathF.Round(sampleDuration * (index / (float)(sampleCount - 1))));
            MdxResolvedCameraState state = MdxCameraResolver.Resolve(summary, camera, sequenceIndex, time);
            float rollRadians = MdxAnimationSampler.SampleScalarTrack(
                camera.RollTrack,
                summary,
                sequenceIndex,
                time,
                0f);

            keys.Add(new M2CameraPathKeyframe
            {
                TimeMs = time,
                Position = state.Position,
                Target = state.Target,
                FovDegrees = ToDegrees(state.FieldOfView),
                RollDegrees = rollRadians * (180f / MathF.PI),
            });
        }

        M2CameraPathDocument result = new()
        {
            Name = string.IsNullOrWhiteSpace(camera.Name)
                ? (cameraFile.ModelName ?? Path.GetFileNameWithoutExtension(cameraFile.SourcePath))
                : camera.Name,
            Keyframes = keys,
        };
        M2CameraPathEvaluator.NormalizeAndValidate(result);
        return result;
    }

    private static int ResolveDuration(MdxSummary summary, MdxCamera camera, int sequenceIndex)
    {
        if (sequenceIndex >= 0 && sequenceIndex < summary.Sequences.Count)
            return Math.Max(0, summary.Sequences[sequenceIndex].Duration);

        int lastPosition = camera.PositionTrack?.LastKeyTime ?? 0;
        int lastTarget = camera.TargetPositionTrack?.LastKeyTime ?? 0;
        int lastRoll = camera.RollTrack?.LastKeyTime ?? 0;
        return Math.Max(lastPosition, Math.Max(lastTarget, lastRoll));
    }

    private static float ToDegrees(float fieldOfView)
    {
        if (!float.IsFinite(fieldOfView) || fieldOfView <= 0f)
            return 45f;

        float degrees = fieldOfView <= MathF.Tau
            ? fieldOfView * (180f / MathF.PI)
            : fieldOfView;
        return Math.Clamp(degrees, 1f, 179f);
    }
}
