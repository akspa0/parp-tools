using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public static class M2CameraPathOverlayBuilder
{
    public static bool CanBuild(M2ModelDocument model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return model.CameraCount > 0 && (model.ViewCount == 0 || LooksLikeStandaloneCameraAsset(model));
    }

    private static bool LooksLikeStandaloneCameraAsset(M2ModelDocument model)
    {
        if (model.RibbonCount != 0 || model.ParticleCount != 0)
            return false;

        string canonicalPath = model.Identity.CanonicalModelPath;
        string fileName = Path.GetFileNameWithoutExtension(canonicalPath);
        string? directory = Path.GetDirectoryName(canonicalPath);

        return fileName.EndsWith("_cam", StringComparison.OrdinalIgnoreCase)
            || string.Equals(directory, "Cameras", StringComparison.OrdinalIgnoreCase);
    }

    public static M2CameraPathVisualization Build(M2ModelDocument model)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (!CanBuild(model))
        {
            throw new InvalidOperationException(
                $"Model '{model.Identity.CanonicalModelPath}' is not a camera-only visualization candidate.");
        }

        List<M2CameraPathOverlay> overlays = new(model.Cameras.Count);
        for (int index = 0; index < model.Cameras.Count; index++)
            overlays.Add(BuildOverlay(model, model.Cameras[index]));

        Vector3 boundsMin = overlays[0].BoundsMin;
        Vector3 boundsMax = overlays[0].BoundsMax;
        for (int index = 1; index < overlays.Count; index++)
        {
            boundsMin = Vector3.Min(boundsMin, overlays[index].BoundsMin);
            boundsMax = Vector3.Max(boundsMax, overlays[index].BoundsMax);
        }

        return new M2CameraPathVisualization(overlays, boundsMin, boundsMax);
    }

    private static M2CameraPathOverlay BuildOverlay(M2ModelDocument model, M2CameraDefinition camera)
    {
        List<CameraSampleDomain> domains = BuildDomains(model, camera);
        List<Vector3> cameraSamples = new();
        List<Vector3> targetSamples = new();

        if (domains.Count == 0)
        {
            cameraSamples.Add(camera.PositionBase);
            targetSamples.Add(camera.TargetPositionBase);
        }
        else
        {
            foreach (CameraSampleDomain domain in domains)
            {
                foreach (int sampleTime in BuildSampleTimes(domain.DurationMs))
                {
                    cameraSamples.Add(SamplePosition(model, camera, domain.SequenceIndex, sampleTime));
                    targetSamples.Add(SampleTarget(model, camera, domain.SequenceIndex, sampleTime));
                }
            }
        }

        cameraSamples = CompactSamples(cameraSamples);
        targetSamples = CompactSamples(targetSamples);

        if (cameraSamples.Count == 0)
            cameraSamples.Add(camera.PositionBase);

        if (targetSamples.Count == 0)
            targetSamples.Add(camera.TargetPositionBase);

        Vector3 boundsMin = cameraSamples[0];
        Vector3 boundsMax = cameraSamples[0];
        foreach (Vector3 point in cameraSamples)
        {
            boundsMin = Vector3.Min(boundsMin, point);
            boundsMax = Vector3.Max(boundsMax, point);
        }

        foreach (Vector3 point in targetSamples)
        {
            boundsMin = Vector3.Min(boundsMin, point);
            boundsMax = Vector3.Max(boundsMax, point);
        }

        float diagonal = MathF.Max((boundsMax - boundsMin).Length(), 10.0f);
        float pinHeight = Math.Clamp(diagonal * 0.04f, 3.0f, 36.0f);
        float pinHeadSize = Math.Clamp(pinHeight * 0.25f, 0.75f, 8.0f);

        return new M2CameraPathOverlay(
            camera.Index,
            camera.Type,
            DescribeCameraType(camera.Type),
            cameraSamples,
            targetSamples,
            boundsMin,
            boundsMax,
            pinHeight,
            pinHeadSize);
    }

    private static List<CameraSampleDomain> BuildDomains(M2ModelDocument model, M2CameraDefinition camera)
    {
        List<CameraSampleDomain> domains = new();
        AddDomains(domains, model, camera.PositionTrack);
        AddDomains(domains, model, camera.TargetPositionTrack);

        List<CameraSampleDomain> deduped = new();
        for (int index = 0; index < domains.Count; index++)
        {
            bool seen = false;
            for (int previous = 0; previous < deduped.Count; previous++)
            {
                if (deduped[previous].Equals(domains[index]))
                {
                    seen = true;
                    break;
                }
            }

            if (!seen)
                deduped.Add(domains[index]);
        }

        return deduped;
    }

    private static void AddDomains<T>(List<CameraSampleDomain> domains, M2ModelDocument model, M2TrackDefinition<T> track)
    {
        if (track.TimestampArray.Count == 0 || track.ValueArray.Count == 0)
            return;

        if (track.UsesGlobalSequence)
        {
            if (track.GlobalSequenceIndex >= 0 && track.GlobalSequenceIndex < model.GlobalLoops.Count)
            {
                int durationMs = Math.Max((int)model.GlobalLoops[track.GlobalSequenceIndex], 1);
                domains.Add(new CameraSampleDomain(0, durationMs));
            }

            return;
        }

        int sequenceCount = (int)Math.Min(Math.Min(track.TimestampArray.Count, track.ValueArray.Count), (uint)model.Sequences.Count);
        for (int sequenceIndex = 0; sequenceIndex < sequenceCount; sequenceIndex++)
        {
            int durationMs = Math.Max((int)model.Sequences[sequenceIndex].Duration, 1);
            domains.Add(new CameraSampleDomain(sequenceIndex, durationMs));
        }
    }

    private static IEnumerable<int> BuildSampleTimes(int durationMs)
    {
        if (durationMs <= 0)
        {
            yield return 0;
            yield break;
        }

        int sampleCount = Math.Clamp((durationMs / 125) + 1, 4, 64);
        if (sampleCount == 1)
        {
            yield return 0;
            yield break;
        }

        for (int index = 0; index < sampleCount; index++)
        {
            float factor = index / (float)(sampleCount - 1);
            yield return (int)MathF.Round(durationMs * factor);
        }
    }

    private static Vector3 SamplePosition(M2ModelDocument model, M2CameraDefinition camera, int sequenceIndex, int timeMs)
    {
        if (!CanSample(model, camera.PositionTrack, sequenceIndex))
            return camera.PositionBase;

        return camera.PositionBase + M2TrackSampler.SampleVector3(model.RawBytes, model, sequenceIndex, timeMs, camera.PositionTrack, Vector3.Zero);
    }

    private static Vector3 SampleTarget(M2ModelDocument model, M2CameraDefinition camera, int sequenceIndex, int timeMs)
    {
        if (!CanSample(model, camera.TargetPositionTrack, sequenceIndex))
            return camera.TargetPositionBase;

        return camera.TargetPositionBase + M2TrackSampler.SampleVector3(model.RawBytes, model, sequenceIndex, timeMs, camera.TargetPositionTrack, Vector3.Zero);
    }

    private static bool CanSample<T>(M2ModelDocument model, M2TrackDefinition<T> track, int sequenceIndex)
    {
        if (track.TimestampArray.Count == 0 || track.ValueArray.Count == 0)
            return false;

        return track.UsesGlobalSequence || (sequenceIndex >= 0 && sequenceIndex < model.Sequences.Count);
    }

    private static List<Vector3> CompactSamples(List<Vector3> samples)
    {
        if (samples.Count <= 1)
            return samples;

        List<Vector3> compacted = new(samples.Count) { samples[0] };
        for (int index = 1; index < samples.Count; index++)
        {
            if (!ApproximatelyEqual(compacted[^1], samples[index]))
                compacted.Add(samples[index]);
        }

        return compacted;
    }

    private static bool ApproximatelyEqual(Vector3 left, Vector3 right)
        => Vector3.DistanceSquared(left, right) <= 0.0001f;

    private static string DescribeCameraType(int cameraType)
    {
        return cameraType switch
        {
            0 => "portrait",
            1 => "character info",
            -1 => "flyby",
            _ => $"type {cameraType}",
        };
    }

    private readonly record struct CameraSampleDomain(int SequenceIndex, int DurationMs);
}