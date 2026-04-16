using System.Numerics;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2CameraPathVisualization
{
    public M2CameraPathVisualization(
        IReadOnlyList<M2CameraPathOverlay> overlays,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentNullException.ThrowIfNull(overlays);

        Overlays = overlays;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public IReadOnlyList<M2CameraPathOverlay> Overlays { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }
}

public sealed class M2CameraPathOverlay
{
    public M2CameraPathOverlay(
        int cameraIndex,
        int cameraType,
        string typeLabel,
        IReadOnlyList<Vector3> cameraSamples,
        IReadOnlyList<Vector3> targetSamples,
        Vector3 boundsMin,
        Vector3 boundsMax,
        float pinHeight,
        float pinHeadSize)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(typeLabel);
        ArgumentNullException.ThrowIfNull(cameraSamples);
        ArgumentNullException.ThrowIfNull(targetSamples);

        CameraIndex = cameraIndex;
        CameraType = cameraType;
        TypeLabel = typeLabel;
        CameraSamples = cameraSamples;
        TargetSamples = targetSamples;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        PinHeight = pinHeight;
        PinHeadSize = pinHeadSize;
    }

    public int CameraIndex { get; }

    public int CameraType { get; }

    public string TypeLabel { get; }

    public string Name => $"Camera {CameraIndex} ({TypeLabel})";

    public IReadOnlyList<Vector3> CameraSamples { get; }

    public IReadOnlyList<Vector3> TargetSamples { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }

    public float PinHeight { get; }

    public float PinHeadSize { get; }
}