namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WeakSignalOptions
{
    public const float DefaultMinHeightBand = -8192f;
    public const float DefaultMaxHeightBand = 512f;

    public float MaxHeightRange { get; init; } = float.MaxValue;
    public float MinHeightBand { get; init; } = DefaultMinHeightBand;
    public float MaxHeightBand { get; init; } = DefaultMaxHeightBand;
    public bool UseAutoFactor { get; init; } = true;
    public float ManualFactor { get; init; } = WeakSignalDetector.ClassicCompressionFactor;
}
