namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureVariantResult
{
    public ValidationCaptureVariantResult(
        ValidationCaptureVariant variant,
        string tileName,
        int tileX,
        int tileY,
        string outputPath,
        ValidationCaptureReadinessState readinessState,
        bool succeeded,
        bool timedOut,
        int framesObserved,
        int settledFrames,
        string? failureReason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        if (!Enum.IsDefined(variant))
            throw new ArgumentOutOfRangeException(nameof(variant), variant, "Validation capture variant must be defined.");
        ArgumentOutOfRangeException.ThrowIfNegative(framesObserved);
        ArgumentOutOfRangeException.ThrowIfNegative(settledFrames);

        Variant = variant;
        TileName = tileName;
        TileX = tileX;
        TileY = tileY;
        OutputPath = outputPath;
        ReadinessState = readinessState;
        Succeeded = succeeded;
        TimedOut = timedOut;
        FramesObserved = framesObserved;
        SettledFrames = settledFrames;
        FailureReason = failureReason;
    }

    public ValidationCaptureVariant Variant { get; }

    public string TileName { get; }

    public int TileX { get; }

    public int TileY { get; }

    public string OutputPath { get; }

    public ValidationCaptureReadinessState ReadinessState { get; }

    public bool Succeeded { get; }

    public bool TimedOut { get; }

    public int FramesObserved { get; }

    public int SettledFrames { get; }

    public string? FailureReason { get; }
}