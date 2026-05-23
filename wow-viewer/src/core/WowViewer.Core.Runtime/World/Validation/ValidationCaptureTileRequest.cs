namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureTileRequest
{
    public ValidationCaptureTileRequest(
        string tileName,
        int tileX,
        int tileY,
        ValidationCaptureVariant variant,
        string outputPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        if (!Enum.IsDefined(variant))
            throw new ArgumentOutOfRangeException(nameof(variant), variant, "Validation capture variant must be defined.");

        TileName = tileName;
        TileX = tileX;
        TileY = tileY;
        Variant = variant;
        OutputPath = outputPath;
    }

    public string TileName { get; }

    public int TileX { get; }

    public int TileY { get; }

    public ValidationCaptureVariant Variant { get; }

    public string OutputPath { get; }
}