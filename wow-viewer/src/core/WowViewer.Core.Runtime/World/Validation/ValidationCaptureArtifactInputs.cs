namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureArtifactInputs
{
    public ValidationCaptureArtifactInputs(
        string tileName,
        string? buildLabel,
        int width,
        int height,
        byte[] primaryRgbaPixels,
        byte[] noObjectsRgbaPixels,
        byte[]? objectsOnlyRgbaPixels)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(primaryRgbaPixels);
        ArgumentNullException.ThrowIfNull(noObjectsRgbaPixels);

        int rgbaLength = checked(width * height * 4);
        if (primaryRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("Primary RGBA payload length must match width * height * 4.", nameof(primaryRgbaPixels));
        if (noObjectsRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("No-objects RGBA payload length must match width * height * 4.", nameof(noObjectsRgbaPixels));
        if (objectsOnlyRgbaPixels != null && objectsOnlyRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("Objects-only RGBA payload length must match width * height * 4.", nameof(objectsOnlyRgbaPixels));

        TileName = tileName;
        BuildLabel = buildLabel;
        Width = width;
        Height = height;
        PrimaryRgbaPixels = primaryRgbaPixels;
        NoObjectsRgbaPixels = noObjectsRgbaPixels;
        ObjectsOnlyRgbaPixels = objectsOnlyRgbaPixels;
    }

    public string TileName { get; }

    public string? BuildLabel { get; }

    public int Width { get; }

    public int Height { get; }

    public byte[] PrimaryRgbaPixels { get; }

    public byte[] NoObjectsRgbaPixels { get; }

    public byte[]? ObjectsOnlyRgbaPixels { get; }
}