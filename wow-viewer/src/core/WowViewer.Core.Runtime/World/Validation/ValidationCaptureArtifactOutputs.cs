namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureArtifactOutputs
{
    public ValidationCaptureArtifactOutputs(
        string tileName,
        string? buildLabel,
        int width,
        int height,
        ValidationObjectMaskStrategy maskStrategy,
        byte[] objectVisibilityMaskL8Pixels,
        string objectVisibilityMaskHash,
        byte[] noObjectMinimapRgbaPixels,
        string noObjectMinimapHash)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentNullException.ThrowIfNull(objectVisibilityMaskL8Pixels);
        ArgumentException.ThrowIfNullOrWhiteSpace(objectVisibilityMaskHash);
        ArgumentNullException.ThrowIfNull(noObjectMinimapRgbaPixels);
        ArgumentException.ThrowIfNullOrWhiteSpace(noObjectMinimapHash);

        int maskLength = checked(width * height);
        int rgbaLength = checked(width * height * 4);
        if (objectVisibilityMaskL8Pixels.Length != maskLength)
            throw new ArgumentException("Object-visibility mask payload length must match width * height.", nameof(objectVisibilityMaskL8Pixels));
        if (noObjectMinimapRgbaPixels.Length != rgbaLength)
            throw new ArgumentException("No-object minimap RGBA payload length must match width * height * 4.", nameof(noObjectMinimapRgbaPixels));

        TileName = tileName;
        BuildLabel = buildLabel;
        Width = width;
        Height = height;
        MaskStrategy = maskStrategy;
        ObjectVisibilityMaskL8Pixels = objectVisibilityMaskL8Pixels;
        ObjectVisibilityMaskHash = objectVisibilityMaskHash;
        NoObjectMinimapRgbaPixels = noObjectMinimapRgbaPixels;
        NoObjectMinimapHash = noObjectMinimapHash;
    }

    public string TileName { get; }

    public string? BuildLabel { get; }

    public int Width { get; }

    public int Height { get; }

    public ValidationObjectMaskStrategy MaskStrategy { get; }

    public byte[] ObjectVisibilityMaskL8Pixels { get; }

    public string ObjectVisibilityMaskHash { get; }

    public byte[] NoObjectMinimapRgbaPixels { get; }

    public string NoObjectMinimapHash { get; }
}