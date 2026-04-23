using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Tool.Converter;

internal static class TerrainMccvGuideTextureBuilder
{
    public static TerrainMccvGuideOutputs? TryWriteOutputs(string adtPath, string tileName, string? sourceMinimapPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(adtPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(tileName);

        IReadOnlyDictionary<int, byte[]> chunkColors = AdtMccvTileImageBuilder.ReadChunkColors(adtPath);
        if (chunkColors.Count == 0)
            return null;

        string outputDirectory = Path.GetDirectoryName(adtPath) ?? ".";
        Directory.CreateDirectory(outputDirectory);

        byte[] rawImageBytes = AdtMccvTileImageBuilder.RenderTileImageRgba(chunkColors);
        string rawMccvPngPath = Path.Combine(outputDirectory, $"{tileName}_mccv.png");
        using Image<Rgba32> rawMccvImage = Image.LoadPixelData<Rgba32>(rawImageBytes, AdtMccvTileImageBuilder.TileImageSize, AdtMccvTileImageBuilder.TileImageSize);
        rawMccvImage.SaveAsPng(rawMccvPngPath);

        string? guideTexturePath = null;
        if (!string.IsNullOrWhiteSpace(sourceMinimapPath) && File.Exists(sourceMinimapPath))
        {
            guideTexturePath = Path.Combine(outputDirectory, $"{tileName}_terrain_guide.png");
            using Image<Rgba32> guideImage = Image.Load<Rgba32>(sourceMinimapPath);
            using Image<Rgba32> overlay = rawMccvImage.Clone();
            if (overlay.Width != guideImage.Width || overlay.Height != guideImage.Height)
                overlay.Mutate(ctx => ctx.Resize(guideImage.Width, guideImage.Height));

            for (int y = 0; y < guideImage.Height; y++)
            {
                for (int x = 0; x < guideImage.Width; x++)
                    guideImage[x, y] = ApplyTint(guideImage[x, y], overlay[x, y]);
            }

            guideImage.SaveAsPng(guideTexturePath);
        }

        return new TerrainMccvGuideOutputs(rawMccvPngPath, guideTexturePath);
    }

    private static Rgba32 ApplyTint(Rgba32 source, Rgba32 rawStoredTint)
    {
        Rgba32 tint = DecodeStoredMccvColor(rawStoredTint);
        return new Rgba32(
            ApplyTint(source.R, tint.R, tint.A),
            ApplyTint(source.G, tint.G, tint.A),
            ApplyTint(source.B, tint.B, tint.A),
            source.A);
    }

    private static byte ApplyTint(byte source, byte tintChannel, byte tintAlpha)
    {
        float tintColor = Math.Clamp((tintChannel / 255f) * 2f, 0f, 2f);
        float tintStrength = Math.Clamp((tintAlpha / 255f) * 2f - 1f, 0f, 1f);
        float tintFactor = Math.Clamp(1f + ((tintColor - 1f) * tintStrength), 1f / 255f, 2f);
        return (byte)Math.Clamp((int)MathF.Round(source * tintFactor), 0, 255);
    }

    private static Rgba32 DecodeStoredMccvColor(Rgba32 rawPixel)
    {
        return new Rgba32(rawPixel.B, rawPixel.G, rawPixel.R, rawPixel.A);
    }

    internal sealed record TerrainMccvGuideOutputs(string RawMccvPngPath, string? GuideTexturePath);
}