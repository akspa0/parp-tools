using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WowViewer.Core.IO.Maps;

/// <summary>Stitches known terrain-tile PNGs without fabricating pixels for missing map coordinates.</summary>
public static class TerrainMinimapStitcher
{
    public const int DefaultMaximumDimension = 16384;

    public static TerrainMinimapStitchResult Stitch(
        IReadOnlyDictionary<(int TileX, int TileY), string> tilePaths,
        string outputPath,
        int tileResolution,
        int maximumDimension = DefaultMaximumDimension)
    {
        ArgumentNullException.ThrowIfNull(tilePaths);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        if (tileResolution <= 0)
            throw new ArgumentOutOfRangeException(nameof(tileResolution));
        if (maximumDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumDimension));

        var available = tilePaths
            .Where(entry => File.Exists(entry.Value))
            .OrderBy(entry => entry.Key.TileY)
            .ThenBy(entry => entry.Key.TileX)
            .ToArray();
        if (available.Length == 0)
            throw new InvalidDataException("Cannot stitch a minimap without at least one emitted tile PNG.");

        int minX = available.Min(entry => entry.Key.TileX);
        int minY = available.Min(entry => entry.Key.TileY);
        int maxX = available.Max(entry => entry.Key.TileX);
        int maxY = available.Max(entry => entry.Key.TileY);
        int requestedWidth = checked((maxX - minX + 1) * tileResolution);
        int requestedHeight = checked((maxY - minY + 1) * tileResolution);

        float scale = Math.Min(1f, Math.Min(
            maximumDimension / (float)requestedWidth,
            maximumDimension / (float)requestedHeight));
        int effectiveTileResolution = Math.Max(1, (int)MathF.Floor(tileResolution * scale));
        int width = checked((maxX - minX + 1) * effectiveTileResolution);
        int height = checked((maxY - minY + 1) * effectiveTileResolution);

        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        using var canvas = new Image<Rgba32>(width, height, Color.Transparent);
        foreach (var entry in available)
        {
            using Image<Rgba32> tile = Image.Load<Rgba32>(entry.Value);
            if (tile.Width != effectiveTileResolution || tile.Height != effectiveTileResolution)
                tile.Mutate(context => context.Resize(effectiveTileResolution, effectiveTileResolution));

            int destinationX = (entry.Key.TileX - minX) * effectiveTileResolution;
            int destinationY = (entry.Key.TileY - minY) * effectiveTileResolution;
            canvas.Mutate(context => context.DrawImage(tile, new Point(destinationX, destinationY), 1f));
        }

        canvas.SaveAsPng(outputPath);
        return new TerrainMinimapStitchResult(
            outputPath,
            minX,
            minY,
            maxX,
            maxY,
            tileResolution,
            effectiveTileResolution,
            width,
            height,
            scale,
            available.Length);
    }
}

/// <summary>Records the coordinate and output shape contract for one stitched minimap PNG.</summary>
public sealed record TerrainMinimapStitchResult(
    string OutputPath,
    int MinTileX,
    int MinTileY,
    int MaxTileX,
    int MaxTileY,
    int RequestedTileResolution,
    int EffectiveTileResolution,
    int Width,
    int Height,
    float Scale,
    int TileCount);
