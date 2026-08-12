using System.Numerics;

namespace WowViewer.Core.Runtime.M2;

/// <summary>
/// Computes the bounded terrain-tile footprint needed by a camera path.
/// Sampled positions are connected in tile space so a fast path cannot skip a
/// tile merely because its midpoint was between two time samples.
/// </summary>
public static class CameraPathTileFootprintSelector
{
    public static HashSet<(int tileX, int tileY)> GetTiles(
        M2CameraPathDocument path,
        float mapOrigin,
        float tileSize,
        int mapEdge,
        int sampleSpacingMs,
        int tileRadius,
        Func<int, int, bool>? tileExists = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        if (tileSize <= 0f)
            throw new ArgumentOutOfRangeException(nameof(tileSize));
        if (mapEdge <= 0)
            throw new ArgumentOutOfRangeException(nameof(mapEdge));
        if (sampleSpacingMs <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleSpacingMs));
        if (tileRadius < 0)
            throw new ArgumentOutOfRangeException(nameof(tileRadius));

        tileExists ??= static (int _, int _) => true;
        var tiles = new HashSet<(int tileX, int tileY)>();
        if (path.Keyframes.Count == 0)
            return tiles;

        int durationMs = Math.Max(0, path.DurationMs);
        int boundedSampleSpacingMs = Math.Max(1, sampleSpacingMs);
        int sampleCount = durationMs == 0
            ? 1
            : Math.Min(8192, (int)MathF.Ceiling(durationMs / (float)boundedSampleSpacingMs) + 1);

        var sampleTimes = new SortedSet<int>();
        for (int index = 0; index < sampleCount; index++)
        {
            int timeMs = sampleCount == 1
                ? 0
                : Math.Min(durationMs, (int)MathF.Round(durationMs * (index / (float)(sampleCount - 1))));
            sampleTimes.Add(timeMs);
        }

        foreach (M2CameraPathKeyframe key in path.Keyframes)
            sampleTimes.Add(Math.Clamp(key.TimeMs, 0, durationMs));

        (int tileX, int tileY)? previousTile = null;
        foreach (int timeMs in sampleTimes)
        {
            Vector3 position = M2CameraPathEvaluator.Sample(path, timeMs).Position;
            (int tileX, int tileY) tile = ToTile(position, mapOrigin, tileSize, mapEdge);

            if (previousTile is { } previous)
                AddTileLine(tiles, previous, tile, mapEdge, tileRadius, tileExists);
            else
                AddTileWindow(tiles, tile.tileX, tile.tileY, mapEdge, tileRadius, tileExists);

            previousTile = tile;
        }

        return tiles;
    }

    private static (int tileX, int tileY) ToTile(Vector3 position, float mapOrigin, float tileSize, int mapEdge)
        => (
            Math.Clamp((int)MathF.Floor((mapOrigin - position.X) / tileSize), 0, mapEdge - 1),
            Math.Clamp((int)MathF.Floor((mapOrigin - position.Y) / tileSize), 0, mapEdge - 1));

    private static void AddTileLine(
        HashSet<(int tileX, int tileY)> tiles,
        (int tileX, int tileY) start,
        (int tileX, int tileY) end,
        int mapEdge,
        int radius,
        Func<int, int, bool> tileExists)
    {
        int x = start.tileX;
        int y = start.tileY;
        int dx = Math.Abs(end.tileX - start.tileX);
        int sx = start.tileX < end.tileX ? 1 : -1;
        int dy = -Math.Abs(end.tileY - start.tileY);
        int sy = start.tileY < end.tileY ? 1 : -1;
        int error = dx + dy;

        while (true)
        {
            AddTileWindow(tiles, x, y, mapEdge, radius, tileExists);
            if (x == end.tileX && y == end.tileY)
                break;

            int doubledError = 2 * error;
            if (doubledError >= dy)
            {
                error += dy;
                x += sx;
            }
            if (doubledError <= dx)
            {
                error += dx;
                y += sy;
            }
        }
    }

    private static void AddTileWindow(
        HashSet<(int tileX, int tileY)> tiles,
        int centerX,
        int centerY,
        int mapEdge,
        int radius,
        Func<int, int, bool> tileExists)
    {
        for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++)
        {
            int tileX = centerX + dx;
            int tileY = centerY + dy;
            if (tileX >= 0 && tileX < mapEdge && tileY >= 0 && tileY < mapEdge && tileExists(tileX, tileY))
                tiles.Add((tileX, tileY));
        }
    }
}
