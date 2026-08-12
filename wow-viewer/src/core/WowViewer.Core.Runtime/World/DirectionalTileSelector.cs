using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Selects the camera tile and up to three immediately adjacent tiles in the
/// camera-facing direction. The selector deliberately has no radial expansion
/// or fog-distance behavior; that belongs to a later, explicitly bounded phase.
/// </summary>
public readonly record struct DirectionalTileCoord(int TileX, int TileY);

public sealed class DirectionalTileSelector
{
    private readonly float _mapOrigin;
    private readonly float _tileSize;
    private readonly int _mapEdge;
    private readonly Func<int, int, bool> _tileExists;

    public DirectionalTileSelector(
        float mapOrigin,
        float tileSize,
        int mapEdge,
        Func<int, int, bool>? tileExists = null)
    {
        if (tileSize <= 0f)
            throw new ArgumentOutOfRangeException(nameof(tileSize));
        if (mapEdge <= 0)
            throw new ArgumentOutOfRangeException(nameof(mapEdge));

        _mapOrigin = mapOrigin;
        _tileSize = tileSize;
        _mapEdge = mapEdge;
        _tileExists = tileExists ?? ((int _, int _) => true);
    }

    /// <summary>
    /// Returns the active tile followed by at most three adjacent tile
    /// coordinates inside the forward cone. <paramref name="fovDegrees"/>
    /// is the cone half-angle so the 45-degree baseline includes the direct
    /// neighbor and its two diagonal forward neighbors.
    /// </summary>
    public List<DirectionalTileCoord> GetVisibleTiles(Vector3 camPos, float yaw, float fovDegrees)
    {
        int activeTileX = ToTileCoordinate(_mapOrigin - camPos.X);
        int activeTileY = ToTileCoordinate(_mapOrigin - camPos.Y);

        var selected = new List<DirectionalTileCoord>(capacity: 4);
        AddIfPresent(selected, activeTileX, activeTileY);

        if (fovDegrees <= 0f)
            return selected;

        float halfAngleRadians = Math.Clamp(fovDegrees, 0f, 180f) * (MathF.PI / 180f);
        float minimumDot = MathF.Cos(halfAngleRadians);
        float yawRadians = yaw * (MathF.PI / 180f);
        Vector2 forward = new(MathF.Cos(yawRadians), MathF.Sin(yawRadians));

        var candidates = new List<(DirectionalTileCoord Coord, float Dot, int Manhattan)>(8);
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;

                int tileX = activeTileX + dx;
                int tileY = activeTileY + dy;
                if (!IsInBounds(tileX, tileY) || !_tileExists(tileX, tileY))
                    continue;

                // Tile coordinates run opposite to world X/Y in the WoW map
                // transform, so negate the grid offset before testing yaw.
                Vector2 direction = Vector2.Normalize(new Vector2(-dx, -dy));
                float dot = Vector2.Dot(forward, direction);
                if (dot + 1e-5f < minimumDot)
                    continue;

                candidates.Add((new DirectionalTileCoord(tileX, tileY), dot, Math.Abs(dx) + Math.Abs(dy)));
            }
        }

        candidates.Sort(static (left, right) =>
        {
            int dotCompare = right.Dot.CompareTo(left.Dot);
            if (dotCompare != 0)
                return dotCompare;

            int distanceCompare = left.Manhattan.CompareTo(right.Manhattan);
            if (distanceCompare != 0)
                return distanceCompare;

            int xCompare = left.Coord.TileX.CompareTo(right.Coord.TileX);
            return xCompare != 0
                ? xCompare
                : left.Coord.TileY.CompareTo(right.Coord.TileY);
        });

        foreach (var candidate in candidates)
        {
            if (selected.Count >= 4)
                break;

            selected.Add(candidate.Coord);
        }

        return selected;
    }

    private int ToTileCoordinate(float distanceFromMapOrigin)
        => Math.Clamp((int)MathF.Floor(distanceFromMapOrigin / _tileSize), 0, _mapEdge - 1);

    private bool IsInBounds(int tileX, int tileY)
        => tileX >= 0 && tileX < _mapEdge && tileY >= 0 && tileY < _mapEdge;

    private void AddIfPresent(List<DirectionalTileCoord> selected, int tileX, int tileY)
    {
        if (IsInBounds(tileX, tileY) && _tileExists(tileX, tileY))
            selected.Add(new DirectionalTileCoord(tileX, tileY));
    }
}
