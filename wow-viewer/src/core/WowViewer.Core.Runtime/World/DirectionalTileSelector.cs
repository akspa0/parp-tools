using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Selects a bounded set of tiles in the camera-facing cone. The selector
/// deliberately has no fog-distance or whole-map expansion; the caller owns
/// the requested detail count and therefore controls the bounded cone depth.
/// </summary>
public readonly record struct DirectionalTileCoord(int TileX, int TileY);

public sealed class DirectionalTileSelector
{
    private readonly float _mapOrigin;
    private readonly float _tileSize;
    private readonly int _mapEdge;
    private readonly Func<int, int, bool> _tileExists;
    private readonly int _maxCandidateRadius;

    public DirectionalTileSelector(
        float mapOrigin,
        float tileSize,
        int mapEdge,
        Func<int, int, bool>? tileExists = null,
        int maxCandidateRadius = 4)
    {
        if (tileSize <= 0f)
            throw new ArgumentOutOfRangeException(nameof(tileSize));
        if (mapEdge <= 0)
            throw new ArgumentOutOfRangeException(nameof(mapEdge));
        if (maxCandidateRadius <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxCandidateRadius));

        _mapOrigin = mapOrigin;
        _tileSize = tileSize;
        _mapEdge = mapEdge;
        _tileExists = tileExists ?? ((int _, int _) => true);
        _maxCandidateRadius = maxCandidateRadius;
    }

    /// <summary>
    /// Returns the legacy four-tile baseline. New callers should use the
    /// overload that supplies <paramref name="maxTileCount"/> so the active
    /// detail control owns the coverage budget.
    /// </summary>
    public List<DirectionalTileCoord> GetVisibleTiles(Vector3 camPos, float yaw, float fovDegrees)
        => GetVisibleTiles(camPos, yaw, fovDegrees, maxTileCount: 4);

    /// <summary>
    /// Returns the active tile and up to <paramref name="maxTileCount"/>
    /// existing tiles inside the forward cone. The search expands only along
    /// bounded forward rings; it does not perform radial or map-wide admission.
    /// </summary>
    public List<DirectionalTileCoord> GetVisibleTiles(
        Vector3 camPos,
        float yaw,
        float fovDegrees,
        int maxTileCount)
    {
        if (maxTileCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxTileCount));

        int activeTileX = ToTileCoordinate(_mapOrigin - camPos.X);
        int activeTileY = ToTileCoordinate(_mapOrigin - camPos.Y);

        int candidateRadius = Math.Min(
            _maxCandidateRadius,
            Math.Max(1, (int)MathF.Ceiling(MathF.Sqrt(maxTileCount)) - 1));
        var selected = new List<DirectionalTileCoord>(capacity: Math.Min(maxTileCount, 25));
        AddIfPresent(selected, activeTileX, activeTileY);

        if (selected.Count >= maxTileCount)
            return selected;

        if (fovDegrees <= 0f)
            return selected;

        float halfAngleRadians = Math.Clamp(fovDegrees, 0f, 180f) * (MathF.PI / 180f);
        float minimumDot = MathF.Cos(halfAngleRadians);
        float yawRadians = yaw * (MathF.PI / 180f);
        Vector2 forward = new(MathF.Cos(yawRadians), MathF.Sin(yawRadians));

        var candidates = new List<(DirectionalTileCoord Coord, float Dot, int Manhattan)>(8);
        for (int dy = -candidateRadius; dy <= candidateRadius; dy++)
        {
            for (int dx = -candidateRadius; dx <= candidateRadius; dx++)
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
            if (selected.Count >= maxTileCount)
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
