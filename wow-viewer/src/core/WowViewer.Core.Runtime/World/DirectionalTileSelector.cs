using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Selects a bounded set of tiles around the camera. The active tile and its
/// immediate neighborhood are admitted first as a near-field safety ring so
/// close terrain cannot disappear at a tile seam. Remaining budget expands
/// only through the camera-facing cone; there is no fog-distance or whole-map
/// expansion.
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
    /// existing tiles. The largest complete near-field square that fits in the
    /// requested budget is filled first, then any remaining budget expands
    /// through bounded forward rings. The near-field ring intentionally has
    /// priority over the FOV so adjacent terrain and objects cannot pop out
    /// beside the camera.
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

        var selectedSet = new HashSet<DirectionalTileCoord>(selected);

        // Protect the largest complete camera-centered square that fits inside
        // the requested budget before spending the remaining slots on distant
        // forward tiles. This gives 9/12 tiles a 3x3 near field and 25 tiles a
        // full 5x5 near field, instead of letting the FOV consume the budget
        // while close side/rear ADTs disappear.
        int nearFieldRadius = 0;
        while ((nearFieldRadius * 2 + 3) * (nearFieldRadius * 2 + 3) <= maxTileCount
            && nearFieldRadius < candidateRadius)
        {
            nearFieldRadius++;
        }

        if (nearFieldRadius > 0)
        {
            var nearCandidates = new List<(DirectionalTileCoord Coord, float Dot, int Manhattan)>(
                (nearFieldRadius * 2 + 1) * (nearFieldRadius * 2 + 1) - 1);
            for (int dy = -nearFieldRadius; dy <= nearFieldRadius; dy++)
            {
                for (int dx = -nearFieldRadius; dx <= nearFieldRadius; dx++)
                {
                    if (dx == 0 && dy == 0)
                        continue;

                    int tileX = activeTileX + dx;
                    int tileY = activeTileY + dy;
                    if (!IsInBounds(tileX, tileY) || !_tileExists(tileX, tileY))
                        continue;

                    DirectionalTileCoord coord = new(tileX, tileY);
                    if (selectedSet.Contains(coord))
                        continue;

                    float dot = GetHeadingDot(forward, dx, dy);
                    nearCandidates.Add((coord, dot, Math.Abs(dx) + Math.Abs(dy)));
                }
            }

            SortCandidates(nearCandidates);
            foreach (var candidate in nearCandidates)
            {
                if (selected.Count >= maxTileCount)
                    return selected;

                selected.Add(candidate.Coord);
                selectedSet.Add(candidate.Coord);
            }
        }

        var candidates = new List<(DirectionalTileCoord Coord, float Dot, int Manhattan)>(16);
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
                DirectionalTileCoord coord = new(tileX, tileY);
                if (selectedSet.Contains(coord))
                    continue;

                // Tile coordinates run opposite to world X/Y in the WoW map
                // transform, so negate the grid offset before testing yaw.
                float dot = GetHeadingDot(forward, dx, dy);
                if (dot + 1e-5f < minimumDot)
                    continue;

                candidates.Add((coord, dot, Math.Abs(dx) + Math.Abs(dy)));
            }
        }

        SortCandidates(candidates);

        foreach (var candidate in candidates)
        {
            if (selected.Count >= maxTileCount)
                break;

            selected.Add(candidate.Coord);
        }

        return selected;
    }

    private static float GetHeadingDot(Vector2 forward, int dx, int dy)
    {
        Vector2 direction = Vector2.Normalize(new Vector2(-dx, -dy));
        return Vector2.Dot(forward, direction);
    }

    private static void SortCandidates(List<(DirectionalTileCoord Coord, float Dot, int Manhattan)> candidates)
    {
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
