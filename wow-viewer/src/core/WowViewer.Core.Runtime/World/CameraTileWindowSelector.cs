using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Selects a bounded camera-centered residency window. This is deliberately
/// separate from directional visibility: the window controls what may stream
/// in and remain resident, not what the renderer must submit this frame.
/// </summary>
public sealed class CameraTileWindowSelector
{
    private readonly float _mapOrigin;
    private readonly float _tileSize;
    private readonly int _mapEdge;
    private readonly Func<int, int, bool> _tileExists;

    public CameraTileWindowSelector(
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
    /// Returns every existing tile in the bounded Chebyshev window around the
    /// camera tile. The active tile comes first, followed by nearer tiles in
    /// deterministic coordinate order.
    /// </summary>
    public List<DirectionalTileCoord> GetTiles(Vector3 camPos, int radius)
    {
        if (radius < 0)
            throw new ArgumentOutOfRangeException(nameof(radius));

        int activeTileX = ToTileCoordinate(_mapOrigin - camPos.X);
        int activeTileY = ToTileCoordinate(_mapOrigin - camPos.Y);
        int boundedRadius = Math.Min(radius, _mapEdge - 1);
        var candidates = new List<(DirectionalTileCoord Coord, int Distance, int DistanceSquared)>(
            capacity: (boundedRadius * 2 + 1) * (boundedRadius * 2 + 1));

        for (int dy = -boundedRadius; dy <= boundedRadius; dy++)
        {
            for (int dx = -boundedRadius; dx <= boundedRadius; dx++)
            {
                int tileX = activeTileX + dx;
                int tileY = activeTileY + dy;
                if (!IsInBounds(tileX, tileY) || !_tileExists(tileX, tileY))
                    continue;

                int distance = Math.Abs(dx) + Math.Abs(dy);
                candidates.Add((
                    new DirectionalTileCoord(tileX, tileY),
                    distance,
                    dx * dx + dy * dy));
            }
        }

        candidates.Sort(static (left, right) =>
        {
            int distanceCompare = left.DistanceSquared.CompareTo(right.DistanceSquared);
            if (distanceCompare != 0)
                return distanceCompare;

            int manhattanCompare = left.Distance.CompareTo(right.Distance);
            if (manhattanCompare != 0)
                return manhattanCompare;

            int xCompare = left.Coord.TileX.CompareTo(right.Coord.TileX);
            return xCompare != 0
                ? xCompare
                : left.Coord.TileY.CompareTo(right.Coord.TileY);
        });

        return candidates.Select(static candidate => candidate.Coord).ToList();
    }

    private int ToTileCoordinate(float distanceFromMapOrigin)
        => Math.Clamp((int)MathF.Floor(distanceFromMapOrigin / _tileSize), 0, _mapEdge - 1);

    private bool IsInBounds(int tileX, int tileY)
        => tileX >= 0 && tileX < _mapEdge && tileY >= 0 && tileY < _mapEdge;
}
