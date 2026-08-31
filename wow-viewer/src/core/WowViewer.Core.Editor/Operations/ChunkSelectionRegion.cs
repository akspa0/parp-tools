namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Manages a selection region of global MCNK chunks across one or more ADT map tiles.
/// </summary>
public sealed class ChunkSelectionRegion
{
    private readonly HashSet<GlobalChunkCoordinate> _selected = new();

    public int Count => _selected.Count;

    public bool IsEmpty => _selected.Count == 0;

    public IEnumerable<GlobalChunkCoordinate> Chunks => _selected;

    public bool Add(GlobalChunkCoordinate coord)
    {
        if (!coord.IsValid) return false;
        return _selected.Add(coord);
    }

    public bool Add(int tileX, int tileY, int chunkX, int chunkY)
    {
        return Add(GlobalChunkCoordinate.FromTileAndChunk(tileX, tileY, chunkX, chunkY));
    }

    public bool Remove(GlobalChunkCoordinate coord)
    {
        return _selected.Remove(coord);
    }

    public bool Remove(int tileX, int tileY, int chunkX, int chunkY)
    {
        return Remove(GlobalChunkCoordinate.FromTileAndChunk(tileX, tileY, chunkX, chunkY));
    }

    public void Toggle(GlobalChunkCoordinate coord)
    {
        if (!coord.IsValid) return;
        if (!_selected.Add(coord))
            _selected.Remove(coord);
    }

    public void Toggle(int tileX, int tileY, int chunkX, int chunkY)
    {
        Toggle(GlobalChunkCoordinate.FromTileAndChunk(tileX, tileY, chunkX, chunkY));
    }

    public void AddRectangle(GlobalChunkCoordinate corner1, GlobalChunkCoordinate corner2)
    {
        int minGx = Math.Max(0, Math.Min(corner1.Gx, corner2.Gx));
        int maxGx = Math.Min(GlobalChunkCoordinate.TotalGlobalChunksPerAxis - 1, Math.Max(corner1.Gx, corner2.Gx));
        int minGy = Math.Max(0, Math.Min(corner1.Gy, corner2.Gy));
        int maxGy = Math.Min(GlobalChunkCoordinate.TotalGlobalChunksPerAxis - 1, Math.Max(corner1.Gy, corner2.Gy));

        for (int gx = minGx; gx <= maxGx; gx++)
        {
            for (int gy = minGy; gy <= maxGy; gy++)
            {
                _selected.Add(new GlobalChunkCoordinate(gx, gy));
            }
        }
    }

    public void AddTile(int tileX, int tileY)
    {
        if (tileX < 0 || tileX >= 64 || tileY < 0 || tileY >= 64) return;
        int baseGx = tileX * GlobalChunkCoordinate.ChunksPerTile;
        int baseGy = tileY * GlobalChunkCoordinate.ChunksPerTile;

        for (int cx = 0; cx < GlobalChunkCoordinate.ChunksPerTile; cx++)
        {
            for (int cy = 0; cy < GlobalChunkCoordinate.ChunksPerTile; cy++)
            {
                _selected.Add(new GlobalChunkCoordinate(baseGx + cx, baseGy + cy));
            }
        }
    }

    public void RemoveTile(int tileX, int tileY)
    {
        if (tileX < 0 || tileX >= 64 || tileY < 0 || tileY >= 64) return;
        int baseGx = tileX * GlobalChunkCoordinate.ChunksPerTile;
        int baseGy = tileY * GlobalChunkCoordinate.ChunksPerTile;

        for (int cx = 0; cx < GlobalChunkCoordinate.ChunksPerTile; cx++)
        {
            for (int cy = 0; cy < GlobalChunkCoordinate.ChunksPerTile; cy++)
            {
                _selected.Remove(new GlobalChunkCoordinate(baseGx + cx, baseGy + cy));
            }
        }
    }

    public void Clear()
    {
        _selected.Clear();
    }

    public bool Contains(GlobalChunkCoordinate coord) => _selected.Contains(coord);

    public bool Contains(int tileX, int tileY, int chunkX, int chunkY)
    {
        return Contains(GlobalChunkCoordinate.FromTileAndChunk(tileX, tileY, chunkX, chunkY));
    }

    public bool TryGetBoundingBox(out GlobalChunkCoordinate min, out GlobalChunkCoordinate max)
    {
        if (_selected.Count == 0)
        {
            min = default;
            max = default;
            return false;
        }

        int minGx = int.MaxValue;
        int maxGx = int.MinValue;
        int minGy = int.MaxValue;
        int maxGy = int.MinValue;

        foreach (var coord in _selected)
        {
            if (coord.Gx < minGx) minGx = coord.Gx;
            if (coord.Gx > maxGx) maxGx = coord.Gx;
            if (coord.Gy < minGy) minGy = coord.Gy;
            if (coord.Gy > maxGy) maxGy = coord.Gy;
        }

        min = new GlobalChunkCoordinate(minGx, minGy);
        max = new GlobalChunkCoordinate(maxGx, maxGy);
        return true;
    }

    public HashSet<(int tileX, int tileY)> GetAffectedTiles()
    {
        var tiles = new HashSet<(int tileX, int tileY)>();
        foreach (var coord in _selected)
            tiles.Add((coord.TileX, coord.TileY));
        return tiles;
    }
}
