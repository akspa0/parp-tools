namespace WowViewer.Core.Maps;

public sealed class WdlSummary
{
    private const int GridSize = 64;
    private readonly WdlTileSummary?[] _tileGrid;

    public WdlSummary(string sourcePath, uint? version, WdlTileSummary?[] tileGrid)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(tileGrid);
        if (tileGrid.Length != GridSize * GridSize)
            throw new ArgumentException($"WDL tile grids must contain exactly {GridSize * GridSize} entries.", nameof(tileGrid));

        SourcePath = sourcePath;
        Version = version;
        _tileGrid = tileGrid;
        Tiles = tileGrid.Where(static tile => tile is not null).Select(static tile => tile!).ToArray();
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public IReadOnlyList<WdlTileSummary> Tiles { get; }

    public int TileCount => Tiles.Count;

    public bool TryGetTile(int tileX, int tileY, out WdlTileSummary? tile)
    {
        if ((uint)tileX >= GridSize || (uint)tileY >= GridSize)
        {
            tile = null;
            return false;
        }

        tile = _tileGrid[(tileY * GridSize) + tileX];
        return tile is not null;
    }
}

public sealed class WdlTileSummary
{
    public const int OuterGridSize = 17;
    public const int InnerGridSize = 16;
    public const int OuterHeightCount = OuterGridSize * OuterGridSize;
    public const int InnerHeightCount = InnerGridSize * InnerGridSize;

    public WdlTileSummary(int tileX, int tileY, short[] outerHeights, short[] innerHeights, short minHeight, short maxHeight)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentNullException.ThrowIfNull(outerHeights);
        ArgumentNullException.ThrowIfNull(innerHeights);
        if (outerHeights.Length != OuterHeightCount)
            throw new ArgumentException($"WDL outer height grids must contain exactly {OuterHeightCount} entries.", nameof(outerHeights));
        if (innerHeights.Length != InnerHeightCount)
            throw new ArgumentException($"WDL inner height grids must contain exactly {InnerHeightCount} entries.", nameof(innerHeights));

        TileX = tileX;
        TileY = tileY;
        OuterHeights = outerHeights;
        InnerHeights = innerHeights;
        MinHeight = minHeight;
        MaxHeight = maxHeight;
    }

    public int TileX { get; }

    public int TileY { get; }

    public short[] OuterHeights { get; }

    public short[] InnerHeights { get; }

    public short MinHeight { get; }

    public short MaxHeight { get; }

    public short GetOuterHeight(int x, int y)
    {
        if ((uint)x >= OuterGridSize || (uint)y >= OuterGridSize)
            throw new ArgumentOutOfRangeException(nameof(x), "Outer WDL coordinates must stay within the 17x17 grid.");

        return OuterHeights[(y * OuterGridSize) + x];
    }

    public short GetInnerHeight(int x, int y)
    {
        if ((uint)x >= InnerGridSize || (uint)y >= InnerGridSize)
            throw new ArgumentOutOfRangeException(nameof(x), "Inner WDL coordinates must stay within the 16x16 grid.");

        return InnerHeights[(y * InnerGridSize) + x];
    }
}