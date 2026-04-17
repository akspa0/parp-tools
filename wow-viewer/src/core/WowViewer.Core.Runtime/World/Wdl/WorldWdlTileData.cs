namespace WowViewer.Core.Runtime.World.Wdl;

public sealed class WorldWdlTileData
{
    public WorldWdlTileData(
        string sourcePath,
        uint? version,
        int tileX,
        int tileY,
        bool sourceFound,
        bool hasData,
        short? minHeight,
        short? maxHeight,
        short? centerHeight,
        short? northWestHeight,
        short? northEastHeight,
        short? southWestHeight,
        short? southEastHeight,
        int outerHeightCount,
        int innerHeightCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);
        ArgumentOutOfRangeException.ThrowIfNegative(outerHeightCount);
        ArgumentOutOfRangeException.ThrowIfNegative(innerHeightCount);

        SourcePath = sourcePath;
        Version = version;
        TileX = tileX;
        TileY = tileY;
        SourceFound = sourceFound;
        HasData = hasData;
        MinHeight = minHeight;
        MaxHeight = maxHeight;
        CenterHeight = centerHeight;
        NorthWestHeight = northWestHeight;
        NorthEastHeight = northEastHeight;
        SouthWestHeight = southWestHeight;
        SouthEastHeight = southEastHeight;
        OuterHeightCount = outerHeightCount;
        InnerHeightCount = innerHeightCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int TileX { get; }

    public int TileY { get; }

    public bool SourceFound { get; }

    public bool HasData { get; }

    public short? MinHeight { get; }

    public short? MaxHeight { get; }

    public short? CenterHeight { get; }

    public short? NorthWestHeight { get; }

    public short? NorthEastHeight { get; }

    public short? SouthWestHeight { get; }

    public short? SouthEastHeight { get; }

    public int OuterHeightCount { get; }

    public int InnerHeightCount { get; }

    public int? HeightRange => MinHeight.HasValue && MaxHeight.HasValue ? MaxHeight.Value - MinHeight.Value : null;

    public static WorldWdlTileData Missing(string sourcePath, int tileX, int tileY)
    {
        return new WorldWdlTileData(sourcePath, null, tileX, tileY, sourceFound: false, hasData: false, null, null, null, null, null, null, null, 0, 0);
    }
}