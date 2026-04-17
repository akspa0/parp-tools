using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Wdl;

public static class WorldWdlTileBuilder
{
    public static WorldWdlTileData Read(string path, int tileX, int tileY)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        WdlSummary summary = WdlSummaryReader.Read(path);
        return Read(summary, tileX, tileY);
    }

    public static WorldWdlTileData Read(Stream stream, string sourcePath, int tileX, int tileY)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        WdlSummary summary = WdlSummaryReader.Read(stream, sourcePath);
        return Read(summary, tileX, tileY);
    }

    public static WorldWdlTileData Read(WdlSummary summary, int tileX, int tileY)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentOutOfRangeException.ThrowIfNegative(tileX);
        ArgumentOutOfRangeException.ThrowIfNegative(tileY);

        if (!summary.TryGetTile(tileX, tileY, out WdlTileSummary? tile) || tile is null)
            return new WorldWdlTileData(summary.SourcePath, summary.Version, tileX, tileY, sourceFound: true, hasData: false, null, null, null, null, null, null, null, 0, 0);

        return new WorldWdlTileData(
            summary.SourcePath,
            summary.Version,
            tileX,
            tileY,
            sourceFound: true,
            hasData: true,
            tile.MinHeight,
            tile.MaxHeight,
            tile.GetOuterHeight(8, 8),
            tile.GetOuterHeight(0, 0),
            tile.GetOuterHeight(16, 0),
            tile.GetOuterHeight(0, 16),
            tile.GetOuterHeight(16, 16),
            WdlTileSummary.OuterHeightCount,
            WdlTileSummary.InnerHeightCount);
    }
}