using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Passes;

public static class WorldTileStageSummaryBuilder
{
    public static WorldTileStageSummary Read(string path, int wdlVisibleTileCount = 1)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary, wdlVisibleTileCount);
    }

    public static WorldTileStageSummary Read(Stream stream, MapFileSummary fileSummary, int wdlVisibleTileCount = 1)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"World tile stage summary requires a root ADT file, but found {fileSummary.Kind}.");

        AdtSummary adtSummary = AdtSummaryReader.Read(stream, fileSummary);
        stream.Position = 0;
        AdtMcnkSummary mcnkSummary = AdtMcnkSummaryReader.Read(stream, fileSummary);
        stream.Position = 0;
        AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, fileSummary);

        int liquidChunkCount = liquidFile.Chunks.Count(static chunk => chunk.Layers.Count > 0);
        int liquidLayerCount = liquidFile.Chunks.Sum(static chunk => chunk.Layers.Count);
        int visibleLiquidTileCount = liquidFile.Chunks.Sum(static chunk => chunk.Layers.Sum(static layer => layer.VisibleTileCount));

        return new WorldTileStageSummary(
            fileSummary.SourcePath,
            fileSummary.Kind,
            wdlVisibleTileCount,
            adtSummary.TerrainChunkCount,
            mcnkSummary.ChunksWithHoles,
            liquidChunkCount,
            liquidLayerCount,
            visibleLiquidTileCount,
            adtSummary.HasWater);
    }
}