using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtSummaryReader
{
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;

    public static AdtSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtSummary Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.Adt or MapFileKind.AdtTex or MapFileKind.AdtObj))
            throw new InvalidDataException($"ADT semantic summary requires an ADT-family file, but found {fileSummary.Kind}.");

        return new AdtSummary(
            fileSummary.SourcePath,
            fileSummary.Kind,
            terrainChunkCount: fileSummary.CountChunks(MapChunkIds.Mcnk),
            textureNameCount: MapSummaryReaderCommon.CountStringEntries(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mtex)),
            modelNameCount: MapSummaryReaderCommon.CountStringEntries(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mmdx)),
            worldModelNameCount: MapSummaryReaderCommon.CountStringEntries(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mwmo)),
            modelPlacementCount: MapSummaryReaderCommon.CountPlacements(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mddf), MddfEntrySize),
            worldModelPlacementCount: MapSummaryReaderCommon.CountPlacements(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Modf), ModfEntrySize),
            hasFlightBounds: fileSummary.HasChunk(MapChunkIds.Mfbo),
            hasWater: fileSummary.HasChunk(MapChunkIds.Mh2o),
            hasTextureParams: fileSummary.HasChunk(MapChunkIds.Mamp),
            hasTextureFlags: fileSummary.HasChunk(MapChunkIds.Mtxf));
    }
}
