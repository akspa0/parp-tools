using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class WdtTileIndexReader
{
    private const int WdtTilesPerAxis = 64;
    private const int WdtTileCount = WdtTilesPerAxis * WdtTilesPerAxis;
    private const int StandardMainCellSize = 8;
    private const int AlphaMainCellSize = 16;

    public static IReadOnlyList<WdtTileCoordinate> ReadOccupiedTiles(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return ReadOccupiedTiles(stream, fileSummary);
    }

    public static IReadOnlyList<WdtTileCoordinate> ReadOccupiedTiles(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Wdt)
            throw new InvalidDataException($"WDT tile indexing requires a WDT file, but found {fileSummary.Kind}.");

        byte[] mainData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Main) ?? [];
        int mainCellSize = InferMainCellSize(mainData);
        if (mainCellSize < sizeof(uint) || mainData.Length < mainCellSize)
            return [];

        List<WdtTileCoordinate> occupiedTiles = [];
        for (int index = 0; index < WdtTileCount; index++)
        {
            int offset = index * mainCellSize;
            if (offset + sizeof(uint) > mainData.Length)
                break;

            uint value = BinaryPrimitives.ReadUInt32LittleEndian(mainData.AsSpan(offset, sizeof(uint)));
            if (value == 0)
                continue;

            occupiedTiles.Add(mainCellSize == AlphaMainCellSize
                ? new WdtTileCoordinate(index / WdtTilesPerAxis, index % WdtTilesPerAxis)
                : new WdtTileCoordinate(index % WdtTilesPerAxis, index / WdtTilesPerAxis));
        }

        return occupiedTiles;
    }

    private static int InferMainCellSize(byte[] mainData)
    {
        if (mainData.Length == WdtTileCount * AlphaMainCellSize)
            return AlphaMainCellSize;

        if (mainData.Length == WdtTileCount * StandardMainCellSize)
            return StandardMainCellSize;

        if (mainData.Length > 0 && mainData.Length % WdtTileCount == 0)
            return mainData.Length / WdtTileCount;

        return 0;
    }
}