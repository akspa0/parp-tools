using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class WdtSummaryReader
{
    private const int WdtTileCount = 64 * 64;
    private const uint MainFlagHasAdt = 0x1;
    private const uint MainFlagAllWater = 0x2;
    private const uint MainFlagLoaded = 0x4;
    private const uint KnownMainFlagMask = MainFlagHasAdt | MainFlagAllWater | MainFlagLoaded;
    private const int StandardMainCellSize = 8;
    private const int AlphaMainCellSize = 16;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;

    public static WdtSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static WdtSummary Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Wdt)
            throw new InvalidDataException($"WDT semantic summary requires a WDT file, but found {fileSummary.Kind}.");

        byte[] mphdData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mphd) ?? [];
        byte[] mainData = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Main) ?? [];
        int mainCellSize = InferMainCellSize(mainData);

        return new WdtSummary(
            fileSummary.SourcePath,
            isWmoBased: IsWmoBased(mphdData),
            tilesWithData: CountTilesWithData(mainData),
            totalTiles: WdtTileCount,
            mainCellSizeBytes: mainCellSize,
            doodadNameCount: MapSummaryReaderCommon.CountStringEntries(MapSummaryReaderCommon.ReadFirstAvailableChunkPayload(stream, fileSummary, [MapChunkIds.Mdnm, MapChunkIds.Mmdx])),
            worldModelNameCount: MapSummaryReaderCommon.CountStringEntries(MapSummaryReaderCommon.ReadFirstAvailableChunkPayload(stream, fileSummary, [MapChunkIds.Monm, MapChunkIds.Mwmo])),
            doodadPlacementCount: MapSummaryReaderCommon.CountPlacements(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mddf), MddfEntrySize),
            worldModelPlacementCount: MapSummaryReaderCommon.CountPlacements(MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Modf), ModfEntrySize),
            mainFlags: ReadMainFlagsSummary(mainData, mainCellSize));
    }

    private static bool IsWmoBased(byte[] mphdData)
    {
        if (mphdData.Length >= 12 && BinaryPrimitives.ReadUInt32LittleEndian(mphdData.AsSpan(8, 4)) == 2)
            return true;

        if (mphdData.Length >= 4)
            return (BinaryPrimitives.ReadUInt32LittleEndian(mphdData.AsSpan(0, 4)) & 0x1u) != 0;

        return false;
    }

    private static int CountTilesWithData(byte[] mainData)
    {
        int cellSize = InferMainCellSize(mainData);
        if (cellSize < sizeof(uint) || mainData.Length < cellSize)
            return 0;

        int occupied = 0;
        for (int index = 0; index < WdtTileCount; index++)
        {
            int offset = index * cellSize;
            if (offset + sizeof(uint) > mainData.Length)
                break;

            if (BinaryPrimitives.ReadUInt32LittleEndian(mainData.AsSpan(offset, sizeof(uint))) != 0)
                occupied++;
        }

        return occupied;
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

    private static WdtMainFlagsSummary? ReadMainFlagsSummary(byte[] mainData, int mainCellSize)
    {
        if (mainCellSize != StandardMainCellSize || mainData.Length < mainCellSize)
            return null;

        int cellsWithAnyFlags = 0;
        int cellsWithHasAdt = 0;
        int cellsWithAllWater = 0;
        int cellsWithLoaded = 0;
        int cellsWithUnknownFlags = 0;
        int cellsWithAsyncId = 0;
        Dictionary<uint, int> distinctNonZeroValues = [];

        for (int index = 0; index < WdtTileCount; index++)
        {
            int offset = index * mainCellSize;
            if (offset + StandardMainCellSize > mainData.Length)
                break;

            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(mainData.AsSpan(offset, sizeof(uint)));
            uint asyncId = BinaryPrimitives.ReadUInt32LittleEndian(mainData.AsSpan(offset + sizeof(uint), sizeof(uint)));

            if (flags != 0)
            {
                cellsWithAnyFlags++;
                distinctNonZeroValues.TryGetValue(flags, out int existingCount);
                distinctNonZeroValues[flags] = existingCount + 1;
            }

            if ((flags & MainFlagHasAdt) != 0)
                cellsWithHasAdt++;

            if ((flags & MainFlagAllWater) != 0)
                cellsWithAllWater++;

            if ((flags & MainFlagLoaded) != 0)
                cellsWithLoaded++;

            if ((flags & ~KnownMainFlagMask) != 0)
                cellsWithUnknownFlags++;

            if (asyncId != 0)
                cellsWithAsyncId++;
        }

        List<WdtMainFlagValueSummary> distinctValues = [];
        foreach (KeyValuePair<uint, int> pair in distinctNonZeroValues.OrderBy(static pair => pair.Key))
            distinctValues.Add(new WdtMainFlagValueSummary(pair.Key, pair.Value));

        return new WdtMainFlagsSummary(
            cellsWithAnyFlags,
            cellsWithHasAdt,
            cellsWithAllWater,
            cellsWithLoaded,
            cellsWithUnknownFlags,
            cellsWithAsyncId,
            distinctValues);
    }
}
