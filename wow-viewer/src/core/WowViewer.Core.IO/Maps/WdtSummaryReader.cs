using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class WdtSummaryReader
{
    private const int WdtTileCount = 64 * 64;
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

        byte[] mphdData = ReadChunkPayload(stream, fileSummary, MapChunkIds.Mphd) ?? [];
        byte[] mainData = ReadChunkPayload(stream, fileSummary, MapChunkIds.Main) ?? [];

        return new WdtSummary(
            fileSummary.SourcePath,
            isWmoBased: IsWmoBased(mphdData),
            tilesWithData: CountTilesWithData(mainData),
            totalTiles: WdtTileCount,
            mainCellSizeBytes: InferMainCellSize(mainData),
            doodadNameCount: CountStringEntries(ReadFirstAvailableChunkPayload(stream, fileSummary, [MapChunkIds.Mdnm, MapChunkIds.Mmdx])),
            worldModelNameCount: CountStringEntries(ReadFirstAvailableChunkPayload(stream, fileSummary, [MapChunkIds.Monm, MapChunkIds.Mwmo])),
            doodadPlacementCount: CountPlacements(ReadChunkPayload(stream, fileSummary, MapChunkIds.Mddf), MddfEntrySize),
            worldModelPlacementCount: CountPlacements(ReadChunkPayload(stream, fileSummary, MapChunkIds.Modf), ModfEntrySize));
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

    private static int CountStringEntries(byte[]? payload)
    {
        if (payload is not { Length: > 0 })
            return 0;

        int count = 0;
        bool sawText = false;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] == 0)
            {
                if (sawText)
                {
                    count++;
                    sawText = false;
                }

                continue;
            }

            sawText = true;
        }

        if (sawText)
            count++;

        return count;
    }

    private static int CountPlacements(byte[]? payload, int stride)
    {
        if (payload is not { Length: > 0 } || stride <= 0)
            return 0;

        return payload.Length / stride;
    }

    private static byte[]? ReadFirstAvailableChunkPayload(Stream stream, MapFileSummary fileSummary, IReadOnlyList<WowViewer.Core.Chunks.FourCC> ids)
    {
        foreach (WowViewer.Core.Chunks.FourCC id in ids)
        {
            byte[]? payload = ReadChunkPayload(stream, fileSummary, id);
            if (payload is { Length: > 0 })
                return payload;
        }

        return null;
    }

    private static byte[]? ReadChunkPayload(Stream stream, MapFileSummary fileSummary, WowViewer.Core.Chunks.FourCC id)
    {
        MapChunkLocation chunk = default;
        bool found = false;
        foreach (MapChunkLocation location in fileSummary.Chunks)
        {
            if (location.Id != id)
                continue;

            chunk = location;
            found = true;
            break;
        }

        if (!found)
            return null;

        if (!stream.CanSeek)
            throw new ArgumentException("Chunk payload reading requires a seekable stream.", nameof(stream));

        long previousPosition = stream.Position;
        try
        {
            stream.Position = chunk.DataOffset;
            byte[] payload = new byte[chunk.Size];
            stream.ReadExactly(payload);
            return payload;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }
}