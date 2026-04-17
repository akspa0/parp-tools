using System.Buffers.Binary;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Terrain;

public static class WorldTerrainTileBuilder
{
    private const int RootMcnkHeaderSize = 128;
    private const uint LiquidFlagMask = 0x3Cu;
    private const uint VertexColorFlagMask = 0x40u;

    public static WorldTerrainTileData Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static WorldTerrainTileData Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"World terrain tile builder requires a root ADT file, but found {fileSummary.Kind}.");

        List<WorldTerrainChunkData> chunks = new(fileSummary.CountChunks(MapChunkIds.Mcnk));
        int chunkOrdinal = 0;
        foreach (MapChunkLocation chunk in fileSummary.Chunks)
        {
            if (chunk.Id != MapChunkIds.Mcnk)
                continue;

            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
            {
                chunkOrdinal++;
                continue;
            }

            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x00, 4));
            int indexX = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x04, 4)));
            int indexY = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x08, 4)));
            int layerCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x0C, 4)));
            uint areaId = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x34, 4));
            ushort holes = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(0x3C, 2));

            chunks.Add(new WorldTerrainChunkData(
                chunkOrdinal,
                indexX,
                indexY,
                areaId,
                flags,
                layerCount,
                holes != 0,
                (flags & LiquidFlagMask) != 0,
                (flags & VertexColorFlagMask) != 0));
            chunkOrdinal++;
        }

        return new WorldTerrainTileData(fileSummary.SourcePath, fileSummary.Kind, chunks);
    }

    private static byte[] ReadChunkPayload(Stream stream, MapChunkLocation chunk)
    {
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