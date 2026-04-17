using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class WdlSummaryReader
{
    private const int GridSize = 64;
    private const int TileCount = GridSize * GridSize;
    private const int MareHeightCount = WdlTileSummary.OuterHeightCount + WdlTileSummary.InnerHeightCount;
    private const int MarePayloadBytes = MareHeightCount * sizeof(short);
    private const int ChunkHeaderBytes = 8;

    private static readonly uint MverTag = ReadTag("REVM");
    private static readonly uint MverAlternateTag = ReadTag("MVER");
    private static readonly uint MaofTag = ReadTag("FOAM");
    private static readonly uint MaofAlternateTag = ReadTag("MAOF");
    private static readonly uint MareTag = ReadTag("ERAM");
    private static readonly uint MareAlternateTag = ReadTag("MARE");

    public static WdlSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WdlSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        byte[] data = ReadAllBytes(stream);
        if (data.Length < ChunkHeaderBytes + sizeof(uint))
            throw new InvalidDataException($"WDL file '{sourcePath}' is too small to contain a valid header.");

        uint? version = null;
        int scanOffset = 0;
        if (TryReadVersion(data, out uint detectedVersion, out int nextOffset))
        {
            version = detectedVersion;
            scanOffset = nextOffset;
        }

        if (!TryFindChunk(data, scanOffset, MaofTag, MaofAlternateTag, out int maofDataOffset, out uint maofSize))
            throw new InvalidDataException($"WDL file '{sourcePath}' does not contain a valid MAOF chunk.");

        int requiredMaofBytes = TileCount * sizeof(uint);
        if (maofSize < requiredMaofBytes || maofDataOffset + requiredMaofBytes > data.Length)
            throw new InvalidDataException($"WDL file '{sourcePath}' has an invalid MAOF payload size of {maofSize} bytes.");

        WdlTileSummary?[] tiles = new WdlTileSummary?[TileCount];
        for (int tileIndex = 0; tileIndex < TileCount; tileIndex++)
        {
            uint rawOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(maofDataOffset + (tileIndex * sizeof(uint)), sizeof(uint)));
            if (rawOffset == 0 || rawOffset > int.MaxValue)
                continue;

            int tileOffset = checked((int)rawOffset);
            int tileX = tileIndex % GridSize;
            int tileY = tileIndex / GridSize;
            if (TryReadTile(data, tileOffset, tileX, tileY, out WdlTileSummary? tile) && tile is not null)
                tiles[tileIndex] = tile;
        }

        return new WdlSummary(sourcePath, version, tiles);
    }

    private static bool TryReadVersion(byte[] data, out uint version, out int nextOffset)
    {
        if (!TryReadChunkHeader(data, 0, out uint tag, out uint payloadSize) || !Matches(tag, MverTag, MverAlternateTag))
        {
            version = 0;
            nextOffset = 0;
            return false;
        }

        if (payloadSize < sizeof(uint) || ChunkHeaderBytes + payloadSize > data.Length)
            throw new InvalidDataException("WDL MVER chunk payload is truncated.");

        version = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(ChunkHeaderBytes, sizeof(uint)));
        nextOffset = checked((int)(ChunkHeaderBytes + payloadSize));
        return true;
    }

    private static bool TryReadTile(byte[] data, int tileOffset, int tileX, int tileY, out WdlTileSummary? tile)
    {
        tile = null;
        if (tileOffset < 0 || tileOffset >= data.Length)
            return false;

        int payloadOffset = tileOffset;
        if (TryReadChunkHeader(data, tileOffset, out uint tag, out uint payloadSize) && Matches(tag, MareTag, MareAlternateTag))
        {
            if (payloadSize < MarePayloadBytes || tileOffset + ChunkHeaderBytes + MarePayloadBytes > data.Length)
                return false;

            payloadOffset += ChunkHeaderBytes;
        }
        else if (tileOffset + MarePayloadBytes > data.Length)
        {
            return false;
        }

        short[] outerHeights = new short[WdlTileSummary.OuterHeightCount];
        short[] innerHeights = new short[WdlTileSummary.InnerHeightCount];
        short minHeight = short.MaxValue;
        short maxHeight = short.MinValue;

        int cursor = payloadOffset;
        for (int index = 0; index < outerHeights.Length; index++)
        {
            short value = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(cursor, sizeof(short)));
            outerHeights[index] = value;
            minHeight = short.Min(minHeight, value);
            maxHeight = short.Max(maxHeight, value);
            cursor += sizeof(short);
        }

        for (int index = 0; index < innerHeights.Length; index++)
        {
            short value = BinaryPrimitives.ReadInt16LittleEndian(data.AsSpan(cursor, sizeof(short)));
            innerHeights[index] = value;
            minHeight = short.Min(minHeight, value);
            maxHeight = short.Max(maxHeight, value);
            cursor += sizeof(short);
        }

        tile = new WdlTileSummary(tileX, tileY, outerHeights, innerHeights, minHeight, maxHeight);
        return true;
    }

    private static bool TryFindChunk(byte[] data, int startOffset, uint tag, uint alternateTag, out int chunkDataOffset, out uint chunkSize)
    {
        int offset = Math.Max(0, startOffset);
        while (TryReadChunkHeader(data, offset, out uint currentTag, out uint currentSize))
        {
            if (Matches(currentTag, tag, alternateTag) && offset + ChunkHeaderBytes + currentSize <= data.Length)
            {
                chunkDataOffset = offset + ChunkHeaderBytes;
                chunkSize = currentSize;
                return true;
            }

            if (currentSize == 0)
            {
                offset += ChunkHeaderBytes;
                continue;
            }

            long nextOffset = offset + ChunkHeaderBytes + currentSize;
            if (nextOffset > data.Length)
                break;

            offset = checked((int)nextOffset);
        }

        for (offset = Math.Max(0, startOffset); offset + ChunkHeaderBytes <= data.Length; offset++)
        {
            uint currentTag = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
            if (!Matches(currentTag, tag, alternateTag))
                continue;

            uint currentSize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + sizeof(uint), sizeof(uint)));
            if (offset + ChunkHeaderBytes + currentSize > data.Length)
                continue;

            chunkDataOffset = offset + ChunkHeaderBytes;
            chunkSize = currentSize;
            return true;
        }

        chunkDataOffset = 0;
        chunkSize = 0;
        return false;
    }

    private static bool TryReadChunkHeader(byte[] data, int offset, out uint tag, out uint payloadSize)
    {
        if (offset < 0 || offset + ChunkHeaderBytes > data.Length)
        {
            tag = 0;
            payloadSize = 0;
            return false;
        }

        tag = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
        payloadSize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + sizeof(uint), sizeof(uint)));
        return true;
    }

    private static bool Matches(uint tag, uint primary, uint alternate)
    {
        return tag == primary || tag == alternate;
    }

    private static uint ReadTag(string text)
    {
        return BinaryPrimitives.ReadUInt32LittleEndian(System.Text.Encoding.ASCII.GetBytes(text));
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        long previousPosition = stream.CanSeek ? stream.Position : 0;
        try
        {
            if (stream.CanSeek)
                stream.Position = 0;

            using MemoryStream memory = new();
            stream.CopyTo(memory);
            return memory.ToArray();
        }
        finally
        {
            if (stream.CanSeek)
                stream.Position = previousPosition;
        }
    }
}