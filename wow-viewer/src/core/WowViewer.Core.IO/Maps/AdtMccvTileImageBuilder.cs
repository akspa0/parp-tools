using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtMccvTileImageBuilder
{
    public const int TileChunks = 16;
    public const int TileImageSize = 145;

    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const int VerticesPerChunk = 145;
    private const int McinEntrySize = 16;
    private const int ExpectedChunkCount = 256;
    private const int McnrConsumedSize = 0x1C0;
    private const byte NeutralChannelValue = 127;

    public static IReadOnlyDictionary<int, byte[]> ReadChunkColors(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return ReadChunkColors(File.ReadAllBytes(path), Path.GetFullPath(path));
    }

    public static IReadOnlyDictionary<int, byte[]> ReadChunkColors(byte[] sourceBytes, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        using MemoryStream stream = new(sourceBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"MCCV extraction requires a root ADT file, but found {fileSummary.Kind}.");

        Dictionary<int, byte[]> chunkColors = [];
        foreach (MapChunkLocation chunk in ResolveTerrainChunkLocations(stream, fileSummary))
        {
            byte[]? colors = ReadChunkColors(sourceBytes, chunk);
            if (colors is null)
                continue;

            ReadOnlySpan<byte> payload = sourceBytes.AsSpan(checked((int)chunk.DataOffset), checked((int)chunk.Size));
            int chunkX = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x04, 4)));
            int chunkY = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x08, 4)));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            chunkColors[(chunkY * TileChunks) + chunkX] = colors;
        }

        return chunkColors;
    }

    public static byte[] RenderTileImageRgba(IReadOnlyDictionary<int, byte[]> chunkColors)
    {
        ArgumentNullException.ThrowIfNull(chunkColors);

        byte[] tileImage = new byte[TileImageSize * TileImageSize * 4];
        byte[] neutralChunk = CreateNeutralChunkColors();
        for (int y = 0; y < TileImageSize; y++)
        {
            float v = y / (float)(TileImageSize - 1);
            float chunkY = v * TileChunks;
            int chunkIy = Math.Clamp((int)chunkY, 0, TileChunks - 1);
            float localY = Math.Clamp(chunkY - chunkIy, 0f, 1f);

            for (int x = 0; x < TileImageSize; x++)
            {
                float u = x / (float)(TileImageSize - 1);
                float chunkX = u * TileChunks;
                int chunkIx = Math.Clamp((int)chunkX, 0, TileChunks - 1);
                float localX = Math.Clamp(chunkX - chunkIx, 0f, 1f);
                int chunkIndex = (chunkIy * TileChunks) + chunkIx;

                byte[] chunkData = chunkColors.TryGetValue(chunkIndex, out byte[]? data)
                    ? NormalizeChunkColors(data)
                    : neutralChunk;

                WritePixel(tileImage, x, y, SampleChunkImage(chunkData, localX, localY));
            }
        }

        return tileImage;
    }

    private static byte[]? ReadChunkColors(byte[] sourceBytes, MapChunkLocation chunk)
    {
        ReadOnlySpan<byte> payload = sourceBytes.AsSpan(checked((int)chunk.DataOffset), checked((int)chunk.Size));
        if (payload.Length < RootMcnkHeaderSize)
            return null;

        int mccvDataOffset = LocateMccvDataOffset(payload);
        if (mccvDataOffset < 0)
            return null;

        return payload.Slice(mccvDataOffset, VerticesPerChunk * 4).ToArray();
    }

    private static int LocateMccvDataOffset(ReadOnlySpan<byte> payload)
    {
        uint headerMcalSize = payload.Length >= 0x2C ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x28, 4)) : 0;
        uint headerMcshSize = payload.Length >= 0x34 ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x30, 4)) : 0;

        int position = RootMcnkSubchunkOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int consumedSize = unchecked((int)header.Size);
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);
            else if (header.Id == AdtChunkIds.Mcal && headerMcalSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, checked((int)headerMcalSize - ChunkHeader.SizeInBytes));
            else if (header.Id == AdtChunkIds.Mcsh && headerMcshSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, checked((int)headerMcshSize - ChunkHeader.SizeInBytes));

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mccv)
            {
                if (header.Size < VerticesPerChunk * 4)
                    throw new InvalidDataException("MCCV chunk is smaller than the expected 145-color payload.");

                return position + ChunkHeader.SizeInBytes;
            }

            position = checked((int)nextOffset);
        }

        return -1;
    }

    private static IReadOnlyList<MapChunkLocation> ResolveTerrainChunkLocations(Stream stream, MapFileSummary fileSummary)
    {
        List<MapChunkLocation> topLevelChunks = fileSummary.Chunks
            .Where(static chunk => chunk.Id == MapChunkIds.Mcnk)
            .ToList();

        if (topLevelChunks.Count >= ExpectedChunkCount || !fileSummary.HasChunk(MapChunkIds.Mcin))
            return topLevelChunks;

        MapChunkLocation mcinChunk = fileSummary.Chunks.First(chunk => chunk.Id == MapChunkIds.Mcin);
        byte[] mcinPayload = ReadChunkPayload(stream, mcinChunk);
        if (mcinPayload.Length < McinEntrySize)
            return topLevelChunks;

        List<MapChunkLocation> resolvedChunks = new(ExpectedChunkCount);
        for (int index = 0; index < ExpectedChunkCount && ((index + 1) * McinEntrySize) <= mcinPayload.Length; index++)
        {
            int entryOffset = index * McinEntrySize;
            uint chunkOffset = BinaryPrimitives.ReadUInt32LittleEndian(mcinPayload.AsSpan(entryOffset, 4));
            if (chunkOffset == 0)
                continue;

            long headerOffset = chunkOffset;
            if (!TryReadChunkHeader(stream, headerOffset, out ChunkHeader header))
                continue;

            if (header.Id != MapChunkIds.Mcnk || header.Size < RootMcnkHeaderSize)
                continue;

            long dataOffset = headerOffset + ChunkHeader.SizeInBytes;
            if (dataOffset > stream.Length || dataOffset + header.Size > stream.Length)
                continue;

            resolvedChunks.Add(new MapChunkLocation(MapChunkIds.Mcnk, header.Size, headerOffset, dataOffset));
        }

        return resolvedChunks.Count > topLevelChunks.Count ? resolvedChunks : topLevelChunks;
    }

    private static bool TryReadChunkHeader(Stream stream, long headerOffset, out ChunkHeader header)
    {
        long previousPosition = stream.Position;
        try
        {
            if (headerOffset < 0 || headerOffset > stream.Length - ChunkHeader.SizeInBytes)
            {
                header = default;
                return false;
            }

            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];
            stream.Position = headerOffset;
            stream.ReadExactly(headerBytes);
            return ChunkHeaderReader.TryRead(headerBytes, out header);
        }
        finally
        {
            stream.Position = previousPosition;
        }
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

    private static byte[] NormalizeChunkColors(byte[] raw)
    {
        if (raw.Length >= VerticesPerChunk * 4)
            return raw;

        return CreateNeutralChunkColors();
    }

    private static byte[] CreateNeutralChunkColors()
    {
        byte[] data = new byte[VerticesPerChunk * 4];
        for (int index = 0; index < VerticesPerChunk; index++)
        {
            int offset = index * 4;
            data[offset + 0] = NeutralChannelValue;
            data[offset + 1] = NeutralChannelValue;
            data[offset + 2] = NeutralChannelValue;
            data[offset + 3] = NeutralChannelValue;
        }

        return data;
    }

    private static (float R, float G, float B, float A) SampleChunkImage(byte[] chunkData, float localX, float localY)
    {
        float gridX = localX * 8f;
        float gridY = localY * 8f;

        int ix = Math.Clamp((int)gridX, 0, 7);
        int iy = Math.Clamp((int)gridY, 0, 7);
        float dx = Math.Clamp(gridX - ix, 0f, 1f);
        float dy = Math.Clamp(gridY - iy, 0f, 1f);

        (byte R, byte G, byte B, byte A) topLeft = GetVertex(chunkData, (iy * 9) + ix);
        (byte R, byte G, byte B, byte A) topRight = GetVertex(chunkData, (iy * 9) + ix + 1);
        (byte R, byte G, byte B, byte A) bottomLeft = GetVertex(chunkData, ((iy + 1) * 9) + ix);
        (byte R, byte G, byte B, byte A) bottomRight = GetVertex(chunkData, ((iy + 1) * 9) + ix + 1);
        (byte R, byte G, byte B, byte A) center = GetVertex(chunkData, 81 + (iy * 8) + ix);

        if (dy < dx && dy < 1.0f - dx)
            return Combine(topLeft, topRight, center, 1 - dx - dy, dx - dy, 2 * dy);

        if (dy > dx && dy > 1.0f - dx)
            return Combine(bottomLeft, bottomRight, center, dy - dx, dx + dy - 1, 2 * (1 - dy));

        if (dx < dy && dx < 1.0f - dy)
            return Combine(topLeft, bottomLeft, center, 1 - dx - dy, dy - dx, 2 * dx);

        return Combine(topRight, bottomRight, center, dx - dy, dy + dx - 1, 2 * (1 - dx));
    }

    private static (float R, float G, float B, float A) Combine(
        (byte R, byte G, byte B, byte A) colorA,
        (byte R, byte G, byte B, byte A) colorB,
        (byte R, byte G, byte B, byte A) colorC,
        float weightA,
        float weightB,
        float weightC)
    {
        return (
            Math.Clamp(colorA.R * weightA + colorB.R * weightB + colorC.R * weightC, 0f, 255f),
            Math.Clamp(colorA.G * weightA + colorB.G * weightB + colorC.G * weightC, 0f, 255f),
            Math.Clamp(colorA.B * weightA + colorB.B * weightB + colorC.B * weightC, 0f, 255f),
            Math.Clamp(colorA.A * weightA + colorB.A * weightB + colorC.A * weightC, 0f, 255f));
    }

    private static (byte R, byte G, byte B, byte A) GetVertex(byte[] chunkData, int vertexIndex)
    {
        int offset = vertexIndex * 4;
        return (chunkData[offset + 0], chunkData[offset + 1], chunkData[offset + 2], chunkData[offset + 3]);
    }

    private static void WritePixel(byte[] image, int x, int y, (float R, float G, float B, float A) color)
    {
        int offset = ((y * TileImageSize) + x) * 4;
        image[offset + 0] = ToByte(color.R);
        image[offset + 1] = ToByte(color.G);
        image[offset + 2] = ToByte(color.B);
        image[offset + 3] = ToByte(color.A);
    }

    private static byte ToByte(float value) => (byte)Math.Clamp((int)MathF.Round(value), 0, 255);
}