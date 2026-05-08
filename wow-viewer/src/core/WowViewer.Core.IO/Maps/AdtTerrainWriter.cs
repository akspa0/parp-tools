using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtTerrainWriter
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const int TileHeightmapSize = 257;
    private const int HalfStepsPerChunk = 16;
    private const int McvtSampleCount = 145;
    private const int McinEntrySize = 16;
    private const int ExpectedChunkCount = 256;
    private const int McnrConsumedSize = 0x1C0;
    private const int McnrSampleByteCount = McvtSampleCount * 3;

    public static byte[] ApplyHeightmap(string path, IReadOnlyList<float> tileHeightmap257)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(tileHeightmap257);

        return ApplyHeightmap(File.ReadAllBytes(path), Path.GetFullPath(path), tileHeightmap257);
    }

    public static byte[] ApplyHeightmap(byte[] sourceBytes, string sourcePath, IReadOnlyList<float> tileHeightmap257)
    {
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(tileHeightmap257);
        if (tileHeightmap257.Count != TileHeightmapSize * TileHeightmapSize)
            throw new ArgumentException($"Terrain heightmap must contain exactly {TileHeightmapSize * TileHeightmapSize} samples.", nameof(tileHeightmap257));

        byte[] updatedBytes = sourceBytes.ToArray();
        using MemoryStream stream = new(updatedBytes, writable: true);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"ADT terrain writing requires a root ADT file, but found {fileSummary.Kind}.");

        IReadOnlyList<MapChunkLocation> terrainChunks = ResolveTerrainChunkLocations(stream, fileSummary);
        if (terrainChunks.Count == 0)
            throw new InvalidDataException("ADT terrain writing requires at least one MCNK chunk.");

        foreach (MapChunkLocation chunk in terrainChunks)
            PatchChunk(updatedBytes, chunk, tileHeightmap257);

        return updatedBytes;
    }

    public static void Write(string inputPath, string outputPath, IReadOnlyList<float> tileHeightmap257)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(tileHeightmap257);

        byte[] updatedBytes = ApplyHeightmap(inputPath, tileHeightmap257);
        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.WriteAllBytes(outputPath, updatedBytes);
    }

    private static void PatchChunk(byte[] bytes, MapChunkLocation chunk, IReadOnlyList<float> tileHeightmap257)
    {
        int payloadOffset = checked((int)chunk.DataOffset);
        int payloadSize = checked((int)chunk.Size);
        Span<byte> payload = bytes.AsSpan(payloadOffset, payloadSize);
        if (payload.Length < RootMcnkHeaderSize)
            throw new InvalidDataException("MCNK payload is smaller than the expected root header size.");

        int chunkX = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x04, 4)));
        int chunkY = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x08, 4)));
        float baseHeight = BinaryPrimitives.ReadSingleLittleEndian(payload.Slice(0x70, 4));

        int mcvtDataOffset = LocateMcvtDataOffset(payload);
        if (mcvtDataOffset < 0)
            throw new InvalidDataException($"MCNK ({chunkX},{chunkY}) has no MCVT subchunk to patch.");

        int mcnrDataOffset = LocateMcnrDataOffset(payload);
        if (mcnrDataOffset < 0)
            throw new InvalidDataException($"MCNK ({chunkX},{chunkY}) has no MCNR subchunk to patch.");

        for (int sampleIndex = 0; sampleIndex < McvtSampleCount; sampleIndex++)
        {
            ResolveTileSampleCoordinates(chunkX, chunkY, sampleIndex, out int sampleX, out int sampleY);
            float absoluteHeight = tileHeightmap257[(sampleY * TileHeightmapSize) + sampleX];
            float rawHeight = absoluteHeight - baseHeight;
            int sampleOffset = payloadOffset + mcvtDataOffset + (sampleIndex * sizeof(float));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(sampleOffset, sizeof(float)), BitConverter.SingleToInt32Bits(rawHeight));

            Vector3 normal = AdtTerrainMath.ComputeNormal(tileHeightmap257, sampleX, sampleY);
            int normalOffset = payloadOffset + mcnrDataOffset + (sampleIndex * 3);
            bytes[normalOffset + 0] = EncodeNormalComponent(normal.X);
            bytes[normalOffset + 1] = EncodeNormalComponent(normal.Z);
            bytes[normalOffset + 2] = EncodeNormalComponent(normal.Y);
        }
    }

    private static int LocateMcvtDataOffset(ReadOnlySpan<byte> payload)
    {
        uint headerMcalSize = payload.Length >= 0x2C ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x28, 4)) : 0;
        uint headerMcshSize = payload.Length >= 0x34 ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x30, 4)) : 0;

        int position = RootMcnkSubchunkOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = unchecked((int)header.Size);
            int consumedSize = declaredSize;
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);
            else if (header.Id == AdtChunkIds.Mcal && headerMcalSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, checked((int)headerMcalSize - ChunkHeader.SizeInBytes));
            else if (header.Id == AdtChunkIds.Mcsh && headerMcshSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, checked((int)headerMcshSize - ChunkHeader.SizeInBytes));

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcvt)
            {
                if (header.Size < McvtSampleCount * sizeof(float))
                    throw new InvalidDataException("MCVT chunk is smaller than the expected 145-float terrain payload.");

                return position + ChunkHeader.SizeInBytes;
            }

            position = checked((int)nextOffset);
        }

        return -1;
    }

    private static int LocateMcnrDataOffset(ReadOnlySpan<byte> payload)
    {
        int position = RootMcnkSubchunkOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int declaredSize = unchecked((int)header.Size);
            int consumedSize = declaredSize;
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcnr)
            {
                if (header.Size < McnrSampleByteCount)
                    throw new InvalidDataException("MCNR chunk is smaller than the expected 145-normal payload.");

                return position + ChunkHeader.SizeInBytes;
            }

            position = checked((int)nextOffset);
        }

        return -1;
    }

    private static byte EncodeNormalComponent(float value)
    {
        int scaled = (int)MathF.Round(Math.Clamp(value, -1f, 1f) * 127f);
        return unchecked((byte)(sbyte)scaled);
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

    private static void ResolveTileSampleCoordinates(int chunkX, int chunkY, int sampleIndex, out int sampleX, out int sampleY)
    {
        GetVertexPosition(sampleIndex, out int row, out int col, out bool isInner);
        int localX = isInner ? (col * 2) + 1 : col * 2;
        int localY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

        sampleX = (chunkX * HalfStepsPerChunk) + localX;
        sampleY = (chunkY * HalfStepsPerChunk) + localY;
    }

    private static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2) != 0;
                return;
            }

            remaining -= rowSize;
        }
    }
}