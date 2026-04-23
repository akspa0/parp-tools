using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtTerrainPatchAudit
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const int McinEntrySize = 16;
    private const int ExpectedChunkCount = 256;
    private const int McnrConsumedSize = 0x1C0;
    private const int McnrSampleByteCount = 145 * 3;
    private const int McvtSampleByteCount = 145 * sizeof(float);

    public static AdtChunkChangeAudit AnalyzeChunkChanges(string inputAdtPath, string outputAdtPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputAdtPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputAdtPath);

        return AnalyzeChunkChanges(
            File.ReadAllBytes(inputAdtPath),
            File.ReadAllBytes(outputAdtPath),
            Path.GetFullPath(inputAdtPath));
    }

    public static AdtChunkChangeAudit AnalyzeChunkChanges(byte[] sourceBytes, byte[] updatedBytes, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentNullException.ThrowIfNull(updatedBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        using MemoryStream sourceStream = new(sourceBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(sourceStream, sourcePath);
        IReadOnlyList<MapChunkLocation> terrainChunks = ResolveTerrainChunkLocations(sourceStream, fileSummary);

        int changedMcvtChunkCount = 0;
        int changedMcnrChunkCount = 0;
        foreach (MapChunkLocation chunk in terrainChunks)
        {
            int sourcePayloadOffset = checked((int)chunk.DataOffset);
            int sourcePayloadSize = checked((int)chunk.Size);
            ReadOnlySpan<byte> sourcePayload = sourceBytes.AsSpan(sourcePayloadOffset, sourcePayloadSize);
            ReadOnlySpan<byte> updatedPayload = updatedBytes.AsSpan(sourcePayloadOffset, sourcePayloadSize);

            int mcvtDataOffset = LocateMcvtDataOffset(sourcePayload);
            int mcnrDataOffset = LocateMcnrDataOffset(sourcePayload);
            if (mcvtDataOffset >= 0 && !sourcePayload.Slice(mcvtDataOffset, McvtSampleByteCount).SequenceEqual(updatedPayload.Slice(mcvtDataOffset, McvtSampleByteCount)))
                changedMcvtChunkCount++;

            if (mcnrDataOffset >= 0 && !sourcePayload.Slice(mcnrDataOffset, McnrSampleByteCount).SequenceEqual(updatedPayload.Slice(mcnrDataOffset, McnrSampleByteCount)))
                changedMcnrChunkCount++;
        }

        return new AdtChunkChangeAudit(terrainChunks.Count, changedMcvtChunkCount, changedMcnrChunkCount);
    }

    public static AdtSeamAudit CreateSeamAudit(
        (int TileX, int TileY) tileCoordinate,
        IReadOnlyList<float> originalHeightmap,
        IReadOnlyList<float> patchedHeightmap,
        IReadOnlyDictionary<(int TileX, int TileY), float[]> referenceHeightmaps)
    {
        ArgumentNullException.ThrowIfNull(originalHeightmap);
        ArgumentNullException.ThrowIfNull(patchedHeightmap);
        ArgumentNullException.ThrowIfNull(referenceHeightmaps);
        ValidateHeightmap(originalHeightmap, nameof(originalHeightmap));
        ValidateHeightmap(patchedHeightmap, nameof(patchedHeightmap));

        SeamAccumulator preHeight = new();
        SeamAccumulator postHeight = new();
        SeamAccumulator preNormal = new();
        SeamAccumulator postNormal = new();
        int neighborTileCount = 0;

        if (referenceHeightmaps.TryGetValue((tileCoordinate.TileX - 1, tileCoordinate.TileY), out float[]? leftNeighbor))
        {
            AccumulateVerticalEdge(originalHeightmap, patchedHeightmap, leftNeighbor, 0, AdtTerrainMath.TileHeightmapSize - 1, preHeight, postHeight, preNormal, postNormal);
            neighborTileCount++;
        }

        if (referenceHeightmaps.TryGetValue((tileCoordinate.TileX + 1, tileCoordinate.TileY), out float[]? rightNeighbor))
        {
            AccumulateVerticalEdge(originalHeightmap, patchedHeightmap, rightNeighbor, AdtTerrainMath.TileHeightmapSize - 1, 0, preHeight, postHeight, preNormal, postNormal);
            neighborTileCount++;
        }

        if (referenceHeightmaps.TryGetValue((tileCoordinate.TileX, tileCoordinate.TileY - 1), out float[]? topNeighbor))
        {
            AccumulateHorizontalEdge(originalHeightmap, patchedHeightmap, topNeighbor, 0, AdtTerrainMath.TileHeightmapSize - 1, preHeight, postHeight, preNormal, postNormal);
            neighborTileCount++;
        }

        if (referenceHeightmaps.TryGetValue((tileCoordinate.TileX, tileCoordinate.TileY + 1), out float[]? bottomNeighbor))
        {
            AccumulateHorizontalEdge(originalHeightmap, patchedHeightmap, bottomNeighbor, AdtTerrainMath.TileHeightmapSize - 1, 0, preHeight, postHeight, preNormal, postNormal);
            neighborTileCount++;
        }

        return new AdtSeamAudit(
            neighborTileCount,
            preHeight.ToMetric(),
            postHeight.ToMetric(),
            preNormal.ToMetric(),
            postNormal.ToMetric());
    }

    private static void AccumulateVerticalEdge(
        IReadOnlyList<float> originalHeightmap,
        IReadOnlyList<float> patchedHeightmap,
        IReadOnlyList<float> neighborHeightmap,
        int tileX,
        int neighborX,
        SeamAccumulator preHeight,
        SeamAccumulator postHeight,
        SeamAccumulator preNormal,
        SeamAccumulator postNormal)
    {
        for (int sampleY = 0; sampleY < AdtTerrainMath.TileHeightmapSize; sampleY++)
        {
            int tileIndex = (sampleY * AdtTerrainMath.TileHeightmapSize) + tileX;
            int neighborIndex = (sampleY * AdtTerrainMath.TileHeightmapSize) + neighborX;
            float neighborHeight = neighborHeightmap[neighborIndex];
            preHeight.Add(MathF.Abs(originalHeightmap[tileIndex] - neighborHeight));
            postHeight.Add(MathF.Abs(patchedHeightmap[tileIndex] - neighborHeight));

            Vector3 neighborNormal = AdtTerrainMath.ComputeNormal(neighborHeightmap, neighborX, sampleY);
            preNormal.Add(ComputeNormalAngleDegrees(AdtTerrainMath.ComputeNormal(originalHeightmap, tileX, sampleY), neighborNormal));
            postNormal.Add(ComputeNormalAngleDegrees(AdtTerrainMath.ComputeNormal(patchedHeightmap, tileX, sampleY), neighborNormal));
        }
    }

    private static void AccumulateHorizontalEdge(
        IReadOnlyList<float> originalHeightmap,
        IReadOnlyList<float> patchedHeightmap,
        IReadOnlyList<float> neighborHeightmap,
        int tileY,
        int neighborY,
        SeamAccumulator preHeight,
        SeamAccumulator postHeight,
        SeamAccumulator preNormal,
        SeamAccumulator postNormal)
    {
        for (int sampleX = 0; sampleX < AdtTerrainMath.TileHeightmapSize; sampleX++)
        {
            int tileIndex = (tileY * AdtTerrainMath.TileHeightmapSize) + sampleX;
            int neighborIndex = (neighborY * AdtTerrainMath.TileHeightmapSize) + sampleX;
            float neighborHeight = neighborHeightmap[neighborIndex];
            preHeight.Add(MathF.Abs(originalHeightmap[tileIndex] - neighborHeight));
            postHeight.Add(MathF.Abs(patchedHeightmap[tileIndex] - neighborHeight));

            Vector3 neighborNormal = AdtTerrainMath.ComputeNormal(neighborHeightmap, sampleX, neighborY);
            preNormal.Add(ComputeNormalAngleDegrees(AdtTerrainMath.ComputeNormal(originalHeightmap, sampleX, tileY), neighborNormal));
            postNormal.Add(ComputeNormalAngleDegrees(AdtTerrainMath.ComputeNormal(patchedHeightmap, sampleX, tileY), neighborNormal));
        }
    }

    private static float ComputeNormalAngleDegrees(Vector3 left, Vector3 right)
    {
        float dot = Math.Clamp(Vector3.Dot(left, right), -1f, 1f);
        return float.RadiansToDegrees(MathF.Acos(dot));
    }

    private static void ValidateHeightmap(IReadOnlyList<float> heightmap, string paramName)
    {
        if (heightmap.Count != AdtTerrainMath.TileHeightmapSize * AdtTerrainMath.TileHeightmapSize)
            throw new ArgumentException($"Heightmap must contain exactly {AdtTerrainMath.TileHeightmapSize * AdtTerrainMath.TileHeightmapSize} samples.", paramName);
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

    private static int LocateMcvtDataOffset(ReadOnlySpan<byte> payload)
    {
        int position = RootMcnkSubchunkOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int consumedSize = checked((int)header.Size);
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcvt)
            {
                if (header.Size < McvtSampleByteCount)
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

            int consumedSize = checked((int)header.Size);
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

    private sealed class SeamAccumulator
    {
        private float _sum;
        private float _max;
        private int _count;

        public void Add(float value)
        {
            _sum += value;
            _max = MathF.Max(_max, value);
            _count++;
        }

        public AdtSeamMetric ToMetric()
        {
            return new AdtSeamMetric(_count, _count == 0 ? 0f : _sum / _count, _max);
        }
    }
}

public sealed record AdtChunkChangeAudit(int PresentChunkCount, int ChangedMcvtChunkCount, int ChangedMcnrChunkCount);

public sealed record AdtSeamMetric(int SampleCount, float MeanAbsoluteDelta, float MaxAbsoluteDelta);

public sealed record AdtSeamAudit(
    int NeighborTileCount,
    AdtSeamMetric PreHeightDelta,
    AdtSeamMetric PostHeightDelta,
    AdtSeamMetric PreNormalAngleDegrees,
    AdtSeamMetric PostNormalAngleDegrees);