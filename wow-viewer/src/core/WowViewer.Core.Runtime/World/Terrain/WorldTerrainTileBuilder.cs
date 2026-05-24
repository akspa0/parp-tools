using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Runtime.World.Terrain;

public static class WorldTerrainTileBuilder
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const int McnrConsumedSize = 0x1C0;
    private const int McinEntrySize = 16;
    private const int ExpectedChunkCount = 256;
    private const int TileHeightmapSize = 257;
    private const int HalfStepsPerChunk = 16;
    private const int McvtSampleCount = 145;
    private const uint LiquidFlagMask = 0x3Cu;
    private const uint VertexColorFlagMask = 0x40u;

    public static WorldTerrainTileData Read(string path, bool applyBaseHeightOffset = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));

        AdtTextureFile? textureFile = null;
        AdtTileFamily family = AdtTileFamilyResolver.Resolve(path);
        if (family.HasTex0 && fileSummary.Kind == MapFileKind.Adt)
            textureFile = AdtTextureReader.Read(family.Tex0Path);

        return Read(stream, fileSummary, textureFile, applyBaseHeightOffset);
    }

    public static WorldTerrainTileData Read(Stream stream, MapFileSummary fileSummary, bool applyBaseHeightOffset = true)
    {
        return Read(stream, fileSummary, textureFile: null, applyBaseHeightOffset);
    }

    public static WorldTerrainTileData Read(Stream stream, MapFileSummary fileSummary, AdtTextureFile? textureFile, bool applyBaseHeightOffset = true)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"World terrain tile builder requires a root ADT file, but found {fileSummary.Kind}.");

        IReadOnlyList<MapChunkLocation> terrainChunkLocations = ResolveTerrainChunkLocations(stream, fileSummary);
    IReadOnlyList<string> inlineTextureNames = ReadTextureNames(stream, fileSummary);
    Dictionary<int, AdtTextureChunk>? externalTextureChunks = textureFile?.Chunks.ToDictionary(static chunk => chunk.ChunkIndex);
        List<WorldTerrainChunkData> chunks = new(terrainChunkLocations.Count);
        int chunkOrdinal = 0;
        foreach (MapChunkLocation chunk in terrainChunkLocations)
        {
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
            float baseHeight = BinaryPrimitives.ReadSingleLittleEndian(payload.AsSpan(0x70, 4));
            float[]? heights = TryReadMcvtHeights(payload, baseHeight, applyBaseHeightOffset);
            AdtTextureChunk? textureChunk = ResolveTextureChunk(chunkOrdinal, payload, fileSummary.Kind, inlineTextureNames, externalTextureChunks);

            chunks.Add(new WorldTerrainChunkData(
                chunkOrdinal,
                indexX,
                indexY,
                areaId,
                flags,
                layerCount,
                holes,
                (flags & LiquidFlagMask) != 0,
                (flags & VertexColorFlagMask) != 0,
                heights,
                textureChunk?.Layers));
            chunkOrdinal++;
        }

        return new WorldTerrainTileData(fileSummary.SourcePath, fileSummary.Kind, chunks, BuildHeightmap(chunks));
    }

    private static IReadOnlyList<string> ReadTextureNames(Stream stream, MapFileSummary fileSummary)
    {
        if (!fileSummary.HasChunk(MapChunkIds.Mtex))
            return [];

        MapChunkLocation mtexChunk = fileSummary.Chunks.First(chunk => chunk.Id == MapChunkIds.Mtex);
        byte[] payload = ReadChunkPayload(stream, mtexChunk);
        return ParseStringEntries(payload);
    }

    private static AdtTextureChunk? ResolveTextureChunk(
        int chunkIndex,
        byte[] payload,
        MapFileKind terrainKind,
        IReadOnlyList<string> inlineTextureNames,
        Dictionary<int, AdtTextureChunk>? externalTextureChunks)
    {
        if (externalTextureChunks is not null && externalTextureChunks.TryGetValue(chunkIndex, out AdtTextureChunk? externalChunk))
            return externalChunk;

        if (inlineTextureNames.Count == 0)
            return null;

        AdtTextureChunk inlineChunk = AdtTextureChunkReader.Read(chunkIndex, payload, terrainKind, inlineTextureNames);
        return inlineChunk.Layers.Count > 0 ? inlineChunk : null;
    }

    private static IReadOnlyList<string> ParseStringEntries(byte[] payload)
    {
        if (payload.Length == 0)
            return [];

        List<string> entries = [];
        int start = 0;
        for (int index = 0; index <= payload.Length; index++)
        {
            if (index < payload.Length && payload[index] != 0)
                continue;

            int length = index - start;
            if (length > 0)
                entries.Add(Encoding.UTF8.GetString(payload, start, length));

            start = index + 1;
        }

        return entries;
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

    private static float[]? TryReadMcvtHeights(byte[] payload, float baseHeight, bool applyBaseHeightOffset)
    {
        int position = RootMcnkSubchunkOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.AsSpan(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            int consumedSize = checked((int)header.Size);
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcvt)
            {
                if (header.Size < McvtSampleCount * sizeof(float))
                    return null;

                int dataOffset = position + ChunkHeader.SizeInBytes;
                float[] heights = new float[McvtSampleCount];
                for (int index = 0; index < heights.Length; index++)
                {
                    float rawHeight = BinaryPrimitives.ReadSingleLittleEndian(payload.AsSpan(dataOffset + (index * sizeof(float)), sizeof(float)));
                    heights[index] = applyBaseHeightOffset ? baseHeight + rawHeight : rawHeight;
                }

                return heights;
            }

            position = checked((int)nextOffset);
        }

        return null;
    }

    private static WorldTerrainHeightmapData? BuildHeightmap(IReadOnlyList<WorldTerrainChunkData> chunks)
    {
        float[] sum = new float[TileHeightmapSize * TileHeightmapSize];
        ushort[] count = new ushort[TileHeightmapSize * TileHeightmapSize];

        foreach (WorldTerrainChunkData chunk in chunks)
        {
            if (!chunk.HasHeights || chunk.Heights is null)
                continue;

            int baseX = chunk.IndexX * HalfStepsPerChunk;
            int baseY = chunk.IndexY * HalfStepsPerChunk;
            for (int index = 0; index < chunk.Heights.Length; index++)
            {
                GetVertexPosition(index, out int row, out int col, out bool isInner);
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;

                int x = baseX + sampleX;
                int y = baseY + sampleY;
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;

                int target = (y * TileHeightmapSize) + x;
                sum[target] += chunk.Heights[index];
                count[target]++;
            }
        }

        int authoritativeSampleCount = count.Count(static value => value > 0);
        if (authoritativeSampleCount == 0)
            return null;

        float[] heights = new float[TileHeightmapSize * TileHeightmapSize];
        float min = float.MaxValue;
        float max = float.MinValue;
        for (int index = 0; index < heights.Length; index++)
        {
            if (count[index] > 0)
            {
                float value = sum[index] / count[index];
                heights[index] = value;
                if (value < min)
                    min = value;

                if (value > max)
                    max = value;
            }
            else
            {
                heights[index] = float.NaN;
            }
        }

        FillMixedParityGaps(heights);
        FillRemainingGaps(heights);
        if (min == float.MaxValue || max == float.MinValue)
        {
            min = 0f;
            max = 0f;
        }

        return new WorldTerrainHeightmapData(TileHeightmapSize, TileHeightmapSize, heights, min, max, authoritativeSampleCount);
    }

    private static void FillMixedParityGaps(float[] heights)
    {
        for (int y = 0; y < TileHeightmapSize; y++)
        {
            for (int x = 0; x < TileHeightmapSize; x++)
            {
                int index = (y * TileHeightmapSize) + x;
                if (!float.IsNaN(heights[index]))
                    continue;

                if ((x & 1) == 1 && (y & 1) == 0)
                {
                    float left = heights[(y * TileHeightmapSize) + (x - 1)];
                    float right = heights[(y * TileHeightmapSize) + (x + 1)];
                    if (!float.IsNaN(left) && !float.IsNaN(right))
                        heights[index] = (left + right) * 0.5f;
                }
                else if ((x & 1) == 0 && (y & 1) == 1)
                {
                    float up = heights[((y - 1) * TileHeightmapSize) + x];
                    float down = heights[((y + 1) * TileHeightmapSize) + x];
                    if (!float.IsNaN(up) && !float.IsNaN(down))
                        heights[index] = (up + down) * 0.5f;
                }
            }
        }
    }

    private static void FillRemainingGaps(float[] heights)
    {
        for (int y = 0; y < TileHeightmapSize; y++)
        {
            for (int x = 0; x < TileHeightmapSize; x++)
            {
                int index = (y * TileHeightmapSize) + x;
                if (!float.IsNaN(heights[index]))
                    continue;

                if (TryFindNearestHeight(heights, x, y, out float nearest))
                    heights[index] = nearest;
                else
                    heights[index] = 0f;
            }
        }
    }

    private static bool TryFindNearestHeight(float[] heights, int x, int y, out float value)
    {
        value = 0f;
        const int maxRadius = 24;
        for (int radius = 1; radius <= maxRadius; radius++)
        {
            int minY = Math.Max(0, y - radius);
            int maxY = Math.Min(TileHeightmapSize - 1, y + radius);
            int minX = Math.Max(0, x - radius);
            int maxX = Math.Min(TileHeightmapSize - 1, x + radius);

            for (int sampleX = minX; sampleX <= maxX; sampleX++)
            {
                float top = heights[(minY * TileHeightmapSize) + sampleX];
                if (!float.IsNaN(top))
                {
                    value = top;
                    return true;
                }

                float bottom = heights[(maxY * TileHeightmapSize) + sampleX];
                if (!float.IsNaN(bottom))
                {
                    value = bottom;
                    return true;
                }
            }

            for (int sampleY = minY + 1; sampleY < maxY; sampleY++)
            {
                float left = heights[(sampleY * TileHeightmapSize) + minX];
                if (!float.IsNaN(left))
                {
                    value = left;
                    return true;
                }

                float right = heights[(sampleY * TileHeightmapSize) + maxX];
                if (!float.IsNaN(right))
                {
                    value = right;
                    return true;
                }
            }
        }

        return false;
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