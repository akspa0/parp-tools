using System.Collections.Concurrent;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.App;

internal sealed class AlphaEmbeddedAdtTileData
{
    public AlphaEmbeddedAdtTileData(
        string sourcePath,
        AdtPlacementCatalog placementCatalog,
        WorldTerrainTileData terrainTileData,
        WorldLiquidTileData liquidTileData,
        WorldTileStageSummary tileStageSummary)
    {
        SourcePath = sourcePath;
        PlacementCatalog = placementCatalog;
        TerrainTileData = terrainTileData;
        LiquidTileData = liquidTileData;
        TileStageSummary = tileStageSummary;
    }

    public string SourcePath { get; }

    public AdtPlacementCatalog PlacementCatalog { get; }

    public WorldTerrainTileData TerrainTileData { get; }

    public WorldLiquidTileData LiquidTileData { get; }

    public WorldTileStageSummary TileStageSummary { get; }
}

internal sealed class AlphaEmbeddedWdtData
{
    public AlphaEmbeddedWdtData(
        byte[] wdtData,
        string sourcePath,
        byte[] mainData,
        IReadOnlyList<string> modelNames,
        IReadOnlyList<string> worldModelNames)
    {
        WdtData = wdtData;
        SourcePath = sourcePath;
        MainData = mainData;
        ModelNames = modelNames;
        WorldModelNames = worldModelNames;
    }

    public byte[] WdtData { get; }

    public string SourcePath { get; }

    public byte[] MainData { get; }

    public IReadOnlyList<string> ModelNames { get; }

    public IReadOnlyList<string> WorldModelNames { get; }
}

internal static class AlphaEmbeddedAdtReader
{
    private const int AlphaMainEntrySize = 16;
    private const int AlphaMcnkHeaderSize = 128;
    private const int AlphaChunkHeaderSize = 8;
    private const int AlphaMcinEntrySize = 16;
    private const int AlphaMcvtSize = 580;
    private const int AlphaMclqTileFlagsOffset = 0x290;
    private const uint AlphaLiquidFlagMask = 0x3Cu;
    private const int TileHeightmapSize = 257;
    private const int HalfStepsPerChunk = 16;
    private const int MapOrigin = 17066;

    private static readonly ConcurrentDictionary<string, AlphaEmbeddedAdtTileData> Cache = new(StringComparer.OrdinalIgnoreCase);
    private static readonly ConcurrentDictionary<string, AlphaEmbeddedWdtData> WdtCache = new(StringComparer.OrdinalIgnoreCase);

    public static bool TryReadPlacementCatalog(
        string clientRoot,
        string mapDirectory,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        out AdtPlacementCatalog? placementCatalog,
        out string sourcePath)
    {
        if (!TryReadWdtData(clientRoot, mapDirectory, archiveCatalog, out AlphaEmbeddedWdtData? wdtData)
            || !TryResolveAlphaTileOffset(wdtData.MainData, tileX, tileY, out int adtOffset))
        {
            placementCatalog = null;
            sourcePath = string.Empty;
            return false;
        }

        sourcePath = $"{wdtData.SourcePath}#alpha-tile({tileX},{tileY})";
        placementCatalog = BuildPlacementCatalog(wdtData, adtOffset, sourcePath);
        return true;
    }

    public static bool TryReadTile(
        string clientRoot,
        string mapDirectory,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        out AlphaEmbeddedAdtTileData? tileData)
    {
        string cacheKey = $"{clientRoot}|{mapDirectory}|{tileX}|{tileY}";
        if (Cache.TryGetValue(cacheKey, out AlphaEmbeddedAdtTileData? cached))
        {
            tileData = cached;
            return true;
        }

        if (!TryReadWdtData(clientRoot, mapDirectory, archiveCatalog, out AlphaEmbeddedWdtData? wdtData))
        {
            tileData = null;
            return false;
        }

        if (!TryResolveAlphaTileOffset(wdtData.MainData, tileX, tileY, out int adtOffset))
        {
            tileData = null;
            return false;
        }

        string sourcePath = $"{wdtData.SourcePath}#alpha-tile({tileX},{tileY})";
        IReadOnlyList<string> textureNames = ReadStringEntries(ReadEmbeddedSubchunkPayload(wdtData.WdtData, adtOffset, 0x04));
        List<WorldTerrainChunkData> terrainChunks = ReadAlphaTerrainChunks(wdtData.WdtData, adtOffset, textureNames);
        WorldTerrainTileData terrainTileData = new(sourcePath, MapFileKind.Adt, terrainChunks, BuildHeightmap(terrainChunks));
        WorldLiquidTileData liquidTileData = new(sourcePath, MapFileKind.Adt, ReadAlphaLiquidChunks(wdtData.WdtData, adtOffset));
        AdtPlacementCatalog placementCatalog = BuildPlacementCatalog(wdtData, adtOffset, sourcePath);
        WorldTileStageSummary tileStageSummary = new(
            sourcePath,
            MapFileKind.Adt,
            0,
            terrainTileData.ChunkCount,
            terrainTileData.HoleChunkCount,
            liquidTileData.ActiveChunkCount,
            liquidTileData.LayerCount,
            liquidTileData.VisibleTileCount,
            liquidTileData.ActiveChunkCount > 0);

        tileData = new AlphaEmbeddedAdtTileData(sourcePath, placementCatalog, terrainTileData, liquidTileData, tileStageSummary);
        Cache[cacheKey] = tileData;
        return true;
    }

    private static bool TryReadWdtData(
        string clientRoot,
        string mapDirectory,
        IArchiveCatalog archiveCatalog,
        out AlphaEmbeddedWdtData? wdtData)
    {
        string cacheKey = $"{clientRoot}|{mapDirectory}";
        if (WdtCache.TryGetValue(cacheKey, out AlphaEmbeddedWdtData? cached))
        {
            wdtData = cached;
            return true;
        }

        string wdtVirtualPath = $@"World\Maps\{mapDirectory}\{mapDirectory}.wdt";
        if (!TryReadVirtualOrLooseFile(clientRoot, wdtVirtualPath, archiveCatalog, out byte[]? rawWdtData, out string wdtSourcePath) || rawWdtData is null)
        {
            wdtData = null;
            return false;
        }

        using MemoryStream stream = new(rawWdtData, writable: false);
        MapFileSummary wdtSummary = MapFileSummaryReader.Read(stream, wdtVirtualPath);
        if (wdtSummary.Kind != MapFileKind.Wdt)
        {
            wdtData = null;
            return false;
        }

        byte[]? mainData = ReadChunkPayload(stream, wdtSummary, MapChunkIds.Main);
        if (mainData is not { Length: > 0 })
        {
            wdtData = null;
            return false;
        }

        wdtData = new AlphaEmbeddedWdtData(
            rawWdtData,
            wdtSourcePath,
            mainData,
            ReadStringEntries(ReadFirstAvailableChunkPayload(stream, wdtSummary, [MapChunkIds.Mdnm, MapChunkIds.Mmdx])),
            ReadStringEntries(ReadFirstAvailableChunkPayload(stream, wdtSummary, [MapChunkIds.Monm, MapChunkIds.Mwmo])));
        WdtCache[cacheKey] = wdtData;
        return true;
    }

    private static AdtPlacementCatalog BuildPlacementCatalog(AlphaEmbeddedWdtData wdtData, int adtOffset, string sourcePath)
    {
        byte[] mddfData = ReadEmbeddedSubchunkPayload(wdtData.WdtData, adtOffset, 0x0C) ?? Array.Empty<byte>();
        byte[] modfData = ReadEmbeddedSubchunkPayload(wdtData.WdtData, adtOffset, 0x14) ?? Array.Empty<byte>();

        return new AdtPlacementCatalog(
            sourcePath,
            MapFileKind.Adt,
            wdtData.ModelNames,
            wdtData.WorldModelNames,
            ReadAlphaModelPlacements(mddfData, wdtData.ModelNames),
            ReadAlphaWorldModelPlacements(modfData, wdtData.WorldModelNames));
    }

    public static bool TryReadVirtualOrLooseFile(
        string clientRoot,
        string virtualPath,
        IArchiveCatalog archiveCatalog,
        out byte[]? data,
        out string sourcePath)
    {
        string normalizedPath = NormalizeVirtualPath(virtualPath);
        foreach (string loosePath in EnumerateLooseCandidates(clientRoot, normalizedPath))
        {
            if (File.Exists(loosePath))
            {
                data = File.ReadAllBytes(loosePath);
                sourcePath = Path.GetFullPath(loosePath);
                return true;
            }

            byte[]? alphaData = AlphaArchiveReader.ReadWithMpqFallback(loosePath);
            if (alphaData is { Length: > 0 })
            {
                data = alphaData;
                sourcePath = ResolveCompanionMpqPath(loosePath);
                return true;
            }
        }

        data = archiveCatalog.ReadFile(normalizedPath) ?? archiveCatalog.ReadFile(normalizedPath.Replace('\\', '/'));
        sourcePath = normalizedPath;
        return data is { Length: > 0 };
    }

    private static IEnumerable<string> EnumerateLooseCandidates(string clientRoot, string normalizedPath)
    {
        yield return Path.Combine(clientRoot, normalizedPath.Replace('\\', Path.DirectorySeparatorChar));
        yield return Path.Combine(clientRoot, "Data", normalizedPath.Replace('\\', Path.DirectorySeparatorChar));
    }

    private static string ResolveCompanionMpqPath(string loosePath)
    {
        string upper = loosePath + ".MPQ";
        if (File.Exists(upper))
            return Path.GetFullPath(upper);

        string lower = loosePath + ".mpq";
        if (File.Exists(lower))
            return Path.GetFullPath(lower);

        return Path.GetFullPath(loosePath);
    }

    private static string NormalizeVirtualPath(string path)
    {
        return path.Trim().Replace('/', '\\');
    }

    private static bool TryResolveAlphaTileOffset(byte[]? mainData, int tileX, int tileY, out int offset)
    {
        offset = 0;
        if (mainData is not { Length: >= AlphaMainEntrySize })
            return false;

        int rowMajorIndex = (tileY * 64) + tileX;
        if (TryReadMainOffset(mainData, rowMajorIndex, out offset))
            return true;

        int columnMajorIndex = (tileX * 64) + tileY;
        return TryReadMainOffset(mainData, columnMajorIndex, out offset);
    }

    private static bool TryReadMainOffset(byte[] mainData, int index, out int offset)
    {
        offset = 0;
        int entryOffset = index * AlphaMainEntrySize;
        if (entryOffset < 0 || entryOffset + sizeof(int) > mainData.Length)
            return false;

        offset = BitConverter.ToInt32(mainData, entryOffset);
        return offset > 0;
    }

    private static byte[]? ReadEmbeddedSubchunkPayload(byte[] container, int adtOffset, int mhdrFieldOffset)
    {
        if (!TryReadChunkSize(container, adtOffset, out int mhdrSize))
            return null;

        int mhdrDataOffset = adtOffset + AlphaChunkHeaderSize;
        if (mhdrDataOffset + mhdrFieldOffset + sizeof(int) > container.Length || mhdrSize < mhdrFieldOffset + sizeof(int))
            return null;

        int relativeOffset = BitConverter.ToInt32(container, mhdrDataOffset + mhdrFieldOffset);
        if (relativeOffset <= 0)
            return null;

        int chunkOffset = mhdrDataOffset + relativeOffset;
        if (!TryReadChunkSize(container, chunkOffset, out int chunkSize))
            return null;

        int payloadOffset = chunkOffset + AlphaChunkHeaderSize;
        if (payloadOffset + chunkSize > container.Length)
            return null;

        byte[] payload = new byte[chunkSize];
        Buffer.BlockCopy(container, payloadOffset, payload, 0, chunkSize);
        return payload;
    }

    private static bool TryReadChunkSize(byte[] data, int offset, out int size)
    {
        size = 0;
        if (offset < 0 || offset + AlphaChunkHeaderSize > data.Length)
            return false;

        size = BitConverter.ToInt32(data, offset + 4);
        return size >= 0 && offset + AlphaChunkHeaderSize + size <= data.Length;
    }

    private static List<int> ReadMcnkOffsets(byte[] container, int adtOffset)
    {
        byte[]? mcinData = ReadEmbeddedSubchunkPayload(container, adtOffset, 0x00);
        if (mcinData is not { Length: >= AlphaMcinEntrySize })
            return [];

        List<int> offsets = new(256);
        for (int index = 0; index < 256 && (index * AlphaMcinEntrySize) + sizeof(int) <= mcinData.Length; index++)
            offsets.Add(BitConverter.ToInt32(mcinData, index * AlphaMcinEntrySize));

        return offsets;
    }

    private static List<WorldTerrainChunkData> ReadAlphaTerrainChunks(byte[] container, int adtOffset, IReadOnlyList<string> textureNames)
    {
        List<WorldTerrainChunkData> chunks = [];
        foreach (int mcnkOffset in ReadMcnkOffsets(container, adtOffset))
        {
            if (!TryReadAlphaTerrainChunk(container, mcnkOffset, textureNames, out WorldTerrainChunkData? chunkData))
                continue;

            chunks.Add(chunkData);
        }

        return chunks;
    }

    private static bool TryReadAlphaTerrainChunk(byte[] container, int mcnkOffset, IReadOnlyList<string> textureNames, out WorldTerrainChunkData? chunkData)
    {
        chunkData = null;
        if (!TryReadChunkSize(container, mcnkOffset, out _) || mcnkOffset + AlphaChunkHeaderSize + AlphaMcnkHeaderSize > container.Length)
            return false;

        int headerOffset = mcnkOffset + AlphaChunkHeaderSize;
        uint flags = BitConverter.ToUInt32(container, headerOffset + 0x00);
        int indexX = BitConverter.ToInt32(container, headerOffset + 0x04);
        int indexY = BitConverter.ToInt32(container, headerOffset + 0x08);
        int layerCount = BitConverter.ToInt32(container, headerOffset + 0x10);
        uint areaId = (uint)(BitConverter.ToInt32(container, headerOffset + 0x38) & 0xFFFF);
        ushort holeMask = (ushort)BitConverter.ToUInt32(container, headerOffset + 0x40);
        int chunkIndex = (indexY * 16) + indexX;
        float[]? heights = ReadAlphaInterleavedHeights(container, mcnkOffset, headerOffset);
        IReadOnlyList<AdtTextureChunkLayer> textureLayers = ReadAlphaTextureLayers(container, mcnkOffset, headerOffset, layerCount, textureNames);

        chunkData = new WorldTerrainChunkData(
            chunkIndex,
            indexX,
            indexY,
            areaId,
            flags,
            Math.Max(0, layerCount),
            holeMask,
            (flags & AlphaLiquidFlagMask) != 0,
            false,
            heights,
            textureLayers);
        return true;
    }

    private static IReadOnlyList<AdtTextureChunkLayer> ReadAlphaTextureLayers(
        byte[] container,
        int mcnkOffset,
        int headerOffset,
        int layerCount,
        IReadOnlyList<string> textureNames)
    {
        if (layerCount <= 0)
            return [];

        int layerLimit = Math.Min(layerCount, 4);
        int mclyRelativeOffset = BitConverter.ToInt32(container, headerOffset + 0x20);
        if (mclyRelativeOffset < 0)
            return [];

        int mclyDataOffset = mcnkOffset + AlphaChunkHeaderSize + AlphaMcnkHeaderSize + mclyRelativeOffset;
        int mclyByteCount = 0;
        if (mclyDataOffset + 8 <= container.Length)
        {
            int mclyPayloadSize = BitConverter.ToInt32(container, mclyDataOffset + 4);
            mclyByteCount = Math.Min(mclyPayloadSize, Math.Max(0, container.Length - mclyDataOffset - 8));
        }

        if (mclyByteCount < 16 * layerLimit)
            return [];

        int availableLayers = Math.Min(layerLimit, mclyByteCount / 16);

        byte[] mclyData = new byte[availableLayers * 16];
        Buffer.BlockCopy(container, mclyDataOffset + 8, mclyData, 0, mclyData.Length);

        int mcalRelativeOffset = BitConverter.ToInt32(container, headerOffset + 0x28);
        int mcalSize = BitConverter.ToInt32(container, headerOffset + 0x2C);
        byte[] mcalData = Array.Empty<byte>();
        if (mcalRelativeOffset >= 0 && mcalSize > 0)
        {
            int mcalDataOffset = mcnkOffset + AlphaChunkHeaderSize + AlphaMcnkHeaderSize + mcalRelativeOffset;
            int availableMcalBytes = Math.Min(mcalSize, container.Length - mcalDataOffset);
            if (mcalDataOffset >= 0 && availableMcalBytes > 0)
            {
                mcalData = new byte[availableMcalBytes];
                Buffer.BlockCopy(container, mcalDataOffset, mcalData, 0, availableMcalBytes);
            }
        }

        return BuildAlphaTextureLayers(mclyData, mcalData, availableLayers, textureNames);
    }

    private static IReadOnlyList<AdtTextureChunkLayer> BuildAlphaTextureLayers(
        byte[] mclyData,
        byte[] mcalData,
        int layerCount,
        IReadOnlyList<string> textureNames)
    {
        if (layerCount <= 0 || mclyData.Length < 16)
            return [];

        List<AdtTextureChunkLayer> layers = new(layerCount);
        int alphaOffset = 0;
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
        {
            int entryOffset = layerIndex * 16;
            if (entryOffset + 16 > mclyData.Length)
                break;

            uint textureId = BitConverter.ToUInt32(mclyData, entryOffset + 0x00);
            uint flags = BitConverter.ToUInt32(mclyData, entryOffset + 0x04);
            uint layerAlphaOffset = BitConverter.ToUInt32(mclyData, entryOffset + 0x08);
            uint effectId = BitConverter.ToUInt32(mclyData, entryOffset + 0x0C);

            string? texturePath = textureId < textureNames.Count ? textureNames[(int)textureId] : null;
            AdtMcalDecodedLayer? decodedAlpha = null;
            if (layerIndex > 0 && alphaOffset < mcalData.Length)
            {
                decodedAlpha = DecodeAlphaLayer(layerIndex, textureId, flags, layerAlphaOffset, mcalData, ref alphaOffset);
            }

            layers.Add(new AdtTextureChunkLayer(
                layerIndex,
                textureId,
                texturePath,
                flags,
                layerAlphaOffset,
                effectId,
                decodedAlpha));
        }

        return layers;
    }

    private static AdtMcalDecodedLayer? DecodeAlphaLayer(
        int layerIndex,
        uint textureId,
        uint flags,
        uint alphaOffset,
        byte[] mcalData,
        ref int sourceOffset)
    {
        int remaining = mcalData.Length - sourceOffset;
        if (remaining <= 0)
            return null;

        bool useBigAlpha = (flags & 0x200) != 0;
        int expectedBytes = useBigAlpha ? 4096 : 2048;
        int consumedBytes = Math.Min(expectedBytes, remaining);
        if (consumedBytes <= 0)
            return null;

        byte[] alphaMap;
        AdtMcalAlphaEncoding encoding;
        bool appliedFixup = false;
        if (!useBigAlpha)
        {
            int packedBytes = Math.Min(2048, consumedBytes);
            alphaMap = new byte[4096];
            for (int index = 0; index < packedBytes; index++)
            {
                byte packed = mcalData[sourceOffset + index];
                alphaMap[index * 2] = (byte)((packed & 0x0F) * 17);
                alphaMap[(index * 2) + 1] = (byte)((packed >> 4) * 17);
            }

            ApplyLegacyEdgeFix(alphaMap);
            encoding = AdtMcalAlphaEncoding.Packed4Bit;
            appliedFixup = true;
            consumedBytes = packedBytes;
        }
        else
        {
            alphaMap = new byte[4096];
            Buffer.BlockCopy(mcalData, sourceOffset, alphaMap, 0, Math.Min(4096, consumedBytes));
            encoding = AdtMcalAlphaEncoding.BigAlpha;
        }

        AdtMcalDecodedLayer decoded = new(
            layerIndex,
            textureId,
            flags,
            checked((int)alphaOffset),
            consumedBytes,
            encoding,
            appliedFixup,
            alphaMap);

        sourceOffset += consumedBytes;
        return decoded;
    }

    private static void ApplyLegacyEdgeFix(byte[] alphaMap)
    {
        const int alphaSize = 64;
        if (alphaMap.Length < alphaSize * alphaSize)
            return;

        for (int row = 0; row < alphaSize; row++)
            alphaMap[(row * alphaSize) + (alphaSize - 1)] = alphaMap[(row * alphaSize) + (alphaSize - 2)];

        Buffer.BlockCopy(alphaMap, (alphaSize - 2) * alphaSize, alphaMap, (alphaSize - 1) * alphaSize, alphaSize);
    }

    private static float[]? ReadAlphaInterleavedHeights(byte[] container, int mcnkOffset, int headerOffset)
    {
        int mcvtRelativeOffset = BitConverter.ToInt32(container, headerOffset + 0x18);
        if (mcvtRelativeOffset < 0)
            return null;

        int mcvtDataOffset = mcnkOffset + AlphaChunkHeaderSize + AlphaMcnkHeaderSize + mcvtRelativeOffset;
        if (mcvtDataOffset + AlphaMcvtSize > container.Length)
            return null;

        float[] heights = new float[145];
        int destination = 0;
        for (int row = 0; row < 17; row++)
        {
            if ((row & 1) == 0)
            {
                int outerRow = row / 2;
                for (int col = 0; col < 9; col++)
                {
                    int sourceIndex = (outerRow * 9) + col;
                    heights[destination++] = BitConverter.ToSingle(container, mcvtDataOffset + (sourceIndex * sizeof(float)));
                }
            }
            else
            {
                int innerRow = row / 2;
                for (int col = 0; col < 8; col++)
                {
                    int sourceIndex = 81 + (innerRow * 8) + col;
                    heights[destination++] = BitConverter.ToSingle(container, mcvtDataOffset + (sourceIndex * sizeof(float)));
                }
            }
        }

    // Ghidra-verified (CMapChunk::CreateVertices, 0.5.3.3368):
        // Alpha MCVT heights are ABSOLUTE world-space Z values — no base height addition.
        // The MCNK header field at offset 0x80 stores the chunk's world Position.Z,
        // which the client uses for bounding-box math and vertex relativization, NOT as
        // an additive base for the heights.

        return heights;
    }

    private static List<WorldLiquidChunkData> ReadAlphaLiquidChunks(byte[] container, int adtOffset)
    {
        List<WorldLiquidChunkData> chunks = [];
        foreach (int mcnkOffset in ReadMcnkOffsets(container, adtOffset))
        {
            if (!TryReadAlphaLiquidChunk(container, mcnkOffset, out WorldLiquidChunkData? chunkData))
                continue;

            chunks.Add(chunkData);
        }

        return chunks;
    }

    private static bool TryReadAlphaLiquidChunk(byte[] container, int mcnkOffset, out WorldLiquidChunkData? chunkData)
    {
        chunkData = null;
        if (!TryReadChunkSize(container, mcnkOffset, out _) || mcnkOffset + AlphaChunkHeaderSize + AlphaMcnkHeaderSize > container.Length)
            return false;

        int headerOffset = mcnkOffset + AlphaChunkHeaderSize;
        uint flags = BitConverter.ToUInt32(container, headerOffset + 0x00);
        int indexX = BitConverter.ToInt32(container, headerOffset + 0x04);
        int indexY = BitConverter.ToInt32(container, headerOffset + 0x08);
        int mclqRelativeOffset = BitConverter.ToInt32(container, headerOffset + 0x64);
        int mcnkChunksSize = BitConverter.ToInt32(container, headerOffset + 0x5C);
        if ((flags & AlphaLiquidFlagMask) == 0 || mclqRelativeOffset <= 0 || mcnkChunksSize <= mclqRelativeOffset)
            return false;

        int payloadOffset = mcnkOffset + AlphaChunkHeaderSize + AlphaMcnkHeaderSize + mclqRelativeOffset;
        int payloadSize = mcnkChunksSize - mclqRelativeOffset;
        if (payloadOffset < 0 || payloadSize < 8 || payloadOffset + payloadSize > container.Length)
            return false;

        byte[] payload = new byte[payloadSize];
        Buffer.BlockCopy(container, payloadOffset, payload, 0, payloadSize);
        payload = StripMclqChunkHeaderIfPresent(payload);
        if (payload.Length < 8)
            return false;

        float minHeight = BitConverter.ToSingle(payload, 0);
        float maxHeight = BitConverter.ToSingle(payload, 4);
        if (float.IsNaN(minHeight) || float.IsNaN(maxHeight))
            return false;

        // Alpha MCLQ heights are absolute — no base height addition needed.

        byte[]? tileFlags = null;
        if (payload.Length >= AlphaMclqTileFlagsOffset + 64)
        {
            tileFlags = new byte[64];
            Buffer.BlockCopy(payload, AlphaMclqTileFlagsOffset, tileFlags, 0, 64);
        }

        AdtLiquidBasicType basicType = ResolveAlphaLiquidBasicType(flags);
        WorldLiquidLayerData layer = new(
            (ushort)basicType,
            basicType,
            AdtLiquidVertexFormat.HeightDepth,
            minHeight,
            maxHeight,
            0,
            0,
            8,
            8,
            CountVisibleLiquidTiles(tileFlags),
            hasDepthData: false,
            hasHeightData: payload.Length >= (8 + (81 * 8)),
            hasUvData: false);

        chunkData = new WorldLiquidChunkData((indexY * 16) + indexX, indexX, indexY, null, null, [layer]);
        return true;
    }

    private static AdtLiquidBasicType ResolveAlphaLiquidBasicType(uint flags)
    {
        if ((flags & 0x08) != 0)
            return AdtLiquidBasicType.Ocean;

        return ((flags >> 4) & 0x3) switch
        {
            1 => AdtLiquidBasicType.Ocean,
            2 => AdtLiquidBasicType.Magma,
            3 => AdtLiquidBasicType.Slime,
            _ => AdtLiquidBasicType.Water,
        };
    }

    private static int CountVisibleLiquidTiles(byte[]? tileFlags)
    {
        if (tileFlags is not { Length: >= 64 })
            return 64;

        int visible = 0;
        for (int index = 0; index < 64; index++)
        {
            if ((tileFlags[index] & 0x0F) != 0x0F)
                visible++;
        }

        return visible;
    }

    private static byte[] StripMclqChunkHeaderIfPresent(byte[] payload)
    {
        if (payload.Length < 8)
            return payload;

        bool isMclq = payload[0] == (byte)'M' && payload[1] == (byte)'C' && payload[2] == (byte)'L' && payload[3] == (byte)'Q';
        bool isReversed = payload[0] == (byte)'Q' && payload[1] == (byte)'L' && payload[2] == (byte)'C' && payload[3] == (byte)'M';
        if (!isMclq && !isReversed)
            return payload;

        uint size = BitConverter.ToUInt32(payload, 4);
        if (size == 0 || payload.Length < 8 + size)
            return payload;

        byte[] stripped = new byte[size];
        Buffer.BlockCopy(payload, 8, stripped, 0, (int)size);
        return stripped;
    }

    private static List<AdtModelPlacement> ReadAlphaModelPlacements(byte[] payload, IReadOnlyList<string> modelNames)
    {
        const int entrySize = 36;
        List<AdtModelPlacement> placements = [];
        for (int offset = 0; offset + entrySize <= payload.Length; offset += entrySize)
        {
            int nameId = BitConverter.ToInt32(payload, offset + 0);
            int uniqueId = BitConverter.ToInt32(payload, offset + 4);
            float filePosX = BitConverter.ToSingle(payload, offset + 0x08);
            float filePosY = BitConverter.ToSingle(payload, offset + 0x0C);
            float filePosZ = BitConverter.ToSingle(payload, offset + 0x10);
            float fileRotX = BitConverter.ToSingle(payload, offset + 0x14);
            float fileRotY = BitConverter.ToSingle(payload, offset + 0x18);
            float fileRotZ = BitConverter.ToSingle(payload, offset + 0x1C);
            ushort scale = BitConverter.ToUInt16(payload, offset + 0x20);

            float rendererX = MapOrigin - filePosZ;
            float rendererY = MapOrigin - filePosX;
            float rendererZ = filePosY;
            float rendererRollDeg = fileRotZ;
            float rendererPitchDeg = fileRotX;
            float rendererYawDeg = fileRotY + 180.0f;
            placements.Add(new AdtModelPlacement(
                nameId,
                ResolveIndexedName(modelNames, nameId),
                uniqueId,
                new Vector3(rendererX, rendererY, rendererZ),
                new Vector3(rendererRollDeg, rendererPitchDeg, rendererYawDeg),
                scale / 1024f));
        }

        return placements;
    }

    private static List<AdtWorldModelPlacement> ReadAlphaWorldModelPlacements(byte[] payload, IReadOnlyList<string> worldModelNames)
    {
        const int entrySize = 64;
        List<AdtWorldModelPlacement> placements = [];
        for (int offset = 0; offset + entrySize <= payload.Length; offset += entrySize)
        {
            int nameId = BitConverter.ToInt32(payload, offset + 0);
            int uniqueId = BitConverter.ToInt32(payload, offset + 4);
            float filePosX = BitConverter.ToSingle(payload, offset + 0x08);
            float filePosY = BitConverter.ToSingle(payload, offset + 0x0C);
            float filePosZ = BitConverter.ToSingle(payload, offset + 0x10);
            float fileRotX = BitConverter.ToSingle(payload, offset + 0x14);
            float fileRotY = BitConverter.ToSingle(payload, offset + 0x18);
            float fileRotZ = BitConverter.ToSingle(payload, offset + 0x1C);
            float extentsTopX = BitConverter.ToSingle(payload, offset + 0x20);
            float extentsTopY = BitConverter.ToSingle(payload, offset + 0x24);
            float extentsTopZ = BitConverter.ToSingle(payload, offset + 0x28);
            float extentsBotX = BitConverter.ToSingle(payload, offset + 0x2C);
            float extentsBotY = BitConverter.ToSingle(payload, offset + 0x30);
            float extentsBotZ = BitConverter.ToSingle(payload, offset + 0x34);
            ushort flags = BitConverter.ToUInt16(payload, offset + 0x38);

            float rendererX = MapOrigin - filePosZ;
            float rendererY = MapOrigin - filePosX;
            float rendererZ = filePosY;
            float rendererRollDeg = fileRotZ;
            float rendererPitchDeg = fileRotX;
            float rendererYawDeg = fileRotY + 180.0f;
            float boundsMinX = MapOrigin - extentsTopZ;
            float boundsMinY = MapOrigin - extentsTopX;
            float boundsMinZ = extentsBotY;
            float boundsMaxX = MapOrigin - extentsBotZ;
            float boundsMaxY = MapOrigin - extentsBotX;
            float boundsMaxZ = extentsTopY;
            placements.Add(new AdtWorldModelPlacement(
                nameId,
                ResolveIndexedName(worldModelNames, nameId),
                uniqueId,
                new Vector3(rendererX, rendererY, rendererZ),
                new Vector3(rendererRollDeg, rendererPitchDeg, rendererYawDeg),
                new Vector3(boundsMinX, boundsMinY, boundsMinZ),
                new Vector3(boundsMaxX, boundsMaxY, boundsMaxZ),
                flags));
        }

        return placements;
    }

    private static string ResolveIndexedName(IReadOnlyList<string> names, int index)
    {
        return index >= 0 && index < names.Count ? NormalizeVirtualPath(names[index]) : $"unknown_{index}";
    }

    private static byte[]? ReadChunkPayload(Stream stream, MapFileSummary fileSummary, FourCC id)
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

    private static byte[]? ReadFirstAvailableChunkPayload(Stream stream, MapFileSummary fileSummary, IReadOnlyList<FourCC> ids)
    {
        foreach (FourCC id in ids)
        {
            byte[]? payload = ReadChunkPayload(stream, fileSummary, id);
            if (payload is { Length: > 0 })
                return payload;
        }

        return null;
    }

    private static IReadOnlyList<string> ReadStringEntries(byte[]? payload)
    {
        if (payload is not { Length: > 0 })
            return Array.Empty<string>();

        List<string> entries = [];
        int start = 0;
        for (int index = 0; index < payload.Length; index++)
        {
            if (payload[index] != 0)
                continue;

            if (index > start)
                entries.Add(Encoding.UTF8.GetString(payload, start, index - start));

            start = index + 1;
        }

        if (start < payload.Length)
            entries.Add(Encoding.UTF8.GetString(payload, start, payload.Length - start));

        return entries;
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
