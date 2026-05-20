using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Builds a <see cref="TerrainTileTensorPack"/> from an ADT root file and its associated
/// texture source (_tex0.adt or inline). Coordinates existing deep readers in WowViewer.Core.IO.
/// </summary>
public static class AdtTensorPackBuilder
{
    private const int RootMcnkHeaderSize = 128;
    private const int RootMcnkSubchunkOffset = 0x80;
    private const int McvtSampleCount = 145;
    private const int McnrConsumedSize = 0x1C0;
    private const int McnrSampleByteCount = McvtSampleCount * 3;
    private const int HalfStepsPerChunk = 16;
    private const int TileHeightmapSize = 257;
    private const int TileChunks = 16;
    private const int ChunkAlphaSize = 64;
    private const int TileAlphaSize = ChunkAlphaSize * TileChunks;
    private const int TileMinimapSize = 256;

    public static TerrainTileTensorPack Build(string adtPath, string? textureSourcePath = null, string? buildVersion = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(adtPath);

        AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
        AdtFormatProfile profile = AdtFormatProfiles.Resolve(buildVersion);

        using FileStream stream = File.OpenRead(adtPath);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(adtPath));
        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"Tensor pack builder requires a root ADT file, but found {fileSummary.Kind}.");

        string tileName = Path.GetFileNameWithoutExtension(adtPath);
        TryParseAdtTileCoords(adtPath, out int tileX, out int tileY);
        HashSet<string> availableSignals = [];

        // ── Resolve terrain chunks ───────────────────────────────────────────
        List<MapChunkLocation> terrainChunks = ResolveTerrainChunkLocations(stream, fileSummary);

        // ── Assemble heightmap (MCVT) ───────────────────────────────────────
        float[,]? height257 = AssembleHeightmap(stream, terrainChunks, availableSignals);

        // ── Assemble normals (MCNR) ─────────────────────────────────────────
        float[,,]? mcnrNormalXyz = AssembleNormals(stream, terrainChunks, availableSignals);

        // ── Assemble vertex colors (MCCV) ────────────────────────────────────
        float[,,]? mccvRgb = AssembleMccv(stream, terrainChunks, availableSignals);

        // ── Assemble baked lighting bytes (MCLV) ─────────────────────────────
        byte[,,]? mclvLightingBytes = AssembleMclv(stream, terrainChunks, availableSignals);

        // ── Read flight bounds (MFBO) ────────────────────────────────────────
        int[,,]? mfboFlightBounds = ReadMfbo(stream, fileSummary, availableSignals);

        // ── Read texture data (MCLY + MCAL) ──────────────────────────────────
        (int[,,]? mclyTextureIds, IReadOnlyList<string> mclyTextureNames, bool[,,]? mclyLayerMask, byte[,,]? mcmtMaterialIds, byte[]? mampValue, float[,,]? mcalAlphaPack, float[,]? mcshShadowMask256) =
            ReadTextureData(adtPath, textureSourcePath, profile, availableSignals);

        // ── Read MH2O liquid ─────────────────────────────────────────────────
        (float[,]? mh2oHeight, float[,]? mh2oDepth, int[,]? mh2oType, bool[,]? mh2oPresence) =
            ReadMh2o(stream, fileSummary, profile, availableSignals);

        // ── Read MTXF texture flags ──────────────────────────────────────────
        (int[,]? mtxfAnimated, int[,]? mtxfTransform) =
            ReadMtxf(stream, fileSummary, mclyTextureIds, availableSignals);

        // ── Read MCLQ legacy liquid ──────────────────────────────────────────
        (float[,]? mclqHeight, int[,]? mclqType, bool[,]? mclqPresence) =
            ReadMclq(stream, terrainChunks, profile, availableSignals);

        // ── Read MCRF object references ──────────────────────────────────────
        (bool[,]? holeMask, int[,]? objectMask16, int[,]? mcrfDoodadRefCounts16, int[]? mcrfDoodadRefIndices, int[,]? mcrfWmoRefCounts16, int[]? mcrfWmoRefIndices) =
            ReadMcrfAndHoles(stream, terrainChunks, availableSignals);

        // ── Read MCNK header flags (liquid type bits 2-5) ───────────────────
        int[,]? mcnkFlags16 = ReadMcnkFlags(stream, terrainChunks, availableSignals);

        // ── Read MCSE sound emitters ────────────────────────────────────────
        (int[,]? mcseEmitterCounts16, int[]? mcseEntryIds, float[,]? mcsePositionXyz, byte[,]? mcseEntryBytes) =
            ReadMcse(stream, terrainChunks, availableSignals);

        // ── Read split Cataclysm+ chunk object references (MCRD/MCRW) ──────
        (int[,]? mcrdRefCounts16, int[]? mcrdRefIndices, int[,]? mcrwRefCounts16, int[]? mcrwRefIndices) =
            ReadSplitPlacementChunkReferences(adtPath, availableSignals);

        // ── Read WL* loose liquid files ──────────────────────────────────────
        (float[,]? wlMask, float[,]? wlHeight) =
            ReadWlFiles(adtPath, availableSignals);

        // ── Read placements once for masks + placement arrays ───────────────
        AdtPlacementCatalog? placementCatalog =
            TryReadPlacementCatalog(adtPath, stream, fileSummary);

        // ── Build object footprint masks from MDDF/MODF ──────────────────────
        (float[,]? objectMask257, float[,]? objectPreciseMask257, int[,]? objectInstanceMask257, float[,]? mddfMask257, float[,]? modfMask257, float[,]? objectFilteredMask257) =
            BuildObjectMasks(adtPath, stream, fileSummary, availableSignals, placementsOverride: placementCatalog);

        float[,]? shadowResidualMask256 = BuildShadowResidualMask256(mcshShadowMask256, objectPreciseMask257, availableSignals);

        // ── Build PM4 path, building footprint, and MPRL portal masks ────────
        (float[,]? pm4PathMask, float[,]? pm4BuildingFootprintMask, float[,]? pm4MprlMask) =
            BuildPm4Masks(adtPath, availableSignals);

        // ── Build unified liquid mask and height ─────────────────────────────
        (float[,]? unifiedLiquidMask, float[,]? unifiedLiquidHeight) =
            BuildUnifiedLiquid(mh2oHeight, mh2oPresence, mclqHeight, mclqPresence, wlMask, wlHeight, availableSignals);

        // ── Compute downsampled heights ──────────────────────────────────────
        float[,]? height65 = DownsampleHeightmap(height257, 65);
        float[,]? height17 = DownsampleHeightmap(height257, 17);

        (int mddfCount, int modfCount, float[,]? mddfData, float[,]? modfData, IReadOnlyList<string> mddfNames, IReadOnlyList<string> modfNames) =
            ExtractPlacementArrays(adtPath, stream, fileSummary, placementsOverride: placementCatalog);

        IReadOnlyList<TerrainRawChunkBlob> rawChunks = AdtRawChunkBlobCollector.Collect(family.RootPath, textureSourcePath);
        if (rawChunks.Count > 0)
            availableSignals.Add("raw_adt_chunks");

        return new TerrainTileTensorPack
        {
            TileName = tileName,
            MapName = ExtractMapName(adtPath),
            BuildKey = buildVersion ?? string.Empty,
            SourceAdtPath = adtPath,
            TileX = tileX,
            TileY = tileY,
            Height257 = height257,
            Height65 = height65,
            Height17 = height17,
            MclyTextureIds = mclyTextureIds,
            MclyTextureNames = mclyTextureNames,
            MclyLayerMask = mclyLayerMask,
            McmtMaterialIds = mcmtMaterialIds,
            MampValue = mampValue,
            McalAlphaPack = mcalAlphaPack,
            McalAlphaPack256 = DownsampleAlpha256(mcalAlphaPack),
            MccvRgb = mccvRgb,
            MclvLightingBytes = mclvLightingBytes,
            McnrNormalXyz = mcnrNormalXyz,
            MfboFlightBounds = mfboFlightBounds,
            McnkFlags16 = mcnkFlags16,
            Mh2oSurfaceHeight = mh2oHeight,
            Mh2oDepth = mh2oDepth,
            Mh2oTypeMask = mh2oType,
            Mh2oPresenceMask = mh2oPresence,
            MclqSurfaceHeight = mclqHeight,
            MclqTypeMask = mclqType,
            MclqPresenceMask = mclqPresence,
            MtxfAnimatedMask = mtxfAnimated,
            MtxfTransformId = mtxfTransform,
            HoleMask16 = holeMask,
            McseEmitterCounts16 = mcseEmitterCounts16,
            McseEntryIds = mcseEntryIds,
            McsePositionXyz = mcsePositionXyz,
            McseEntryBytes = mcseEntryBytes,
            McrfDoodadRefCounts16 = mcrfDoodadRefCounts16,
            McrfDoodadRefIndices = mcrfDoodadRefIndices,
            McrfWmoRefCounts16 = mcrfWmoRefCounts16,
            McrfWmoRefIndices = mcrfWmoRefIndices,
            McrdRefCounts16 = mcrdRefCounts16,
            McrdRefIndices = mcrdRefIndices,
            McrwRefCounts16 = mcrwRefCounts16,
            McrwRefIndices = mcrwRefIndices,
            WlLiquidMask = wlMask,
            WlLiquidHeight = wlHeight,
            UnifiedLiquidMask = unifiedLiquidMask,
            UnifiedLiquidHeight = unifiedLiquidHeight,
            ObjectMask257 = objectMask257,
            ObjectPreciseMask257 = objectPreciseMask257,
            ObjectInstanceMask257 = objectInstanceMask257,
            MddfMask257 = mddfMask257,
            ModfMask257 = modfMask257,
            ObjectFilteredMask257 = objectFilteredMask257,
            Pm4PathMask = pm4PathMask,
            Pm4BuildingFootprintMask = pm4BuildingFootprintMask,
            Pm4MprlMask = pm4MprlMask,
            McshShadowMask256 = mcshShadowMask256,
            ShadowResidualMask256 = shadowResidualMask256,
            PlacementMddfCount = mddfCount,
            PlacementModfCount = modfCount,
            PlacementMddfData = mddfData,
            PlacementModfData = modfData,
            PlacementMddfNames = mddfNames,
            PlacementModfNames = modfNames,
            RawChunks = rawChunks,
            AvailableSignals = availableSignals,
        };
    }

    public static TerrainTileTensorPack BuildFromBytes(
        string sourceAdtPath,
        byte[] adtBytes,
        byte[]? textureSourceBytes = null,
        byte[]? placementSourceBytes = null,
        string? buildVersion = null,
        string? textureSourcePath = null,
        string? placementSourcePath = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceAdtPath);
        ArgumentNullException.ThrowIfNull(adtBytes);

        AdtFormatProfile profile = AdtFormatProfiles.Resolve(buildVersion);

        using MemoryStream stream = new(adtBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourceAdtPath);
        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"Tensor pack builder requires a root ADT file, but found {fileSummary.Kind}.");

        string tileName = Path.GetFileNameWithoutExtension(sourceAdtPath);
        TryParseAdtTileCoords(sourceAdtPath, out int tileX, out int tileY);
        HashSet<string> availableSignals = [];

        List<MapChunkLocation> terrainChunks = ResolveTerrainChunkLocations(stream, fileSummary);
        float[,]? height257 = AssembleHeightmap(stream, terrainChunks, availableSignals);
        float[,,]? mcnrNormalXyz = AssembleNormals(stream, terrainChunks, availableSignals);
        float[,,]? mccvRgb = AssembleMccv(stream, terrainChunks, availableSignals);
        byte[,,]? mclvLightingBytes = AssembleMclv(stream, terrainChunks, availableSignals);
        int[,,]? mfboFlightBounds = ReadMfbo(stream, fileSummary, availableSignals);

        (int[,,]? mclyTextureIds, IReadOnlyList<string> mclyTextureNames, bool[,,]? mclyLayerMask, byte[,,]? mcmtMaterialIds, byte[]? mampValue, float[,,]? mcalAlphaPack, float[,]? mcshShadowMask256) =
            ReadTextureDataFromBytes(sourceAdtPath, adtBytes, textureSourcePath, textureSourceBytes, profile, availableSignals);

        (float[,]? mh2oHeight, float[,]? mh2oDepth, int[,]? mh2oType, bool[,]? mh2oPresence) =
            ReadMh2o(stream, fileSummary, profile, availableSignals);

        (int[,]? mtxfAnimated, int[,]? mtxfTransform) =
            ReadMtxf(stream, fileSummary, mclyTextureIds, availableSignals);

        (float[,]? mclqHeight, int[,]? mclqType, bool[,]? mclqPresence) =
            ReadMclq(stream, terrainChunks, profile, availableSignals);

        (bool[,]? holeMask, int[,]? objectMask16, int[,]? mcrfDoodadRefCounts16, int[]? mcrfDoodadRefIndices, int[,]? mcrfWmoRefCounts16, int[]? mcrfWmoRefIndices) =
            ReadMcrfAndHoles(stream, terrainChunks, availableSignals);

        int[,]? mcnkFlags16 = ReadMcnkFlags(stream, terrainChunks, availableSignals);

        (int[,]? mcseEmitterCounts16, int[]? mcseEntryIds, float[,]? mcsePositionXyz, byte[,]? mcseEntryBytes) =
            ReadMcse(stream, terrainChunks, availableSignals);

        (int[,]? mcrdRefCounts16, int[]? mcrdRefIndices, int[,]? mcrwRefCounts16, int[]? mcrwRefIndices) =
            ReadSplitPlacementChunkReferencesFromBytes(placementSourcePath, placementSourceBytes, availableSignals);

        AdtPlacementCatalog? placementCatalog =
            TryReadPlacementCatalog(sourceAdtPath, stream, fileSummary, placementSourcePath, placementSourceBytes);

        (float[,]? objectMask257, float[,]? objectPreciseMask257, int[,]? objectInstanceMask257, float[,]? mddfMask257, float[,]? modfMask257, float[,]? objectFilteredMask257) =
            BuildObjectMasks(
                sourceAdtPath,
                stream,
                fileSummary,
                availableSignals,
                placementSourcePath,
                placementSourceBytes,
                placementCatalog);

        float[,]? shadowResidualMask256 = BuildShadowResidualMask256(mcshShadowMask256, objectPreciseMask257, availableSignals);

        (float[,]? pm4PathMask, float[,]? pm4BuildingFootprintMask, float[,]? pm4MprlMask) =
            BuildPm4Masks(sourceAdtPath, availableSignals);

        (float[,]? unifiedLiquidMask, float[,]? unifiedLiquidHeight) =
            BuildUnifiedLiquid(mh2oHeight, mh2oPresence, mclqHeight, mclqPresence, null, null, availableSignals);

        float[,]? height65 = DownsampleHeightmap(height257, 65);
        float[,]? height17 = DownsampleHeightmap(height257, 17);

        (int mddfCount, int modfCount, float[,]? mddfData, float[,]? modfData, IReadOnlyList<string> mddfNames, IReadOnlyList<string> modfNames) =
            ExtractPlacementArrays(
                sourceAdtPath,
                stream,
                fileSummary,
                placementSourcePath,
                placementSourceBytes,
                placementCatalog);

        IReadOnlyList<TerrainRawChunkBlob> rawChunks = AdtRawChunkBlobCollector.CollectMemory(
            sourceAdtPath,
            adtBytes,
            textureSourcePath,
            textureSourceBytes,
            placementSourcePath,
            placementSourceBytes);
        if (rawChunks.Count > 0)
            availableSignals.Add("raw_adt_chunks");

        return new TerrainTileTensorPack
        {
            TileName = tileName,
            MapName = ExtractMapName(sourceAdtPath),
            BuildKey = buildVersion ?? string.Empty,
            SourceAdtPath = sourceAdtPath,
            TileX = tileX,
            TileY = tileY,
            Height257 = height257,
            Height65 = height65,
            Height17 = height17,
            MclyTextureIds = mclyTextureIds,
            MclyTextureNames = mclyTextureNames,
            MclyLayerMask = mclyLayerMask,
            McmtMaterialIds = mcmtMaterialIds,
            MampValue = mampValue,
            McalAlphaPack = mcalAlphaPack,
            McalAlphaPack256 = DownsampleAlpha256(mcalAlphaPack),
            MccvRgb = mccvRgb,
            MclvLightingBytes = mclvLightingBytes,
            McnrNormalXyz = mcnrNormalXyz,
            MfboFlightBounds = mfboFlightBounds,
            McnkFlags16 = mcnkFlags16,
            Mh2oSurfaceHeight = mh2oHeight,
            Mh2oDepth = mh2oDepth,
            Mh2oTypeMask = mh2oType,
            Mh2oPresenceMask = mh2oPresence,
            MclqSurfaceHeight = mclqHeight,
            MclqTypeMask = mclqType,
            MclqPresenceMask = mclqPresence,
            MtxfAnimatedMask = mtxfAnimated,
            MtxfTransformId = mtxfTransform,
            HoleMask16 = holeMask,
            McseEmitterCounts16 = mcseEmitterCounts16,
            McseEntryIds = mcseEntryIds,
            McsePositionXyz = mcsePositionXyz,
            McseEntryBytes = mcseEntryBytes,
            McrfDoodadRefCounts16 = mcrfDoodadRefCounts16,
            McrfDoodadRefIndices = mcrfDoodadRefIndices,
            McrfWmoRefCounts16 = mcrfWmoRefCounts16,
            McrfWmoRefIndices = mcrfWmoRefIndices,
            McrdRefCounts16 = mcrdRefCounts16,
            McrdRefIndices = mcrdRefIndices,
            McrwRefCounts16 = mcrwRefCounts16,
            McrwRefIndices = mcrwRefIndices,
            WlLiquidMask = null,
            WlLiquidHeight = null,
            UnifiedLiquidMask = unifiedLiquidMask,
            UnifiedLiquidHeight = unifiedLiquidHeight,
            ObjectMask257 = objectMask257,
            ObjectPreciseMask257 = objectPreciseMask257,
            ObjectInstanceMask257 = objectInstanceMask257,
            MddfMask257 = mddfMask257,
            ModfMask257 = modfMask257,
            ObjectFilteredMask257 = objectFilteredMask257,
            Pm4PathMask = pm4PathMask,
            Pm4BuildingFootprintMask = pm4BuildingFootprintMask,
            Pm4MprlMask = pm4MprlMask,
            McshShadowMask256 = mcshShadowMask256,
            ShadowResidualMask256 = shadowResidualMask256,
            PlacementMddfCount = mddfCount,
            PlacementModfCount = modfCount,
            PlacementMddfData = mddfData,
            PlacementModfData = modfData,
            PlacementMddfNames = mddfNames,
            PlacementModfNames = modfNames,
            RawChunks = rawChunks,
            AvailableSignals = availableSignals,
        };
    }

    /// <summary>
    /// Builds a placeholder <see cref="TerrainTileTensorPack"/> for a tile that has PM4 data
    /// and a minimap but no root ADT. Height, MCAL, and MCLY fields are left null.
    /// This supports Tier 2 tiles in the development-map corpus.
    /// </summary>
    /// <param name="mapDirectory">Path to the map directory containing PM4 files.</param>
    /// <param name="mapName">Map name (e.g. "development").</param>
    /// <param name="tileX">ADT tile X coordinate.</param>
    /// <param name="tileY">ADT tile Y coordinate.</param>
    /// <param name="minimapRgb256">Pre-loaded 256×256×3 minimap RGB data, or null.</param>
    /// <param name="buildKey">Build provenance tag (e.g. "4.0.0.11927").</param>
    public static TerrainTileTensorPack BuildPlaceholder(
        string mapDirectory,
        string mapName,
        int tileX,
        int tileY,
        byte[,,]? minimapRgb256,
        string buildKey)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mapDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);

        string tileName = $"{mapName}_{tileX}_{tileY}";
        HashSet<string> availableSignals = [];

        // ── Build PM4 masks from the map directory ──────────────────────────
        // We construct a synthetic ADT path so the PM4 mask builder can parse
        // tile coordinates from the filename.
        string syntheticAdtPath = Path.Combine(mapDirectory, $"{tileName}.adt");
        (float[,]? pm4PathMask, float[,]? pm4BuildingFootprintMask, float[,]? pm4MprlMask) =
            BuildPm4Masks(syntheticAdtPath, availableSignals);

        // ── Attach minimap if provided ──────────────────────────────────────
        if (minimapRgb256 is not null)
        {
            availableSignals.Add("minimap_rgb_256");
        }

        return new TerrainTileTensorPack
        {
            TileName = tileName,
            MapName = mapName,
            BuildKey = buildKey,
            SourceAdtPath = string.Empty, // no ADT source
            TileX = tileX,
            TileY = tileY,
            Height257 = null,
            Height65 = null,
            Height17 = null,
            MclyTextureIds = null,
            MclyTextureNames = Array.Empty<string>(),
            MclyLayerMask = null,
            McmtMaterialIds = null,
            MampValue = null,
            McalAlphaPack256 = null,
            MccvRgb = null,
            MclvLightingBytes = null,
            McnrNormalXyz = null,
            MfboFlightBounds = null,
            Mh2oSurfaceHeight = null,
            Mh2oDepth = null,
            Mh2oTypeMask = null,
            Mh2oPresenceMask = null,
            MclqSurfaceHeight = null,
            MclqTypeMask = null,
            MclqPresenceMask = null,
            MtxfAnimatedMask = null,
            MtxfTransformId = null,
            HoleMask16 = null,
            McseEmitterCounts16 = null,
            McseEntryIds = null,
            McsePositionXyz = null,
            McseEntryBytes = null,
            McrfDoodadRefCounts16 = null,
            McrfDoodadRefIndices = null,
            McrfWmoRefCounts16 = null,
            McrfWmoRefIndices = null,
            McrdRefCounts16 = null,
            McrdRefIndices = null,
            McrwRefCounts16 = null,
            McrwRefIndices = null,
            WlLiquidMask = null,
            WlLiquidHeight = null,
            UnifiedLiquidMask = null,
            UnifiedLiquidHeight = null,
            ObjectMask257 = null,
            ObjectPreciseMask257 = null,
            Pm4PathMask = pm4PathMask,
            Pm4BuildingFootprintMask = pm4BuildingFootprintMask,
            Pm4MprlMask = pm4MprlMask,
            McshShadowMask256 = null,
            ShadowResidualMask256 = null,
            MinimapRgb256 = minimapRgb256,
            MinimapSourceTag = minimapRgb256 is not null ? "raw" : string.Empty,
            AvailableSignals = availableSignals,
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Heightmap assembly (MCVT)
    // ═══════════════════════════════════════════════════════════════════════

    private static float[,]? AssembleHeightmap(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return null;

        float[,] heightmap = new float[TileHeightmapSize, TileHeightmapSize];
        bool any = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            float baseHeight = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x70, 4)));

            int mcvtOffset = LocateMcvtDataOffset(payload);
            if (mcvtOffset < 0)
                continue;

            any = true;
            for (int sampleIndex = 0; sampleIndex < McvtSampleCount; sampleIndex++)
            {
                float rawHeight = BitConverter.ToSingle(payload, mcvtOffset + (sampleIndex * sizeof(float)));
                float absoluteHeight = rawHeight + baseHeight;

                ResolveTileSampleCoordinates(chunkX, chunkY, sampleIndex, out int sampleX, out int sampleY);
                heightmap[sampleY, sampleX] = absoluteHeight;
            }
        }

        if (!any)
            return null;

        FillHeightmapGaps(heightmap);
        signals.Add("height_257");
        return heightmap;
    }

    private static void FillHeightmapGaps(float[,] hm)
    {
        int size = hm.GetLength(0);
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                if (hm[y, x] != 0f) continue;
                if (x > 0 && hm[y, x - 1] != 0f) hm[y, x] = hm[y, x - 1];
                else if (y > 0 && hm[y - 1, x] != 0f) hm[y, x] = hm[y - 1, x];
                else if (x < size - 1 && hm[y, x + 1] != 0f) hm[y, x] = hm[y, x + 1];
                else if (y < size - 1 && hm[y + 1, x] != 0f) hm[y, x] = hm[y + 1, x];
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Normal assembly (MCNR)
    // ═══════════════════════════════════════════════════════════════════════

    private static float[,,]? AssembleNormals(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return null;

        float[, ,] normals = new float[TileHeightmapSize, TileHeightmapSize, 3];
        bool any = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            int mcnrOffset = LocateMcnrDataOffset(payload);
            if (mcnrOffset < 0)
                continue;

            any = true;
            for (int sampleIndex = 0; sampleIndex < McvtSampleCount; sampleIndex++)
            {
                int normalOffset = mcnrOffset + (sampleIndex * 3);
                float nx = DecodeNormalComponent(payload[normalOffset + 0]);
                float ny = DecodeNormalComponent(payload[normalOffset + 1]);
                float nz = DecodeNormalComponent(payload[normalOffset + 2]);

                ResolveTileSampleCoordinates(chunkX, chunkY, sampleIndex, out int sampleX, out int sampleY);
                normals[sampleY, sampleX, 0] = nx;
                normals[sampleY, sampleX, 1] = ny;
                normals[sampleY, sampleX, 2] = nz;
            }
        }

        if (!any)
            return null;

        signals.Add("mcnr_normal_xyz");
        return normals;
    }

    private static float DecodeNormalComponent(byte value)
    {
        return (sbyte)value / 127f;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MCCV assembly
    // ═══════════════════════════════════════════════════════════════════════

    private static float[,,]? AssembleMccv(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return null;

        float[, ,] colors = new float[TileHeightmapSize, TileHeightmapSize, 3];
        bool any = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            int mccvOffset = LocateMccvDataOffset(payload);
            if (mccvOffset < 0)
                continue;

            any = true;
            for (int sampleIndex = 0; sampleIndex < McvtSampleCount; sampleIndex++)
            {
                int colorOffset = mccvOffset + (sampleIndex * 4);
                float r = payload[colorOffset + 0] / 255f;
                float g = payload[colorOffset + 1] / 255f;
                float b = payload[colorOffset + 2] / 255f;

                ResolveTileSampleCoordinates(chunkX, chunkY, sampleIndex, out int sampleX, out int sampleY);
                colors[sampleY, sampleX, 0] = r;
                colors[sampleY, sampleX, 1] = g;
                colors[sampleY, sampleX, 2] = b;
            }
        }

        if (!any)
            return null;

        signals.Add("mccv_rgb");
        return colors;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Texture data (MCLY + MCAL)
    // ═══════════════════════════════════════════════════════════════════════

    private static (int[,,]? textureIds, IReadOnlyList<string> textureNames, bool[,,]? layerMask, byte[,,]? materialIds, byte[]? mampValue, float[,,]? alphaPack, float[,]? mcshShadowMask256)
        ReadTextureData(string adtPath, string? textureSourcePath, AdtFormatProfile profile, HashSet<string> signals)
    {
        string? effectiveTexturePath = textureSourcePath;
        if (string.IsNullOrWhiteSpace(effectiveTexturePath))
        {
            AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
            effectiveTexturePath = family.TextureSourcePath;
        }

        if (string.IsNullOrWhiteSpace(effectiveTexturePath) || !File.Exists(effectiveTexturePath))
        {
            if (!profile.PreferTex0ForTextureData && File.Exists(adtPath))
                effectiveTexturePath = adtPath;
            else
                return (null, Array.Empty<string>(), null, null, null, null, null);
        }

        try
        {
            AdtTextureFile textureFile = AdtTextureReader.Read(effectiveTexturePath, profile.DecodeProfile);
            if (textureFile.Chunks.Count == 0)
                return (null, textureFile.TextureNames, null, null, textureFile.MampValue.HasValue ? [textureFile.MampValue.Value] : null, null, null);

            int[,,] textureIds = new int[TileChunks, TileChunks, 4];
            bool[,,] layerMask = new bool[TileChunks, TileChunks, 4];
            byte[,,] materialIds = new byte[TileChunks, TileChunks, 4];
            float[,,] alphaPack = new float[TileAlphaSize, TileAlphaSize, 4];
            float[,] shadowAccum256 = new float[TileMinimapSize, TileMinimapSize];
            int[,] shadowCount256 = new int[TileMinimapSize, TileMinimapSize];

            // Initialize texture IDs to -1 (missing)
            for (int y = 0; y < TileChunks; y++)
                for (int x = 0; x < TileChunks; x++)
                    for (int l = 0; l < 4; l++)
                        textureIds[y, x, l] = -1;

            bool any = false;
            foreach (AdtTextureChunk chunk in textureFile.Chunks)
            {
                if (chunk.Layers.Count == 0)
                    continue;

                int chunkX = chunk.ChunkX;
                int chunkY = chunk.ChunkY;
                if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                    continue;

                any = true;

                if (chunk.MaterialIds is { Length: > 0 })
                {
                    for (int materialIndex = 0; materialIndex < chunk.MaterialIds.Length && materialIndex < 4; materialIndex++)
                        materialIds[chunkY, chunkX, materialIndex] = chunk.MaterialIds[materialIndex];
                }

                for (int layerIndex = 0; layerIndex < chunk.Layers.Count && layerIndex < 4; layerIndex++)
                {
                    AdtTextureChunkLayer layer = chunk.Layers[layerIndex];
                    textureIds[chunkY, chunkX, layerIndex] = (int)layer.TextureId;
                    layerMask[chunkY, chunkX, layerIndex] = true;

                    byte[]? alphaMap = layer.DecodedAlpha?.AlphaMap;
                    if (alphaMap is null || alphaMap.Length != ChunkAlphaSize * ChunkAlphaSize)
                        continue;

                    for (int localY = 0; localY < ChunkAlphaSize; localY++)
                    {
                        for (int localX = 0; localX < ChunkAlphaSize; localX++)
                        {
                            int globalX = (chunkX * ChunkAlphaSize) + localX;
                            int globalY = (chunkY * ChunkAlphaSize) + localY;
                            alphaPack[globalY, globalX, layerIndex] = alphaMap[(localY * ChunkAlphaSize) + localX] / 255f;
                        }
                    }
                }

                byte[]? shadowMap = chunk.ShadowMap;
                if (shadowMap is { Length: ChunkAlphaSize * ChunkAlphaSize })
                {
                    for (int localY = 0; localY < ChunkAlphaSize; localY++)
                    {
                        for (int localX = 0; localX < ChunkAlphaSize; localX++)
                        {
                            int globalX = (chunkX * ChunkAlphaSize) + localX;
                            int globalY = (chunkY * ChunkAlphaSize) + localY;
                            int minimapX = Math.Clamp(globalX / 4, 0, TileMinimapSize - 1);
                            int minimapY = Math.Clamp(globalY / 4, 0, TileMinimapSize - 1);
                            shadowAccum256[minimapY, minimapX] += shadowMap[(localY * ChunkAlphaSize) + localX] / 255f;
                            shadowCount256[minimapY, minimapX]++;
                        }
                    }
                }
            }

            if (!any)
                return (null, textureFile.TextureNames, null, null, textureFile.MampValue.HasValue ? [textureFile.MampValue.Value] : null, null, null);

            float[,]? mcshShadowMask256 = null;
            bool anyShadow = false;
            for (int y = 0; y < TileMinimapSize; y++)
            {
                for (int x = 0; x < TileMinimapSize; x++)
                {
                    int count = shadowCount256[y, x];
                    if (count <= 0)
                        continue;

                    mcshShadowMask256 ??= new float[TileMinimapSize, TileMinimapSize];
                    float value = shadowAccum256[y, x] / count;
                    mcshShadowMask256[y, x] = value;
                    anyShadow |= value > 0f;
                }
            }

            signals.Add("mcly_texture_ids");
            signals.Add("mcly_layer_mask");
            signals.Add("mcal_alpha_pack_256");
            if (textureFile.MampValue.HasValue)
                signals.Add("mamp_value");
            if (textureFile.Chunks.Any(static chunk => chunk.MaterialIds is { Length: > 0 }))
                signals.Add("mcmt_material_ids");
            if (textureFile.TextureNames.Count > 0)
                signals.Add("mcly_texture_names");
            if (anyShadow)
                signals.Add("mcsh_shadow_mask_256");
            return (textureIds, textureFile.TextureNames, layerMask, materialIds, textureFile.MampValue.HasValue ? [textureFile.MampValue.Value] : null, alphaPack, anyShadow ? mcshShadowMask256 : null);
        }
        catch
        {
            return (null, Array.Empty<string>(), null, null, null, null, null);
        }
    }

    private static byte[,,]? AssembleMclv(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return null;

        byte[,,] colors = new byte[TileHeightmapSize, TileHeightmapSize, 4];
        bool any = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            int mclvOffset = LocateMcnkSubchunkDataOffset(payload, AdtChunkIds.Mclv, 145 * 4);
            if (mclvOffset < 0)
                continue;

            any = true;
            for (int sampleIndex = 0; sampleIndex < McvtSampleCount; sampleIndex++)
            {
                int colorOffset = mclvOffset + (sampleIndex * 4);
                ResolveTileSampleCoordinates(chunkX, chunkY, sampleIndex, out int sampleX, out int sampleY);
                colors[sampleY, sampleX, 0] = payload[colorOffset + 0];
                colors[sampleY, sampleX, 1] = payload[colorOffset + 1];
                colors[sampleY, sampleX, 2] = payload[colorOffset + 2];
                colors[sampleY, sampleX, 3] = payload[colorOffset + 3];
            }
        }

        if (!any)
            return null;

        signals.Add("mclv_lighting_bytes");
        return colors;
    }

    private static int[,,]? ReadMfbo(Stream stream, MapFileSummary fileSummary, HashSet<string> signals)
    {
        byte[]? payload = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Mfbo);
        if (payload is not { Length: >= 36 })
            return null;

        int[,,] heights = new int[2, 3, 3];
        for (int plane = 0; plane < 2; plane++)
        {
            for (int row = 0; row < 3; row++)
            {
                for (int column = 0; column < 3; column++)
                {
                    int offset = (((plane * 3) + row) * 3 + column) * sizeof(short);
                    heights[plane, row, column] = BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset, sizeof(short)));
                }
            }
        }

        signals.Add("mfbo_flight_bounds");
        return heights;
    }

    private static float[,]? BuildShadowResidualMask256(float[,]? shadowMask256, float[,]? objectPreciseMask257, HashSet<string> signals)
    {
        if (shadowMask256 is null || objectPreciseMask257 is null)
            return null;

        float[,] objectMask256 = ResizeScalarGrid(objectPreciseMask257, TileMinimapSize);
        float[,] residual = new float[TileMinimapSize, TileMinimapSize];
        bool any = false;

        for (int y = 0; y < TileMinimapSize; y++)
        {
            for (int x = 0; x < TileMinimapSize; x++)
            {
                float value = MathF.Max(0f, shadowMask256[y, x] - Math.Clamp(objectMask256[y, x], 0f, 1f));
                residual[y, x] = value;
                any |= value > 0f;
            }
        }

        if (!any)
            return null;

        signals.Add("shadow_residual_mask_256");
        return residual;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MH2O liquid
    // ═══════════════════════════════════════════════════════════════════════

    private static (int[,,]? MclyTextureIds, IReadOnlyList<string> MclyTextureNames, bool[,,]? MclyLayerMask, byte[,,]? McmtMaterialIds, byte[]? MampValue, float[,,]? McalAlphaPack, float[,]? McshShadowMask256)
        ReadTextureDataFromBytes(
            string sourceAdtPath,
            byte[] adtBytes,
            string? textureSourcePath,
            byte[]? textureSourceBytes,
            AdtFormatProfile profile,
            HashSet<string> signals)
    {
        string? effectiveTextureSourcePath = textureSourcePath;
        byte[]? effectiveTextureBytes = textureSourceBytes;

        try
        {
            AdtTextureFile textureFile;
            if (effectiveTextureBytes is not null)
            {
                effectiveTextureSourcePath ??= $"{sourceAdtPath}_tex0";
                using MemoryStream textureStream = new(effectiveTextureBytes, writable: false);
                MapFileSummary textureSummary = MapFileSummaryReader.Read(textureStream, effectiveTextureSourcePath);
                textureFile = AdtTextureReader.Read(textureStream, textureSummary, profile.DecodeProfile);
            }
            else
            {
                // Mixed-era archive tiles can omit _tex0 entirely while still carrying
                // valid inline MCLY/MCAL in the root ADT. Match the older file-path
                // builder behavior by falling back to the root bytes in that case.
                using MemoryStream rootStream = new(adtBytes, writable: false);
                MapFileSummary rootSummary = MapFileSummaryReader.Read(rootStream, sourceAdtPath);
                textureFile = AdtTextureReader.Read(rootStream, rootSummary, profile.DecodeProfile);
            }

            if (textureFile.Chunks.Count == 0)
                return (null, textureFile.TextureNames, null, null, textureFile.MampValue.HasValue ? [textureFile.MampValue.Value] : null, null, null);

            int[,,] textureIds = new int[TileChunks, TileChunks, 4];
            bool[,,] layerMask = new bool[TileChunks, TileChunks, 4];
            byte[,,] materialIds = new byte[TileChunks, TileChunks, 4];
            float[,,] alphaPack = new float[TileAlphaSize, TileAlphaSize, 4];
            float[,] shadowAccum256 = new float[TileMinimapSize, TileMinimapSize];
            int[,] shadowCount256 = new int[TileMinimapSize, TileMinimapSize];

            for (int y = 0; y < TileChunks; y++)
                for (int x = 0; x < TileChunks; x++)
                    for (int l = 0; l < 4; l++)
                        textureIds[y, x, l] = -1;

            bool any = false;
            foreach (AdtTextureChunk chunk in textureFile.Chunks)
            {
                if (chunk.Layers.Count == 0)
                    continue;

                int chunkX = chunk.ChunkX;
                int chunkY = chunk.ChunkY;
                if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                    continue;

                any = true;

                if (chunk.MaterialIds is { Length: > 0 })
                {
                    for (int materialIndex = 0; materialIndex < chunk.MaterialIds.Length && materialIndex < 4; materialIndex++)
                        materialIds[chunkY, chunkX, materialIndex] = chunk.MaterialIds[materialIndex];
                }

                for (int layerIndex = 0; layerIndex < chunk.Layers.Count && layerIndex < 4; layerIndex++)
                {
                    AdtTextureChunkLayer layer = chunk.Layers[layerIndex];
                    textureIds[chunkY, chunkX, layerIndex] = (int)layer.TextureId;
                    layerMask[chunkY, chunkX, layerIndex] = true;

                    byte[]? alphaMap = layer.DecodedAlpha?.AlphaMap;
                    if (alphaMap is null || alphaMap.Length != ChunkAlphaSize * ChunkAlphaSize)
                        continue;

                    for (int localY = 0; localY < ChunkAlphaSize; localY++)
                    {
                        for (int localX = 0; localX < ChunkAlphaSize; localX++)
                        {
                            int globalX = (chunkX * ChunkAlphaSize) + localX;
                            int globalY = (chunkY * ChunkAlphaSize) + localY;
                            alphaPack[globalY, globalX, layerIndex] = alphaMap[(localY * ChunkAlphaSize) + localX] / 255f;
                        }
                    }
                }

                byte[]? shadowMap = chunk.ShadowMap;
                if (shadowMap is { Length: ChunkAlphaSize * ChunkAlphaSize })
                {
                    for (int localY = 0; localY < ChunkAlphaSize; localY++)
                    {
                        for (int localX = 0; localX < ChunkAlphaSize; localX++)
                        {
                            int globalX = (chunkX * ChunkAlphaSize) + localX;
                            int globalY = (chunkY * ChunkAlphaSize) + localY;
                            int minimapX = Math.Clamp(globalX / 4, 0, TileMinimapSize - 1);
                            int minimapY = Math.Clamp(globalY / 4, 0, TileMinimapSize - 1);
                            shadowAccum256[minimapY, minimapX] += shadowMap[(localY * ChunkAlphaSize) + localX] / 255f;
                            shadowCount256[minimapY, minimapX]++;
                        }
                    }
                }
            }

            if (!any)
                return (null, textureFile.TextureNames, null, null, textureFile.MampValue.HasValue ? [textureFile.MampValue.Value] : null, null, null);

            float[,]? mcshShadowMask256 = null;
            bool anyShadow = false;
            for (int y = 0; y < TileMinimapSize; y++)
            {
                for (int x = 0; x < TileMinimapSize; x++)
                {
                    int count = shadowCount256[y, x];
                    if (count <= 0)
                        continue;

                    mcshShadowMask256 ??= new float[TileMinimapSize, TileMinimapSize];
                    float value = shadowAccum256[y, x] / count;
                    mcshShadowMask256[y, x] = value;
                    anyShadow |= value > 0f;
                }
            }

            signals.Add("mcly_texture_ids");
            signals.Add("mcly_layer_mask");
            signals.Add("mcal_alpha_pack_256");
            if (textureFile.MampValue.HasValue)
                signals.Add("mamp_value");
            if (textureFile.Chunks.Any(static chunk => chunk.MaterialIds is { Length: > 0 }))
                signals.Add("mcmt_material_ids");
            if (anyShadow)
                signals.Add("mcsh_shadow_mask_256");

            return (textureIds, textureFile.TextureNames, layerMask, materialIds, textureFile.MampValue.HasValue ? [textureFile.MampValue.Value] : null, alphaPack, mcshShadowMask256);
        }
        catch
        {
            return (null, Array.Empty<string>(), null, null, null, null, null);
        }
    }

    private static (float[,]? height, float[,]? depth, int[,]? typeMask, bool[,]? presenceMask)
        ReadMh2o(Stream stream, MapFileSummary fileSummary, AdtFormatProfile profile, HashSet<string> signals)
    {
        try
        {
            AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, fileSummary, profile);
            if (liquidFile.Chunks.Count == 0)
                return (null, null, null, null);

            float[,] heights = new float[TileHeightmapSize, TileHeightmapSize];
            float[,] depths = new float[TileHeightmapSize, TileHeightmapSize];
            int[,] typeMask = new int[TileHeightmapSize, TileHeightmapSize];
            bool[,] presenceMask = new bool[TileHeightmapSize, TileHeightmapSize];
            bool any = false;

            foreach (AdtLiquidChunk chunk in liquidFile.Chunks)
            {
                if (chunk.Layers.Count == 0)
                    continue;

                int chunkIndex = chunk.ChunkIndex;
                int chunkX = chunkIndex % TileChunks;
                int chunkY = chunkIndex / TileChunks;

                // Each MH2O chunk covers an 8×8 cell region within the MCNK
                // We map the chunk-level liquid data to the tile grid
                foreach (AdtLiquidLayer layer in chunk.Layers)
                {
                    if (layer.Heights is null)
                        continue;

                    any = true;
                    int vertW = layer.Width + 1;
                    int vertH = layer.Height + 1;

                    for (int localY = 0; localY < vertH; localY++)
                    {
                        for (int localX = 0; localX < vertW; localX++)
                        {
                            int vertexIndex = (localY * vertW) + localX;
                            if (vertexIndex >= layer.Heights.Length)
                                continue;

                            int globalX = (chunkX * HalfStepsPerChunk) + layer.XOffset + localX;
                            int globalY = (chunkY * HalfStepsPerChunk) + layer.YOffset + localY;
                            if (globalX >= TileHeightmapSize || globalY >= TileHeightmapSize)
                                continue;

                            heights[globalY, globalX] = layer.Heights[vertexIndex];
                            typeMask[globalY, globalX] = (int)layer.BasicType;
                            presenceMask[globalY, globalX] = true;

                            if (layer.Depths is not null && vertexIndex < layer.Depths.Length)
                                depths[globalY, globalX] = layer.Depths[vertexIndex];
                        }
                    }
                }
            }

            if (!any)
                return (null, null, null, null);

            signals.Add("mh2o_surface_height");
            signals.Add("mh2o_depth");
            signals.Add("mh2o_type_mask");
            return (heights, depths, typeMask, presenceMask);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MH2O] Warning: Failed to parse liquid data for tile: {ex.GetType().Name}: {ex.Message}");
            return (null, null, null, null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════

    private static (float[,]? height, int[,]? typeMask, bool[,]? presenceMask)
        ReadMclq(Stream stream, List<MapChunkLocation> chunks, AdtFormatProfile profile, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return (null, null, null);

        const int mclqVertsPerChunk = 9;
        const int mclqCellsPerChunk = 8;
        const int gridSize = (mclqCellsPerChunk * TileChunks) + 1; // 129

        float[,] heights = new float[gridSize, gridSize];
        int[,] types = new int[gridSize, gridSize];
        bool[,] presenceMask = new bool[gridSize, gridSize];
        bool any = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            AdtMclqData? mclq = AdtMclqReader.Read(payload, profile);
            if (mclq is null)
                continue;

            any = true;
            for (int vy = 0; vy < mclqVertsPerChunk; vy++)
            {
                for (int vx = 0; vx < mclqVertsPerChunk; vx++)
                {
                    int globalX = (chunkX * mclqCellsPerChunk) + vx;
                    int globalY = (chunkY * mclqCellsPerChunk) + vy;
                    int vertexIndex = (vy * mclqVertsPerChunk) + vx;

                    if (vertexIndex < mclq.Heights.Length)
                    {
                        heights[globalY, globalX] = mclq.Heights[vertexIndex];
                        types[globalY, globalX] = mclq.LiquidType;
                        presenceMask[globalY, globalX] = true;
                    }
                }
            }
        }

        if (!any)
            return (null, null, null);

        signals.Add("mclq_surface_height");
        signals.Add("mclq_type_mask");
        return (heights, types, presenceMask);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MTXF texture flags
    // ═══════════════════════════════════════════════════════════════════════

    private static (int[,]? animatedMask, int[,]? transformId)
        ReadMtxf(Stream stream, MapFileSummary fileSummary, int[,,]? mclyTextureIds, HashSet<string> signals)
    {
        AdtMtxfData? mtxf = AdtMtxfReader.Read(stream, fileSummary);
        if (mtxf is null || mclyTextureIds is null)
            return (null, null);

        int[,] animated = new int[TileChunks, TileChunks];
        int[,] transform = new int[TileChunks, TileChunks];
        bool any = false;

        for (int cy = 0; cy < TileChunks; cy++)
        {
            for (int cx = 0; cx < TileChunks; cx++)
            {
                int textureId = mclyTextureIds[cy, cx, 0];
                if (textureId < 0 || textureId >= mtxf.Flags.Length)
                    continue;

                any = true;
                animated[cy, cx] = (mtxf.Flags[textureId] & 0x01) != 0 ? 1 : 0;
                animated[cy, cx] |= mtxf.Flags[textureId] & ~0x01; // preserve other flags too

                if (mtxf.TransformIds is not null && textureId < mtxf.TransformIds.Length)
                    transform[cy, cx] = mtxf.TransformIds[textureId];
            }
        }

        if (!any)
            return (null, null);

        signals.Add("mtxf_animated_mask");
        signals.Add("mtxf_transform_id");
        return (animated, transform);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MCRF object references + hole mask
    // ═══════════════════════════════════════════════════════════════════════

    private static (bool[,]? holeMask, int[,]? objectMask16, int[,]? mcrfDoodadRefCounts16, int[]? mcrfDoodadRefIndices, int[,]? mcrfWmoRefCounts16, int[]? mcrfWmoRefIndices)
        ReadMcrfAndHoles(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return (null, null, null, null, null, null);

        bool[,] holes = new bool[TileChunks, TileChunks];
        int[,] objects = new int[TileChunks, TileChunks];
        int[,] mcrfDoodadRefCounts16 = new int[TileChunks, TileChunks];
        int[,] mcrfWmoRefCounts16 = new int[TileChunks, TileChunks];
        List<int> mcrfDoodadRefIndices = [];
        List<int> mcrfWmoRefIndices = [];
        bool anyHoles = false;
        bool anyObjects = false;
        bool anyMcrfDoodads = false;
        bool anyMcrfWmos = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            // Hole mask from MCNK header flags (bits 8-15 in some formats)
            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x00, 4));
            if ((flags & 0x0000FF00) != 0)
            {
                holes[chunkY, chunkX] = true;
                anyHoles = true;
            }

            // MCRF presence = objects in chunk (full 257x257 projection requires MDDF/MODF parsing)
            if (AdtMcrfReader.TryLocateMcrfPayload(payload, out int mcrfOffset, out int mcrfSize))
            {
                if (mcrfSize >= 4)
                {
                    int doodadCount = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x14, 4));
                    int wmoCount = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x3C, 4));
                    AdtMcrfData refs = AdtMcrfReader.Read(payload.AsSpan(mcrfOffset, mcrfSize).ToArray(), doodadCount, wmoCount);

                    int totalRefCount = refs.DoodadIndices.Length + refs.WmoIndices.Length;
                    objects[chunkY, chunkX] = totalRefCount;
                    anyObjects |= totalRefCount > 0;

                    if (refs.DoodadIndices.Length > 0)
                    {
                        mcrfDoodadRefCounts16[chunkY, chunkX] = refs.DoodadIndices.Length;
                        mcrfDoodadRefIndices.AddRange(refs.DoodadIndices);
                        anyMcrfDoodads = true;
                    }

                    if (refs.WmoIndices.Length > 0)
                    {
                        mcrfWmoRefCounts16[chunkY, chunkX] = refs.WmoIndices.Length;
                        mcrfWmoRefIndices.AddRange(refs.WmoIndices);
                        anyMcrfWmos = true;
                    }
                }
            }
        }

        if (anyHoles)
            signals.Add("hole_mask_16");
        if (anyObjects)
            signals.Add("object_mask_257"); // coarse 16x16 stored in int array; consumer may upsample
        if (anyMcrfDoodads)
            signals.Add("mcrf_doodad_ref_indices");
        if (anyMcrfWmos)
            signals.Add("mcrf_wmo_ref_indices");

        return (
            anyHoles ? holes : null,
            anyObjects ? objects : null,
            anyMcrfDoodads ? mcrfDoodadRefCounts16 : null,
            anyMcrfDoodads ? mcrfDoodadRefIndices.ToArray() : null,
            anyMcrfWmos ? mcrfWmoRefCounts16 : null,
            anyMcrfWmos ? mcrfWmoRefIndices.ToArray() : null);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MCNK header flags (liquid type bits 2-5)
    // ═══════════════════════════════════════════════════════════════════════

    private static int[,]? ReadMcnkFlags(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return null;

        int[,] flags16 = new int[TileChunks, TileChunks];
        bool anyNonZero = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < 12)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            uint flags = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0x00, 4));
            flags16[chunkY, chunkX] = (int)flags;
            if ((flags & 0x3C) != 0)
                anyNonZero = true;
        }

        if (anyNonZero)
            signals.Add("mcnk_flags_16");

        return flags16;
    }

    private static (int[,]? mcseEmitterCounts16, int[]? mcseEntryIds, float[,]? mcsePositionXyz, byte[,]? mcseEntryBytes)
        ReadMcse(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return (null, null, null, null);

        int[,] mcseEmitterCounts16 = new int[TileChunks, TileChunks];
        List<int> mcseEntryIds = [];
        List<float> mcsePositions = [];
        List<byte[]> mcseEntryRows = [];
        int? entrySize = null;
        bool anyMcse = false;

        foreach (MapChunkLocation chunk in chunks)
        {
            byte[] payload = ReadChunkPayload(stream, chunk);
            if (payload.Length < RootMcnkHeaderSize)
                continue;

            int chunkX = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x04, 4));
            int chunkY = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x08, 4));
            if ((uint)chunkX >= TileChunks || (uint)chunkY >= TileChunks)
                continue;

            if (!AdtMcseReader.TryLocateMcsePayload(payload, out int mcseOffset, out int mcsePayloadSize))
                continue;

            int declaredEmitterCount = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0x5C, 4));
            AdtMcseData mcse = AdtMcseReader.Read(payload.AsSpan(mcseOffset, mcsePayloadSize).ToArray(), declaredEmitterCount);
            if (mcse.EntryCount <= 0 || mcse.EntryBytes is null)
                return (null, null, null, null);

            entrySize ??= mcse.EntrySize;
            if (entrySize != mcse.EntrySize)
                return (null, null, null, null);

            mcseEmitterCounts16[chunkY, chunkX] = mcse.EntryCount;
            anyMcse = true;

            for (int entryIndex = 0; entryIndex < mcse.EntryCount; entryIndex++)
            {
                byte[] row = new byte[mcse.EntrySize];
                for (int byteIndex = 0; byteIndex < mcse.EntrySize; byteIndex++)
                    row[byteIndex] = mcse.EntryBytes[entryIndex, byteIndex];
                mcseEntryRows.Add(row);
            }

            if (mcse.EntryIds is not null)
                mcseEntryIds.AddRange(mcse.EntryIds);

            if (mcse.PositionXyz is not null)
            {
                for (int entryIndex = 0; entryIndex < mcse.PositionXyz.GetLength(0); entryIndex++)
                {
                    mcsePositions.Add(mcse.PositionXyz[entryIndex, 0]);
                    mcsePositions.Add(mcse.PositionXyz[entryIndex, 1]);
                    mcsePositions.Add(mcse.PositionXyz[entryIndex, 2]);
                }
            }
        }

        if (!anyMcse || entrySize is null)
            return (null, null, null, null);

        byte[,] entryBytes = new byte[mcseEntryRows.Count, entrySize.Value];
        for (int rowIndex = 0; rowIndex < mcseEntryRows.Count; rowIndex++)
            for (int byteIndex = 0; byteIndex < entrySize.Value; byteIndex++)
                entryBytes[rowIndex, byteIndex] = mcseEntryRows[rowIndex][byteIndex];

        float[,]? positionXyz = null;
        if (mcsePositions.Count > 0)
        {
            positionXyz = new float[mcsePositions.Count / 3, 3];
            for (int entryIndex = 0; entryIndex < positionXyz.GetLength(0); entryIndex++)
            {
                int baseIndex = entryIndex * 3;
                positionXyz[entryIndex, 0] = mcsePositions[baseIndex];
                positionXyz[entryIndex, 1] = mcsePositions[baseIndex + 1];
                positionXyz[entryIndex, 2] = mcsePositions[baseIndex + 2];
            }
        }

        signals.Add("mcse_entry_bytes");
        if (mcseEntryIds.Count > 0)
            signals.Add("mcse_entry_ids");
        if (positionXyz is not null)
            signals.Add("mcse_position_xyz");

        return (
            mcseEmitterCounts16,
            mcseEntryIds.Count > 0 ? mcseEntryIds.ToArray() : null,
            positionXyz,
            entryBytes);
    }

    private static (int[,]? mcrdRefCounts16, int[]? mcrdRefIndices, int[,]? mcrwRefCounts16, int[]? mcrwRefIndices)
        ReadSplitPlacementChunkReferencesFromBytes(string? placementSourcePath, byte[]? placementBytes, HashSet<string> signals)
    {
        if (placementBytes is null || string.IsNullOrWhiteSpace(placementSourcePath))
            return (null, null, null, null);

        using MemoryStream stream = new(placementBytes, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, placementSourcePath);
        List<MapChunkLocation> placementChunks = fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk).ToList();
        if (placementChunks.Count == 0)
            return (null, null, null, null);

        int[,] mcrdCounts = new int[TileChunks, TileChunks];
        int[,] mcrwCounts = new int[TileChunks, TileChunks];
        List<int> mcrdIndices = [];
        List<int> mcrwIndices = [];
        bool anyMcrd = false;
        bool anyMcrw = false;

        for (int chunkIndex = 0; chunkIndex < placementChunks.Count && chunkIndex < TileChunks * TileChunks; chunkIndex++)
        {
            int chunkX = chunkIndex % TileChunks;
            int chunkY = chunkIndex / TileChunks;
            byte[] payload = ReadChunkPayload(stream, placementChunks[chunkIndex]);

            byte[]? mcrdPayload = TryReadSplitMcnkSubchunkPayload(payload, AdtChunkIds.Mcrd);
            if (mcrdPayload is { Length: >= 4 })
            {
                int count = mcrdPayload.Length / sizeof(int);
                mcrdCounts[chunkY, chunkX] = count;
                for (int index = 0; index < count; index++)
                    mcrdIndices.Add(BinaryPrimitives.ReadInt32LittleEndian(mcrdPayload.AsSpan(index * sizeof(int), sizeof(int))));
                anyMcrd = true;
            }

            byte[]? mcrwPayload = TryReadSplitMcnkSubchunkPayload(payload, AdtChunkIds.Mcrw);
            if (mcrwPayload is { Length: >= 4 })
            {
                int count = mcrwPayload.Length / sizeof(int);
                mcrwCounts[chunkY, chunkX] = count;
                for (int index = 0; index < count; index++)
                    mcrwIndices.Add(BinaryPrimitives.ReadInt32LittleEndian(mcrwPayload.AsSpan(index * sizeof(int), sizeof(int))));
                anyMcrw = true;
            }
        }

        if (anyMcrd)
        {
            signals.Add("mcrd_ref_counts_16");
            signals.Add("mcrd_ref_indices");
        }

        if (anyMcrw)
        {
            signals.Add("mcrw_ref_counts_16");
            signals.Add("mcrw_ref_indices");
        }

        return (anyMcrd ? mcrdCounts : null, anyMcrd ? [.. mcrdIndices] : null, anyMcrw ? mcrwCounts : null, anyMcrw ? [.. mcrwIndices] : null);
    }

    private static (int[,]? mcrdRefCounts16, int[]? mcrdRefIndices, int[,]? mcrwRefCounts16, int[]? mcrwRefIndices)
        ReadSplitPlacementChunkReferences(string adtPath, HashSet<string> signals)
    {
        AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
        string? placementPath = family.PlacementSourcePath;
        if (string.IsNullOrWhiteSpace(placementPath) || !File.Exists(placementPath))
            return (null, null, null, null);

        using FileStream stream = File.OpenRead(placementPath);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(placementPath));
        List<MapChunkLocation> placementChunks = fileSummary.Chunks.Where(static chunk => chunk.Id == MapChunkIds.Mcnk).ToList();
        if (placementChunks.Count == 0)
            return (null, null, null, null);

        int[,] mcrdCounts = new int[TileChunks, TileChunks];
        int[,] mcrwCounts = new int[TileChunks, TileChunks];
        List<int> mcrdIndices = [];
        List<int> mcrwIndices = [];
        bool anyMcrd = false;
        bool anyMcrw = false;

        for (int chunkIndex = 0; chunkIndex < placementChunks.Count && chunkIndex < TileChunks * TileChunks; chunkIndex++)
        {
            int chunkX = chunkIndex % TileChunks;
            int chunkY = chunkIndex / TileChunks;
            byte[] payload = ReadChunkPayload(stream, placementChunks[chunkIndex]);

            byte[]? mcrdPayload = TryReadSplitMcnkSubchunkPayload(payload, AdtChunkIds.Mcrd);
            if (mcrdPayload is { Length: >= 4 })
            {
                int count = mcrdPayload.Length / sizeof(int);
                mcrdCounts[chunkY, chunkX] = count;
                for (int index = 0; index < count; index++)
                    mcrdIndices.Add(BinaryPrimitives.ReadInt32LittleEndian(mcrdPayload.AsSpan(index * sizeof(int), sizeof(int))));
                anyMcrd = true;
            }

            byte[]? mcrwPayload = TryReadSplitMcnkSubchunkPayload(payload, AdtChunkIds.Mcrw);
            if (mcrwPayload is { Length: >= 4 })
            {
                int count = mcrwPayload.Length / sizeof(int);
                mcrwCounts[chunkY, chunkX] = count;
                for (int index = 0; index < count; index++)
                    mcrwIndices.Add(BinaryPrimitives.ReadInt32LittleEndian(mcrwPayload.AsSpan(index * sizeof(int), sizeof(int))));
                anyMcrw = true;
            }
        }

        if (anyMcrd)
            signals.Add("mcrd_ref_indices");
        if (anyMcrw)
            signals.Add("mcrw_ref_indices");

        return (
            anyMcrd ? mcrdCounts : null,
            anyMcrd ? mcrdIndices.ToArray() : null,
            anyMcrw ? mcrwCounts : null,
            anyMcrw ? mcrwIndices.ToArray() : null);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // WL* loose liquid files
    // ═══════════════════════════════════════════════════════════════════════

    private const float WlTileSize = 533.333f;
    private const float WlMapSize = 17066.666f;

    private static (float[,]? mask, float[,]? height)
        ReadWlFiles(string adtPath, HashSet<string> signals)
    {
        string? mapDir = Path.GetDirectoryName(adtPath);
        if (string.IsNullOrEmpty(mapDir))
            return (null, null);

        if (!TryParseAdtTileCoords(adtPath, out int targetTileX, out int targetTileY))
            return (null, null);

        // Scan ALL WL files in the map directory (MdxViewer pattern)
        string[] wlFiles;
        try
        {
            wlFiles = Directory.GetFiles(mapDir, "*.wlw", SearchOption.TopDirectoryOnly)
                .Concat(Directory.GetFiles(mapDir, "*.wlm", SearchOption.TopDirectoryOnly))
                .Concat(Directory.GetFiles(mapDir, "*.wlq", SearchOption.TopDirectoryOnly))
                .Concat(Directory.GetFiles(mapDir, "*.wll", SearchOption.TopDirectoryOnly))
                .ToArray();
        }
        catch
        {
            return (null, null);
        }

        if (wlFiles.Length == 0)
            return (null, null);

        float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] heights = new float[TileHeightmapSize, TileHeightmapSize];
        bool any = false;

        // Each WL block is ~33.33m on a 533.33m tile = ~16 pixels on 257 grid
        // Block local coords are in [0, WlTileSize) range within the tile
        float blockWorldSize = WlTileSize / 16f; // ~33.33m per block
        float pixelsPerBlock = (TileHeightmapSize - 1) / 16f; // ~16 pixels per block

        foreach (string wlPath in wlFiles)
        {
            WlFile wl;
            try
            {
                wl = WlFileReader.Read(wlPath);
            }
            catch
            {
                continue;
            }

            foreach (WlBlock block in wl.Blocks)
            {
                Vector3 pos = block.WorldPosition;
                int tileX = Math.Clamp((int)Math.Floor((WlMapSize - pos.Y) / WlTileSize), 0, 63);
                int tileY = Math.Clamp((int)Math.Floor((WlMapSize - pos.X) / WlTileSize), 0, 63);

                if (tileX != targetTileX || tileY != targetTileY)
                    continue;

                // Get per-vertex heights in standard row-major order
                float[] vertexHeights = block.GetHeights4x4();

                // Block origin in tile-local coordinates (meters from tile corner)
                float blockLocalX = (WlMapSize - pos.Y) - (tileX * WlTileSize);
                float blockLocalY = (WlMapSize - pos.X) - (tileY * WlTileSize);

                // Map each of the 16 vertices to the 257x257 grid
                for (int vi = 0; vi < 4; vi++)
                {
                    for (int vj = 0; vj < 4; vj++)
                    {
                        float vh = vertexHeights[vi * 4 + vj];

                        // Vertex position in tile-local meters
                        float vx = blockLocalX + vj * blockWorldSize;
                        float vy = blockLocalY + vi * blockWorldSize;

                        // Convert to pixel coordinate
                        float px = vx / WlTileSize * (TileHeightmapSize - 1);
                        float py = vy / WlTileSize * (TileHeightmapSize - 1);

                        // Write to nearest pixel with 2-pixel radius for blending
                        int ix = Math.Clamp((int)Math.Round(px), 0, TileHeightmapSize - 1);
                        int iy = Math.Clamp((int)Math.Round(py), 0, TileHeightmapSize - 1);

                        for (int dy = -2; dy <= 2; dy++)
                        {
                            for (int dx = -2; dx <= 2; dx++)
                            {
                                int x = Math.Clamp(ix + dx, 0, TileHeightmapSize - 1);
                                int y = Math.Clamp(iy + dy, 0, TileHeightmapSize - 1);
                                float dist = MathF.Sqrt(dx * dx + dy * dy);
                                if (dist > 2.5f)
                                    continue;
                                float w = 1.0f / (1.0f + dist);
                                if (w > mask[y, x])
                                {
                                    mask[y, x] = w;
                                    heights[y, x] = vh;
                                }
                            }
                        }
                        any = true;
                    }
                }
            }
        }

        if (!any)
            return (null, null);

        // Normalize mask to [0, 1]
        float maxMask = 0f;
        for (int y = 0; y < TileHeightmapSize; y++)
            for (int x = 0; x < TileHeightmapSize; x++)
                if (mask[y, x] > maxMask)
                    maxMask = mask[y, x];
        if (maxMask > 0f)
            for (int y = 0; y < TileHeightmapSize; y++)
                for (int x = 0; x < TileHeightmapSize; x++)
                    mask[y, x] = MathF.Min(mask[y, x] / maxMask, 1.0f);

        signals.Add("wl_liquid_mask");
        signals.Add("wl_liquid_height");
        return (mask, heights);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Object footprint masks (MDDF/MODF → 257×257)
    // ═══════════════════════════════════════════════════════════════════════

    private const float ObjectWorldTileSize = 533.33333f;
    private const float ObjectMapOrigin = 17066.666f;

    private static AdtPlacementCatalog? TryReadPlacementCatalog(
        string adtPath,
        Stream stream,
        MapFileSummary fileSummary,
        string? placementSourcePathOverride = null,
        byte[]? placementBytesOverride = null)
    {
        try
        {
            if (placementBytesOverride is not null && !string.IsNullOrWhiteSpace(placementSourcePathOverride))
            {
                using MemoryStream placementStream = new(placementBytesOverride, writable: false);
                MapFileSummary placementSummary = MapFileSummaryReader.Read(placementStream, placementSourcePathOverride);
                return AdtPlacementReader.Read(placementStream, placementSummary);
            }

            AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
            string? placementSourcePath = placementSourcePathOverride ?? family.PlacementSourcePath;
            return !string.IsNullOrWhiteSpace(placementSourcePath) && File.Exists(placementSourcePath)
                ? AdtPlacementReader.Read(placementSourcePath)
                : AdtPlacementReader.Read(stream, fileSummary);
        }
        catch
        {
            return null;
        }
    }

    private static readonly System.Text.RegularExpressions.Regex ExcludeDoodadRegex = new(
        @"^(Tree|Bush|Flower|Plant|Vine|Fern|Mushroom|Herb|Ivy|Reed|Cattress|Lilypad|Kelp|Seaweed|Coral)",
        System.Text.RegularExpressions.RegexOptions.IgnoreCase | System.Text.RegularExpressions.RegexOptions.Compiled);

    private const float DoodadHeightThreshold = 50f;

    private static (float[,]? mask, float[,]? preciseMask, int[,]? instanceMask, float[,]? mddfMask, float[,]? modfMask, float[,]? filteredMask)
        BuildObjectMasks(
            string adtPath,
            Stream stream,
            MapFileSummary fileSummary,
            HashSet<string> signals,
            string? placementSourcePathOverride = null,
            byte[]? placementBytesOverride = null,
            AdtPlacementCatalog? placementsOverride = null)
    {
        if (!TryParseAdtTileCoords(fileSummary.SourcePath, out int tileX, out int tileY))
            return (null, null, null, null, null, null);

        AdtPlacementCatalog? placements = placementsOverride
            ?? TryReadPlacementCatalog(adtPath, stream, fileSummary, placementSourcePathOverride, placementBytesOverride);
        if (placements is null)
            return (null, null, null, null, null, null);

        if (placements.ModelPlacements.Count == 0 && placements.WorldModelPlacements.Count == 0)
            return (null, null, null, null, null, null);

        float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] preciseMask = new float[TileHeightmapSize, TileHeightmapSize];
        int[,] instanceMask = new int[TileHeightmapSize, TileHeightmapSize];
        float[,] mddfMask = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] modfMask = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] filteredMask = new float[TileHeightmapSize, TileHeightmapSize];
        int instanceId = 1;

        // Compute tile base height from heightmap for height gate
        float tileBaseHeight = 0f;

        foreach (AdtModelPlacement placement in placements.ModelPlacements)
        {
            if (!TryProjectPlacementToTilePixel(placement.Position, tileX, tileY, out int px, out int py))
                continue;

            float radiusBinary = 2f;
            float radiusPrecise = MathF.Max(1.5f, placement.Scale * 2f);

            PaintCircle(mask, px, py, radiusBinary, value: 1.0f);
            PaintSoftCircle(preciseMask, px, py, radiusPrecise);
            PaintCircle(instanceMask, px, py, radiusBinary, value: instanceId);
            PaintCircle(mddfMask, px, py, radiusBinary, value: 1.0f);

            // Height gate: exclude objects whose estimated top exceeds tile base + threshold
            float estimatedTop = placement.Position.Y + placement.Scale * 10f;
            bool exceedsHeight = estimatedTop > tileBaseHeight + DoodadHeightThreshold;

            // Regex gate: exclude decorative doodads
            string assetName = Path.GetFileNameWithoutExtension(placement.ModelPath);
            bool matchesExclusion = ExcludeDoodadRegex.IsMatch(assetName);

            if (!exceedsHeight && !matchesExclusion)
                PaintCircle(filteredMask, px, py, radiusBinary, value: 1.0f);

            instanceId++;
        }

        foreach (AdtWorldModelPlacement placement in placements.WorldModelPlacements)
        {
            if (!TryProjectPlacementToTilePixel(placement.Position, tileX, tileY, out int px, out int py))
                continue;

            Vector3 min = placement.BoundsMin;
            Vector3 max = placement.BoundsMax;
            bool hasValidBounds =
                min.X < max.X && min.Y < max.Y &&
                !float.IsNaN(min.X) && !float.IsNaN(max.X);

            if (hasValidBounds)
            {
                ProjectBoundsToTilePixels(min, max, tileX, tileY,
                    out int minPx, out int minPy, out int maxPx, out int maxPy);

                PaintRect(mask, minPx, minPy, maxPx, maxPy, value: 1.0f);
                PaintSoftRect(preciseMask, minPx, minPy, maxPx, maxPy);
                PaintRect(instanceMask, minPx, minPy, maxPx, maxPy, value: instanceId);
                PaintRect(modfMask, minPx, minPy, maxPx, maxPy, value: 1.0f);
                // WMOs always included in filtered mask
                PaintRect(filteredMask, minPx, minPy, maxPx, maxPy, value: 1.0f);
            }
            else
            {
                PaintCircle(mask, px, py, radius: 3f, value: 1.0f);
                PaintSoftCircle(preciseMask, px, py, radius: 3f);
                PaintCircle(instanceMask, px, py, radius: 3f, value: instanceId);
                PaintCircle(modfMask, px, py, radius: 3f, value: 1.0f);
                PaintCircle(filteredMask, px, py, radius: 3f, value: 1.0f);
            }
            instanceId++;
        }

        signals.Add("object_mask_257");
        signals.Add("object_precise_mask_257");
        signals.Add("object_instance_mask_257");
        signals.Add("mddf_mask_257");
        signals.Add("modf_mask_257");
        signals.Add("object_filtered_mask_257");
        return (mask, preciseMask, instanceMask, mddfMask, modfMask, filteredMask);
    }

    private static bool TryProjectPlacementToTilePixel(Vector3 position, int tileX, int tileY, out int pixelX, out int pixelY)
    {
        pixelX = 0;
        pixelY = 0;

        (float U, float V)[] candidates =
        [
            ((position.X / ObjectWorldTileSize) - tileX, (position.Z / ObjectWorldTileSize) - tileY),
            (((ObjectMapOrigin - position.Z) / ObjectWorldTileSize) - tileX, ((ObjectMapOrigin - position.X) / ObjectWorldTileSize) - tileY),
            ((position.X / ObjectWorldTileSize) - tileX, (position.Y / ObjectWorldTileSize) - tileY),
            (((ObjectMapOrigin - position.Y) / ObjectWorldTileSize) - tileX, ((ObjectMapOrigin - position.X) / ObjectWorldTileSize) - tileY),
        ];

        float bestScore = float.MinValue;
        (float U, float V) best = default;
        bool found = false;
        foreach ((float U, float V) candidate in candidates)
        {
            if (candidate.U < -0.25f || candidate.U > 1.25f || candidate.V < -0.25f || candidate.V > 1.25f)
                continue;

            float distanceToCenter = MathF.Abs(candidate.U - 0.5f) + MathF.Abs(candidate.V - 0.5f);
            float score = -distanceToCenter;
            if (score > bestScore)
            {
                bestScore = score;
                best = candidate;
                found = true;
            }
        }

        if (!found)
            return false;

        pixelX = Math.Clamp((int)MathF.Round(best.U * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
        pixelY = Math.Clamp((int)MathF.Round(best.V * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
        return true;
    }

    private static void ProjectBoundsToTilePixels(Vector3 min, Vector3 max, int tileX, int tileY,
        out int minPx, out int minPy, out int maxPx, out int maxPy)
    {
        minPx = int.MaxValue;
        minPy = int.MaxValue;
        maxPx = int.MinValue;
        maxPy = int.MinValue;

        // Project all 4 corners of the bounds rectangle to find the pixel AABB
        ReadOnlySpan<Vector3> corners =
        [
            new Vector3(min.X, min.Y, min.Z),
            new Vector3(min.X, max.Y, min.Z),
            new Vector3(max.X, min.Y, min.Z),
            new Vector3(max.X, max.Y, min.Z),
            new Vector3(min.X, min.Y, max.Z),
            new Vector3(min.X, max.Y, max.Z),
            new Vector3(max.X, min.Y, max.Z),
            new Vector3(max.X, max.Y, max.Z),
        ];

        foreach (Vector3 corner in corners)
        {
            if (TryProjectPlacementToTilePixel(corner, tileX, tileY, out int px, out int py))
            {
                minPx = Math.Min(minPx, px);
                minPy = Math.Min(minPy, py);
                maxPx = Math.Max(maxPx, px);
                maxPy = Math.Max(maxPy, py);
            }
        }

        if (minPx == int.MaxValue)
        {
            minPx = 0; minPy = 0; maxPx = 0; maxPy = 0;
        }
    }

    private static void PaintCircle(float[,] buffer, int cx, int cy, float radius, float value)
    {
        int r = (int)MathF.Ceiling(radius);
        int rSq = r * r;
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                if ((dx * dx) + (dy * dy) > rSq)
                    continue;

                int x = cx + dx;
                int y = cy + dy;
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;

                buffer[y, x] = value;
            }
        }
    }

    private static void PaintCircle(int[,] buffer, int cx, int cy, float radius, int value)
    {
        int r = (int)MathF.Ceiling(radius);
        int rSq = r * r;
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                if ((dx * dx) + (dy * dy) > rSq)
                    continue;

                int x = cx + dx;
                int y = cy + dy;
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;

                buffer[y, x] = value;
            }
        }
    }

    private static void PaintSoftCircle(float[,] buffer, int cx, int cy, float radius)
    {
        int r = (int)MathF.Ceiling(radius * 1.5f);
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                float dist = MathF.Sqrt((dx * dx) + (dy * dy));
                if (dist > radius * 1.5f)
                    continue;

                float alpha = 1f - MathF.Min(1f, dist / radius);
                if (alpha <= 0f)
                    continue;

                int x = cx + dx;
                int y = cy + dy;
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;

                buffer[y, x] = Math.Max(buffer[y, x], alpha);
            }
        }
    }

    private static void PaintRect(float[,] buffer, int minX, int minY, int maxX, int maxY, float value)
    {
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;
                buffer[y, x] = value;
            }
        }
    }

    private static void PaintRect(int[,] buffer, int minX, int minY, int maxX, int maxY, int value)
    {
        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;
                buffer[y, x] = value;
            }
        }
    }

    private static void PaintSoftRect(float[,] buffer, int minX, int minY, int maxX, int maxY)
    {
        int pad = 2;
        for (int y = minY - pad; y <= maxY + pad; y++)
        {
            for (int x = minX - pad; x <= maxX + pad; x++)
            {
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;

                float dx = 0f;
                if (x < minX) dx = minX - x;
                else if (x > maxX) dx = x - maxX;

                float dy = 0f;
                if (y < minY) dy = minY - y;
                else if (y > maxY) dy = y - maxY;

                float dist = MathF.Sqrt((dx * dx) + (dy * dy));
                float alpha = 1f - MathF.Min(1f, dist / pad);

                buffer[y, x] = Math.Max(buffer[y, x], alpha);
            }
        }
    }

    private static bool TryParseAdtTileCoords(string adtPath, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;
        string fileName = Path.GetFileNameWithoutExtension(adtPath);
        int lastUnderscore = fileName.LastIndexOf('_');
        if (lastUnderscore < 0) return false;
        int secondLast = fileName.LastIndexOf('_', lastUnderscore - 1);
        if (secondLast < 0) return false;

        return int.TryParse(fileName.AsSpan(secondLast + 1, lastUnderscore - secondLast - 1), out tileX)
            && int.TryParse(fileName.AsSpan(lastUnderscore + 1), out tileY);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Downsampling helpers
    // ═══════════════════════════════════════════════════════════════════════

    private static float[,]? DownsampleHeightmap(float[,]? source, int targetSize)
    {
        if (source is null)
            return null;

        float[,] result = new float[targetSize, targetSize];
        int sourceSize = source.GetLength(0);
        float scale = (float)(sourceSize - 1) / (targetSize - 1);

        for (int y = 0; y < targetSize; y++)
        {
            for (int x = 0; x < targetSize; x++)
            {
                float sourceX = x * scale;
                float sourceY = y * scale;
                int ix = Math.Clamp((int)sourceX, 0, sourceSize - 2);
                int iy = Math.Clamp((int)sourceY, 0, sourceSize - 2);
                float fx = sourceX - ix;
                float fy = sourceY - iy;

                float v00 = source[iy, ix];
                float v10 = source[iy, ix + 1];
                float v01 = source[iy + 1, ix];
                float v11 = source[iy + 1, ix + 1];

                result[y, x] = BilinearInterpolate(v00, v10, v01, v11, fx, fy);
            }
        }

        return result;
    }

    private static float[,] ResizeScalarGrid(float[,] source, int targetSize)
    {
        float[,] result = new float[targetSize, targetSize];
        int sourceHeight = source.GetLength(0);
        int sourceWidth = source.GetLength(1);
        float scaleX = (float)(sourceWidth - 1) / Math.Max(1, targetSize - 1);
        float scaleY = (float)(sourceHeight - 1) / Math.Max(1, targetSize - 1);

        for (int y = 0; y < targetSize; y++)
        {
            for (int x = 0; x < targetSize; x++)
            {
                float sourceX = x * scaleX;
                float sourceY = y * scaleY;
                int ix = Math.Clamp((int)sourceX, 0, sourceWidth - 2);
                int iy = Math.Clamp((int)sourceY, 0, sourceHeight - 2);
                float fx = sourceX - ix;
                float fy = sourceY - iy;

                float v00 = source[iy, ix];
                float v10 = source[iy, ix + 1];
                float v01 = source[iy + 1, ix];
                float v11 = source[iy + 1, ix + 1];

                result[y, x] = BilinearInterpolate(v00, v10, v01, v11, fx, fy);
            }
        }

        return result;
    }

    private static float BilinearInterpolate(float v00, float v10, float v01, float v11, float fx, float fy)
    {
        float top = v00 + (v10 - v00) * fx;
        float bottom = v01 + (v11 - v01) * fx;
        return top + (bottom - top) * fy;
    }

    private static float[,,]? DownsampleAlpha256(float[,,]? alpha)
    {
        if (alpha is null)
            return null;

        int srcSize = alpha.GetLength(0);
        int channels = alpha.GetLength(2);
        const int DstSize = 256;

        float scale = (float)srcSize / DstSize;
        float[,,] result = new float[DstSize, DstSize, channels];

        for (int y = 0; y < DstSize; y++)
        {
            for (int x = 0; x < DstSize; x++)
            {
                int srcX0 = (int)(x * scale);
                int srcY0 = (int)(y * scale);
                int srcX1 = Math.Min(srcX0 + 1, srcSize - 1);
                int srcY1 = Math.Min(srcY0 + 1, srcSize - 1);
                float fx = (x * scale) - srcX0;
                float fy = (y * scale) - srcY0;

                for (int c = 0; c < channels; c++)
                {
                    float v00 = alpha[srcY0, srcX0, c];
                    float v10 = alpha[srcY0, srcX1, c];
                    float v01 = alpha[srcY1, srcX0, c];
                    float v11 = alpha[srcY1, srcX1, c];
                    result[y, x, c] = v00 + (v10 - v00) * fx + (v01 - v00) * fy + (v00 - v10 - v01 + v11) * fx * fy;
                }
            }
        }

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Coordinate mapping (copied from AdtTerrainWriter)
    // ═══════════════════════════════════════════════════════════════════════

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

    // ═══════════════════════════════════════════════════════════════════════
    // Chunk payload utilities
    // ═══════════════════════════════════════════════════════════════════════

    private static List<MapChunkLocation> ResolveTerrainChunkLocations(Stream stream, MapFileSummary fileSummary)
    {
        List<MapChunkLocation> topLevelChunks = fileSummary.Chunks
            .Where(static chunk => chunk.Id == MapChunkIds.Mcnk)
            .ToList();

        if (topLevelChunks.Count >= 256 || !fileSummary.HasChunk(MapChunkIds.Mcin))
            return topLevelChunks;

        MapChunkLocation mcinChunk = fileSummary.Chunks.First(chunk => chunk.Id == MapChunkIds.Mcin);
        byte[] mcinPayload = MapSummaryReaderCommon.ReadChunkPayload(stream, mcinChunk);
        if (mcinPayload.Length < 16)
            return topLevelChunks;

        List<MapChunkLocation> resolvedChunks = new(256);
        for (int index = 0; index < 256 && ((index + 1) * 16) <= mcinPayload.Length; index++)
        {
            int entryOffset = index * 16;
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
        uint headerMcalSize = payload.Length >= 0x2C ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x28, 4)) : 0;
        uint headerMcshSize = payload.Length >= 0x34 ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x30, 4)) : 0;

        int position = RootMcnkSubchunkOffset;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            long consumedSize = (long)header.Size;
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);
            else if (header.Id == AdtChunkIds.Mcal && headerMcalSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcalSize - ChunkHeader.SizeInBytes);
            else if (header.Id == AdtChunkIds.Mcsh && headerMcshSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcshSize - ChunkHeader.SizeInBytes);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcvt)
            {
                if (header.Size < McvtSampleCount * sizeof(float))
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = (int)nextOffset;
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

            long consumedSize = (long)header.Size;
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mcnr)
            {
                if (header.Size < McnrSampleByteCount)
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = (int)nextOffset;
        }

        return -1;
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

            long consumedSize = (long)header.Size;
            if (header.Id == AdtChunkIds.Mcnr)
                consumedSize = Math.Max(consumedSize, McnrConsumedSize);
            else if (header.Id == AdtChunkIds.Mcal && headerMcalSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcalSize - ChunkHeader.SizeInBytes);
            else if (header.Id == AdtChunkIds.Mcsh && headerMcshSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcshSize - ChunkHeader.SizeInBytes);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (nextOffset > payload.Length)
                break;

            if (header.Id == AdtChunkIds.Mccv)
            {
                if (header.Size < McvtSampleCount * 4)
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = (int)nextOffset;
        }

        return -1;
    }

private static int LocateMcnkSubchunkDataOffset(ReadOnlySpan<byte> payload, FourCC chunkId, int minimumPayloadSize)
    {
        uint headerMcshSize = payload.Length >= 0x34 ? BinaryPrimitives.ReadUInt32LittleEndian(payload.Slice(0x30, 4)) : 0;
        int position = RootMcnkSubchunkOffset;

        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            long declaredSize = (long)header.Size;
            long consumedSize = declaredSize;
            if (header.Id == AdtChunkIds.Mcsh && headerMcshSize >= ChunkHeader.SizeInBytes)
                consumedSize = Math.Max(consumedSize, (long)headerMcshSize - ChunkHeader.SizeInBytes);

            long nextOffset = (long)position + ChunkHeader.SizeInBytes + consumedSize;
            if (declaredSize < 0 || nextOffset > payload.Length)
                break;

            if (header.Id == chunkId && declaredSize >= minimumPayloadSize)
                return position + ChunkHeader.SizeInBytes;

            position = (int)nextOffset;
        }

        return -1;
    }

private static byte[]? TryReadSplitMcnkSubchunkPayload(ReadOnlySpan<byte> payload, FourCC chunkId)
    {
        int position = 0;
        while (position <= payload.Length - ChunkHeader.SizeInBytes)
        {
            if (!ChunkHeaderReader.TryRead(payload.Slice(position, ChunkHeader.SizeInBytes), out ChunkHeader header))
                break;

            long declaredSize = (long)header.Size;
            long nextOffset = (long)position + ChunkHeader.SizeInBytes + declaredSize;
            if (declaredSize < 0 || nextOffset > payload.Length)
                break;

            if (header.Id == chunkId && declaredSize <= int.MaxValue)
                return payload.Slice(position + ChunkHeader.SizeInBytes, (int)declaredSize).ToArray();

            position = (int)nextOffset;
        }

        return null;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PM4 path and building footprint masks
    // ═══════════════════════════════════════════════════════════════════════

    private static (float[,]? pathMask, float[,]? buildingFootprintMask, float[,]? mprlMask)
        BuildPm4Masks(string adtPath, HashSet<string> signals)
    {
        if (!AdtPm4MaskBuilder.TryBuild(adtPath, out float[,]? pathMask, out float[,]? buildingMask, out float[,]? mprlMask))
            return (null, null, null);

        if (pathMask is not null)
            signals.Add("pm4_path_mask");
        if (buildingMask is not null)
            signals.Add("pm4_building_footprint_mask");
        if (mprlMask is not null)
            signals.Add("pm4_mprl_mask");

        return (pathMask, buildingMask, mprlMask);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Unified liquid mask and height
    // ═══════════════════════════════════════════════════════════════════════

    internal static (float[,]? mask, float[,]? height)
        BuildUnifiedLiquid(
            float[,]? mh2oHeight,
            bool[,]? mh2oPresence,
            float[,]? mclqHeight,
            bool[,]? mclqPresence,
            float[,]? wlMask,
            float[,]? wlHeight,
            HashSet<string> signals)
    {
        // Priority: MH2O > MCLQ > WL*
        // MH2O is the richest source (WotLK+) with per-vertex heights at 257×257.
        if (mh2oHeight is not null && mh2oPresence is not null)
        {
            float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
            float[,] height = new float[TileHeightmapSize, TileHeightmapSize];
            bool any = false;

            for (int y = 0; y < TileHeightmapSize; y++)
            {
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    if (!mh2oPresence[y, x])
                        continue;

                    float h = mh2oHeight[y, x];
                    mask[y, x] = 1.0f;
                    height[y, x] = h;
                    any = true;
                }
            }

            if (any)
            {
                signals.Add("unified_liquid_mask");
                signals.Add("unified_liquid_height");
                return (mask, height);
            }
        }

        // MCLQ is pre-WotLK, stored at 129×129 resolution.
        if (mclqHeight is not null && mclqPresence is not null)
        {
            int mclqSize = mclqHeight.GetLength(0);
            float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
            float[,] height = new float[TileHeightmapSize, TileHeightmapSize];
            bool any = false;

            // Upsample 129×129 → 257×257 via bilinear
            float scale = (float)(mclqSize - 1) / (TileHeightmapSize - 1);
            for (int y = 0; y < TileHeightmapSize; y++)
            {
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    float sourceX = x * scale;
                    float sourceY = y * scale;
                    int ix = Math.Clamp((int)sourceX, 0, mclqSize - 2);
                    int iy = Math.Clamp((int)sourceY, 0, mclqSize - 2);
                    float fx = sourceX - ix;
                    float fy = sourceY - iy;

                    if (!mclqPresence[iy, ix]
                        && !mclqPresence[iy, ix + 1]
                        && !mclqPresence[iy + 1, ix]
                        && !mclqPresence[iy + 1, ix + 1])
                    {
                        continue;
                    }

                    float h = BilinearInterpolate(
                        mclqHeight[iy, ix], mclqHeight[iy, ix + 1],
                        mclqHeight[iy + 1, ix], mclqHeight[iy + 1, ix + 1],
                        fx, fy);

                    mask[y, x] = 1.0f;
                    height[y, x] = h;
                    any = true;
                }
            }

            if (any)
            {
                signals.Add("unified_liquid_mask");
                signals.Add("unified_liquid_height");
                return (mask, height);
            }
        }

        // WL* loose files are the last resort.
        if (wlMask is not null)
        {
            float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
            float[,] height = new float[TileHeightmapSize, TileHeightmapSize];
            bool any = false;

            for (int y = 0; y < TileHeightmapSize; y++)
            {
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    if (wlMask[y, x] == 0f)
                        continue;

                    mask[y, x] = 1.0f;
                    height[y, x] = wlHeight?[y, x] ?? 0f;
                    any = true;
                }
            }

            if (any)
            {
                signals.Add("unified_liquid_mask");
                signals.Add("unified_liquid_height");
                return (mask, height);
            }
        }

        return (null, null);
    }

    private static string ExtractMapName(string adtPath)
    {
        // Path pattern: .../World/Maps/<MapName>/<MapName>_<x>_<y>.adt
        string? mapsDir = Path.GetDirectoryName(adtPath);
        if (mapsDir is null)
            return ExtractMapNameFromTileStem(adtPath);

        string? mapName = Path.GetFileName(mapsDir);
        if (!string.IsNullOrWhiteSpace(mapName)
            && !mapName.StartsWith("wowviewer_", StringComparison.OrdinalIgnoreCase)
            && !mapName.StartsWith("tmp", StringComparison.OrdinalIgnoreCase))
        {
            return mapName;
        }

        return ExtractMapNameFromTileStem(adtPath);
    }

    private static string ExtractMapNameFromTileStem(string adtPath)
    {
        string stem = Path.GetFileNameWithoutExtension(adtPath);
        string[] parts = stem.Split('_');
        if (parts.Length >= 3
            && int.TryParse(parts[^1], out _)
            && int.TryParse(parts[^2], out _))
        {
            return string.Join("_", parts[..^2]);
        }

        return string.Empty;
    }

    private static (int mddfCount, int modfCount, float[,]? mddfData, float[,]? modfData, IReadOnlyList<string> mddfNames, IReadOnlyList<string> modfNames)
        ExtractPlacementArrays(
            string adtPath,
            Stream stream,
            MapFileSummary fileSummary,
            string? placementSourcePathOverride = null,
            byte[]? placementBytesOverride = null,
            AdtPlacementCatalog? placementsOverride = null)
    {
        try
        {
            AdtPlacementCatalog? placements = placementsOverride
                ?? TryReadPlacementCatalog(adtPath, stream, fileSummary, placementSourcePathOverride, placementBytesOverride);
            if (placements is null)
                return (0, 0, null, null, Array.Empty<string>(), Array.Empty<string>());

            float[,]? mddfData = null;
            List<string> mddfNames = [];
            if (placements.ModelPlacements.Count > 0)
            {
                mddfData = new float[placements.ModelPlacements.Count, 9];
                for (int i = 0; i < placements.ModelPlacements.Count; i++)
                {
                    var p = placements.ModelPlacements[i];
                    mddfData[i, 0] = p.NameId; mddfData[i, 1] = p.UniqueId;
                    mddfData[i, 2] = p.Position.X; mddfData[i, 3] = p.Position.Y; mddfData[i, 4] = p.Position.Z;
                    mddfData[i, 5] = p.Rotation.X; mddfData[i, 6] = p.Rotation.Y; mddfData[i, 7] = p.Rotation.Z;
                    mddfData[i, 8] = p.Scale;
                    SetIndexedName(mddfNames, p.NameId, p.ModelPath);
                }
            }

            float[,]? modfData = null;
            List<string> modfNames = [];
            if (placements.WorldModelPlacements.Count > 0)
            {
                modfData = new float[placements.WorldModelPlacements.Count, 14];
                for (int i = 0; i < placements.WorldModelPlacements.Count; i++)
                {
                    var p = placements.WorldModelPlacements[i];
                    modfData[i, 0] = p.NameId; modfData[i, 1] = p.UniqueId;
                    modfData[i, 2] = p.Position.X; modfData[i, 3] = p.Position.Y; modfData[i, 4] = p.Position.Z;
                    modfData[i, 5] = p.Rotation.X; modfData[i, 6] = p.Rotation.Y; modfData[i, 7] = p.Rotation.Z;
                    modfData[i, 8] = p.BoundsMin.X; modfData[i, 9] = p.BoundsMin.Y; modfData[i, 10] = p.BoundsMin.Z;
                    modfData[i, 11] = p.BoundsMax.X; modfData[i, 12] = p.BoundsMax.Y; modfData[i, 13] = p.BoundsMax.Z;
                    SetIndexedName(modfNames, p.NameId, p.ModelPath);
                }
            }

            return (placements.ModelPlacements.Count, placements.WorldModelPlacements.Count, mddfData, modfData, mddfNames, modfNames);
        }
        catch
        {
            return (0, 0, null, null, Array.Empty<string>(), Array.Empty<string>());
        }
    }

    private static void SetIndexedName(List<string> names, int index, string path)
    {
        if (index < 0)
            return;

        while (names.Count <= index)
            names.Add(string.Empty);

        if (names[index].Length == 0)
            names[index] = path;
    }
}
