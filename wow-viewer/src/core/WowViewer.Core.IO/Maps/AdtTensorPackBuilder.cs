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

        AdtFormatProfile profile = AdtFormatProfiles.Resolve(buildVersion);

        using FileStream stream = File.OpenRead(adtPath);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(adtPath));
        if (fileSummary.Kind != MapFileKind.Adt)
            throw new InvalidDataException($"Tensor pack builder requires a root ADT file, but found {fileSummary.Kind}.");

        string tileName = Path.GetFileNameWithoutExtension(adtPath);
        HashSet<string> availableSignals = [];

        // ── Resolve terrain chunks ───────────────────────────────────────────
        List<MapChunkLocation> terrainChunks = ResolveTerrainChunkLocations(stream, fileSummary);

        // ── Assemble heightmap (MCVT) ───────────────────────────────────────
        float[,]? height257 = AssembleHeightmap(stream, terrainChunks, availableSignals);

        // ── Assemble normals (MCNR) ─────────────────────────────────────────
        float[,,]? mcnrNormalXyz = AssembleNormals(stream, terrainChunks, availableSignals);

        // ── Assemble vertex colors (MCCV) ────────────────────────────────────
        float[,,]? mccvRgb = AssembleMccv(stream, terrainChunks, availableSignals);

        // ── Read texture data (MCLY + MCAL) ──────────────────────────────────
        (int[,,]? mclyTextureIds, IReadOnlyList<string> mclyTextureNames, bool[,,]? mclyLayerMask, float[,,]? mcalAlphaPack, float[,]? mcshShadowMask256) =
            ReadTextureData(adtPath, textureSourcePath, profile, availableSignals);

        // ── Read MH2O liquid ─────────────────────────────────────────────────
        (float[,]? mh2oHeight, float[,]? mh2oDepth, int[,]? mh2oType) =
            ReadMh2o(stream, fileSummary, availableSignals);

        // ── Read MTXF texture flags ──────────────────────────────────────────
        (int[,]? mtxfAnimated, int[,]? mtxfTransform) =
            ReadMtxf(stream, fileSummary, mclyTextureIds, availableSignals);

        // ── Read MCLQ legacy liquid ──────────────────────────────────────────
        (float[,]? mclqHeight, int[,]? mclqType) =
            ReadMclq(stream, terrainChunks, availableSignals);

        // ── Read MCRF object references ──────────────────────────────────────
        (bool[,]? holeMask, int[,]? objectMask16) =
            ReadMcrfAndHoles(stream, terrainChunks, availableSignals);

        // ── Read WL* loose liquid files ──────────────────────────────────────
        (float[,]? wlMask, float[,]? wlHeight) =
            ReadWlFiles(adtPath, availableSignals);

        // ── Build object footprint masks from MDDF/MODF ──────────────────────
        (float[,]? objectMask257, float[,]? objectPreciseMask257) =
            BuildObjectMasks(adtPath, stream, fileSummary, availableSignals);

        float[,]? shadowResidualMask256 = BuildShadowResidualMask256(mcshShadowMask256, objectPreciseMask257, availableSignals);

        // ── Build PM4 path, building footprint, and MPRL portal masks ────────
        (float[,]? pm4PathMask, float[,]? pm4BuildingFootprintMask, float[,]? pm4MprlMask) =
            BuildPm4Masks(adtPath, availableSignals);

        // ── Build unified liquid mask and height ─────────────────────────────
        (float[,]? unifiedLiquidMask, float[,]? unifiedLiquidHeight) =
            BuildUnifiedLiquid(mh2oHeight, mclqHeight, wlMask, wlHeight, availableSignals);

        // ── Compute downsampled heights ──────────────────────────────────────
        float[,]? height65 = DownsampleHeightmap(height257, 65);
        float[,]? height17 = DownsampleHeightmap(height257, 17);

        (int mddfCount, int modfCount, float[,]? mddfData, float[,]? modfData, IReadOnlyList<string> mddfNames, IReadOnlyList<string> modfNames) =
            ExtractPlacementArrays(adtPath, stream, fileSummary);

        return new TerrainTileTensorPack
        {
            TileName = tileName,
            MapName = ExtractMapName(adtPath),
            BuildKey = buildVersion ?? string.Empty,
            SourceAdtPath = adtPath,
            Height257 = height257,
            Height65 = height65,
            Height17 = height17,
            MclyTextureIds = mclyTextureIds,
            MclyTextureNames = mclyTextureNames,
            MclyLayerMask = mclyLayerMask,
            McalAlphaPack256 = mcalAlphaPack,
            MccvRgb = mccvRgb,
            McnrNormalXyz = mcnrNormalXyz,
            Mh2oSurfaceHeight = mh2oHeight,
            Mh2oDepth = mh2oDepth,
            Mh2oTypeMask = mh2oType,
            MclqSurfaceHeight = mclqHeight,
            MclqTypeMask = mclqType,
            MtxfAnimatedMask = mtxfAnimated,
            MtxfTransformId = mtxfTransform,
            HoleMask16 = holeMask,
            WlLiquidMask = wlMask,
            WlLiquidHeight = wlHeight,
            UnifiedLiquidMask = unifiedLiquidMask,
            UnifiedLiquidHeight = unifiedLiquidHeight,
            ObjectMask257 = objectMask257,
            ObjectPreciseMask257 = objectPreciseMask257,
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
            Height257 = null,
            Height65 = null,
            Height17 = null,
            MclyTextureIds = null,
            MclyTextureNames = Array.Empty<string>(),
            MclyLayerMask = null,
            McalAlphaPack256 = null,
            MccvRgb = null,
            McnrNormalXyz = null,
            Mh2oSurfaceHeight = null,
            Mh2oDepth = null,
            Mh2oTypeMask = null,
            MclqSurfaceHeight = null,
            MclqTypeMask = null,
            MtxfAnimatedMask = null,
            MtxfTransformId = null,
            HoleMask16 = null,
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

    private static (int[,,]? textureIds, IReadOnlyList<string> textureNames, bool[,,]? layerMask, float[,,]? alphaPack, float[,]? mcshShadowMask256)
        ReadTextureData(string adtPath, string? textureSourcePath, AdtFormatProfile profile, HashSet<string> signals)
    {
        string? effectiveTexturePath = textureSourcePath;
        if (string.IsNullOrWhiteSpace(effectiveTexturePath))
        {
            AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
            effectiveTexturePath = family.TextureSourcePath;
        }

        if (string.IsNullOrWhiteSpace(effectiveTexturePath) || !File.Exists(effectiveTexturePath))
            return (null, Array.Empty<string>(), null, null, null);

        try
        {
            AdtTextureFile textureFile = AdtTextureReader.Read(effectiveTexturePath, profile.DecodeProfile);
            if (textureFile.Chunks.Count == 0)
                return (null, textureFile.TextureNames, null, null, null);

            int[,,] textureIds = new int[TileChunks, TileChunks, 4];
            bool[,,] layerMask = new bool[TileChunks, TileChunks, 4];
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
                return (null, textureFile.TextureNames, null, null, null);

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
            if (textureFile.TextureNames.Count > 0)
                signals.Add("mcly_texture_names");
            if (anyShadow)
                signals.Add("mcsh_shadow_mask_256");
            return (textureIds, textureFile.TextureNames, layerMask, alphaPack, anyShadow ? mcshShadowMask256 : null);
        }
        catch
        {
            return (null, Array.Empty<string>(), null, null, null);
        }
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

    private static (float[,]? height, float[,]? depth, int[,]? typeMask)
        ReadMh2o(Stream stream, MapFileSummary fileSummary, HashSet<string> signals)
    {
        try
        {
            AdtLiquidFile liquidFile = AdtLiquidReader.Read(stream, fileSummary);
            if (liquidFile.Chunks.Count == 0)
                return (null, null, null);

            float[,] heights = new float[TileHeightmapSize, TileHeightmapSize];
            float[,] depths = new float[TileHeightmapSize, TileHeightmapSize];
            int[,] typeMask = new int[TileHeightmapSize, TileHeightmapSize];
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

                            if (layer.Depths is not null && vertexIndex < layer.Depths.Length)
                                depths[globalY, globalX] = layer.Depths[vertexIndex];
                        }
                    }
                }
            }

            if (!any)
                return (null, null, null);

            signals.Add("mh2o_surface_height");
            signals.Add("mh2o_depth");
            signals.Add("mh2o_type_mask");
            return (heights, depths, typeMask);
        }
        catch
        {
            return (null, null, null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MCLQ legacy liquid
    // ═══════════════════════════════════════════════════════════════════════

    private static (float[,]? height, int[,]? typeMask)
        ReadMclq(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return (null, null);

        const int mclqVertsPerChunk = 9;
        const int mclqCellsPerChunk = 8;
        const int gridSize = (mclqCellsPerChunk * TileChunks) + 1; // 129

        float[,] heights = new float[gridSize, gridSize];
        int[,] types = new int[gridSize, gridSize];
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

            int mclqOffset = AdtMclqReader.LocateMclqOffset(payload);
            if (mclqOffset < 0)
                continue;

            int mclqSize = payload.Length - mclqOffset;
            byte[] mclqPayload = payload.AsSpan(mclqOffset, mclqSize).ToArray();
            AdtMclqData? mclq = AdtMclqReader.Read(mclqPayload);
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
                    }
                }
            }
        }

        if (!any)
            return (null, null);

        signals.Add("mclq_surface_height");
        signals.Add("mclq_type_mask");
        return (heights, types);
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

    private static (bool[,]? holeMask, int[,]? objectMask16)
        ReadMcrfAndHoles(Stream stream, List<MapChunkLocation> chunks, HashSet<string> signals)
    {
        if (chunks.Count == 0)
            return (null, null);

        bool[,] holes = new bool[TileChunks, TileChunks];
        int[,] objects = new int[TileChunks, TileChunks];
        bool anyHoles = false;
        bool anyObjects = false;

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
            int mcrfOffset = AdtMcrfReader.LocateMcrfOffset(payload);
            if (mcrfOffset >= 0)
            {
                int mcrfSize = payload.Length - mcrfOffset;
                if (mcrfSize >= 4)
                {
                    objects[chunkY, chunkX] = mcrfSize / 4; // count of references
                    anyObjects = true;
                }
            }
        }

        if (anyHoles)
            signals.Add("hole_mask_16");
        if (anyObjects)
            signals.Add("object_mask_257"); // coarse 16x16 stored in int array; consumer may upsample

        return (anyHoles ? holes : null, anyObjects ? objects : null);
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

                float avgHeight = block.Vertices.Average(v => v.Z);

                // Map block center to 257x257 grid
                float localX = (WlMapSize - pos.Y) - (tileX * WlTileSize);
                float localY = (WlMapSize - pos.X) - (tileY * WlTileSize);
                int gx = Math.Clamp((int)(localX / WlTileSize * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
                int gy = Math.Clamp((int)(localY / WlTileSize * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);

                // Write to a small neighborhood
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int px = Math.Clamp(gx + dx, 0, TileHeightmapSize - 1);
                        int py = Math.Clamp(gy + dy, 0, TileHeightmapSize - 1);
                        mask[py, px] = 1.0f;
                        heights[py, px] = avgHeight;
                    }
                }
                any = true;
            }
        }

        if (!any)
            return (null, null);

        signals.Add("wl_liquid_mask");
        signals.Add("wl_liquid_height");
        return (mask, heights);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Object footprint masks (MDDF/MODF → 257×257)
    // ═══════════════════════════════════════════════════════════════════════

    private const float ObjectWorldTileSize = 533.33333f;
    private const float ObjectMapOrigin = 17066.666f;

    private static (float[,]? mask, float[,]? preciseMask)
        BuildObjectMasks(string adtPath, Stream stream, MapFileSummary fileSummary, HashSet<string> signals)
    {
        if (!TryParseAdtTileCoords(fileSummary.SourcePath, out int tileX, out int tileY))
            return (null, null);

        AdtPlacementCatalog placements;
        try
        {
            AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
            string? placementSourcePath = family.PlacementSourcePath;
            placements = !string.IsNullOrWhiteSpace(placementSourcePath) && File.Exists(placementSourcePath)
                ? AdtPlacementReader.Read(placementSourcePath)
                : AdtPlacementReader.Read(stream, fileSummary);
        }
        catch
        {
            return (null, null);
        }

        if (placements.ModelPlacements.Count == 0 && placements.WorldModelPlacements.Count == 0)
            return (null, null);

        float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] preciseMask = new float[TileHeightmapSize, TileHeightmapSize];

        foreach (AdtModelPlacement placement in placements.ModelPlacements)
        {
            if (!TryProjectPlacementToTilePixel(placement.Position, tileX, tileY, out int px, out int py))
                continue;

            float radiusBinary = 2f;
            float radiusPrecise = MathF.Max(1.5f, placement.Scale * 2f);

            PaintCircle(mask, px, py, radiusBinary, value: 1.0f);
            PaintSoftCircle(preciseMask, px, py, radiusPrecise);
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
            }
            else
            {
                // Fallback to centroid circle when bounds are missing
                PaintCircle(mask, px, py, radius: 3f, value: 1.0f);
                PaintSoftCircle(preciseMask, px, py, radius: 3f);
            }
        }

        signals.Add("object_mask_257");
        signals.Add("object_precise_mask_257");
        return (mask, preciseMask);
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
                    return -1;
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
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = checked((int)nextOffset);
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
                if (header.Size < McvtSampleCount * 4)
                    return -1;
                return position + ChunkHeader.SizeInBytes;
            }

            position = checked((int)nextOffset);
        }

        return -1;
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

    private static (float[,]? mask, float[,]? height)
        BuildUnifiedLiquid(float[,]? mh2oHeight, float[,]? mclqHeight, float[,]? wlMask, float[,]? wlHeight, HashSet<string> signals)
    {
        // Priority: MH2O > MCLQ > WL*
        // MH2O is the richest source (WotLK+) with per-vertex heights at 257×257.
        if (mh2oHeight is not null)
        {
            float[,] mask = new float[TileHeightmapSize, TileHeightmapSize];
            float[,] height = new float[TileHeightmapSize, TileHeightmapSize];
            bool any = false;

            for (int y = 0; y < TileHeightmapSize; y++)
            {
                for (int x = 0; x < TileHeightmapSize; x++)
                {
                    float h = mh2oHeight[y, x];
                    if (h == 0f)
                        continue;

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
        if (mclqHeight is not null)
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

                    float h = BilinearInterpolate(
                        mclqHeight[iy, ix], mclqHeight[iy, ix + 1],
                        mclqHeight[iy + 1, ix], mclqHeight[iy + 1, ix + 1],
                        fx, fy);

                    if (h == 0f)
                        continue;

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
            return string.Empty;

        string? mapName = Path.GetFileName(mapsDir);
        return mapName ?? string.Empty;
    }

    private static (int mddfCount, int modfCount, float[,]? mddfData, float[,]? modfData, IReadOnlyList<string> mddfNames, IReadOnlyList<string> modfNames)
        ExtractPlacementArrays(string adtPath, Stream stream, MapFileSummary fileSummary)
    {
        try
        {
            AdtTileFamily family = AdtTileFamilyResolver.Resolve(adtPath);
            string? sourcePath = family.PlacementSourcePath;
            var placements = !string.IsNullOrWhiteSpace(sourcePath) && File.Exists(sourcePath)
                ? AdtPlacementReader.Read(sourcePath)
                : AdtPlacementReader.Read(stream, fileSummary);

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
                    mddfNames.Add(p.ModelPath);
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
                    modfNames.Add(p.ModelPath);
                }
            }

            return (placements.ModelPlacements.Count, placements.WorldModelPlacements.Count, mddfData, modfData, mddfNames, modfNames);
        }
        catch
        {
            return (0, 0, null, null, Array.Empty<string>(), Array.Empty<string>());
        }
    }
}
