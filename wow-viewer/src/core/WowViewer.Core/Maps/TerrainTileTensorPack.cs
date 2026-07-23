using System.Numerics;

namespace WowViewer.Core.Maps;

/// <summary>
/// A unified container for all extractable training signals from a single ADT tile.
/// Every field is nullable — signals that are missing from the source tile remain null.
/// This is the canonical data contract between wow-viewer library extraction and the v10 training pipeline.
/// </summary>
public sealed class TerrainTileTensorPack
{
    public string TileName { get; init; } = string.Empty;
    public string MapName { get; init; } = string.Empty;
    public string BuildKey { get; init; } = string.Empty;
    public string SourceAdtPath { get; init; } = string.Empty;
    public int? TileX { get; init; }
    public int? TileY { get; init; }

    // ── Height fields ──────────────────────────────────────────────────────

    /// <summary>
    /// 257×257 compatibility view assembled from MCVT. Mixed-parity positions
    /// are filled for raster consumers and are not real terrain vertices; use
    /// <see cref="TerrainVertices"/> for training and mesh reconstruction.
    /// </summary>
    public float[,]? Height257 { get; init; }

    /// <summary>
    /// Canonical raw MCVT terrain vertices in native per-chunk sample order.
    /// Unlike <see cref="Height257"/>, this contract never fills non-vertex
    /// positions and preserves legitimate zero-height samples.
    /// </summary>
    public TerrainVertexLattice? TerrainVertices { get; init; }

    public TerrainWdlLattice? WdlLattice { get; init; }

    /// <summary>65×65 downsampled height field (area-averaged from 257×257).</summary>
    public float[,]? Height65 { get; init; }

    /// <summary>17×17 coarse height field (area-averaged from 257×257).</summary>
    public float[,]? Height17 { get; init; }

    // ── Texture layer data ─────────────────────────────────────────────────

    /// <summary>
    /// 16×16 chunk grid × up to 4 layers.
    /// Each entry is the texture ID for that layer in that chunk.
    /// A value of -1 means the layer is not present in that chunk.
    /// </summary>
    public int[,,]? MclyTextureIds { get; init; }

    /// <summary>
    /// Tile-level MTEX texture table used by MCLY texture IDs.
    /// </summary>
    public IReadOnlyList<string> MclyTextureNames { get; init; } = Array.Empty<string>();

    /// <summary>Per-texture native-resolution RGB pixel arrays (name-aligned with MclyTextureNames).</summary>
    public IReadOnlyList<byte[,,]>? MclyTexturePixels { get; set; }

    /// <summary>
    /// Explicit RGB-proxy substitutions used by derived minimap/dataset consumers when a referenced
    /// MTEX BLP cannot decode. Original MTEX names and IDs are never rewritten.
    /// </summary>
    public IReadOnlyDictionary<int, TerrainTextureFallbackResolution> MinimapTextureFallbacks { get; set; }
        = new Dictionary<int, TerrainTextureFallbackResolution>();

    /// <summary>
    /// 16×16 chunk grid × 4 layers.
    /// Boolean flags: is this layer active in this chunk?
    /// </summary>
    public bool[,,]? MclyLayerMask { get; init; }

    /// <summary>
    /// 16×16 chunk grid × up to 4 layers.
    /// Texture material ids from MCMT in on-disk uint8 form.
    /// </summary>
    public byte[,,]? McmtMaterialIds { get; init; }

    /// <summary>
    /// Single-byte Cataclysm+ texture sizing parameter from MAMP.
    /// </summary>
    public byte[]? MampValue { get; init; }

    /// <summary>
    /// Full-resolution tile-level alpha pack at 1024×1024 (4 channels, 0-1 float).
    /// Assembled from per-chunk MCAL decode at 64×64 pixels per chunk.
    /// Channel 0 = base layer (always implied, may be all-ones).
    /// Channels 1-3 = blend weights for additional layers.
    /// </summary>
    public float[,,]? McalAlphaPack { get; init; }

    /// <summary>
    /// 256×256 downsampled alpha pack (4 channels, 0-1 float), derived from
    /// <see cref="McalAlphaPack"/> by averaging 4×4 blocks. Suitable for
    /// minimap-resolution compositing. Channel layout matches <see cref="McalAlphaPack"/>.
    /// </summary>
    public float[,,]? McalAlphaPack256 { get; init; }

    // ── Vertex colors and normals ──────────────────────────────────────────

    /// <summary>257×257 × 3 RGB vertex ambient colors (MCCV).</summary>
    public float[,,]? MccvRgb { get; init; }

    /// <summary>
    /// 257×257 × 4 per-vertex baked lighting bytes from MCLV in on-disk channel order.
    /// </summary>
    public byte[,,]? MclvLightingBytes { get; init; }

    /// <summary>
    /// 2 × 3 × 3 flight-bound planes from MFBO stored as signed heights widened to int32.
    /// Plane 0 = maximum, plane 1 = minimum.
    /// </summary>
    public int[,,]? MfboFlightBounds { get; init; }

    /// <summary>257×257 × 3 XYZ per-vertex normals (MCNR).</summary>
    public float[,,]? McnrNormalXyz { get; init; }

    /// <summary>257×257 bool mask: true where MCNR had original data (x%2 == y%2).</summary>
    public bool[,]? McnrMask257 { get; init; }

    // ── Liquid data ────────────────────────────────────────────────────────

    /// <summary>16×16 per-chunk MCNK header flags. Bits 2-5 indicate liquid type.</summary>
    public int[,]? McnkFlags16 { get; init; }

    /// <summary>257×257 liquid surface height from MH2O (WotLK+).</summary>
    public float[,]? Mh2oSurfaceHeight { get; init; }

    /// <summary>257×257 liquid depth from MH2O (WotLK+).</summary>
    public float[,]? Mh2oDepth { get; init; }

    /// <summary>257×257 liquid type mask from MH2O (WotLK+).</summary>
    public int[,]? Mh2oTypeMask { get; init; }

    /// <summary>257×257 boolean MH2O presence mask (true only where liquid is defined).</summary>
    public bool[,]? Mh2oPresenceMask { get; init; }

    /// <summary>257×257 liquid surface height from MCLQ (pre-WotLK).</summary>
    public float[,]? MclqSurfaceHeight { get; init; }

    /// <summary>257×257 liquid type mask from MCLQ (pre-WotLK).</summary>
    public int[,]? MclqTypeMask { get; init; }

    /// <summary>257×257 boolean MCLQ presence mask (true only where liquid is defined).</summary>
    public bool[,]? MclqPresenceMask { get; init; }

    /// <summary>257×257 liquid mask from WLW/WLM/WLQ/WLL loose files.</summary>
    public float[,]? WlLiquidMask { get; set; }

    /// <summary>257×257 liquid height from WLW/WLM/WLQ/WLL loose files.</summary>
    public float[,]? WlLiquidHeight { get; set; }

    /// <summary>
    /// 257×257 unified liquid mask from decoded MH2O, MCLQ, or WL* sources.
    /// MCNK flags are metadata only and never invent liquid coverage. Source
    /// precedence is MH2O, then MCLQ, then WL*; 1.0 marks decoded water.
    /// </summary>
    public float[,]? UnifiedLiquidMask { get; set; }

    /// <summary>
    /// 257×257 unified liquid surface height from decoded MH2O, MCLQ, or WL*
    /// sources. MCNK flags never supply a surface height; precedence matches
    /// <see cref="UnifiedLiquidMask"/>.
    /// </summary>
    public float[,]? UnifiedLiquidHeight { get; set; }

    /// <summary>
    /// 257×257 per-vertex resolved <see cref="AdtLiquidBasicType"/> (0=Water, 1=Ocean,
    /// 2=Magma, 3=Slime). Populated at harvest time using
    /// <c>McnkFlagDecoder</c> for MCNK/MCLQ-driven tiles and
    /// <c>DbcLiquidTypeTable</c> for MH2O-driven tiles. This is the canonical
    /// liquid type field consumed by the viewer at runtime — the viewer MUST
    /// NOT re-resolve liquid type from MCNK flags or MH2O LiquidTypeId at
    /// render time. 0xFF where no liquid is present.
    /// </summary>
    public byte[,]? LiquidBasicType257 { get; set; }

    // ── Object and footprint masks ─────────────────────────────────────────

    /// <summary>257×257 binary mask: any object footprint present.</summary>
    public float[,]? ObjectMask257 { get; init; }

    /// <summary>257×257 anti-aliased object silhouette.</summary>
    public float[,]? ObjectPreciseMask257 { get; init; }

    /// <summary>
    /// 257×257 strict training target: transformed M2/WMO geometry fragments
    /// only, retained per raster sample only when above raw MCVT terrain Z.
    /// This is deliberately distinct from historical approximate mask fields.
    /// </summary>
    public float[,]? ObjectGeometryVisibleMask257 { get; init; }

    /// <summary>
    /// 257×257 world elevation of the highest strict geometry fragment at each
    /// positive <see cref="ObjectGeometryVisibleMask257"/> pixel. Diagnostic
    /// provenance only; never a model input.
    /// </summary>
    public float[,]? ObjectGeometryVisibleTopElevation257 { get; init; }

    /// <summary>
    /// 257×257 raw-MCVT terrain elevation sampled at the same raster fragment
    /// as <see cref="ObjectGeometryVisibleTopElevation257"/>. Diagnostic
    /// provenance only; never a model input.
    /// </summary>
    public float[,]? ObjectGeometryVisibleTerrainElevation257 { get; init; }

    /// <summary>
    /// 257×257 source tag for strict geometry pixels: 0 none, 1 M2 triangle,
    /// 2 WMO triangle. Diagnostic provenance only; never a model input.
    /// </summary>
    public byte[,]? ObjectGeometryVisibleSource257 { get; init; }

    /// <summary>
    /// 257×257 per-object instance label for strict geometry pixels (Spec 118
    /// FR-002): 0 = no visible object, 1..K = per-tile compact instance id of
    /// the front-most visible fragment, assigned deterministically (MDDF
    /// placements first, then MODF) and resolved to placements/assets via
    /// <see cref="ObjectGeometryVisibleInstances"/>. Painted under the same
    /// front-most rule as <see cref="ObjectGeometryVisibleSource257"/>.
    /// </summary>
    public int[,]? ObjectGeometryVisibleInstance257 { get; init; }

    /// <summary>
    /// Instance table for <see cref="ObjectGeometryVisibleInstance257"/>:
    /// compact id → placement unique id, asset index, source class, and
    /// visible pixel count. Audit metadata only and never a model input.
    /// </summary>
    public IReadOnlyList<ObjectGeometryVisibleInstance> ObjectGeometryVisibleInstances { get; init; }
        = Array.Empty<ObjectGeometryVisibleInstance>();

    /// <summary>
    /// Per-tile completeness/provenance for the strict geometry target. Only
    /// complete-empty and complete-visible status may supervise M0.
    /// </summary>
    public ObjectGeometryTargetProvenance? ObjectGeometryTargetProvenance { get; init; }

    /// <summary>
    /// Asset table for <see cref="ObjectGeometryFragmentTrace"/>. This retains
    /// the exact normalized M2/WMO path referenced by each strict fragment;
    /// it is audit metadata only and never a model input.
    /// </summary>
    public IReadOnlyList<ObjectGeometryTargetAsset> ObjectGeometryTargetAssets { get; init; }
        = Array.Empty<ObjectGeometryTargetAsset>();

    /// <summary>
    /// Explicit placement/asset failures for a nonmaterialized strict target.
    /// These records are audit facts only and make incomplete geometry visible
    /// rather than silently converting it into an empty mask.
    /// </summary>
    public IReadOnlyList<ObjectGeometryTargetUnresolvedPlacement> ObjectGeometryTargetUnresolvedPlacements { get; init; }
        = Array.Empty<ObjectGeometryTargetUnresolvedPlacement>();

    /// <summary>
    /// Lossless uncollapsed triangle/raster-fragment evidence for the strict
    /// geometry target. The union mask remains the M0 label; this sidecar
    /// preserves transformed object XYZ, source identity, raw-MCVT
    /// interpolation, liquid state, and classification for audit.
    /// </summary>
    public ObjectGeometryFragmentTrace? ObjectGeometryFragmentTrace { get; init; }

    /// <summary>
    /// 257×257 per-instance object label mask.
    /// 0 = terrain (no object), 1..N = placement index (1-based).
    /// MDDF placements are labeled first (starting at 1), then MODF placements.
    /// Matches row indices in PlacementMddfData and PlacementModfData.
    /// </summary>
    public int[,]? ObjectInstanceMask257 { get; init; }

    /// <summary>257×257 raw MDDF (doodad) binary mask — unfiltered, all doodads.</summary>
    public float[,]? MddfMask257 { get; init; }

    /// <summary>257×257 raw MODF (WMO) binary mask — all WMOs.</summary>
    public float[,]? ModfMask257 { get; init; }

    /// <summary>
    /// 257×257 filtered combined object mask for terrain loss weighting.
    /// Excludes MDDF objects matching the exclusion regex (trees, bushes, etc.)
    /// and objects exceeding the height threshold. All MODF (WMO) are included.
    /// </summary>
    public float[,]? ObjectFilteredMask257 { get; init; }

    /// <summary>257×257 PM4 navigable path mask.</summary>
    public float[,]? Pm4PathMask { get; init; }

    /// <summary>257×257 PM4 building footprint mask.</summary>
    public float[,]? Pm4BuildingFootprintMask { get; init; }

    /// <summary>
    /// 257×257 PM4 MPRL portal mask. MPRL entries mark transitions between
    /// PM4 zones and correlate with ADT hole regions — the hole will exist
    /// within the same footprint as MPRL portal areas.
    /// </summary>
    public float[,]? Pm4MprlMask { get; init; }

    /// <summary>256×256 decoded MCSH shadow occupancy aligned to minimap space.</summary>
    public float[,]? McshShadowMask256 { get; init; }

    /// <summary>256×256 shadow occupancy not currently explained by the object explanation mask.</summary>
    public float[,]? ShadowResidualMask256 { get; init; }

    /// <summary>256×256 roof-focused object mask aligned to minimap space.</summary>
    public float[,]? ObjectRoofMask256 { get; init; }

    /// <summary>256×256 per-pixel confidence for <see cref="ObjectRoofMask256"/>.</summary>
    public float[,]? ObjectRoofConfidence256 { get; init; }

    /// <summary>Provenance label for the roof mask source (metadata, learned, heuristic, none).</summary>
    public string ObjectRoofMaskSource { get; init; } = string.Empty;

    // ── Minimap inputs ─────────────────────────────────────────────────────

    /// <summary>256×256 × 3 RGB minimap image used as the primary Stage 1 visual input.</summary>
    public byte[,,]? MinimapRgb256 { get; set; }

    // ── Hole and flag data ─────────────────────────────────────────────────

    /// <summary>16×16 chunk grid: does this chunk have holes?</summary>
    public bool[,]? HoleMask16 { get; init; }

    /// <summary>16×16 chunk grid: texture animation flags (MTXF).</summary>
    public int[,]? MtxfAnimatedMask { get; init; }

    /// <summary>16×16 chunk grid: texture transform IDs (MTXF).</summary>
    public int[,]? MtxfTransformId { get; init; }

    /// <summary>16×16 chunk grid: per-chunk MCSE sound-emitter counts.</summary>
    public int[,]? McseEmitterCounts16 { get; init; }

    /// <summary>Flattened decoded MCSE entry ids for standard 0x1C-byte emitters.</summary>
    public int[]? McseEntryIds { get; init; }

    /// <summary>Flattened decoded MCSE positions as an N×3 matrix for standard 0x1C-byte emitters.</summary>
    public float[,]? McsePositionXyz { get; init; }

    /// <summary>Exact MCSE emitter entry bytes as an N×stride matrix.</summary>
    public byte[,]? McseEntryBytes { get; init; }

    /// <summary>16×16 chunk grid: per-chunk pre-Cata doodad reference counts from MCRF.</summary>
    public int[,]? McrfDoodadRefCounts16 { get; init; }

    /// <summary>Flattened pre-Cata doodad reference indices from MCRF.</summary>
    public int[]? McrfDoodadRefIndices { get; init; }

    /// <summary>16×16 chunk grid: per-chunk pre-Cata world-model reference counts from MCRF.</summary>
    public int[,]? McrfWmoRefCounts16 { get; init; }

    /// <summary>Flattened pre-Cata world-model reference indices from MCRF.</summary>
    public int[]? McrfWmoRefIndices { get; init; }

    /// <summary>16×16 chunk grid: per-chunk Cataclysm+ doodad reference counts from MCRD.</summary>
    public int[,]? McrdRefCounts16 { get; init; }

    /// <summary>Flattened Cataclysm+ doodad reference indices from MCRD.</summary>
    public int[]? McrdRefIndices { get; init; }

    /// <summary>16×16 chunk grid: per-chunk Cataclysm+ world-model reference counts from MCRW.</summary>
    public int[,]? McrwRefCounts16 { get; init; }

    /// <summary>Flattened Cataclysm+ world-model reference indices from MCRW.</summary>
    public int[]? McrwRefIndices { get; init; }

    // ── Metadata ───────────────────────────────────────────────────────────

    /// <summary>Which signals were present in the source tile.</summary>
    public IReadOnlySet<string> AvailableSignals { get; set; } = new HashSet<string>();

    /// <summary>Minimap source tag (terrain_only, no_liquid, no_object, raw, etc.).</summary>
    public string MinimapSourceTag { get; set; } = string.Empty;

    /// <summary>
    /// Optional comparison of the authored minimap with an unlit terrain reconstruction. This is
    /// downstream bucketing evidence only: an inferred time is never a claim of exact capture time.
    /// </summary>
    public MinimapLightingProvenance? MinimapLightingProvenance { get; set; }

    /// <summary>
    /// Raw ADT-family chunks that the current tensor-pack builder does not yet decode into
    /// first-class signals. These are persisted into the NPZ shard as uint8 blobs so later
    /// passes can recover full-fidelity data without rereading the original client files.
    /// </summary>
    public IReadOnlyList<TerrainRawChunkBlob> RawChunks { get; init; } = Array.Empty<TerrainRawChunkBlob>();

    // ── Placement data ──────────────────────────────────────────────────────

    /// <summary>MDDF (M2) and MODF (WMO) model payloads for the V22 stream, keyed by canonical path.</summary>
    public IReadOnlyDictionary<string, V22ModelPayload>? PerTileModelPayloads { get; init; }

    /// <summary>Number of MDDF (M2 model) placements on this tile.</summary>
    public int PlacementMddfCount { get; init; }

    /// <summary>Number of MODF (WMO model) placements on this tile.</summary>
    public int PlacementModfCount { get; init; }

    /// <summary>
    /// MDDF placement flat array [N, 9]: nameId, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, scale.
    /// </summary>
    public float[,]? PlacementMddfData { get; init; }

    /// <summary>
    /// MODF placement flat array [N, 14]: nameId, uniqueId, posX, posY, posZ, rotX, rotY, rotZ,
    /// bbMinX, bbMinY, bbMinZ, bbMaxX, bbMaxY, bbMaxZ, flags.
    /// </summary>
    public float[,]? PlacementModfData { get; init; }

    /// <summary>MDDF model paths (index maps to nameId in placement_mddf_data).</summary>
    public IReadOnlyList<string> PlacementMddfNames { get; init; } = Array.Empty<string>();

    /// <summary>MODF model paths (index maps to nameId in placement_modf_data).</summary>
    public IReadOnlyList<string> PlacementModfNames { get; init; } = Array.Empty<string>();

    public TileLoadResult ToTileLoadResult(int tileX, int tileY)
    {
        const int chunksPerTile = 16;
        const int tileSize = 257;
        const int alphaSize = 64;
        const float mapOrigin = 17066f;
        const float chunkSize = 533.33333f;
        const float chunkSmall = chunkSize / chunksPerTile;

        float tileWorldX = mapOrigin - tileX * chunkSize;
        float tileWorldY = mapOrigin - tileY * chunkSize;

        var chunks = new List<TerrainChunkData>(256);

        for (int cy = 0; cy < chunksPerTile; cy++)
        {
            for (int cx = 0; cx < chunksPerTile; cx++)
            {
                float[] heights = SliceChunkHeights(tileX, tileY, cx, cy, tileSize);
                if (heights == null) continue;

                var layers = new List<TerrainLayer>();
                var alphaMaps = new Dictionary<int, byte[]>();

                if (MclyLayerMask != null && MclyTextureIds != null)
                {
                    for (int l = 0; l < 4; l++)
                    {
                        if (!MclyLayerMask[cx, cy, l])
                            break;

                        layers.Add(new TerrainLayer
                        {
                            TextureIndex = MclyTextureIds[cx, cy, l],
                            Flags = 0,
                            AlphaOffset = 0,
                            EffectId = 0
                        });

                        if (l > 0 && McalAlphaPack != null)
                        {
                            var alpha = SliceChunkAlpha256(McalAlphaPack, cx, cy, l, alphaSize);
                            if (alpha != null)
                                alphaMaps[l] = alpha;
                        }
                    }
                }

                int holeMask = 0;
                if (HoleMask16 != null && cx < HoleMask16.GetLength(0) && cy < HoleMask16.GetLength(1))
                    holeMask = HoleMask16[cx, cy] ? 1 : 0;

                int mcnkFlags = 0;
                if (McnkFlags16 != null && cx < McnkFlags16.GetLength(0) && cy < McnkFlags16.GetLength(1))
                    mcnkFlags = McnkFlags16[cx, cy];

                var liquid = FindLiquid(cx, cy);
                if (mcnkFlags == 0 && liquid != null)
                    mcnkFlags |= 0x3C;

                float chunkWorldX = tileWorldX - cy * chunkSmall;
                float chunkWorldY = tileWorldY - cx * chunkSmall;

                chunks.Add(new TerrainChunkData
                {
                    TileX = tileX,
                    TileY = tileY,
                    ChunkX = cx,
                    ChunkY = cy,
                    Heights = heights,
                    Normals = SliceChunkNormals(cx, cy),
                    HoleMask = holeMask,
                    Layers = layers.ToArray(),
                    AlphaMaps = alphaMaps,
                    ShadowMap = SliceChunkShadow(cx, cy),
                    Liquid = liquid,
                    WorldPosition = new Vector3(chunkWorldX, chunkWorldY, 0f),
                    AreaId = 0,
                    McnkFlags = mcnkFlags
                });
            }
        }

        return new TileLoadResult
        {
            Chunks = chunks,
            MddfPlacements = [],
            ModfPlacements = []
        };
    }

    private float[]? SliceChunkHeights(int tileX, int tileY, int cx, int cy, int tileSize)
    {
        if (Height257 == null) return null;

        var heights = new float[145];
        int baseX = cy * 16;
        int baseY = cx * 16;
        int idx = 0;

        for (int row = 0; row < 17; row++)
        {
            bool isInner = (row & 1) != 0;
            int cols = isInner ? 8 : 9;
            for (int col = 0; col < cols; col++)
            {
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                int px = baseX + sampleX;
                int py = baseY + sampleY;

                if ((uint)px < tileSize && (uint)py < tileSize)
                    heights[idx] = Height257[py, px];

                idx++;
            }
        }

        return heights;
    }

    private Vector3[] SliceChunkNormals(int cx, int cy)
    {
        if (McnrNormalXyz == null) return [];

        const int n = 145;
        var normals = new Vector3[n];
        int baseX = cy * 16;
        int baseY = cx * 16;
        int idx = 0;

        for (int row = 0; row < 17; row++)
        {
            bool isInner = (row & 1) != 0;
            int cols = isInner ? 8 : 9;
            for (int col = 0; col < cols; col++)
            {
                int sampleX = isInner ? (col * 2) + 1 : col * 2;
                int sampleY = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                int px = baseX + sampleX;
                int py = baseY + sampleY;

                if ((uint)px < 257 && (uint)py < 257)
                    normals[idx] = new Vector3(McnrNormalXyz[py, px, 0], McnrNormalXyz[py, px, 1], McnrNormalXyz[py, px, 2]);

                idx++;
            }
        }

        return normals;
    }

    private byte[]? SliceChunkShadow(int cx, int cy)
    {
        if (McshShadowMask256 == null) return null;

        const int srcSize = 256;
        const int dstSize = 64;
        var shadow = new byte[dstSize * dstSize];
        int srcBaseX = cy * dstSize;
        int srcBaseY = cx * dstSize;

        for (int y = 0; y < dstSize; y++)
        {
            for (int x = 0; x < dstSize; x++)
            {
                int sy = srcBaseY + y;
                int sx = srcBaseX + x;
                if (sy < srcSize && sx < srcSize)
                    shadow[y * dstSize + x] = (byte)(McshShadowMask256[sy, sx] * 255f);
            }
        }

        return shadow;
    }

    private byte[]? SliceChunkAlpha256(float[,,] alphaPack, int cx, int cy, int layer, int alphaSize)
    {
        var alpha = new byte[alphaSize * alphaSize];
        int srcBaseY = cx * alphaSize;
        int srcBaseX = cy * alphaSize;

        for (int y = 0; y < alphaSize; y++)
        {
            for (int x = 0; x < alphaSize; x++)
            {
                int sy = srcBaseY + y;
                int sx = srcBaseX + x;
                if (sy < alphaPack.GetLength(0) && sx < alphaPack.GetLength(1))
                {
                    float f = alphaPack[sy, sx, layer];
                    alpha[y * alphaSize + x] = (byte)Math.Clamp((int)(f * 255f), 0, 255);
                }
            }
        }

        return alpha;
    }

    private LiquidChunkData? FindLiquid(int cx, int cy)
    {
        bool hasLiquid = false;
        float minH = float.MaxValue, maxH = float.MinValue;
        int liquidType = 0;

        if (MclqSurfaceHeight != null)
        {
            int baseX = cy * 16;
            int baseY = cx * 16;
            bool found = false;
            for (int y = 0; y < 16 && baseY + y < 257; y++)
            {
                for (int x = 0; x < 16 && baseX + x < 257; x++)
                {
                    float h = MclqSurfaceHeight[baseY + y, baseX + x];
                    if (h != 0f)
                    {
                        found = true;
                        minH = Math.Min(minH, h);
                        maxH = Math.Max(maxH, h);
                    }
                }
            }

            if (found)
            {
                hasLiquid = true;
                if (MclqTypeMask != null && cx < 16 && cy < 16)
                    liquidType = MclqTypeMask[cx, cy];
            }
        }

        if (!hasLiquid && Mh2oSurfaceHeight != null)
        {
            int baseX = cy * 16;
            int baseY = cx * 16;
            bool found = false;
            for (int y = 0; y < 16 && baseY + y < 257; y++)
            {
                for (int x = 0; x < 16 && baseX + x < 257; x++)
                {
                    float h = Mh2oSurfaceHeight[baseY + y, baseX + x];
                    if (h != 0f)
                    {
                        found = true;
                        minH = Math.Min(minH, h);
                        maxH = Math.Max(maxH, h);
                    }
                }
            }

            if (found)
            {
                hasLiquid = true;
                if (Mh2oTypeMask != null && cx < 16 && cy < 16)
                    liquidType = Mh2oTypeMask[cx, cy];
            }
        }

        return hasLiquid
            ? new LiquidChunkData { LiquidType = liquidType, MinHeight = minH, MaxHeight = maxH }
            : null;
    }
}
