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

    // ── Height fields ──────────────────────────────────────────────────────

    /// <summary>257×257 per-vertex terrain heights (MCVT assembled across 16×16 chunks).</summary>
    public float[,]? Height257 { get; init; }

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

    /// <summary>
    /// 16×16 chunk grid × 4 layers.
    /// Boolean flags: is this layer active in this chunk?
    /// </summary>
    public bool[,,]? MclyLayerMask { get; init; }

    /// <summary>
    /// 256×256 tile-level alpha pack (4 channels, 0-1 float).
    /// Assembled from per-chunk MCAL decode.
    /// Channel 0 = base layer (always implied, may be all-ones).
    /// Channels 1-3 = blend weights for additional layers.
    /// </summary>
    public float[,,]? McalAlphaPack256 { get; init; }

    // ── Vertex colors and normals ──────────────────────────────────────────

    /// <summary>257×257 × 3 RGB vertex ambient colors (MCCV).</summary>
    public float[,,]? MccvRgb { get; init; }

    /// <summary>257×257 × 3 XYZ per-vertex normals (MCNR).</summary>
    public float[,,]? McnrNormalXyz { get; init; }

    // ── Liquid data ────────────────────────────────────────────────────────

    /// <summary>257×257 liquid surface height from MH2O (WotLK+).</summary>
    public float[,]? Mh2oSurfaceHeight { get; init; }

    /// <summary>257×257 liquid depth from MH2O (WotLK+).</summary>
    public float[,]? Mh2oDepth { get; init; }

    /// <summary>257×257 liquid type mask from MH2O (WotLK+).</summary>
    public int[,]? Mh2oTypeMask { get; init; }

    /// <summary>257×257 liquid surface height from MCLQ (pre-WotLK).</summary>
    public float[,]? MclqSurfaceHeight { get; init; }

    /// <summary>257×257 liquid type mask from MCLQ (pre-WotLK).</summary>
    public int[,]? MclqTypeMask { get; init; }

    /// <summary>257×257 liquid mask from WLW/WLM/WLQ/WLL loose files.</summary>
    public float[,]? WlLiquidMask { get; set; }

    /// <summary>257×257 liquid height from WLW/WLM/WLQ/WLL loose files.</summary>
    public float[,]? WlLiquidHeight { get; set; }

    /// <summary>
    /// 257×257 unified liquid mask combining MH2O, MCLQ, and WL* sources.
    /// Priority: MH2O > MCLQ > WL*. 1.0 where any liquid source indicates water.
    /// </summary>
    public float[,]? UnifiedLiquidMask { get; init; }

    /// <summary>
    /// 257×257 unified liquid surface height combining MH2O, MCLQ, and WL* sources.
    /// Priority: MH2O > MCLQ > WL*.
    /// </summary>
    public float[,]? UnifiedLiquidHeight { get; init; }

    // ── Object and footprint masks ─────────────────────────────────────────

    /// <summary>257×257 binary mask: any object footprint present.</summary>
    public float[,]? ObjectMask257 { get; init; }

    /// <summary>257×257 anti-aliased object silhouette.</summary>
    public float[,]? ObjectPreciseMask257 { get; init; }

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

    // ── Metadata ───────────────────────────────────────────────────────────

    /// <summary>Which signals were present in the source tile.</summary>
    public IReadOnlySet<string> AvailableSignals { get; set; } = new HashSet<string>();

    /// <summary>Minimap source tag (terrain_only, no_liquid, no_object, raw, etc.).</summary>
    public string MinimapSourceTag { get; set; } = string.Empty;

    // ── Placement data ──────────────────────────────────────────────────────

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

                        if (l > 0 && McalAlphaPack256 != null)
                        {
                            var alpha = SliceChunkAlpha256(McalAlphaPack256, cx, cy, l, alphaSize);
                            if (alpha != null)
                                alphaMaps[l] = alpha;
                        }
                    }
                }

                int holeMask = 0;
                if (HoleMask16 != null && cx < HoleMask16.GetLength(0) && cy < HoleMask16.GetLength(1))
                    holeMask = HoleMask16[cx, cy] ? 1 : 0;

                int mcnkFlags = 0;
                var liquid = FindLiquid(cx, cy);
                if (liquid != null)
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
