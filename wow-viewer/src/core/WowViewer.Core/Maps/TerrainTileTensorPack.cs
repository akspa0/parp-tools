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
    public float[,]? WlLiquidMask { get; init; }

    /// <summary>257×257 liquid height from WLW/WLM/WLQ/WLL loose files.</summary>
    public float[,]? WlLiquidHeight { get; init; }

    // ── Object and footprint masks ─────────────────────────────────────────

    /// <summary>257×257 binary mask: any object footprint present.</summary>
    public float[,]? ObjectMask257 { get; init; }

    /// <summary>257×257 anti-aliased object silhouette.</summary>
    public float[,]? ObjectPreciseMask257 { get; init; }

    /// <summary>257×257 PM4 navigable path mask.</summary>
    public float[,]? Pm4PathMask { get; init; }

    /// <summary>257×257 PM4 building footprint mask.</summary>
    public float[,]? Pm4BuildingFootprintMask { get; init; }

    // ── Hole and flag data ─────────────────────────────────────────────────

    /// <summary>16×16 chunk grid: does this chunk have holes?</summary>
    public bool[,]? HoleMask16 { get; init; }

    /// <summary>16×16 chunk grid: texture animation flags (MTXF).</summary>
    public int[,]? MtxfAnimatedMask { get; init; }

    /// <summary>16×16 chunk grid: texture transform IDs (MTXF).</summary>
    public int[,]? MtxfTransformId { get; init; }

    // ── Metadata ───────────────────────────────────────────────────────────

    /// <summary>Which signals were present in the source tile.</summary>
    public IReadOnlySet<string> AvailableSignals { get; init; } = new HashSet<string>();

    /// <summary>Minimap source tag (terrain_only, no_liquid, no_object, raw, etc.).</summary>
    public string MinimapSourceTag { get; init; } = string.Empty;
}
