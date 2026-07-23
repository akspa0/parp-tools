using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;

namespace WowViewer.Core.Maps;

/// <summary>
/// The only statuses accepted by the strict object-mask training contract are
/// <see cref="CompleteEmpty"/> and <see cref="CompleteVisible"/>. Every other
/// status means that a tile had an unavailable geometry/terrain fact and must
/// be rejected rather than approximated with a bounds or centroid mask.
/// </summary>
public enum ObjectGeometryTargetStatus
{
    Unavailable = 0,
    CompleteEmpty = 1,
    CompleteVisible = 2,
    PlacementCatalogUnavailable = 3,
    TerrainSurfaceUnavailable = 4,
    IncompleteGeometry = 5,
    /// <summary>
    /// Liquid parsing or liquid-height evidence was incomplete. Treat this as
    /// an unavailable fact, never as dry terrain.
    /// </summary>
    LiquidVisibilityUnknown = 6,
}

/// <summary>
/// Whether liquid evidence was sufficient for the strict target. Initial M0
/// curation is deliberately dry-only; liquid-bearing targets remain useful
/// provenance for a later liquid-aware contract but never authorize water
/// pixels as background negatives.
/// </summary>
public enum ObjectGeometryLiquidEvidenceStatus
{
    Unknown = 0,
    Dry = 1,
    LiquidPresent = 2,
}

/// <summary>Exact source geometry that supplied a strict object-mask pixel.</summary>
public enum ObjectGeometryPixelSource : byte
{
    None = 0,
    M2Triangle = 1,
    WmoTriangle = 2,
}

/// <summary>
/// The independently auditable disposition of one triangle/raster-sample
/// fragment. This is intentionally separate from the final union mask: two
/// placements may overlap one pixel and both source records must survive.
/// </summary>
public enum ObjectGeometryFragmentClassification : byte
{
    TerrainVisible = 1,
    TerrainHidden = 2,
    WaterHidden = 3,
    TerrainUnknown = 4,
    LiquidUnknown = 5,
}

/// <summary>Stable per-tile asset table entry referenced by fragment records.</summary>
public sealed record ObjectGeometryTargetAsset(
    int AssetIndex,
    ObjectGeometryPixelSource Source,
    string NormalizedAssetPath);

/// <summary>
/// An explicit reason a placement could not contribute to a strict target.
/// This prevents missing geometry from collapsing into an unexplained empty
/// mask and preserves the exact placement/asset investigation handle.
/// </summary>
public sealed record ObjectGeometryTargetUnresolvedPlacement(
    int PlacementUniqueId,
    ObjectGeometryPixelSource Source,
    string NormalizedAssetPath,
    string Reason);

/// <summary>
/// Per-tile compact instance table for the strict visible-object target
/// (Spec 118 FR-002). <see cref="InstanceId"/> is the value painted into the
/// dense visible-instance array (1..K, deterministic assignment order: MDDF
/// placements first, then MODF); it links back to the placement's unique id
/// and asset. A resolved placement that is fully occluded or underground is
/// retained with <see cref="VisiblePixelCount"/> 0 rather than dropped, so the
/// table always accounts for every placement the dense array could name.
/// </summary>
public sealed record ObjectGeometryVisibleInstance(
    int InstanceId,
    int PlacementUniqueId,
    int AssetIndex,
    ObjectGeometryPixelSource Source,
    int VisiblePixelCount);

/// <summary>
/// One uncollapsed transformed-triangle/raster-sample trace record. The three
/// terrain nodes are the raw MCVT interpolation triangle in dense coordinates;
/// their original chunk/local identity and world coordinates remain available
/// from the tile's canonical MCVT arrays.
/// </summary>
public readonly record struct ObjectGeometryFragmentRecord(
    int RasterX,
    int RasterY,
    float ObjectWorldX,
    float ObjectWorldY,
    float ObjectWorldZ,
    float ObjectElevation,
    int PlacementUniqueId,
    int AssetIndex,
    int SourceTriangleIndex,
    ObjectGeometryPixelSource Source,
    ObjectGeometryFragmentClassification Classification,
    int TerrainVertex0X,
    int TerrainVertex0Y,
    int TerrainVertex1X,
    int TerrainVertex1Y,
    int TerrainVertex2X,
    int TerrainVertex2Y,
    float TerrainVertex0Z,
    float TerrainVertex1Z,
    float TerrainVertex2Z,
    bool TerrainVertex0Present,
    bool TerrainVertex1Present,
    bool TerrainVertex2Present,
    float TerrainWeight0,
    float TerrainWeight1,
    float TerrainWeight2,
    float TerrainElevation,
    float LiquidSurfaceElevation);

/// <summary>
/// Lossless tabular transport for every strict target fragment. It is an audit
/// sidecar, not an M0 forward input or target tensor. Arrays use a row-major
/// record layout so NPZ/raw writers can serialize them without inventing an
/// image representation for mesh facts.
/// </summary>
public sealed class ObjectGeometryFragmentTrace
{
    private ObjectGeometryFragmentTrace(
        int[,] rasterXy,
        float[,] objectWorldXyzElevation,
        int[,] sourceIds,
        byte[,] sourceClassification,
        int[,] terrainVertexDenseXy,
        float[,] terrainVertexZ,
        byte[,] terrainVertexPresent,
        float[,] terrainWeights,
        float[,] terrainLiquidElevation,
        string contentSha256)
    {
        RasterXy = rasterXy;
        ObjectWorldXyzElevation = objectWorldXyzElevation;
        SourceIds = sourceIds;
        SourceClassification = sourceClassification;
        TerrainVertexDenseXy = terrainVertexDenseXy;
        TerrainVertexZ = terrainVertexZ;
        TerrainVertexPresent = terrainVertexPresent;
        TerrainWeights = terrainWeights;
        TerrainLiquidElevation = terrainLiquidElevation;
        ContentSha256 = contentSha256;
    }

    /// <summary>Rows × [raster X, raster Y].</summary>
    public int[,] RasterXy { get; }

    /// <summary>
    /// Rows × transformed object world [X, Y, Z, comparison elevation]. The
    /// fourth value makes the terrain-axis convention explicit rather than
    /// assuming Z is height for every legacy placement convention.
    /// </summary>
    public float[,] ObjectWorldXyzElevation { get; }

    /// <summary>Rows × [placement unique id, asset-table id, source triangle id].</summary>
    public int[,] SourceIds { get; }

    /// <summary>Rows × [M2/WMO source kind, fragment classification].</summary>
    public byte[,] SourceClassification { get; }

    /// <summary>Rows × [v0 X, v0 Y, v1 X, v1 Y, v2 X, v2 Y].</summary>
    public int[,] TerrainVertexDenseXy { get; }

    /// <summary>Rows × raw MCVT [v0 Z, v1 Z, v2 Z]; unavailable nodes are NaN.</summary>
    public float[,] TerrainVertexZ { get; }

    /// <summary>Rows × raw MCVT [v0 present, v1 present, v2 present].</summary>
    public byte[,] TerrainVertexPresent { get; }

    /// <summary>Rows × barycentric [v0, v1, v2] interpolation weights.</summary>
    public float[,] TerrainWeights { get; }

    /// <summary>Rows × [interpolated terrain Z, liquid surface Z]; unavailable values are NaN.</summary>
    public float[,] TerrainLiquidElevation { get; }

    /// <summary>SHA-256 over the canonical asset table and every trace row.</summary>
    public string ContentSha256 { get; }

    public int Count => RasterXy.GetLength(0);

    public static ObjectGeometryFragmentTrace Create(
        IReadOnlyList<ObjectGeometryFragmentRecord> records,
        IReadOnlyList<ObjectGeometryTargetAsset> assets)
    {
        ArgumentNullException.ThrowIfNull(records);
        ArgumentNullException.ThrowIfNull(assets);

        int count = records.Count;
        int[,] rasterXy = new int[count, 2];
        float[,] objectWorldXyzElevation = new float[count, 4];
        int[,] sourceIds = new int[count, 3];
        byte[,] sourceClassification = new byte[count, 2];
        int[,] terrainVertexDenseXy = new int[count, 6];
        float[,] terrainVertexZ = new float[count, 3];
        byte[,] terrainVertexPresent = new byte[count, 3];
        float[,] terrainWeights = new float[count, 3];
        float[,] terrainLiquidElevation = new float[count, 2];

        for (int index = 0; index < count; index++)
        {
            ObjectGeometryFragmentRecord record = records[index];
            rasterXy[index, 0] = record.RasterX;
            rasterXy[index, 1] = record.RasterY;
            objectWorldXyzElevation[index, 0] = record.ObjectWorldX;
            objectWorldXyzElevation[index, 1] = record.ObjectWorldY;
            objectWorldXyzElevation[index, 2] = record.ObjectWorldZ;
            objectWorldXyzElevation[index, 3] = record.ObjectElevation;
            sourceIds[index, 0] = record.PlacementUniqueId;
            sourceIds[index, 1] = record.AssetIndex;
            sourceIds[index, 2] = record.SourceTriangleIndex;
            sourceClassification[index, 0] = (byte)record.Source;
            sourceClassification[index, 1] = (byte)record.Classification;
            terrainVertexDenseXy[index, 0] = record.TerrainVertex0X;
            terrainVertexDenseXy[index, 1] = record.TerrainVertex0Y;
            terrainVertexDenseXy[index, 2] = record.TerrainVertex1X;
            terrainVertexDenseXy[index, 3] = record.TerrainVertex1Y;
            terrainVertexDenseXy[index, 4] = record.TerrainVertex2X;
            terrainVertexDenseXy[index, 5] = record.TerrainVertex2Y;
            terrainVertexZ[index, 0] = record.TerrainVertex0Z;
            terrainVertexZ[index, 1] = record.TerrainVertex1Z;
            terrainVertexZ[index, 2] = record.TerrainVertex2Z;
            terrainVertexPresent[index, 0] = record.TerrainVertex0Present ? (byte)1 : (byte)0;
            terrainVertexPresent[index, 1] = record.TerrainVertex1Present ? (byte)1 : (byte)0;
            terrainVertexPresent[index, 2] = record.TerrainVertex2Present ? (byte)1 : (byte)0;
            terrainWeights[index, 0] = record.TerrainWeight0;
            terrainWeights[index, 1] = record.TerrainWeight1;
            terrainWeights[index, 2] = record.TerrainWeight2;
            terrainLiquidElevation[index, 0] = record.TerrainElevation;
            terrainLiquidElevation[index, 1] = record.LiquidSurfaceElevation;
        }

        return new ObjectGeometryFragmentTrace(
            rasterXy,
            objectWorldXyzElevation,
            sourceIds,
            sourceClassification,
            terrainVertexDenseXy,
            terrainVertexZ,
            terrainVertexPresent,
            terrainWeights,
            terrainLiquidElevation,
            ComputeSha256(
                rasterXy,
                objectWorldXyzElevation,
                sourceIds,
                sourceClassification,
                terrainVertexDenseXy,
                terrainVertexZ,
                terrainVertexPresent,
                terrainWeights,
                terrainLiquidElevation,
                assets));
    }

    /// <summary>
    /// Verifies that the exposed trace arrays still match the immutable hash
    /// produced at construction time. Callers can mutate CLR arrays after a
    /// trace is created, so serializers must check this before treating the
    /// sidecar as audit evidence.
    /// </summary>
    public bool HasConsistentContentHash(IReadOnlyList<ObjectGeometryTargetAsset> assets)
    {
        ArgumentNullException.ThrowIfNull(assets);
        return string.Equals(
            ContentSha256,
            ComputeSha256(
                RasterXy,
                ObjectWorldXyzElevation,
                SourceIds,
                SourceClassification,
                TerrainVertexDenseXy,
                TerrainVertexZ,
                TerrainVertexPresent,
                TerrainWeights,
                TerrainLiquidElevation,
                assets),
            StringComparison.Ordinal);
    }

    private static string ComputeSha256(
        int[,] rasterXy,
        float[,] objectWorldXyzElevation,
        int[,] sourceIds,
        byte[,] sourceClassification,
        int[,] terrainVertexDenseXy,
        float[,] terrainVertexZ,
        byte[,] terrainVertexPresent,
        float[,] terrainWeights,
        float[,] terrainLiquidElevation,
        IReadOnlyList<ObjectGeometryTargetAsset> assets)
    {
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        byte[] scalar = new byte[sizeof(int)];

        void AddInt(int value)
        {
            BinaryPrimitives.WriteInt32LittleEndian(scalar, value);
            hash.AppendData(scalar);
        }

        void AddFloat(float value)
        {
            BinaryPrimitives.WriteInt32LittleEndian(scalar, BitConverter.SingleToInt32Bits(value));
            hash.AppendData(scalar);
        }

        void AddByte(byte value)
        {
            scalar[0] = value;
            hash.AppendData(scalar.AsSpan(0, 1));
        }

        AddInt(assets.Count);
        foreach (ObjectGeometryTargetAsset asset in assets.OrderBy(static asset => asset.AssetIndex))
        {
            AddInt(asset.AssetIndex);
            AddByte((byte)asset.Source);
            byte[] path = Encoding.UTF8.GetBytes(asset.NormalizedAssetPath ?? string.Empty);
            AddInt(path.Length);
            hash.AppendData(path);
        }

        int count = rasterXy.GetLength(0);
        AddInt(count);
        for (int index = 0; index < count; index++)
        {
            AddInt(rasterXy[index, 0]);
            AddInt(rasterXy[index, 1]);
            AddFloat(objectWorldXyzElevation[index, 0]);
            AddFloat(objectWorldXyzElevation[index, 1]);
            AddFloat(objectWorldXyzElevation[index, 2]);
            AddFloat(objectWorldXyzElevation[index, 3]);
            AddInt(sourceIds[index, 0]);
            AddInt(sourceIds[index, 1]);
            AddInt(sourceIds[index, 2]);
            AddByte(sourceClassification[index, 0]);
            AddByte(sourceClassification[index, 1]);
            AddInt(terrainVertexDenseXy[index, 0]);
            AddInt(terrainVertexDenseXy[index, 1]);
            AddInt(terrainVertexDenseXy[index, 2]);
            AddInt(terrainVertexDenseXy[index, 3]);
            AddInt(terrainVertexDenseXy[index, 4]);
            AddInt(terrainVertexDenseXy[index, 5]);
            AddFloat(terrainVertexZ[index, 0]);
            AddFloat(terrainVertexZ[index, 1]);
            AddFloat(terrainVertexZ[index, 2]);
            AddByte(terrainVertexPresent[index, 0]);
            AddByte(terrainVertexPresent[index, 1]);
            AddByte(terrainVertexPresent[index, 2]);
            AddFloat(terrainWeights[index, 0]);
            AddFloat(terrainWeights[index, 1]);
            AddFloat(terrainWeights[index, 2]);
            AddFloat(terrainLiquidElevation[index, 0]);
            AddFloat(terrainLiquidElevation[index, 1]);
        }

        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }
}

/// <summary>
/// Per-tile provenance for <c>object_geometry_visible_mask_257</c>. This is
/// deliberately separate from the historical <c>object_precise_mask_257</c>,
/// which may contain legacy approximation fallbacks in existing corpora.
/// </summary>
public sealed record ObjectGeometryTargetProvenance(
    ObjectGeometryTargetStatus Status,
    int PlacementCount,
    int GeometryResolvedPlacementCount,
    int GeometryUnresolvedPlacementCount,
    int FallbackRequiredPlacementCount,
    int TriangleCount,
    int VisiblePixelCount,
    int OccludedPixelCount,
    int TerrainUnknownPixelCount,
    ObjectGeometryLiquidEvidenceStatus LiquidEvidenceStatus = ObjectGeometryLiquidEvidenceStatus.Unknown,
    int LiquidCoveredPixelCount = 0,
    int LiquidSurfaceUnknownPixelCount = 0,
    int LiquidCoveredFragmentCount = 0,
    int LiquidHiddenFragmentCount = 0,
    int LiquidAboveSurfaceFragmentCount = 0,
    int LiquidUnknownFragmentCount = 0)
{
    public const string ContractVersion = "strict-geometry-terrain-liquid-fragment-trace-v3";

    public bool IsMaterialized => (Status is ObjectGeometryTargetStatus.CompleteEmpty
        or ObjectGeometryTargetStatus.CompleteVisible)
        && LiquidEvidenceStatus != ObjectGeometryLiquidEvidenceStatus.Unknown;
}
