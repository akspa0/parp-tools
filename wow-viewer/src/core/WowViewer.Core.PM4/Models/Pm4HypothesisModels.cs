namespace WowViewer.Core.PM4.Models;

public sealed record Pm4ObjectHypothesis(
    string Family,
    int FamilyObjectIndex,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    IReadOnlyList<int> SurfaceIndices,
    IReadOnlyList<uint> MdosIndices,
    IReadOnlyList<byte> GroupKeys,
    IReadOnlyList<uint> MslkGroupObjectIds,
    IReadOnlyList<ushort> MslkRefIndices,
    uint DominantLinkGroupObjectId,
    Pm4ForensicsPlacementComparison PlacementComparison,
    Pm4Bounds3? Bounds,
    Pm4MprlFootprintSummary MprlFootprint);

public sealed record Pm4TileObjectHypothesisReport(
    string? SourcePath,
    int? TileX,
    int? TileY,
    uint Version,
    int Ck24GroupCount,
    int TotalHypothesisCount,
    IReadOnlyList<Pm4ObjectHypothesis> Objects,
    IReadOnlyList<string> Notes,
    IReadOnlyList<string> Diagnostics);

public sealed record Pm4CompactHypothesisFamilySummary(
    string Family,
    int ObjectCount,
    int MaxSurfaceCount,
    int MaxIndexCount,
    int TotalLinkedRefCount,
    int TotalLinkedInBoundsCount);

public sealed record Pm4CompactObjectHypothesis(
    string Family,
    int FamilyObjectIndex,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    int MdosCount,
    int GroupKeyCount,
    int MslkGroupObjectIdCount,
    int MslkRefIndexCount,
    uint DominantLinkGroupObjectId,
    Pm4CoordinateMode CoordinateMode,
    Pm4PlanarTransform PlanarTransform,
    float FrameYawDegrees,
    float? MprlHeadingMeanDegrees,
    float? HeadingDeltaDegrees,
    Pm4Bounds3? Bounds,
    Pm4MprlFootprintSummary MprlFootprint);

public sealed record Pm4CompactTileObjectHypothesisReport(
    string? SourcePath,
    int? TileX,
    int? TileY,
    uint Version,
    int Ck24GroupCount,
    int TotalHypothesisCount,
    IReadOnlyList<Pm4CompactHypothesisFamilySummary> Families,
    IReadOnlyList<Pm4CompactObjectHypothesis> Objects,
    IReadOnlyList<string> Notes,
    IReadOnlyList<string> Diagnostics);