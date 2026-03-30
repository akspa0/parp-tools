using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public sealed record Pm4ForensicsSurfaceRow(
    int SurfaceIndex,
    byte GroupKey,
    byte AttributeMask,
    uint MdosIndex,
    byte Ck24Type,
    ushort Ck24ObjectId,
    byte IndexCount,
    uint MsviFirstIndex,
    float Height,
    Vector3 Normal);

public sealed record Pm4ForensicsMslkRow(
    int LinkIndex,
    byte TypeFlags,
    byte Subtype,
    uint GroupObjectId,
    int MspiFirstIndex,
    byte MspiIndexCount,
    uint LinkId,
    ushort RefIndex,
    ushort SystemFlag,
    bool ReferencesSelectedSurface,
    bool ReferencesAnyPositionRef);

public sealed record Pm4ForensicsMprlRow(
    int RefIndex,
    ushort Unk00,
    short Unk02,
    ushort Unk04,
    ushort Unk06,
    Vector3 Position,
    short Unk14,
    ushort Unk16,
    float HeadingDegrees,
    bool IsTerminator);

public sealed record Pm4MprlFootprintSummary(
    int TileRefCount,
    int LinkedRefCount,
    int LinkedNormalCount,
    int LinkedTerminatorCount,
    int TileInBoundsCount,
    int TileNearBoundsCount,
    int LinkedInBoundsCount,
    int LinkedNearBoundsCount,
    short? LinkedFloorMin,
    short? LinkedFloorMax);

public sealed record Pm4ForensicsPlacementComparison(
    int? TileX,
    int? TileY,
    Pm4AxisConvention AxisConvention,
    Pm4CoordinateMode CoordinateMode,
    Pm4PlanarTransform PlanarTransform,
    bool UsedCoordinateFallback,
    float? TileLocalScore,
    float? WorldSpaceScore,
    Vector3 WorldPivot,
    float FrameYawDegrees,
    float? MprlHeadingMeanDegrees,
    float? HeadingDeltaDegrees);

public sealed record Pm4ForensicsLinkGroupReport(
    uint LinkGroupObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    IReadOnlyList<byte> AttributeMasks,
    IReadOnlyList<byte> GroupKeys,
    IReadOnlyList<uint> MdosIndices,
    IReadOnlyList<int> SurfaceIndices,
    IReadOnlyList<int> ReferencedPositionRefIndices,
    Pm4Bounds3? Bounds,
    Pm4LinkedPositionRefSummary LinkedPositionRefSummary,
    Pm4MprlFootprintSummary MprlFootprint,
    IReadOnlyList<Pm4ForensicsSurfaceRow> Surfaces,
    IReadOnlyList<Pm4ForensicsMslkRow> MslkRows,
    IReadOnlyList<Pm4ForensicsMprlRow> MprlRows,
    Pm4ForensicsPlacementComparison PlacementComparison);

public sealed record Pm4Ck24ForensicsReport(
    string? SourcePath,
    uint Version,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    int DistinctLinkGroupCount,
    int DistinctMdosCount,
    int DistinctAttributeMaskCount,
    int DistinctGroupKeyCount,
    IReadOnlyList<Pm4TerminologyEntry> Terminology,
    IReadOnlyList<byte> AttributeMasks,
    IReadOnlyList<byte> GroupKeys,
    IReadOnlyList<uint> MdosIndices,
    Pm4ForensicsPlacementComparison PlacementComparison,
    IReadOnlyList<Pm4ForensicsLinkGroupReport> LinkGroups,
    IReadOnlyList<string> Notes,
    IReadOnlyList<string> Diagnostics);