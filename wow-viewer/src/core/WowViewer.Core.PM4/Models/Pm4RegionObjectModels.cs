using System.Numerics;

namespace WowViewer.Core.PM4.Models;

/// <summary>
/// A sub-object within a CK24 object, partitioned by MSLK.GroupObjectId.
/// Contains the surface indices, linked MPRL position references, and placement metadata.
/// </summary>
public sealed record Pm4SubObject(
    uint GroupObjectId,
    IReadOnlyList<int> SurfaceIndices,
    IReadOnlyList<Pm4MprlEntry> PositionRefs,
    IReadOnlyList<uint> MslkGroupObjectIds,
    uint DominantLinkGroupObjectId,
    Pm4Bounds3? Bounds,
    float AverageSurfaceHeight)
{
    public int SurfaceCount => SurfaceIndices.Count;
    public int PositionRefCount => PositionRefs.Count;
}

/// <summary>
/// A CK24 object within a region, containing one or more sub-objects.
/// </summary>
public sealed record Pm4RegionObject(
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    IReadOnlyList<Pm4SubObject> SubObjects,
    IReadOnlyList<string> TileCoordinates,
    int TotalSurfaceCount,
    int TotalIndexCount)
{
    public int SubObjectCount => SubObjects.Count;
    public int TileCount => TileCoordinates.Count;
    public bool SpansMultipleTiles => TileCount > 1;
}

/// <summary>
/// A region grouping all objects belonging to a single MSHD.Field04 value.
/// Regions can span multiple ADT tiles and contain multiple CK24 objects.
/// </summary>
public sealed record Pm4Region(
    uint RegionId,
    IReadOnlyList<Pm4RegionObject> Objects,
    IReadOnlyList<string> TileCoordinates,
    int TotalSurfaceCount,
    int TotalObjectCount)
{
    public int TileCount => TileCoordinates.Count;
    public bool IsEmptyStubRegion => RegionId == 1;
}

/// <summary>
/// Complete region-aware grouping report for a map directory.
/// </summary>
public sealed record Pm4RegionGroupingReport(
    string Directory,
    int TotalFiles,
    int NonEmptyFiles,
    int TotalRegions,
    int TotalObjects,
    int TotalSubObjects,
    IReadOnlyList<Pm4Region> Regions,
    IReadOnlyList<string> Notes)
{
    public IReadOnlyList<Pm4Region> NonEmptyRegions => Regions.Where(r => !r.IsEmptyStubRegion).ToList();
}

/// <summary>
/// Decoded world-space placement for a single sub-object.
/// </summary>
public sealed record Pm4DecodedObjectPlacement(
    uint RegionId,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    uint SubObjectId,
    Vector3 WorldPosition,
    float WorldHeadingDegrees,
    Pm4Bounds3 WorldBounds,
    int SurfaceCount,
    int PositionRefCount,
    int TotalIndexCount,
    IReadOnlyList<string> TileCoordinates,
    Pm4CoordinateMode CoordinateMode,
    Pm4AxisConvention AxisConvention,
    Pm4PlanarTransform PlanarTransform,
    float FrameYawDegrees)
{
    public string ObjectType => Ck24Type switch
    {
        0x00 => "NavMesh",
        0x40 => "M2Interior",
        0x41 => "M2Interior",
        0x42 => "WMO",
        0x43 => "WMO",
        0xC0 => "M2Exterior",
        0xC1 => "M2Exterior",
        0xC2 => "M2Exterior",
        0xC3 => "M2Exterior",
        _ => $"Unknown(0x{Ck24Type:X2})"
    };
}

/// <summary>
/// Complete decoded report for a region grouping.
/// </summary>
public sealed record Pm4DecodedRegionReport(
    string Directory,
    int TotalFiles,
    int NonEmptyFiles,
    int TotalRegions,
    int TotalDecodedObjects,
    IReadOnlyList<Pm4DecodedRegion> DecodedRegions,
    IReadOnlyList<string> Notes);

/// <summary>
/// Decoded objects within a single region.
/// </summary>
public sealed record Pm4DecodedRegion(
    uint RegionId,
    IReadOnlyList<Pm4DecodedObjectPlacement> Objects,
    int TileCount,
    int TotalSurfaceCount)
{
    public int ObjectCount => Objects.Count;
}

// Pm4Bounds3 is defined in Pm4ResearchChunkModels.cs
