using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Caching;

/// <summary>
/// Lightweight, library-owned record of one tile's worth of decoded PM4
/// overlay data, suitable for in-memory and on-disk per-file cache
/// payloads. Does not depend on the viewer-side
/// <c>Pm4OverlayCacheTile</c> / <c>Pm4OverlayCacheObject</c> records so
/// the per-file cache types can live in
/// <c>WowViewer.Core.PM4.Caching</c> without pulling in viewer types.
/// </summary>
public sealed record Pm4CachedTile(
    int TileX,
    int TileY,
    IReadOnlyList<Vector3> PositionRefs,
    IReadOnlyList<Pm4CachedObject> Objects);

public sealed record Pm4CachedObject(
    string SourcePath,
    uint MshdField00,
    uint MshdRegionId,
    uint MshdField08,
    uint Ck24,
    byte Ck24Type,
    int ObjectPartId,
    uint LinkGroupObjectId,
    int LinkedPositionRefCount,
    Pm4LinkedPositionRefSummary LinkedPositionRefSummary,
    int SurfaceCount,
    int TotalIndexCount,
    byte DominantGroupKey,
    byte DominantAttributeMask,
    uint DominantMscnRefIndex,
    float AverageSurfaceHeight,
    Vector3 PlacementAnchor,
    float BaseRotationRadians,
    bool PlanarSwapPlanarAxes,
    bool PlanarInvertU,
    bool PlanarInvertV,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    IReadOnlyList<Pm4CachedConnectorKey> ConnectorKeys,
    IReadOnlyList<Pm4CachedLineSegment> Lines,
    IReadOnlyList<Pm4CachedTriangle> Triangles);

public readonly record struct Pm4CachedConnectorKey(int X, int Y, int Z);

public readonly record struct Pm4CachedLineSegment(Vector3 From, Vector3 To);

public readonly record struct Pm4CachedTriangle(Vector3 A, Vector3 B, Vector3 C);

/// <summary>
/// One per-PM4-file entry in the per-file PM4 overlay cache.
/// The payload is the list of <see cref="Pm4CachedTile"/>s that the
/// PM4 file produced (one tile per file in the common case, two if the
/// file sits on a corner and a viewer-side split maps it to two tiles).
/// The stamp is the file's (Length, LastWriteTicks) at the time of
/// decode, so a re-read with a different stamp is treated as a miss.
/// </summary>
public sealed record Pm4PerFileCacheEntry(
    long FileLength,
    long LastWriteTicks,
    IReadOnlyList<Pm4CachedTile> Tiles);
