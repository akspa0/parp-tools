namespace WowViewer.Core.PM4.Models;

/// <summary>
/// Population of MSLK path windows, split by the three states a window can be in.
/// The three counts plus <see cref="ActiveWindows"/> partition <see cref="TotalMslkEntries"/>.
/// </summary>
public sealed record Pm4MslkWindowPopulation(
    int TotalMslkEntries,
    int ActiveWindows,
    int NegativeFirstIndexEntries,
    int ZeroCountEntries,
    long TotalWindowIndices,
    double MeanIndicesPerWindow,
    int MinWindowSize,
    int MaxWindowSize);

/// <summary>
/// One window-length bucket. <see cref="Size"/> is an MSLK MspiIndexCount value.
/// </summary>
public sealed record Pm4WindowSizeBucket(
    int Size,
    int WindowCount,
    double Fraction);

/// <summary>
/// Topology evidence for a set of windows. These are the measurements that can actually
/// separate a polyline from a triangle run; the legacy indices-vs-triangles mode counters
/// cannot, because the triangles bound implies the indices bound for every input.
/// </summary>
public sealed record Pm4WindowTopologyEvidence(
    int WindowsMeasured,
    int ClosedWindows,
    int MultipleOfThreeWindows,
    int WindowsWithDuplicateVertices,
    int CollinearWindows,
    int CoplanarWindows,
    long TriplesTested,
    long DegenerateTriples,
    double DegenerateTripleFraction);

/// <summary>
/// Per-TypeFlags/Subtype family view. Window semantics may differ by family rather than
/// being uniform, which is why nothing here is reported only in aggregate.
/// </summary>
public sealed record Pm4WindowFamilySummary(
    string FamilyKey,
    byte TypeFlags,
    byte Subtype,
    int FileCount,
    int TotalEntries,
    int ActiveWindows,
    int NegativeFirstIndexEntries,
    double MeanWindowSize,
    int ModalWindowSize,
    double MultipleOfThreeFraction,
    double ClosedFraction,
    Pm4WindowTopologyEvidence Topology,
    IReadOnlyList<Pm4WindowSizeBucket> TopSizes);

/// <summary>
/// MSUR._0x18 -> MSCN linkage. Prior art describes MSCN as the per-object exterior
/// boundary; this measures whether that edge resolves and how much of MSCN it reaches.
/// </summary>
public sealed record Pm4MscnLinkageSummary(
    int FilesWithMscn,
    long MsurToMscnFits,
    long MsurToMscnMisses,
    long TotalMscnPoints,
    long DistinctMscnReferenced,
    long MscnPointsUnreferenced,
    double ReferencedFraction,
    double MscnToMsvtRatio);

/// <summary>
/// Orientation of a set of planar faces, expressed without assuming which axis is up.
/// Each face is bucketed by which component of its unit normal is largest.
/// </summary>
/// <remarks>
/// Comparing the MSPV/MSPI quad orientation against MSUR's own <c>Normal</c> field is the
/// assumption-free way to ask whether the second stream is walls and the surface mesh is floors:
/// if one set concentrates on an axis and the other concentrates perpendicular to it, that is the
/// answer regardless of which axis the format calls up.
/// </remarks>
public sealed record Pm4FaceOrientationSummary(
    string Name,
    long FacesMeasured,
    long DominantX,
    long DominantY,
    long DominantZ,
    double MeanAbsNormalX,
    double MeanAbsNormalY,
    double MeanAbsNormalZ,
    long NearAxisAligned,
    long NearPerpendicularToDominantAxis);

/// <summary>
/// Whether the path-vertex stream and the mesh-vertex stream meet in space. If the vertical
/// faces terminate on the walkable mesh's own vertices, the two streams close a volume rather
/// than merely coexisting in one coordinate frame.
/// </summary>
public sealed record Pm4StreamCoincidenceSummary(
    long MspvPointsTested,
    long MspvPointsCoincidentWithMsvt,
    double CoincidentFraction,
    float Epsilon,
    long MsvtPointsTested,
    long MsvtPointsCoincidentWithMspv,
    double MsvtCoincidentFraction);

public sealed record Pm4ConnectiveGeometryReport(
    string InputDirectory,
    int FileCount,
    int NonEmptyFileCount,
    Pm4MslkWindowPopulation WindowPopulation,
    IReadOnlyList<Pm4WindowSizeBucket> SizeHistogram,
    Pm4WindowTopologyEvidence Topology,
    Pm4FaceOrientationSummary PathWindowOrientation,
    Pm4FaceOrientationSummary SurfaceNormalOrientation,
    Pm4StreamCoincidenceSummary StreamCoincidence,
    IReadOnlyList<Pm4WindowFamilySummary> Families,
    Pm4MscnLinkageSummary MscnLinkage,
    IReadOnlyList<string> Notes);

/// <summary>Per-tile horizontal bounds and how far geometry spills past each tile edge.</summary>
public sealed record Pm4TileBoundsRecord(
    string FileName,
    int TileX,
    int TileY,
    int VertexCount,
    int VerticesOutside,
    double OutsideFraction,
    float MinX,
    float MaxX,
    float MinY,
    float MaxY,
    float MinZ,
    float MaxZ,
    float SpillNegX,
    float SpillPosX,
    float SpillNegZ,
    float SpillPosZ);

/// <summary>Corpus totals of tile-boundary spill, kept per side.</summary>
public sealed record Pm4BoundsSideSummary(
    double TotalNegX,
    double TotalPosX,
    double TotalNegZ,
    double TotalPosZ,
    int TilesNegX,
    int TilesPosX,
    int TilesNegZ,
    int TilesPosZ);

public sealed record Pm4BoundsAuditReport(
    string InputDirectory,
    int FilesWithGeometry,
    int FilesOverflowing,
    long VerticesTotal,
    long VerticesOutside,
    double OutsideFraction,
    Pm4BoundsSideSummary SideSummary,
    IReadOnlyList<Pm4TileBoundsRecord> WorstTiles,
    IReadOnlyList<Pm4TileBoundsRecord> AllTiles,
    IReadOnlyList<string> Notes);

/// <summary>Bound-test fit of an MPRR field against one chunk domain.</summary>
public sealed record Pm4MprrDomainFit(string Domain, long Fits, long Misses, double FitFraction);

/// <summary>
/// How often the sentinel-delimited MPRR run count equals a chunk's entry count.
/// A high match rate identifies MPRR's owner structurally, without decoding any value.
/// </summary>
public sealed record Pm4MprrRunCountMatch(string Domain, int FilesMatching, int FilesTotal, double MatchFraction);

public sealed record Pm4MprrReport(
    string InputDirectory,
    int FilesWithMprr,
    long TotalEntries,
    long SentinelEntries,
    long NonSentinelEntries,
    long TotalRuns,
    double Value1WithinRunLengthFraction,
    double Value1WithinRunIndexFraction,
    IReadOnlyList<Pm4MprrDomainFit> Value1DomainFits,
    IReadOnlyList<Pm4MprrDomainFit> Value2DomainFits,
    IReadOnlyList<Pm4MprrRunCountMatch> RunCountMatches,
    IReadOnlyList<Pm4ValueFrequency> RunLengthHistogram,
    IReadOnlyList<string> Notes);
