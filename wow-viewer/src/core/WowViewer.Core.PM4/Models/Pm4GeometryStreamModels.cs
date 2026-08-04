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

/// <summary>One CK24 object, the frame the placement fitter resolved for it, and where that puts it.</summary>
public sealed record Pm4RegionFrameObjectRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    uint Ck24,
    int SurfaceCount,
    string ResolvedFrame,
    bool MatchesCanonicalFrame,
    int CanonicalTileX,
    int CanonicalTileY,
    int ResolvedTileX,
    int ResolvedTileY,
    int TileOffsetX,
    int TileOffsetY,
    float YawCorrectionDegrees);

/// <summary>Per-file bounds in ADT placement space, plus agreement with real ADT placements.</summary>
public sealed record Pm4RegionFrameFileRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    int VertexCount,
    float PlacementMinX,
    float PlacementMaxX,
    float PlacementMinY,
    float PlacementMaxY,
    bool RawBandMatchesFileName,
    int ObjectCount,
    int ObjectsOffCanonicalFrame,
    int ReferencePlacements,
    int ReferencePlacementsInside,
    double ReferenceAgreement);

/// <summary>How many objects resolved to one frame token.</summary>
public sealed record Pm4FrameFamilyCount(string Frame, int ObjectCount);

/// <summary>How many objects the resolved frame displaced by one whole-tile offset.</summary>
public sealed record Pm4TileOffsetFamilyCount(int OffsetX, int OffsetY, int ObjectCount);

/// <summary>Everything one MSHD.Field04 region contributes, aggregated over its files.</summary>
public sealed record Pm4RegionFrameSummary(
    uint RegionId,
    bool IsSharedBucket,
    bool IsEmptyStubRegion,
    int FileCount,
    int VertexCount,
    IReadOnlyList<string> Files,
    int ObjectCount,
    IReadOnlyList<Pm4FrameFamilyCount> Frames,
    bool IsFrameHomogeneous,
    IReadOnlyList<Pm4TileOffsetFamilyCount> TileOffsets,
    bool IsTileOffsetHomogeneous,
    int FilesOffRawBand,
    int ReferencePlacements,
    int ReferencePlacementsInside,
    double ReferenceAgreement);

public sealed record Pm4RegionFrameAuditReport(
    string InputDirectory,
    int FilesWithGeometry,
    int ObjectCount,
    int DistinctRegionCount,
    int MultiFileRegionCount,
    int MultiFileRegionsWithMixedFrames,
    int ObjectsOnCanonicalFrame,
    int ObjectsOffCanonicalFrame,
    int FilesOffRawBand,
    IReadOnlyList<Pm4FrameFamilyCount> CorpusFrames,
    IReadOnlyList<Pm4TileOffsetFamilyCount> CorpusTileOffsets,
    long ReferencePlacements,
    long ReferencePlacementsInside,
    double ReferenceAgreement,
    IReadOnlyList<Pm4RegionFrameSummary> Regions,
    IReadOnlyList<Pm4RegionFrameFileRecord> Files,
    IReadOnlyList<Pm4RegionFrameObjectRecord> Objects,
    IReadOnlyList<string> Notes);

/// <summary>A WMO placement's world bounding box in ADT placement space, supplied by the caller.</summary>
public readonly record struct Pm4PlacementBox(
    float MinX,
    float MinY,
    float MaxX,
    float MaxY,
    string ModelPath,
    int UniqueId);

/// <summary>The yaw test's outcome for one object, including whether its box could see a rotation.</summary>
public readonly record struct Pm4YawDecision(
    double InsideCanonical,
    double InsideYawOnly,
    double InsideControl45,
    bool HasDiscriminatingPower,
    string Verdict);

/// <summary>One object scored against the WMO box it stands in, with and without the yaw correction.</summary>
public sealed record Pm4YawEvidenceObjectRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    uint Ck24,
    int VertexCount,
    string ModelPath,
    int UniqueId,
    float YawCorrectionDegrees,
    double InsideCanonical,
    double InsideYawOnly,
    double InsideResolved,
    double InsideControl45,
    bool HasDiscriminatingPower,
    string Verdict);

public sealed record Pm4YawEvidenceReport(
    string InputDirectory,
    int FilesScored,
    int ObjectsSeen,
    int ObjectsMatched,
    int ObjectsUnmatched,
    int ObjectsWithYaw,
    int ObjectsDecidable,
    int ObjectsWithoutPower,
    int YawHelps,
    int YawHurts,
    int Ties,
    double MeanInsideCanonical,
    double MeanInsideYawOnly,
    double MeanInsideResolved,
    double MeanInsideControl45,
    string Verdict,
    IReadOnlyList<Pm4YawEvidenceObjectRecord> WorstObjects,
    IReadOnlyList<string> Notes);

/// <summary>An ADT doodad position in placement space, with the model it places.</summary>
public readonly record struct Pm4NamedPoint(float X, float Y, float Z, string ModelPath, int UniqueId);

/// <summary>One tile's ADT placements, split by asset class because the two score differently.</summary>
public sealed record Pm4TilePlacements(
    IReadOnlyList<Pm4NamedPoint> DoodadPositions,
    IReadOnlyList<Pm4PlacementBox> WorldModelBoxes);

/// <summary>One CK24 object scored against both asset classes.</summary>
public sealed record Pm4DoodadSplitObjectRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    uint Ck24,
    bool IsZeroBucket,
    int SurfaceCount,
    int VertexCount,
    float NearestDoodadDistance,
    bool SitsOnDoodad,
    string NearestDoodadPath,
    bool InsideWorldModel,
    string WorldModelPath,
    int DistinctGroupObjectIds,
    int DistinctGroupKeys,
    int AnchorOnlyLinks);

/// <summary>Per-tile counts, used to screen candidate per-doodad identity fields.</summary>
public sealed record Pm4DoodadSplitTileRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    int DoodadPlacements,
    int WorldModelPlacements,
    int ZeroBucketObjects,
    int ZeroBucketOnDoodad,
    int NonZeroObjects,
    int NonZeroInWorldModel,
    int MslkCount,
    int AnchorOnlyLinks,
    int MprlCount,
    int DistinctGroupObjectIds);

/// <summary>How closely a candidate field's cardinality tracks the tile's doodad count.</summary>
public sealed record Pm4DoodadSeparatorFit(
    string Field,
    int TilesTested,
    double MeanRatioToDoodadCount,
    double MedianRatioToDoodadCount,
    int TilesMatchingExactly);

/// <summary>
/// Whether the count of keyed (non-zero CK24) objects tracks the tile's WMO placement count.
/// </summary>
/// <remarks>
/// This is the falsifiable form of "a non-zero CK24 is a WMO instance". Counting beats both spatial
/// tests: proximity and containment can be satisfied by coincidence, but a tile with no WMOs that
/// nonetheless carries keyed objects refutes the claim outright.
/// </remarks>
public sealed record Pm4Ck24WmoCorrespondence(
    int TilesTested,
    int WmoFreeTiles,
    int WmoFreeTilesWithKeyedObjects,
    int TilesWithExactCountMatch,
    int TilesWithinOne,
    int TilesWithAnyZeroBucket,
    int TilesWithExactlyOneZeroBucket,
    long TotalKeyedObjects,
    long TotalWorldModelPlacements);

public sealed record Pm4DoodadSplitReport(
    string InputDirectory,
    int TilesScored,
    int ObjectsScored,
    int ZeroBucketObjects,
    int NonZeroObjects,
    double ZeroBucketOnDoodadFraction,
    double ZeroBucketInWorldModelFraction,
    double NonZeroOnDoodadFraction,
    double NonZeroInWorldModelFraction,
    Pm4Ck24WmoCorrespondence WmoCorrespondence,
    string Verdict,
    IReadOnlyList<Pm4DoodadSeparatorFit> SeparatorFits,
    IReadOnlyList<Pm4DoodadSplitTileRecord> TopTiles,
    IReadOnlyList<Pm4DoodadSplitObjectRecord> MatchedDoodadSamples,
    IReadOnlyList<string> Notes);

/// <summary>One spatially connected component of the CK24 0 remainder.</summary>
public sealed record Pm4ComponentRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    int ComponentIndex,
    int SurfaceCount,
    int VertexCount,
    float ExtentX,
    float ExtentY,
    float ExtentZ,
    float NearestDoodadDistance,
    bool LandsOnDoodad,
    string NearestDoodadPath,
    int DistinctGroupObjectIds,
    int DistinctLinkIds,
    int DistinctTypeFlags,
    int DistinctGroupKeys,
    int DistinctAttributeMasks,
    int AnchorOnlyLinks,
    uint SoleGroupObjectId,
    uint SoleLinkId,
    int SoleTypeFlags,
    int SoleGroupKey,
    int SoleAttributeMask);

public sealed record Pm4ComponentTileRecord(
    string FileName,
    int TileFirst,
    int TileSecond,
    uint RegionId,
    int ZeroBucketSurfaces,
    int ComponentCount,
    int DoodadPlacements,
    int ComponentsOnDoodad);

/// <summary>
/// How well a field reproduces the geometric components.
/// </summary>
/// <remarks>
/// <see cref="Purity"/> alone is not evidence — a constant field is perfectly pure and identifies
/// nothing. It has to be read next to how often the value is reused by another component.
/// </remarks>
public sealed record Pm4FieldSeparatorScore(
    string Field,
    int PureComponents,
    double Purity,
    double AbsentFraction,
    int DistinctComponents,
    double Distinctness,
    int DistinctValuesPerTileMedian);

public sealed record Pm4ComponentIdentityReport(
    string InputDirectory,
    int TilesScored,
    int ComponentCount,
    int ComponentsOnDoodad,
    double ComponentsOnDoodadFraction,
    double ComponentsPerDoodadPlacement,
    int PureGroupObjectIdComponents,
    int ReusedGroupObjectIdComponents,
    string Verdict,
    IReadOnlyList<Pm4FieldSeparatorScore> SeparatorScores,
    IReadOnlyList<Pm4ComponentTileRecord> TopTiles,
    IReadOnlyList<Pm4ComponentRecord> ClosestMatches,
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
