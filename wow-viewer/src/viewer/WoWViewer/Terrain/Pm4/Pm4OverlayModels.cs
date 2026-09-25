using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Population;
using WoWViewer.Rendering;
using WoWViewer.Audio;
using WowViewer.Core.Audio;
using WowViewer.Core.Maps;
using Silk.NET.OpenGL;
using CorePm4AxisConvention = WowViewer.Core.PM4.Models.Pm4AxisConvention;
using CorePm4CorrelationCandidateScore = WowViewer.Core.PM4.Models.Pm4CorrelationCandidateScore;
using CorePm4CorrelationMetrics = WowViewer.Core.PM4.Models.Pm4CorrelationMetrics;
using CorePm4CorrelationObjectDescriptor = WowViewer.Core.PM4.Models.Pm4CorrelationObjectDescriptor;
using CorePm4CorrelationGeometryInput = WowViewer.Core.PM4.Models.Pm4CorrelationGeometryInput;
using CorePm4CorrelationObjectInput = WowViewer.Core.PM4.Models.Pm4CorrelationObjectInput;
using CorePm4CorrelationObjectState = WowViewer.Core.PM4.Models.Pm4CorrelationObjectState;
using CorePm4CorrelationMath = WowViewer.Core.PM4.Services.Pm4CorrelationMath;
using CorePm4ConnectorKey = WowViewer.Core.PM4.Models.Pm4ConnectorKey;
using CorePm4ConnectorMergeCandidate = WowViewer.Core.PM4.Models.Pm4ConnectorMergeCandidate;
using CorePm4CoordinateMode = WowViewer.Core.PM4.Models.Pm4CoordinateMode;
using CorePm4GeometryLineSegment = WowViewer.Core.PM4.Models.Pm4GeometryLineSegment;
using CorePm4GeometryTriangle = WowViewer.Core.PM4.Models.Pm4GeometryTriangle;
using CorePm4LinkedPositionRefSummary = WowViewer.Core.PM4.Models.Pm4LinkedPositionRefSummary;
using CorePm4MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using CorePm4MshdGroupingService = WowViewer.Core.PM4.Services.Pm4MshdGroupingService;
using CorePm4MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using CorePm4MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using CorePm4CoordinateModeResolution = WowViewer.Core.PM4.Models.Pm4CoordinateModeResolution;
using CorePm4ObjectGroupKey = WowViewer.Core.PM4.Models.Pm4ObjectGroupKey;
using CorePm4CachedTile = WowViewer.Core.PM4.Caching.Pm4CachedTile;
using CorePm4CachedObject = WowViewer.Core.PM4.Caching.Pm4CachedObject;
using CorePm4CachedConnectorKey = WowViewer.Core.PM4.Caching.Pm4CachedConnectorKey;
using CorePm4CachedLineSegment = WowViewer.Core.PM4.Caching.Pm4CachedLineSegment;
using CorePm4CachedTriangle = WowViewer.Core.PM4.Caching.Pm4CachedTriangle;
using CorePm4PerFileCacheEntry = WowViewer.Core.PM4.Caching.Pm4PerFileCacheEntry;
using CorePm4PerFileCache = WowViewer.Core.PM4.Caching.Pm4PerFileCache;
using CorePm4PerFileCacheService = WowViewer.Core.PM4.Caching.Pm4PerFileCacheService;
using CorePm4PlacementContract = WowViewer.Core.PM4.Services.Pm4PlacementContract;
using CorePm4PlacementMath = WowViewer.Core.PM4.Services.Pm4PlacementMath;
using CorePm4PlacementSolution = WowViewer.Core.PM4.Models.Pm4PlacementSolution;
using Pm4PlanarTransform = WowViewer.Core.PM4.Models.Pm4PlanarTransform;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using CorePm4DecodeAuditReport = WowViewer.Core.PM4.Models.Pm4DecodeAuditReport;
using CorePm4ExplorationSnapshot = WowViewer.Core.PM4.Models.Pm4ExplorationSnapshot;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using CorePm4ObjectHypothesis = WowViewer.Core.PM4.Models.Pm4ObjectHypothesis;
using MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using Pm4VersionFormatter = WowViewer.Core.PM4.Services.Pm4VersionFormatter;
using MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using Pm4File = WowViewer.Core.PM4.Research.Pm4ResearchDocument;
using CorePm4ReferenceAudit = WowViewer.Core.PM4.Models.Pm4ReferenceAudit;
using CorePm4ResearchAuditAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchAuditAnalyzer;
using CorePm4ResearchHierarchyAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchHierarchyAnalyzer;
using CorePm4ResearchSnapshotBuilder = WowViewer.Core.PM4.Research.Pm4ResearchSnapshotBuilder;
using CorePm4TileObjectHypothesisReport = WowViewer.Core.PM4.Models.Pm4TileObjectHypothesisReport;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WorldFramePassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldFramePassCoordinator;
using WorldFramePassOptions = WowViewer.Core.Runtime.World.Passes.WorldFramePassOptions;
using WorldFramePasses = WowViewer.Core.Runtime.World.Passes.WorldFramePasses;
using WorldObjectPassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldObjectPassCoordinator;
using WorldObjectPassFrame = WowViewer.Core.Runtime.World.Passes.WorldObjectPassFrame;
using WorldModelBatchGate = WowViewer.Core.Runtime.World.Passes.WorldModelBatchGate;
using WorldModelRenderPath = WowViewer.Core.Runtime.World.Passes.WorldModelRenderPath;
using WorldModelSubmissionOutcome = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionOutcome;
using WorldModelSubmissionTally = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionTally;
using VisibleMdxInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry;
using VisibleWmoInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.World;

namespace WoWViewer.Terrain;

/// <summary>
/// What the PM4 overlay colours objects by.
/// </summary>
/// <remarks>
/// Four modes were removed on 2026-08-24 because each grouped by a measured misreading:
/// <c>Ck24Type</c> is the EXPONENT BAND of the placement-Z float, <c>Ck24ObjectId</c> its MANTISSA
/// bytes, <c>Ck24TypeVsTypeFlags</c> a cross-tab of that exponent band, and <c>AttributeMask</c> is
/// the LENGTH of a surface's MSLK window - so it coloured objects by how many neighbours they have.
/// None of them named a property of an object. See docs/wowdev-wiki/pm4-pd4-draft.md.
/// </remarks>
public enum Pm4OverlayColorMode
{
    /// <summary>MSUR._0x1C as a float: the producing placement's Z. Groups WMO objects; every doodad surface is 0.</summary>
    PlacementZ,

    /// <summary>Placed (carries a placement height) versus doodad collision (does not).</summary>
    Population,

    Tile,
    MshdRegionId,

    /// <summary>Surfaces per object - a size bucket, not an identity.</summary>
    SurfaceCount,

    /// <summary>MSUR._0x00. Unmeasured: 100% pure across only 9 corpus values, so it separates nothing.</summary>
    GroupKey,

    Height,

    /// <summary>MSLK._0x00. Partial: observed buckets, not corpus-closed.</summary>
    TypeFlags,
}

public readonly struct Pm4ObjectDebugInfo
{
    public Pm4ObjectDebugInfo(
        uint ck24,
        byte ck24Type,
        ushort ck24ObjectId,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        Pm4LinkedPositionRefSummary linkedPositionRefSummary,
        int tileX,
        int tileY,
        uint mshdField00,
        uint mshdRegionId,
        uint mshdField08,
        int surfaceCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMscnRefIndex,
        float averageSurfaceHeight,
        Vector3 boundsMin,
        Vector3 boundsMax,
        Vector3 center,
        float nearestPositionRefDistance,
        bool swapPlanarAxes,
        bool invertU,
        bool invertV,
        bool invertsWinding,
        uint distinctTypeFlags = 0)
    {
        Ck24 = ck24;
        Ck24Type = ck24Type;
        Ck24ObjectId = ck24ObjectId;
        ObjectPartId = objectPartId;
        LinkGroupObjectId = linkGroupObjectId;
        LinkedPositionRefCount = linkedPositionRefCount;
        LinkedPositionRefSummary = linkedPositionRefSummary;
        TileX = tileX;
        TileY = tileY;
        MshdField00 = mshdField00;
        MshdRegionId = mshdRegionId;
        DistinctTypeFlags = distinctTypeFlags;
        MshdField08 = mshdField08;
        SurfaceCount = surfaceCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMscnRefIndex = dominantMscnRefIndex;
        AverageSurfaceHeight = averageSurfaceHeight;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
        Center = center;
        NearestPositionRefDistance = nearestPositionRefDistance;
        SwapPlanarAxes = swapPlanarAxes;
        InvertU = invertU;
        InvertV = invertV;
        InvertsWinding = invertsWinding;
    }

    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId { get; }
    public int ObjectPartId { get; }
    public uint LinkGroupObjectId { get; }
    public int LinkedPositionRefCount { get; }
    public Pm4LinkedPositionRefSummary LinkedPositionRefSummary { get; }
    public int TileX { get; }
    public int TileY { get; }
    public uint MshdField00 { get; }
    public uint MshdRegionId { get; }
    public uint MshdField08 { get; }
    public uint DistinctTypeFlags { get; }
    public int SurfaceCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMscnRefIndex { get; }
    public float AverageSurfaceHeight { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public Vector3 Center { get; }
    public float NearestPositionRefDistance { get; }
    public bool SwapPlanarAxes { get; }
    public bool InvertU { get; }
    public bool InvertV { get; }
    public bool InvertsWinding { get; }
}

/// <summary>
/// Summary of MSLK linking statistics across all loaded PM4 files.
/// Produced by <see cref="WorldScene.GetPm4MslkLinkingStats"/>.
/// </summary>
public readonly struct Pm4MslkLinkingStats
{
    public Pm4MslkLinkingStats(
        int totalFiles,
        int totalMslkEntries,
        int anchorOnlyLinks,
        int pathWindowLinks,
        int totalComponents,
        int componentsWithLinks,
        int componentsWithoutLinks,
        int refIndexMismatches)
    {
        TotalFiles = totalFiles;
        TotalMslkEntries = totalMslkEntries;
        AnchorOnlyLinks = anchorOnlyLinks;
        PathWindowLinks = pathWindowLinks;
        TotalComponents = totalComponents;
        ComponentsWithLinks = componentsWithLinks;
        ComponentsWithoutLinks = componentsWithoutLinks;
        RefIndexMismatches = refIndexMismatches;
    }

    public int TotalFiles { get; }
    public int TotalMslkEntries { get; }
    public int AnchorOnlyLinks { get; }
    public int PathWindowLinks { get; }
    public int TotalComponents { get; }
    public int ComponentsWithLinks { get; }
    public int ComponentsWithoutLinks { get; }
    public int RefIndexMismatches { get; }
}

public readonly struct Pm4LinkedPositionRefSummary
{
    public Pm4LinkedPositionRefSummary(
        int totalCount,
        int normalCount,
        int terminatorCount,
        int floorMin,
        int floorMax,
        float headingMinDegrees,
        float headingMaxDegrees,
        float headingMeanDegrees)
    {
        TotalCount = totalCount;
        NormalCount = normalCount;
        TerminatorCount = terminatorCount;
        FloorMin = floorMin;
        FloorMax = floorMax;
        HeadingMinDegrees = headingMinDegrees;
        HeadingMaxDegrees = headingMaxDegrees;
        HeadingMeanDegrees = headingMeanDegrees;
    }

    public int TotalCount { get; }
    public int NormalCount { get; }
    public int TerminatorCount { get; }
    public int FloorMin { get; }
    public int FloorMax { get; }
    public float HeadingMinDegrees { get; }
    public float HeadingMaxDegrees { get; }
    public float HeadingMeanDegrees { get; }
    public bool HasNormalHeadings => NormalCount > 0 && !float.IsNaN(HeadingMeanDegrees);
}

public readonly struct Pm4ResearchHypothesisMatch
{
    public Pm4ResearchHypothesisMatch(
        string family,
        int familyObjectIndex,
        int surfaceCount,
        int totalIndexCount,
        int mscnRefCount,
        int groupKeyCount,
        int linkGroupCount,
        uint dominantLinkGroupObjectId,
        int linkedMprlRefCount,
        int linkedMprlInBoundsCount,
        CorePm4CoordinateMode coordinateMode,
        Pm4PlanarTransform planarTransform,
        float frameYawDegrees,
        float? mprlHeadingMeanDegrees,
        float? headingDeltaDegrees,
        float similarityScore)
    {
        Family = family;
        FamilyObjectIndex = familyObjectIndex;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        MscnRefCount = mscnRefCount;
        GroupKeyCount = groupKeyCount;
        LinkGroupCount = linkGroupCount;
        DominantLinkGroupObjectId = dominantLinkGroupObjectId;
        LinkedMprlRefCount = linkedMprlRefCount;
        LinkedMprlInBoundsCount = linkedMprlInBoundsCount;
        CoordinateMode = coordinateMode;
        PlanarTransform = planarTransform;
        FrameYawDegrees = frameYawDegrees;
        MprlHeadingMeanDegrees = mprlHeadingMeanDegrees;
        HeadingDeltaDegrees = headingDeltaDegrees;
        SimilarityScore = similarityScore;
    }

    public string Family { get; }
    public int FamilyObjectIndex { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public int MscnRefCount { get; }
    public int GroupKeyCount { get; }
    public int LinkGroupCount { get; }
    public uint DominantLinkGroupObjectId { get; }
    public int LinkedMprlRefCount { get; }
    public int LinkedMprlInBoundsCount { get; }
    public CorePm4CoordinateMode CoordinateMode { get; }
    public Pm4PlanarTransform PlanarTransform { get; }
    public float FrameYawDegrees { get; }
    public float? MprlHeadingMeanDegrees { get; }
    public float? HeadingDeltaDegrees { get; }
    public float SimilarityScore { get; }
}

public readonly struct Pm4SelectedObjectResearchInfo
{
    public Pm4SelectedObjectResearchInfo(
        string sourcePath,
        uint version,
        int mslkCount,
        int msurCount,
        int mscnCount,
        int mprlCount,
        int invalidRefIndexCount,
        int totalHypothesisCount,
        int matchingCk24HypothesisCount,
        int diagnosticCount,
        IReadOnlyList<string> diagnostics,
        IReadOnlyList<Pm4ResearchHypothesisMatch> topMatches,
        string? mshdRawFields = null,
        IReadOnlyList<string>? mslkRawEntries = null)
    {
        SourcePath = sourcePath;
        Version = version;
        MslkCount = mslkCount;
        MsurCount = msurCount;
        MscnCount = mscnCount;
        MprlCount = mprlCount;
        InvalidRefIndexCount = invalidRefIndexCount;
        TotalHypothesisCount = totalHypothesisCount;
        MatchingCk24HypothesisCount = matchingCk24HypothesisCount;
        DiagnosticCount = diagnosticCount;
        Diagnostics = diagnostics;
        TopMatches = topMatches;
        MshdRawFields = mshdRawFields;
        MslkRawEntries = mslkRawEntries ?? Array.Empty<string>();
    }

    public string SourcePath { get; }
    public uint Version { get; }
    public int MslkCount { get; }
    public int MsurCount { get; }
    public int MscnCount { get; }
    public int MprlCount { get; }
    public int InvalidRefIndexCount { get; }
    public int TotalHypothesisCount { get; }
    public int MatchingCk24HypothesisCount { get; }
    public int DiagnosticCount { get; }
    public IReadOnlyList<string> Diagnostics { get; }
    public IReadOnlyList<Pm4ResearchHypothesisMatch> TopMatches { get; }
    public string? MshdRawFields { get; }
    public IReadOnlyList<string> MslkRawEntries { get; }
}

public readonly struct Pm4ColorLegendEntry
{
    public Pm4ColorLegendEntry(string label, Vector3 color, int objectCount, bool isSelected)
    {
        Label = label;
        Color = color;
        ObjectCount = objectCount;
        IsSelected = isSelected;
    }

    public string Label { get; }
    public Vector3 Color { get; }
    public int ObjectCount { get; }
    public bool IsSelected { get; }
}

public readonly struct Pm4ColorLegendInfo
{
    public Pm4ColorLegendInfo(
        Pm4OverlayColorMode mode,
        bool isContinuous,
        string description,
        int totalEntryCount,
        IReadOnlyList<Pm4ColorLegendEntry> entries)
    {
        Mode = mode;
        IsContinuous = isContinuous;
        Description = description;
        TotalEntryCount = totalEntryCount;
        Entries = entries;
    }

    public Pm4OverlayColorMode Mode { get; }
    public bool IsContinuous { get; }
    public string Description { get; }
    public int TotalEntryCount { get; }
    public IReadOnlyList<Pm4ColorLegendEntry> Entries { get; }
    public bool IsTruncated => Entries.Count < TotalEntryCount;
    public int HiddenEntryCount => Math.Max(0, TotalEntryCount - Entries.Count);
}

public readonly struct Pm4VisibleTypeBucket
{
    public Pm4VisibleTypeBucket(byte ck24Type, int objectCount)
    {
        Ck24Type = ck24Type;
        ObjectCount = objectCount;
    }

    public byte Ck24Type { get; }
    public int ObjectCount { get; }
}

public readonly struct Pm4VisibleRegionSummary
{
    public Pm4VisibleRegionSummary(
        uint regionId,
        int objectCount,
        int tileCount,
        int uniqueCk24Count,
        int uniqueLinkGroupCount,
        float averageCenterHeight,
        bool isSelectedRegion,
        IReadOnlyList<Pm4VisibleTypeBucket> typeBuckets)
    {
        RegionId = regionId;
        ObjectCount = objectCount;
        TileCount = tileCount;
        UniqueCk24Count = uniqueCk24Count;
        UniqueLinkGroupCount = uniqueLinkGroupCount;
        AverageCenterHeight = averageCenterHeight;
        IsSelectedRegion = isSelectedRegion;
        TypeBuckets = typeBuckets;
    }

    public uint RegionId { get; }
    public int ObjectCount { get; }
    public int TileCount { get; }
    public int UniqueCk24Count { get; }
    public int UniqueLinkGroupCount { get; }
    public float AverageCenterHeight { get; }
    public bool IsSelectedRegion { get; }
    public IReadOnlyList<Pm4VisibleTypeBucket> TypeBuckets { get; }
}

public readonly struct Pm4VisibleOverlaySummaryInfo
{
    public Pm4VisibleOverlaySummaryInfo(
        int visibleObjectCount,
        int visibleTileCount,
        int regionCount,
        uint? selectedRegionId,
        IReadOnlyList<Pm4VisibleRegionSummary> regions)
    {
        VisibleObjectCount = visibleObjectCount;
        VisibleTileCount = visibleTileCount;
        RegionCount = regionCount;
        SelectedRegionId = selectedRegionId;
        Regions = regions;
    }

    public int VisibleObjectCount { get; }
    public int VisibleTileCount { get; }
    public int RegionCount { get; }
    public uint? SelectedRegionId { get; }
    public IReadOnlyList<Pm4VisibleRegionSummary> Regions { get; }
}

public readonly struct Pm4RegionPeerSummary
{
    public Pm4RegionPeerSummary(
        (int tileX, int tileY, uint ck24, int objectPart) objectKey,
        byte ck24Type,
        ushort ck24ObjectId,
        int surfaceCount,
        uint linkGroupObjectId,
        uint dominantMscnRefIndex,
        Vector3 center,
        bool isSelected,
        bool sameCk24,
        bool sameLinkGroup,
        bool sameMscnRefIndex)
    {
        ObjectKey = objectKey;
        Ck24Type = ck24Type;
        Ck24ObjectId = ck24ObjectId;
        SurfaceCount = surfaceCount;
        LinkGroupObjectId = linkGroupObjectId;
        DominantMscnRefIndex = dominantMscnRefIndex;
        Center = center;
        IsSelected = isSelected;
        SameCk24 = sameCk24;
        SameLinkGroup = sameLinkGroup;
        SameMscnRefIndex = sameMscnRefIndex;
    }

    public (int tileX, int tileY, uint ck24, int objectPart) ObjectKey { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId { get; }
    public int SurfaceCount { get; }
    public uint LinkGroupObjectId { get; }
    public uint DominantMscnRefIndex { get; }
    public Vector3 Center { get; }
    public bool IsSelected { get; }
    public bool SameCk24 { get; }
    public bool SameLinkGroup { get; }
    public bool SameMscnRefIndex { get; }
}

public readonly struct Pm4SelectedObjectRegionInfo
{
    public Pm4SelectedObjectRegionInfo(
        uint regionId,
        int visibleObjectCount,
        int visibleTileCount,
        int uniqueCk24Count,
        int uniqueLinkGroupCount,
        int uniqueMscnRefCount,
        int sameCk24Count,
        int sameLinkGroupCount,
        int sameMscnRefCount,
        float averageSurfaceCount,
        float averageCenterHeight,
        IReadOnlyList<Pm4VisibleTypeBucket> typeBuckets,
        IReadOnlyList<Pm4RegionPeerSummary> peers)
    {
        RegionId = regionId;
        VisibleObjectCount = visibleObjectCount;
        VisibleTileCount = visibleTileCount;
        UniqueCk24Count = uniqueCk24Count;
        UniqueLinkGroupCount = uniqueLinkGroupCount;
        UniqueMscnRefCount = uniqueMscnRefCount;
        SameCk24Count = sameCk24Count;
        SameLinkGroupCount = sameLinkGroupCount;
        SameMscnRefCount = sameMscnRefCount;
        AverageSurfaceCount = averageSurfaceCount;
        AverageCenterHeight = averageCenterHeight;
        TypeBuckets = typeBuckets;
        Peers = peers;
    }

    public uint RegionId { get; }
    public int VisibleObjectCount { get; }
    public int VisibleTileCount { get; }
    public int UniqueCk24Count { get; }
    public int UniqueLinkGroupCount { get; }
    public int UniqueMscnRefCount { get; }
    public int SameCk24Count { get; }
    public int SameLinkGroupCount { get; }
    public int SameMscnRefCount { get; }
    public float AverageSurfaceCount { get; }
    public float AverageCenterHeight { get; }
    public IReadOnlyList<Pm4VisibleTypeBucket> TypeBuckets { get; }
    public IReadOnlyList<Pm4RegionPeerSummary> Peers { get; }
}

public readonly struct Pm4SelectedObjectGraphPartNode
{
    public Pm4SelectedObjectGraphPartNode(
        int tileX,
        int tileY,
        int objectPartId,
        int surfaceCount,
        int totalIndexCount,
        int lineCount,
        int triangleCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMscnRefIndex,
        bool isSelected)
    {
        TileX = tileX;
        TileY = tileY;
        ObjectPartId = objectPartId;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        LineCount = lineCount;
        TriangleCount = triangleCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMscnRefIndex = dominantMscnRefIndex;
        IsSelected = isSelected;
    }

    public int TileX { get; }
    public int TileY { get; }
    public int ObjectPartId { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public int LineCount { get; }
    public int TriangleCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMscnRefIndex { get; }
    public bool IsSelected { get; }
}

public readonly struct Pm4SelectedObjectGraphMscnRefNode
{
    public Pm4SelectedObjectGraphMscnRefNode(
        uint mscnRefIndex,
        int partCount,
        int surfaceCount,
        int totalIndexCount,
        IReadOnlyList<byte> attributeMasks,
        IReadOnlyList<byte> groupKeys,
        IReadOnlyList<Pm4SelectedObjectGraphPartNode> parts)
    {
        MscnRefIndex = mscnRefIndex;
        PartCount = partCount;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        AttributeMasks = attributeMasks;
        GroupKeys = groupKeys;
        Parts = parts;
    }

    public uint MscnRefIndex { get; }
    public int PartCount { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public IReadOnlyList<byte> AttributeMasks { get; }
    public IReadOnlyList<byte> GroupKeys { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphPartNode> Parts { get; }
}

public readonly struct Pm4SelectedObjectGraphLinkNode
{
    public Pm4SelectedObjectGraphLinkNode(
        uint linkGroupObjectId,
        int partCount,
        int surfaceCount,
        int totalIndexCount,
        int linkedPositionRefCount,
        Pm4LinkedPositionRefSummary linkedPositionRefSummary,
        IReadOnlyList<uint> mscnRefIndices,
        IReadOnlyList<byte> attributeMasks,
        IReadOnlyList<byte> groupKeys,
        IReadOnlyList<Pm4SelectedObjectGraphMscnRefNode> mscnRefGroups)
    {
        LinkGroupObjectId = linkGroupObjectId;
        PartCount = partCount;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        LinkedPositionRefCount = linkedPositionRefCount;
        LinkedPositionRefSummary = linkedPositionRefSummary;
        MscnRefIndices = mscnRefIndices;
        AttributeMasks = attributeMasks;
        GroupKeys = groupKeys;
        MscnRefGroups = mscnRefGroups;
    }

    public uint LinkGroupObjectId { get; }
    public int PartCount { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public int LinkedPositionRefCount { get; }
    public Pm4LinkedPositionRefSummary LinkedPositionRefSummary { get; }
    public IReadOnlyList<uint> MscnRefIndices { get; }
    public IReadOnlyList<byte> AttributeMasks { get; }
    public IReadOnlyList<byte> GroupKeys { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphMscnRefNode> MscnRefGroups { get; }
}

public readonly struct Pm4SelectedObjectGraphTypeBucket
{
    public Pm4SelectedObjectGraphTypeBucket(
        byte ck24Type,
        string typeLabel,
        int linkGroupCount,
        int surfaceCount,
        IReadOnlyList<Pm4SelectedObjectGraphLinkNode> linkGroups)
    {
        Ck24Type = ck24Type;
        TypeLabel = typeLabel;
        LinkGroupCount = linkGroupCount;
        SurfaceCount = surfaceCount;
        LinkGroups = linkGroups;
    }

    public byte Ck24Type { get; }
    public string TypeLabel { get; }
    public int LinkGroupCount { get; }
    public int SurfaceCount { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphLinkNode> LinkGroups { get; }
}

public readonly struct Pm4SelectedObjectGraphInfo
{
    public Pm4SelectedObjectGraphInfo(
        int selectedTileX,
        int selectedTileY,
        uint ck24,
        byte ck24Type,
        ushort ck24ObjectId,
        int selectedObjectPartId,
        bool splitByMscnRef,
        bool splitByConnectivity,
        int tileCount,
        int linkGroupCount,
        int mscnRefGroupCount,
        int partCount,
        int surfaceCount,
        int totalIndexCount,
        int attributeMaskCount,
        int groupKeyCount,
        IReadOnlyList<Pm4SelectedObjectGraphLinkNode> linkGroups,
        IReadOnlyList<Pm4SelectedObjectGraphTypeBucket> typeBuckets)
    {
        SelectedTileX = selectedTileX;
        SelectedTileY = selectedTileY;
        Ck24 = ck24;
        Ck24Type = ck24Type;
        Ck24ObjectId = ck24ObjectId;
        SelectedObjectPartId = selectedObjectPartId;
        SplitByMscnRef = splitByMscnRef;
        SplitByConnectivity = splitByConnectivity;
        TileCount = tileCount;
        LinkGroupCount = linkGroupCount;
        MscnRefGroupCount = mscnRefGroupCount;
        PartCount = partCount;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        AttributeMaskCount = attributeMaskCount;
        GroupKeyCount = groupKeyCount;
        LinkGroups = linkGroups;
        TypeBuckets = typeBuckets;
    }

    public int SelectedTileX { get; }
    public int SelectedTileY { get; }
    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId { get; }
    public int SelectedObjectPartId { get; }
    public bool SplitByMscnRef { get; }
    public bool SplitByConnectivity { get; }
    public int TileCount { get; }
    public int LinkGroupCount { get; }
    public int MscnRefGroupCount { get; }
    public int PartCount { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public int AttributeMaskCount { get; }
    public int GroupKeyCount { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphLinkNode> LinkGroups { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphTypeBucket> TypeBuckets { get; }
}

internal readonly record struct Pm4ConnectorKey(int X, int Y, int Z);
