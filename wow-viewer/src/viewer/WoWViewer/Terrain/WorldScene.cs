using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Population;
using WoWViewer.Rendering;
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
using VisibleMdxInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry;
using VisibleWmoInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.SceneGraph;
using WowViewer.Core.Runtime.World.Visibility;

namespace WoWViewer.Terrain;

public enum Pm4OverlayColorMode
{
    Ck24Type,
    Ck24ObjectId,
    Ck24Key,
    Tile,
    MshdRegionId,
    GroupKey,
    AttributeMask,
    Height,
    TypeFlags,
    Ck24TypeVsTypeFlags,
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

/// <summary>
/// Combines terrain (WDT/ADT), WMO placements (MODF), and MDX placements (MDDF)
/// into a single world scene — the same way the game client renders a map.
/// 
/// Uses <see cref="WorldAssetManager"/> to ensure each model is loaded exactly once.
/// Instances are lightweight structs holding only a model key + transform.
/// </summary>
public readonly record struct TaxiActorPose(
    int RouteId,
    Vector3 Position,
    Vector3 Forward,
    float YawRadians,
    float Scale,
    string ModelPath);

public class WorldScene : ISceneRenderer
{
    private const float TaxiActorHeadingSampleWindow = 18f;
    private const float TaxiActorHeadingSmoothingHz = 8f;
    public const float TaxiActorNormalSpeedSetting = 0.10f;
    public const float TaxiActorMinSpeedSetting = 0.01f;
    public const float TaxiActorMaxSpeedSetting = 0.50f;
    private static readonly string[] TaxiActorDefaultModelCandidates =
    {
        @"Creature\Gryphon\Gryphon.mdx",
        @"Creature\FelBat\BatTaxi.mdx",
    };

    private readonly record struct SelectedSceneObjectKey(
        ObjectType ObjectType,
        int UniqueId,
        int PlacementEntryIndex,
        int TileX,
        int TileY,
        bool HasTileCoordinate,
        string ModelKey,
        Vector3 PlacementPosition);

    public static IReadOnlyList<string> DefaultTaxiActorModelPaths => TaxiActorDefaultModelCandidates;

    private static float? JsonFiniteOrNull(float value) => float.IsFinite(value) ? value : null;

    private static float DecodeRawMprlPackedAngleRadians(MprlEntry positionRef)
    {
        return positionRef.Unk04 * (2f * MathF.PI / 65536f);
    }

    private readonly GL _gl;
    private readonly TerrainManager _terrainManager;
    private readonly WorldAssetManager _assets;
    private readonly Pm4OverlayCacheService? _pm4OverlayCacheService;
    // Spec 054: on-disk per-file PM4 overlay cache. Constructed lazily
    // on first PM4 load from the same cache root the per-window cache
    // uses. Holds small per-PM4-file gzip blobs; one entry per file.
    private CorePm4PerFileCacheService? _pm4PerFileDiskCache;

    // Lightweight instance lists — just a key + transform, no renderer reference
    // These are rebuilt from _tileMdxInstances/_tileWmoInstances when tiles change
    private List<ObjectInstance> _mdxInstances = new();
    private List<ObjectInstance> _skyboxInstances = new();
    private List<ObjectInstance> _wmoInstances = new();

    // Per-tile instance storage for lazy load/unload
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileSkyboxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances = new();
    // These buckets mirror the graph's tile/chunk partition, but remain a flat collector aid.
    // They reject whole candidate lists; the existing per-instance collector stays authoritative.
    private readonly Dictionary<(int, int), List<FlatVisibilityBucket>> _tileMdxVisibilityBuckets = new();
    private readonly Dictionary<(int, int), List<FlatVisibilityBucket>> _tileWmoVisibilityBuckets = new();
    private readonly Dictionary<(int, int), (Vector3 Min, Vector3 Max)> _tileMdxBounds = new();
    private readonly Dictionary<(int, int), (Vector3 Min, Vector3 Max)> _tileWmoBounds = new();
    private readonly List<ObjectInstance> _externalMdxInstances = new();
    private readonly List<ObjectInstance> _externalSkyboxInstances = new();
    private readonly List<ObjectInstance> _externalWmoInstances = new();
    private readonly List<ObjectInstance> _taxiActorInstances = new();
    private WorldSceneGraphBuildSet? _sceneGraphBuild;
    private readonly Dictionary<string, WorldScenePortalAdapterResult> _sceneGraphPortalAdapters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, WorldScenePortalVisibilityResult> _sceneGraphPortalVisibility = new(StringComparer.Ordinal);
    private readonly List<ObjectInstance> _sceneGraphVisibleMdxInstances = new();
    private readonly List<ObjectInstance> _sceneGraphVisibleWmoInstances = new();
    private bool _sceneGraphFrameVisibilityPrepared;
    private WorldSceneTraversalDiagnostics _lastSceneGraphTraversalDiagnostics = new();
    // The hierarchical graph remains available as an explicit investigation path, but it is
    // not yet a proven replacement for the production flat visibility collectors. Real Azeroth
    // captures measured tens of milliseconds in graph traversal per object pass, so keep the
    // stable legacy path as the runtime default until the graph has a bounded cost budget.
    private bool _useHierarchicalSceneTraversal;
    private bool _instancesDirty = false;
    private readonly Dictionary<string, float> _pendingVisibleMdxLoadDistances = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, float> _pendingVisibleWmoLoadDistances = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<KeyValuePair<string, float>> _pendingVisibleMdxLoadScratch = new();
    private readonly List<KeyValuePair<string, float>> _pendingVisibleWmoLoadScratch = new();

    private sealed class FlatVisibilityBucket
    {
        public List<ObjectInstance> Instances { get; } = new();
        public Vector3 Min { get; private set; } = new(float.MaxValue);
        public Vector3 Max { get; private set; } = new(float.MinValue);
        public bool BoundsKnown { get; private set; } = true;

        public void Add(in ObjectInstance instance)
        {
            Instances.Add(instance);
            if (!instance.BoundsResolved
                || !AreFiniteOrderedBounds(instance.BoundsMin, instance.BoundsMax))
            {
                BoundsKnown = false;
                return;
            }

            Min = Vector3.Min(Min, instance.BoundsMin);
            Max = Vector3.Max(Max, instance.BoundsMax);
        }
    }

    private bool _objectsVisible = true;
    private bool _wmosVisible = true;
    private bool _doodadsVisible = true;
    private bool _objectFogEnabled = true;
    private bool _objectPathFiltersEnabled = true;
    private bool _limitHoveredAssetRange = true;
    private bool _useDynamicHoveredAssetRange = false;
    private bool _showSelectedObjectBounds = true;
    private float _hoveredAssetMaxDistance = 533.33f;
    private float _lastHoverPickFogEnd = 1500f;
    private float _objectStreamingRangeMultiplier = 0.5f;
    private float _maxVisibleMdxBoundsHeight;
    private bool _hideTerrainOccludedMdx;
    private WorldObjectVisibilityProfile _objectVisibilityProfile = WorldObjectVisibilityProfile.Performance;

    // Frustum culling
    private readonly FrustumCuller _frustumCuller = new();
    private const float DoodadCullDistance = 16000f; // Hard ceiling for very small doodads when fog allows farther visibility
    private const float DoodadCullDistanceSq = DoodadCullDistance * DoodadCullDistance;
    private const float DoodadSmallThreshold = 10f; // AABB diagonal below this = "small" (relaxed — only cull tiny objects)
    private const float FadeStartFraction = 0.80f;  // Fade begins at 80% of cull distance
    private const float WmoCullDistance = 1600f;     // Default world-object visibility should stay close to terrain fog unless explicitly widened
    private const float NoCullRadius = 512f;         // Objects within this radius are never frustum-culled
    private const float NoCullRadiusSq = NoCullRadius * NoCullRadius;
    private const float ObjectNearHoldRadius = 384f;
    private const float ObjectNearHoldRadiusSq = ObjectNearHoldRadius * ObjectNearHoldRadius;
    private const float VisionConeFrontDot = 0.15f;
    private const float VisionConeRearDot = -0.35f;
    private const float RearConeCullFraction = 0.45f;
    private const float MinOffFrustumConeFactor = 0.35f;
    private const float RearConeFadeFloor = 0.25f;
    private const float RearConeLoadPenalty = 2.5f;
    private const float HoverInfoBrushPixels = 32f;
    private const float HoverInfoMaxScreenRadius = 96f;
    private const float WireframeRevealBrushPixels = 96f;
    private const float WireframeRevealMaxScreenRadius = 220f;
    private const float MaxWorldObjectViewDistance = 20000f;
    private const float MaxWorldObjectViewDistanceSq = MaxWorldObjectViewDistance * MaxWorldObjectViewDistance;

    private readonly record struct TerrainAssetLoadPolicy(
        bool PrewarmTileAssets,
        int MaxNewMdxLoadsPerFrame,
        int MaxNewWmoLoadsPerFrame,
        int MaxDeferredLoadsPerFrame,
        double MaxDeferredLoadBudgetMs);

    private static readonly TerrainAssetLoadPolicy WmoOnlyAssetLoadPolicy = new(
        PrewarmTileAssets: false,
        MaxNewMdxLoadsPerFrame: 6,
        MaxNewWmoLoadsPerFrame: 3,
        MaxDeferredLoadsPerFrame: 2,
        MaxDeferredLoadBudgetMs: 6.0);

    private static readonly TerrainAssetLoadPolicy StreamingTerrainAssetLoadPolicy = new(
        PrewarmTileAssets: false,
        MaxNewMdxLoadsPerFrame: 12,
        MaxNewWmoLoadsPerFrame: 6,
        MaxDeferredLoadsPerFrame: 4,
        MaxDeferredLoadBudgetMs: 3.5);

    private sealed class WorldRenderFrame
    {
        public WorldVisibilityFrame Visibility { get; } = new();
        public WorldObjectPassFrame ObjectPasses { get; } = new();
        public Dictionary<string, WmoRenderer> VisibleWmoRendererCache { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, IModelRenderer> VisibleMdxRendererCache { get; } = new(StringComparer.OrdinalIgnoreCase);

        public List<VisibleWmoInstance> VisibleWmoInstances => Visibility.VisibleWmos;
        public List<VisibleMdxInstance> VisibleMdxInstances => Visibility.VisibleMdx;
        public int VisibleTaxiMdxCount
        {
            get => Visibility.VisibleTaxiMdxCount;
            set => Visibility.VisibleTaxiMdxCount = value;
        }

        public int OpaqueBatchedMdxCount { get; set; }
        public int OpaqueUnbatchedMdxCount { get; set; }
        public int TransparentBatchedMdxCount { get; set; }
        public int TransparentUnbatchedMdxCount { get; set; }
        public int WmoDrawCallCount { get; set; }
        public int WmoBatchDrawCallCount { get; set; }
        public int WmoOpaqueBatchInstanceCount { get; set; }
        public int WmoGroupFallbackDrawCallCount { get; set; }
        public int WmoLiquidDrawCallCount { get; set; }
        public int WmoDoodadSubmissionCount { get; set; }
        public int WmoVisibleGroupSubmissionCount { get; set; }

        public double DeferredAssetLoadMs { get; set; }
        public double TaxiActorUpdateMs { get; set; }
        public double LightingMs { get; set; }
        public double SkyMs { get; set; }
        public double SkyboxBackdropMs { get; set; }
        public double WdlMs { get; set; }
        public double TerrainMs { get; set; }
        public double WmoVisibilityMs { get; set; }
        public double WmoSubmissionMs { get; set; }
        public double WmoTransparentSubmissionMs { get; set; }
        public double MdxAnimationMs { get; set; }
        public double MdxVisibilityMs { get; set; }
        public double MdxOpaqueSubmissionMs { get; set; }
        public double LiquidMs { get; set; }
        public double MdxTransparentSortMs { get; set; }
        public double MdxTransparentSubmissionMs { get; set; }
        public double OverlayMs { get; set; }
        public double SceneMaintenanceMs { get; set; }
        public List<WorldOverlayOwnerFrameStats> OverlayOwners { get; } = new(WorldOverlayOwners.All.Count);

        public void Reset()
        {
            Visibility.Reset();
            ObjectPasses.Reset();
            VisibleWmoRendererCache.Clear();
            VisibleMdxRendererCache.Clear();
            OpaqueBatchedMdxCount = 0;
            OpaqueUnbatchedMdxCount = 0;
            TransparentBatchedMdxCount = 0;
            TransparentUnbatchedMdxCount = 0;
            WmoDrawCallCount = 0;
            WmoBatchDrawCallCount = 0;
            WmoOpaqueBatchInstanceCount = 0;
            WmoGroupFallbackDrawCallCount = 0;
            WmoLiquidDrawCallCount = 0;
            WmoDoodadSubmissionCount = 0;
            WmoVisibleGroupSubmissionCount = 0;
            DeferredAssetLoadMs = 0;
            TaxiActorUpdateMs = 0;
            LightingMs = 0;
            SkyMs = 0;
            SkyboxBackdropMs = 0;
            WdlMs = 0;
            TerrainMs = 0;
            WmoVisibilityMs = 0;
            WmoSubmissionMs = 0;
            WmoTransparentSubmissionMs = 0;
            MdxAnimationMs = 0;
            MdxVisibilityMs = 0;
            MdxOpaqueSubmissionMs = 0;
            LiquidMs = 0;
            MdxTransparentSortMs = 0;
            MdxTransparentSubmissionMs = 0;
            OverlayMs = 0;
            SceneMaintenanceMs = 0;
            OverlayOwners.Clear();
            foreach (string ownerId in WorldOverlayOwners.All)
                OverlayOwners.Add(WorldOverlayOwnerFrameStats.Disabled(ownerId));
        }

        public void SetOverlayOwner(
            string ownerId,
            double durationMs,
            bool enabled,
            int preparedPrimitiveCount = 0,
            int submittedPrimitiveCount = 0,
            string cacheStatus = "not_cached",
            int deferredCount = 0)
        {
            WorldOverlayOwnerFrameStats stats = new(
                ownerId,
                Math.Max(0, durationMs),
                enabled,
                Math.Max(0, preparedPrimitiveCount),
                Math.Max(0, submittedPrimitiveCount),
                cacheStatus,
                Math.Max(0, deferredCount));

            for (int i = 0; i < OverlayOwners.Count; i++)
            {
                if (string.Equals(OverlayOwners[i].OwnerId, ownerId, StringComparison.Ordinal))
                {
                    OverlayOwners[i] = stats;
                    return;
                }
            }

            throw new InvalidOperationException($"Unknown world overlay owner '{ownerId}'.");
        }

        public double OverlayOwnerDurationSum => OverlayOwners.Sum(static owner => owner.DurationMs);

        public WorldRenderFrameStats ToStats(
            double totalCpuMs,
            int pendingAssetLoadCount,
            int terrainChunksRendered,
            int terrainChunksCulled,
            int wdlVisibleTileCount,
            int wdlHiddenTileCount)
        {
            int visibleMdxCount = Math.Max(0, VisibleMdxInstances.Count - VisibleTaxiMdxCount);
            return new WorldRenderFrameStats(
                totalCpuMs,
                pendingAssetLoadCount,
                terrainChunksRendered,
                terrainChunksCulled,
                wdlVisibleTileCount,
                wdlHiddenTileCount,
                VisibleWmoInstances.Count,
                visibleMdxCount,
                VisibleTaxiMdxCount,
                OpaqueBatchedMdxCount,
                OpaqueUnbatchedMdxCount,
                TransparentBatchedMdxCount,
                TransparentUnbatchedMdxCount,
                WmoDrawCallCount,
                WmoBatchDrawCallCount,
                WmoOpaqueBatchInstanceCount,
                WmoGroupFallbackDrawCallCount,
                WmoLiquidDrawCallCount,
                WmoDoodadSubmissionCount,
                WmoVisibleGroupSubmissionCount,
                new WorldRenderStageStats(DeferredAssetLoadMs),
                new WorldRenderStageStats(TaxiActorUpdateMs),
                new WorldRenderStageStats(LightingMs),
                new WorldRenderStageStats(SkyMs),
                new WorldRenderStageStats(SkyboxBackdropMs),
                new WorldRenderStageStats(WdlMs, wdlVisibleTileCount),
                new WorldRenderStageStats(TerrainMs, terrainChunksRendered),
                new WorldRenderStageStats(WmoVisibilityMs, VisibleWmoInstances.Count),
                new WorldRenderStageStats(WmoSubmissionMs, VisibleWmoInstances.Count, VisibleWmoInstances.Count),
                new WorldRenderStageStats(WmoTransparentSubmissionMs, VisibleWmoInstances.Count, WmoDrawCallCount),
                new WorldRenderStageStats(MdxAnimationMs),
                new WorldRenderStageStats(MdxVisibilityMs, VisibleMdxInstances.Count),
                new WorldRenderStageStats(MdxOpaqueSubmissionMs, VisibleMdxInstances.Count, OpaqueBatchedMdxCount + OpaqueUnbatchedMdxCount),
                new WorldRenderStageStats(LiquidMs),
                new WorldRenderStageStats(MdxTransparentSortMs, ObjectPasses.TransparentVisibleMdxRoutes.Count),
                new WorldRenderStageStats(MdxTransparentSubmissionMs, ObjectPasses.TransparentVisibleMdxRoutes.Count, TransparentBatchedMdxCount + TransparentUnbatchedMdxCount),
                new WorldRenderStageStats(OverlayMs),
                new WorldRenderStageStats(SceneMaintenanceMs))
            {
                OverlayOwners = OverlayOwners.ToArray(),
            };
        }
    }

    // Scratch collections reused every frame to avoid hot-path allocations.
    private readonly WorldRenderFrame _renderFrame = new();
    private readonly HashSet<WmoRenderer> _worldFrameWmoRenderers = new();
    private readonly List<int> _wireframeRevealWmoIndices = new();
    private readonly List<int> _wireframeRevealMdxIndices = new();
    private readonly MinimapRenderer? _minimapRenderer;
    private HoveredAssetInfo? _hoveredAssetInfo;
    private TerrainAssetLoadPolicy _assetLoadPolicy = StreamingTerrainAssetLoadPolicy;
    private bool _wireframeRevealEnabled;
    private bool _showHoveredAssetTooltips = true;
    private const int UniqueIdLayerGapThreshold = 100;
    private bool _uniqueIdFilterEnabled;
    private UniqueIdVisibilityScope _uniqueIdVisibilityScope = UniqueIdVisibilityScope.PerMap;
    private int _uniqueIdFilterMin = -1;
    private int _uniqueIdFilterMax = -1;
    private (int tileX, int tileY)? _uniqueIdFilterTile;
    private readonly List<ObjectPathFilterEntry> _objectPathFilters = new();

    // PM4 debug overlay
    private const int Pm4MaxLinesTotal = int.MaxValue;
    private const int Pm4MaxLinesPerTile = int.MaxValue;
    private const int Pm4MaxTrianglesTotal = int.MaxValue;
    private const int Pm4MaxTrianglesPerTile = int.MaxValue;
    private const int Pm4MaxPositionRefsTotal = int.MaxValue;
    private const int Pm4MaxPositionRefsPerTile = int.MaxValue;
    private const float Pm4MaxEdgeLength = 512f;
    private const int Pm4MinCameraTileRadius = 1;
    private const int Pm4MaxCameraTileRadius = 2;
    private const double Pm4ExpandWindowThresholdMs = 120.0;
    private const double Pm4ShrinkWindowThresholdMs = 300.0;
    private const long Pm4ProgressStatusIntervalMs = 1000;
    private const long Pm4ProgressLogIntervalMs = 5000;
    private bool _showPm4Overlay;
    private bool _showPm4SolidOverlay = true;
    private bool _showPm4ObjectBounds;
    private bool _showPm4Ck24Bounds;
    private bool _pm4OverlayIgnoreDepth;
    private bool _pm4FlipAllObjectsY;
    private bool _showPm4PositionRefs;
    private bool _showPm4ObjectCentroids;
    private bool _showPm4MscnNodes;
    private bool _showPm4MspvNodes;
    private float _pm4MscnCubeSize = 0.8f;
    private float _pm4MspvCubeSize = 1.0f;
    private float _pm4MscnCubeAlpha = 0.95f;
    private float _pm4MspvCubeAlpha = 0.95f;
    private float _pm4WireframeLineWidth = 2.5f;
    private bool _pm4RenderNodesAsCubes = true;
    private bool _pm4SplitCk24ByConnectivity;
    private bool _showPm4Type40 = true;
    private bool _showPm4Type80 = true;
    private bool _showPm4TypeOther = true;
    private bool _pm4SplitCk24ByMscnRef = true;
    // MSPV/MSPI path windows are a vertical planar quad mesh — the walls that stand between the
    // MSUR walkable surfaces. Measured corpus-wide: 98% of windows are exactly 4 indices, 99.6%
    // coplanar, and not one of 598,790 faces has Z as its dominant normal. The viewer has never
    // drawn them, so half the decoded geometry has been invisible.
    private bool _pm4ShowPathWalls = true;
    private Pm4OverlayColorMode _pm4ColorMode = Pm4OverlayColorMode.Ck24ObjectId;
    private Vector3 _pm4OverlayTranslation = Vector3.Zero;
    private Vector3 _pm4OverlayRotationDegrees = Vector3.Zero;
    private Vector3 _pm4OverlayScale = Vector3.One;
    private bool _pm4LoadAttempted;
    private string _pm4Status = "PM4 overlay not loaded.";
    private int _pm4TotalFiles;
    private int _pm4LoadedFiles;
    private int _pm4ObjectCount;
    private int _pm4LineCount;
    private int _pm4TriangleCount;
    private int _pm4RejectedLongEdges;
    private int _pm4VisibleObjectCount;
    private int _pm4VisibleLineCount;
    private int _pm4VisibleTriangleCount;
    private int _pm4PositionRefCount;
    private int _pm4VisiblePositionRefCount;
    private int _pm4TotalMsurCount;
    private int _pm4DroppedShortIndexCount;
    private int _pm4WallFaceCount;
    private int _pm4DroppedOutOfRangeMsviCount;
    private int _pm4DroppedEmptyComponentCount;
    private float _pm4MinObjectZ;
    private float _pm4MaxObjectZ;
    private int _pm4CameraTileRadius = Pm4MinCameraTileRadius;
    private double _pm4AverageLoadMs = -1.0;
    private (int minTileX, int minTileY, int maxTileX, int maxTileY)? _pm4LoadedCameraWindow;
    private ((int tileX, int tileY, uint ck24, int objectPart) key, (int tileX, int tileY, uint ck24) group)? _pm4GraphInfoCacheKey;
    private Pm4SelectedObjectGraphInfo? _pm4GraphInfoCacheValue;
    private bool _pm4GraphInfoCacheSplitByMscnRef;
    private bool _pm4GraphInfoCacheSplitByConnectivity;

    // Click-freeze / per-frame instrumentation. Toggle via the static
    // Pm4Profiling.Enabled flag. Counts and cumulative milliseconds per hot spot.
    // Logged on a coarser cadence than every call so we don't drown the log.
    private static readonly System.Diagnostics.Stopwatch s_pm4PickSw = new();
    private static long s_pm4PickCallCount;
    private static long s_pm4PickAabbHitCount;
    private static double s_pm4PickTotalMs;
    private static double s_pm4PickMaxMs;
    private static long s_pm4PickReportCount;

    private static readonly System.Diagnostics.Stopwatch s_pm4ResearchSw = new();
    private static long s_pm4ResearchCallCount;
    private static double s_pm4ResearchTotalMs;
    private static double s_pm4ResearchMaxMs;
    private static long s_pm4ResearchReportCount;

    private static readonly System.Diagnostics.Stopwatch s_pm4GraphBuildSw = new();
    private static long s_pm4GraphBuildCallCount;
    private static double s_pm4GraphBuildTotalMs;
    private static double s_pm4GraphBuildMaxMs;
    private static long s_pm4GraphBuildReportCount;
    private static long s_pm4GraphBuildLastObjectCount;
    private static long s_pm4GraphBuildLastRegionCount;

    /// <summary>
    /// Receiver for per-frame graph-build timings reported by the viewer-app
    /// side. Centralises the per-section log so all PM4 hot-spot reports
    /// share one cadence and one tag.
    /// </summary>
    public static class Pm4ProfilingAccumulator
    {
        public static void RecordGraphBuild(double elapsedMs, int walkedObjectCount, int regionCount)
        {
            if (!Pm4Profiling.Enabled) return;
            s_pm4GraphBuildCallCount++;
            s_pm4GraphBuildTotalMs += elapsedMs;
            if (elapsedMs > s_pm4GraphBuildMaxMs) s_pm4GraphBuildMaxMs = elapsedMs;
            s_pm4GraphBuildLastObjectCount = walkedObjectCount;
            s_pm4GraphBuildLastRegionCount = regionCount;
            s_pm4GraphBuildReportCount++;
            if (elapsedMs >= 50.0 || s_pm4GraphBuildReportCount >= 200)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[PM4-PROFILE] DrawPm4SceneGraph.Build: call={s_pm4GraphBuildCallCount} last={elapsedMs:0.0}ms max={s_pm4GraphBuildMaxMs:0.0}ms avg={s_pm4GraphBuildTotalMs / s_pm4GraphBuildCallCount:0.0}ms walked={walkedObjectCount} regions={regionCount}");
                s_pm4GraphBuildReportCount = 0;
            }
        }
    }

    private readonly HashSet<(int tileX, int tileY)> _pm4KnownMapTiles = new();
    private readonly HashSet<(int tileX, int tileY)> _pm4CoveredMapTiles = new();
    private Task<Pm4OverlayAsyncLoadResult>? _pm4LoadTask;
    private CancellationTokenSource? _pm4LoadCancellation;
    private int _pm4LoadRequestId;
    private Vector3 _lastRenderedCameraPosition;
    private bool _hasLastRenderedCameraPosition;
    private readonly Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> _pm4TileObjects = new();
    // Per-file in-memory PM4 cache (spec 054). Lets a camera shift or
    // re-visit reuse already-decoded PM4 payloads without re-running
    // BuildPm4TileObjects. Bounded by a simple LRU cap; cleared on
    // ReloadPm4Overlay().
    private readonly CorePm4PerFileCache _pm4PerFileInMemoryCache = new(capacity: 256);
    // MSCN = scene-graph connector anchors; one per MSUR surface (placed via MSUR.MscnRefIndex).
    // See wow-viewer/docs/architecture/pm4-chunk-semantics.md.
    private readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TileMscnPoints = new();
    // MSPV = path-vertex positions reached via MSPI from MSLK link records. Only present when surfaces are connected.
    private readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TileMspvPoints = new();
    private readonly Dictionary<(int tileX, int tileY), Pm4OverlayTileStats> _pm4TileStats = new();
    private readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TilePositionRefs = new();
    private readonly Dictionary<string, Pm4ResearchContext> _pm4ResearchBySourcePath = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _pm4ResearchUnavailablePaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4OverlayObject> _pm4ObjectLookup = new();
    private readonly HashSet<(int tileX, int tileY, uint ck24, int objectPart)> _highlightedPm4ObjectKeys = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), (int tileX, int tileY, uint ck24)> _pm4MergedObjectGroupKeys = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), List<(int tileX, int tileY, uint ck24, int objectPart)>> _pm4GroupToObjectKeys = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max)> _pm4ObjectGroupBounds = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max)> _pm4TileCk24Bounds = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4ObjectTranslations = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4ObjectRotationsDegrees = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4ObjectScales = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4TileCk24Translations = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4TileCk24RotationsDegrees = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), Vector3> _pm4TileCk24Scales = new();
    private (int tileX, int tileY, uint ck24, int objectPart)? _selectedPm4ObjectKey;
    private (int tileX, int tileY, uint ck24)? _selectedPm4ObjectGroupKey;

    // Culling stats (updated each frame)
    public int WmoRenderedCount { get; private set; }
    public int WmoCulledCount { get; private set; }
    public int MdxRenderedCount { get; private set; }
    public int MdxCulledCount { get; private set; }
    public int LastUnloadedWmoTileX { get; private set; } = -1;
    public int LastUnloadedWmoTileY { get; private set; } = -1;
    public int LastUnloadedWmoInstanceCount { get; private set; }
    public int WmoTileUnloadEventCount { get; private set; }
    public WorldRenderFrameStats LastRenderFrameStats { get; private set; } = WorldRenderFrameStats.Empty;
    public string RendererOptimizationHint => WorldRenderOptimizationAdvisor.BuildHint(LastRenderFrameStats);
    public bool UseHierarchicalSceneTraversal
    {
        get => _useHierarchicalSceneTraversal;
        set
        {
            if (_useHierarchicalSceneTraversal == value)
                return;

            _useHierarchicalSceneTraversal = value;
            _sceneGraphBuild = null;
            _sceneGraphFrameVisibilityPrepared = false;
            _instancesDirty = true;
        }
    }
    public bool IsHierarchicalSceneTraversalActive => UseHierarchicalSceneTraversal && _sceneGraphBuild is not null;
    public int SceneGraphResidentAdtCount => _sceneGraphBuild?.AdtGraphs.Count ?? 0;
    public bool SceneGraphHasExternalRoot => _sceneGraphBuild?.ExternalGraph is not null;
    public WorldSceneGraphSnapshot? SceneGraphSnapshot => _sceneGraphBuild?.CreateSnapshot();
    public WorldSceneTraversalDiagnostics SceneGraphTraversalDiagnostics => _lastSceneGraphTraversalDiagnostics;
    public IReadOnlyDictionary<string, WorldScenePortalAdapterResult> SceneGraphPortalAdapters => _sceneGraphPortalAdapters;
    public IReadOnlyDictionary<string, WorldScenePortalVisibilityResult> SceneGraphPortalVisibility => _sceneGraphPortalVisibility;

    // Stats
    public int MdxInstanceCount => _mdxInstances.Count;
    public int SkyboxInstanceCount => _skyboxInstances.Count;
    public int WmoInstanceCount => _wmoInstances.Count;
    public int UniqueMdxModels => _assets.MdxModelsLoaded;
    public int UniqueWmoModels => _assets.WmoModelsLoaded;
    public int ExternalSpawnMdxCount => _externalMdxInstances.Count;
    public int ExternalSpawnWmoCount => _externalWmoInstances.Count;
    public int ExternalSpawnInstanceCount => ExternalSpawnMdxCount + ExternalSpawnWmoCount;
    public float SqlGameObjectMdxScaleMultiplier { get; set; } = 1.0f;
    public TerrainManager Terrain => _terrainManager;
    public WorldAssetManager Assets => _assets;
    public bool IsWmoBased => _terrainManager.Adapter.IsWmoBased;

    // Expose raw placement data for UI object list
    public IReadOnlyList<MddfPlacement> MddfPlacements => _terrainManager.Adapter.MddfPlacements;
    public IReadOnlyList<ModfPlacement> ModfPlacements => _terrainManager.Adapter.ModfPlacements;
    public IReadOnlyList<string> MdxModelNames => _terrainManager.Adapter.MdxModelNames;
    public IReadOnlyList<string> WmoModelNames => _terrainManager.Adapter.WmoModelNames;

    // Sky dome
    private readonly SkyDomeRenderer _skyDome;
    public SkyDomeRenderer SkyDome => _skyDome;

    // WDL low-res terrain (far terrain background)
    private WdlTerrainRenderer? _wdlTerrain;
    public WdlTerrainRenderer? WdlTerrain => _wdlTerrain;
    public bool ShowWdlTerrain { get; set; } = true;
    public bool ShowSky { get; set; } = true;

    // Bounding box debug rendering
    private bool _showBoundingBoxes = false;
    private BoundingBoxRenderer? _bbRenderer;
    public bool ShowBoundingBoxes { get => _showBoundingBoxes; set => _showBoundingBoxes = value; }

    // Object selection
    private ObjectType _selectedObjectType = ObjectType.None;
    private int _selectedObjectIndex = -1;
    private SelectedSceneObjectKey? _selectedSceneObjectKey;
    public ObjectType SelectedObjectType => _selectedObjectType;
    public int SelectedObjectIndex => _selectedObjectIndex;
    public bool WireframeRevealEnabled => _wireframeRevealEnabled;
    public bool TerrainWireframeEnabled => _terrainManager.IsWireframe;
    public bool ObjectWireframeEnabled => _assets.ObjectWireframeEnabled;
    public HoveredAssetInfo? HoveredAssetInfo => _hoveredAssetInfo;
    public bool ShowHoveredAssetTooltips { get => _showHoveredAssetTooltips; set => _showHoveredAssetTooltips = value; }
    public bool LimitHoveredAssetRange { get => _limitHoveredAssetRange; set => _limitHoveredAssetRange = value; }
    public bool UseDynamicHoveredAssetRange { get => _useDynamicHoveredAssetRange; set => _useDynamicHoveredAssetRange = value; }
    public int PendingAssetLoadCount => _assets.PendingAssetLoadCount;
    public int PendingDeferredWmoDoodadLoadCount => _assets.PendingDeferredWmoDoodadLoadCount;
    public int PendingDeferredWmoMaterialTextureLoadCount => _assets.PendingDeferredWmoMaterialTextureLoadCount;
    public int PendingWorldObjectLoadCount => PendingAssetLoadCount + PendingDeferredWmoDoodadLoadCount;
    public int PendingCapturePreloadLoadCount => PendingWorldObjectLoadCount + PendingDeferredWmoMaterialTextureLoadCount;
    private bool _capturePreloadActive;
    private readonly HashSet<(int tileX, int tileY)> _capturePreloadTiles = new();
    public bool CapturePreloadActive
    {
        get => _capturePreloadActive;
        set
        {
            _capturePreloadActive = value;
            if (!value)
                _capturePreloadTiles.Clear();
        }
    }
    public float ObjectStreamingRangeMultiplier
    {
        get => _objectStreamingRangeMultiplier;
        set => _objectStreamingRangeMultiplier = Math.Clamp(value, 0.25f, 4.0f);
    }
    public float MaxVisibleMdxBoundsHeight
    {
        get => _maxVisibleMdxBoundsHeight;
        set => _maxVisibleMdxBoundsHeight = value > 0f ? value : 0f;
    }
    public bool HideTerrainOccludedMdx
    {
        get => _hideTerrainOccludedMdx;
        set => _hideTerrainOccludedMdx = value;
    }
    public string? SecondaryOverlayMap
    {
        get => _terrainManager?.OverlayMapName;
        set => _terrainManager?.SetOverlayMap(value);
    }
    public bool EnableRuntimeWmoGroupVisibility
    {
        get => _assets.EnableRuntimeWmoGroupVisibility;
        set => _assets.EnableRuntimeWmoGroupVisibility = value;
    }
    public bool EnableRuntimeWmoGroupLiquids
    {
        get => _assets.EnableRuntimeWmoGroupLiquids;
        set => _assets.EnableRuntimeWmoGroupLiquids = value;
    }
    public WorldObjectVisibilityProfile ObjectVisibilityProfile
    {
        get => _objectVisibilityProfile;
        set => _objectVisibilityProfile = value;
    }
    public bool ObjectsVisible { get => _objectsVisible; set => _objectsVisible = value; }
    public bool WmosVisible { get => _wmosVisible; set => _wmosVisible = value; }
    public bool DoodadsVisible { get => _doodadsVisible; set => _doodadsVisible = value; }
    public bool ObjectPathFiltersEnabled { get => _objectPathFiltersEnabled; set => _objectPathFiltersEnabled = value; }
    public IReadOnlyList<ObjectPathFilterEntry> ObjectPathFilters => _objectPathFilters;
    public bool ShowSelectedObjectBounds { get => _showSelectedObjectBounds; set => _showSelectedObjectBounds = value; }
    public float HoveredAssetMaxDistance
    {
        get => _hoveredAssetMaxDistance;
        set => _hoveredAssetMaxDistance = Math.Clamp(value, 10f, MaxWorldObjectViewDistance);
    }
    public float EffectiveHoveredAssetMaxDistance => ComputeEffectiveHoveredAssetMaxDistance();
    public bool UniqueIdFilterEnabled { get => _uniqueIdFilterEnabled; set => _uniqueIdFilterEnabled = value; }
    public UniqueIdVisibilityScope UniqueIdVisibilityScope { get => _uniqueIdVisibilityScope; set => _uniqueIdVisibilityScope = value; }
    public int UniqueIdFilterMin { get => _uniqueIdFilterMin; set => _uniqueIdFilterMin = value; }
    public int UniqueIdFilterMax { get => _uniqueIdFilterMax; set => _uniqueIdFilterMax = value; }
    public (int tileX, int tileY)? UniqueIdFilterTile => _uniqueIdFilterTile;
    public bool ShowPm4Overlay
    {
        get => _showPm4Overlay;
        set
        {
            if (_showPm4Overlay == value)
                return;

            _showPm4Overlay = value;
            if (_showPm4Overlay)
                BeginPm4OverlayLoad();
        }
    }

    public bool Pm4LoadAttempted => _pm4LoadAttempted;
    public bool IsPm4Loading => _pm4LoadTask != null && !_pm4LoadTask.IsCompleted;
    public string Pm4Status => _pm4Status;
    public int Pm4TotalFiles => _pm4TotalFiles;
    public int Pm4LoadedFiles => _pm4LoadedFiles;
    public int Pm4ObjectCount => _pm4ObjectCount;

    public bool LoadLoosePm4File(string filePath)
    {
        if (!File.Exists(filePath))
            return false;

        try
        {
            byte[] bytes = File.ReadAllBytes(filePath);
            Pm4File pm4 = CorePm4DocumentReader.Read(bytes, filePath);

            Pm4CoordinateService.TryParseTileCoordinates(filePath, out int tileX, out int tileY);

            int lineBudget = int.MaxValue;
            int triBudget = int.MaxValue;
            int rejectedLong = 0;
            List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                pm4,
                filePath,
                tileX,
                tileY,
                _pm4SplitCk24ByMscnRef,
                _pm4SplitCk24ByConnectivity,
                _pm4ShowPathWalls,
                ref lineBudget,
                ref triBudget,
                ref rejectedLong,
                out _);

            if (objects.Count == 0)
                return false;

            _pm4TileObjects[(tileX, tileY)] = objects;
            _pm4LoadedCameraWindow = (tileX, tileY, tileX, tileY);
            _pm4LoadAttempted = true;
            _showPm4Overlay = true;
            _pm4Status = $"Loaded loose PM4/PD4 '{Path.GetFileName(filePath)}' ({Pm4VersionFormatter.Format(pm4.Version)}): {objects.Count} objects, {pm4.KnownChunks.Msvt.Count} verts, {pm4.KnownChunks.Msur.Count} surfaces.";
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
            return true;
        }
        catch (Exception ex)
        {
            _pm4Status = $"Failed to load loose PM4/PD4 '{Path.GetFileName(filePath)}': {ex.Message}";
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
            return false;
        }
    }

    public void SetUniqueIdFilterTile(int tileX, int tileY)
    {
        _uniqueIdFilterTile = (tileX, tileY);
    }

    public void SetUniqueIdFilterRange(int minUniqueId, int maxUniqueId)
    {
        if (minUniqueId <= maxUniqueId)
        {
            _uniqueIdFilterMin = minUniqueId;
            _uniqueIdFilterMax = maxUniqueId;
            return;
        }

        _uniqueIdFilterMin = maxUniqueId;
        _uniqueIdFilterMax = minUniqueId;
    }

    public void ResetUniqueIdFilter()
    {
        _uniqueIdFilterEnabled = false;
        _uniqueIdFilterMin = -1;
        _uniqueIdFilterMax = -1;
    }

    public bool AddObjectPathFilter(string pathPrefix, bool appliesToWmo, bool appliesToMdx)
    {
        string normalizedPrefix = ObjectPathFilterEntry.NormalizePrefix(pathPrefix);
        if (string.IsNullOrWhiteSpace(normalizedPrefix) || (!appliesToWmo && !appliesToMdx))
            return false;

        ObjectPathFilterEntry entry = new(normalizedPrefix, appliesToWmo, appliesToMdx);
        if (_objectPathFilters.Contains(entry))
            return false;

        _objectPathFilters.Add(entry);
        _objectPathFilters.Sort(static (left, right) => string.Compare(left.PathPrefix, right.PathPrefix, StringComparison.OrdinalIgnoreCase));
        return true;
    }

    public bool RemoveObjectPathFilter(string pathPrefix, bool appliesToWmo, bool appliesToMdx)
    {
        string normalizedPrefix = ObjectPathFilterEntry.NormalizePrefix(pathPrefix);
        if (string.IsNullOrWhiteSpace(normalizedPrefix))
            return false;

        return _objectPathFilters.RemoveAll(entry =>
            string.Equals(entry.PathPrefix, normalizedPrefix, StringComparison.OrdinalIgnoreCase)
            && entry.AppliesToWmo == appliesToWmo
            && entry.AppliesToMdx == appliesToMdx) > 0;
    }

    public void ClearObjectPathFilters()
    {
        _objectPathFilters.Clear();
    }

    public bool TryGetUniqueIdFilterRange(out int minUniqueId, out int maxUniqueId, out int instanceCount)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        minUniqueId = int.MaxValue;
        maxUniqueId = int.MinValue;
        instanceCount = 0;

        AccumulateUniqueIdFilterRange(_wmoInstances, ref minUniqueId, ref maxUniqueId, ref instanceCount);
        AccumulateUniqueIdFilterRange(_mdxInstances, ref minUniqueId, ref maxUniqueId, ref instanceCount);

        if (instanceCount <= 0)
        {
            minUniqueId = 0;
            maxUniqueId = 0;
            return false;
        }

        return true;
    }

    public IReadOnlyList<UniqueIdArchaeologyLayer> GetUniqueIdArchaeologyLayers()
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        var countsById = new SortedDictionary<int, (int wmoCount, int mdxCount)>();
        AccumulateUniqueIdLayerCandidates(_wmoInstances, isWmo: true, countsById);
        AccumulateUniqueIdLayerCandidates(_mdxInstances, isWmo: false, countsById);

        if (countsById.Count == 0)
            return Array.Empty<UniqueIdArchaeologyLayer>();

        var layers = new List<UniqueIdArchaeologyLayer>();
        int layerNumber = 1;
        int layerStart = 0;
        int layerEnd = 0;
        int previousId = 0;
        int placementCount = 0;
        int wmoCount = 0;
        int mdxCount = 0;
        bool hasLayer = false;

        foreach ((int uniqueId, (int layerWmoCount, int layerMdxCount) counts) in countsById)
        {
            if (!hasLayer)
            {
                layerStart = uniqueId;
                hasLayer = true;
            }
            else if (uniqueId - previousId > UniqueIdLayerGapThreshold)
            {
                layers.Add(new UniqueIdArchaeologyLayer(layerNumber++, layerStart, layerEnd, placementCount, wmoCount, mdxCount));
                layerStart = uniqueId;
                placementCount = 0;
                wmoCount = 0;
                mdxCount = 0;
            }

            layerEnd = uniqueId;
            previousId = uniqueId;
            placementCount += counts.layerWmoCount + counts.layerMdxCount;
            wmoCount += counts.layerWmoCount;
            mdxCount += counts.layerMdxCount;
        }

        if (hasLayer)
            layers.Add(new UniqueIdArchaeologyLayer(layerNumber, layerStart, layerEnd, placementCount, wmoCount, mdxCount));

        return layers;
    }

    public void ApplyTextureSamplingSettings()
    {
        _terrainManager.Renderer.ApplyTextureSamplingSettings();
        _assets.ApplyTextureSamplingSettings();
    }
    public int Pm4LineCount => _pm4LineCount;
    public int Pm4TriangleCount => _pm4TriangleCount;
    public int Pm4RejectedLongEdges => _pm4RejectedLongEdges;
    public int Pm4TotalMsurCount => _pm4TotalMsurCount;
    public int Pm4DroppedShortIndexCount => _pm4DroppedShortIndexCount;

    /// <summary>MSPV/MSPI wall faces emitted across the loaded tiles.</summary>
    public int Pm4WallFaceCount => _pm4WallFaceCount;
    public int Pm4DroppedOutOfRangeMsviCount => _pm4DroppedOutOfRangeMsviCount;
    public int Pm4DroppedEmptyComponentCount => _pm4DroppedEmptyComponentCount;
    public int Pm4VisibleObjectCount => _pm4VisibleObjectCount;
    public int Pm4VisibleLineCount => _pm4VisibleLineCount;
    public int Pm4VisibleTriangleCount => _pm4VisibleTriangleCount;
    public int Pm4PositionRefCount => _pm4PositionRefCount;
    public int Pm4VisiblePositionRefCount => _pm4VisiblePositionRefCount;
    public bool ShowPm4SolidOverlay { get => _showPm4SolidOverlay; set => _showPm4SolidOverlay = value; }
    public bool ShowPm4ObjectBounds { get => _showPm4ObjectBounds; set => _showPm4ObjectBounds = value; }
    public bool ShowPm4Ck24Bounds { get => _showPm4Ck24Bounds; set => _showPm4Ck24Bounds = value; }
    public bool Pm4OverlayIgnoreDepth { get => _pm4OverlayIgnoreDepth; set => _pm4OverlayIgnoreDepth = value; }
    public bool Pm4FlipAllObjectsY
    {
        get => _pm4FlipAllObjectsY;
        set
        {
            if (_pm4FlipAllObjectsY == value)
                return;

            _pm4FlipAllObjectsY = value;

            // Bake global Y-flip at PM4 decode time to avoid per-frame vertex transform cost.
            if (_pm4LoadAttempted)
                ReloadPm4Overlay();
        }
    }
    public bool ShowPm4PositionRefs { get => _showPm4PositionRefs; set => _showPm4PositionRefs = value; }
    public bool ShowPm4ObjectCentroids { get => _showPm4ObjectCentroids; set => _showPm4ObjectCentroids = value; }
    public bool ShowPm4MscnNodes { get => _showPm4MscnNodes; set => _showPm4MscnNodes = value; }
    public bool ShowPm4MspvNodes { get => _showPm4MspvNodes; set => _showPm4MspvNodes = value; }
    public float Pm4MscnCubeSize { get => _pm4MscnCubeSize; set => _pm4MscnCubeSize = MathF.Max(0.1f, value); }
    public float Pm4MspvCubeSize { get => _pm4MspvCubeSize; set => _pm4MspvCubeSize = MathF.Max(0.1f, value); }
    public float Pm4MscnCubeAlpha { get => _pm4MscnCubeAlpha; set => _pm4MscnCubeAlpha = Math.Clamp(value, 0.1f, 1f); }
    public float Pm4MspvCubeAlpha { get => _pm4MspvCubeAlpha; set => _pm4MspvCubeAlpha = Math.Clamp(value, 0.1f, 1f); }
    public float Pm4WireframeLineWidth { get => _pm4WireframeLineWidth; set => _pm4WireframeLineWidth = MathF.Max(1.0f, value); }
    public bool Pm4RenderNodesAsCubes { get => _pm4RenderNodesAsCubes; set => _pm4RenderNodesAsCubes = value; }
    public bool Pm4SplitCk24ByConnectivity { get => _pm4SplitCk24ByConnectivity; set => _pm4SplitCk24ByConnectivity = value; }
    public bool ShowPm4Type40 { get => _showPm4Type40; set => _showPm4Type40 = value; }
    public bool ShowPm4Type80 { get => _showPm4Type80; set => _showPm4Type80 = value; }
    public bool ShowPm4TypeOther { get => _showPm4TypeOther; set => _showPm4TypeOther = value; }
    public bool Pm4SplitCk24ByMscnRef { get => _pm4SplitCk24ByMscnRef; set => _pm4SplitCk24ByMscnRef = value; }
    public bool Pm4ShowPathWalls { get => _pm4ShowPathWalls; set => _pm4ShowPathWalls = value; }
    public Pm4OverlayColorMode Pm4ColorMode { get => _pm4ColorMode; set => _pm4ColorMode = value; }
    public Vector3 Pm4OverlayTranslation { get => _pm4OverlayTranslation; set => _pm4OverlayTranslation = value; }
    public Vector3 Pm4OverlayRotationDegrees { get => _pm4OverlayRotationDegrees; set => _pm4OverlayRotationDegrees = value; }
    public Vector3 Pm4OverlayScale { get => _pm4OverlayScale; set => _pm4OverlayScale = value; }
    public bool HasSelectedPm4Object => _selectedPm4ObjectKey.HasValue;
    public (int tileX, int tileY, uint ck24, int objectPart)? SelectedPm4ObjectKey => _selectedPm4ObjectKey;

    private const uint Pm4SyntheticZeroCk24GroupMask = 0x80000000u;

    private static (int tileX, int tileY, uint ck24) BuildPm4BaseObjectGroupKey(
        (int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        uint groupKey = objectKey.ck24 != 0
            ? objectKey.ck24
            : Pm4SyntheticZeroCk24GroupMask | (uint)objectKey.objectPart;
        return (0, 0, groupKey);
    }

    private (int tileX, int tileY, uint ck24) ResolvePm4ObjectGroupKey((int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        var baseGroupKey = BuildPm4BaseObjectGroupKey(objectKey);
        return _pm4MergedObjectGroupKeys.TryGetValue(baseGroupKey, out var mergedGroupKey)
            ? mergedGroupKey
            : baseGroupKey;
    }

    private bool IsPm4ObjectInGroup(
        (int tileX, int tileY, uint ck24) groupKey,
        (int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        return ResolvePm4ObjectGroupKey(objectKey) == groupKey;
    }

    public Vector3 SelectedPm4ObjectTranslation
    {
        get
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return Vector3.Zero;

            return _pm4ObjectTranslations.TryGetValue(_selectedPm4ObjectGroupKey.Value, out Vector3 translation)
                ? translation
                : Vector3.Zero;
        }
        set
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return;

            if (value.LengthSquared() < 0.0001f)
                _pm4ObjectTranslations.Remove(_selectedPm4ObjectGroupKey.Value);
            else
                _pm4ObjectTranslations[_selectedPm4ObjectGroupKey.Value] = value;
        }
    }
    public Vector3 SelectedPm4ObjectRotationDegrees
    {
        get
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return Vector3.Zero;

            return _pm4ObjectRotationsDegrees.TryGetValue(_selectedPm4ObjectGroupKey.Value, out Vector3 rotationDegrees)
                ? rotationDegrees
                : Vector3.Zero;
        }
        set
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return;

            if (IsNearZeroVector(value))
                _pm4ObjectRotationsDegrees.Remove(_selectedPm4ObjectGroupKey.Value);
            else
                _pm4ObjectRotationsDegrees[_selectedPm4ObjectGroupKey.Value] = value;
        }
    }
    public Vector3 SelectedPm4ObjectScale
    {
        get
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return Vector3.One;

            return _pm4ObjectScales.TryGetValue(_selectedPm4ObjectGroupKey.Value, out Vector3 scale)
                ? scale
                : Vector3.One;
        }
        set
        {
            if (!_selectedPm4ObjectGroupKey.HasValue)
                return;

            Vector3 sanitized = SanitizeScale(value);
            if (IsNearOneVector(sanitized))
                _pm4ObjectScales.Remove(_selectedPm4ObjectGroupKey.Value);
            else
                _pm4ObjectScales[_selectedPm4ObjectGroupKey.Value] = sanitized;
        }
    }
    public uint? SelectedPm4RawCk24 => _selectedPm4ObjectKey?.ck24;
    public (int tileX, int tileY, uint ck24)? SelectedPm4TileCk24Key
        => _selectedPm4ObjectKey.HasValue
            ? (_selectedPm4ObjectKey.Value.tileX, _selectedPm4ObjectKey.Value.tileY, _selectedPm4ObjectKey.Value.ck24)
            : null;
    public Vector3 SelectedPm4Ck24LayerTranslation
    {
        get
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return Vector3.Zero;

            return _pm4TileCk24Translations.TryGetValue(SelectedPm4TileCk24Key.Value, out Vector3 translation)
                ? translation
                : Vector3.Zero;
        }
        set
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return;

            if (value.LengthSquared() < 0.0001f)
                _pm4TileCk24Translations.Remove(SelectedPm4TileCk24Key.Value);
            else
                _pm4TileCk24Translations[SelectedPm4TileCk24Key.Value] = value;
        }
    }
    public Vector3 SelectedPm4Ck24LayerRotationDegrees
    {
        get
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return Vector3.Zero;

            return _pm4TileCk24RotationsDegrees.TryGetValue(SelectedPm4TileCk24Key.Value, out Vector3 rotationDegrees)
                ? rotationDegrees
                : Vector3.Zero;
        }
        set
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return;

            if (IsNearZeroVector(value))
                _pm4TileCk24RotationsDegrees.Remove(SelectedPm4TileCk24Key.Value);
            else
                _pm4TileCk24RotationsDegrees[SelectedPm4TileCk24Key.Value] = value;
        }
    }
    public Vector3 SelectedPm4Ck24LayerScale
    {
        get
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return Vector3.One;

            return _pm4TileCk24Scales.TryGetValue(SelectedPm4TileCk24Key.Value, out Vector3 scale)
                ? scale
                : Vector3.One;
        }
        set
        {
            if (!SelectedPm4TileCk24Key.HasValue)
                return;

            Vector3 sanitized = SanitizeScale(value);
            if (IsNearOneVector(sanitized))
                _pm4TileCk24Scales.Remove(SelectedPm4TileCk24Key.Value);
            else
                _pm4TileCk24Scales[SelectedPm4TileCk24Key.Value] = sanitized;
        }
    }
    public float Pm4OverlayYawDegrees
    {
        get => _pm4OverlayRotationDegrees.Z;
        set => _pm4OverlayRotationDegrees = new Vector3(_pm4OverlayRotationDegrees.X, _pm4OverlayRotationDegrees.Y, value);
    }
    public IReadOnlyCollection<Pm4OverlayTileStats> Pm4TileStats => _pm4TileStats.Values;

    public bool TryGetSelectedPm4Ck24LayerStats(out int tileCount, out int objectCount)
    {
        tileCount = 0;
        objectCount = 0;

        var tileCk24Key = SelectedPm4TileCk24Key;
        if (!tileCk24Key.HasValue)
            return false;

        foreach (var objectKey in _pm4ObjectLookup.Keys)
        {
            if (objectKey.tileX != tileCk24Key.Value.tileX
                || objectKey.tileY != tileCk24Key.Value.tileY
                || objectKey.ck24 != tileCk24Key.Value.ck24)
                continue;

            objectCount++;
        }

        tileCount = objectCount > 0 ? 1 : 0;
        return objectCount > 0;
    }

    public string BuildPm4OverlayInterchangeJson(bool includeGeometry = true)
    {
        static float[] VectorToArray(Vector3 v) => new[] { v.X, v.Y, v.Z };
        static float[] LineToArray(Pm4LineSegment line, in Matrix4x4 transform)
        {
            Vector3 from = ApplyPm4OverlayTransform(line.From, transform);
            Vector3 to = ApplyPm4OverlayTransform(line.To, transform);
            return new[] { from.X, from.Y, from.Z, to.X, to.Y, to.Z };
        }

        static float[] TriangleToArray(Pm4Triangle tri, in Matrix4x4 transform)
        {
            Vector3 a = ApplyPm4OverlayTransform(tri.A, transform);
            Vector3 b = ApplyPm4OverlayTransform(tri.B, transform);
            Vector3 c = ApplyPm4OverlayTransform(tri.C, transform);
            return new[] { a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z };
        }

        var tiles = _pm4TileObjects
            .OrderBy(kvp => kvp.Key.tileX)
            .ThenBy(kvp => kvp.Key.tileY)
            .Select(kvp => new
            {
                tileX = kvp.Key.tileX,
                tileY = kvp.Key.tileY,
                objectCount = kvp.Value.Count,
                objects = kvp.Value
                    .OrderBy(obj => obj.Ck24)
                    .ThenBy(obj => obj.ObjectPartId)
                    .Select(obj =>
                    {
                        var objectKey = (kvp.Key.tileX, kvp.Key.tileY, obj.Ck24, obj.ObjectPartId);
                        var objectGroupKey = ResolvePm4ObjectGroupKey(objectKey);
                        var tileCk24Key = (kvp.Key.tileX, kvp.Key.tileY, obj.Ck24);
                        bool hasLayerOffset = _pm4TileCk24Translations.TryGetValue(tileCk24Key, out Vector3 layerOffset)
                            && !IsNearZeroVector(layerOffset);
                        bool hasLayerRotation = _pm4TileCk24RotationsDegrees.TryGetValue(tileCk24Key, out Vector3 layerRotationDegrees)
                            && !IsNearZeroVector(layerRotationDegrees);
                        bool hasLayerScale = _pm4TileCk24Scales.TryGetValue(tileCk24Key, out Vector3 layerScale)
                            && !IsNearOneVector(layerScale);
                        bool hasObjectOffset = _pm4ObjectTranslations.TryGetValue(objectGroupKey, out Vector3 objectOffset);
                        bool hasObjectRotation = _pm4ObjectRotationsDegrees.TryGetValue(objectGroupKey, out Vector3 objectRotationDegrees)
                            && !IsNearZeroVector(objectRotationDegrees);
                        bool hasObjectScale = _pm4ObjectScales.TryGetValue(objectGroupKey, out Vector3 objectScale)
                            && !IsNearOneVector(objectScale);
                        Matrix4x4 baseGeometryTransform = obj.BaseTransform;

                        return new
                        {
                            ck24 = obj.Ck24,
                            ck24Type = obj.Ck24Type,
                            ck24ObjectId = obj.Ck24ObjectId,
                            // Byte-decomposed view of the 24-bit ck24. Ck24ObjectId
                            // above is the lossy flattening of these two bytes into
                            // a single 16-bit ID. See Pm4OverlayObject.Ck24HighByte /
                            // Ck24LowByte for the model. Additive - existing readers
                            // ignore the new fields.
                            ck24HighByte = obj.Ck24HighByte,
                            ck24LowByte = obj.Ck24LowByte,
                            objectPartId = obj.ObjectPartId,
                            mshd = new
                            {
                                field00 = obj.MshdField00,
                                regionId = obj.MshdRegionId,
                                field08 = obj.MshdField08,
                            },
                            linkGroupObjectId = obj.LinkGroupObjectId,
                            objectGroupKey = new
                            {
                                tileX = objectGroupKey.tileX,
                                tileY = objectGroupKey.tileY,
                                ck24 = objectGroupKey.ck24,
                            },
                            linkedPositionRefCount = obj.LinkedPositionRefCount,
                            linkedPositionRefSummary = new
                            {
                                totalCount = obj.LinkedPositionRefSummary.TotalCount,
                                normalCount = obj.LinkedPositionRefSummary.NormalCount,
                                terminatorCount = obj.LinkedPositionRefSummary.TerminatorCount,
                                floorMin = obj.LinkedPositionRefSummary.FloorMin,
                                floorMax = obj.LinkedPositionRefSummary.FloorMax,
                                headingMinDegrees = JsonFiniteOrNull(obj.LinkedPositionRefSummary.HeadingMinDegrees),
                                headingMaxDegrees = JsonFiniteOrNull(obj.LinkedPositionRefSummary.HeadingMaxDegrees),
                                headingMeanDegrees = JsonFiniteOrNull(obj.LinkedPositionRefSummary.HeadingMeanDegrees),
                            },
                            surfaceCount = obj.SurfaceCount,
                            dominantGroupKey = obj.DominantGroupKey,
                            dominantAttributeMask = obj.DominantAttributeMask,
                            dominantMscnRefIndex = obj.DominantMscnRefIndex,
                            averageSurfaceHeight = JsonFiniteOrNull(obj.AverageSurfaceHeight),
                            boundsMin = VectorToArray(obj.BoundsMin),
                            boundsMax = VectorToArray(obj.BoundsMax),
                            center = VectorToArray(obj.Center),
                            planarTransform = new
                            {
                                swapPlanarAxes = obj.PlanarTransform.SwapPlanarAxes,
                                invertU = obj.PlanarTransform.InvertU,
                                invertV = obj.PlanarTransform.InvertV,
                                invertsWinding = obj.PlanarTransform.InvertsWinding,
                            },
                            rawCk24Layer = new
                            {
                                tileX = kvp.Key.tileX,
                                tileY = kvp.Key.tileY,
                                ck24 = obj.Ck24,
                                hasLayerOffset,
                                layerOffset = hasLayerOffset ? VectorToArray(layerOffset) : VectorToArray(Vector3.Zero),
                                hasLayerRotation,
                                layerRotationDegrees = hasLayerRotation ? VectorToArray(layerRotationDegrees) : VectorToArray(Vector3.Zero),
                                hasLayerScale,
                                layerScale = hasLayerScale ? VectorToArray(layerScale) : VectorToArray(Vector3.One),
                            },
                            hasObjectOffset,
                            objectOffset = hasObjectOffset ? VectorToArray(objectOffset) : VectorToArray(Vector3.Zero),
                            hasObjectRotation,
                            objectRotationDegrees = hasObjectRotation ? VectorToArray(objectRotationDegrees) : VectorToArray(Vector3.Zero),
                            hasObjectScale,
                            objectScale = hasObjectScale ? VectorToArray(objectScale) : VectorToArray(Vector3.One),
                            baseTransformRotationDegreesZ = obj.BaseRotationRadians * (180f / MathF.PI),
                            lineCount = obj.Lines.Count,
                            triangleCount = obj.Triangles.Count,
                            baseTransformTranslation = VectorToArray(obj.PlacementAnchor),
                            lines = includeGeometry
                                ? obj.Lines.Select(line => LineToArray(line, baseGeometryTransform))
                                    .ToList()
                                : new List<float[]>(),
                            triangles = includeGeometry
                                ? obj.Triangles.Select(tri => TriangleToArray(tri, baseGeometryTransform))
                                    .ToList()
                                : new List<float[]>(),
                        };
                    })
                    .ToList(),
            })
            .ToList();

        var positionRefs = _pm4TilePositionRefs
            .OrderBy(kvp => kvp.Key.tileX)
            .ThenBy(kvp => kvp.Key.tileY)
            .Select(kvp => new
            {
                tileX = kvp.Key.tileX,
                tileY = kvp.Key.tileY,
                refs = kvp.Value.Select(VectorToArray).ToList(),
            })
            .ToList();

        var payload = new
        {
            generatedAtUtc = DateTime.UtcNow,
            status = _pm4Status,
            includeGeometry,
            summary = new
            {
                totalFiles = _pm4TotalFiles,
                loadedFiles = _pm4LoadedFiles,
                objectCount = _pm4ObjectCount,
                lineCount = _pm4LineCount,
                triangleCount = _pm4TriangleCount,
                positionRefCount = _pm4PositionRefCount,
                rejectedLongEdges = _pm4RejectedLongEdges,
            },
            overlayAlignment = new
            {
                translation = VectorToArray(_pm4OverlayTranslation),
                rotationDegrees = VectorToArray(_pm4OverlayRotationDegrees),
                scale = VectorToArray(_pm4OverlayScale),
            },
            tiles,
            tilePositionRefs = positionRefs,
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true,
        });
    }

    public Pm4OfflineObjExportSummary ExportPm4ObjectsAsObjDirectory(string outputDirectory)
    {
        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("Output directory is required.", nameof(outputDirectory));

        if (_dataSource == null)
            throw new InvalidOperationException("PM4 export is unavailable: no data source.");

        string mapName = _terrainManager.MapName;
        List<string> mapPm4Candidates = _dataSource
            .GetFileList(".pm4")
            .Where(path => IsMapPm4Path(path, mapName))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (mapPm4Candidates.Count == 0)
            throw new InvalidOperationException($"PM4 export found no files for map '{mapName}'.");

        string exportRoot = Path.Combine(outputDirectory, SanitizePm4ExportPathSegment(mapName));
        Directory.CreateDirectory(exportRoot);

        var exportedTiles = new Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>>();
        var fileSummaries = new List<object>(mapPm4Candidates.Count);
        int exportedObjectCount = 0;
        int exportedTileCount = 0;
        int tileParseRejected = 0;
        int tileRangeRejected = 0;
            int readFailed = 0;
            int decodeFailed = 0;
            int zeroObjectFiles = 0;
            int memCacheHits = 0;
            int diskCacheHits = 0;
            int memCacheMisses = 0;

        foreach (string pm4Path in mapPm4Candidates)
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int fileTileX, out int fileTileY))
            {
                tileParseRejected++;
                fileSummaries.Add(new
                {
                    sourcePath = pm4Path,
                    tileParsed = false,
                    exported = false,
                    reason = "tile-parse-failed"
                });
                continue;
            }

            if (!TryMapPm4FileTileToTerrainTile(fileTileX, fileTileY, out int effectiveTileX, out int effectiveTileY))
            {
                tileRangeRejected++;
                fileSummaries.Add(new
                {
                    sourcePath = pm4Path,
                    tileParsed = true,
                    fileTileX,
                    fileTileY,
                    effectiveTileX = (int?)null,
                    effectiveTileY = (int?)null,
                    exported = false,
                    reason = "tile-out-of-range"
                });
                continue;
            }

            byte[]? bytes = _dataSource.ReadFile(pm4Path);
            if (bytes == null || bytes.Length == 0)
            {
                readFailed++;
                fileSummaries.Add(new
                {
                    sourcePath = pm4Path,
                    tileParsed = true,
                    fileTileX,
                    fileTileY,
                    effectiveTileX,
                    effectiveTileY,
                    exported = false,
                    reason = "read-failed"
                });
                continue;
            }

            try
            {
                Pm4File pm4 = CorePm4DocumentReader.Read(bytes, pm4Path);
                int remainingLineBudget = int.MaxValue;
                int remainingTriangleBudget = int.MaxValue;
                int rejectedLongEdges = 0;
                List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                    pm4,
                    pm4Path,
                    effectiveTileX,
                    effectiveTileY,
                    _pm4SplitCk24ByMscnRef,
                    _pm4SplitCk24ByConnectivity,
                    _pm4ShowPathWalls,
                    ref remainingLineBudget,
                    ref remainingTriangleBudget,
                    ref rejectedLongEdges,
                    out _);

                if (objects.Count == 0)
                    zeroObjectFiles++;

                if (exportedTiles.TryGetValue((effectiveTileX, effectiveTileY), out List<Pm4OverlayObject>? existingObjects))
                {
                    int objectPartOffset = existingObjects.Count;
                    objects = RebasePm4ObjectParts(objects, objectPartOffset);
                    existingObjects.AddRange(objects);
                }
                else
                {
                    exportedTiles[(effectiveTileX, effectiveTileY)] = objects;
                }

                exportedObjectCount += objects.Count;
                fileSummaries.Add(new
                {
                    sourcePath = pm4Path,
                    tileParsed = true,
                    fileTileX,
                    fileTileY,
                    effectiveTileX,
                    effectiveTileY,
                    exported = true,
                    version = pm4.Version,
                    meshVertexCount = pm4.KnownChunks.Msvt.Count,
                    meshIndexCount = pm4.KnownChunks.Msvi.Count,
                    surfaceCount = pm4.KnownChunks.Msur.Count,
                    ck24SurfaceCount = pm4.KnownChunks.Msur.Count(surface => surface.Ck24 != 0),
                    linkCount = pm4.KnownChunks.Mslk.Count,
                    positionRefCount = pm4.KnownChunks.Mprl.Count,
                    exportedObjectCount = objects.Count,
                    exportedLineCount = objects.Sum(static obj => obj.Lines.Count),
                    exportedTriangleCount = objects.Sum(static obj => obj.Triangles.Count),
                    rejectedLongEdges,
                    zeroObjects = objects.Count == 0
                });
            }
            catch (Exception ex)
            {
                decodeFailed++;
                fileSummaries.Add(new
                {
                    sourcePath = pm4Path,
                    tileParsed = true,
                    fileTileX,
                    fileTileY,
                    effectiveTileX,
                    effectiveTileY,
                    exported = false,
                    reason = "decode-failed",
                    error = ex.Message
                });
            }
        }

        var tileSummaries = new List<object>(exportedTiles.Count);
        foreach (var tileEntry in exportedTiles
            .OrderBy(static entry => entry.Key.tileX)
            .ThenBy(static entry => entry.Key.tileY))
        {
            exportedTileCount++;
            int tileX = tileEntry.Key.tileX;
            int tileY = tileEntry.Key.tileY;
            List<Pm4OverlayObject> objects = tileEntry.Value
                .OrderBy(static obj => obj.Ck24)
                .ThenBy(static obj => obj.ObjectPartId)
                .ToList();
            string tileDirectory = Path.Combine(exportRoot, $"tile_{tileX:D2}_{tileY:D2}");
            Directory.CreateDirectory(tileDirectory);

            string tileObjPath = Path.Combine(tileDirectory, $"tile_{tileX:D2}_{tileY:D2}.obj");
            File.WriteAllText(tileObjPath, BuildPm4ObjText(objects, tileX, tileY), Encoding.UTF8);

            foreach (Pm4OverlayObject obj in objects)
            {
                string fileName = $"ck24_{obj.Ck24:X6}_part_{obj.ObjectPartId:D4}_type_{obj.Ck24Type:X2}_obj_{obj.Ck24ObjectId:D5}.obj";
                string objectPath = Path.Combine(tileDirectory, fileName);
                File.WriteAllText(objectPath, BuildPm4ObjText(new[] { obj }, tileX, tileY), Encoding.UTF8);
            }

            tileSummaries.Add(new
            {
                tileX,
                tileY,
                tileObjPath,
                objectCount = objects.Count,
                lineCount = objects.Sum(static obj => obj.Lines.Count),
                triangleCount = objects.Sum(static obj => obj.Triangles.Count),
                ck24Count = objects.Select(static obj => obj.Ck24).Distinct().Count(),
                sourceFiles = objects.Select(static obj => obj.SourcePath).Distinct(StringComparer.OrdinalIgnoreCase).OrderBy(static path => path, StringComparer.OrdinalIgnoreCase).ToList()
            });
        }

        string manifestPath = Path.Combine(exportRoot, "pm4_obj_manifest.json");
        var manifest = new
        {
            generatedAtUtc = DateTime.UtcNow,
            mapName,
            exportRoot,
            splitCk24ByMscnRef = _pm4SplitCk24ByMscnRef,
            splitCk24ByConnectivity = _pm4SplitCk24ByConnectivity,
            includePathWalls = _pm4ShowPathWalls,
            summary = new
            {
                sourceFileCount = mapPm4Candidates.Count,
                exportedTileCount,
                exportedObjectCount,
                tileParseRejected,
                tileRangeRejected,
                readFailed,
                decodeFailed,
                zeroObjectFiles
            },
            tiles = tileSummaries,
            files = fileSummaries
        };
        File.WriteAllText(
            manifestPath,
            JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }),
            Encoding.UTF8);

        return new Pm4OfflineObjExportSummary(
            exportRoot,
            manifestPath,
            mapPm4Candidates.Count,
            exportedTileCount,
            exportedObjectCount,
            zeroObjectFiles,
            decodeFailed,
            readFailed);
    }

    internal Pm4WmoCorrelationReport BuildPm4WmoPlacementCorrelationReport(int maxMatchesPerPlacement = 8)
    {
        EnsurePm4OverlayMatchesCameraWindow(GetPm4LoadAnchorCameraPosition());

        if (_instancesDirty)
            RebuildInstanceLists();

        int resolvedMaxMatches = Math.Max(1, maxMatchesPerPlacement);
        List<CorePm4CorrelationObjectState> pm4Objects = BuildPm4CorrelationObjectStates();
        int mergedPm4ObjectCount = pm4Objects.Select(static candidate => candidate.GroupKey).Distinct().Count();
        Dictionary<int, ModfPlacement> modfByUniqueId = _terrainManager.Adapter.ModfPlacements
            .GroupBy(static placement => placement.UniqueId)
            .ToDictionary(static group => group.Key, static group => group.First());

        int placementCount = 0;
        int meshResolvedCount = 0;
        int placementsWithCandidates = 0;
        int placementsWithNearCandidates = 0;

        List<Pm4WmoCorrelationPlacement> placementReports = _tileWmoInstances
            .OrderBy(static kvp => kvp.Key.Item1)
            .ThenBy(static kvp => kvp.Key.Item2)
            .SelectMany(tileEntry => tileEntry.Value
                .OrderBy(static instance => instance.ModelPath, StringComparer.OrdinalIgnoreCase)
                .Select(instance =>
                {
                    placementCount++;

                    bool hasMeshSummary = _assets.TryGetWmoMeshSummary(instance.ModelKey, out WmoMeshSummary meshSummary);
                    if (hasMeshSummary)
                        meshResolvedCount++;

                    Vector3 worldBoundsMin = instance.BoundsMin;
                    Vector3 worldBoundsMax = instance.BoundsMax;
                    Vector2[] wmoFootprintHull = Array.Empty<Vector2>();
                    float wmoFootprintArea = 0f;
                    if (hasMeshSummary)
                    {
                        TransformBounds(meshSummary.BoundsMin, meshSummary.BoundsMax, instance.Transform, out worldBoundsMin, out worldBoundsMax);
                        wmoFootprintHull = CorePm4CorrelationMath.BuildTransformedFootprintHull(meshSummary.FootprintSampleVertices, instance.Transform);
                        wmoFootprintArea = CorePm4CorrelationMath.ComputeFootprintArea(wmoFootprintHull);
                    }

                    bool hasRawPlacement = modfByUniqueId.TryGetValue(instance.UniqueId, out ModfPlacement rawPlacement);

                    var candidateMetrics = pm4Objects
                        .Where(candidate => Math.Abs(candidate.TileX - tileEntry.Key.Item1) <= 1
                            && Math.Abs(candidate.TileY - tileEntry.Key.Item2) <= 1)
                        .Select(candidate =>
                        {
                            CorePm4CorrelationMetrics metrics = CorePm4CorrelationMath.EvaluateMetrics(
                                worldBoundsMin,
                                worldBoundsMax,
                                instance.PlacementPosition,
                                wmoFootprintHull,
                                wmoFootprintArea,
                                candidate.BoundsMin,
                                candidate.BoundsMax,
                                candidate.Center,
                                candidate.FootprintHull,
                                candidate.FootprintArea);

                            bool sameTile = candidate.TileX == tileEntry.Key.Item1 && candidate.TileY == tileEntry.Key.Item2;
                            CorePm4CorrelationCandidateScore score = new(
                                sameTile,
                                metrics,
                                candidate.BoundsMin,
                                candidate.BoundsMax,
                                candidate.Center);

                            return new
                            {
                                candidate,
                                score,
                            };
                        })
                        .GroupBy(static candidate => candidate.candidate.GroupKey)
                        .Select(group => group
                            .OrderBy(static candidate => candidate.score, Comparer<CorePm4CorrelationCandidateScore>.Create(CorePm4CorrelationMath.CompareCandidateScores))
                            .First())
                        .OrderBy(static candidate => candidate.score, Comparer<CorePm4CorrelationCandidateScore>.Create(CorePm4CorrelationMath.CompareCandidateScores))
                        .ToList();

                    if (candidateMetrics.Count > 0)
                        placementsWithCandidates++;

                    int nearCandidateCount = candidateMetrics.Count(candidate => candidate.score.Metrics.PlanarGap <= 32f && candidate.score.Metrics.VerticalGap <= 64f);
                    if (nearCandidateCount > 0)
                        placementsWithNearCandidates++;

                    Pm4WmoCorrelationAdtPlacementInfo adtPlacementInfo = new(
                        hasRawPlacement,
                        hasRawPlacement ? rawPlacement.Flags : (ushort)0,
                        hasRawPlacement ? rawPlacement.BoundsMin : Vector3.Zero,
                        hasRawPlacement ? rawPlacement.BoundsMax : Vector3.Zero);

                    Pm4WmoCorrelationMeshInfo wmoMeshInfo = hasMeshSummary
                        ? new Pm4WmoCorrelationMeshInfo(
                            true,
                            meshSummary.Version,
                            meshSummary.GroupCount,
                            meshSummary.VertexCount,
                            meshSummary.IndexCount,
                            meshSummary.TriangleCount,
                            meshSummary.BatchCount,
                            meshSummary.BoundsMin,
                            meshSummary.BoundsMax,
                            meshSummary.FootprintSampleCount,
                            wmoFootprintHull.Length,
                            wmoFootprintArea)
                        : new Pm4WmoCorrelationMeshInfo(
                            false,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            Vector3.Zero,
                            Vector3.Zero,
                            0,
                            0,
                            0f);

                    List<Pm4WmoCorrelationMatch> matches = candidateMetrics
                        .Take(resolvedMaxMatches)
                        .Select(candidate => new Pm4WmoCorrelationMatch(
                            candidate.candidate.TileX,
                            candidate.candidate.TileY,
                            candidate.candidate.Object.Ck24,
                            candidate.candidate.Object.Ck24Type,
                            candidate.candidate.Object.Ck24ObjectId,
                            candidate.candidate.Object.ObjectPartId,
                            candidate.candidate.Object.LinkGroupObjectId,
                            candidate.candidate.Object.SurfaceCount,
                            candidate.candidate.Object.LinkedPositionRefCount,
                            candidate.candidate.Object.DominantGroupKey,
                            candidate.candidate.Object.DominantAttributeMask,
                            candidate.candidate.Object.DominantMscnRefIndex,
                            candidate.candidate.Object.AverageSurfaceHeight,
                            candidate.score.SameTile,
                            candidate.score.Metrics.PlanarGap,
                            candidate.score.Metrics.VerticalGap,
                            candidate.score.Metrics.CenterDistance,
                            candidate.score.Metrics.PlanarOverlapRatio,
                            candidate.score.Metrics.VolumeOverlapRatio,
                            candidate.score.Metrics.FootprintOverlapRatio,
                            candidate.score.Metrics.FootprintAreaRatio,
                            candidate.score.Metrics.FootprintDistance,
                            candidate.candidate.BoundsMin,
                            candidate.candidate.BoundsMax,
                            candidate.candidate.Center))
                        .ToList();

                    return new Pm4WmoCorrelationPlacement(
                        tileEntry.Key.Item1,
                        tileEntry.Key.Item2,
                        instance.UniqueId,
                        instance.ModelName,
                        instance.ModelPath,
                        instance.ModelKey,
                        instance.PlacementPosition,
                        instance.PlacementRotation,
                        instance.PlacementScale,
                        adtPlacementInfo,
                        worldBoundsMin,
                        worldBoundsMax,
                        wmoMeshInfo,
                        candidateMetrics.Count,
                        nearCandidateCount,
                        matches);
                }))
            .ToList();

        return new Pm4WmoCorrelationReport(
            DateTime.UtcNow,
            _pm4Status,
            new Pm4WmoCorrelationSummary(
                placementCount,
                meshResolvedCount,
                mergedPm4ObjectCount,
                placementsWithCandidates,
                placementsWithNearCandidates,
                resolvedMaxMatches),
            placementReports);
    }

    public string BuildPm4WmoPlacementCorrelationJson(int maxMatchesPerPlacement = 8)
    {
        static float[] VectorToArray(Vector3 value) => new[] { value.X, value.Y, value.Z };

        Pm4WmoCorrelationReport report = BuildPm4WmoPlacementCorrelationReport(maxMatchesPerPlacement);
        var payload = new
        {
            generatedAtUtc = report.GeneratedAtUtc,
            pm4Status = report.Pm4Status,
            summary = new
            {
                wmoPlacementCount = report.Summary.WmoPlacementCount,
                wmoMeshResolvedCount = report.Summary.WmoMeshResolvedCount,
                pm4ObjectCount = report.Summary.Pm4ObjectCount,
                placementsWithCandidates = report.Summary.PlacementsWithCandidates,
                placementsWithNearCandidates = report.Summary.PlacementsWithNearCandidates,
                maxMatchesPerPlacement = report.Summary.MaxMatchesPerPlacement,
            },
            placements = report.Placements.Select(placement => new
            {
                tileX = placement.TileX,
                tileY = placement.TileY,
                uniqueId = placement.UniqueId,
                modelName = placement.ModelName,
                modelPath = placement.ModelPath,
                modelKey = placement.ModelKey,
                placementPosition = VectorToArray(placement.PlacementPosition),
                placementRotation = VectorToArray(placement.PlacementRotation),
                placementScale = JsonFiniteOrNull(placement.PlacementScale),
                adtPlacement = new
                {
                    found = placement.AdtPlacement.Found,
                    flags = placement.AdtPlacement.Flags,
                    rawBoundsMin = VectorToArray(placement.AdtPlacement.RawBoundsMin),
                    rawBoundsMax = VectorToArray(placement.AdtPlacement.RawBoundsMax),
                },
                worldBoundsMin = VectorToArray(placement.WorldBoundsMin),
                worldBoundsMax = VectorToArray(placement.WorldBoundsMax),
                wmoMesh = new
                {
                    available = placement.WmoMesh.Available,
                    version = placement.WmoMesh.Version,
                    groupCount = placement.WmoMesh.GroupCount,
                    vertexCount = placement.WmoMesh.VertexCount,
                    indexCount = placement.WmoMesh.IndexCount,
                    triangleCount = placement.WmoMesh.TriangleCount,
                    batchCount = placement.WmoMesh.BatchCount,
                    localBoundsMin = VectorToArray(placement.WmoMesh.LocalBoundsMin),
                    localBoundsMax = VectorToArray(placement.WmoMesh.LocalBoundsMax),
                    footprintSampleCount = placement.WmoMesh.FootprintSampleCount,
                    worldFootprintHullPointCount = placement.WmoMesh.WorldFootprintHullPointCount,
                    worldFootprintArea = JsonFiniteOrNull(placement.WmoMesh.WorldFootprintArea),
                },
                pm4CandidateCount = placement.Pm4CandidateCount,
                pm4NearCandidateCount = placement.Pm4NearCandidateCount,
                pm4Matches = placement.Pm4Matches.Select(match => new
                {
                    tileX = match.TileX,
                    tileY = match.TileY,
                    ck24 = match.Ck24,
                    ck24Type = match.Ck24Type,
                    ck24ObjectId = match.Ck24ObjectId,
                    objectPartId = match.ObjectPartId,
                    linkGroupObjectId = match.LinkGroupObjectId,
                    surfaceCount = match.SurfaceCount,
                    linkedPositionRefCount = match.LinkedPositionRefCount,
                    dominantGroupKey = match.DominantGroupKey,
                    dominantAttributeMask = match.DominantAttributeMask,
                    dominantMscnRefIndex = match.DominantMscnRefIndex,
                    averageSurfaceHeight = JsonFiniteOrNull(match.AverageSurfaceHeight),
                    sameTile = match.SameTile,
                    planarGap = JsonFiniteOrNull(match.PlanarGap),
                    verticalGap = JsonFiniteOrNull(match.VerticalGap),
                    centerDistance = JsonFiniteOrNull(match.CenterDistance),
                    planarOverlapRatio = JsonFiniteOrNull(match.PlanarOverlapRatio),
                    volumeOverlapRatio = JsonFiniteOrNull(match.VolumeOverlapRatio),
                    footprintOverlapRatio = JsonFiniteOrNull(match.FootprintOverlapRatio),
                    footprintAreaRatio = JsonFiniteOrNull(match.FootprintAreaRatio),
                    footprintDistance = JsonFiniteOrNull(match.FootprintDistance),
                    boundsMin = VectorToArray(match.BoundsMin),
                    boundsMax = VectorToArray(match.BoundsMax),
                    center = VectorToArray(match.Center),
                }).ToList(),
            }).ToList(),
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true,
        });
    }

    internal Pm4ObjectMatchReport BuildPm4ObjectMatchReport(int maxMatchesPerObject = 8)
    {
        EnsurePm4OverlayMatchesCameraWindow(GetPm4LoadAnchorCameraPosition());

        if (_instancesDirty)
            RebuildInstanceLists();

        int resolvedMaxMatches = Math.Max(1, maxMatchesPerObject);
        List<Pm4ObjectMatchState> pm4Objects = BuildPm4ObjectMatchStates();
        List<Pm4PlacementMatchState> placements = BuildPm4PlacementMatchStates();
        List<Pm4AssetProfileState> assetProfiles = BuildPm4AssetProfileStates(placements);

        int objectsWithCandidates = 0;
        int objectsWithNearCandidates = 0;
        List<Pm4ObjectMatchObject> reports = new(pm4Objects.Count);

        foreach (Pm4ObjectMatchState pm4Object in pm4Objects)
        {
            Pm4ObjectMatchObject report = BuildPm4ObjectMatchObject(pm4Object, placements, assetProfiles, resolvedMaxMatches);
            if (report.CandidateCount > 0)
                objectsWithCandidates++;

            if (report.NearCandidateCount > 0)
                objectsWithNearCandidates++;

            reports.Add(report);
        }

        return new Pm4ObjectMatchReport(
            DateTime.UtcNow,
            _terrainManager.MapName,
            _pm4Status,
            new Pm4ObjectMatchSummary(
                pm4Objects.Count,
                placements.Count(static placement => placement.Kind == "wmo"),
                placements.Count(static placement => placement.Kind == "m2"),
                objectsWithCandidates,
                objectsWithNearCandidates,
                resolvedMaxMatches),
            reports);
    }

    internal bool TryBuildSelectedPm4ObjectMatch(int maxMatchesPerObject, out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        EnsurePm4OverlayMatchesCameraWindow(GetPm4LoadAnchorCameraPosition());

        if (_instancesDirty)
            RebuildInstanceLists();

        if (!_selectedPm4ObjectKey.HasValue)
            return false;

        var objectKey = _selectedPm4ObjectKey.Value;
        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        Pm4ObjectMatchState pm4Object = BuildPm4ObjectMatchState(objectKey.tileX, objectKey.tileY, objectKey, obj);
        List<Pm4PlacementMatchState> placements = BuildPm4PlacementMatchStates();
        List<Pm4AssetProfileState> assetProfiles = BuildPm4AssetProfileStates(placements);
        objectMatch = BuildPm4ObjectMatchObject(pm4Object, placements, assetProfiles, Math.Max(1, maxMatchesPerObject));
        return true;
    }

    private List<Pm4ObjectMatchState> BuildPm4ObjectMatchStates()
    {
        List<Pm4ObjectMatchState> states = new(_pm4ObjectLookup.Count);

        foreach (var tileEntry in _pm4TileObjects)
        {
            foreach (Pm4OverlayObject obj in tileEntry.Value)
            {
                var objectKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                states.Add(BuildPm4ObjectMatchState(tileEntry.Key.tileX, tileEntry.Key.tileY, objectKey, obj));
            }
        }

        return states;
    }

    private Pm4ObjectMatchState BuildPm4ObjectMatchState(
        int tileX,
        int tileY,
        (int tileX, int tileY, uint ck24, int objectPart) objectKey,
        Pm4OverlayObject obj)
    {
        bool applyPm4Transform = !IsNearZeroVector(_pm4OverlayTranslation)
            || !IsNearZeroVector(_pm4OverlayRotationDegrees)
            || !IsNearOneVector(_pm4OverlayScale);
        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
        Vector3 boundsMin = obj.BoundsMin;
        Vector3 boundsMax = obj.BoundsMax;
        Vector3 center = obj.Center;
        Vector3 placementAnchor = obj.PlacementAnchor;
        if (applyObjectTransform)
        {
            TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);
            center = ApplyPm4OverlayTransform(obj.Center, objectTransform);
            placementAnchor = ApplyPm4OverlayTransform(obj.PlacementAnchor, objectTransform);
        }

        Vector2[] footprintHull = BuildPm4BoundsFootprintHull(boundsMin, boundsMax);
        float footprintArea = CorePm4CorrelationMath.ComputeFootprintArea(footprintHull);
        Pm4ShapeSignature shapeSignature = BuildPm4ShapeSignature(boundsMin, boundsMax, footprintHull);
        return new Pm4ObjectMatchState(tileX, tileY, objectKey, obj, placementAnchor, boundsMin, boundsMax, center, footprintHull, footprintArea, shapeSignature);
    }

    private List<Pm4PlacementMatchState> BuildPm4PlacementMatchStates()
    {
        Dictionary<int, ModfPlacement> modfByUniqueId = _terrainManager.Adapter.ModfPlacements
            .GroupBy(static placement => placement.UniqueId)
            .ToDictionary(static group => group.Key, static group => group.First());
        List<Pm4PlacementMatchState> states = new(_tileWmoInstances.Count * 4 + _tileMdxInstances.Count * 4);

        foreach (var tileEntry in _tileWmoInstances)
        {
            foreach (ObjectInstance instance in tileEntry.Value)
            {
                bool hasMeshSummary = _assets.TryGetWmoMeshSummary(instance.ModelKey, out WmoMeshSummary meshSummary);
                Vector3 worldBoundsMin = instance.BoundsMin;
                Vector3 worldBoundsMax = instance.BoundsMax;
                Vector2[] footprintHull = BuildPm4BoundsFootprintHull(worldBoundsMin, worldBoundsMax);
                float footprintArea = CorePm4CorrelationMath.ComputeFootprintArea(footprintHull);
                Vector3 localBoundsMin = instance.LocalBoundsMin;
                Vector3 localBoundsMax = instance.LocalBoundsMax;
                Vector2[] localFootprintHull = instance.BoundsResolved
                    ? BuildPm4BoundsFootprintHull(localBoundsMin, localBoundsMax)
                    : BuildPm4BoundsFootprintHull(worldBoundsMin, worldBoundsMax);
                int meshGroupCount = 0;
                int meshVertexCount = 0;
                int meshTriangleCount = 0;
                int footprintSampleCount = 0;
                float worldFootprintArea = footprintArea;
                string evidenceSource = "modf-bounds";
                var geometryVariants = new List<Pm4PlacementGeometryVariant>();

                if (hasMeshSummary)
                {
                    TransformBounds(meshSummary.BoundsMin, meshSummary.BoundsMax, instance.Transform, out worldBoundsMin, out worldBoundsMax);
                    footprintHull = CorePm4CorrelationMath.BuildTransformedFootprintHull(meshSummary.FootprintSampleVertices, instance.Transform);
                    footprintArea = CorePm4CorrelationMath.ComputeFootprintArea(footprintHull);
                    localBoundsMin = meshSummary.BoundsMin;
                    localBoundsMax = meshSummary.BoundsMax;
                    localFootprintHull = meshSummary.FootprintSampleVertices.Length > 0
                        ? CorePm4CorrelationMath.BuildFootprintHull(meshSummary.FootprintSampleVertices)
                        : BuildPm4BoundsFootprintHull(localBoundsMin, localBoundsMax);
                    meshGroupCount = meshSummary.GroupCount;
                    meshVertexCount = meshSummary.VertexCount;
                    meshTriangleCount = meshSummary.TriangleCount;
                    footprintSampleCount = meshSummary.FootprintSampleCount;
                    worldFootprintArea = footprintArea;
                    evidenceSource = "wmo-mesh";
                }

                geometryVariants.Add(new Pm4PlacementGeometryVariant(
                    BuildPm4AssetProfileKey("wmo", instance.ModelKey, evidenceSource, null),
                    evidenceSource,
                    worldBoundsMin,
                    worldBoundsMax,
                    footprintHull,
                    footprintArea,
                    meshGroupCount,
                    meshVertexCount,
                    meshTriangleCount,
                    footprintSampleCount,
                    worldFootprintArea,
                    BuildPm4ShapeSignature(localBoundsMin, localBoundsMax, localFootprintHull),
                    null));

                if (hasMeshSummary && meshSummary.GroupSummaries.Length > 1)
                {
                    foreach (WmoGroupMeshSummary groupSummary in meshSummary.GroupSummaries)
                    {
                        if (groupSummary.VertexCount <= 0 || groupSummary.TriangleCount <= 0)
                            continue;

                        TransformBounds(groupSummary.BoundsMin, groupSummary.BoundsMax, instance.Transform, out Vector3 groupWorldBoundsMin, out Vector3 groupWorldBoundsMax);
                        Vector2[] groupFootprintHull = groupSummary.FootprintSampleVertices.Length > 0
                            ? CorePm4CorrelationMath.BuildTransformedFootprintHull(groupSummary.FootprintSampleVertices, instance.Transform)
                            : BuildPm4BoundsFootprintHull(groupWorldBoundsMin, groupWorldBoundsMax);
                        float groupFootprintArea = CorePm4CorrelationMath.ComputeFootprintArea(groupFootprintHull);
                        Vector2[] groupLocalFootprintHull = groupSummary.FootprintSampleVertices.Length > 0
                            ? CorePm4CorrelationMath.BuildFootprintHull(groupSummary.FootprintSampleVertices)
                            : BuildPm4BoundsFootprintHull(groupSummary.BoundsMin, groupSummary.BoundsMax);
                        byte? correlatedGroupKey = groupSummary.GroupIndex <= byte.MaxValue ? (byte)groupSummary.GroupIndex : null;
                        geometryVariants.Add(new Pm4PlacementGeometryVariant(
                            BuildPm4AssetProfileKey("wmo", instance.ModelKey, "wmo-group-mesh", correlatedGroupKey),
                            "wmo-group-mesh",
                            groupWorldBoundsMin,
                            groupWorldBoundsMax,
                            groupFootprintHull,
                            groupFootprintArea,
                            1,
                            groupSummary.VertexCount,
                            groupSummary.TriangleCount,
                            groupSummary.FootprintSampleCount,
                            groupFootprintArea,
                            BuildPm4ShapeSignature(groupSummary.BoundsMin, groupSummary.BoundsMax, groupLocalFootprintHull),
                            correlatedGroupKey));
                    }
                }

                ushort flags = modfByUniqueId.TryGetValue(instance.UniqueId, out ModfPlacement rawPlacement)
                    ? rawPlacement.Flags
                    : (ushort)0;

                states.Add(new Pm4PlacementMatchState(
                    tileEntry.Key.Item1,
                    tileEntry.Key.Item2,
                    "wmo",
                    instance.UniqueId,
                    instance.ModelName,
                    instance.ModelPath,
                    instance.ModelKey,
                    geometryVariants[0].AssetProfileKey,
                    true,
                    evidenceSource,
                    flags,
                    instance.PlacementPosition,
                    instance.PlacementRotation,
                    instance.PlacementScale,
                    worldBoundsMin,
                    worldBoundsMax,
                    footprintHull,
                    footprintArea,
                    meshGroupCount,
                    meshVertexCount,
                    meshTriangleCount,
                    footprintSampleCount,
                    worldFootprintArea,
                    geometryVariants));
            }
        }

        foreach (var tileEntry in _tileMdxInstances)
        {
            foreach (ObjectInstance instance in tileEntry.Value)
            {
                Vector3 worldBoundsMin = instance.BoundsMin;
                Vector3 worldBoundsMax = instance.BoundsMax;
                Vector2[] footprintHull = BuildPm4BoundsFootprintHull(worldBoundsMin, worldBoundsMax);
                float footprintArea = CorePm4CorrelationMath.ComputeFootprintArea(footprintHull);
                Vector3 localBoundsMin = instance.BoundsResolved ? instance.LocalBoundsMin : worldBoundsMin;
                Vector3 localBoundsMax = instance.BoundsResolved ? instance.LocalBoundsMax : worldBoundsMax;
                Vector2[] localFootprintHull = BuildPm4BoundsFootprintHull(localBoundsMin, localBoundsMax);
                var geometryVariants = new List<Pm4PlacementGeometryVariant>
                {
                    new(
                        BuildPm4AssetProfileKey("m2", instance.ModelKey, "instance-bounds", null),
                        "instance-bounds",
                        worldBoundsMin,
                        worldBoundsMax,
                        footprintHull,
                        footprintArea,
                        0,
                        0,
                        0,
                        0,
                        footprintArea,
                        BuildPm4ShapeSignature(localBoundsMin, localBoundsMax, localFootprintHull),
                        null)
                };

                if (_assets.TryGetMdxCollisionSummary(instance.ModelKey, out MdxCollisionMeshSummary collisionSummary))
                {
                    TransformBounds(collisionSummary.BoundsMin, collisionSummary.BoundsMax, instance.Transform, out Vector3 collisionWorldBoundsMin, out Vector3 collisionWorldBoundsMax);
                    Vector2[] collisionFootprintHull = collisionSummary.FootprintSampleVertices.Length > 0
                        ? CorePm4CorrelationMath.BuildTransformedFootprintHull(collisionSummary.FootprintSampleVertices, instance.Transform)
                        : BuildPm4BoundsFootprintHull(collisionWorldBoundsMin, collisionWorldBoundsMax);
                    float collisionFootprintArea = CorePm4CorrelationMath.ComputeFootprintArea(collisionFootprintHull);
                    Vector2[] collisionLocalFootprintHull = collisionSummary.FootprintSampleVertices.Length > 0
                        ? CorePm4CorrelationMath.BuildFootprintHull(collisionSummary.FootprintSampleVertices)
                        : BuildPm4BoundsFootprintHull(collisionSummary.BoundsMin, collisionSummary.BoundsMax);
                    geometryVariants.Add(new Pm4PlacementGeometryVariant(
                        BuildPm4AssetProfileKey("m2", instance.ModelKey, "mdx-collision", null),
                        "mdx-collision",
                        collisionWorldBoundsMin,
                        collisionWorldBoundsMax,
                        collisionFootprintHull,
                        collisionFootprintArea,
                        0,
                        collisionSummary.VertexCount,
                        collisionSummary.TriangleCount,
                        collisionSummary.FootprintSampleCount,
                        collisionFootprintArea,
                        BuildPm4ShapeSignature(collisionSummary.BoundsMin, collisionSummary.BoundsMax, collisionLocalFootprintHull),
                        null));
                }

                states.Add(new Pm4PlacementMatchState(
                    tileEntry.Key.Item1,
                    tileEntry.Key.Item2,
                    "m2",
                    instance.UniqueId,
                    instance.ModelName,
                    instance.ModelPath,
                    instance.ModelKey,
                    geometryVariants[0].AssetProfileKey,
                    true,
                    "instance-bounds",
                    0,
                    instance.PlacementPosition,
                    instance.PlacementRotation,
                    instance.PlacementScale,
                    worldBoundsMin,
                    worldBoundsMax,
                    footprintHull,
                    footprintArea,
                    0,
                    0,
                    0,
                    0,
                    footprintArea,
                    geometryVariants));
            }
        }

        return states;
    }

    private static List<Pm4AssetProfileState> BuildPm4AssetProfileStates(IReadOnlyList<Pm4PlacementMatchState> placements)
    {
        Dictionary<string, Pm4AssetProfileState> profiles = new(StringComparer.OrdinalIgnoreCase);

        foreach (Pm4PlacementMatchState placement in placements)
        {
            for (int index = 0; index < placement.GeometryVariants.Count; index++)
            {
                Pm4PlacementGeometryVariant variant = placement.GeometryVariants[index];
                if (string.IsNullOrWhiteSpace(variant.AssetProfileKey))
                    continue;

                var profile = new Pm4AssetProfileState(
                    variant.AssetProfileKey,
                    placement.Kind,
                    placement.ModelName,
                    placement.ModelPath,
                    placement.ModelKey,
                    variant.EvidenceSource,
                    variant.CorrelatedGroupKey,
                    variant.MeshGroupCount,
                    variant.MeshVertexCount,
                    variant.MeshTriangleCount,
                    variant.FootprintSampleCount,
                    variant.ShapeSignature);

                if (!profiles.TryGetValue(variant.AssetProfileKey, out Pm4AssetProfileState existingProfile)
                    || ComparePm4AssetProfileRichness(profile, existingProfile) < 0)
                {
                    profiles[variant.AssetProfileKey] = profile;
                }
            }
        }

        return profiles.Values.ToList();
    }

    private static Pm4ObjectMatchObject BuildPm4ObjectMatchObject(
        Pm4ObjectMatchState pm4Object,
        IReadOnlyList<Pm4PlacementMatchState> placements,
        IReadOnlyList<Pm4AssetProfileState> assetProfiles,
        int maxMatchesPerObject)
    {
        HashSet<string>? preferredAssetProfileKeys = ResolvePreferredPm4AssetProfileKeys(pm4Object, assetProfiles, maxMatchesPerObject);

        List<Pm4PlacementMatchEvaluation> evaluatedCandidates = placements
            .Where(placement => Math.Abs(placement.TileX - pm4Object.TileX) <= 1
                && Math.Abs(placement.TileY - pm4Object.TileY) <= 1)
            .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, preferredAssetProfileKeys))
            .Where(static candidate => candidate.HasValue)
            .Select(static candidate => candidate!.Value)
            .ToList();

        if (evaluatedCandidates.Count == 0)
        {
            evaluatedCandidates = placements
                .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, preferredAssetProfileKeys))
                .Where(static candidate => candidate.HasValue)
                .Select(static candidate => candidate!.Value)
                .ToList();
        }

        if (evaluatedCandidates.Count == 0)
        {
            evaluatedCandidates = placements
                .Where(placement => Math.Abs(placement.TileX - pm4Object.TileX) <= 1
                    && Math.Abs(placement.TileY - pm4Object.TileY) <= 1)
                .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, null))
                .Where(static candidate => candidate.HasValue)
                .Select(static candidate => candidate!.Value)
                .ToList();
        }

        if (evaluatedCandidates.Count == 0)
        {
            evaluatedCandidates = placements
                .Select(placement => EvaluatePm4PlacementMatch(pm4Object, placement, null))
                .Where(static candidate => candidate.HasValue)
                .Select(static candidate => candidate!.Value)
                .ToList();
        }

        List<Pm4PlacementMatchEvaluation> rankedCandidates = evaluatedCandidates
            .OrderBy(candidate => new CorePm4CorrelationCandidateScore(
                    candidate.Placement.SameTile(pm4Object.TileX, pm4Object.TileY),
                    candidate.Metrics,
                    candidate.Placement.WorldBoundsMin,
                    candidate.Placement.WorldBoundsMax,
                    candidate.Placement.Center),
                Comparer<CorePm4CorrelationCandidateScore>.Create(CorePm4CorrelationMath.CompareCandidateScores))
            .ThenBy(candidate => pm4Object.Object.LinkedPositionRefCount > 0 ? candidate.AnchorPlanarGap : float.MaxValue)
            .ThenBy(candidate => GetPm4ObjectMatchEvidenceRank(pm4Object, candidate.Placement))
            .ToList();

        int nearCandidateCount = rankedCandidates.Count(static candidate =>
            candidate.Metrics.PlanarOverlapRatio > 0f
            || candidate.Metrics.VolumeOverlapRatio > 0f
            || candidate.AnchorPlanarGap <= 64f
            || (candidate.Metrics.PlanarGap <= 32f && candidate.Metrics.VerticalGap <= 96f));

        List<Pm4ObjectMatchCandidate> candidates = rankedCandidates
            .Take(maxMatchesPerObject)
            .Select(candidate => new Pm4ObjectMatchCandidate(
                candidate.Placement.TileX,
                candidate.Placement.TileY,
                candidate.Placement.Kind,
                candidate.Placement.UniqueId,
                candidate.Placement.ModelName,
                candidate.Placement.ModelPath,
                candidate.Placement.ModelKey,
                candidate.Placement.SameTile(pm4Object.TileX, pm4Object.TileY),
                candidate.Placement.AssetResolved,
                candidate.Placement.EvidenceSource,
                candidate.Placement.PlacementFlags,
                candidate.Placement.PlacementPosition,
                candidate.Placement.PlacementRotation,
                candidate.Placement.PlacementScale,
                candidate.AnchorPlanarGap,
                candidate.Metrics.PlanarGap,
                candidate.Metrics.VerticalGap,
                candidate.Metrics.CenterDistance,
                candidate.Metrics.PlanarOverlapRatio,
                candidate.Metrics.VolumeOverlapRatio,
                candidate.Metrics.FootprintOverlapRatio,
                candidate.Metrics.FootprintAreaRatio,
                candidate.Metrics.FootprintDistance,
                candidate.Placement.WorldBoundsMin,
                candidate.Placement.WorldBoundsMax,
                candidate.Placement.Center,
                candidate.Placement.MeshGroupCount,
                candidate.Placement.MeshVertexCount,
                candidate.Placement.MeshTriangleCount,
                candidate.Placement.FootprintSampleCount,
                candidate.Placement.WorldFootprintArea))
            .ToList();

        return new Pm4ObjectMatchObject(
            pm4Object.TileX,
            pm4Object.TileY,
            pm4Object.Object.Ck24,
            pm4Object.Object.Ck24Type,
            pm4Object.Object.Ck24ObjectId,
            pm4Object.Object.ObjectPartId,
            pm4Object.Object.LinkGroupObjectId,
            pm4Object.Object.SurfaceCount,
            pm4Object.Object.LinkedPositionRefCount,
            pm4Object.Object.DominantGroupKey,
            pm4Object.Object.DominantAttributeMask,
            pm4Object.Object.DominantMscnRefIndex,
            pm4Object.Object.AverageSurfaceHeight,
            pm4Object.Object.LinkedPositionRefSummary,
            pm4Object.PlacementAnchor,
            pm4Object.BoundsMin,
            pm4Object.BoundsMax,
            pm4Object.Center,
            rankedCandidates.Count,
            nearCandidateCount,
            rankedCandidates.Count(candidate => candidate.Placement.Kind == "wmo"),
            rankedCandidates.Count(candidate => candidate.Placement.Kind == "m2"),
            candidates);
    }

    private static HashSet<string>? ResolvePreferredPm4AssetProfileKeys(
        Pm4ObjectMatchState pm4Object,
        IReadOnlyList<Pm4AssetProfileState> assetProfiles,
        int maxMatchesPerObject)
    {
        if (assetProfiles.Count == 0)
            return null;

        int shortlistSize = Math.Clamp(maxMatchesPerObject * 6, 12, 48);
        List<Pm4AssetProfileMatchEvaluation> rankedProfiles = assetProfiles
            .Select(profile => new Pm4AssetProfileMatchEvaluation(profile, EvaluatePm4AssetProfileMetrics(pm4Object, profile)))
            .OrderBy(evaluation => evaluation, Comparer<Pm4AssetProfileMatchEvaluation>.Create((left, right) => ComparePm4AssetProfiles(pm4Object, left, right)))
            .Take(shortlistSize)
            .ToList();

        if (rankedProfiles.Count == 0)
            return null;

        HashSet<string> preferredKeys = new(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index < rankedProfiles.Count; index++)
            preferredKeys.Add(rankedProfiles[index].Profile.AssetProfileKey);

        return preferredKeys;
    }

    private static CorePm4CorrelationMetrics EvaluatePm4AssetProfileMetrics(Pm4ObjectMatchState pm4Object, Pm4AssetProfileState profile)
    {
        return CorePm4CorrelationMath.EvaluateMetrics(
            pm4Object.ShapeSignature.BoundsMin,
            pm4Object.ShapeSignature.BoundsMax,
            Vector3.Zero,
            pm4Object.ShapeSignature.FootprintHull,
            pm4Object.ShapeSignature.FootprintArea,
            profile.ShapeSignature.BoundsMin,
            profile.ShapeSignature.BoundsMax,
            Vector3.Zero,
            profile.ShapeSignature.FootprintHull,
            profile.ShapeSignature.FootprintArea);
    }

    private static int ComparePm4AssetProfiles(
        Pm4ObjectMatchState pm4Object,
        Pm4AssetProfileMatchEvaluation left,
        Pm4AssetProfileMatchEvaluation right)
    {
        bool leftGroupMatch = left.Profile.CorrelatedGroupKey.HasValue && left.Profile.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        bool rightGroupMatch = right.Profile.CorrelatedGroupKey.HasValue && right.Profile.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        int compareGroupMatch = rightGroupMatch.CompareTo(leftGroupMatch);
        if (compareGroupMatch != 0)
            return compareGroupMatch;

        int compareScore = CorePm4CorrelationMath.CompareCandidateScores(
            new CorePm4CorrelationCandidateScore(false, left.Metrics, left.Profile.ShapeSignature.BoundsMin, left.Profile.ShapeSignature.BoundsMax, Vector3.Zero),
            new CorePm4CorrelationCandidateScore(false, right.Metrics, right.Profile.ShapeSignature.BoundsMin, right.Profile.ShapeSignature.BoundsMax, Vector3.Zero));
        if (compareScore != 0)
            return compareScore;

        int compareEvidence = GetPlacementGeometryEvidenceRank(left.Profile.EvidenceSource).CompareTo(GetPlacementGeometryEvidenceRank(right.Profile.EvidenceSource));
        if (compareEvidence != 0)
            return compareEvidence;

        return right.Profile.MeshTriangleCount.CompareTo(left.Profile.MeshTriangleCount);
    }

    private static int ComparePm4AssetProfileRichness(Pm4AssetProfileState left, Pm4AssetProfileState right)
    {
        int compareEvidence = GetPlacementGeometryEvidenceRank(left.EvidenceSource).CompareTo(GetPlacementGeometryEvidenceRank(right.EvidenceSource));
        if (compareEvidence != 0)
            return compareEvidence;

        int compareTriangles = right.MeshTriangleCount.CompareTo(left.MeshTriangleCount);
        if (compareTriangles != 0)
            return compareTriangles;

        return right.FootprintSampleCount.CompareTo(left.FootprintSampleCount);
    }

    private static Pm4PlacementMatchEvaluation? EvaluatePm4PlacementMatch(
        Pm4ObjectMatchState pm4Object,
        Pm4PlacementMatchState placement,
        ISet<string>? preferredAssetProfileKeys)
    {
        if (!TryResolveBestPlacementGeometryVariant(pm4Object, placement, preferredAssetProfileKeys, out Pm4PlacementMatchState effectivePlacement, out CorePm4CorrelationMetrics metrics))
            return null;

        float anchorPlanarGap = ComputePm4ObjectAnchorPlanarGap(pm4Object.PlacementAnchor, effectivePlacement.PlacementPosition);
        return new Pm4PlacementMatchEvaluation(effectivePlacement, anchorPlanarGap, metrics);
    }

    private static bool TryResolveBestPlacementGeometryVariant(
        Pm4ObjectMatchState pm4Object,
        Pm4PlacementMatchState placement,
        ISet<string>? preferredAssetProfileKeys,
        out Pm4PlacementMatchState resolvedPlacement,
        out CorePm4CorrelationMetrics metrics)
    {
        IReadOnlyList<Pm4PlacementGeometryVariant> variants = placement.GeometryVariants;
        List<Pm4PlacementGeometryVariant>? filteredVariants = null;
        if (preferredAssetProfileKeys != null)
        {
            filteredVariants = variants
                .Where(variant => preferredAssetProfileKeys.Contains(variant.AssetProfileKey))
                .ToList();
            if (filteredVariants.Count > 0)
                variants = filteredVariants;
        }

        if (variants.Count == 0)
        {
            metrics = CorePm4CorrelationMath.EvaluateMetrics(
                pm4Object.BoundsMin,
                pm4Object.BoundsMax,
                pm4Object.Center,
                pm4Object.FootprintHull,
                pm4Object.FootprintArea,
                placement.WorldBoundsMin,
                placement.WorldBoundsMax,
                placement.Center,
                placement.FootprintHull,
                placement.FootprintArea);
            resolvedPlacement = placement;
            return preferredAssetProfileKeys == null;
        }

        bool sameTile = placement.SameTile(pm4Object.TileX, pm4Object.TileY);
        Pm4PlacementGeometryVariant bestVariant = variants[0];
        CorePm4CorrelationMetrics bestMetrics = EvaluatePlacementVariantMetrics(pm4Object, bestVariant);

        for (int index = 1; index < variants.Count; index++)
        {
            Pm4PlacementGeometryVariant candidateVariant = variants[index];
            CorePm4CorrelationMetrics candidateMetrics = EvaluatePlacementVariantMetrics(pm4Object, candidateVariant);
            if (ComparePlacementGeometryVariants(pm4Object, sameTile, candidateVariant, candidateMetrics, bestVariant, bestMetrics) < 0)
            {
                bestVariant = candidateVariant;
                bestMetrics = candidateMetrics;
            }
        }

        metrics = bestMetrics;
        resolvedPlacement = placement with
        {
            AssetProfileKey = bestVariant.AssetProfileKey,
            EvidenceSource = bestVariant.EvidenceSource,
            WorldBoundsMin = bestVariant.WorldBoundsMin,
            WorldBoundsMax = bestVariant.WorldBoundsMax,
            FootprintHull = bestVariant.FootprintHull,
            FootprintArea = bestVariant.FootprintArea,
            MeshGroupCount = bestVariant.MeshGroupCount,
            MeshVertexCount = bestVariant.MeshVertexCount,
            MeshTriangleCount = bestVariant.MeshTriangleCount,
            FootprintSampleCount = bestVariant.FootprintSampleCount,
            WorldFootprintArea = bestVariant.WorldFootprintArea
        };
        return true;
    }

    private static Pm4ShapeSignature BuildPm4ShapeSignature(Vector3 boundsMin, Vector3 boundsMax, IReadOnlyList<Vector2> footprintHull)
    {
        Vector2[] resolvedFootprintHull = footprintHull.Count > 0
            ? footprintHull.ToArray()
            : BuildPm4BoundsFootprintHull(boundsMin, boundsMax);
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        float scale = MathF.Max(MathF.Max(boundsMax.X - boundsMin.X, boundsMax.Y - boundsMin.Y), boundsMax.Z - boundsMin.Z);
        if (!float.IsFinite(scale) || scale <= 0.001f)
            scale = 1f;

        Vector3 normalizedBoundsMin = (boundsMin - center) / scale;
        Vector3 normalizedBoundsMax = (boundsMax - center) / scale;
        Vector2 planarCenter = new(center.X, center.Y);
        Vector2[] normalizedFootprintHull = new Vector2[resolvedFootprintHull.Length];
        for (int index = 0; index < resolvedFootprintHull.Length; index++)
            normalizedFootprintHull[index] = (resolvedFootprintHull[index] - planarCenter) / scale;

        float normalizedFootprintArea = CorePm4CorrelationMath.ComputeFootprintArea(normalizedFootprintHull);
        return new Pm4ShapeSignature(normalizedBoundsMin, normalizedBoundsMax, normalizedFootprintHull, normalizedFootprintArea);
    }

    private static string BuildPm4AssetProfileKey(string kind, string modelKey, string evidenceSource, byte? correlatedGroupKey)
    {
        string groupKey = correlatedGroupKey.HasValue
            ? correlatedGroupKey.Value.ToString(CultureInfo.InvariantCulture)
            : "-";
        return $"{kind}|{modelKey}|{evidenceSource}|{groupKey}";
    }

    private static CorePm4CorrelationMetrics EvaluatePlacementVariantMetrics(Pm4ObjectMatchState pm4Object, Pm4PlacementGeometryVariant variant)
    {
        return CorePm4CorrelationMath.EvaluateMetrics(
            pm4Object.BoundsMin,
            pm4Object.BoundsMax,
            pm4Object.Center,
            pm4Object.FootprintHull,
            pm4Object.FootprintArea,
            variant.WorldBoundsMin,
            variant.WorldBoundsMax,
            variant.Center,
            variant.FootprintHull,
            variant.FootprintArea);
    }

    private static int ComparePlacementGeometryVariants(
        Pm4ObjectMatchState pm4Object,
        bool sameTile,
        Pm4PlacementGeometryVariant leftVariant,
        CorePm4CorrelationMetrics leftMetrics,
        Pm4PlacementGeometryVariant rightVariant,
        CorePm4CorrelationMetrics rightMetrics)
    {
        bool leftGroupMatch = leftVariant.CorrelatedGroupKey.HasValue && leftVariant.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        bool rightGroupMatch = rightVariant.CorrelatedGroupKey.HasValue && rightVariant.CorrelatedGroupKey.Value == pm4Object.Object.DominantGroupKey;
        int compareGroupMatch = rightGroupMatch.CompareTo(leftGroupMatch);
        if (compareGroupMatch != 0)
            return compareGroupMatch;

        int compareScore = CorePm4CorrelationMath.CompareCandidateScores(
            new CorePm4CorrelationCandidateScore(sameTile, leftMetrics, leftVariant.WorldBoundsMin, leftVariant.WorldBoundsMax, leftVariant.Center),
            new CorePm4CorrelationCandidateScore(sameTile, rightMetrics, rightVariant.WorldBoundsMin, rightVariant.WorldBoundsMax, rightVariant.Center));
        if (compareScore != 0)
            return compareScore;

        return GetPlacementGeometryEvidenceRank(leftVariant.EvidenceSource).CompareTo(GetPlacementGeometryEvidenceRank(rightVariant.EvidenceSource));
    }

    private static int GetPlacementGeometryEvidenceRank(string evidenceSource)
    {
        return evidenceSource.ToLowerInvariant() switch
        {
            "wmo-group-mesh" => 0,
            "mdx-collision" => 1,
            "wmo-mesh" => 2,
            "modf-bounds" => 3,
            _ => 4,
        };
    }

    private static int GetPm4ObjectMatchEvidenceRank(Pm4ObjectMatchState pm4Object, Pm4PlacementMatchState placement)
    {
        bool zeroOrRootObject = pm4Object.Object.Ck24 == 0 || pm4Object.Object.LinkGroupObjectId == 0;
        if (zeroOrRootObject)
        {
            if (pm4Object.Object.LinkedPositionRefCount > 0)
                return string.Equals(placement.Kind, "m2", StringComparison.OrdinalIgnoreCase) ? 0 : 1;

            return 0;
        }

        if (placement.Kind == "wmo" && string.Equals(placement.EvidenceSource, "wmo-group-mesh", StringComparison.OrdinalIgnoreCase))
            return 0;

        if (string.Equals(placement.EvidenceSource, "mdx-collision", StringComparison.OrdinalIgnoreCase))
            return 1;

        if (placement.Kind == "wmo" && string.Equals(placement.EvidenceSource, "wmo-mesh", StringComparison.OrdinalIgnoreCase))
            return 2;

        if (placement.Kind == "wmo")
            return 3;

        return 4;
    }

    private static float ComputePm4ObjectAnchorPlanarGap(Vector3 anchor, Vector3 placementPosition)
    {
        if (!float.IsFinite(anchor.X) || !float.IsFinite(anchor.Y) || !float.IsFinite(placementPosition.X) || !float.IsFinite(placementPosition.Y))
            return float.MaxValue;

        return Vector2.Distance(new Vector2(anchor.X, anchor.Y), new Vector2(placementPosition.X, placementPosition.Y));
    }

    private static Vector2[] BuildPm4BoundsFootprintHull(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];
    }

    /// <summary>Get the currently selected object instance, or null if nothing selected.</summary>
    public ObjectInstance? SelectedInstance => TryGetSelectedSceneInstance(out ObjectInstance instance) ? instance : null;

    public bool TryGetSelectedPlacementSourceData(out string sourcePath, out byte[] sourceBytes)
    {
        sourcePath = string.Empty;
        sourceBytes = Array.Empty<byte>();

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
            return false;

        ObjectInstance instance = selected.Value;
        if (!instance.HasTileCoordinate || instance.PlacementEntryIndex < 0)
            return false;

        return _terrainManager.Adapter.TryGetPlacementSourceData(instance.TileX, instance.TileY, out sourcePath, out sourceBytes);
    }

    public bool TryGetSelectedPlacementWritablePath(out string? fullPath)
    {
        fullPath = null;

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
            return false;

        ObjectInstance instance = selected.Value;
        if (!instance.HasTileCoordinate || instance.PlacementEntryIndex < 0)
            return false;

        return _terrainManager.Adapter.TryGetPlacementWritablePath(instance.TileX, instance.TileY, out fullPath);
    }

    public bool TryUpdateSelectedPlacementPosition(Vector3 newPosition, out string error)
    {
        error = string.Empty;

        ObjectInstance? selected = SelectedInstance;
        if (!selected.HasValue)
        {
            error = "No world object is selected.";
            return false;
        }

        ObjectInstance current = selected.Value;
        if (!current.HasTileCoordinate || current.PlacementEntryIndex < 0)
        {
            error = "The selected object is not backed by a writable ADT placement entry.";
            return false;
        }

        if (_selectedObjectType is not (ObjectType.Mdx or ObjectType.Wmo))
        {
            error = "Only ADT MDDF and MODF placements are supported by the current save seam.";
            return false;
        }

        Dictionary<(int, int), List<ObjectInstance>> tileInstances = _selectedObjectType == ObjectType.Mdx
            ? _tileMdxInstances
            : _tileWmoInstances;

        if (!tileInstances.TryGetValue((current.TileX, current.TileY), out List<ObjectInstance>? instances))
        {
            error = $"Tile ({current.TileX}, {current.TileY}) is not currently loaded.";
            return false;
        }

        int instanceIndex = FindPlacementInstanceIndex(instances, current);
        if (instanceIndex < 0)
        {
            error = "The selected placement could not be matched back to the loaded tile instance list.";
            return false;
        }

        ObjectInstance updated = MovePlacementInstance(current, newPosition, _selectedObjectType);
        instances[instanceIndex] = updated;

        _terrainManager.TryUpdateCachedPlacementPosition(_selectedObjectType, current.TileX, current.TileY, current.PlacementEntryIndex, newPosition);
        UpdateAdapterPlacementPosition(_selectedObjectType, current, newPosition);

        _instancesDirty = true;
        RebuildInstanceLists();
        return true;
    }

    // Area POI (lazy-loaded on first toggle)
    private AreaPoiLoader? _poiLoader;
    private bool _showPoi = false;
    private bool _poiLoadAttempted = false;
    public bool ShowPoi
    {
        get => _showPoi;
        set { _showPoi = value; if (value && !_poiLoadAttempted) LazyLoadPoi(); }
    }
    public AreaPoiLoader? PoiLoader => _poiLoader;
    public bool PoiLoadAttempted => _poiLoadAttempted;

    // Taxi paths (lazy-loaded on first toggle)
    private TaxiPathLoader? _taxiLoader;
    private bool _showTaxi = false;
    private bool _taxiLoadAttempted = false;
    public bool ShowTaxi
    {
        get => _showTaxi;
        set { _showTaxi = value; if (value && !_taxiLoadAttempted) LazyLoadTaxi(); }
    }
    public TaxiPathLoader? TaxiLoader => _taxiLoader;
    public bool TaxiLoadAttempted => _taxiLoadAttempted;

    // AreaTriggers (lazy-loaded on first toggle)
    private AreaTriggerLoader? _areaTriggerLoader;
    private bool _showAreaTriggers = false;
    private bool _areaTriggerLoadAttempted = false;
    public bool ShowAreaTriggers
    {
        get => _showAreaTriggers;
        set { _showAreaTriggers = value; if (value && !_areaTriggerLoadAttempted) LazyLoadAreaTriggers(); }
    }
    public AreaTriggerLoader? AreaTriggerLoader => _areaTriggerLoader;
    public bool AreaTriggerLoadAttempted => _areaTriggerLoadAttempted;

    // WL loose liquid files (auto-loaded on scene init)
    private WlLiquidLoader? _wlLoader;
    private bool _showWlLiquids = true; // Auto-enable by default
    private bool _wlLoadAttempted = false;
    private IDataSource? _dataSource;
    private bool _clientStarsProbeComplete;
    private string? _clientStarsFallbackModelPath;
    private string? _activeLightSkyboxSourcePath;
    private string? _activeLightSkyboxModelKey;
    public bool ShowWlLiquids
    {
        get => _showWlLiquids;
        set
        {
            _showWlLiquids = value;
            if (value && !_wlLoadAttempted) LazyLoadWlLiquids();
            _terrainManager.LiquidRenderer.ShowWlLiquids = value;
        }
    }
    public WlLiquidLoader? WlLoader => _wlLoader;
    public bool WlLoadAttempted => _wlLoadAttempted;

    // Stored DBC credentials for lazy loading
    private DBCD.Providers.IDBCProvider? _dbcProvider;
    private string? _dbdDir;
    private string? _dbcBuild;
    private int _mapId = -1;

    // DBC Lighting
    private LightService? _lightService;
    public LightService? LightService => _lightService;

    // Alpha LIT lighting (lazy-loaded on first request)
    private LitLoader? _litLoader;
    private bool _showLitLights;
    private bool _showLitMinimapMarkers;
    private bool _litLoadAttempted;
    private bool _useLitFogOverride;
    // Local Light* spatial selection is retained for diagnostics, but its
    // renderer application remains opt-in until the native local-zone
    // transform/falloff contract is proven for the active build.
    private bool _useLocalDbcLightingOverlay;
    private bool _hasGlobalViewerFogRange;
    private float _globalViewerFogStart;
    private float _globalViewerFogEnd;
    private bool _hasPreLitFogRange;
    private float _preLitFogStart;
    private float _preLitFogEnd;
    private bool _hasUserFogRangeOverride;
    private float _userFogStart = TerrainLightingMath.DefaultFogStart;
    private float _userFogEnd = TerrainLightingMath.DefaultFogEnd;
    private float _activeFogStart = TerrainLightingMath.DefaultFogStart;
    private float _activeFogEnd = TerrainLightingMath.DefaultFogEnd;
    private string _activeFogRangeSource = "Fallback";
    private bool _activeFogRangeAdjusted;
    private string _litStatus = "LIT not loaded.";
    private int _selectedLitLightIndex = -1;
    private string? _selectedLitSourcePath;
    private LitLoader.LitLightingSample? _lastLitSample;
    public bool ShowLitLights
    {
        get => _showLitLights;
        set
        {
            _showLitLights = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    /// <summary>Shows loaded positional LIT entries on shared minimap surfaces without changing lighting.</summary>
    public bool ShowLitMinimapMarkers
    {
        get => _showLitMinimapMarkers;
        set
        {
            _showLitMinimapMarkers = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    public bool UseLitFogOverride
    {
        get => _useLitFogOverride;
        set
        {
            if (_useLitFogOverride == value)
                return;
            if (!value)
                RestorePreLitFogRange(_terrainManager?.Lighting);
            _useLitFogOverride = value;
            if (value && !_litLoadAttempted)
                LazyLoadLit();
        }
    }

    public bool UseLocalDbcLightingOverlay
    {
        get => _useLocalDbcLightingOverlay;
        set => _useLocalDbcLightingOverlay = value;
    }

    public LitLoader? LitLoader => _litLoader;
    public bool LitLoadAttempted => _litLoadAttempted;
    public string LitStatus => _litStatus;
    public int SelectedLitLightIndex { get => _selectedLitLightIndex; set => _selectedLitLightIndex = value; }
    public string? SelectedLitSourcePath => _selectedLitSourcePath ?? _litLoader?.SourcePath;
    public IReadOnlyList<string> AvailableLitSourcePaths => _litLoader?.AvailableSourcePaths ?? Array.Empty<string>();
    public LitLoader.LitLightingSample? LastLitSample => _lastLitSample;

    /// <summary>User-selected fog range that is intentionally independent from lighting recommendations.</summary>
    public bool HasUserFogRangeOverride => _hasUserFogRangeOverride;

    public float UserFogStart => _userFogStart;

    public float UserFogEnd => _userFogEnd;

    public float ActiveFogStart => _activeFogStart;

    public float ActiveFogEnd => _activeFogEnd;

    public string ActiveFogRangeSource => _activeFogRangeSource;

    public bool ActiveFogRangeAdjusted => _activeFogRangeAdjusted;

    public void SetUserFogRangeOverride(float fogStart, float fogEnd)
    {
        (_userFogStart, _userFogEnd) = TerrainLightingMath.NormalizeFogRange(fogStart, fogEnd);
        _hasUserFogRangeOverride = true;
    }

    public void ClearUserFogRangeOverride()
    {
        _hasUserFogRangeOverride = false;
    }

    private void CapturePreLitFogRange(TerrainLighting lighting)
    {
        if (_hasPreLitFogRange)
            return;

        _preLitFogStart = lighting.FogStart;
        _preLitFogEnd = lighting.FogEnd;
        _hasPreLitFogRange = true;
    }

    private void RestoreGlobalViewerFogRange(TerrainLighting lighting)
    {
        if (!_hasGlobalViewerFogRange)
        {
            _globalViewerFogStart = lighting.FogStart;
            _globalViewerFogEnd = lighting.FogEnd;
            _hasGlobalViewerFogRange = true;
        }

        lighting.FogStart = _globalViewerFogStart;
        lighting.FogEnd = _globalViewerFogEnd;
    }

    private void RestorePreLitFogRange(TerrainLighting? lighting)
    {
        if (!_hasPreLitFogRange)
            return;
        if (lighting != null)
        {
            lighting.FogStart = _preLitFogStart;
            lighting.FogEnd = _preLitFogEnd;
        }
        _hasPreLitFogRange = false;
    }

    private void ResolveActiveFogRange(TerrainLighting lighting, string recommendationSource)
    {
        float rawRecommendedStart = lighting.FogStart;
        float rawRecommendedEnd = lighting.FogEnd;
        float fallbackStart = _hasPreLitFogRange ? _preLitFogStart : TerrainLightingMath.DefaultFogStart;
        float fallbackEnd = _hasPreLitFogRange ? _preLitFogEnd : TerrainLightingMath.DefaultFogEnd;
        (float recommendedStart, float recommendedEnd) = TerrainLightingMath.NormalizeFogRange(
            rawRecommendedStart,
            rawRecommendedEnd,
            fallbackStart,
            fallbackEnd);

        (float activeStart, float activeEnd) = _hasUserFogRangeOverride
            ? TerrainLightingMath.NormalizeFogRange(_userFogStart, _userFogEnd, recommendedStart, recommendedEnd)
            : (recommendedStart, recommendedEnd);

        _activeFogRangeAdjusted = !FogRangesEqual(rawRecommendedStart, rawRecommendedEnd, recommendedStart, recommendedEnd)
            || (_hasUserFogRangeOverride && !FogRangesEqual(_userFogStart, _userFogEnd, activeStart, activeEnd));
        _activeFogRangeSource = _hasUserFogRangeOverride ? "User override" : recommendationSource;
        _activeFogStart = activeStart;
        _activeFogEnd = activeEnd;
        lighting.FogStart = activeStart;
        lighting.FogEnd = activeEnd;
    }

    private static bool FogRangesEqual(float leftStart, float leftEnd, float rightStart, float rightEnd)
        => MathF.Abs(leftStart - rightStart) < 0.001f && MathF.Abs(leftEnd - rightEnd) < 0.001f;

    // Taxi selection: -1 = show all (or none if !_showTaxi)
    private int _selectedTaxiNodeId = -1;
    private int _selectedTaxiRouteId = -1;
    private readonly Dictionary<int, string> _taxiActorModelOverrideByPath = new();
    private readonly Dictionary<int, float> _taxiActorTravelByPath = new();
    private readonly Dictionary<int, TaxiActorPose> _taxiActorPoseByPath = new();
    private readonly Dictionary<int, Vector3> _taxiActorSmoothedForwardByPath = new();
    private long _lastTaxiActorTick;
    private bool _taxiActorClockInitialized;
    private bool _showTaxiActors = true;
    private float _taxiActorSpeedMultiplier = TaxiActorNormalSpeedSetting;
    private float _taxiActorScaleMultiplier = 1.0f;
    private const float TaxiActorBaseUnitsPerSecond = 650f;
    private const float TaxiActorHoverOffset = 12f;
    public int SelectedTaxiNodeId { get => _selectedTaxiNodeId; set { _selectedTaxiNodeId = value; _selectedTaxiRouteId = -1; } }
    public int SelectedTaxiRouteId { get => _selectedTaxiRouteId; set { _selectedTaxiRouteId = value; _selectedTaxiNodeId = -1; } }
    public void ClearTaxiSelection() { _selectedTaxiNodeId = -1; _selectedTaxiRouteId = -1; }
    public bool ShowTaxiActors { get => _showTaxiActors; set => _showTaxiActors = value; }
    public float TaxiActorSpeedMultiplier
    {
        get => _taxiActorSpeedMultiplier;
        set
        {
            float normalized = float.IsFinite(value) ? value : TaxiActorNormalSpeedSetting;
            _taxiActorSpeedMultiplier = Math.Clamp(normalized, TaxiActorMinSpeedSetting, TaxiActorMaxSpeedSetting);
        }
    }

    public float TaxiActorScaleMultiplier
    {
        get => _taxiActorScaleMultiplier;
        set => _taxiActorScaleMultiplier = Math.Max(0f, value);
    }

    public bool IsTaxiRouteVisible(TaxiPathLoader.TaxiRoute route)
    {
        if (_selectedTaxiRouteId >= 0) return route.PathId == _selectedTaxiRouteId;
        if (_selectedTaxiNodeId >= 0) return route.FromNodeId == _selectedTaxiNodeId || route.ToNodeId == _selectedTaxiNodeId;
        return true; // no selection = show all
    }

    public bool IsTaxiNodeVisible(TaxiPathLoader.TaxiNode node)
    {
        if (_selectedTaxiNodeId >= 0) return node.Id == _selectedTaxiNodeId;
        if (_selectedTaxiRouteId >= 0)
        {
            var route = _taxiLoader?.Routes.FirstOrDefault(r => r.PathId == _selectedTaxiRouteId);
            return route != null && (route.FromNodeId == node.Id || route.ToNodeId == node.Id);
        }
        return true; // no selection = show all
    }

    public TaxiPathLoader.TaxiNode? GetTaxiNode(int nodeId)
        => _taxiLoader?.Nodes.FirstOrDefault(node => node.Id == nodeId);

    public TaxiPathLoader.TaxiRoute? GetTaxiRoute(int pathId)
        => _taxiLoader?.Routes.FirstOrDefault(route => route.PathId == pathId);

    public string? GetTaxiActorModelOverride(int pathId)
        => _taxiActorModelOverrideByPath.TryGetValue(pathId, out string? modelPath) ? modelPath : null;

    public void SetTaxiActorModelOverride(int pathId, string? modelPath)
    {
        string normalizedPath = string.IsNullOrWhiteSpace(modelPath)
            ? string.Empty
            : modelPath.Trim().Replace('/', '\\');

        if (string.IsNullOrWhiteSpace(normalizedPath))
        {
            _taxiActorModelOverrideByPath.Remove(pathId);
            return;
        }

        _taxiActorModelOverrideByPath[pathId] = normalizedPath;
        _assets.QueueMdxLoad(WorldAssetManager.NormalizeKey(normalizedPath));
    }

    public string? GetResolvedTaxiActorModelPath(int pathId)
    {
        if (_taxiActorModelOverrideByPath.TryGetValue(pathId, out string? overrideModelPath)
            && !string.IsNullOrWhiteSpace(overrideModelPath))
        {
            return overrideModelPath;
        }

        TaxiPathLoader.TaxiRoute? route = GetTaxiRoute(pathId);
        if (route == null)
            return ResolveDefaultTaxiActorModelPath();

        TaxiPathLoader.TaxiNode? mountNode = ResolveTaxiActorNode(route);
        if (!string.IsNullOrWhiteSpace(mountNode?.MountModelPath))
            return mountNode.MountModelPath.Replace('/', '\\');

        return ResolveDefaultTaxiActorModelPath();
    }

    private string ResolveDefaultTaxiActorModelPath()
    {
        if (_dataSource != null)
        {
            foreach (string candidate in TaxiActorDefaultModelCandidates)
            {
                if (_dataSource.FileExists(candidate))
                    return candidate;
            }
        }

        return TaxiActorDefaultModelCandidates[0];
    }

    public bool TryGetTaxiActorPose(int pathId, out TaxiActorPose pose)
        => _taxiActorPoseByPath.TryGetValue(pathId, out pose);

    public bool TryGetSelectedTaxiActorPose(out TaxiActorPose pose)
    {
        if (_selectedTaxiRouteId < 0)
        {
            pose = default;
            return false;
        }

        return _taxiActorPoseByPath.TryGetValue(_selectedTaxiRouteId, out pose);
    }

    public bool TryGetTaxiRouteSelectionPoint(int pathId, out Vector3 point)
    {
        TaxiPathLoader.TaxiRoute? route = GetTaxiRoute(pathId);
        if (route == null)
        {
            point = Vector3.Zero;
            return false;
        }

        return TryGetTaxiRouteSelectionPoint(route, out point);
    }

    /// <summary>
    /// Store DBC credentials for lazy loading of POI, Taxi, and Lighting.
    /// </summary>
    public void SetDbcCredentials(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _dbcProvider = dbcProvider;
        _dbdDir = dbdDir;
        _dbcBuild = build;
        _mapId = mapId;
        _assets.SetBuildVersion(build);
    }

    private void LazyLoadWlLiquids()
    {
        _wlLoadAttempted = true;
        if (_dataSource == null) return;
        _wlLoader = new WlLiquidLoader(_dataSource, _terrainManager.MapName);
        _wlLoader.LoadAll();
        if (_wlLoader.HasData)
            _terrainManager.LiquidRenderer.AddWlBodies(_wlLoader.Bodies);
    }

    private void LazyLoadLit()
    {
        _litLoadAttempted = true;
        _lastLitSample = null;

        if (_dataSource == null)
        {
            _litStatus = "LIT unavailable: no data source.";
            return;
        }

        _litLoader = new LitLoader(_dataSource, _terrainManager.MapName, _selectedLitSourcePath);
        if (_litLoader.Load())
        {
            _litStatus = _litLoader.Status;
            _selectedLitSourcePath = _litLoader.SourcePath;
            if (_selectedLitLightIndex < 0 && _litLoader.Lights.Count > 0)
                _selectedLitLightIndex = 0;
            return;
        }

        _litStatus = _litLoader.Status;
    }

    public void ReloadLit(string? sourcePath = null)
    {
        _selectedLitSourcePath = string.IsNullOrWhiteSpace(sourcePath) ? null : sourcePath;
        _selectedLitLightIndex = -1;
        _litLoader = null;
        _litLoadAttempted = false;
        _litStatus = "LIT reload queued.";
        LazyLoadLit();
    }

    private void BeginPm4OverlayLoad(bool ignoreCache = false)
    {
        if (_dataSource == null)
        {
            _pm4LoadAttempted = true;
            _pm4Status = "PM4 unavailable: no data source.";
            return;
        }

        if (!ignoreCache && _pm4LoadTask != null && !_pm4LoadTask.IsCompleted)
            return;

        ReleasePm4LoadCancellation(cancelPendingLoad: true);

        _pm4LoadAttempted = true;
        int requestId = ++_pm4LoadRequestId;
        var selectedObjectKey = _selectedPm4ObjectKey;
        var cancellation = new CancellationTokenSource();
        _pm4LoadCancellation = cancellation;
        _pm4Status = ignoreCache
            ? "PM4 reload queued: decoding map-wide overlay in background..."
            : "PM4 loading: decoding map-wide overlay in background...";
        _pm4LoadTask = Task.Run(() => LoadPm4OverlayAsync(requestId, ignoreCache, selectedObjectKey, cancellation.Token), cancellation.Token);
    }

    private void TryFinalizePm4OverlayLoad()
    {
        Task<Pm4OverlayAsyncLoadResult>? loadTask = _pm4LoadTask;
        if (loadTask == null || !loadTask.IsCompleted)
            return;

        _pm4LoadTask = null;

        Pm4OverlayAsyncLoadResult result;
        try
        {
            result = loadTask.GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            _pm4Status = $"PM4 load failed: {ex.Message}";
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
            return;
        }

        if (result.RequestId != _pm4LoadRequestId || result.Cancelled)
            return;

        bool replaceExisting = !_pm4LoadedCameraWindow.HasValue || _pm4TileObjects.Count == 0;

        if (result.KnownMapTiles.Count > 0)
            _pm4KnownMapTiles.UnionWith(result.KnownMapTiles);

        if (result.CacheData != null)
        {
            if (replaceExisting)
                ClearPm4OverlayRuntimeState();

            MergePm4OverlayFromCache(result.CacheData);
            RestoreSelectedPm4Object(result.SelectedObjectKey);
            UpdatePm4AdaptiveWindow(result.LoadElapsedMs);
        }

        if (result.CoveredMapTiles.Count > 0)
            _pm4CoveredMapTiles.UnionWith(result.CoveredMapTiles);

        if (result.LoadedCameraWindow.HasValue)
            ExpandPm4LoadedCameraWindow(result.LoadedCameraWindow.Value);

        _pm4Status = result.StatusMessage;
        LogPm4FinalStatus(_pm4Status);
    }

    private static bool ShouldSuppressPm4FinalStatusLog(string status)
    {
        return status.StartsWith("PM4: no files intersect camera window", StringComparison.Ordinal)
            || status.StartsWith("PM4: 0/", StringComparison.Ordinal)
            || status.Contains("none decoded into overlay data", StringComparison.Ordinal);
    }

    private void LogPm4FinalStatus(string status)
    {
        if (string.IsNullOrWhiteSpace(status) || ShouldSuppressPm4FinalStatusLog(status))
            return;

        if (status.StartsWith("PM4 ready:", StringComparison.Ordinal)
            || status.StartsWith("PM4 load failed:", StringComparison.Ordinal)
            || status.StartsWith("PM4 unavailable:", StringComparison.Ordinal))
        {
            ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + status);
            return;
        }

        ViewerLog.Info(ViewerLog.Category.Terrain, "[PM4] " + status);
    }

    private void ReleasePm4LoadCancellation(bool cancelPendingLoad)
    {
        CancellationTokenSource? cancellation = _pm4LoadCancellation;
        _pm4LoadCancellation = null;
        if (cancellation == null)
            return;

        if (cancelPendingLoad)
            cancellation.Cancel();

        cancellation.Dispose();
    }

    private void ReportPm4LoadProgress(
        int requestId,
        string phase,
        int processedFiles,
        int totalFiles,
        int loadedFiles,
        int objectCount,
        int lineCount,
        int triangleCount,
        int readFailed,
        int decodeFailed,
        int zeroObjectFiles,
        int memCacheHits,
        int diskCacheHits,
        string? currentPath,
        bool emitLog)
    {
        if (requestId != _pm4LoadRequestId)
            return;

        string currentFileSuffix = string.IsNullOrWhiteSpace(currentPath)
            ? string.Empty
            : $", file={Path.GetFileName(currentPath)}";
        string status =
            $"PM4 loading: {phase} {processedFiles}/{totalFiles} files, loaded={loadedFiles}, objects={objectCount}, lines={lineCount}, tris={triangleCount}, readFail={readFailed}, decodeFail={decodeFailed}, zero={zeroObjectFiles} (mem-cache {memCacheHits} hit, disk-cache {diskCacheHits} hit){currentFileSuffix}";
        _pm4Status = status;

        if (emitLog)
            ViewerLog.Info(ViewerLog.Category.Terrain, "[PM4] " + status);
    }

    private Pm4OverlayAsyncLoadResult LoadPm4OverlayAsync(
        int requestId,
        bool ignoreCache,
        (int tileX, int tileY, uint ck24, int objectPart)? selectedObjectKey,
        CancellationToken cancellationToken)
    {
        try
        {
            if (_dataSource == null)
                return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, "PM4 unavailable: no data source.", cancelled: false);

            string mapName = _terrainManager.MapName;
            List<string> mapPm4Candidates = _dataSource
                .GetFileList(".pm4")
                .Where(path => IsMapPm4Path(path, mapName))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();

            int mapPm4CandidateCount = mapPm4Candidates.Count;
            if (mapPm4CandidateCount == 0)
                return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, $"PM4: no files found for map '{mapName}'.", cancelled: false);

            int tileParseRejected = 0;
            int tileRangeRejected = 0;
            var pm4Candidates = new List<(string path, int tileX, int tileY)>();
            foreach (string pm4Path in mapPm4Candidates)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int fileTileX, out int fileTileY))
                {
                    tileParseRejected++;
                    continue;
                }

                if (!TryMapPm4FileTileToTerrainTile(fileTileX, fileTileY, out int effectiveTileX, out int effectiveTileY))
                {
                    tileRangeRejected++;
                    continue;
                }

                pm4Candidates.Add((pm4Path, effectiveTileX, effectiveTileY));
            }

            HashSet<(int tileX, int tileY)> knownMapTiles = pm4Candidates
                .Select(static candidate => (candidate.tileX, candidate.tileY))
                .ToHashSet();

            int totalFiles = pm4Candidates.Count;
            if (totalFiles == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    null,
                    knownMapTiles,
                    [],
                    selectedObjectKey,
                    0.0,
                    $"PM4: 0/{mapPm4CandidateCount} valid map files after tile mapping (tileParse={tileParseRejected}, tileRange={tileRangeRejected}).",
                    cancelled: false);
            }

            Vector3 loadAnchorCameraPosition = GetPm4LoadAnchorCameraPosition();
            var cameraWindow = GetPm4CameraWindow(loadAnchorCameraPosition, _pm4CameraTileRadius);
            List<(string path, int tileX, int tileY)> loadCandidates = pm4Candidates
                .Where(candidate => IsPm4TileInsideCameraWindow(candidate.tileX, candidate.tileY, cameraWindow))
                .ToList();

            if (loadCandidates.Count == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    cameraWindow,
                    knownMapTiles,
                    [],
                    selectedObjectKey,
                    0.0,
                    $"PM4: no files intersect camera window ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}) out of {totalFiles} valid map files.",
                    cancelled: false);
            }

            HashSet<(int tileX, int tileY)> loadCandidateTiles = loadCandidates
                .Select(static candidate => (candidate.tileX, candidate.tileY))
                .ToHashSet();

            if (ignoreCache && _pm4OverlayCacheService != null)
            {
                if (!_pm4OverlayCacheService.TryDelete(mapName, out string? cacheDeleteError) && !string.IsNullOrWhiteSpace(cacheDeleteError))
                    ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] {cacheDeleteError}");
            }

            string candidateSignature = Pm4OverlayCacheService.BuildCandidateSignature(
                _dataSource,
                loadCandidates.Select(static candidate => candidate.path).ToList(),
                _pm4SplitCk24ByMscnRef,
                _pm4SplitCk24ByConnectivity,
                _pm4ShowPathWalls);
            var loadStopwatch = Stopwatch.StartNew();
            string? cacheLoadError = null;
            if (!ignoreCache
                && _pm4OverlayCacheService != null
                && _pm4OverlayCacheService.TryLoad(mapName, candidateSignature, out Pm4OverlayCacheData? cachedOverlay, out cacheLoadError)
                && cachedOverlay != null)
            {
                loadStopwatch.Stop();
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    cachedOverlay,
                    cameraWindow,
                    knownMapTiles,
                    loadCandidateTiles,
                    selectedObjectKey,
                    loadStopwatch.Elapsed.TotalMilliseconds,
                    $"PM4 ready: {cachedOverlay.LoadedFiles}/{cachedOverlay.TotalFiles} camera-window files restored from disk cache for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}), avg {_pm4AverageLoadMs:0} ms, next radius {_pm4CameraTileRadius}, from {mapPm4CandidateCount} map files, {cachedOverlay.ObjectCount} objects, {cachedOverlay.LineCount} lines, {cachedOverlay.TriangleCount} triangles, {cachedOverlay.PositionRefCount} refs, {cachedOverlay.RejectedLongEdges} long edges rejected, {loadStopwatch.ElapsedMilliseconds} ms.",
                    cancelled: false);
            }

            if (!string.IsNullOrWhiteSpace(cacheLoadError))
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] {cacheLoadError}");

            _pm4Status = $"PM4 loading: decoding {loadCandidates.Count} camera-window files (per-file cache active)...";

            int remainingLineBudget = Pm4MaxLinesTotal;
            int remainingTriangleBudget = Pm4MaxTrianglesTotal;
            int remainingPositionRefBudget = Pm4MaxPositionRefsTotal;
            int loadedFiles = 0;
            int objectCount = 0;
            int lineCount = 0;
            int triangleCount = 0;
            int positionRefCount = 0;
            int rejectedLongEdgesTotal = 0;
            int readFailed = 0;
            int decodeFailed = 0;
            int zeroObjectFiles = 0;
            int memCacheHits = 0;
            int memCacheMisses = 0;
            int diskCacheHits = 0;
            float minObjectZ = float.MaxValue;
            float maxObjectZ = float.MinValue;
            var tileCandidateCounts = loadCandidates
                .GroupBy(static candidate => (candidate.tileX, candidate.tileY))
                .ToDictionary(static group => group.Key, static group => group.Count());
            var tileSatisfiedCounts = tileCandidateCounts.Keys.ToDictionary(static tile => tile, static _ => 0);
            var tileObjects = new Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>>();
            var tilePositionRefs = new Dictionary<(int tileX, int tileY), List<Vector3>>();
            var progressStopwatch = Stopwatch.StartNew();
            long lastStatusReportMs = -Pm4ProgressStatusIntervalMs;
            long lastLogReportMs = -Pm4ProgressLogIntervalMs;
            int processedFiles = 0;

            foreach (var candidate in loadCandidates)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (remainingLineBudget <= 0)
                    break;

                string pm4Path = candidate.path;
                int effectiveTileX = candidate.tileX;
                int effectiveTileY = candidate.tileY;
                processedFiles++;

                byte[]? bytes = _dataSource.ReadFile(pm4Path);
                if (bytes == null)
                {
                    readFailed++;
                    long readFailElapsedMs = progressStopwatch.ElapsedMilliseconds;
                    if (readFailElapsedMs - lastStatusReportMs >= Pm4ProgressStatusIntervalMs)
                    {
                        bool emitLog = readFailElapsedMs - lastLogReportMs >= Pm4ProgressLogIntervalMs;
                        ReportPm4LoadProgress(requestId, "reading", processedFiles, loadCandidates.Count, loadedFiles, objectCount, lineCount, triangleCount, readFailed, decodeFailed, zeroObjectFiles, memCacheHits, diskCacheHits, pm4Path, emitLog);
                        lastStatusReportMs = readFailElapsedMs;
                        if (emitLog)
                            lastLogReportMs = readFailElapsedMs;
                    }
                    continue;
                }

                if (bytes.Length == 0)
                {
                    tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                    zeroObjectFiles++;
                    ViewerLog.Debug(ViewerLog.Category.Terrain,
                        $"[PM4] Skipping empty PM4 carrier '{pm4Path}' for tile ({effectiveTileX},{effectiveTileY}).");
                    continue;
                }

                // Spec 054: per-file in-memory cache check. A hit lets the
                // camera-window load skip the per-tile BuildPm4TileObjects
                // work entirely for files already decoded in this session.
                // The stamp folds (file length, split-flag bits) so a
                // content edit invalidates the entry and a split-flag
                // toggle does too. The on-disk per-file cache below uses
                // the loose-file write-tick for stronger stamp coverage
                // when the data source exposes it.
                string normalizedPm4Path = pm4Path.Replace('\\', '/');
                long memStamp = ((_pm4ShowPathWalls ? 1L : 0L) << 33)
                    | ((_pm4SplitCk24ByMscnRef ? 1L : 0L) << 32)
                    | (_pm4SplitCk24ByConnectivity ? 1L : 0L);
                if (_pm4PerFileInMemoryCache.TryGet(normalizedPm4Path, bytes.Length, memStamp, out CorePm4PerFileCacheEntry? cachedEntry)
                    && cachedEntry != null
                    && cachedEntry.Tiles.Count > 0)
                {
                    int cachedObjectCount = ApplyCachedTilesToTileDictionaries(
                        cachedEntry,
                        effectiveTileX,
                        effectiveTileY,
                        tileObjects,
                        tilePositionRefs,
                        ref minObjectZ,
                        ref maxObjectZ,
                        ref objectCount,
                        ref lineCount,
                        ref triangleCount,
                        ref positionRefCount);
                    if (cachedObjectCount > 0)
                    {
                        loadedFiles++;
                        memCacheHits++;
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        ViewerLog.Debug(
                            ViewerLog.Category.Terrain,
                            $"[PM4] Per-file in-memory cache hit for '{pm4Path}' ({cachedObjectCount} objects, {cachedEntry.Tiles.Count} tiles).");
                        continue;
                    }
                }
                memCacheMisses++;

                // Spec 054: per-file on-disk cache check. Falls through
                // when the in-memory cache misses. We use the loose-file
                // write-tick as the file stamp when available (falls
                // back to 0 for MPQ-only data sources).
                CorePm4PerFileCacheService? onDiskCache = EnsurePerFileDiskCache(mapName);
                if (onDiskCache != null
                    && TryReadPerFileDiskCache(onDiskCache, normalizedPm4Path, bytes.Length, out CorePm4PerFileCacheEntry? diskCachedEntry)
                    && diskCachedEntry != null
                    && diskCachedEntry.Tiles.Count > 0)
                {
                    int cachedObjectCount = ApplyCachedTilesToTileDictionaries(
                        diskCachedEntry,
                        effectiveTileX,
                        effectiveTileY,
                        tileObjects,
                        tilePositionRefs,
                        ref minObjectZ,
                        ref maxObjectZ,
                        ref objectCount,
                        ref lineCount,
                        ref triangleCount,
                        ref positionRefCount);
                    if (cachedObjectCount > 0)
                    {
                        loadedFiles++;
                        diskCacheHits++;
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        ViewerLog.Debug(
                            ViewerLog.Category.Terrain,
                            $"[PM4] Per-file on-disk cache hit for '{pm4Path}' ({cachedObjectCount} objects, {diskCachedEntry.Tiles.Count} tiles).");
                        continue;
                    }
                }

                try
                {
                    Pm4File pm4 = CorePm4DocumentReader.Read(bytes, pm4Path);
                    int rejectedLongEdges = 0;
                    List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                        pm4,
                        pm4Path,
                        effectiveTileX,
                        effectiveTileY,
                _pm4SplitCk24ByMscnRef,
                        _pm4SplitCk24ByConnectivity,
                        _pm4ShowPathWalls,
                        ref remainingLineBudget,
                        ref remainingTriangleBudget,
                        ref rejectedLongEdges,
                        out Pm4TileBuildDiagnostics fileDiagnostics);
                    _pm4TotalMsurCount += fileDiagnostics.TotalMsurCount;
                    _pm4DroppedShortIndexCount += fileDiagnostics.DroppedShortIndexCount;
                    _pm4WallFaceCount += fileDiagnostics.WallFaceCount;
                    _pm4DroppedOutOfRangeMsviCount += fileDiagnostics.DroppedOutOfRangeMsviCount;
                    _pm4DroppedEmptyComponentCount += fileDiagnostics.DroppedEmptyComponentCount;
                    if (objects.Count == 0)
                    {
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        zeroObjectFiles++;
                        continue;
                    }

                    if (tileObjects.TryGetValue((effectiveTileX, effectiveTileY), out List<Pm4OverlayObject>? existingObjects))
                    {
                        ViewerLog.Debug(
                            ViewerLog.Category.Terrain,
                            $"[PM4] Multiple files mapped to tile ({effectiveTileX},{effectiveTileY}); merging '{Path.GetFileName(pm4Path)}' into existing overlay tile.");

                        int objectPartOffset = existingObjects.Count;
                        objects = RebasePm4ObjectParts(objects, objectPartOffset);
                        existingObjects.AddRange(objects);
                    }
                    else
                    {
                        tileObjects[(effectiveTileX, effectiveTileY)] = objects;
                    }

                    foreach (Pm4OverlayObject obj in objects)
                    {
                        minObjectZ = MathF.Min(minObjectZ, obj.Center.Z);
                        maxObjectZ = MathF.Max(maxObjectZ, obj.Center.Z);
                    }

                    // Store MSCN points in global world space (X↔Y swapped relative to tile coords)
                    // MSCN/MSPV extraction removed from construction path — too slow.
                    // Extracted on-demand via EnsurePm4MscnData() / EnsurePm4MspvData() when the
                    // "MSCN Nodes" / "MSPV Nodes" checkboxes are enabled. See
                    // wow-viewer/docs/architecture/pm4-chunk-semantics.md for what these streams are.

                    if (remainingPositionRefBudget > 0)
                    {
                        List<Vector3> positionRefs = BuildPm4PositionRefMarkers(pm4, Math.Min(Pm4MaxPositionRefsPerTile, remainingPositionRefBudget));
                        if (positionRefs.Count > 0)
                        {
                            if (tilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? existingPositionRefs))
                                existingPositionRefs.AddRange(positionRefs);
                            else
                                tilePositionRefs[(effectiveTileX, effectiveTileY)] = positionRefs;

                            positionRefCount += positionRefs.Count;
                            remainingPositionRefBudget -= positionRefs.Count;
                        }
                    }

                    tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                    loadedFiles++;
                    objectCount += objects.Count;
                    lineCount += objects.Sum(obj => obj.Lines.Count);
                    triangleCount += objects.Sum(obj => obj.Triangles.Count);
                    rejectedLongEdgesTotal += rejectedLongEdges;

                    // Spec 054: store the per-file payload in the
                    // in-memory per-file cache so a future camera-window
                    // shift that touches this same file can skip the
                    // BuildPm4TileObjects + budget enforcement work.
                    StorePerFileInMemoryCache(
                        normalizedPm4Path,
                        bytes.Length,
                        effectiveTileX,
                        effectiveTileY,
                        objects,
                        tilePositionRefs,
                        _pm4SplitCk24ByMscnRef,
                        _pm4SplitCk24ByConnectivity);

                    // Spec 054: also persist to the on-disk per-file
                    // cache. Best-effort; a failure here does not break
                    // the load, it just means the next session's reload
                    // of this file will decode fresh instead of reading
                    // from disk. The on-disk entry uses the loose-file
                    // write-tick as its stamp (falls back to 0 for MPQ
                    // data sources where stamps are not available).
                    CorePm4PerFileCacheService? perFileDiskCache = EnsurePerFileDiskCache(mapName);
                    if (perFileDiskCache != null)
                    {
                        long diskStamp = 0L;
                        if (_dataSource != null
                            && Pm4OverlayCacheService.TryGetLooseFileStamp(_dataSource, pm4Path, out _, out long looseTicks))
                        {
                            diskStamp = looseTicks;
                        }

                        perFileDiskCache.Write(
                            normalizedPm4Path,
                            new CorePm4PerFileCacheEntry(
                                FileLength: bytes.Length,
                                LastWriteTicks: diskStamp,
                                Tiles: new[]
                                {
                                    new CorePm4CachedTile(
                                        TileX: effectiveTileX,
                                        TileY: effectiveTileY,
                                        PositionRefs: tilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? refs)
                                            ? new List<Vector3>(refs)
                                            : new List<Vector3>(),
                                        Objects: BuildCachedObjectsForDiskWrite(objects))
                                }));
                    }
                }
                catch (Exception ex)
                {
                    decodeFailed++;
                    ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] Failed to decode '{pm4Path}': {ex.Message}");
                }

                long elapsedMs = progressStopwatch.ElapsedMilliseconds;
                if (elapsedMs - lastStatusReportMs >= Pm4ProgressStatusIntervalMs || processedFiles == loadCandidates.Count)
                {
                    bool emitLog = elapsedMs - lastLogReportMs >= Pm4ProgressLogIntervalMs || processedFiles == loadCandidates.Count;
                    ReportPm4LoadProgress(requestId, "decoding", processedFiles, loadCandidates.Count, loadedFiles, objectCount, lineCount, triangleCount, readFailed, decodeFailed, zeroObjectFiles, memCacheHits, diskCacheHits, pm4Path, emitLog);
                    lastStatusReportMs = elapsedMs;
                    if (emitLog)
                        lastLogReportMs = elapsedMs;
                }
            }

            HashSet<(int tileX, int tileY)> coveredMapTiles = tileCandidateCounts
                .Where(entry => tileSatisfiedCounts[entry.Key] >= entry.Value)
                .Select(static entry => entry.Key)
                .ToHashSet();

            if (loadedFiles == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    cameraWindow,
                    knownMapTiles,
                    coveredMapTiles,
                    selectedObjectKey,
                    loadStopwatch.Elapsed.TotalMilliseconds,
                    $"PM4: {loadCandidates.Count}/{totalFiles} camera-window files found, none decoded into overlay data for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}) (tileParse={tileParseRejected}, tileRange={tileRangeRejected}, read={readFailed}, decode={decodeFailed}, zeroObjects={zeroObjectFiles}).",
                    cancelled: false);
            }

            if (minObjectZ > maxObjectZ)
            {
                minObjectZ = 0f;
                maxObjectZ = 1f;
            }

            loadStopwatch.Stop();
            Pm4OverlayCacheData cacheData = BuildPm4OverlayCacheData(
                mapName,
                candidateSignature,
                totalFiles,
                loadedFiles,
                objectCount,
                lineCount,
                triangleCount,
                positionRefCount,
                rejectedLongEdgesTotal,
                minObjectZ,
                maxObjectZ,
                tileObjects,
                tilePositionRefs);
            if (_pm4OverlayCacheService != null)
            {
                if (!_pm4OverlayCacheService.TrySave(cacheData, out string? cacheSaveError) && !string.IsNullOrWhiteSpace(cacheSaveError))
                    ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4] {cacheSaveError}");
            }

            return new Pm4OverlayAsyncLoadResult(
                requestId,
                cacheData,
                cameraWindow,
                knownMapTiles,
                coveredMapTiles,
                selectedObjectKey,
                loadStopwatch.Elapsed.TotalMilliseconds,
                $"PM4 ready: {loadedFiles}/{loadCandidates.Count} camera-window files (mem-cache {memCacheHits} hit, {memCacheMisses} fresh-decode) for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}), avg {_pm4AverageLoadMs:0} ms, next radius {_pm4CameraTileRadius}, from {mapPm4CandidateCount} map files, {objectCount} objects, {lineCount} lines, {triangleCount} triangles, {positionRefCount} refs, {rejectedLongEdgesTotal} long edges rejected, {loadStopwatch.ElapsedMilliseconds} ms.",
                cancelled: false);
        }
        catch (OperationCanceledException)
        {
            return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, "PM4 load cancelled.", cancelled: true);
        }
        catch (Exception ex)
        {
            return new Pm4OverlayAsyncLoadResult(requestId, null, null, [], [], selectedObjectKey, 0.0, $"PM4 load failed: {ex.Message}", cancelled: false);
        }
    }

    private void ClearPm4OverlayRuntimeState()
    {
        _pm4LoadedCameraWindow = null;
        _pm4CoveredMapTiles.Clear();
        _pm4KnownMapTiles.Clear();
        _pm4TileObjects.Clear();
        _pm4TileMscnPoints.Clear();
        _pm4TileMspvPoints.Clear();
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ResearchBySourcePath.Clear();
        _pm4ResearchUnavailablePaths.Clear();
        _pm4ObjectLookup.Clear();
        _pm4MergedObjectGroupKeys.Clear();
        _pm4GroupToObjectKeys.Clear();
        _pm4ObjectGroupBounds.Clear();
        _pm4TotalFiles = 0;
        _pm4LoadedFiles = 0;
        _pm4ObjectCount = 0;
        _pm4LineCount = 0;
        _pm4TriangleCount = 0;
        _pm4RejectedLongEdges = 0;
        _pm4VisibleObjectCount = 0;
        _pm4VisibleLineCount = 0;
        _pm4VisibleTriangleCount = 0;
        _pm4PositionRefCount = 0;
        _pm4VisiblePositionRefCount = 0;
        _pm4MinObjectZ = float.MaxValue;
        _pm4MaxObjectZ = float.MinValue;
    }

    private static Pm4OverlayCacheData BuildPm4OverlayCacheData(
        string mapName,
        string candidateSignature,
        int totalFiles,
        int loadedFiles,
        int objectCount,
        int lineCount,
        int triangleCount,
        int positionRefCount,
        int rejectedLongEdges,
        float minObjectZ,
        float maxObjectZ,
        Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> tileObjects,
        Dictionary<(int tileX, int tileY), List<Vector3>> tilePositionRefs)
    {
        var tiles = new List<Pm4OverlayCacheTile>(tileObjects.Count);
        foreach (var tileEntry in tileObjects.OrderBy(static entry => entry.Key.tileX).ThenBy(static entry => entry.Key.tileY))
        {
            List<Vector3> positionRefs = tilePositionRefs.TryGetValue(tileEntry.Key, out List<Vector3>? existingPositionRefs)
                ? existingPositionRefs
                : new List<Vector3>();
            var objects = new List<Pm4OverlayCacheObject>(tileEntry.Value.Count);
            for (int i = 0; i < tileEntry.Value.Count; i++)
            {
                Pm4OverlayObject obj = tileEntry.Value[i];
                objects.Add(new Pm4OverlayCacheObject(
                    obj.SourcePath,
                    obj.MshdField00,
                    obj.MshdRegionId,
                    obj.MshdField08,
                    obj.Ck24,
                    obj.Ck24Type,
                    obj.ObjectPartId,
                    obj.LinkGroupObjectId,
                    obj.LinkedPositionRefCount,
                    obj.LinkedPositionRefSummary,
                    obj.Lines,
                    obj.Triangles,
                    obj.SurfaceCount,
                    obj.TotalIndexCount,
                    obj.DominantGroupKey,
                    obj.DominantAttributeMask,
                    obj.DominantMscnRefIndex,
                    obj.AverageSurfaceHeight,
                    obj.PlacementAnchor,
                    obj.BaseRotationRadians,
                    obj.PlanarTransform,
                    obj.BoundsMin,
                    obj.BoundsMax,
                    obj.ConnectorKeys.ToList()));
            }

            tiles.Add(new Pm4OverlayCacheTile(tileEntry.Key.tileX, tileEntry.Key.tileY, objects, positionRefs));
        }

        return new Pm4OverlayCacheData(
            mapName,
            candidateSignature,
            totalFiles,
            loadedFiles,
            objectCount,
            lineCount,
            triangleCount,
            positionRefCount,
            rejectedLongEdges,
            minObjectZ,
            maxObjectZ,
            tiles);
    }

    private void RestoreSelectedPm4Object((int tileX, int tileY, uint ck24, int objectPart)? selectedObjectKey)
    {
        if (!selectedObjectKey.HasValue)
        {
            _selectedPm4ObjectKey = null;
            _selectedPm4ObjectGroupKey = null;
            return;
        }

        if (_pm4ObjectLookup.ContainsKey(selectedObjectKey.Value))
        {
            _selectedPm4ObjectKey = selectedObjectKey;
            _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(selectedObjectKey.Value);
            return;
        }

        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
    }

    private Vector3 GetPm4LoadAnchorCameraPosition()
    {
        if (_hasLastRenderedCameraPosition)
            return _lastRenderedCameraPosition;

        return _terrainManager.GetInitialCameraPosition();
    }

    private static (int minTileX, int minTileY, int maxTileX, int maxTileY) GetPm4CameraWindow(Vector3 cameraPos, int tileRadius)
    {
        GetPm4CameraTile(cameraPos, out int centerTileX, out int centerTileY);
        int minTileX = Math.Max(0, centerTileX - tileRadius);
        int minTileY = Math.Max(0, centerTileY - tileRadius);
        int maxTileX = Math.Min(63, centerTileX + tileRadius);
        int maxTileY = Math.Min(63, centerTileY + tileRadius);
        return (minTileX, minTileY, maxTileX, maxTileY);
    }

    private static void GetPm4CameraTile(Vector3 cameraPos, out int tileX, out int tileY)
    {
        // PM4 filenames and terrain AOI both operate on ADT tile coordinates (64x64 grid).
        // WoWConstants.TileSize is the larger WDL tile span, which collapses camera-window
        // PM4 loads into a tiny corner of the map. Use the ADT tile span instead.
        float camTileX = (WoWConstants.MapOrigin - cameraPos.X) / WoWConstants.ChunkSize;
        float camTileY = (WoWConstants.MapOrigin - cameraPos.Y) / WoWConstants.ChunkSize;
        tileX = Math.Clamp((int)MathF.Floor(camTileX), 0, 63);
        tileY = Math.Clamp((int)MathF.Floor(camTileY), 0, 63);
    }

    private static bool IsPm4TileInsideCameraWindow(
        int tileX,
        int tileY,
        (int minTileX, int minTileY, int maxTileX, int maxTileY) cameraWindow)
    {
        return tileX >= cameraWindow.minTileX
            && tileX <= cameraWindow.maxTileX
            && tileY >= cameraWindow.minTileY
            && tileY <= cameraWindow.maxTileY;
    }

    private void EnsurePm4OverlayMatchesCameraWindow(Vector3 cameraPos)
    {
        if (!_showPm4Overlay)
            return;

        if (_pm4LoadTask != null && !_pm4LoadTask.IsCompleted)
            return;

        if (!_pm4LoadAttempted || !_pm4LoadedCameraWindow.HasValue)
        {
            BeginPm4OverlayLoad();
            return;
        }

        var desiredWindow = GetPm4CameraWindow(cameraPos, _pm4CameraTileRadius);
        if (!IsPm4CameraWindowCovered(desiredWindow))
            BeginPm4OverlayLoad();
    }

    private bool IsPm4CameraWindowCovered((int minTileX, int minTileY, int maxTileX, int maxTileY) cameraWindow)
    {
        if (_pm4KnownMapTiles.Count > 0)
        {
            bool hasKnownTileInWindow = false;
            foreach ((int tileX, int tileY) in _pm4KnownMapTiles)
            {
                if (!IsPm4TileInsideCameraWindow(tileX, tileY, cameraWindow))
                    continue;

                hasKnownTileInWindow = true;

                if (!_pm4CoveredMapTiles.Contains((tileX, tileY)))
                    return false;
            }

            if (hasKnownTileInWindow)
                return true;
        }

        if (!_pm4LoadedCameraWindow.HasValue)
            return false;

        var loadedWindow = _pm4LoadedCameraWindow.Value;
        return cameraWindow.minTileX >= loadedWindow.minTileX
            && cameraWindow.minTileY >= loadedWindow.minTileY
            && cameraWindow.maxTileX <= loadedWindow.maxTileX
            && cameraWindow.maxTileY <= loadedWindow.maxTileY;
    }

    private void ExpandPm4LoadedCameraWindow((int minTileX, int minTileY, int maxTileX, int maxTileY) window)
    {
        if (!_pm4LoadedCameraWindow.HasValue)
        {
            _pm4LoadedCameraWindow = window;
            return;
        }

        var existing = _pm4LoadedCameraWindow.Value;
        _pm4LoadedCameraWindow = (
            Math.Min(existing.minTileX, window.minTileX),
            Math.Min(existing.minTileY, window.minTileY),
            Math.Max(existing.maxTileX, window.maxTileX),
            Math.Max(existing.maxTileY, window.maxTileY));
    }

    private void UpdatePm4AdaptiveWindow(double loadElapsedMs)
    {
        _pm4AverageLoadMs = _pm4AverageLoadMs < 0.0
            ? loadElapsedMs
            : _pm4AverageLoadMs * 0.65 + loadElapsedMs * 0.35;

        int previousRadius = _pm4CameraTileRadius;
        if (_pm4AverageLoadMs >= Pm4ShrinkWindowThresholdMs && _pm4CameraTileRadius > Pm4MinCameraTileRadius)
            _pm4CameraTileRadius--;
        else if (_pm4AverageLoadMs <= Pm4ExpandWindowThresholdMs && _pm4CameraTileRadius < Pm4MaxCameraTileRadius)
            _pm4CameraTileRadius++;

        if (previousRadius != _pm4CameraTileRadius)
        {
            ViewerLog.Info(
                ViewerLog.Category.Terrain,
                $"[PM4] Adaptive window radius changed {previousRadius} -> {_pm4CameraTileRadius} (avg {_pm4AverageLoadMs:0} ms).");
        }
    }

    private Pm4OverlayCacheData BuildPm4OverlayCacheData(string mapName, string candidateSignature)
    {
        var tiles = new List<Pm4OverlayCacheTile>(_pm4TileObjects.Count);
        foreach (var tileEntry in _pm4TileObjects.OrderBy(static entry => entry.Key.tileX).ThenBy(static entry => entry.Key.tileY))
        {
            List<Vector3> positionRefs = _pm4TilePositionRefs.TryGetValue(tileEntry.Key, out List<Vector3>? existingPositionRefs)
                ? existingPositionRefs
                : new List<Vector3>();
            var objects = new List<Pm4OverlayCacheObject>(tileEntry.Value.Count);
            for (int i = 0; i < tileEntry.Value.Count; i++)
            {
                Pm4OverlayObject obj = tileEntry.Value[i];
                objects.Add(new Pm4OverlayCacheObject(
                    obj.SourcePath,
                    obj.MshdField00,
                    obj.MshdRegionId,
                    obj.MshdField08,
                    obj.Ck24,
                    obj.Ck24Type,
                    obj.ObjectPartId,
                    obj.LinkGroupObjectId,
                    obj.LinkedPositionRefCount,
                    obj.LinkedPositionRefSummary,
                    obj.Lines,
                    obj.Triangles,
                    obj.SurfaceCount,
                    obj.TotalIndexCount,
                    obj.DominantGroupKey,
                    obj.DominantAttributeMask,
                    obj.DominantMscnRefIndex,
                    obj.AverageSurfaceHeight,
                    obj.PlacementAnchor,
                    obj.BaseRotationRadians,
                    obj.PlanarTransform,
                    obj.BoundsMin,
                    obj.BoundsMax,
                    obj.ConnectorKeys.ToList()));
            }

            tiles.Add(new Pm4OverlayCacheTile(tileEntry.Key.tileX, tileEntry.Key.tileY, objects, positionRefs));
        }

        return new Pm4OverlayCacheData(
            mapName,
            candidateSignature,
            _pm4TotalFiles,
            _pm4LoadedFiles,
            _pm4ObjectCount,
            _pm4LineCount,
            _pm4TriangleCount,
            _pm4PositionRefCount,
            _pm4RejectedLongEdges,
            _pm4MinObjectZ,
            _pm4MaxObjectZ,
            tiles);
    }

    private void MergePm4OverlayFromCache(Pm4OverlayCacheData cacheData)
    {
        for (int tileIndex = 0; tileIndex < cacheData.Tiles.Count; tileIndex++)
        {
            Pm4OverlayCacheTile tile = cacheData.Tiles[tileIndex];
            var tileKey = (tile.TileX, tile.TileY);

            var objects = new List<Pm4OverlayObject>(tile.Objects.Count);
            for (int objectIndex = 0; objectIndex < tile.Objects.Count; objectIndex++)
            {
                Pm4OverlayCacheObject cachedObject = tile.Objects[objectIndex];
                Pm4OverlayObject restored = Pm4OverlayObject.FromCachedLocalized(
                    cachedObject.SourcePath,
                    cachedObject.MshdField00,
                    cachedObject.MshdRegionId,
                    cachedObject.MshdField08,
                    cachedObject.Ck24,
                    cachedObject.Ck24Type,
                    cachedObject.ObjectPartId,
                    cachedObject.LinkGroupObjectId,
                    cachedObject.LinkedPositionRefCount,
                    cachedObject.LinkedPositionRefSummary,
                    new List<Pm4LineSegment>(cachedObject.Lines),
                    new List<Pm4Triangle>(cachedObject.Triangles),
                    cachedObject.SurfaceCount,
                    cachedObject.TotalIndexCount,
                    cachedObject.DominantGroupKey,
                    cachedObject.DominantAttributeMask,
                    cachedObject.DominantMscnRefIndex,
                    cachedObject.AverageSurfaceHeight,
                    cachedObject.PlacementAnchor,
                    cachedObject.BaseRotationRadians,
                    cachedObject.PlanarTransform,
                    cachedObject.BoundsMin,
                    cachedObject.BoundsMax,
                    cachedObject.ConnectorKeys.ToList());
                objects.Add(restored);
                _pm4ObjectLookup[(tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId)] = restored;
                var groupKey = BuildPm4BaseObjectGroupKey((tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId));
                if (!_pm4GroupToObjectKeys.TryGetValue(groupKey, out var groupObjectKeys))
                {
                    groupObjectKeys = new List<(int, int, uint, int)>();
                    _pm4GroupToObjectKeys[groupKey] = groupObjectKeys;
                }
                groupObjectKeys.Add((tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId));
            }

            _pm4TileObjects[tileKey] = objects;
            _pm4TileStats[tileKey] = new Pm4OverlayTileStats(
                tile.TileX,
                tile.TileY,
                objects.Count,
                objects.Sum(static obj => obj.Lines.Count),
                objects.Sum(static obj => obj.Triangles.Count));

            if (tile.PositionRefs.Count > 0)
                _pm4TilePositionRefs[tileKey] = new List<Vector3>(tile.PositionRefs);
        }

        if (_pm4MinObjectZ > _pm4MaxObjectZ)
        {
            _pm4MinObjectZ = 0f;
            _pm4MaxObjectZ = 1f;
        }

        _pm4TotalFiles = Math.Max(_pm4TotalFiles, cacheData.TotalFiles);
        RecalculatePm4OverlayRuntimeTotals();
    }

    /// <summary>
    /// Spec 054: Apply a per-file in-memory cache hit to the camera-window
    /// load's per-tile dictionaries. The cache value is a list of
    /// <see cref="CorePm4CachedTile"/>s produced by an earlier decode of
    /// the same PM4 file; we materialize each one into
    /// <see cref="Pm4OverlayObject"/> records and add it to the local
    /// <paramref name="tileObjects"/> / <paramref name="tilePositionRefs"/>
    /// dictionaries that the load path consumes.
    /// </summary>
    private static int ApplyCachedTilesToTileDictionaries(
        CorePm4PerFileCacheEntry cachedEntry,
        int fallbackTileX,
        int fallbackTileY,
        Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> tileObjects,
        Dictionary<(int tileX, int tileY), List<Vector3>> tilePositionRefs,
        ref float minObjectZ,
        ref float maxObjectZ,
        ref int objectCount,
        ref int lineCount,
        ref int triangleCount,
        ref int positionRefCount)
    {
        int addedObjectCount = 0;
        for (int tileIndex = 0; tileIndex < cachedEntry.Tiles.Count; tileIndex++)
        {
            CorePm4CachedTile cachedTile = cachedEntry.Tiles[tileIndex];
            int tileX = cachedTile.TileX;
            int tileY = cachedTile.TileY;
            var tileKey = (tileX, tileY);

            var objects = new List<Pm4OverlayObject>(cachedTile.Objects.Count);
            for (int objectIndex = 0; objectIndex < cachedTile.Objects.Count; objectIndex++)
            {
                CorePm4CachedObject cached = cachedTile.Objects[objectIndex];
                Pm4OverlayObject restored = Pm4OverlayObject.FromCachedLocalized(
                    cached.SourcePath,
                    cached.MshdField00,
                    cached.MshdRegionId,
                    cached.MshdField08,
                    cached.Ck24,
                    cached.Ck24Type,
                    cached.ObjectPartId,
                    cached.LinkGroupObjectId,
                    cached.LinkedPositionRefCount,
                    FromCorePm4LinkedPositionRefSummary(cached.LinkedPositionRefSummary),
                    cached.Lines
                        .Select(static seg => new Pm4LineSegment(seg.From, seg.To))
                        .ToList(),
                    cached.Triangles
                        .Select(static tri => new Pm4Triangle(tri.A, tri.B, tri.C))
                        .ToList(),
                    cached.SurfaceCount,
                    cached.TotalIndexCount,
                    cached.DominantGroupKey,
                    cached.DominantAttributeMask,
                    cached.DominantMscnRefIndex,
                    cached.AverageSurfaceHeight,
                    cached.PlacementAnchor,
                    cached.BaseRotationRadians,
                    new Pm4PlanarTransform(
                        cached.PlanarSwapPlanarAxes,
                        cached.PlanarInvertU,
                        cached.PlanarInvertV),
                    cached.BoundsMin,
                    cached.BoundsMax,
                    cached.ConnectorKeys
                        .Select(static k => new Pm4ConnectorKey(k.X, k.Y, k.Z))
                        .ToList());
                objects.Add(restored);
            }

            if (tileObjects.TryGetValue(tileKey, out List<Pm4OverlayObject>? existingObjects))
            {
                int objectPartOffset = existingObjects.Count;
                List<Pm4OverlayObject> rebased = RebasePm4ObjectParts(objects, objectPartOffset);
                existingObjects.AddRange(rebased);
                objects = rebased;
            }
            else
            {
                tileObjects[tileKey] = objects;
            }

            for (int objIndex = 0; objIndex < objects.Count; objIndex++)
            {
                minObjectZ = MathF.Min(minObjectZ, objects[objIndex].Center.Z);
                maxObjectZ = MathF.Max(maxObjectZ, objects[objIndex].Center.Z);
            }

            if (cachedTile.PositionRefs.Count > 0)
            {
                if (tilePositionRefs.TryGetValue(tileKey, out List<Vector3>? existingRefs))
                    existingRefs.AddRange(cachedTile.PositionRefs);
                else
                    tilePositionRefs[tileKey] = new List<Vector3>(cachedTile.PositionRefs);
            }

            addedObjectCount += objects.Count;
        }

        objectCount += addedObjectCount;
        for (int tileIndex = 0; tileIndex < cachedEntry.Tiles.Count; tileIndex++)
        {
            CorePm4CachedTile cachedTile = cachedEntry.Tiles[tileIndex];
            for (int objectIndex = 0; objectIndex < cachedTile.Objects.Count; objectIndex++)
            {
                CorePm4CachedObject cached = cachedTile.Objects[objectIndex];
                lineCount += cached.Lines.Count;
                triangleCount += cached.Triangles.Count;
            }
            positionRefCount += cachedTile.PositionRefs.Count;
        }
        return addedObjectCount;
    }

    /// <summary>
    /// Spec 054: After a successful per-file decode, store the payload in
    /// the in-memory per-file cache. The split flags are folded into the
    /// entry's stamp so an entry decoded with one set of split flags is
    /// treated as a miss when the user toggles a split-flag and re-loads.
    /// </summary>
    private void StorePerFileInMemoryCache(
        string normalizedPath,
        long fileLength,
        int effectiveTileX,
        int effectiveTileY,
        IReadOnlyList<Pm4OverlayObject> objects,
        IReadOnlyDictionary<(int tileX, int tileY), List<Vector3>> tilePositionRefs,
        bool splitByMscnRef,
        bool splitByConnectivity)
    {
        if (objects.Count == 0)
            return;

        List<CorePm4CachedObject> cachedObjects = new(objects.Count);
        for (int i = 0; i < objects.Count; i++)
        {
            Pm4OverlayObject obj = objects[i];
            cachedObjects.Add(new CorePm4CachedObject(
                SourcePath: obj.SourcePath,
                MshdField00: obj.MshdField00,
                MshdRegionId: obj.MshdRegionId,
                MshdField08: obj.MshdField08,
                Ck24: obj.Ck24,
                Ck24Type: obj.Ck24Type,
                ObjectPartId: obj.ObjectPartId,
                LinkGroupObjectId: obj.LinkGroupObjectId,
                LinkedPositionRefCount: obj.LinkedPositionRefCount,
                LinkedPositionRefSummary: new CorePm4LinkedPositionRefSummary(
                    obj.LinkedPositionRefSummary.TotalCount,
                    obj.LinkedPositionRefSummary.NormalCount,
                    obj.LinkedPositionRefSummary.TerminatorCount,
                    obj.LinkedPositionRefSummary.FloorMin,
                    obj.LinkedPositionRefSummary.FloorMax,
                    obj.LinkedPositionRefSummary.HeadingMinDegrees,
                    obj.LinkedPositionRefSummary.HeadingMaxDegrees,
                    obj.LinkedPositionRefSummary.HeadingMeanDegrees),
                SurfaceCount: obj.SurfaceCount,
                TotalIndexCount: obj.TotalIndexCount,
                DominantGroupKey: obj.DominantGroupKey,
                DominantAttributeMask: obj.DominantAttributeMask,
                DominantMscnRefIndex: obj.DominantMscnRefIndex,
                AverageSurfaceHeight: obj.AverageSurfaceHeight,
                PlacementAnchor: obj.PlacementAnchor,
                BaseRotationRadians: obj.BaseRotationRadians,
                PlanarSwapPlanarAxes: obj.PlanarTransform.SwapPlanarAxes,
                PlanarInvertU: obj.PlanarTransform.InvertU,
                PlanarInvertV: obj.PlanarTransform.InvertV,
                BoundsMin: obj.BoundsMin,
                BoundsMax: obj.BoundsMax,
                ConnectorKeys: obj.ConnectorKeys
                    .Select(static k => new CorePm4CachedConnectorKey(k.X, k.Y, k.Z))
                    .ToList(),
                Lines: obj.Lines
                    .Select(static seg => new CorePm4CachedLineSegment(seg.From, seg.To))
                    .ToList(),
                Triangles: obj.Triangles
                    .Select(static tri => new CorePm4CachedTriangle(tri.A, tri.B, tri.C))
                    .ToList()));
        }

        List<Vector3> positionRefs = tilePositionRefs.TryGetValue((effectiveTileX, effectiveTileY), out List<Vector3>? refs)
            ? new List<Vector3>(refs)
            : new List<Vector3>();

        CorePm4CachedTile cachedTile = new(
            TileX: effectiveTileX,
            TileY: effectiveTileY,
            PositionRefs: positionRefs,
            Objects: cachedObjects);

        long splitStamp = ((splitByMscnRef ? 1L : 0L) << 32) | (splitByConnectivity ? 1L : 0L);

        _pm4PerFileInMemoryCache.Set(
            normalizedPath,
            new CorePm4PerFileCacheEntry(
                FileLength: fileLength,
                LastWriteTicks: splitStamp,
                Tiles: new[] { cachedTile }));
    }

    /// <summary>
    /// Spec 054: lazy accessor for the on-disk per-file cache. The
    /// service is constructed once per (dataSource, mapName) and
    /// cached for the lifetime of <see cref="WorldScene"/>. The cache
    /// root is derived from the per-window cache service's
    /// <see cref="Pm4OverlayCacheService.CacheRoot"/> so the two layers
    /// share the on-disk parent.
    /// </summary>
    private CorePm4PerFileCacheService? EnsurePerFileDiskCache(string mapName)
    {
        if (_pm4PerFileDiskCache != null)
            return _pm4PerFileDiskCache;
        if (_pm4OverlayCacheService == null)
            return null;
        if (_dataSource == null)
            return null;
        if (string.IsNullOrWhiteSpace(mapName))
            return null;

        string identity = _dataSource.Name ?? "default";
        _pm4PerFileDiskCache = CorePm4PerFileCacheService.CreateForDataSource(
            Path.Combine(_pm4OverlayCacheService.CacheRoot, "files"),
            identity,
            mapName);
        return _pm4PerFileDiskCache;
    }

    /// <summary>
    /// Spec 054: read a per-file entry from disk and gate on the file
    /// stamp. A stamp mismatch (file content changed since the entry
    /// was written) is treated as a miss and the on-disk entry is
    /// deleted so the next read is also a miss.
    /// </summary>
    private static bool TryReadPerFileDiskCache(
        CorePm4PerFileCacheService service,
        string normalizedPath,
        long fileLength,
        out CorePm4PerFileCacheEntry? entry)
    {
        entry = null;
        if (!service.TryRead(normalizedPath, out CorePm4PerFileCacheEntry? read))
            return false;
        if (read == null)
            return false;

        // We don't have a reliable loose-file write-tick at read time
        // (it requires the data source's overlay roots to be probed,
        // and the viewer side already passed us bytes.Length as a
        // proxy). For now we accept the entry when its recorded length
        // matches the current file length; a content edit that does
        // not change the byte count is rare for binary PM4 and is
        // accepted as a hit (a future enhancement can wire the loose
        // write-tick through the data-source interface).
        if (read.FileLength != fileLength)
        {
            service.Delete(normalizedPath);
            return false;
        }

        entry = read;
        return true;
    }

    /// <summary>
    /// Spec 054: build a list of <see cref="CorePm4CachedObject"/>s
    /// from a list of <see cref="Pm4OverlayObject"/>s for on-disk
    /// persistence. The library's record shape is independent of the
    /// viewer-side <c>Pm4OverlayCacheObject</c> so this is a separate
    /// pass (not a direct reuse of the in-memory cache's construction).
    /// </summary>
    private static List<CorePm4CachedObject> BuildCachedObjectsForDiskWrite(
        IReadOnlyList<Pm4OverlayObject> objects)
    {
        var cachedObjects = new List<CorePm4CachedObject>(objects.Count);
        for (int i = 0; i < objects.Count; i++)
        {
            Pm4OverlayObject obj = objects[i];
            cachedObjects.Add(new CorePm4CachedObject(
                SourcePath: obj.SourcePath,
                MshdField00: obj.MshdField00,
                MshdRegionId: obj.MshdRegionId,
                MshdField08: obj.MshdField08,
                Ck24: obj.Ck24,
                Ck24Type: obj.Ck24Type,
                ObjectPartId: obj.ObjectPartId,
                LinkGroupObjectId: obj.LinkGroupObjectId,
                LinkedPositionRefCount: obj.LinkedPositionRefCount,
                LinkedPositionRefSummary: new CorePm4LinkedPositionRefSummary(
                    obj.LinkedPositionRefSummary.TotalCount,
                    obj.LinkedPositionRefSummary.NormalCount,
                    obj.LinkedPositionRefSummary.TerminatorCount,
                    obj.LinkedPositionRefSummary.FloorMin,
                    obj.LinkedPositionRefSummary.FloorMax,
                    obj.LinkedPositionRefSummary.HeadingMinDegrees,
                    obj.LinkedPositionRefSummary.HeadingMaxDegrees,
                    obj.LinkedPositionRefSummary.HeadingMeanDegrees),
                SurfaceCount: obj.SurfaceCount,
                TotalIndexCount: obj.TotalIndexCount,
                DominantGroupKey: obj.DominantGroupKey,
                DominantAttributeMask: obj.DominantAttributeMask,
                DominantMscnRefIndex: obj.DominantMscnRefIndex,
                AverageSurfaceHeight: obj.AverageSurfaceHeight,
                PlacementAnchor: obj.PlacementAnchor,
                BaseRotationRadians: obj.BaseRotationRadians,
                PlanarSwapPlanarAxes: obj.PlanarTransform.SwapPlanarAxes,
                PlanarInvertU: obj.PlanarTransform.InvertU,
                PlanarInvertV: obj.PlanarTransform.InvertV,
                BoundsMin: obj.BoundsMin,
                BoundsMax: obj.BoundsMax,
                ConnectorKeys: obj.ConnectorKeys
                    .Select(static k => new CorePm4CachedConnectorKey(k.X, k.Y, k.Z))
                    .ToList(),
                Lines: obj.Lines
                    .Select(static seg => new CorePm4CachedLineSegment(seg.From, seg.To))
                    .ToList(),
                Triangles: obj.Triangles
                    .Select(static tri => new CorePm4CachedTriangle(tri.A, tri.B, tri.C))
                    .ToList()));
        }
        return cachedObjects;
    }

    private void RecalculatePm4OverlayRuntimeTotals()
    {
        _pm4LoadedFiles = _pm4TileObjects.Count;
        _pm4ObjectCount = 0;
        _pm4LineCount = 0;
        _pm4TriangleCount = 0;
        _pm4PositionRefCount = 0;
        _pm4RejectedLongEdges = 0;
        _pm4TotalMsurCount = 0;
        _pm4DroppedShortIndexCount = 0;
        _pm4WallFaceCount = 0;
        _pm4DroppedOutOfRangeMsviCount = 0;
        _pm4DroppedEmptyComponentCount = 0;
        _pm4MinObjectZ = float.MaxValue;
        _pm4MaxObjectZ = float.MinValue;

        foreach (var tileEntry in _pm4TileObjects)
        {
            List<Pm4OverlayObject> objects = tileEntry.Value;
            _pm4ObjectCount += objects.Count;
            _pm4LineCount += objects.Sum(static obj => obj.Lines.Count);
            _pm4TriangleCount += objects.Sum(static obj => obj.Triangles.Count);

            for (int i = 0; i < objects.Count; i++)
            {
                _pm4MinObjectZ = MathF.Min(_pm4MinObjectZ, objects[i].Center.Z);
                _pm4MaxObjectZ = MathF.Max(_pm4MaxObjectZ, objects[i].Center.Z);
            }
        }

        foreach (var refsEntry in _pm4TilePositionRefs)
            _pm4PositionRefCount += refsEntry.Value.Count;

        if (_pm4MinObjectZ > _pm4MaxObjectZ)
        {
            _pm4MinObjectZ = 0f;
            _pm4MaxObjectZ = 1f;
        }
    }

    private static bool IsMapPm4Path(string path, string mapName)
    {
        string normalized = path.Replace('\\', '/');
        string fileName = Path.GetFileName(normalized);
        if (fileName.StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase))
            return true;

        string mapSegment = "/" + mapName + "/";
        return normalized.Contains(mapSegment, StringComparison.OrdinalIgnoreCase);
    }

    private static string BuildPm4ObjText(IReadOnlyList<Pm4OverlayObject> objects, int tileX, int tileY)
    {
        var builder = new StringBuilder();
        builder.AppendLine($"# PM4 tile {tileX:D2}_{tileY:D2}");
        builder.AppendLine($"# object_count {objects.Count}");

        int vertexIndex = 1;
        foreach (Pm4OverlayObject obj in objects)
        {
            string objectName = $"tile_{tileX:D2}_{tileY:D2}_ck24_{obj.Ck24:X6}_part_{obj.ObjectPartId:D4}";
            builder.AppendLine();
            builder.AppendLine($"o {objectName}");
            builder.AppendLine($"# source {obj.SourcePath}");
            builder.AppendLine($"# lines {obj.Lines.Count} triangles {obj.Triangles.Count} surfaces {obj.SurfaceCount} total_indices {obj.TotalIndexCount}");

            Matrix4x4 transform = obj.BaseTransform;
            for (int i = 0; i < obj.Triangles.Count; i++)
            {
                Pm4Triangle tri = obj.Triangles[i];
                Vector3 a = ApplyPm4OverlayTransform(tri.A, transform);
                Vector3 b = ApplyPm4OverlayTransform(tri.B, transform);
                Vector3 c = ApplyPm4OverlayTransform(tri.C, transform);
                AppendObjVertex(builder, a);
                AppendObjVertex(builder, b);
                AppendObjVertex(builder, c);
                builder.Append("f ")
                    .Append(vertexIndex)
                    .Append(' ')
                    .Append(vertexIndex + 1)
                    .Append(' ')
                    .Append(vertexIndex + 2)
                    .AppendLine();
                vertexIndex += 3;
            }

            for (int i = 0; i < obj.Lines.Count; i++)
            {
                Pm4LineSegment line = obj.Lines[i];
                Vector3 from = ApplyPm4OverlayTransform(line.From, transform);
                Vector3 to = ApplyPm4OverlayTransform(line.To, transform);
                AppendObjVertex(builder, from);
                AppendObjVertex(builder, to);
                builder.Append("l ")
                    .Append(vertexIndex)
                    .Append(' ')
                    .Append(vertexIndex + 1)
                    .AppendLine();
                vertexIndex += 2;
            }
        }

        return builder.ToString();
    }

    private static void AppendObjVertex(StringBuilder builder, Vector3 vertex)
    {
        builder.Append("v ")
            .Append(vertex.X.ToString("G9", CultureInfo.InvariantCulture))
            .Append(' ')
            .Append(vertex.Y.ToString("G9", CultureInfo.InvariantCulture))
            .Append(' ')
            .Append(vertex.Z.ToString("G9", CultureInfo.InvariantCulture))
            .AppendLine();
    }

    private static string SanitizePm4ExportPathSegment(string value)
    {
        char[] invalidChars = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Length);
        for (int i = 0; i < value.Length; i++)
        {
            char current = value[i];
            builder.Append(invalidChars.Contains(current) ? '_' : current);
        }

        return builder.Length == 0 ? "pm4" : builder.ToString();
    }

    private static bool TryMapPm4FileTileToTerrainTile(int fileTileX, int fileTileY, out int terrainTileX, out int terrainTileY)
    {
        // PM4 filename tiles are transposed relative to ADT terrain tile naming on the
        // development corpus. Map PM4 file XX_YY onto terrain tile YY_XX so camera-window
        // loads and tile-local placement land on the same ADT tile the user is viewing.
        terrainTileX = fileTileY;
        terrainTileY = fileTileX;

        return terrainTileX is >= 0 and <= 63
            && terrainTileY is >= 0 and <= 63;
    }

    private bool ShouldRenderPm4Tile(int tileX, int tileY)
    {
        // PM4 overlay loading is already constrained by the PM4 camera window and object-level
        // culling. Gating PM4 by terrain AOI slices large structures across adjacent tiles,
        // which makes multi-tile WMO footprints like Stormwind Harbour disappear in pieces.
        return true;
    }

    private static List<Pm4OverlayObject> RebasePm4ObjectParts(IReadOnlyList<Pm4OverlayObject> objects, int objectPartOffset)
    {
        if (objects.Count == 0 || objectPartOffset == 0)
            return objects.ToList();

        var rebased = new List<Pm4OverlayObject>(objects.Count);
        for (int i = 0; i < objects.Count; i++)
        {
            Pm4OverlayObject obj = objects[i];
            rebased.Add(Pm4OverlayObject.FromCachedLocalized(
                obj.SourcePath,
                obj.MshdField00,
                obj.MshdRegionId,
                obj.MshdField08,
                obj.Ck24,
                obj.Ck24Type,
                obj.ObjectPartId + objectPartOffset,
                obj.LinkGroupObjectId,
                obj.LinkedPositionRefCount,
                obj.LinkedPositionRefSummary,
                obj.Lines,
                obj.Triangles,
                obj.SurfaceCount,
                obj.TotalIndexCount,
                obj.DominantGroupKey,
                obj.DominantAttributeMask,
                obj.DominantMscnRefIndex,
                obj.AverageSurfaceHeight,
                obj.PlacementAnchor,
                obj.BaseRotationRadians,
                obj.PlanarTransform,
                obj.BoundsMin,
                obj.BoundsMax,
                obj.ConnectorKeys));
        }

        return rebased;
    }

    private readonly struct Pm4IndexedSurface
    {
        public Pm4IndexedSurface(int surfaceIndex, MsurEntry surface)
        {
            SurfaceIndex = surfaceIndex;
            Surface = surface;
        }

        public int SurfaceIndex { get; }
        public MsurEntry Surface { get; }
    }

    private readonly struct Pm4OverlaySeedGroup
    {
        public Pm4OverlaySeedGroup(uint displayCk24, byte displayCk24Type, bool requiresConnectivitySeedSplit, List<Pm4IndexedSurface> surfaces)
        {
            DisplayCk24 = displayCk24;
            DisplayCk24Type = displayCk24Type;
            RequiresConnectivitySeedSplit = requiresConnectivitySeedSplit;
            Surfaces = surfaces;
        }

        public uint DisplayCk24 { get; }
        public byte DisplayCk24Type { get; }
        public bool RequiresConnectivitySeedSplit { get; }
        public List<Pm4IndexedSurface> Surfaces { get; }
    }

    private static List<Pm4OverlaySeedGroup> BuildPm4OverlaySeedGroups(Pm4File pm4)
    {
        List<Pm4IndexedSurface> indexedSurfaces = pm4.KnownChunks.Msur
            .Select((surface, surfaceIndex) => new Pm4IndexedSurface(surfaceIndex, surface))
            .Where(static indexedSurface => indexedSurface.Surface.IndexCount >= 3)
            .ToList();

        var groups = new List<Pm4OverlaySeedGroup>();
        foreach (IGrouping<uint, Pm4IndexedSurface> ck24Group in indexedSurfaces
            .Where(static indexedSurface => indexedSurface.Surface.Ck24 != 0)
            .GroupBy(static indexedSurface => indexedSurface.Surface.Ck24)
            .OrderBy(static group => group.Key))
        {
            groups.Add(new Pm4OverlaySeedGroup(
                ck24Group.Key,
                (byte)(ck24Group.Key >> 16),
                requiresConnectivitySeedSplit: false,
                ck24Group.ToList()));
        }

        foreach (IGrouping<(byte groupKey, byte attributeMask), Pm4IndexedSurface> zeroGroup in indexedSurfaces
            .Where(static indexedSurface => indexedSurface.Surface.Ck24 == 0)
            .GroupBy(static indexedSurface => (indexedSurface.Surface.GroupKey, indexedSurface.Surface.AttributeMask))
            .OrderBy(static group => group.Key.GroupKey)
            .ThenBy(static group => group.Key.AttributeMask))
        {
            groups.Add(new Pm4OverlaySeedGroup(
                0u,
                0,
                requiresConnectivitySeedSplit: true,
                zeroGroup.ToList()));
        }

        return groups;
    }

    private static List<Pm4OverlayObject> BuildPm4TileObjects(
        Pm4File pm4,
        string sourcePath,
        int tileX,
        int tileY,
        bool splitCk24ByMscnRef,
        bool splitCk24ByConnectivity,
        bool includePathWalls,
        ref int remainingLineBudget,
        ref int remainingTriangleBudget,
        ref int rejectedLongEdges,
        out Pm4TileBuildDiagnostics diagnostics)
    {
        diagnostics = new Pm4TileBuildDiagnostics
        {
            TotalMsurCount = pm4.KnownChunks.Msur.Count,
        };

        var objects = new List<Pm4OverlayObject>();
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;

        if (remainingLineBudget <= 0 || meshVertices.Count == 0)
            return objects;

        List<Pm4OverlaySeedGroup> seedGroups = BuildPm4OverlaySeedGroups(pm4);
        // The build also drops short-index surfaces at the seed-group stage; count them here.
        diagnostics.DroppedShortIndexCount = pm4.KnownChunks.Msur.Count(static s => s.IndexCount < 3);
        if (seedGroups.Count == 0)
            return objects;

        // MSLK path windows, indexed once per tile by the MSUR surface they reference.
        Dictionary<int, List<int>> mslkWindowsBySurface = includePathWalls
            ? BuildMslkWindowsBySurface(pm4)
            : [];

        var mshdGrouping = CorePm4MshdGroupingService.Describe(pm4.KnownChunks.Mshd);
        Pm4AxisConvention fileAxisConvention = DetectPm4AxisConvention(pm4);
        bool fallbackTileLocalCoordinates = IsLikelyTileLocal(meshVertices);
        int tileLineBudget = Math.Min(Pm4MaxLinesPerTile, remainingLineBudget);
        int tileTriangleBudget = Math.Min(Pm4MaxTrianglesPerTile, remainingTriangleBudget);

        // Viewer-generated split id used as a stable handle for this overlay build.
        // This is not a raw PM4 field from disk.
        int nextObjectPartId = 0;
        foreach (Pm4OverlaySeedGroup seedGroup in seedGroups)
        {
            if (tileLineBudget <= 0)
                break;

            uint ck24 = seedGroup.DisplayCk24;
            byte ck24Type = seedGroup.DisplayCk24Type;
            List<Pm4IndexedSurface> surfaceGroup = seedGroup.Surfaces;
            Pm4AxisConvention ck24AxisConvention = fileAxisConvention;
            List<MsurEntry> ck24Surfaces = surfaceGroup.Select(static entry => entry.Surface).ToList();
            List<MprlEntry> ck24PositionRefs = CollectLinkedPositionRefs(pm4, surfaceGroup);
            CorePm4CoordinateModeResolution seedCoordinateModeResolution = ResolveCk24CoordinateModeResolution(
                pm4,
                ck24Surfaces,
                ck24PositionRefs,
                tileX,
                tileY,
                ck24AxisConvention,
                fallbackTileLocalCoordinates);
            bool seedUseTileLocalCoordinates = seedCoordinateModeResolution.CoordinateMode == CorePm4CoordinateMode.TileLocal;
            // Keep one shared planar transform per CK24 so split linked/components stay on one coordinate plane.
            CorePm4PlacementSolution seedPlacement = ResolvePlacementSolution(
                pm4,
                ck24Surfaces,
                ck24PositionRefs,
                tileX,
                tileY,
                seedUseTileLocalCoordinates,
                ck24AxisConvention);
            Pm4PlanarTransform seedPlanarTransform = seedCoordinateModeResolution.PlanarTransform;
            Vector3 seedWorldPivot = seedPlacement.WorldPivot;
            float seedWorldYawCorrection = seedPlacement.WorldYawCorrectionRadians;
            float seedRendererFrameRotationRadians = ConvertWorldYawCorrectionToRendererRotationRadians(seedWorldYawCorrection);
            IReadOnlyList<Pm4ConnectorKey> seedConnectorKeys = BuildCk24ConnectorKeys(pm4, ck24Surfaces, seedPlacement);
            List<List<Pm4IndexedSurface>> linkedGroups = seedGroup.RequiresConnectivitySeedSplit
                ? SplitZeroCk24SeedGroup(pm4, surfaceGroup)
                : SplitSurfaceGroupByMslk(pm4, surfaceGroup);

            foreach (List<Pm4IndexedSurface> linkedGroup in linkedGroups)
            {
                if (linkedGroup.Count == 0 || tileLineBudget <= 0)
                    continue;

                uint dominantLinkGroupObjectId = SelectDominantMslkGroupObjectId(pm4, linkedGroup);
                List<MsurEntry> linkedSurfaces = linkedGroup.Select(static entry => entry.Surface).ToList();
                List<MprlEntry> linkedPositionRefs = CollectLinkedPositionRefs(pm4, linkedGroup);
                Pm4LinkedPositionRefSummary linkedPositionRefSummary = SummarizeLinkedPositionRefs(linkedPositionRefs);

                CorePm4CoordinateModeResolution linkedCoordinateModeResolution = ResolveCk24CoordinateModeResolution(
                    pm4,
                    linkedSurfaces,
                    linkedPositionRefs,
                    tileX,
                    tileY,
                    ck24AxisConvention,
                    fallbackTileLocalCoordinates);
                bool linkedUseTileLocalCoordinates = linkedCoordinateModeResolution.CoordinateMode == CorePm4CoordinateMode.TileLocal;

                CorePm4PlacementSolution linkedPlacement = ResolvePlacementSolution(
                    pm4,
                    linkedSurfaces,
                    linkedPositionRefs,
                    tileX,
                    tileY,
                    linkedUseTileLocalCoordinates,
                    ck24AxisConvention);

                Pm4PlanarTransform linkedPlanarTransform = linkedCoordinateModeResolution.PlanarTransform;
                Vector3 linkedWorldPivot = linkedPlacement.WorldPivot;
                float linkedWorldYawCorrection = linkedPlacement.WorldYawCorrectionRadians;
                float linkedRendererFrameRotationRadians = ConvertWorldYawCorrectionToRendererRotationRadians(linkedWorldYawCorrection);
                IReadOnlyList<Pm4ConnectorKey> linkedConnectorKeys = BuildCk24ConnectorKeys(pm4, linkedSurfaces, linkedPlacement);

                Vector3 linkedPlacementAnchor = ComputeSurfaceRendererCentroid(
                    pm4,
                    linkedSurfaces,
                    tileX,
                    tileY,
                    linkedUseTileLocalCoordinates,
                    ck24AxisConvention,
                    linkedPlanarTransform,
                    linkedWorldPivot,
                    linkedWorldYawCorrection);
                // MSUR.MsviFirstIndex is the surface's window start, so it recovers the surface
                // index after the split helpers have reduced indexed surfaces to bare entries.
                Dictionary<uint, int> surfaceIndexByMsviFirst = [];
                if (includePathWalls)
                {
                    foreach (Pm4IndexedSurface indexed in linkedGroup)
                        surfaceIndexByMsviFirst[indexed.Surface.MsviFirstIndex] = indexed.SurfaceIndex;
                }

                bool allowNestedSeedSplits = !seedGroup.RequiresConnectivitySeedSplit;
                List<List<MsurEntry>> anchorGroups = splitCk24ByMscnRef && allowNestedSeedSplits
                    ? SplitSurfaceGroupByMscnRef(linkedSurfaces)
                    : new List<List<MsurEntry>> { linkedSurfaces };

                foreach (List<MsurEntry> anchorGroup in anchorGroups)
                {
                    List<List<MsurEntry>> components = splitCk24ByConnectivity && allowNestedSeedSplits
                        ? SplitSurfaceGroupByConnectivity(pm4, anchorGroup)
                        : new List<List<MsurEntry>> { anchorGroup };

                    foreach (List<MsurEntry> component in components)
                    {
                        if (tileLineBudget <= 0)
                            break;

                        // Keep split components under one linked-group frame basis.
                        // MSUR 0x1C / CK24 is not sufficient to guarantee one shared object rotation
                        // across every linked sub-object in a seed group, especially on large WMO
                        // interiors where repeated carriers can appear under the same CK24 value.
                        List<Pm4LineSegment> lines = BuildCk24ObjectLines(pm4, component, tileX, tileY, linkedUseTileLocalCoordinates, ck24AxisConvention, linkedPlanarTransform, linkedWorldPivot, linkedWorldYawCorrection, tileLineBudget, ref rejectedLongEdges);
                        int componentRejectedOutOfRange = 0;
                        List<Pm4Triangle> triangles = tileTriangleBudget > 0
                            ? BuildCk24ObjectTriangles(pm4, component, tileX, tileY, linkedUseTileLocalCoordinates, ck24AxisConvention, linkedPlanarTransform, linkedWorldPivot, linkedWorldYawCorrection, tileTriangleBudget, out componentRejectedOutOfRange)
                            : new List<Pm4Triangle>();

                        diagnostics.DroppedOutOfRangeMsviCount += componentRejectedOutOfRange;

                        // Append this component's wall faces to the same mesh, so they render,
                        // pick and select as part of the object they stand on.
                        if (includePathWalls && mslkWindowsBySurface.Count > 0 && tileTriangleBudget > 0)
                        {
                            HashSet<int> componentSurfaceIndices = [];
                            foreach (MsurEntry componentSurface in component)
                            {
                                if (surfaceIndexByMsviFirst.TryGetValue(componentSurface.MsviFirstIndex, out int surfaceIndex))
                                    componentSurfaceIndices.Add(surfaceIndex);
                            }

                            if (componentSurfaceIndices.Count > 0)
                            {
                                var wallLines = new List<Pm4LineSegment>();
                                List<Pm4Triangle> wallTriangles = BuildMslkWallTriangles(
                                    pm4,
                                    componentSurfaceIndices,
                                    mslkWindowsBySurface,
                                    tileX,
                                    tileY,
                                    linkedUseTileLocalCoordinates,
                                    ck24AxisConvention,
                                    linkedPlanarTransform,
                                    Math.Max(0, tileTriangleBudget - triangles.Count),
                                    wallLines,
                                    Math.Max(0, tileLineBudget - lines.Count),
                                    out int componentWallFaces);

                                diagnostics.WallFaceCount += componentWallFaces;
                                triangles.AddRange(wallTriangles);
                                lines.AddRange(wallLines);
                            }
                        }

                        if (lines.Count == 0 && triangles.Count == 0)
                        {
                            diagnostics.DroppedEmptyComponentCount++;
                            continue;
                        }

                        byte dominantGroupKey = SelectDominantSurfaceValue(component, static surface => surface.GroupKey);
                        byte dominantAttributeMask = SelectDominantSurfaceValue(component, static surface => surface.AttributeMask);
                        uint dominantMscnRefIndex = SelectDominantSurfaceValue(component, static surface => surface.MscnRefIndex);
                        float averageSurfaceHeight = component.Count > 0 ? component.Average(static surface => surface.Height) : 0f;
                        int totalIndexCount = component.Sum(static surface => surface.IndexCount);

                        objects.Add(new Pm4OverlayObject(
                            sourcePath,
                            mshdGrouping.Field00,
                            mshdGrouping.RegionId,
                            mshdGrouping.Field08,
                            ck24,
                            ck24Type,
                            nextObjectPartId++,
                            dominantLinkGroupObjectId,
                            linkedPositionRefs.Count,
                            linkedPositionRefSummary,
                            lines,
                            triangles,
                            component.Count,
                            totalIndexCount,
                            dominantGroupKey,
                            dominantAttributeMask,
                            dominantMscnRefIndex,
                            averageSurfaceHeight,
                            linkedPlacementAnchor,
                            linkedRendererFrameRotationRadians,
                            linkedPlanarTransform,
                            linkedConnectorKeys));

                        // Collect MSLK.TypeFlags from surfaces in this component.
                        // Match MSLK.RefIndex against the MSUR entry at that index position.
                        uint typeFlagsMask = 0;
                        if (pm4.KnownChunks.Mslk.Count > 0)
                        {
                            foreach (MslkEntry mslk in pm4.KnownChunks.Mslk)
                            {
                                if (mslk.TypeFlags == 0)
                                    continue;
                                if ((uint)mslk.RefIndex < (uint)pm4.KnownChunks.Msur.Count &&
                                    component.Contains(pm4.KnownChunks.Msur[mslk.RefIndex]))
                                {
                                    typeFlagsMask |= 1u << mslk.TypeFlags;
                                }
                            }
                        }
                        if (typeFlagsMask != 0)
                            objects[^1].DistinctTypeFlags = typeFlagsMask;

                        tileLineBudget -= lines.Count;
                        tileTriangleBudget -= triangles.Count;
                    }
                }
            }
        }

        int linesUsed = objects.Sum(obj => obj.Lines.Count);
        int trianglesUsed = objects.Sum(obj => obj.Triangles.Count);
        remainingLineBudget -= linesUsed;
        remainingTriangleBudget -= trianglesUsed;
        diagnostics.DroppedLongEdgeLines = rejectedLongEdges;
        return objects;
    }

    /// <summary>
    /// The one frame PM4 geometry is drawn in. MSVT needs no fitting, so nothing is fitted.
    /// </summary>
    /// <remarks>
    /// MSVT is stored in ADT placement space — an origin-relative coordinate, like a raw MDDF
    /// position — so <c>WorldSpace</c> with the identity planar transform composes with
    /// <see cref="ConvertWorldToRenderer"/> into <c>(MapOrigin - X, MapOrigin - Y, Z)</c>. That is
    /// already, letter for letter, what <see cref="EnsurePm4MscnData"/> and
    /// <see cref="EnsurePm4MspvData"/> do to place MSCN and MSPV, and those land correctly. MSPV,
    /// MSVT and MSCN share one chunk frame, so the mesh has no business using a different one.
    /// </remarks>
    private static readonly CorePm4CoordinateModeResolution CanonicalCoordinateModeResolution =
        new(
            CorePm4CoordinateMode.WorldSpace,
            CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace),
            0f,
            0f,
            false);

    /// <summary>
    /// Returns the canonical frame instead of fitting one per object.
    /// </summary>
    /// <remarks>
    /// This used to call <c>CorePm4PlacementMath.ResolveCoordinateMode</c>, which scored candidate
    /// coordinate modes and planar transforms against MPRL. Measured over the whole development
    /// corpus, that fitter was wrong in both directions:
    ///
    /// - it selected <c>TileLocal</c> for 18 objects whose coordinates are absolute, then added tile
    ///   offsets to them. The human tents in <c>development_01_00.pm4</c> were one, thrown from tile
    ///   (0,1) to (1,-1) while all three of that tile's real ADT placements sit inside the canonical
    ///   footprint;
    /// - the yaw correction it produced rotated 974 of 1,895 objects by 15-45 degrees. Scored
    ///   against MODF world bounding boxes over the 127 objects whose box can actually see a
    ///   rotation, containment fell from 93.3% to 88.2%, against 79.0% for a deliberately wrong
    ///   45-degree control. It hurt 96 objects and helped 3.
    ///
    /// `pm4 bounds-audit --by-region` and `pm4 yaw-evidence` reproduce both numbers.
    /// <c>CorePm4PlacementMath</c> keeps the fitter for callers that still want to explore it; the
    /// render path simply no longer asks.
    /// </remarks>
    private static CorePm4CoordinateModeResolution ResolveCk24CoordinateModeResolution(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry> anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        bool fallbackTileLocalCoordinates)
    {
        return CanonicalCoordinateModeResolution;
    }

    private static List<List<Pm4IndexedSurface>> SplitSurfaceGroupByMslk(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        var groups = new List<List<Pm4IndexedSurface>>();
        if (surfaces.Count == 0)
            return groups;

        if (!TryPartitionSurfaceGroupByMslk(pm4, surfaces, out List<List<Pm4IndexedSurface>> linkedComponents, out List<Pm4IndexedSurface> unlinked))
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        if (linkedComponents.Count <= 1)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        foreach (List<Pm4IndexedSurface> component in linkedComponents.OrderBy(component => component.Min(entry => entry.SurfaceIndex)))
            groups.Add(component);

        if (unlinked.Count > 0)
            groups.Add(unlinked);

        return groups;
    }

    private static List<List<Pm4IndexedSurface>> SplitZeroCk24SeedGroup(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        if (!TryPartitionSurfaceGroupByMslk(pm4, surfaces, out List<List<Pm4IndexedSurface>> linkedComponents, out List<Pm4IndexedSurface> unlinked))
            return SplitIndexedSurfaceGroupByConnectivity(pm4, surfaces);

        if (linkedComponents.Count == 0)
            return SplitIndexedSurfaceGroupByConnectivity(pm4, surfaces);

        var groups = new List<List<Pm4IndexedSurface>>();
        foreach (List<Pm4IndexedSurface> component in linkedComponents.OrderBy(component => component.Min(entry => entry.SurfaceIndex)))
            groups.Add(component);

        if (unlinked.Count > 0)
            groups.AddRange(SplitIndexedSurfaceGroupByConnectivity(pm4, unlinked));

        return groups;
    }

    private static bool TryPartitionSurfaceGroupByMslk(
        Pm4File pm4,
        IReadOnlyList<Pm4IndexedSurface> surfaces,
        out List<List<Pm4IndexedSurface>> linkedComponents,
        out List<Pm4IndexedSurface> unlinked)
    {
        linkedComponents = new List<List<Pm4IndexedSurface>>();
        unlinked = new List<Pm4IndexedSurface>();

        IReadOnlyList<CorePm4MslkEntry> linkEntries = pm4.KnownChunks.Mslk;
        int surfaceCount = pm4.KnownChunks.Msur.Count;
        if (surfaces.Count <= 1 || linkEntries.Count == 0)
            return false;

        var surfaceIndexToLocal = new Dictionary<int, int>(surfaces.Count);
        for (int i = 0; i < surfaces.Count; i++)
            surfaceIndexToLocal[surfaces[i].SurfaceIndex] = i;

        var groupToMembers = new Dictionary<uint, HashSet<int>>();
        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if (link.GroupObjectId == 0)
                continue;

            if (link.RefIndex >= surfaceCount || !surfaceIndexToLocal.TryGetValue(link.RefIndex, out int localRefIndex))
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = new HashSet<int>();
                groupToMembers[link.GroupObjectId] = members;
            }

            members.Add(localRefIndex);
        }

        if (groupToMembers.Count == 0)
            return false;

        int[] parent = new int[surfaces.Count];
        for (int i = 0; i < parent.Length; i++)
            parent[i] = i;

        static int Find(int[] parentArray, int index)
        {
            while (parentArray[index] != index)
            {
                parentArray[index] = parentArray[parentArray[index]];
                index = parentArray[index];
            }

            return index;
        }

        static void Union(int[] parentArray, int a, int b)
        {
            int rootA = Find(parentArray, a);
            int rootB = Find(parentArray, b);
            if (rootA != rootB)
                parentArray[rootB] = rootA;
        }

        var linkedLocalIndices = new HashSet<int>();
        foreach (HashSet<int> members in groupToMembers.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linkedLocalIndices.Add(first);
            foreach (int member in members)
            {
                linkedLocalIndices.Add(member);
                Union(parent, first, member);
            }
        }

        if (linkedLocalIndices.Count < 2)
            return false;

        var linkedByRoot = new Dictionary<int, List<Pm4IndexedSurface>>();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!linkedLocalIndices.Contains(i))
            {
                unlinked.Add(surfaces[i]);
                continue;
            }

            int root = Find(parent, i);
            if (!linkedByRoot.TryGetValue(root, out List<Pm4IndexedSurface>? component))
            {
                component = new List<Pm4IndexedSurface>();
                linkedByRoot[root] = component;
            }

            component.Add(surfaces[i]);
        }

        if (linkedByRoot.Count == 0)
            return false;

        linkedComponents = linkedByRoot.Values.ToList();
        return true;
    }

    private static List<List<Pm4IndexedSurface>> SplitIndexedSurfaceGroupByConnectivity(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var components = new List<List<Pm4IndexedSurface>>();
        if (surfaces.Count == 0)
            return components;
        if (surfaces.Count == 1)
        {
            components.Add(new List<Pm4IndexedSurface> { surfaces[0] });
            return components;
        }

        var surfaceVertices = new List<List<int>>(surfaces.Count);
        var vertexToSurfaceIndices = new Dictionary<int, List<int>>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s].Surface;
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            var vertices = new List<int>();
            var unique = new HashSet<int>();

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int idx = firstIndex; idx < endExclusive; idx++)
                {
                    int vertexIndex = (int)meshIndices[idx];
                    if ((uint)vertexIndex >= (uint)meshVertices.Count)
                        continue;
                    if (!unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = new List<int>();
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }

                    owners.Add(s);
                }
            }

            surfaceVertices.Add(vertices);
        }

        var visited = new bool[surfaces.Count];
        var queue = new Queue<int>();
        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            var component = new List<Pm4IndexedSurface>();

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                List<int> vertices = surfaceVertices[current];
                for (int v = 0; v < vertices.Count; v++)
                {
                    int vertexIndex = vertices[v];
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    for (int n = 0; n < neighbors.Count; n++)
                    {
                        int neighborSurface = neighbors[n];
                        if (visited[neighborSurface])
                            continue;

                        visited[neighborSurface] = true;
                        queue.Enqueue(neighborSurface);
                    }
                }
            }

            components.Add(component);
        }

        return components;
    }

    private static uint SelectDominantMslkGroupObjectId(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        IReadOnlyList<CorePm4MslkEntry> linkEntries = pm4.KnownChunks.Mslk;
        if (surfaces.Count == 0 || linkEntries.Count == 0)
            return 0;

        int surfaceCount = pm4.KnownChunks.Msur.Count;
        var surfaceIndices = new HashSet<int>(surfaces.Select(static surface => surface.SurfaceIndex));
        var counts = new Dictionary<uint, int>();

        uint bestGroupObjectId = 0;
        int bestCount = 0;
        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if (link.GroupObjectId == 0)
                continue;

            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            int nextCount = 1;
            if (counts.TryGetValue(link.GroupObjectId, out int existingCount))
                nextCount = existingCount + 1;
            counts[link.GroupObjectId] = nextCount;

            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestGroupObjectId = link.GroupObjectId;
            }
        }

        return bestGroupObjectId;
    }

    private static List<MprlEntry> CollectLinkedPositionRefs(Pm4File pm4, IReadOnlyList<Pm4IndexedSurface> surfaces)
    {
        var refs = new List<MprlEntry>();
        IReadOnlyList<CorePm4MslkEntry> linkEntries = pm4.KnownChunks.Mslk;
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;
        if (surfaces.Count == 0 || linkEntries.Count == 0 || positionRefs.Count == 0)
            return refs;

        int surfaceCount = pm4.KnownChunks.Msur.Count;
        var surfaceIndices = new HashSet<int>(surfaces.Select(static surface => surface.SurfaceIndex));
        var seenRefIndices = new HashSet<int>();
        HashSet<uint> groupObjectIds = CollectMslkGroupObjectIds(linkEntries, surfaceIndices, surfaceCount);

        if (groupObjectIds.Count > 0)
        {
            for (int i = 0; i < linkEntries.Count; i++)
            {
                CorePm4MslkEntry link = linkEntries[i];
                if (link.GroupObjectId == 0 || !groupObjectIds.Contains(link.GroupObjectId))
                    continue;
                if ((uint)link.RefIndex >= (uint)positionRefs.Count)
                    continue;
                if (!seenRefIndices.Add(link.RefIndex))
                    continue;

                refs.Add(positionRefs[link.RefIndex]);
            }

            if (refs.Count > 0)
                return refs;
        }

        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if ((uint)link.RefIndex >= (uint)positionRefs.Count)
                continue;

            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            if (!seenRefIndices.Add(link.RefIndex))
                continue;

            refs.Add(positionRefs[link.RefIndex]);
        }

        return refs;
    }

    private static HashSet<uint> CollectMslkGroupObjectIds(
        IReadOnlyList<CorePm4MslkEntry> linkEntries,
        HashSet<int> surfaceIndices,
        int surfaceCount)
    {
        var groupObjectIds = new HashSet<uint>();
        for (int i = 0; i < linkEntries.Count; i++)
        {
            CorePm4MslkEntry link = linkEntries[i];
            if (link.GroupObjectId == 0)
                continue;
            if (!LinkReferencesSurface(link, surfaceIndices, surfaceCount))
                continue;

            groupObjectIds.Add(link.GroupObjectId);
        }

        return groupObjectIds;
    }

    private static bool LinkReferencesSurface(MslkEntry link, HashSet<int> surfaceIndices, int surfaceCount)
    {
        // The current shared PM4 reader exposes surface linkage through RefIndex.
        if (link.RefIndex < surfaceCount && surfaceIndices.Contains(link.RefIndex))
            return true;

        return false;
    }

    private static Pm4LinkedPositionRefSummary SummarizeLinkedPositionRefs(IReadOnlyList<MprlEntry> positionRefs)
    {
        return FromCorePm4LinkedPositionRefSummary(
            CorePm4PlacementMath.SummarizeLinkedPositionRefs(ConvertToCorePm4PositionRefs(positionRefs)));
    }

    private static bool TryComputeExpectedMprlYawRadians(IReadOnlyList<MprlEntry> positionRefs, out float yawRadians)
    {
        yawRadians = 0f;
        if (positionRefs.Count == 0)
            return false;

        double sumSin = 0d;
        double sumCos = 0d;
        int count = 0;
        for (int i = 0; i < positionRefs.Count; i++)
        {
            // Keep MPRL low-16 orientation as a raw packed angle until its world-yaw semantics
            // are proven. Basis/sign ambiguity is handled later by the comparison fallback path.
            float angleRadians = DecodeRawMprlPackedAngleRadians(positionRefs[i]);
            sumSin += Math.Sin(angleRadians);
            sumCos += Math.Cos(angleRadians);
            count++;
        }

        if (count == 0)
            return false;

        double length = Math.Sqrt(sumSin * sumSin + sumCos * sumCos);
        if (length < 1e-4)
            return false;

        yawRadians = (float)Math.Atan2(sumSin, sumCos);
        return true;
    }

    private static bool TryComputePlanarPrincipalYaw(
        IReadOnlyList<Vector3> objectVertices,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        out float yawRadians)
    {
        yawRadians = 0f;
        if (objectVertices.Count < 3)
            return false;

        int sampleCount = Math.Min(512, objectVertices.Count);
        int stride = Math.Max(1, objectVertices.Count / sampleCount);
        double meanX = 0d;
        double meanY = 0d;
        int used = 0;

        for (int i = 0; i < objectVertices.Count; i += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            meanX += world.X;
            meanY += world.Y;
            used++;
        }

        if (used < 3)
            return false;

        meanX /= used;
        meanY /= used;

        double covXX = 0d;
        double covYY = 0d;
        double covXY = 0d;
        for (int i = 0; i < objectVertices.Count; i += stride)
        {
            Vector3 world = ConvertPm4VertexToWorld(objectVertices[i], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            double dx = world.X - meanX;
            double dy = world.Y - meanY;
            covXX += dx * dx;
            covYY += dy * dy;
            covXY += dx * dy;
        }

        if (covXX + covYY < 1e-4)
            return false;

        yawRadians = 0.5f * (float)Math.Atan2(2.0 * covXY, covXX - covYY);
        return true;
    }

    private static float ComputeUndirectedAngleDelta(float a, float b)
    {
        float delta = MathF.Abs(a - b);
        while (delta > MathF.PI)
            delta -= 2f * MathF.PI;
        delta = MathF.Abs(delta);
        if (delta > MathF.PI * 0.5f)
            delta = MathF.PI - delta;

        return MathF.Abs(delta);
    }

    private static float NormalizeSignedRadians(float radians)
    {
        while (radians > MathF.PI)
            radians -= 2f * MathF.PI;
        while (radians < -MathF.PI)
            radians += 2f * MathF.PI;

        return radians;
    }

    private static float ComputeMprlYawDeltaWithQuarterTurnFallback(float candidateYaw, float expectedYaw)
    {
        float bestDelta = ComputeUndirectedAngleDelta(candidateYaw, expectedYaw);
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw));

        const float quarterTurn = MathF.PI * 0.5f;
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, expectedYaw + quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, expectedYaw - quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw + quarterTurn));
        bestDelta = MathF.Min(bestDelta, ComputeUndirectedAngleDelta(candidateYaw, -expectedYaw - quarterTurn));

        return bestDelta;
    }

    private static float ComputeBestSignedYawDeltaWithBasisFallback(float candidateYaw, float expectedYaw)
    {
        const float quarterTurn = MathF.PI * 0.5f;
        float[] expectedCandidates =
        {
            expectedYaw,
            -expectedYaw,
            expectedYaw + quarterTurn,
            expectedYaw - quarterTurn,
            -expectedYaw + quarterTurn,
            -expectedYaw - quarterTurn,
        };

        float bestDelta = 0f;
        float bestAbsDelta = float.MaxValue;
        for (int i = 0; i < expectedCandidates.Length; i++)
        {
            float target = expectedCandidates[i];
            for (int parity = 0; parity < 2; parity++)
            {
                float orientedTarget = target + (parity == 0 ? 0f : MathF.PI);
                float delta = NormalizeSignedRadians(orientedTarget - candidateYaw);
                float absDelta = MathF.Abs(delta);
                if (absDelta < bestAbsDelta)
                {
                    bestAbsDelta = absDelta;
                    bestDelta = delta;
                }
            }
        }

        return bestDelta;
    }

    private static Vector3 ComputeSurfaceRendererCentroid(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3 worldPivot,
        float worldYawCorrectionRadians)
    {
        List<Vector3> objectVertices = CollectSurfaceVertices(pm4, surfaces);
        if (objectVertices.Count == 0)
            return Vector3.Zero;

        Vector3 centroid = Vector3.Zero;
        for (int i = 0; i < objectVertices.Count; i++)
            centroid += objectVertices[i];
        centroid /= objectVertices.Count;

        return ConvertPm4VertexToRenderer(
            centroid,
            tileX,
            tileY,
            useTileLocalCoordinates,
            axisConvention,
            planarTransform,
            worldPivot,
            worldYawCorrectionRadians);
    }

    private static Vector3 ConvertMprlPositionToWorld(Vector3 refPos)
    {
        // Older PM4 R&D exported MSVT in a fixed viewer/world basis of (Y, X, Z).
        // The raw forensic mapping on the development dataset was:
        //   MPRL X -> raw MSVT Y
        //   MPRL Z -> raw MSVT X
        //   MPRL Y -> raw MSVT Z
        // Folding those together gives viewer/world coordinates of (X, Z, Y).
        return new Vector3(refPos.X, refPos.Z, refPos.Y);
    }

    private static List<Pm4LineSegment> BuildCk24ObjectLines(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3 worldPivot,
        float worldYawCorrectionRadians,
        int lineBudget,
        ref int rejectedLongEdges)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var lines = new List<Pm4LineSegment>();
        var uniqueEdges = new HashSet<ulong>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (lines.Count >= lineBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 2 || firstIndex < 0 || firstIndex >= meshIndices.Count)
                continue;

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, meshIndices.Count);
            if (endExclusive - firstIndex < 2)
                continue;

            int prevVertex = (int)meshIndices[firstIndex];
            if ((uint)prevVertex >= (uint)meshVertices.Count)
                continue;

            for (int idx = firstIndex + 1; idx < endExclusive && lines.Count < lineBudget; idx++)
            {
                int nextVertex = (int)meshIndices[idx];
                if ((uint)nextVertex >= (uint)meshVertices.Count)
                    continue;

                AddUniqueEdge(pm4, prevVertex, nextVertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
                prevVertex = nextVertex;
            }

            // Close each surface loop so CK24 objects stay visually self-contained.
            if (lines.Count < lineBudget)
            {
                int firstVertex = (int)meshIndices[firstIndex];
                int lastVertex = (int)meshIndices[endExclusive - 1];
                if ((uint)firstVertex < (uint)meshVertices.Count && (uint)lastVertex < (uint)meshVertices.Count)
                    AddUniqueEdge(pm4, lastVertex, firstVertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            }
        }

        return lines;
    }

    private static List<Pm4Triangle> BuildCk24ObjectTriangles(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3 worldPivot,
        float worldYawCorrectionRadians,
        int triangleBudget,
        out int rejectedOutOfRange)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var triangles = new List<Pm4Triangle>();
        rejectedOutOfRange = 0;

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (triangles.Count >= triangleBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 3 || firstIndex < 0 || firstIndex >= meshIndices.Count)
            {
                rejectedOutOfRange++;
                continue;
            }

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, meshIndices.Count);
            int indexCount = endExclusive - firstIndex;
            if (indexCount < 3)
            {
                rejectedOutOfRange++;
                continue;
            }

            // Most PM4 surfaces are listed as loops; use a fan from the first vertex.
            int i0 = (int)meshIndices[firstIndex];
            if ((uint)i0 >= (uint)meshVertices.Count)
            {
                rejectedOutOfRange++;
                continue;
            }

            Vector3 v0 = ConvertPm4VertexToRenderer(meshVertices[i0], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            for (int idx = firstIndex + 1; idx + 1 < endExclusive && triangles.Count < triangleBudget; idx++)
            {
                int i1 = (int)meshIndices[idx];
                int i2 = (int)meshIndices[idx + 1];
                if ((uint)i1 >= (uint)meshVertices.Count || (uint)i2 >= (uint)meshVertices.Count)
                    continue;

                Vector3 v1 = ConvertPm4VertexToRenderer(meshVertices[i1], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
                Vector3 v2 = ConvertPm4VertexToRenderer(meshVertices[i2], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
                triangles.Add(planarTransform.InvertsWinding
                    ? new Pm4Triangle(v0, v2, v1)
                    : new Pm4Triangle(v0, v1, v2));
            }
        }

        return triangles;
    }

    /// <summary>
    /// Builds the vertical wall faces that belong to a set of surfaces, from the MSLK path windows
    /// that reference them (MSLK.RefIndex -> MSUR) through MSPI into MSPV.
    /// </summary>
    /// <remarks>
    /// Each window is one planar polygon, emitted as a fan. Measured over the 616-file development
    /// corpus: 98.05% of windows hold exactly 4 indices, 1.84% hold 6, 99.6% are coplanar, and zero
    /// of 598,790 faces have Z as their dominant normal component. They are walls; MSUR is floors.
    /// This is the geometry the viewer has never drawn.
    /// </remarks>
    private static List<Pm4Triangle> BuildMslkWallTriangles(
        Pm4File pm4,
        IReadOnlyCollection<int> surfaceIndices,
        IReadOnlyDictionary<int, List<int>> mslkWindowsBySurface,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int triangleBudget,
        List<Pm4LineSegment> wallLines,
        int lineBudget,
        out int wallFaceCount)
    {
        var triangles = new List<Pm4Triangle>();
        wallFaceCount = 0;

        IReadOnlyList<Vector3> pathVertices = pm4.KnownChunks.Mspv;
        IReadOnlyList<uint> pathIndices = pm4.KnownChunks.Mspi;
        if (pathVertices.Count == 0 || pathIndices.Count == 0 || mslkWindowsBySurface.Count == 0)
            return triangles;

        var scratch = new List<Vector3>(8);

        foreach (int surfaceIndex in surfaceIndices)
        {
            if (!mslkWindowsBySurface.TryGetValue(surfaceIndex, out List<int>? linkIndices))
                continue;

            foreach (int linkIndex in linkIndices)
            {
                if (triangles.Count >= triangleBudget)
                    return triangles;

                MslkEntry link = pm4.KnownChunks.Mslk[linkIndex];
                int first = link.MspiFirstIndex;
                int count = link.MspiIndexCount;
                if (first < 0 || count < 3 || first + count > pathIndices.Count)
                    continue;

                scratch.Clear();
                for (int i = 0; i < count; i++)
                {
                    uint vertexIndex = pathIndices[first + i];
                    if (vertexIndex < (uint)pathVertices.Count)
                        scratch.Add(ConvertPm4VertexToRenderer(pathVertices[(int)vertexIndex], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform));
                }

                if (scratch.Count < 3)
                    continue;

                wallFaceCount++;
                for (int i = 1; i + 1 < scratch.Count && triangles.Count < triangleBudget; i++)
                {
                    triangles.Add(planarTransform.InvertsWinding
                        ? new Pm4Triangle(scratch[0], scratch[i + 1], scratch[i])
                        : new Pm4Triangle(scratch[0], scratch[i], scratch[i + 1]));
                }

                // The overlay draws lines unless "PM4 Solid Fill" is on, so a triangle-only wall
                // would be invisible in the default wireframe view. Emit the quad outline too.
                for (int i = 0; i < scratch.Count && wallLines.Count < lineBudget; i++)
                    wallLines.Add(new Pm4LineSegment(scratch[i], scratch[(i + 1) % scratch.Count]));
            }
        }

        return triangles;
    }

    /// <summary>
    /// Indexes MSLK path windows by the MSUR surface they reference, once per tile.
    /// </summary>
    /// <remarks>
    /// Entries with a negative MspiFirstIndex carry no path at all — 53% of the corpus — and are
    /// skipped here rather than being treated as empty geometry. Prior art reads them as doodad
    /// placements; that is a separate question from wall rendering.
    /// </remarks>
    private static Dictionary<int, List<int>> BuildMslkWindowsBySurface(Pm4File pm4)
    {
        var windowsBySurface = new Dictionary<int, List<int>>();
        int surfaceCount = pm4.KnownChunks.Msur.Count;
        IReadOnlyList<MslkEntry> links = pm4.KnownChunks.Mslk;

        for (int i = 0; i < links.Count; i++)
        {
            MslkEntry link = links[i];
            if (link.MspiFirstIndex < 0 || link.MspiIndexCount < 3 || link.RefIndex >= surfaceCount)
                continue;

            if (!windowsBySurface.TryGetValue(link.RefIndex, out List<int>? linkIndices))
            {
                linkIndices = [];
                windowsBySurface[link.RefIndex] = linkIndices;
            }

            linkIndices.Add(i);
        }

        return windowsBySurface;
    }

    private static List<Pm4LineSegment> BuildFallbackMeshLines(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int lineBudget,
        ref int rejectedLongEdges)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var lines = new List<Pm4LineSegment>();
        var uniqueEdges = new HashSet<ulong>();

        for (int i = 0; i + 2 < meshIndices.Count && lines.Count < lineBudget; i += 3)
        {
            int i0 = (int)meshIndices[i];
            int i1 = (int)meshIndices[i + 1];
            int i2 = (int)meshIndices[i + 2];

            if ((uint)i0 >= (uint)meshVertices.Count ||
                (uint)i1 >= (uint)meshVertices.Count ||
                (uint)i2 >= (uint)meshVertices.Count)
                continue;

            AddUniqueEdge(pm4, i0, i1, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            AddUniqueEdge(pm4, i1, i2, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
            AddUniqueEdge(pm4, i2, i0, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, uniqueEdges, lines, lineBudget, ref rejectedLongEdges);
        }

        return lines;
    }

    private static List<List<MsurEntry>> SplitSurfaceGroupByConnectivity(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var components = new List<List<MsurEntry>>();
        if (surfaces.Count == 0)
            return components;
        if (surfaces.Count == 1)
        {
            components.Add(new List<MsurEntry> { surfaces[0] });
            return components;
        }

        var surfaceVertices = new List<List<int>>(surfaces.Count);
        var vertexToSurfaceIndices = new Dictionary<int, List<int>>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            var vertices = new List<int>();
            var unique = new HashSet<int>();

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int idx = firstIndex; idx < endExclusive; idx++)
                {
                    int vertexIndex = (int)meshIndices[idx];
                    if ((uint)vertexIndex >= (uint)meshVertices.Count)
                        continue;
                    if (!unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = new List<int>();
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }
                    owners.Add(s);
                }
            }

            surfaceVertices.Add(vertices);
        }

        var visited = new bool[surfaces.Count];
        var queue = new Queue<int>();
        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            var component = new List<MsurEntry>();

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                List<int> vertices = surfaceVertices[current];
                for (int v = 0; v < vertices.Count; v++)
                {
                    int vertexIndex = vertices[v];
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    for (int n = 0; n < neighbors.Count; n++)
                    {
                        int neighborSurface = neighbors[n];
                        if (visited[neighborSurface])
                            continue;

                        visited[neighborSurface] = true;
                        queue.Enqueue(neighborSurface);
                    }
                }
            }

            components.Add(component);
        }

        return components;
    }

    private static List<List<MsurEntry>> SplitSurfaceGroupByMscnRef(IReadOnlyList<MsurEntry> surfaces)
    {
        if (surfaces.Count <= 1)
            return new List<List<MsurEntry>> { surfaces.ToList() };

        var groups = surfaces
            .GroupBy(static surface => surface.MscnRefIndex)
            .Select(static group => group.ToList())
            .Where(static group => group.Count > 0)
            .ToList();

        return groups.Count > 0 ? groups : new List<List<MsurEntry>> { surfaces.ToList() };
    }

    private static IReadOnlyList<Pm4ConnectorKey> BuildCk24ConnectorKeys(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        CorePm4PlacementSolution placement)
    {
        if (surfaces.Count == 0 || pm4.KnownChunks.Mscn.Count == 0)
            return Array.Empty<Pm4ConnectorKey>();

        return CorePm4PlacementMath.BuildConnectorKeys(
                pm4.KnownChunks.Mscn,
                ConvertToCorePm4Surfaces(surfaces),
                placement)
            .Select(FromCorePm4ConnectorKey)
            .ToList();
    }

    private void RebuildPm4MergedObjectGroups()
    {
        _pm4MergedObjectGroupKeys.Clear();

        var groups = new List<CorePm4ConnectorMergeCandidate>();
        foreach (var tileEntry in _pm4TileObjects)
        {
            foreach (IGrouping<(int tileX, int tileY, uint ck24), Pm4OverlayObject> objectGroup in tileEntry.Value.GroupBy(obj => BuildPm4BaseObjectGroupKey((tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId))))
            {
                var baseGroupKey = objectGroup.Key;
                Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
                Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);
                bool hasBounds = false;
                var connectorKeys = new HashSet<CorePm4ConnectorKey>();

                foreach (Pm4OverlayObject obj in objectGroup)
                {
                    IncludePointInBounds(obj.BoundsMin, ref boundsMin, ref boundsMax, ref hasBounds);
                    IncludePointInBounds(obj.BoundsMax, ref boundsMin, ref boundsMax, ref hasBounds);

                    for (int i = 0; i < obj.ConnectorKeys.Count; i++)
                        connectorKeys.Add(ToCorePm4ConnectorKey(obj.ConnectorKeys[i]));
                }

                if (!hasBounds)
                {
                    boundsMin = Vector3.Zero;
                    boundsMax = Vector3.Zero;
                }

                Vector3 center = (boundsMin + boundsMax) * 0.5f;
                groups.Add(new CorePm4ConnectorMergeCandidate(
                    new CorePm4ObjectGroupKey(baseGroupKey.tileX, baseGroupKey.tileY, baseGroupKey.ck24),
                    boundsMin,
                    boundsMax,
                    center,
                    connectorKeys));

                _pm4MergedObjectGroupKeys[baseGroupKey] = baseGroupKey;
            }
        }

        IReadOnlyDictionary<CorePm4ObjectGroupKey, CorePm4ObjectGroupKey> mergedGroupMap = CorePm4PlacementMath.BuildMergedGroupMap(groups);
        foreach ((CorePm4ObjectGroupKey sourceKey, CorePm4ObjectGroupKey mergedKey) in mergedGroupMap)
            _pm4MergedObjectGroupKeys[(sourceKey.TileX, sourceKey.TileY, sourceKey.Ck24)] = (mergedKey.TileX, mergedKey.TileY, mergedKey.Ck24);
    }

    private static CorePm4ConnectorKey ToCorePm4ConnectorKey(Pm4ConnectorKey key) => new(key.X, key.Y, key.Z);

    private static Pm4ConnectorKey FromCorePm4ConnectorKey(CorePm4ConnectorKey key) => new(key.X, key.Y, key.Z);

    private static void IncludePointInBounds(Vector3 point, ref Vector3 boundsMin, ref Vector3 boundsMax, ref bool hasBounds)
    {
        if (!hasBounds)
        {
            boundsMin = point;
            boundsMax = point;
            hasBounds = true;
            return;
        }

        boundsMin = Vector3.Min(boundsMin, point);
        boundsMax = Vector3.Max(boundsMax, point);
    }

    private static byte SelectDominantSurfaceValue(IReadOnlyList<MsurEntry> surfaces, Func<MsurEntry, byte> selector)
    {
        if (surfaces.Count == 0)
            return 0;

        Span<int> counts = stackalloc int[256];
        for (int i = 0; i < surfaces.Count; i++)
            counts[selector(surfaces[i])]++;

        int bestCount = -1;
        byte bestValue = 0;
        for (int i = 0; i < counts.Length; i++)
        {
            int count = counts[i];
            if (count <= bestCount)
                continue;

            bestCount = count;
            bestValue = (byte)i;
        }

        return bestValue;
    }

    private static uint SelectDominantSurfaceValue(IReadOnlyList<MsurEntry> surfaces, Func<MsurEntry, uint> selector)
    {
        if (surfaces.Count == 0)
            return 0;

        var counts = new Dictionary<uint, int>();
        uint bestValue = 0;
        int bestCount = -1;

        for (int i = 0; i < surfaces.Count; i++)
        {
            uint value = selector(surfaces[i]);
            int count = 1;
            if (counts.TryGetValue(value, out int existing))
                count = existing + 1;
            counts[value] = count;

            if (count > bestCount)
            {
                bestCount = count;
                bestValue = value;
            }
        }

        return bestValue;
    }

    private static List<Vector3> BuildPm4PositionRefMarkers(Pm4File pm4, int limit)
    {
        var markers = new List<Vector3>();
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;
        int count = Math.Min(limit, positionRefs.Count);
        for (int i = 0; i < count; i++)
        {
            Vector3 world = ConvertMprlPositionToWorld(positionRefs[i].Position);
            markers.Add(new Vector3(
                WoWConstants.MapOrigin - world.Y,
                WoWConstants.MapOrigin - world.X,
                world.Z + 0.5f));
        }

        return markers;
    }

    private static List<Pm4Triangle> BuildFallbackMeshTriangles(
        Pm4File pm4,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        int triangleBudget)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var triangles = new List<Pm4Triangle>();

        for (int i = 0; i + 2 < meshIndices.Count && triangles.Count < triangleBudget; i += 3)
        {
            int i0 = (int)meshIndices[i];
            int i1 = (int)meshIndices[i + 1];
            int i2 = (int)meshIndices[i + 2];

            if ((uint)i0 >= (uint)meshVertices.Count ||
                (uint)i1 >= (uint)meshVertices.Count ||
                (uint)i2 >= (uint)meshVertices.Count)
                continue;

            Vector3 v0 = ConvertPm4VertexToRenderer(meshVertices[i0], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            Vector3 v1 = ConvertPm4VertexToRenderer(meshVertices[i1], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            Vector3 v2 = ConvertPm4VertexToRenderer(meshVertices[i2], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
            triangles.Add(planarTransform.InvertsWinding
                ? new Pm4Triangle(v0, v2, v1)
                : new Pm4Triangle(v0, v1, v2));
        }

        return triangles;
    }

    private static void AddUniqueEdge(Pm4File pm4, int ia, int ib,
        int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform,
        HashSet<ulong> uniqueEdges, List<Pm4LineSegment> lines, int tileLineBudget,
        ref int rejectedLongEdges,
        Vector3? worldPivot = null,
        float worldYawCorrectionRadians = 0f)
    {
        if (ia == ib || lines.Count >= tileLineBudget)
            return;

        ulong key = PackEdgeKey(ia, ib);
        if (!uniqueEdges.Add(key))
            return;

        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        Vector3 from = ConvertPm4VertexToRenderer(meshVertices[ia], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, worldPivot, worldYawCorrectionRadians);
        Vector3 to = ConvertPm4VertexToRenderer(meshVertices[ib], tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform, worldPivot, worldYawCorrectionRadians);

        if (Vector3.DistanceSquared(from, to) > Pm4MaxEdgeLength * Pm4MaxEdgeLength)
        {
            rejectedLongEdges++;
            return;
        }

        lines.Add(new Pm4LineSegment(from, to));
    }

    private static ulong PackEdgeKey(int ia, int ib)
    {
        uint lo = ia < ib ? (uint)ia : (uint)ib;
        uint hi = ia < ib ? (uint)ib : (uint)ia;
        return ((ulong)lo << 32) | hi;
    }

    private enum Pm4AxisConvention
    {
        XZPlaneYUp,
        XYPlaneZUp,
        YZPlaneXUp
    }

    private static Pm4AxisConvention DetectPm4AxisConvention(Pm4File pm4)
    {
        // Pick the basis that yields the most horizontal (floor-like) triangles.
        // This avoids forcing users to manually undo a 90-degree wall orientation.
        var candidates = new[]
        {
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        };

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionByTriangleNormals(pm4, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        return DetectAxisConventionByRanges(pm4.KnownChunks.Msvt);
    }

    private static Pm4AxisConvention DetectPm4AxisConvention(Pm4File pm4, IEnumerable<MsurEntry> surfaces)
    {
        var surfaceList = surfaces as List<MsurEntry> ?? surfaces.ToList();
        if (surfaceList.Count == 0)
            return DetectPm4AxisConvention(pm4);

        var candidates = new[]
        {
            Pm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.XYPlaneZUp,
            Pm4AxisConvention.YZPlaneXUp
        };

        Pm4AxisConvention bestConvention = Pm4AxisConvention.XYPlaneZUp;
        float bestScore = float.MinValue;
        foreach (Pm4AxisConvention candidate in candidates)
        {
            float score = ScoreAxisConventionBySurfaceNormals(pm4, surfaceList, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestConvention = candidate;
            }
        }

        if (bestScore > 0f)
            return bestConvention;

        List<Vector3> groupVertices = CollectSurfaceVertices(pm4, surfaceList);
        return groupVertices.Count > 0
            ? DetectAxisConventionByRanges(groupVertices)
            : DetectPm4AxisConvention(pm4);
    }

    private static CorePm4CoordinateMode ToCoreCoordinateMode(bool useTileLocalCoordinates)
    {
        return useTileLocalCoordinates
            ? CorePm4CoordinateMode.TileLocal
            : CorePm4CoordinateMode.WorldSpace;
    }

    private static CorePm4AxisConvention ToCoreAxisConvention(Pm4AxisConvention convention)
    {
        return convention switch
        {
            Pm4AxisConvention.XZPlaneYUp => CorePm4AxisConvention.XZPlaneYUp,
            Pm4AxisConvention.YZPlaneXUp => CorePm4AxisConvention.YZPlaneXUp,
            _ => CorePm4AxisConvention.XYPlaneZUp
        };
    }

    private static List<CorePm4MsurEntry> ConvertToCorePm4Surfaces(IReadOnlyList<MsurEntry> surfaces)
    {
        return surfaces as List<CorePm4MsurEntry> ?? surfaces.ToList();
    }

    private static List<CorePm4MprlEntry> ConvertToCorePm4PositionRefs(IReadOnlyList<MprlEntry> positionRefs)
    {
        return positionRefs as List<CorePm4MprlEntry> ?? positionRefs.ToList();
    }

    private static Pm4LinkedPositionRefSummary FromCorePm4LinkedPositionRefSummary(CorePm4LinkedPositionRefSummary summary)
    {
        return new Pm4LinkedPositionRefSummary(
            summary.TotalCount,
            summary.NormalCount,
            summary.TerminatorCount,
            summary.FloorMin,
            summary.FloorMax,
            summary.HeadingMinDegrees,
            summary.HeadingMaxDegrees,
            summary.HeadingMeanDegrees);
    }

    /// <summary>
    /// Builds the placement for a surface group in the canonical frame, with no yaw correction.
    /// </summary>
    /// <remarks>
    /// The pivot is still the group's real world centroid, because selection and connector merging
    /// use it. Only the fitted rotation is dropped — see
    /// <see cref="ResolveCk24CoordinateModeResolution"/> for the evidence that it was wrong.
    /// </remarks>
    private static CorePm4PlacementSolution ResolvePlacementSolution(
        Pm4File pm4,
        IEnumerable<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry>? anchorPositionRefs,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention)
    {
        var surfaceList = surfaces as List<MsurEntry> ?? surfaces.ToList();
        CorePm4CoordinateMode coordinateMode = CanonicalCoordinateModeResolution.CoordinateMode;
        Pm4PlanarTransform planarTransform = CanonicalCoordinateModeResolution.PlanarTransform;

        Vector3 worldPivot = CorePm4PlacementMath.ComputeSurfaceWorldCentroid(
            pm4.KnownChunks.Msvt,
            pm4.KnownChunks.Msvi,
            ConvertToCorePm4Surfaces(surfaceList),
            tileX,
            tileY,
            coordinateMode,
            ToCoreAxisConvention(axisConvention),
            planarTransform);

        return new CorePm4PlacementSolution(
            tileX,
            tileY,
            coordinateMode,
            ToCoreAxisConvention(axisConvention),
            planarTransform,
            worldPivot,
            WorldYawCorrectionRadians: 0f);
    }

    private static float NearestPositionRefDistanceSquared(IReadOnlyList<MprlEntry> positionRefs, Vector3 world)
    {
        float best = float.MaxValue;
        for (int i = 0; i < positionRefs.Count; i++)
        {
            Vector3 refWorld = ConvertMprlPositionToWorld(positionRefs[i].Position);
            float dx = refWorld.X - world.X;
            float dy = refWorld.Y - world.Y;
            float distSq = dx * dx + dy * dy;
            if (distSq < best)
                best = distSq;
        }

        return best;
    }

    private static float ScoreAxisConventionByTriangleNormals(Pm4File pm4, Pm4AxisConvention convention)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        if (meshVertices.Count == 0 || meshIndices.Count < 3)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        for (int i = 0; i + 2 < meshIndices.Count && samples < maxSamples; i += 3)
        {
            int i0 = (int)meshIndices[i];
            int i1 = (int)meshIndices[i + 1];
            int i2 = (int)meshIndices[i + 2];
            if ((uint)i0 >= (uint)meshVertices.Count ||
                (uint)i1 >= (uint)meshVertices.Count ||
                (uint)i2 >= (uint)meshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(meshVertices[i0], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
            Vector3 b = ConvertPm4VertexToWorld(meshVertices[i1], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
            Vector3 c = ConvertPm4VertexToWorld(meshVertices[i2], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));

            Vector3 normal = Vector3.Cross(b - a, c - a);
            float length = normal.Length();
            if (length < 1e-5f)
                continue;

            // Higher |normal.Z| means more floor-like orientation in this renderer.
            sum += MathF.Abs(normal.Z / length);
            samples++;
        }

        return samples > 0 ? sum / samples : 0f;
    }

    private static float ScoreAxisConventionBySurfaceNormals(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces, Pm4AxisConvention convention)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        if (meshVertices.Count == 0 || meshIndices.Count < 3 || surfaces.Count == 0)
            return 0f;

        float sum = 0f;
        int samples = 0;
        const int maxSamples = 1024;

        for (int s = 0; s < surfaces.Count && samples < maxSamples; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            if (surface.IndexCount < 3 || firstIndex < 0 || endExclusive - firstIndex < 3)
                continue;

            int i0 = (int)meshIndices[firstIndex];
            if ((uint)i0 >= (uint)meshVertices.Count)
                continue;

            Vector3 a = ConvertPm4VertexToWorld(meshVertices[i0], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
            for (int idx = firstIndex + 1; idx + 1 < endExclusive && samples < maxSamples; idx++)
            {
                int i1 = (int)meshIndices[idx];
                int i2 = (int)meshIndices[idx + 1];
                if ((uint)i1 >= (uint)meshVertices.Count || (uint)i2 >= (uint)meshVertices.Count)
                    continue;

                Vector3 b = ConvertPm4VertexToWorld(meshVertices[i1], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));
                Vector3 c = ConvertPm4VertexToWorld(meshVertices[i2], 0, 0, false, convention, CorePm4PlacementContract.GetDefaultPlanarTransform(CorePm4CoordinateMode.WorldSpace));

                Vector3 normal = Vector3.Cross(b - a, c - a);
                float length = normal.Length();
                if (length < 1e-5f)
                    continue;

                sum += MathF.Abs(normal.Z / length);
                samples++;
            }
        }

        return samples > 0 ? sum / samples : 0f;
    }

    private static List<Vector3> CollectSurfaceVertices(Pm4File pm4, IReadOnlyList<MsurEntry> surfaces)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var vertices = new List<Vector3>();
        var seen = new HashSet<int>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            if (surface.IndexCount <= 0 || firstIndex < 0 || endExclusive <= firstIndex)
                continue;

            for (int idx = firstIndex; idx < endExclusive; idx++)
            {
                int vertexIndex = (int)meshIndices[idx];
                if ((uint)vertexIndex >= (uint)meshVertices.Count)
                    continue;
                if (!seen.Add(vertexIndex))
                    continue;

                vertices.Add(meshVertices[vertexIndex]);
            }
        }

        return vertices;
    }

    private static Pm4AxisConvention DetectAxisConventionByRanges(IReadOnlyList<Vector3> vertices)
    {
        if (vertices.Count == 0)
            return Pm4AxisConvention.XYPlaneZUp;

        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            if (v.X < minX) minX = v.X;
            if (v.Y < minY) minY = v.Y;
            if (v.Z < minZ) minZ = v.Z;
            if (v.X > maxX) maxX = v.X;
            if (v.Y > maxY) maxY = v.Y;
            if (v.Z > maxZ) maxZ = v.Z;
        }

        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        const float tieTolerance = 8f;

        if (rangeY + tieTolerance < rangeX && rangeY + tieTolerance < rangeZ)
            return Pm4AxisConvention.XZPlaneYUp;
        if (rangeZ + tieTolerance < rangeX && rangeZ + tieTolerance < rangeY)
            return Pm4AxisConvention.XYPlaneZUp;
        if (rangeX + tieTolerance < rangeY && rangeX + tieTolerance < rangeZ)
            return Pm4AxisConvention.YZPlaneXUp;

        // Ambiguous ranges: default to WoW-style XY plane with Z up.
        return Pm4AxisConvention.XYPlaneZUp;
    }

    private static bool IsLikelyTileLocal(IReadOnlyList<Vector3> vertices)
    {
        float minX = float.MaxValue;
        float minY = float.MaxValue;
        float minZ = float.MaxValue;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float maxZ = float.MinValue;

        for (int i = 0; i < vertices.Count; i++)
        {
            Vector3 v = vertices[i];
            if (v.X < minX) minX = v.X;
            if (v.Y < minY) minY = v.Y;
            if (v.Z < minZ) minZ = v.Z;
            if (v.X > maxX) maxX = v.X;
            if (v.Y > maxY) maxY = v.Y;
            if (v.Z > maxZ) maxZ = v.Z;
        }

        const float tolerance = 64f;
        float tileSpan = Pm4CoordinateService.TileSize;

        bool xyLocal = minX >= -tolerance && minY >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxY <= tileSpan + tolerance;
        bool xzLocal = minX >= -tolerance && minZ >= -tolerance &&
                       maxX <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;
        bool yzLocal = minY >= -tolerance && minZ >= -tolerance &&
                       maxY <= tileSpan + tolerance && maxZ <= tileSpan + tolerance;

        return xyLocal || xzLocal || yzLocal;
    }

    private static Vector3 ConvertPm4VertexToWorld(Vector3 pm4Vertex, int tileX, int tileY, bool useTileLocalCoordinates, Pm4AxisConvention axisConvention, Pm4PlanarTransform planarTransform)
    {
        float localU;
        float localV;
        float localUp;

        switch (axisConvention)
        {
            case Pm4AxisConvention.XZPlaneYUp:
                localU = pm4Vertex.X;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.Y;
                break;
            case Pm4AxisConvention.YZPlaneXUp:
                localU = pm4Vertex.Y;
                localV = pm4Vertex.Z;
                localUp = pm4Vertex.X;
                break;
            case Pm4AxisConvention.XYPlaneZUp:
            default:
                // The older PM4 R&D exporter that matched placed WMO/M2 assets used
                // a fixed MSVT planar order of (Y, X, Z), not raw (X, Y, Z).
                // Keep Z-up, but preserve that planar basis here so the viewer stops
                // trying to approximate it with per-object swap/invert heuristics.
                localU = pm4Vertex.Y;
                localV = pm4Vertex.X;
                localUp = pm4Vertex.Z;
                break;
        }

        if (planarTransform.SwapPlanarAxes)
            (localU, localV) = (localV, localU);

        float tileSpan = Pm4CoordinateService.TileSize;
        float worldX;
        float worldY;

        if (useTileLocalCoordinates)
        {
            float mappedU = planarTransform.InvertU ? tileSpan - localU : localU;
            float mappedV = planarTransform.InvertV ? tileSpan - localV : localV;

            // Viewer world uses the standard WoW tile convention where file tile X advances along
            // world Y and file tile Y advances along world X. Keeping these unswapped only happens
            // to look correct on origin tiles and shifts non-origin tile-local PM4 onto the wrong grid.
            worldX = tileY * tileSpan + mappedU;
            worldY = tileX * tileSpan + mappedV;
        }
        else
        {
            if (planarTransform.InvertU)
                localU = -localU;
            if (planarTransform.InvertV)
                localV = -localV;

            worldX = localU;
            worldY = localV;
        }

        return new Vector3(worldX, worldY, localUp);
    }

    private static Vector3 RotateWorldAroundPivot(Vector3 world, Vector3 pivot, float yawRadians)
    {
        if (MathF.Abs(yawRadians) < 1e-6f)
            return world;

        float sin = MathF.Sin(yawRadians);
        float cos = MathF.Cos(yawRadians);
        float dx = world.X - pivot.X;
        float dy = world.Y - pivot.Y;

        float rx = dx * cos - dy * sin;
        float ry = dx * sin + dy * cos;
        return new Vector3(pivot.X + rx, pivot.Y + ry, world.Z);
    }

    private static Vector3 ConvertWorldToRenderer(Vector3 world)
    {
        return new Vector3(
            WoWConstants.MapOrigin - world.Y,
            WoWConstants.MapOrigin - world.X,
            world.Z + 0.5f);
    }

    private static float ConvertWorldYawCorrectionToRendererRotationRadians(float worldYawCorrectionRadians)
    {
        return -worldYawCorrectionRadians;
    }

    private static Vector3 ConvertPm4VertexToRenderer(
        Vector3 pm4Vertex,
        int tileX,
        int tileY,
        bool useTileLocalCoordinates,
        Pm4AxisConvention axisConvention,
        Pm4PlanarTransform planarTransform,
        Vector3? worldPivot = null,
        float worldYawCorrectionRadians = 0f)
    {
        Vector3 world = ConvertPm4VertexToWorld(pm4Vertex, tileX, tileY, useTileLocalCoordinates, axisConvention, planarTransform);
        if (worldPivot.HasValue && MathF.Abs(worldYawCorrectionRadians) > 1e-6f)
            world = RotateWorldAroundPivot(world, worldPivot.Value, worldYawCorrectionRadians);

        // Canonical world->renderer transform used across terrain/object pipelines.
        // rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ
        return ConvertWorldToRenderer(world);
    }

    public void ReloadPm4Overlay()
    {
        ClearPm4OverlayRuntimeState();
        _pm4PerFileInMemoryCache.Clear();
        _pm4PerFileDiskCache?.ClearForMap();
        _pm4PerFileDiskCache = null;
        _pm4LoadAttempted = false;
        BeginPm4OverlayLoad(ignoreCache: true);
    }

    /// <summary>
    /// Reload WL loose liquid bodies (WLW/WLQ/WLM) and rebuild GPU meshes.
    /// Useful when tweaking WL transform settings in the UI.
    /// </summary>
    public void ReloadWlLiquids()
    {
        _terrainManager.LiquidRenderer.ClearWlBodies();
        _wlLoader = null;
        _wlLoadAttempted = false;
        LazyLoadWlLiquids();
    }

    private void LazyLoadPoi()
    {
        _poiLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null) return;
        _poiLoader = new AreaPoiLoader();
        _poiLoader.Load(_dbcProvider, _dbdDir, _dbcBuild, _terrainManager.MapName);
    }

    private void LazyLoadTaxi()
    {
        _taxiLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _taxiLoader = new TaxiPathLoader();
        var dbcd = new DBCD.DBCD(_dbcProvider, new DBCD.Providers.FilesystemDBDProvider(_dbdDir));
        _taxiLoader.Load(dbcd, _dbcBuild, _mapId);
        _taxiActorTravelByPath.Clear();
        _taxiActorPoseByPath.Clear();
        _taxiActorSmoothedForwardByPath.Clear();
        _taxiActorClockInitialized = false;
    }

    private void UpdateTaxiActorInstances()
    {
        _taxiActorInstances.Clear();

        bool hasTaxiSelection = _selectedTaxiNodeId >= 0 || _selectedTaxiRouteId >= 0;
        if (!_showTaxi || !_showTaxiActors || _taxiLoader == null || !hasTaxiSelection)
        {
            _taxiActorPoseByPath.Clear();
            _taxiActorSmoothedForwardByPath.Clear();
            _taxiActorClockInitialized = false;
            return;
        }

        long now = Stopwatch.GetTimestamp();
        float deltaSeconds = 0f;
        if (_taxiActorClockInitialized)
            deltaSeconds = (float)((now - _lastTaxiActorTick) / (double)Stopwatch.Frequency);
        _lastTaxiActorTick = now;
        _taxiActorClockInitialized = true;

        float distanceStep = TaxiActorBaseUnitsPerSecond * _taxiActorSpeedMultiplier * Math.Max(0f, deltaSeconds);
        var activePathIds = new HashSet<int>();

        foreach (var route in _taxiLoader.Routes)
        {
            if (!IsTaxiRouteVisible(route) || route.Waypoints.Count < 2)
                continue;

            string? actorModelPath = GetResolvedTaxiActorModelPath(route.PathId);
            if (string.IsNullOrWhiteSpace(actorModelPath))
                continue;

            TaxiPathLoader.TaxiNode? mountNode = ResolveTaxiActorNode(route);
            float scale = mountNode?.MountScale > 0.01f ? mountNode.MountScale : 1.0f;

            scale *= _taxiActorScaleMultiplier;

            float routeLength = GetRouteLength(route.Waypoints);
            if (routeLength <= 1f)
                continue;

            activePathIds.Add(route.PathId);

            float travel = _taxiActorTravelByPath.TryGetValue(route.PathId, out float existingTravel)
                ? existingTravel
                : 0f;
            if (distanceStep > 0f)
                travel = (travel + distanceStep) % routeLength;
            _taxiActorTravelByPath[route.PathId] = travel;

            SampleRoute(route.Waypoints, travel, out Vector3 actorPosition, out Vector3 actorDirection);
            actorPosition.Z += TaxiActorHoverOffset;

            Vector3 sampledForward = SampleSmoothedTaxiRouteDirection(route.Waypoints, travel, routeLength);
            Vector3 actorForward = sampledForward;
            if (_taxiActorSmoothedForwardByPath.TryGetValue(route.PathId, out Vector3 previousForward)
                && previousForward.LengthSquared() > 0.0001f)
            {
                float blend = 1f - MathF.Exp(-TaxiActorHeadingSmoothingHz * Math.Max(0f, deltaSeconds));
                if (blend <= 0f)
                {
                    actorForward = previousForward;
                }
                else if (blend < 0.999f)
                {
                    Vector3 blendedForward = Vector3.Lerp(previousForward, sampledForward, blend);
                    actorForward = blendedForward.LengthSquared() > 0.0001f
                        ? Vector3.Normalize(blendedForward)
                        : sampledForward;
                }
            }

            if (actorForward.LengthSquared() <= 0.0001f)
            {
                actorForward = actorDirection.LengthSquared() > 0.0001f
                    ? Vector3.Normalize(actorDirection)
                    : Vector3.UnitX;
            }

            _taxiActorSmoothedForwardByPath[route.PathId] = actorForward;

            float yawRadians = ComputeTaxiActorYawRadians(actorForward);
            string modelPath = actorModelPath.Replace('/', '\\');
            string key = WorldAssetManager.NormalizeKey(modelPath);
            _assets.QueueMdxLoad(key);

            _taxiActorPoseByPath[route.PathId] = new TaxiActorPose(
                route.PathId,
                actorPosition,
                actorForward,
                yawRadians,
                scale,
                modelPath);

            var transform = Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationZ(yawRadians)
                * Matrix4x4.CreateTranslation(actorPosition);

            Vector3 boundsMin;
            Vector3 boundsMax;
            Vector3 localMin = Vector3.Zero;
            Vector3 localMax = Vector3.Zero;
            bool boundsResolved = false;
            if (_assets.TryGetMdxBounds(key, out Vector3 modelMin, out Vector3 modelMax))
            {
                localMin = modelMin;
                localMax = modelMax;
                boundsResolved = true;
                TransformBounds(modelMin, modelMax, transform, out boundsMin, out boundsMax);
            }
            else
            {
                boundsMin = actorPosition - new Vector3(2f);
                boundsMax = actorPosition + new Vector3(2f);
            }

            _taxiActorInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = boundsMin,
                BoundsMax = boundsMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = boundsResolved,
                ModelName = Path.GetFileName(modelPath),
                ModelPath = modelPath,
                PlacementPosition = actorPosition,
                PlacementRotation = new Vector3(0f, 0f, yawRadians * (180f / MathF.PI)),
                PlacementScale = scale,
                UniqueId = -route.PathId
            });
        }

        foreach (int stalePathId in _taxiActorTravelByPath.Keys.Except(activePathIds).ToList())
            _taxiActorTravelByPath.Remove(stalePathId);

        foreach (int stalePathId in _taxiActorPoseByPath.Keys.Except(activePathIds).ToList())
            _taxiActorPoseByPath.Remove(stalePathId);

        foreach (int stalePathId in _taxiActorSmoothedForwardByPath.Keys.Except(activePathIds).ToList())
            _taxiActorSmoothedForwardByPath.Remove(stalePathId);
    }

    private TaxiPathLoader.TaxiNode? ResolveTaxiActorNode(TaxiPathLoader.TaxiRoute route)
    {
        if (_taxiLoader == null)
            return null;

        if (_selectedTaxiNodeId >= 0)
        {
            var selectedNode = GetTaxiNode(_selectedTaxiNodeId);
            if (selectedNode != null && (route.FromNodeId == selectedNode.Id || route.ToNodeId == selectedNode.Id))
                return selectedNode;
        }

        var fromNode = GetTaxiNode(route.FromNodeId);
        if (fromNode != null && !string.IsNullOrWhiteSpace(fromNode.MountModelPath))
            return fromNode;

        var toNode = GetTaxiNode(route.ToNodeId);
        if (toNode != null && !string.IsNullOrWhiteSpace(toNode.MountModelPath))
            return toNode;

        return fromNode ?? toNode;
    }

    private static bool TryGetTaxiRouteSelectionPoint(TaxiPathLoader.TaxiRoute route, out Vector3 point)
    {
        if (route.Waypoints.Count == 0)
        {
            point = Vector3.Zero;
            return false;
        }

        float routeLength = GetRouteLength(route.Waypoints);
        if (routeLength <= 1f)
        {
            point = route.Waypoints[route.Waypoints.Count / 2];
            return true;
        }

        SampleRoute(route.Waypoints, routeLength * 0.5f, out point, out _);
        return true;
    }

    private static float GetRouteLength(List<Vector3> waypoints)
    {
        float total = 0f;
        for (int i = 0; i < waypoints.Count - 1; i++)
            total += Vector3.Distance(waypoints[i], waypoints[i + 1]);
        return total;
    }

    private static void SampleRoute(List<Vector3> waypoints, float distance, out Vector3 position, out Vector3 direction)
    {
        float remaining = distance;
        for (int i = 0; i < waypoints.Count - 1; i++)
        {
            Vector3 start = waypoints[i];
            Vector3 end = waypoints[i + 1];
            Vector3 segment = end - start;
            float segmentLength = segment.Length();
            if (segmentLength <= 0.001f)
                continue;

            if (remaining <= segmentLength)
            {
                float t = remaining / segmentLength;
                position = Vector3.Lerp(start, end, t);
                direction = Vector3.Normalize(segment);
                return;
            }

            remaining -= segmentLength;
        }

        position = waypoints[^1];
        direction = waypoints[^1] - waypoints[^2];
        if (direction.LengthSquared() > 0.0001f)
            direction = Vector3.Normalize(direction);
    }

    private static Vector3 SampleSmoothedTaxiRouteDirection(List<Vector3> waypoints, float distance, float routeLength)
    {
        if (waypoints.Count < 2)
            return Vector3.UnitX;

        float sampleWindow = MathF.Min(TaxiActorHeadingSampleWindow, MathF.Max(1f, routeLength * 0.1f));
        float behindDistance = WrapTaxiRouteDistance(distance - sampleWindow * 0.5f, routeLength);
        float aheadDistance = WrapTaxiRouteDistance(distance + sampleWindow * 0.5f, routeLength);

        SampleRoute(waypoints, behindDistance, out Vector3 behindPosition, out Vector3 behindDirection);
        SampleRoute(waypoints, aheadDistance, out Vector3 aheadPosition, out Vector3 aheadDirection);

        Vector3 tangent = aheadPosition - behindPosition;
        if (tangent.LengthSquared() > 0.0001f)
            return Vector3.Normalize(tangent);

        Vector3 fallback = aheadDirection.LengthSquared() > 0.0001f ? aheadDirection : behindDirection;
        if (fallback.LengthSquared() > 0.0001f)
            return Vector3.Normalize(fallback);

        return Vector3.UnitX;
    }

    private static float WrapTaxiRouteDistance(float distance, float routeLength)
    {
        if (routeLength <= 0f)
            return 0f;

        while (distance < 0f)
            distance += routeLength;

        while (distance >= routeLength)
            distance -= routeLength;

        return distance;
    }

    private static float ComputeTaxiActorYawRadians(Vector3 actorForward)
    {
        Vector3 horizontalForward = new Vector3(actorForward.X, actorForward.Y, 0f);
        if (horizontalForward.LengthSquared() <= 0.0001f)
            return 0f;

        horizontalForward = Vector3.Normalize(horizontalForward);
        return MathF.Atan2(horizontalForward.Y, horizontalForward.X);
    }

    private void LazyLoadAreaTriggers()
    {
        _areaTriggerLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _areaTriggerLoader = new AreaTriggerLoader();
        _areaTriggerLoader.Load(_dbcProvider, _dbdDir, _dbcBuild, _mapId);
    }

    /// <summary>
    /// Load the exact-build Light* DBC chain for zone-based lighting, with the flattened
    /// LightData table retained only as a later-build compatibility fallback.
    /// </summary>
    public void LoadLighting(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _lightService = new LightService();
        _lightService.Load(dbcProvider, dbdDir, build, mapId);
    }

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        string? buildVersion = null,
        MinimapRenderer? minimapRenderer = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
        _minimapRenderer = minimapRenderer;
        _pm4OverlayCacheService = Pm4OverlayCacheService.CreateForDataSource(dataSource);
        _assets = new WorldAssetManager(gl, dataSource, texResolver, buildVersion);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);

        // Create terrain manager (uses AOI-based lazy loading — tiles load as camera moves)
        onStatus?.Invoke("Loading WDT...");
        _terrainManager = new TerrainManager(gl, wdtPath, dataSource);

        InitFromAdapter(onStatus);
    }

    /// <summary>
    /// Create a WorldScene with a pre-built TerrainManager (for Standard WDT, etc.).
    /// </summary>
    public WorldScene(GL gl, TerrainManager terrainManager, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        string? buildVersion = null,
        MinimapRenderer? minimapRenderer = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
        _minimapRenderer = minimapRenderer;
        _pm4OverlayCacheService = Pm4OverlayCacheService.CreateForDataSource(dataSource);
        _assets = new WorldAssetManager(gl, dataSource, texResolver, buildVersion);
        _bbRenderer = new BoundingBoxRenderer(gl);
        _skyDome = new SkyDomeRenderer(gl);
        _terrainManager = terrainManager;

        InitFromAdapter(onStatus);
    }

    private void InitFromAdapter(Action<string>? onStatus)
    {
        var adapter = _terrainManager.Adapter;
        _assetLoadPolicy = ResolveTerrainAssetLoadPolicy(adapter);

        if (adapter.ModfPlacements.Count > 0)
        {
            // Pre-load WDT global WMO placements + models
            var manifest = _assets.BuildManifest(
                adapter.MdxModelNames, adapter.WmoModelNames,
                adapter.MddfPlacements, adapter.ModfPlacements);
            _assets.LoadManifest(manifest);
            BuildInstances(adapter);
        }

        if (adapter.IsWmoBased)
        {
            if (adapter.ModfPlacements.Count > 0)
            {
                var p = adapter.ModfPlacements[0];
                var bbCenter = (p.BoundsMin + p.BoundsMax) * 0.5f;
                var bbExtent = p.BoundsMax - p.BoundsMin;
                float dist = MathF.Max(bbExtent.Length() * 0.5f, 100f);
                _wmoCameraOverride = bbCenter + new Vector3(dist, 0, bbExtent.Z * 0.3f);
                ViewerLog.Info(ViewerLog.Category.Terrain, $"WMO-only map, camera at BB center: ({bbCenter.X:F1}, {bbCenter.Y:F1}, {bbCenter.Z:F1}), dist={dist:F0}");
            }

            // Still subscribe for any late-loaded tiles
            _terrainManager.OnTileLoaded += OnTileLoaded;
            _terrainManager.OnTileUnloaded += OnTileUnloaded;
            onStatus?.Invoke("World loaded (WMO-only map).");
        }
        else
        {
            // Terrain maps: load WDL low-res mesh first for instant overview,
            // then stream detailed ADT tiles via AOI as the camera moves.
            if (_dataSource != null)
            {
                onStatus?.Invoke("Loading WDL terrain...");
                _wdlTerrain = new WdlTerrainRenderer(_gl, _minimapRenderer);
                if (!_wdlTerrain.Load(_dataSource, _terrainManager.MapName))
                {
                    _wdlTerrain.Dispose();
                    _wdlTerrain = null;
                }
            }

            _terrainManager.OnTileLoaded += OnTileLoaded;
            _terrainManager.OnTileUnloaded += OnTileUnloaded;
            onStatus?.Invoke("World loaded (tiles stream as you move).");
        }

        if (!adapter.IsWmoBased)
        {
            AdtProfile adtProfile = FormatProfileRegistry.ResolveAdtProfile(_dbcBuild);
            ViewerLog.Info(
                ViewerLog.Category.Terrain,
                $"Terrain asset load policy: build={_dbcBuild ?? "unknown"}, adtProfile={adtProfile.ProfileId}, prewarmTileAssets={_assetLoadPolicy.PrewarmTileAssets}, visibleMdx={_assetLoadPolicy.MaxNewMdxLoadsPerFrame}, visibleWmo={_assetLoadPolicy.MaxNewWmoLoadsPerFrame}, deferredLoads={_assetLoadPolicy.MaxDeferredLoadsPerFrame}, deferredBudgetMs={_assetLoadPolicy.MaxDeferredLoadBudgetMs:F1}");
        }
        
        // Auto-load WL liquids if enabled
        if (_showWlLiquids && !_wlLoadAttempted)
        {
            LazyLoadWlLiquids();
        }
    }

    private TerrainAssetLoadPolicy ResolveTerrainAssetLoadPolicy(ITerrainAdapter adapter)
    {
        return adapter.IsWmoBased
            ? WmoOnlyAssetLoadPolicy
            : StreamingTerrainAssetLoadPolicy;
    }

    private Vector3? _wmoCameraOverride;
    /// <summary>For WMO-only maps, returns the WMO position as camera start. Otherwise null.</summary>
    public Vector3? WmoCameraOverride => _wmoCameraOverride;

    private void BuildInstances(ITerrainAdapter adapter)
    {
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Placement transform for terrain maps.
        // Positions are already converted to renderer coords in AlphaTerrainAdapter:
        //   rendererX = MapOrigin - wowY, rendererY = MapOrigin - wowX, rendererZ = wowZ
        // Triangle winding is reversed at upload (CW→CCW for OpenGL), which flips the
        // model's facing direction by 180°. Compensate with a 180° Z rotation.
        var rot180Z = Matrix4x4.CreateRotationZ(MathF.PI);
        bool wmoBased = adapter.IsWmoBased;

        // MDX (doodad) placements — same rotation as WMO (wiki confirms "same as MODF"),
        // with scale added. Rotation stored as degrees in file.
        foreach (var p in adapter.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;
            // Rotation stored as degrees in WoW coords (X=North, Y=West, Z=Up).
            // Position axes are swapped: wowX→rendererY, wowY→rendererX (both negated).
            // Rotation axes must follow the same swap:
            //   WoW rotX (tilt around North) → renderer RotationY (negated)
            //   WoW rotY (tilt around West)  → renderer RotationX (negated)
            //   WoW rotZ (heading around Up)  → renderer RotationZ (as-is)
            float rx = -p.Rotation.Y * MathF.PI / 180f;
            float ry = -p.Rotation.X * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transform = rot180Z
                * Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Use actual model bounds if available, transformed to world space
            Vector3 bbMin, bbMax;
            Vector3 localMin = Vector3.Zero;
            Vector3 localMax = Vector3.Zero;
            bool boundsResolved = false;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
            {
                localMin = modelMin;
                localMax = modelMax;
                boundsResolved = true;
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            }
            else
            {
                bbMin = p.Position - new Vector3(2f);
                bbMax = p.Position + new Vector3(2f);
            }
            string modelPath = mdxNames[p.NameIndex];
            var instance = new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = bbMin,
                BoundsMax = bbMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = boundsResolved,
                ModelName = Path.GetFileName(modelPath),
                ModelPath = modelPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = scale,
                UniqueId = p.UniqueId,
                PlacementEntryIndex = -1,
                TileX = -1,
                TileY = -1,
                HasTileCoordinate = false
            };

            if (IsSkyboxModelPath(modelPath))
                _skyboxInstances.Add(instance);
            else
                _mdxInstances.Add(instance);
        }

        // WMO placements
        foreach (var p in adapter.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;

            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            var transform = rot180Z
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Get geometry-tight local bounds for the WMO placement and transform them to world space.
            // Falls back to MODF file bounds if the model summary is unavailable.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
            {
                TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
            }
            else
            {
                localMin = localMax = Vector3.Zero;
                worldMin = p.BoundsMin;
                worldMax = p.BoundsMax;
            }

            string wmoPath = wmoNames[p.NameIndex];
            _wmoInstances.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
                ModelName = Path.GetFileName(wmoPath),
                ModelPath = wmoPath,
                PlacementPosition = p.Position,
                PlacementRotation = p.Rotation,
                PlacementScale = 1.0f,
                UniqueId = p.UniqueId,
                PlacementEntryIndex = -1,
                TileX = -1,
                TileY = -1,
                HasTileCoordinate = false
            });
        }

        ViewerLog.Important(ViewerLog.Category.Terrain, $"Instances: {_mdxInstances.Count} MDX, {_skyboxInstances.Count} skybox, {_wmoInstances.Count} WMO");
        // Diagnostic: terrain chunk WorldPosition range
        var camPos = _terrainManager.GetInitialCameraPosition();
        ViewerLog.Info(ViewerLog.Category.Terrain, $"Camera: ({camPos.X:F1}, {camPos.Y:F1}, {camPos.Z:F1})");
        // Compute terrain bounding box from chunk WorldPositions
        float tMinX = float.MaxValue, tMinY = float.MaxValue, tMinZ = float.MaxValue;
        float tMaxX = float.MinValue, tMaxY = float.MinValue, tMaxZ = float.MinValue;
        foreach (var chunk in _terrainManager.Adapter.LastLoadedChunkPositions)
        {
            tMinX = Math.Min(tMinX, chunk.X); tMaxX = Math.Max(tMaxX, chunk.X);
            tMinY = Math.Min(tMinY, chunk.Y); tMaxY = Math.Max(tMaxY, chunk.Y);
            tMinZ = Math.Min(tMinZ, chunk.Z); tMaxZ = Math.Max(tMaxZ, chunk.Z);
        }
        ViewerLog.Info(ViewerLog.Category.Terrain, $"TERRAIN  X:[{tMinX:F1} .. {tMaxX:F1}]  Y:[{tMinY:F1} .. {tMaxY:F1}]  Z:[{tMinZ:F1} .. {tMaxZ:F1}]");

        // Compute object bounding box (from stored positions, which are already transformed)
        float oMinX = float.MaxValue, oMinY = float.MaxValue, oMinZ = float.MaxValue;
        float oMaxX = float.MinValue, oMaxY = float.MinValue, oMaxZ = float.MinValue;
        foreach (var p in adapter.MddfPlacements)
        {
            oMinX = Math.Min(oMinX, p.Position.X); oMaxX = Math.Max(oMaxX, p.Position.X);
            oMinY = Math.Min(oMinY, p.Position.Y); oMaxY = Math.Max(oMaxY, p.Position.Y);
            oMinZ = Math.Min(oMinZ, p.Position.Z); oMaxZ = Math.Max(oMaxZ, p.Position.Z);
        }
        foreach (var p in adapter.ModfPlacements)
        {
            oMinX = Math.Min(oMinX, p.Position.X); oMaxX = Math.Max(oMaxX, p.Position.X);
            oMinY = Math.Min(oMinY, p.Position.Y); oMaxY = Math.Max(oMaxY, p.Position.Y);
            oMinZ = Math.Min(oMinZ, p.Position.Z); oMaxZ = Math.Max(oMaxZ, p.Position.Z);
        }
        ViewerLog.Info(ViewerLog.Category.Terrain, $"OBJECTS  X:[{oMinX:F1} .. {oMaxX:F1}]  Y:[{oMinY:F1} .. {oMaxY:F1}]  Z:[{oMinZ:F1} .. {oMaxZ:F1}]");
        ViewerLog.Info(ViewerLog.Category.Terrain, $"DELTA    X:{(tMinX+tMaxX)/2 - (oMinX+oMaxX)/2:F1}  Y:{(tMinY+tMaxY)/2 - (oMinY+oMaxY)/2:F1}  Z:{(tMinZ+tMaxZ)/2 - (oMinZ+oMaxZ)/2:F1}");

        // Print first 3 MDDF raw values for manual inspection
        for (int i = 0; i < Math.Min(3, adapter.MddfPlacements.Count); i++)
        {
            var p = adapter.MddfPlacements[i];
            string name = p.NameIndex < mdxNames.Count ? Path.GetFileName(mdxNames[p.NameIndex]) : "?";
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  MDDF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
        for (int i = 0; i < Math.Min(3, adapter.ModfPlacements.Count); i++)
        {
            var p = adapter.ModfPlacements[i];
            string name = p.NameIndex < wmoNames.Count ? Path.GetFileName(wmoNames[p.NameIndex]) : "?";
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  MODF[{i}] pos=({p.Position.X:F1}, {p.Position.Y:F1}, {p.Position.Z:F1}) model={name}");
        }
    }

    /// <summary>
    /// Called by TerrainManager when a new tile enters the AOI.
    /// Builds object instances for the tile and lazy-loads any new models.
    /// </summary>
    private void OnTileLoaded(int tileX, int tileY, TileLoadResult result)
    {
        var adapter = _terrainManager.Adapter;
        var mdxNames = adapter.MdxModelNames;
        var wmoNames = adapter.WmoModelNames;

        // Build MDX instances for this tile
        var tileMdx = new List<ObjectInstance>();
        var tileSkyboxes = new List<ObjectInstance>();
        int tileMddfEntryIndex = 0;
        foreach (var p in result.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            float scale = p.Scale > 0 ? p.Scale : 1.0f;

            // Rotation stored as degrees in WoW coords — axes swapped to match position swap.
            float rx = -p.Rotation.Y * MathF.PI / 180f;
            float ry = -p.Rotation.X * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // 180° Z rotation compensates for winding reversal (CW→CCW)
            var rot180Z = Matrix4x4.CreateRotationZ(MathF.PI);
            var transform = rot180Z
                * Matrix4x4.CreateScale(scale)
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);
            Vector3 bbMin, bbMax;
            Vector3 localMin = Vector3.Zero;
            Vector3 localMax = Vector3.Zero;
            bool boundsResolved = false;
            if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
            {
                localMin = modelMin;
                localMax = modelMax;
                boundsResolved = true;
                TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
            }
            else
            { bbMin = p.Position - new Vector3(2f); bbMax = p.Position + new Vector3(2f); }
            string modelPath = mdxNames[p.NameIndex];
            var instance = new ObjectInstance
            {
                ModelKey = key, Transform = transform, BoundsMin = bbMin, BoundsMax = bbMax,
                LocalBoundsMin = localMin, LocalBoundsMax = localMax, BoundsResolved = boundsResolved,
                ModelName = Path.GetFileName(modelPath), ModelPath = modelPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = scale,
                UniqueId = p.UniqueId,
                PlacementEntryIndex = tileMddfEntryIndex,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true
            };

            if (IsSkyboxModelPath(modelPath))
                tileSkyboxes.Add(instance);
            else
                tileMdx.Add(instance);

            tileMddfEntryIndex++;
        }

        // Build WMO instances for this tile
        var tileWmo = new List<ObjectInstance>();
        int tileModfEntryIndex = 0;
        foreach (var p in result.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            float rx = p.Rotation.X * MathF.PI / 180f;
            float ry = p.Rotation.Y * MathF.PI / 180f;
            float rz = p.Rotation.Z * MathF.PI / 180f;

            // 180° Z rotation compensates for winding reversal (CW→CCW)
            var rot180Z = Matrix4x4.CreateRotationZ(MathF.PI);
            var transform = rot180Z
                * Matrix4x4.CreateRotationX(rx)
                * Matrix4x4.CreateRotationY(ry)
                * Matrix4x4.CreateRotationZ(rz)
                * Matrix4x4.CreateTranslation(p.Position);

            // Get geometry-tight local bounds and transform to world space.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
            {
                TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
            }
            else
            {
                localMin = localMax = Vector3.Zero;
                worldMin = p.BoundsMin;
                worldMax = p.BoundsMax;
            }

            string wmoPath = wmoNames[p.NameIndex];
            tileWmo.Add(new ObjectInstance
            {
                ModelKey = key,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
                ModelName = Path.GetFileName(wmoPath), ModelPath = wmoPath,
                PlacementPosition = p.Position, PlacementRotation = p.Rotation, PlacementScale = 1.0f,
                UniqueId = p.UniqueId,
                PlacementEntryIndex = tileModfEntryIndex,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true
            });

            tileModfEntryIndex++;
        }

        _tileMdxInstances[(tileX, tileY)] = tileMdx;
        _tileSkyboxInstances[(tileX, tileY)] = tileSkyboxes;
        _tileWmoInstances[(tileX, tileY)] = tileWmo;
        UpdateObjectBucketBounds(_tileMdxBounds, (tileX, tileY), tileMdx);
        UpdateObjectBucketBounds(_tileWmoBounds, (tileX, tileY), tileWmo);
        _instancesDirty = true;

        if (_assetLoadPolicy.PrewarmTileAssets
            || (CapturePreloadActive && _capturePreloadTiles.Contains((tileX, tileY))))
            QueueTileAssetLoads(tileMdx, tileSkyboxes, tileWmo);

        // Hide WDL low-res tile now that detailed ADT is loaded
        _wdlTerrain?.MarkDetailedTileResident(tileX, tileY);

        if ((tileMdx.Count > 0 || tileSkyboxes.Count > 0 || tileWmo.Count > 0) && ViewerLog.Verbose)
            ViewerLog.Trace($"[Terrain] Tile ({tileX},{tileY}) loaded: {tileMdx.Count} MDX, {tileSkyboxes.Count} skybox, {tileWmo.Count} WMO instances");
    }

    /// <summary>
    /// Called by TerrainManager when a tile leaves the AOI.
    /// </summary>
    private void OnTileUnloaded(int tileX, int tileY)
    {
        int wmoInstanceCount = _tileWmoInstances.TryGetValue((tileX, tileY), out List<ObjectInstance>? wmoInstances)
            ? wmoInstances.Count
            : 0;
        _tileMdxInstances.Remove((tileX, tileY));
        _tileSkyboxInstances.Remove((tileX, tileY));
        _tileWmoInstances.Remove((tileX, tileY));
        _tileMdxBounds.Remove((tileX, tileY));
        _tileWmoBounds.Remove((tileX, tileY));
        _wdlTerrain?.MarkDetailedTileUnloaded(tileX, tileY);
        LastUnloadedWmoTileX = tileX;
        LastUnloadedWmoTileY = tileY;
        LastUnloadedWmoInstanceCount = wmoInstanceCount;
        WmoTileUnloadEventCount++;
        _instancesDirty = true;
    }

    /// <summary>
    /// Rebuild flat instance lists from per-tile dictionaries.
    /// Called lazily before rendering when _instancesDirty is true.
    /// </summary>
    private void RebuildInstanceLists()
    {
        _mdxInstances.Clear();
        foreach (var list in _tileMdxInstances.Values)
            _mdxInstances.AddRange(list);
        _mdxInstances.AddRange(_externalMdxInstances);

        _skyboxInstances.Clear();
        foreach (var list in _tileSkyboxInstances.Values)
            _skyboxInstances.AddRange(list);
        _skyboxInstances.AddRange(_externalSkyboxInstances);

        _wmoInstances.Clear();
        foreach (var list in _tileWmoInstances.Values)
            _wmoInstances.AddRange(list);
        _wmoInstances.AddRange(_externalWmoInstances);

        RebuildFlatVisibilityBuckets();

        RestoreSelectedSceneObjectAfterRebuild();
        if (UseHierarchicalSceneTraversal)
            RebuildSceneGraphObjectIndex();
        else
            _sceneGraphBuild = null;

        _instancesDirty = false;
    }

    private void RebuildFlatVisibilityBuckets()
    {
        RebuildFlatVisibilityBuckets(_tileMdxInstances, _tileMdxVisibilityBuckets);
        RebuildFlatVisibilityBuckets(_tileWmoInstances, _tileWmoVisibilityBuckets);
    }

    private static void RebuildFlatVisibilityBuckets(
        IReadOnlyDictionary<(int, int), List<ObjectInstance>> source,
        Dictionary<(int, int), List<FlatVisibilityBucket>> destination)
    {
        destination.Clear();
        foreach (KeyValuePair<(int, int), List<ObjectInstance>> tile in source)
        {
            Dictionary<(int chunkX, int chunkY), FlatVisibilityBucket> byChunk = new();
            FlatVisibilityBucket? fallback = null;
            foreach (ObjectInstance instance in tile.Value)
            {
                if (!TryGetSceneObjectChunkKey(instance, out (int tileX, int tileY, int chunkX, int chunkY) chunkKey)
                    || chunkKey.tileX != tile.Key.Item1
                    || chunkKey.tileY != tile.Key.Item2)
                {
                    fallback ??= new FlatVisibilityBucket();
                    fallback.Add(instance);
                    continue;
                }

                if (!byChunk.TryGetValue((chunkKey.chunkX, chunkKey.chunkY), out FlatVisibilityBucket? bucket))
                {
                    bucket = new FlatVisibilityBucket();
                    byChunk.Add((chunkKey.chunkX, chunkKey.chunkY), bucket);
                }

                bucket.Add(instance);
            }

            List<FlatVisibilityBucket> buckets = new(byChunk.Count + (fallback is null ? 0 : 1));
            buckets.AddRange(byChunk.Values);
            if (fallback is not null)
                buckets.Add(fallback);
            destination[tile.Key] = buckets;
        }
    }

    private void RebuildSceneGraphObjectIndex()
    {
        _sceneGraphPortalAdapters.Clear();
        _sceneGraphPortalVisibility.Clear();
        List<WorldSceneGraphObjectPlacement> placements = new(
            _mdxInstances.Count + _skyboxInstances.Count + _wmoInstances.Count);

        bool hasPartitionedSources = _tileMdxInstances.Count > 0
            || _tileSkyboxInstances.Count > 0
            || _tileWmoInstances.Count > 0
            || _externalMdxInstances.Count > 0
            || _externalSkyboxInstances.Count > 0
            || _externalWmoInstances.Count > 0;
        if (hasPartitionedSources)
        {
            AppendSceneGraphPlacements(placements, _tileMdxInstances, WorldSceneNodeKind.M2Placement, isSkybox: false, isExternal: false, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _tileSkyboxInstances, WorldSceneNodeKind.M2Placement, isSkybox: true, isExternal: false, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _tileWmoInstances, WorldSceneNodeKind.WmoPlacement, isSkybox: false, isExternal: false, requiresUpdate: false, childFactory: BuildWmoSceneGraphChildren);
            AppendSceneGraphPlacements(placements, _externalMdxInstances, WorldSceneNodeKind.M2Placement, isSkybox: false, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _externalSkyboxInstances, WorldSceneNodeKind.M2Placement, isSkybox: true, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _externalWmoInstances, WorldSceneNodeKind.WmoPlacement, isSkybox: false, isExternal: true, requiresUpdate: false, childFactory: BuildWmoSceneGraphChildren);
        }
        else
        {
            AppendSceneGraphPlacements(placements, _mdxInstances, WorldSceneNodeKind.M2Placement, isSkybox: false, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _skyboxInstances, WorldSceneNodeKind.M2Placement, isSkybox: true, isExternal: true, requiresUpdate: true);
            AppendSceneGraphPlacements(placements, _wmoInstances, WorldSceneNodeKind.WmoPlacement, isSkybox: false, isExternal: true, requiresUpdate: false, childFactory: BuildWmoSceneGraphChildren);
        }

        _sceneGraphBuild = WorldSceneGraphObjectAdapter.BuildPerAdt(placements);
        _sceneGraphFrameVisibilityPrepared = false;
        _lastSceneGraphTraversalDiagnostics = new WorldSceneTraversalDiagnostics();
    }

    private static void AppendSceneGraphPlacements(
        List<WorldSceneGraphObjectPlacement> destination,
        IEnumerable<KeyValuePair<(int, int), List<ObjectInstance>>> tileInstances,
        WorldSceneNodeKind kind,
        bool isSkybox,
        bool isExternal,
        bool requiresUpdate,
        Func<string, ObjectInstance, IReadOnlyList<WorldSceneGraphChildNode>?>? childFactory = null)
    {
        foreach (KeyValuePair<(int, int), List<ObjectInstance>> tile in tileInstances)
        {
            AppendSceneGraphPlacements(destination, tile.Value, kind, isSkybox, isExternal, requiresUpdate, childFactory, tile.Key);
        }
    }

    private static void AppendSceneGraphPlacements(
        List<WorldSceneGraphObjectPlacement> destination,
        IReadOnlyList<ObjectInstance> instances,
        WorldSceneNodeKind kind,
        bool isSkybox,
        bool isExternal,
        bool requiresUpdate,
        Func<string, ObjectInstance, IReadOnlyList<WorldSceneGraphChildNode>?>? childFactory = null,
        (int tileX, int tileY)? tileKey = null)
    {
        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance instance = instances[index];
            string sourceToken = isExternal
                ? "external"
                : $"tile/{tileKey!.Value.tileX:D2}/{tileKey.Value.tileY:D2}";
            string id = $"world/object/{GetSceneGraphKindToken(kind, isSkybox)}/{sourceToken}/{index:D6}";
            destination.Add(new WorldSceneGraphObjectPlacement(
                id,
                kind,
                instance,
                isExternal,
                WorldSceneRenderPass.Opaque,
                IsQueryable: true,
                RequiresUpdate: requiresUpdate,
                IsSkybox: isSkybox,
                Children: childFactory?.Invoke(id, instance),
                SpatialBucket: GetSceneGraphSpatialBucket(kind, isSkybox, isExternal, tileKey, instance)));
        }
    }

    private static WorldSceneGraphSpatialBucket? GetSceneGraphSpatialBucket(
        WorldSceneNodeKind kind,
        bool isSkybox,
        bool isExternal,
        (int tileX, int tileY)? tileKey,
        in ObjectInstance instance)
    {
        if (kind != WorldSceneNodeKind.M2Placement
            || isSkybox
            || isExternal
            || !tileKey.HasValue
            || !instance.HasTileCoordinate
            || !TryGetSceneObjectChunkKey(instance, out (int tileX, int tileY, int chunkX, int chunkY) chunkKey)
            || chunkKey.tileX != tileKey.Value.tileX
            || chunkKey.tileY != tileKey.Value.tileY)
        {
            return null;
        }

        return new WorldSceneGraphSpatialBucket(
            WorldSceneNodeKind.Chunk,
            $"{chunkKey.chunkX:D2}/{chunkKey.chunkY:D2}");
    }

    private IReadOnlyList<WorldSceneGraphChildNode>? BuildWmoSceneGraphChildren(
        string parentId,
        ObjectInstance instance)
    {
        if (_assets.TryGetLoadedWmo(instance.ModelKey, out WmoRenderer? renderer) && renderer is not null)
        {
            _sceneGraphPortalAdapters[parentId] = WorldScenePortalAdapter.Build(
                renderer.GetSceneGraphPortalGroups(),
                renderer.GetSceneGraphPortalReadModels(),
                parentId);
        }

        if (!_assets.TryGetCachedWmoMeshSummary(instance.ModelKey, out WmoMeshSummary summary)
            || summary.GroupSummaries is null
            || summary.GroupSummaries.Length == 0)
        {
            return null;
        }

        List<WorldSceneGraphChildNode> children = new(summary.GroupSummaries.Length);
        foreach (WmoGroupMeshSummary group in summary.GroupSummaries.OrderBy(group => group.GroupIndex))
        {
            bool boundsKnown = AreFiniteOrderedBounds(group.BoundsMin, group.BoundsMax);
            children.Add(new WorldSceneGraphChildNode(
                $"{parentId}/group/{group.GroupIndex:D4}",
                WorldSceneNodeKind.WmoGroup,
                Matrix4x4.Identity,
                boundsKnown ? group.BoundsMin : Vector3.Zero,
                boundsKnown ? group.BoundsMax : Vector3.Zero,
                BoundsKnown: boundsKnown,
                IsRenderable: true,
                IsQueryable: true,
                RequiresUpdate: false,
                AssetKey: $"{instance.ModelKey}#group/{group.GroupIndex:D4}",
                RenderPassMask: WorldSceneRenderPass.Opaque,
                PortalGroup: group.GroupIndex));
        }

        return children;
    }

    private static bool AreFiniteOrderedBounds(Vector3 min, Vector3 max)
    {
        return float.IsFinite(min.X) && float.IsFinite(min.Y) && float.IsFinite(min.Z)
            && float.IsFinite(max.X) && float.IsFinite(max.Y) && float.IsFinite(max.Z)
            && min.X <= max.X && min.Y <= max.Y && min.Z <= max.Z;
    }

    private static string GetSceneGraphKindToken(WorldSceneNodeKind kind, bool isSkybox)
    {
        if (kind == WorldSceneNodeKind.WmoPlacement)
            return "wmo";

        return isSkybox ? "m2-skybox" : "m2";
    }

    private bool TryGetSelectedSceneInstance(out ObjectInstance instance)
    {
        // Asset promotion can resolve bounds after the frame-maintenance pass and mark the
        // placement lists dirty. Never rebuild the full world synchronously from a read/query
        // accessor (the render-time selected-bounds overlay is one such accessor); let the next
        // frame's scene-maintenance pass perform the rebuild once.
        if (_instancesDirty)
        {
            instance = default;
            return false;
        }

        if (TryGetSceneObjectByIndex(_selectedObjectType, _selectedObjectIndex, out instance))
        {
            if (!_selectedSceneObjectKey.HasValue || !IsSameSceneObject(instance, _selectedSceneObjectKey.Value))
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(_selectedObjectType, instance);

            return true;
        }

        if (_selectedSceneObjectKey.HasValue
            && TryResolveSelectedSceneObject(_selectedSceneObjectKey.Value, out ObjectType resolvedType, out int resolvedIndex, out instance))
        {
            _selectedObjectType = resolvedType;
            _selectedObjectIndex = resolvedIndex;
            return true;
        }

        instance = default;
        return false;
    }

    private void RestoreSelectedSceneObjectAfterRebuild()
    {
        if (!_selectedSceneObjectKey.HasValue)
            return;

        if (TryResolveSelectedSceneObject(_selectedSceneObjectKey.Value, out ObjectType resolvedType, out int resolvedIndex, out _))
        {
            _selectedObjectType = resolvedType;
            _selectedObjectIndex = resolvedIndex;
            return;
        }

        _selectedObjectIndex = -1;
    }

    private bool TryResolveSelectedSceneObject(SelectedSceneObjectKey key, out ObjectType objectType, out int objectIndex, out ObjectInstance instance)
    {
        List<ObjectInstance> instances = key.ObjectType switch
        {
            ObjectType.Wmo => _wmoInstances,
            ObjectType.Mdx => _mdxInstances,
            _ => []
        };

        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance candidate = instances[index];
            if (!IsSameSceneObject(candidate, key))
                continue;

            objectType = key.ObjectType;
            objectIndex = index;
            instance = candidate;
            return true;
        }

        objectType = ObjectType.None;
        objectIndex = -1;
        instance = default;
        return false;
    }

    private bool TryGetSceneObjectByIndex(ObjectType objectType, int objectIndex, out ObjectInstance instance)
    {
        switch (objectType)
        {
            case ObjectType.Wmo when objectIndex >= 0 && objectIndex < _wmoInstances.Count:
                instance = _wmoInstances[objectIndex];
                return true;
            case ObjectType.Mdx when objectIndex >= 0 && objectIndex < _mdxInstances.Count:
                instance = _mdxInstances[objectIndex];
                return true;
            default:
                instance = default;
                return false;
        }
    }

    private static SelectedSceneObjectKey CreateSelectedSceneObjectKey(ObjectType objectType, ObjectInstance instance)
    {
        return new SelectedSceneObjectKey(
            objectType,
            instance.UniqueId,
            instance.PlacementEntryIndex,
            instance.TileX,
            instance.TileY,
            instance.HasTileCoordinate,
            instance.ModelKey,
            instance.PlacementPosition);
    }

    private static bool IsSameSceneObject(ObjectInstance candidate, SelectedSceneObjectKey key)
    {
        if (candidate.UniqueId != key.UniqueId || candidate.PlacementEntryIndex != key.PlacementEntryIndex)
            return false;

        if (candidate.HasTileCoordinate != key.HasTileCoordinate)
            return false;

        if (candidate.HasTileCoordinate)
            return candidate.TileX == key.TileX && candidate.TileY == key.TileY;

        if (!string.Equals(candidate.ModelKey, key.ModelKey, StringComparison.OrdinalIgnoreCase))
            return false;

        return Vector3.DistanceSquared(candidate.PlacementPosition, key.PlacementPosition) < 0.0001f;
    }

    private IModelRenderer? TryGetQueuedMdx(string modelKey)
    {
        if (_assets.TryGetLoadedMdx(modelKey, out var renderer))
            return renderer;
        return null;
    }

    private WmoRenderer? TryGetQueuedWmo(string modelKey)
    {
        if (_assets.TryGetLoadedWmo(modelKey, out var renderer))
            return renderer;
        return null;
    }

    private IModelRenderer? ResolveVisibleMdxRenderer(WorldRenderFrame frame, string modelKey)
    {
        if (frame.VisibleMdxRendererCache.TryGetValue(modelKey, out IModelRenderer? renderer))
            return renderer;

        renderer = TryGetQueuedMdx(modelKey);
        if (renderer != null)
            frame.VisibleMdxRendererCache[modelKey] = renderer;

        return renderer;
    }

    private WmoRenderer? ResolveVisibleWmoRenderer(WorldRenderFrame frame, string modelKey)
    {
        if (frame.VisibleWmoRendererCache.TryGetValue(modelKey, out WmoRenderer? renderer))
        {
            renderer?.SetRuntimeDoodadsVisible(_doodadsVisible);
            return renderer;
        }

        renderer = TryGetQueuedWmo(modelKey);
        if (renderer != null)
        {
            renderer.SetRuntimeDoodadsVisible(_doodadsVisible);
            frame.VisibleWmoRendererCache[modelKey] = renderer;
        }

        return renderer;
    }

    private void PlanVisibleMdxPasses(WorldRenderFrame frame)
    {
        WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
            frame.ObjectPasses,
            frame.Visibility,
            visible =>
            {
                // Keep world MDX on the established per-instance RenderWithTransform
                // contract until the shared/GPU batch paths have visual parity proof for
                // every direct MDX and adapted M2 material route. WMO shell/doodad batching
                // remains independent of this fallback.
                return true;
            });

        WorldObjectPassCoordinator.PlanTransparentMdxRoutes(
            frame.ObjectPasses,
            frame.Visibility,
            visible =>
            {
                IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                return renderer != null && renderer.HasTransparentWorldPass;
            });
    }

    private IModelRenderer? ResolveFirstOpaqueBatchedVisibleMdxRenderer(WorldRenderFrame frame)
    {
        if (frame.ObjectPasses.FirstOpaqueBatchedVisibleMdxIndex < 0)
            return null;

        VisibleMdxInstance visible = frame.Visibility.VisibleMdx[frame.ObjectPasses.FirstOpaqueBatchedVisibleMdxIndex];
        return ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
    }

    private static void AccumulateWmoRenderStats(WorldRenderFrame frame, WmoRenderStats stats)
    {
        frame.WmoDrawCallCount += stats.DrawCalls;
        frame.WmoBatchDrawCallCount += stats.BatchDrawCalls;
        frame.WmoOpaqueBatchInstanceCount += stats.OpaqueBatchInstanceCount;
        frame.WmoGroupFallbackDrawCallCount += stats.GroupFallbackDrawCalls;
        frame.WmoLiquidDrawCallCount += stats.LiquidDrawCalls;
        frame.WmoDoodadSubmissionCount += stats.DoodadSubmissions;
        frame.WmoVisibleGroupSubmissionCount += stats.VisibleGroupSubmissions;
    }

    private void TrackPendingVisibleLoad(Dictionary<string, float> pendingLoads, string modelKey, float distanceSq)
    {
        if (pendingLoads.TryGetValue(modelKey, out float existingDistanceSq) && existingDistanceSq <= distanceSq)
            return;

        pendingLoads[modelKey] = distanceSq;
    }

    private void QueueTileAssetLoads(List<ObjectInstance> tileMdx, List<ObjectInstance> tileSkyboxes, List<ObjectInstance> tileWmo)
    {
        for (int i = 0; i < tileWmo.Count; i++)
            _assets.QueueWmoLoad(tileWmo[i].ModelKey);

        for (int i = 0; i < tileMdx.Count; i++)
            _assets.QueueMdxLoad(tileMdx[i].ModelKey);

        for (int i = 0; i < tileSkyboxes.Count; i++)
            _assets.QueueMdxLoad(tileSkyboxes[i].ModelKey);
    }

    /// <summary>
    /// Queue every unique world asset referenced by the supplied resident tiles.
    /// This is the capture-path warmup seam: it uses the same placement lists and
    /// asset queues as normal streaming, but does not make the render path visit
    /// or submit the objects early.
    /// </summary>
    public void QueueCapturePreloadAssets(IEnumerable<(int tileX, int tileY)> tiles)
    {
        ArgumentNullException.ThrowIfNull(tiles);

        _capturePreloadTiles.Clear();
        foreach (var tile in tiles)
        {
            _capturePreloadTiles.Add(tile);
            if (_tileMdxInstances.TryGetValue(tile, out List<ObjectInstance>? mdx)
                && _tileSkyboxInstances.TryGetValue(tile, out List<ObjectInstance>? skyboxes)
                && _tileWmoInstances.TryGetValue(tile, out List<ObjectInstance>? wmo)
                )
            {
                QueueTileAssetLoads(mdx, skyboxes, wmo);
                continue;
            }

            if (_tileMdxInstances.TryGetValue(tile, out mdx))
                QueueTileAssetLoads(mdx, [], []);
            if (_tileSkyboxInstances.TryGetValue(tile, out skyboxes))
                QueueTileAssetLoads([], skyboxes, []);
            if (_tileWmoInstances.TryGetValue(tile, out wmo))
                QueueTileAssetLoads([], [], wmo);
        }
    }

    private void FlushPendingVisibleMdxLoads()
    {
        if (_pendingVisibleMdxLoadDistances.Count == 0)
            return;

        _pendingVisibleMdxLoadScratch.Clear();
        _pendingVisibleMdxLoadScratch.AddRange(_pendingVisibleMdxLoadDistances);
        _pendingVisibleMdxLoadScratch.Sort((left, right) => left.Value.CompareTo(right.Value));

        int queued = 0;
        for (int i = 0; i < _pendingVisibleMdxLoadScratch.Count && queued < _assetLoadPolicy.MaxNewMdxLoadsPerFrame; i++)
        {
            _assets.PrioritizeMdxLoad(_pendingVisibleMdxLoadScratch[i].Key);
            queued++;
        }
    }

    private void FlushPendingVisibleWmoLoads()
    {
        if (_pendingVisibleWmoLoadDistances.Count == 0)
            return;

        _pendingVisibleWmoLoadScratch.Clear();
        _pendingVisibleWmoLoadScratch.AddRange(_pendingVisibleWmoLoadDistances);
        _pendingVisibleWmoLoadScratch.Sort((left, right) => left.Value.CompareTo(right.Value));

        int queued = 0;
        for (int i = 0; i < _pendingVisibleWmoLoadScratch.Count && queued < _assetLoadPolicy.MaxNewWmoLoadsPerFrame; i++)
        {
            _assets.PrioritizeWmoLoad(_pendingVisibleWmoLoadScratch[i].Key);
            queued++;
        }
    }

    private void ProcessDeferredAssetLoads()
    {
        int pendingLoadCount = _assets.PendingAssetLoadCount;
        int maxLoads = _assetLoadPolicy.MaxDeferredLoadsPerFrame;
        double maxBudgetMs = _assetLoadPolicy.MaxDeferredLoadBudgetMs;

        double previousFrameCpuMs = LastRenderFrameStats.TotalCpuMs;
        if (previousFrameCpuMs >= 33.0)
        {
            maxLoads = Math.Min(maxLoads, 1);
            maxBudgetMs = Math.Min(maxBudgetMs, 1.0);
        }
        else if (previousFrameCpuMs >= 20.0)
        {
            maxLoads = Math.Min(maxLoads, 2);
            maxBudgetMs = Math.Min(maxBudgetMs, 1.5);
        }

        if (!_assetLoadPolicy.PrewarmTileAssets && previousFrameCpuMs < 20.0)
        {
            if (pendingLoadCount >= 96)
            {
                maxLoads = Math.Max(maxLoads, 6);
                maxBudgetMs = Math.Max(maxBudgetMs, 4.0);
            }
            else if (pendingLoadCount >= 32)
            {
                maxLoads = Math.Max(maxLoads, 5);
                maxBudgetMs = Math.Max(maxBudgetMs, 3.0);
            }
        }

        if (_assetLoadPolicy.PrewarmTileAssets)
        {
            if (pendingLoadCount >= 96)
            {
                maxLoads = Math.Max(maxLoads, 16);
                maxBudgetMs = Math.Max(maxBudgetMs, 18.0);
            }
            else if (pendingLoadCount >= 32)
            {
                maxLoads = Math.Max(maxLoads, 12);
                maxBudgetMs = Math.Max(maxBudgetMs, 14.0);
            }
        }

        if (CapturePreloadActive)
        {
            maxLoads = Math.Max(maxLoads, 24);
            maxBudgetMs = Math.Max(maxBudgetMs, 16.0);
        }

        int processed = _assets.ProcessPendingLoads(maxLoads, maxBudgetMs);
        if (processed <= 0)
            return;

        bool flatVisibilityBucketsDirty = false;
        foreach (var pair in _tileMdxInstances)
        {
            if (!RefreshMdxInstanceBounds(pair.Value, pair.Key, isSkybox: false, isExternal: false))
                continue;

            UpdateObjectBucketBounds(_tileMdxBounds, pair.Key, pair.Value);
            flatVisibilityBucketsDirty = true;
        }

        foreach (var pair in _tileSkyboxInstances)
            RefreshMdxInstanceBounds(pair.Value, pair.Key, isSkybox: true, isExternal: false);

        foreach (var pair in _tileWmoInstances)
        {
            if (!RefreshWmoInstanceBounds(pair.Value, pair.Key, isSkybox: false, isExternal: false))
                continue;

            UpdateObjectBucketBounds(_tileWmoBounds, pair.Key, pair.Value);
            flatVisibilityBucketsDirty = true;
        }

        RefreshMdxInstanceBounds(_externalMdxInstances, tileKey: null, isSkybox: false, isExternal: true);
        RefreshMdxInstanceBounds(_externalSkyboxInstances, tileKey: null, isSkybox: true, isExternal: true);
        RefreshWmoInstanceBounds(_externalWmoInstances, tileKey: null, isSkybox: false, isExternal: true);

        if (flatVisibilityBucketsDirty)
            RebuildFlatVisibilityBuckets();

        if (CapturePreloadActive)
        {
            _assets.ProcessDeferredWmoDoodadLoads(maxLoads: 24, maxBudgetMs: 12.0);
            _assets.ProcessDeferredWmoMaterialTextureLoads(maxLoads: 24, maxBudgetMs: 12.0);
        }
    }

    private static void UpdateObjectBucketBounds(
        Dictionary<(int, int), (Vector3 Min, Vector3 Max)> boundsByTile,
        (int, int) tileKey,
        IReadOnlyList<ObjectInstance> instances)
    {
        if (instances.Count == 0)
        {
            boundsByTile.Remove(tileKey);
            return;
        }

        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance instance = instances[i];
            min = Vector3.Min(min, instance.BoundsMin);
            max = Vector3.Max(max, instance.BoundsMax);
        }

        boundsByTile[tileKey] = (min, max);
    }

    private static bool AreMdxTileBoundsResolved(IReadOnlyList<ObjectInstance> instances)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            if (!instances[i].BoundsResolved)
                return false;
        }

        return true;
    }

    private bool ShouldVisitObjectBucket(
        Vector3 bucketMin,
        Vector3 bucketMax,
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        bool isWmo,
        bool countAsTaxiActor)
    {
        float boundsDistSq = DistanceSquaredPointToAabb(cameraPos, bucketMin, bucketMax);
        float noCullDistanceSq = ComputeNoCullDistanceSq(bucketMin, bucketMax);
        bool frustumVisible = _frustumCuller.TestAABB(bucketMin, bucketMax);
        Vector3 bucketCenter = (bucketMin + bucketMax) * 0.5f;
        float centerDistanceSq = Vector3.DistanceSquared(cameraPos, bucketCenter);
        float coneFactor = ComputeVisionConeFactor(cameraPos, cameraForward, bucketCenter, centerDistanceSq);

        if (boundsDistSq > noCullDistanceSq && !frustumVisible && coneFactor < MinOffFrustumConeFactor)
            return false;

        float bucketDiagonal = (bucketMax - bucketMin).Length();
        float baseCullDistance = isWmo
            ? ComputeWmoCullDistance(fogEnd, _objectStreamingRangeMultiplier)
            : ComputeMdxCullDistance(fogEnd, bucketDiagonal, countAsTaxiActor, _objectStreamingRangeMultiplier);
        float coneCullDistance = ComputeConeCullDistance(baseCullDistance, coneFactor);
        if (boundsDistSq > coneCullDistance * coneCullDistance)
            return false;

        return centerDistanceSq <= MaxWorldObjectViewDistanceSq;
    }

    private bool ShouldVisitFlatVisibilityBucket(
        FlatVisibilityBucket bucket,
        Vector3 cameraPos,
        float fogEnd,
        bool isWmo)
    {
        // Unresolved or malformed members remain fail-open. The bucket is only a coarse
        // accelerator and must never become a second correctness culler.
        if (!bucket.BoundsKnown || bucket.Instances.Count == 0)
            return true;

        float boundsDistSq = DistanceSquaredPointToAabb(cameraPos, bucket.Min, bucket.Max);
        bool frustumVisible = _frustumCuller.TestAABB(bucket.Min, bucket.Max);
        if (!frustumVisible && boundsDistSq > ComputeNoCullDistanceSq(bucket.Min, bucket.Max))
            return false;

        float bucketDiagonal = (bucket.Max - bucket.Min).Length();
        float baseCullDistance = isWmo
            ? ComputeWmoCullDistance(fogEnd, _objectStreamingRangeMultiplier)
            : ComputeMdxCullDistance(fogEnd, bucketDiagonal, isTaxiActor: false, _objectStreamingRangeMultiplier);
        if (boundsDistSq > baseCullDistance * baseCullDistance)
            return false;

        return boundsDistSq <= MaxWorldObjectViewDistanceSq;
    }

    private bool RefreshMdxInstanceBounds(
        List<ObjectInstance> instances,
        (int tileX, int tileY)? tileKey,
        bool isSkybox,
        bool isExternal)
    {
        bool changed = false;

        for (int i = 0; i < instances.Count; i++)
        {
            var inst = instances[i];
            if (inst.BoundsResolved)
                continue;

            if (!_assets.TryGetMdxBounds(inst.ModelKey, out var localMin, out var localMax))
                continue;

            TransformBounds(localMin, localMax, inst.Transform, out var worldMin, out var worldMax);
            inst.LocalBoundsMin = localMin;
            inst.LocalBoundsMax = localMax;
            inst.BoundsMin = worldMin;
            inst.BoundsMax = worldMax;
            inst.BoundsResolved = true;
            instances[i] = inst;
            UpdateSceneGraphPlacementBounds(tileKey, WorldSceneNodeKind.M2Placement, i, inst, isSkybox, isExternal);
            changed = true;
        }

        return changed;
    }

    private bool RefreshWmoInstanceBounds(
        List<ObjectInstance> instances,
        (int tileX, int tileY)? tileKey,
        bool isSkybox,
        bool isExternal)
    {
        bool changed = false;

        for (int i = 0; i < instances.Count; i++)
        {
            var inst = instances[i];
            if (inst.BoundsResolved)
                continue;

            if (!_assets.TryGetWmoPlacementBounds(inst.ModelKey, out var localMin, out var localMax))
                continue;

            TransformBounds(localMin, localMax, inst.Transform, out var worldMin, out var worldMax);
            inst.LocalBoundsMin = localMin;
            inst.LocalBoundsMax = localMax;
            inst.BoundsMin = worldMin;
            inst.BoundsMax = worldMax;
            inst.BoundsResolved = true;
            instances[i] = inst;
            UpdateSceneGraphPlacementBounds(tileKey, WorldSceneNodeKind.WmoPlacement, i, inst, isSkybox, isExternal);
            changed = true;
        }

        return changed;
    }

    private void UpdateSceneGraphPlacementBounds(
        (int tileX, int tileY)? tileKey,
        WorldSceneNodeKind kind,
        int instanceIndex,
        in ObjectInstance instance,
        bool isSkybox,
        bool isExternal)
    {
        if (!UseHierarchicalSceneTraversal || _sceneGraphBuild is null)
            return;

        string kindToken = GetSceneGraphKindToken(kind, isSkybox);
        string sourceToken = isExternal
            ? "external"
            : tileKey.HasValue
                ? $"tile/{tileKey.Value.tileX:D2}/{tileKey.Value.tileY:D2}"
                : string.Empty;
        if (string.IsNullOrEmpty(sourceToken))
            return;

        string placementId = $"world/object/{kindToken}/{sourceToken}/{instanceIndex:D6}";
        if (!_sceneGraphBuild.TryGetGraphForPlacement(placementId, out WorldSceneGraphBuildResult? graph)
            || graph is null
            || !graph.Graph.TryGetNode(placementId, out WorldSceneNode? node)
            || node is null)
        {
            return;
        }

        // Bounds promotion changes the placement payload, not the scene topology. Update only
        // this placement node; the streaming-safe node API deliberately avoids refreshing every
        // sibling branch, and the authoritative tile root remains unchanged.
        graph.TryUpdatePlacementInstance(placementId, instance);
        node.UpdateLocalBoundsForStreaming(instance.LocalBoundsMin, instance.LocalBoundsMax, instance.BoundsResolved);
    }

    private static int FindPlacementInstanceIndex(List<ObjectInstance> instances, ObjectInstance current)
    {
        for (int index = 0; index < instances.Count; index++)
        {
            ObjectInstance candidate = instances[index];
            if (candidate.UniqueId == current.UniqueId
                && candidate.PlacementEntryIndex == current.PlacementEntryIndex
                && candidate.TileX == current.TileX
                && candidate.TileY == current.TileY)
            {
                return index;
            }
        }

        return -1;
    }

    private ObjectInstance MovePlacementInstance(ObjectInstance current, Vector3 newPosition, ObjectType objectType)
    {
        Vector3 delta = newPosition - current.PlacementPosition;
        current.PlacementPosition = newPosition;

        switch (objectType)
        {
            case ObjectType.Mdx:
            {
                float rx = -current.PlacementRotation.Y * MathF.PI / 180f;
                float ry = -current.PlacementRotation.X * MathF.PI / 180f;
                float rz = current.PlacementRotation.Z * MathF.PI / 180f;
                var transform = Matrix4x4.CreateRotationZ(MathF.PI)
                    * Matrix4x4.CreateScale(current.PlacementScale)
                    * Matrix4x4.CreateRotationX(rx)
                    * Matrix4x4.CreateRotationY(ry)
                    * Matrix4x4.CreateRotationZ(rz)
                    * Matrix4x4.CreateTranslation(newPosition);

                current.Transform = transform;
                if (_assets.TryGetMdxBounds(current.ModelKey, out Vector3 localMin, out Vector3 localMax))
                {
                    current.LocalBoundsMin = localMin;
                    current.LocalBoundsMax = localMax;
                    current.BoundsResolved = true;
                    TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);
                    current.BoundsMin = worldMin;
                    current.BoundsMax = worldMax;
                }
                else
                {
                    current.BoundsMin += delta;
                    current.BoundsMax += delta;
                    current.BoundsResolved = false;
                }

                return current;
            }

            case ObjectType.Wmo:
            {
                float rx = current.PlacementRotation.X * MathF.PI / 180f;
                float ry = current.PlacementRotation.Y * MathF.PI / 180f;
                float rz = current.PlacementRotation.Z * MathF.PI / 180f;
                var transform = Matrix4x4.CreateRotationZ(MathF.PI)
                    * Matrix4x4.CreateRotationX(rx)
                    * Matrix4x4.CreateRotationY(ry)
                    * Matrix4x4.CreateRotationZ(rz)
                    * Matrix4x4.CreateTranslation(newPosition);

                current.Transform = transform;
                if (_assets.TryGetWmoPlacementBounds(current.ModelKey, out Vector3 localMin, out Vector3 localMax))
                {
                    current.LocalBoundsMin = localMin;
                    current.LocalBoundsMax = localMax;
                    current.BoundsResolved = true;
                    TransformBounds(localMin, localMax, transform, out Vector3 worldMin, out Vector3 worldMax);
                    current.BoundsMin = worldMin;
                    current.BoundsMax = worldMax;
                }
                else
                {
                    current.BoundsMin += delta;
                    current.BoundsMax += delta;
                    current.BoundsResolved = false;
                }

                return current;
            }

            default:
                return current;
        }
    }

    private void UpdateAdapterPlacementPosition(ObjectType objectType, ObjectInstance current, Vector3 newPosition)
    {
        switch (objectType)
        {
            case ObjectType.Mdx:
                UpdateMddfPlacementPosition(current, newPosition);
                break;

            case ObjectType.Wmo:
                UpdateModfPlacementPosition(current, newPosition);
                break;
        }
    }

    private void UpdateMddfPlacementPosition(ObjectInstance current, Vector3 newPosition)
    {
        List<MddfPlacement> placements = _terrainManager.Adapter.MddfPlacements;
        int index = FindPlacementIndexByUniqueIdAndPosition(placements, current.UniqueId, current.PlacementPosition);
        if (index < 0)
            index = FindPlacementIndexByUniqueId(placements, current.UniqueId);
        if (index < 0)
            return;

        MddfPlacement updated = placements[index];
        updated.Position = newPosition;
        placements[index] = updated;
    }

    private void UpdateModfPlacementPosition(ObjectInstance current, Vector3 newPosition)
    {
        List<ModfPlacement> placements = _terrainManager.Adapter.ModfPlacements;
        int index = FindPlacementIndexByUniqueIdAndPosition(placements, current.UniqueId, current.PlacementPosition);
        if (index < 0)
            index = FindPlacementIndexByUniqueId(placements, current.UniqueId);
        if (index < 0)
            return;

        ModfPlacement updated = placements[index];
        Vector3 delta = newPosition - updated.Position;
        updated.Position = newPosition;
        updated.BoundsMin += delta;
        updated.BoundsMax += delta;
        placements[index] = updated;
    }

    private static int FindPlacementIndexByUniqueIdAndPosition(List<MddfPlacement> placements, int uniqueId, Vector3 position)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId && Vector3.DistanceSquared(placements[index].Position, position) < 0.0001f)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueId(List<MddfPlacement> placements, int uniqueId)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueIdAndPosition(List<ModfPlacement> placements, int uniqueId, Vector3 position)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId && Vector3.DistanceSquared(placements[index].Position, position) < 0.0001f)
                return index;
        }

        return -1;
    }

    private static int FindPlacementIndexByUniqueId(List<ModfPlacement> placements, int uniqueId)
    {
        for (int index = 0; index < placements.Count; index++)
        {
            if (placements[index].UniqueId == uniqueId)
                return index;
        }

        return -1;
    }

    public void ClearExternalSpawns()
    {
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _instancesDirty = true;
    }

    public void SetExternalSpawns(IEnumerable<WorldSpawnRecord> spawns)
    {
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();

        foreach (var spawn in spawns)
        {
            if (string.IsNullOrWhiteSpace(spawn.ModelPath))
                continue;

            string modelPath = spawn.ModelPath.Replace('/', '\\');
            bool isWmo = modelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);

            string key = WorldAssetManager.NormalizeKey(modelPath);
            float orientationRadians = spawn.OrientationWowRadians;
            float yawOffsetRadians = spawn.SpawnType == WorldSpawnType.Creature ? MathF.PI : 0f;
            float finalYawRadians = orientationRadians + yawOffsetRadians;
            float finalYawDegrees = finalYawRadians * (180f / MathF.PI);
            float baseScale = spawn.EffectiveScale > 0 ? spawn.EffectiveScale : 1.0f;
            float mdxScale = baseScale;
            if (spawn.SpawnType == WorldSpawnType.GameObject)
                mdxScale *= SqlGameObjectMdxScaleMultiplier > 0 ? SqlGameObjectMdxScaleMultiplier : 1.0f;

            var pos = SqlSpawnCoordinateConverter.ToRendererPosition(spawn.PositionWow);
            var (tileX, tileY) = ComputeTileCoordinates(pos);

            if (isWmo)
            {
                var transform = Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 localMin, localMax, worldMin, worldMax;
                if (_assets.TryGetWmoPlacementBounds(key, out localMin, out localMax))
                {
                    TransformBounds(localMin, localMax, transform, out worldMin, out worldMax);
                }
                else
                {
                    localMin = localMax = Vector3.Zero;
                    worldMin = pos - new Vector3(2f);
                    worldMax = pos + new Vector3(2f);
                }

                _externalWmoInstances.Add(new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = worldMin,
                    BoundsMax = worldMax,
                    LocalBoundsMin = localMin,
                    LocalBoundsMax = localMax,
                    BoundsResolved = localMin != Vector3.Zero || localMax != Vector3.Zero,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = 1.0f,
                    UniqueId = spawn.SpawnId,
                    PlacementEntryIndex = -1,
                    TileX = tileX,
                    TileY = tileY,
                    HasTileCoordinate = true
                });
            }
            else
            {
                var transform = Matrix4x4.CreateScale(mdxScale)
                    * Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 bbMin, bbMax;
                Vector3 localMin = Vector3.Zero;
                Vector3 localMax = Vector3.Zero;
                bool boundsResolved = false;
                if (_assets.TryGetMdxBounds(key, out var modelMin, out var modelMax))
                {
                    localMin = modelMin;
                    localMax = modelMax;
                    boundsResolved = true;
                    TransformBounds(modelMin, modelMax, transform, out bbMin, out bbMax);
                }
                else
                {
                    bbMin = pos - new Vector3(2f);
                    bbMax = pos + new Vector3(2f);
                }

                var instance = new ObjectInstance
                {
                    ModelKey = key,
                    Transform = transform,
                    BoundsMin = bbMin,
                    BoundsMax = bbMax,
                    LocalBoundsMin = localMin,
                    LocalBoundsMax = localMax,
                    BoundsResolved = boundsResolved,
                    ModelName = Path.GetFileName(modelPath),
                    ModelPath = modelPath,
                    PlacementPosition = pos,
                    PlacementRotation = new Vector3(0f, 0f, finalYawDegrees),
                    PlacementScale = mdxScale,
                    UniqueId = spawn.SpawnId,
                    PlacementEntryIndex = -1,
                    TileX = tileX,
                    TileY = tileY,
                    HasTileCoordinate = true
                };

                if (IsSkyboxModelPath(modelPath))
                    _externalSkyboxInstances.Add(instance);
                else
                    _externalMdxInstances.Add(instance);
            }
        }

        ViewerLog.Info(ViewerLog.Category.Terrain,
            $"SQL spawns injected: {_externalMdxInstances.Count} MDX, {_externalSkyboxInstances.Count} skybox, {_externalWmoInstances.Count} WMO");

        _instancesDirty = true;
    }

    /// <summary>
    /// Transform an axis-aligned bounding box through a matrix by transforming all 8 corners
    /// and computing the new AABB that encloses them.
    /// </summary>
    private static void TransformBounds(Vector3 min, Vector3 max, Matrix4x4 m, out Vector3 outMin, out Vector3 outMax)
    {
        outMin = new Vector3(float.MaxValue);
        outMax = new Vector3(float.MinValue);
        Span<float> xs = stackalloc float[] { min.X, max.X };
        Span<float> ys = stackalloc float[] { min.Y, max.Y };
        Span<float> zs = stackalloc float[] { min.Z, max.Z };
        foreach (var x in xs)
        foreach (var y in ys)
        foreach (var z in zs)
        {
            var p = Vector3.Transform(new Vector3(x, y, z), m);
            outMin = Vector3.Min(outMin, p);
            outMax = Vector3.Max(outMax, p);
        }
    }

    private static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float dx = point.X < min.X ? min.X - point.X : point.X > max.X ? point.X - max.X : 0f;
        float dy = point.Y < min.Y ? min.Y - point.Y : point.Y > max.Y ? point.Y - max.Y : 0f;
        float dz = point.Z < min.Z ? min.Z - point.Z : point.Z > max.Z ? point.Z - max.Z : 0f;
        return dx * dx + dy * dy + dz * dz;
    }

    private static float ComputeNoCullDistanceSq(Vector3 min, Vector3 max)
    {
        float halfDiagonal = (max - min).Length() * 0.5f;
        float graceRadius = MathF.Max(NoCullRadius, MathF.Min(halfDiagonal + 96f, 1024f));
        return graceRadius * graceRadius;
    }

    private static float ComputeObjectFogStart(float fogStart, float fogEnd)
    {
        if (fogEnd <= 0f)
            return fogStart;

        float delayedStart = fogEnd * 0.6f;
        return MathF.Min(fogEnd - 64f, MathF.Max(fogStart, delayedStart));
    }

    private static (float start, float end) ComputeObjectFogRange(float fogStart, float fogEnd, bool enabled)
    {
        if (enabled)
            return (ComputeObjectFogStart(fogStart, fogEnd), fogEnd);

        float disabledStart = MathF.Max(fogEnd, fogStart) + 100000f;
        return (disabledStart, disabledStart + 1f);
    }

    private static float ComputeWmoCullDistance(float fogEnd, float rangeMultiplier)
    {
        float clampedMultiplier = Math.Clamp(rangeMultiplier, 0.25f, 4.0f);
        if (fogEnd <= 0f)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Min(WmoCullDistance, MaxWorldObjectViewDistance) * clampedMultiplier);

        float baseDistance = MathF.Min(MaxWorldObjectViewDistance, MathF.Max(WmoCullDistance, fogEnd + 256f));
        return MathF.Min(MaxWorldObjectViewDistance, baseDistance * clampedMultiplier);
    }

    private static float ComputeMdxCullDistance(float fogEnd, float boundsDiagonal, bool isTaxiActor, float rangeMultiplier)
    {
        float clampedMultiplier = Math.Clamp(rangeMultiplier, 0.25f, 4.0f);
        if (isTaxiActor)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Max(1024f, fogEnd + 384f) * clampedMultiplier);

        if (fogEnd <= 0f)
            return MathF.Min(MaxWorldObjectViewDistance, MathF.Min(DoodadCullDistance, MaxWorldObjectViewDistance) * clampedMultiplier);

        float objectAllowance = MathF.Min(512f, boundsDiagonal * 0.5f + 96f);
        float baseDistance = MathF.Min(DoodadCullDistance, MathF.Max(1024f, fogEnd + objectAllowance));
        return MathF.Min(MaxWorldObjectViewDistance, baseDistance * clampedMultiplier);
    }

    private void AccumulateUniqueIdFilterRange(
        IReadOnlyList<ObjectInstance> instances,
        ref int minUniqueId,
        ref int maxUniqueId,
        ref int instanceCount)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance inst = instances[i];
            if (inst.UniqueId <= 0 || !MatchesUniqueIdFilterScope(inst))
                continue;

            minUniqueId = Math.Min(minUniqueId, inst.UniqueId);
            maxUniqueId = Math.Max(maxUniqueId, inst.UniqueId);
            instanceCount++;
        }
    }

    private void AccumulateUniqueIdLayerCandidates(
        IReadOnlyList<ObjectInstance> instances,
        bool isWmo,
        SortedDictionary<int, (int wmoCount, int mdxCount)> countsById)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance inst = instances[i];
            if (inst.UniqueId <= 0 || !MatchesUniqueIdFilterScope(inst))
                continue;

            countsById.TryGetValue(inst.UniqueId, out (int wmoCount, int mdxCount) counts);
            counts = isWmo
                ? (counts.wmoCount + 1, counts.mdxCount)
                : (counts.wmoCount, counts.mdxCount + 1);
            countsById[inst.UniqueId] = counts;
        }
    }

    private bool MatchesUniqueIdFilterScope(in ObjectInstance inst)
    {
        if (_uniqueIdVisibilityScope != UniqueIdVisibilityScope.CameraTile)
            return true;

        if (!_uniqueIdFilterTile.HasValue || !inst.HasTileCoordinate)
            return false;

        return inst.TileX == _uniqueIdFilterTile.Value.tileX
            && inst.TileY == _uniqueIdFilterTile.Value.tileY;
    }

    private bool ShouldHideObjectInstanceByUniqueId(in ObjectInstance inst)
    {
        if (ShouldHideObjectInstanceByPathFilter(inst))
            return true;

        if (!_uniqueIdFilterEnabled
            || _uniqueIdFilterMin < 0
            || _uniqueIdFilterMax < 0
            || inst.UniqueId <= 0
            || !MatchesUniqueIdFilterScope(inst))
        {
            return false;
        }

        int minUniqueId = Math.Min(_uniqueIdFilterMin, _uniqueIdFilterMax);
        int maxUniqueId = Math.Max(_uniqueIdFilterMin, _uniqueIdFilterMax);
        return inst.UniqueId < minUniqueId || inst.UniqueId > maxUniqueId;
    }

    private bool ShouldHideObjectInstanceByPathFilter(in ObjectInstance inst)
    {
        if (!_objectPathFiltersEnabled || _objectPathFilters.Count == 0 || string.IsNullOrWhiteSpace(inst.ModelPath))
            return false;

        string normalizedPath = ObjectPathFilterEntry.NormalizePrefix(inst.ModelPath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return false;

        for (int i = 0; i < _objectPathFilters.Count; i++)
        {
            ObjectPathFilterEntry entry = _objectPathFilters[i];
            if (entry.MatchesModelPath(normalizedPath))
                return true;
        }

        return false;
    }

    private bool ShouldHideVisibleMdxInstance(in ObjectInstance inst)
    {
        if (ShouldHideObjectInstanceByUniqueId(inst))
            return true;

        if (_maxVisibleMdxBoundsHeight > 0f)
        {
            float boundsHeight = MathF.Abs(inst.BoundsMax.Z - inst.BoundsMin.Z);
            if (float.IsFinite(boundsHeight) && boundsHeight > _maxVisibleMdxBoundsHeight)
                return true;
        }

        return _hideTerrainOccludedMdx && IsMdxFullyOccludedByTerrain(inst);
    }

    private bool IsMdxFullyOccludedByTerrain(in ObjectInstance inst)
    {
        if (!TrySampleLoadedTerrainHeight(inst.PlacementPosition.X, inst.PlacementPosition.Y, out float terrainHeight))
            return false;

        float objectTop = MathF.Max(inst.BoundsMin.Z, inst.BoundsMax.Z);
        if (!float.IsFinite(objectTop))
            return false;

        const float terrainOcclusionMargin = 1.0f;
        return terrainHeight >= objectTop + terrainOcclusionMargin;
    }

    /// <summary>
    /// Resolves a camera-path sample against the loaded world. Terrain collision is
    /// heightfield-only; WMO collision uses the resident placement bounds as a
    /// conservative sweep volume. Both are deliberately opt-in because the viewer
    /// also supports free-fly inspection through geometry.
    /// </summary>
    public bool TryResolveCameraPathCollision(
        Vector3 previousPosition,
        Vector3 desiredPosition,
        float clearance,
        bool terrainCollision,
        bool wmoCollision,
        out Vector3 resolvedPosition)
    {
        resolvedPosition = desiredPosition;
        float safeClearance = float.IsFinite(clearance) ? Math.Clamp(clearance, 0f, 32f) : 0f;
        bool collided = false;

        if (terrainCollision && TrySampleLoadedTerrainHeight(desiredPosition.X, desiredPosition.Y, out float terrainHeight))
        {
            float minimumCameraZ = terrainHeight + safeClearance;
            if (resolvedPosition.Z < minimumCameraZ)
            {
                resolvedPosition.Z = minimumCameraZ;
                collided = true;
            }
        }

        if (wmoCollision)
        {
            if (_instancesDirty)
                RebuildInstanceLists();

            Vector3 segmentStart = previousPosition;
            Vector3 segmentEnd = resolvedPosition;
            foreach (ObjectInstance instance in _wmoInstances)
            {
                if (!AreFiniteOrderedBounds(instance.BoundsMin, instance.BoundsMax))
                    continue;

                Vector3 boundsMin = instance.BoundsMin - new Vector3(safeClearance);
                Vector3 boundsMax = instance.BoundsMax + new Vector3(safeClearance);
                if (!TrySegmentAabb(segmentStart, segmentEnd, boundsMin, boundsMax, out float entryT))
                    continue;

                bool startInside = IsPointInsideAabb(segmentStart, boundsMin, boundsMax);
                // A placement AABB is an exterior shell, not an indoor collision mesh.
                // Preserve paths that start inside a WMO instead of ejecting them from
                // the entire building; only stop an outside-to-inside sweep here.
                if (startInside)
                    continue;

                if (entryT > 0f)
                {
                    float stopT = Math.Clamp(entryT - 0.0025f, 0f, 1f);
                    resolvedPosition = Vector3.Lerp(segmentStart, segmentEnd, stopT);
                }
                else if (IsPointInsideAabb(segmentEnd, boundsMin, boundsMax))
                    resolvedPosition = segmentStart;

                collided = true;
                segmentEnd = resolvedPosition;
            }
        }

        return collided;
    }

    private static bool IsPointInsideAabb(Vector3 point, Vector3 min, Vector3 max)
        => point.X >= min.X && point.X <= max.X
            && point.Y >= min.Y && point.Y <= max.Y
            && point.Z >= min.Z && point.Z <= max.Z;

    private static bool TrySegmentAabb(Vector3 start, Vector3 end, Vector3 min, Vector3 max, out float entryT)
    {
        entryT = 0f;
        float exitT = 1f;
        Vector3 delta = end - start;
        for (int axis = 0; axis < 3; axis++)
        {
            float origin = start[axis];
            float direction = delta[axis];
            float axisMin = min[axis];
            float axisMax = max[axis];
            if (MathF.Abs(direction) < 0.000001f)
            {
                if (origin < axisMin || origin > axisMax)
                    return false;
                continue;
            }

            float inverse = 1f / direction;
            float near = (axisMin - origin) * inverse;
            float far = (axisMax - origin) * inverse;
            if (near > far)
                (near, far) = (far, near);
            entryT = MathF.Max(entryT, near);
            exitT = MathF.Min(exitT, far);
            if (entryT > exitT)
                return false;
        }

        return entryT >= 0f && entryT <= 1f;
    }

    private bool TrySampleLoadedTerrainHeight(float worldX, float worldY, out float height)
    {
        height = 0f;

        return TrySampleLoadedTerrainHeight(_terrainManager, _terrainManager.Renderer, worldX, worldY, out height);
    }

    private static bool TrySampleLoadedTerrainHeight(TerrainManager terrainManager, TerrainRenderer renderer, float worldX, float worldY, out float height)
    {
        height = 0f;

        TerrainRenderer.TerrainChunkInfo? chunkInfo = renderer.GetChunkInfoAt(worldX, worldY);
        if (!chunkInfo.HasValue)
            return false;

        if (!terrainManager.TryGetTileLoadResult(chunkInfo.Value.TileX, chunkInfo.Value.TileY, out TileLoadResult tile))
            return false;

        TerrainChunkData? chunk = tile.Chunks.FirstOrDefault(c => c.ChunkX == chunkInfo.Value.ChunkX && c.ChunkY == chunkInfo.Value.ChunkY);
        if (chunk == null || chunk.Heights == null || chunk.Heights.Length < 145)
            return false;

        float localX = chunk.WorldPosition.Y - worldY;
        float localY = chunk.WorldPosition.X - worldX;
        localX = Math.Clamp(localX, 0f, WoWConstants.ChunkSize);
        localY = Math.Clamp(localY, 0f, WoWConstants.ChunkSize);
        height = SampleHeightOuterGrid(chunk, localX, localY);
        return true;
    }

    private static float SampleHeightOuterGrid(TerrainChunkData chunk, float localX, float localY)
    {
        if (chunk.Heights == null || chunk.Heights.Length < 145)
            return chunk.WorldPosition.Z;

        float cellSize = WoWConstants.ChunkSize / 16f;
        float subCellSize = cellSize / 8f;

        Span<float> grid = stackalloc float[9 * 9];
        grid.Clear();

        for (int i = 0; i < 145; i++)
        {
            GetChunkVertexPosition(i, out int row, out int col, out bool isInner);
            if (isInner)
                continue;

            int gridY = row / 2;
            if ((uint)gridY >= 9u || (uint)col >= 9u)
                continue;

            grid[(gridY * 9) + col] = chunk.Heights[i];
        }

        float gridX = localX / subCellSize;
        float gridYFloat = localY / subCellSize;
        int ix = Math.Clamp((int)MathF.Floor(gridX), 0, 7);
        int iy = Math.Clamp((int)MathF.Floor(gridYFloat), 0, 7);
        float fx = Math.Clamp(gridX - ix, 0f, 1f);
        float fy = Math.Clamp(gridYFloat - iy, 0f, 1f);

        float h00 = grid[(iy * 9) + ix];
        float h10 = grid[(iy * 9) + (ix + 1)];
        float h01 = grid[((iy + 1) * 9) + ix];
        float h11 = grid[((iy + 1) * 9) + (ix + 1)];

        float h0 = h00 + ((h10 - h00) * fx);
        float h1 = h01 + ((h11 - h01) * fx);
        return h0 + ((h1 - h0) * fy);
    }

    private static void GetChunkVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2 == 1);
                return;
            }

            remaining -= rowSize;
        }
    }

    private static (int tileX, int tileY) ComputeTileCoordinates(Vector3 rendererPosition)
    {
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - rendererPosition.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - rendererPosition.Y) / WoWConstants.ChunkSize);
        return (tileX, tileY);
    }

    private void PrepareSceneGraphFrameVisibility(
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        float verticalFieldOfViewRadians)
    {
        if (_sceneGraphFrameVisibilityPrepared)
            return;

        _sceneGraphVisibleMdxInstances.Clear();
        _sceneGraphVisibleWmoInstances.Clear();
        _sceneGraphPortalVisibility.Clear();
        _lastSceneGraphTraversalDiagnostics = new WorldSceneTraversalDiagnostics();

        if (_sceneGraphBuild is null)
        {
            _sceneGraphFrameVisibilityPrepared = true;
            return;
        }

        foreach ((string placementId, WorldScenePortalAdapterResult adapter) in _sceneGraphPortalAdapters)
        {
            if (_sceneGraphBuild.TryGetGraphForPlacement(placementId, out WorldSceneGraphBuildResult? placementGraph)
                && placementGraph.Graph.TryGetNode(placementId, out WorldSceneNode? placementNode))
            {
                _sceneGraphPortalVisibility[placementId] = WorldScenePortalVisibilityEvaluator.Evaluate(
                    adapter,
                    placementNode,
                    cameraPos,
                    maximumDepth: 4);
            }
        }

        foreach (WorldSceneGraphBuildResult graphBuild in _sceneGraphBuild.EnumerateGraphs())
        {
            WorldSceneTraversalResult traversal = WorldSceneTraversal.Traverse(
                graphBuild.Graph,
                IsSceneGraphNodeVisible,
                node => node.Kind is WorldSceneNodeKind.M2Placement or WorldSceneNodeKind.WmoPlacement,
                shouldEvaluateVisibility: static node =>
                    node.Kind != WorldSceneNodeKind.M2Placement
                    || node.Parent?.Kind != WorldSceneNodeKind.Chunk,
                validateGraph: false);
            _lastSceneGraphTraversalDiagnostics.Accumulate(traversal.Diagnostics);

            foreach (WorldSceneNode node in traversal.VisibleNodes)
            {
                if (!graphBuild.PlacementsByNodeId.TryGetValue(node.Id, out WorldSceneGraphObjectPlacement placement)
                    || placement.IsSkybox)
                {
                    continue;
                }

                if (node.Kind == WorldSceneNodeKind.WmoPlacement)
                    _sceneGraphVisibleWmoInstances.Add(placement.Instance);
                else if (node.Kind == WorldSceneNodeKind.M2Placement)
                    _sceneGraphVisibleMdxInstances.Add(placement.Instance);
            }
        }

        _sceneGraphFrameVisibilityPrepared = true;
    }

    private bool IsSceneGraphNodeVisible(WorldSceneNode node)
    {
        if (node.Kind == WorldSceneNodeKind.WmoGroup
            && node.Parent is not null
            && _sceneGraphPortalVisibility.TryGetValue(node.Parent.Id, out WorldScenePortalVisibilityResult? portalVisibility)
            && !portalVisibility.Diagnostics.FallbackRequired
            && !portalVisibility.VisibleNodeIds.Contains(node.Id, StringComparer.Ordinal))
        {
            return false;
        }

        return _frustumCuller.TestAABB(node.WorldBoundsMin, node.WorldBoundsMax);
    }

    private void CollectVisibleWmoInstances(WorldRenderFrame frame, Vector3 cameraPos, Vector3 cameraForward, float fogEnd, float verticalFieldOfViewRadians)
    {
        if (UseHierarchicalSceneTraversal && _sceneGraphBuild is not null)
        {
            PrepareSceneGraphFrameVisibility(cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians);
            WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                frame.Visibility,
                _sceneGraphVisibleWmoInstances,
                new WorldObjectVisibilityContext(
                    cameraPos,
                    cameraForward,
                    fogEnd,
                    _objectStreamingRangeMultiplier,
                    CullSmallDoodadsOnly: false,
                    CountAsTaxiActor: false,
                    VerticalFieldOfViewRadians: verticalFieldOfViewRadians,
                    VisibilityProfile: _objectVisibilityProfile),
                inst => ShouldHideObjectInstanceByUniqueId(inst),
                (min, max) => _frustumCuller.TestAABB(min, max),
                modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore));
            return;
        }

        var context = new WorldObjectVisibilityContext(
            cameraPos,
            cameraForward,
            fogEnd,
            _objectStreamingRangeMultiplier,
            CullSmallDoodadsOnly: false,
            CountAsTaxiActor: false,
            VerticalFieldOfViewRadians: verticalFieldOfViewRadians,
            VisibilityProfile: _objectVisibilityProfile);

        WmoCulledCount = 0;
        foreach (var pair in _tileWmoInstances)
        {
            if (_tileWmoBounds.TryGetValue(pair.Key, out var bounds)
                && !ShouldVisitObjectBucket(bounds.Min, bounds.Max, cameraPos, cameraForward, fogEnd, isWmo: true, countAsTaxiActor: false))
            {
                WmoCulledCount += pair.Value.Count;
                continue;
            }

            if (!_tileWmoVisibilityBuckets.TryGetValue(pair.Key, out List<FlatVisibilityBucket>? buckets))
            {
                WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                    frame.Visibility,
                    pair.Value,
                    context,
                    inst => ShouldHideObjectInstanceByUniqueId(inst),
                    (min, max) => _frustumCuller.TestAABB(min, max),
                    modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                    (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore));
                continue;
            }

            foreach (FlatVisibilityBucket bucket in buckets)
            {
                if (!ShouldVisitFlatVisibilityBucket(bucket, cameraPos, fogEnd, isWmo: true))
                {
                    WmoCulledCount += bucket.Instances.Count;
                    continue;
                }

                WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                    frame.Visibility,
                    bucket.Instances,
                    context,
                    inst => ShouldHideObjectInstanceByUniqueId(inst),
                    (min, max) => _frustumCuller.TestAABB(min, max),
                    modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                    (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore));
            }
        }

        if (_externalWmoInstances.Count > 0)
        {
            WmoCulledCount += WorldObjectVisibilityCollector.CollectVisibleWmos(
                frame.Visibility,
                _externalWmoInstances,
                context,
                inst => ShouldHideObjectInstanceByUniqueId(inst),
                (min, max) => _frustumCuller.TestAABB(min, max),
                modelKey => ResolveVisibleWmoRenderer(frame, modelKey) != null,
                (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleWmoLoadDistances, modelKey, priorityScore));
        }
    }

    private void CollectVisibleMdxInstances(
        WorldRenderFrame frame,
        List<ObjectInstance> instances,
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        float verticalFieldOfViewRadians,
        bool cullSmallDoodadsOnly,
        bool countAsTaxiActor)
    {
        MdxCulledCount += WorldObjectVisibilityCollector.CollectVisibleMdx(
            frame.Visibility,
            instances,
            new WorldObjectVisibilityContext(
                cameraPos,
                cameraForward,
                fogEnd,
                _objectStreamingRangeMultiplier,
                cullSmallDoodadsOnly,
                countAsTaxiActor,
                verticalFieldOfViewRadians,
                _objectVisibilityProfile),
            inst => ShouldHideVisibleMdxInstance(inst),
            (min, max) => _frustumCuller.TestAABB(min, max),
            modelKey => ResolveVisibleMdxRenderer(frame, modelKey) != null,
            (modelKey, priorityScore) => TrackPendingVisibleLoad(_pendingVisibleMdxLoadDistances, modelKey, priorityScore));
    }

    private void CollectVisibleMdxBuckets(
        WorldRenderFrame frame,
        Vector3 cameraPos,
        Vector3 cameraForward,
        float fogEnd,
        float verticalFieldOfViewRadians)
    {
        if (UseHierarchicalSceneTraversal && _sceneGraphBuild is not null)
        {
            PrepareSceneGraphFrameVisibility(cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians);
            CollectVisibleMdxInstances(
                frame,
                _sceneGraphVisibleMdxInstances,
                cameraPos,
                cameraForward,
                fogEnd,
                verticalFieldOfViewRadians,
                cullSmallDoodadsOnly: true,
                countAsTaxiActor: false);
            return;
        }

        foreach (var pair in _tileMdxInstances)
        {
            if (_tileMdxBounds.TryGetValue(pair.Key, out var bounds)
                && AreMdxTileBoundsResolved(pair.Value)
                && !ShouldVisitObjectBucket(bounds.Min, bounds.Max, cameraPos, cameraForward, fogEnd, isWmo: false, countAsTaxiActor: false))
            {
                MdxCulledCount += pair.Value.Count;
                continue;
            }

            if (!_tileMdxVisibilityBuckets.TryGetValue(pair.Key, out List<FlatVisibilityBucket>? buckets))
            {
                CollectVisibleMdxInstances(frame, pair.Value, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: true, countAsTaxiActor: false);
                continue;
            }

            foreach (FlatVisibilityBucket bucket in buckets)
            {
                if (!ShouldVisitFlatVisibilityBucket(bucket, cameraPos, fogEnd, isWmo: false))
                {
                    MdxCulledCount += bucket.Instances.Count;
                    continue;
                }

                CollectVisibleMdxInstances(frame, bucket.Instances, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: true, countAsTaxiActor: false);
            }
        }

        if (_externalMdxInstances.Count > 0)
            CollectVisibleMdxInstances(frame, _externalMdxInstances, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: true, countAsTaxiActor: false);
    }

    private static float ExtractVerticalFieldOfViewRadians(Matrix4x4 projection)
    {
        float inverseTanHalfFov = projection.M22;
        if (!float.IsFinite(inverseTanHalfFov) || inverseTanHalfFov <= 1e-6f)
            return MathF.PI / 3f;

        return 2f * MathF.Atan(1f / inverseTanHalfFov);
    }

    private static Vector3 ExtractCameraForward(Matrix4x4 viewInverse)
    {
        Vector3 forward = Vector3.TransformNormal(-Vector3.UnitZ, viewInverse);
        float lengthSq = forward.LengthSquared();
        if (lengthSq <= 1e-6f)
            return Vector3.UnitY;

        return forward / MathF.Sqrt(lengthSq);
    }

    private static float ComputeVisionConeFactor(Vector3 cameraPos, Vector3 cameraForward, Vector3 targetPos, float targetDistanceSq)
    {
        if (targetDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        float forwardLengthSq = cameraForward.LengthSquared();
        if (forwardLengthSq <= 1e-6f)
            return 1.0f;

        Vector3 toTarget = targetPos - cameraPos;
        float toTargetLengthSq = toTarget.LengthSquared();
        if (toTargetLengthSq <= 1e-6f)
            return 1.0f;

        float invTargetLength = 1.0f / MathF.Sqrt(toTargetLengthSq);
        float alignment = Vector3.Dot(toTarget * invTargetLength, cameraForward);
        float factor = (alignment - VisionConeRearDot) / MathF.Max(0.001f, VisionConeFrontDot - VisionConeRearDot);
        return Math.Clamp(factor, 0.0f, 1.0f);
    }

    private static float ComputeConeCullDistance(float baseCullDistance, float coneFactor)
    {
        if (baseCullDistance <= 0f)
            return ObjectNearHoldRadius;

        float scale = RearConeCullFraction + (1.0f - RearConeCullFraction) * coneFactor;
        return MathF.Max(ObjectNearHoldRadius, baseCullDistance * scale);
    }

    private static float ComputeConeFade(float coneFactor, float centerDistanceSq)
    {
        if (centerDistanceSq <= ObjectNearHoldRadiusSq)
            return 1.0f;

        return RearConeFadeFloor + (1.0f - RearConeFadeFloor) * coneFactor;
    }

    private static float ComputeLoadPriorityScore(float centerDistanceSq, float coneFactor)
    {
        float penalty = RearConeLoadPenalty - (RearConeLoadPenalty - 1.0f) * coneFactor;
        return centerDistanceSq * penalty;
    }

    private static double MeasureDurationMs(Action action)
    {
        var stageTimer = Stopwatch.StartNew();
        action();
        return stageTimer.Elapsed.TotalMilliseconds;
    }

    private void FinalizeRenderFrameStats(WorldRenderFrame frame, Stopwatch frameTimer)
    {
        int terrainChunksRendered = _terrainManager.Renderer.ChunksRendered;
        int terrainChunksCulled = _terrainManager.Renderer.ChunksCulled;
        int wdlVisibleTiles = _wdlTerrain?.VisibleTiles ?? 0;
        int wdlHiddenTiles = _wdlTerrain?.HiddenTiles ?? 0;
        LastRenderFrameStats = frame.ToStats(
            frameTimer.Elapsed.TotalMilliseconds,
            _assets.PendingAssetLoadCount,
            terrainChunksRendered,
            terrainChunksCulled,
            wdlVisibleTiles,
            wdlHiddenTiles);
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    private bool _renderDiagPrinted = false;
    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        WorldRenderFrame frame = _renderFrame;
        frame.Reset();
        var frameTimer = Stopwatch.StartNew();
        _pendingVisibleMdxLoadDistances.Clear();
        _pendingVisibleWmoLoadDistances.Clear();
        _sceneGraphFrameVisibilityPrepared = false;
        _sceneGraphVisibleMdxInstances.Clear();
        _sceneGraphVisibleWmoInstances.Clear();

        frame.SceneMaintenanceMs = MeasureDurationMs(() =>
        {
            TryFinalizePm4OverlayLoad();

            // Rebuild flat instance lists if tiles changed.
            if (_instancesDirty)
                RebuildInstanceLists();
            else if (UseHierarchicalSceneTraversal && _sceneGraphBuild is null)
                RebuildSceneGraphObjectIndex();
        });

        frame.DeferredAssetLoadMs = MeasureDurationMs(() =>
        {
            ProcessDeferredAssetLoads();
            _assets.ProcessDeferredWmoDoodadLoads();
        });
        frame.TaxiActorUpdateMs = MeasureDurationMs(UpdateTaxiActorInstances);

        // Extract camera position for sky dome
        Matrix4x4.Invert(view, out var viewInvSky);
        var camPos = new Vector3(viewInvSky.M41, viewInvSky.M42, viewInvSky.M43);
        var lighting = _terrainManager.Lighting;
        Vector3 fogColor;
        float fogStart;
        float fogEnd;

        // 0. Resolve frame lighting before any world pass so terrain, WDL, liquids,
        // skybackdrops, WMOs, and MDXs all sample one lighting state.
        fogColor = Vector3.Zero;
        fogStart = 0f;
        fogEnd = 0f;
        float objectFogStart = 0f;
        float objectFogEnd = 0f;
        Vector3 cameraPos = Vector3.Zero;
        Vector3 cameraForward = Vector3.UnitZ;
        float verticalFieldOfViewRadians = ExtractVerticalFieldOfViewRadians(proj);
        double overlayElapsedMs = 0;
        double objectWireframeMs = 0;
        int objectWireframePreparedCount = 0;
        int objectWireframeSubmittedCount = 0;
        bool objectWireframeEnabled = false;
        double selectionBoundsMs = 0;
        int selectionBoundsPreparedCount = 0;
        double pm4BoundsMs = 0;
        int pm4BoundsPreparedCount = 0;
        double pm4GeometryPrepareMs = 0;
        double pm4GeometrySubmitMs = 0;
        double pm4NodesMs = 0;
        int pm4GeometryPreparedCount = 0;
        int pm4GeometrySubmittedCount = 0;
        int pm4NodesPreparedCount = 0;
        int poiTaxiPreparedCount = 0;
        double poiTaxiMs = 0;
        int areaTriggerPreparedCount = 0;
        double areaTriggersMs = 0;

        bool continuedPastTerrain = WorldFramePassCoordinator.Execute(
            new WorldFramePassOptions(_objectsVisible, _wmosVisible, _doodadsVisible),
            new WorldFramePasses(
                () =>
                {
                    frame.LightingMs = MeasureDurationMs(() =>
                    {
                        LitLoader.LitLightingSample? litSample = null;
                        string fogRecommendationSource;
                        _lightService?.Update(camPos);
                        UpdateActiveSkyboxModel();
                        if (_lightService != null && !lighting.HasManualGameTimeOverride)
                            lighting.GameTime = Math.Clamp(_lightService.TimeOfDay / 2880f, 0f, 1f);

                        if (_litLoader != null && _litLoader.HasData)
                            litSample = _litLoader.EvaluateLighting(camPos, lighting.GameTime);

                        // The viewer global sun is unconditional. DBC/LightData colors and fog
                        // are spatial overlays; a missing record is therefore an identity case,
                        // never a reason to darken the terrain or retain a departed zone's fog.
                        RestoreGlobalViewerFogRange(lighting);
                        lighting.ClearExternalLighting();
                        lighting.Update();
                        _skyDome.UpdateFromLighting(lighting.GameTime, lighting.LightDirection);
                        fogRecommendationSource = "Global viewer light";

                        if (_useLocalDbcLightingOverlay
                            && _lightService is { HasActiveLocalOverlay: true } localLighting)
                        {
                            (float dbcFogStart, float dbcFogEnd) =
                                TerrainLightingMath.ComputeClientFogRange(
                                    localLighting.FogEnd,
                                    localLighting.FogScaler);

                            var globalState = new TerrainViewerLightingState(
                                lighting.LightColor,
                                lighting.AmbientColor,
                                lighting.FogColor,
                                lighting.FogStart,
                                lighting.FogEnd);
                            var localState = new TerrainViewerLightingState(
                                localLighting.DirectColor,
                                localLighting.AmbientColor,
                                localLighting.FogColor,
                                dbcFogStart,
                                dbcFogEnd);
                            TerrainViewerLightingState composed =
                                TerrainViewerLightingComposer.ComposeGlobalWithLocal(
                                    globalState,
                                    localState,
                                    localLighting.ActiveLocalWeight);

                            Vector3 globalSkyTop = _skyDome.ZenithColor;
                            Vector3 globalSkyHorizon = _skyDome.HorizonColor;
                            lighting.ApplyExternalLighting(
                                composed.DirectionalColor,
                                composed.AmbientColor,
                                composed.FogColor);
                            lighting.FogStart = composed.FogStart;
                            lighting.FogEnd = composed.FogEnd;
                            lighting.Update();
                            fogRecommendationSource = "Global viewer light + local DBC overlay";

                            _skyDome.ZenithColor = Vector3.Lerp(
                                globalSkyTop,
                                localLighting.SkyTopColor,
                                localLighting.ActiveLocalWeight);
                            _skyDome.HorizonColor = Vector3.Lerp(
                                globalSkyHorizon,
                                lighting.FogColor,
                                localLighting.ActiveLocalWeight);
                            _skyDome.SkyFogColor = lighting.FogColor;
                        }

                        if (_useLitFogOverride && litSample != null)
                        {
                            CapturePreLitFogRange(lighting);
                            // LIT tracks 0/1/7 are the global diffuse, ambient, and fog colors.
                            // Apply the profile as one coherent source; silently mixing DBC colors
                            // with LIT fog produced a profile that no client file actually authored.
                            lighting.ApplyExternalLighting(
                                litSample.DirectColor,
                                litSample.AmbientColor,
                                litSample.FogColor);
                            lighting.FogStart = litSample.FogStart;
                            lighting.FogEnd = litSample.FogEnd;
                            lighting.Update();
                            fogRecommendationSource = "LIT lighting";

                            _skyDome.ZenithColor = litSample.SkyTopColor;
                            _skyDome.HorizonColor = litSample.SkyHorizonColor;
                            _skyDome.SkyFogColor = litSample.FogColor;
                        }

                        ResolveActiveFogRange(lighting, fogRecommendationSource);
                        _skyDome.UpdateFromLighting(lighting.GameTime, lighting.LightDirection);
                        fogColor = lighting.FogColor;
                        fogStart = lighting.FogStart;
                        fogEnd = lighting.FogEnd;
                        _lastLitSample = litSample;
                    });

                    _lastHoverPickFogEnd = fogEnd;
                    (objectFogStart, objectFogEnd) = ComputeObjectFogRange(fogStart, fogEnd, _objectFogEnabled);
                },
                () =>
                {
                    if (!ShowSky)
                    {
                        frame.SkyMs = 0;
                        return;
                    }

                    frame.SkyMs = MeasureDurationMs(() => _skyDome.Render(view, proj, camPos));
                },
                () =>
                {
                    if (!ShowSky)
                    {
                        _gl.ClearColor(0f, 0f, 0f, 1f);
                        frame.SkyboxBackdropMs = 0;
                        return;
                    }

                    // Also set clear color to horizon color so any gaps match the sky
                    _gl.ClearColor(_skyDome.HorizonColor.X, _skyDome.HorizonColor.Y, _skyDome.HorizonColor.Z, 1f);
                    frame.SkyboxBackdropMs = MeasureDurationMs(() => RenderSkyboxBackdrop(view, proj, camPos, fogColor, fogStart, fogEnd, lighting));
                },
                () =>
                {
                    // 0. Render WDL low-res terrain (far background — hidden tiles replaced by detailed ADTs)
                    frame.WdlMs = MeasureDurationMs(() =>
                    {
                        if (ShowWdlTerrain && _wdlTerrain != null)
                        {
                            bool renderWdlAsOpaqueFallback = _terrainManager.LoadedTileCount == 0;
                            _wdlTerrain.Render(view, proj, camPos, _terrainManager.Lighting, _frustumCuller, renderWdlAsOpaqueFallback);
                        }
                    });
                },
                () =>
                {
                    // 1. Render terrain (with frustum culling)
                    frame.TerrainMs = MeasureDurationMs(() => _terrainManager.Render(view, proj, camPos, _frustumCuller));

                    // Reset GL state after terrain
                    _gl.DepthFunc(DepthFunction.Lequal);
                    _gl.DepthMask(true);
                    _gl.Disable(EnableCap.Blend);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.UseProgram(0); // unbind terrain shader
                },
                () =>
                {
                    // One-time render diagnostic
                    if (!_renderDiagPrinted)
                    {
                        int wmoFound = 0, wmoMissing = 0;
                        foreach (var inst in _wmoInstances)
                        {
                            if (_assets.TryGetLoadedWmo(inst.ModelKey, out _)) wmoFound++;
                            else { wmoMissing++; if (wmoMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Wmo, $"NOT FOUND: \"{inst.ModelKey}\""); }
                        }
                        int mdxFound = 0, mdxMissing = 0;
                        foreach (var inst in _mdxInstances)
                        {
                            if (_assets.TryGetLoadedMdx(inst.ModelKey, out _)) mdxFound++;
                            else { mdxMissing++; if (mdxMissing <= 3) ViewerLog.Debug(ViewerLog.Category.Mdx, $"NOT FOUND: \"{inst.ModelKey}\""); }
                        }
                        ViewerLog.Info(ViewerLog.Category.Terrain, $"Render check: WMO {wmoFound} found / {wmoMissing} missing, MDX {mdxFound} found / {mdxMissing} missing");
                    }

                    // Extract camera position from view matrix (inverse of view translation)
                    Matrix4x4.Invert(view, out var viewInv);
                    cameraPos = new Vector3(viewInv.M41, viewInv.M42, viewInv.M43);
                    cameraForward = ExtractCameraForward(viewInv);
                    _lastRenderedCameraPosition = cameraPos;
                    _hasLastRenderedCameraPosition = true;

                    EnsurePm4OverlayMatchesCameraWindow(cameraPos);

                    // Update frustum planes for culling
                    var vp = view * proj;
                    _frustumCuller.Update(vp);

                    // ── PASS 1: OPAQUE ──────────────────────────────────────────────
                    // Render all opaque geometry first with depth write ON.
                    // This ensures correct depth buffer before any transparent rendering.
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Less);
                    _gl.DepthMask(true);
                    _gl.Disable(EnableCap.Blend);

                    WmoRenderedCount = 0;
                    WmoCulledCount = 0;
                    MdxRenderedCount = 0;
                    MdxCulledCount = 0;
                },
                () =>
                {
                    frame.WmoVisibilityMs = MeasureDurationMs(() => CollectVisibleWmoInstances(frame, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians));
                    FlushPendingVisibleWmoLoads();

                    // State is constant for this pass; set once to reduce per-instance churn and
                    // keep WMO submission running through one explicit visible-instance bucket.
                    frame.WmoSubmissionMs = MeasureDurationMs(() =>
                    {
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(true);

                        var visibleWmoRenderers = new WmoRenderer?[frame.Visibility.VisibleWmos.Count];
                        var wmoBatchCandidates = new List<WorldObjectPassCoordinator.WorldWmoOpaqueBatchCandidate>(
                            frame.Visibility.VisibleWmos.Count);
                        _worldFrameWmoRenderers.Clear();
                        WmoRenderedCount = 0;
                        for (int visibleIndex = 0; visibleIndex < frame.Visibility.VisibleWmos.Count; visibleIndex++)
                        {
                            VisibleWmoInstance visible = frame.Visibility.VisibleWmos[visibleIndex];
                            WmoRenderedCount++;
                            WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visible.Instance.ModelKey);
                            visibleWmoRenderers[visibleIndex] = renderer;
                            if (renderer != null)
                                _worldFrameWmoRenderers.Add(renderer);

                            bool canBatch = renderer is IGpuInstancedWmoRenderer gpuRenderer
                                && gpuRenderer.SupportsGpuInstancedOpaque;
                            wmoBatchCandidates.Add(new(
                                visible.Instance.ModelKey,
                                canBatch,
                                visibleIndex));
                        }

                        foreach (WmoRenderer renderer in _worldFrameWmoRenderers)
                            renderer.BeginWorldFrame();

                        WorldObjectPassCoordinator.WorldWmoOpaqueBatchPlan wmoBatchPlan =
                            WorldObjectPassCoordinator.PlanOpaqueWmoBatches(wmoBatchCandidates);
                        foreach (int visibleIndex in wmoBatchPlan.FallbackVisibleIndices)
                        {
                            VisibleWmoInstance visible = frame.Visibility.VisibleWmos[visibleIndex];
                            WmoRenderer? renderer = visibleWmoRenderers[visibleIndex];
                            if (renderer == null)
                                continue;

                            renderer.RenderWithTransform(visible.Instance.Transform, view, proj, WmoRenderPass.Opaque,
                                fogColor, objectFogStart, objectFogEnd, cameraPos,
                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                            AccumulateWmoRenderStats(frame, renderer.LastRenderStats);
                        }

                        foreach (WorldObjectPassCoordinator.WorldWmoOpaqueBatch batch in wmoBatchPlan.Batches)
                        {
                            int firstVisibleIndex = batch.VisibleIndices[0];
                            WmoRenderer renderer = visibleWmoRenderers[firstVisibleIndex]!;
                            IGpuInstancedWmoRenderer gpuRenderer = (IGpuInstancedWmoRenderer)renderer;
                            var instances = new List<VisibleWmoInstance>(batch.VisibleIndices.Count);
                            foreach (int visibleIndex in batch.VisibleIndices)
                                instances.Add(frame.Visibility.VisibleWmos[visibleIndex]);

                            gpuRenderer.BeginGpuInstanceBatch(
                                view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                            foreach (VisibleWmoInstance visible in instances)
                                gpuRenderer.QueueGpuInstance(visible.Instance.Transform);
                            gpuRenderer.EndGpuInstanceBatch();

                            // The shell is shared across placements, but WMO-internal doodads
                            // retain placement-local visibility, animation, and M2 fallback rules.
                            foreach (VisibleWmoInstance visible in instances)
                            {
                                gpuRenderer.RenderOpaqueDoodadsForPlacement(
                                    visible.Instance.Transform, view, proj,
                                    fogColor, objectFogStart, objectFogEnd, cameraPos,
                                    lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                            }

                            AccumulateWmoRenderStats(frame, gpuRenderer.LastRenderStats);
                        }
                    });
                    if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Wmo, $"WMO render: {WmoRenderedCount} drawn, {WmoCulledCount} culled");
                },
                () =>
                {
                    frame.MdxVisibilityMs = MeasureDurationMs(() =>
                    {
                        CollectVisibleMdxBuckets(frame, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians);
                        CollectVisibleMdxInstances(frame, _taxiActorInstances, cameraPos, cameraForward, fogEnd, verticalFieldOfViewRadians, cullSmallDoodadsOnly: false, countAsTaxiActor: true);
                    });
                    FlushPendingVisibleMdxLoads();

                    // Advance animation only for renderers that survived visibility admission.
                    // The previous path scanned every placed MDX instance every frame, which was
                    // pure idle CPU cost on large maps even when only a fraction were visible.
                    frame.MdxAnimationMs = MeasureDurationMs(() =>
                    {
                        var updatedRenderers = new HashSet<IModelRenderer>();
                        WorldObjectPassCoordinator.ExecuteVisibleMdxAnimation(frame.ObjectPasses, frame.Visibility, visible =>
                        {
                            IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                            if (renderer != null && updatedRenderers.Add(renderer))
                            {
                                renderer.UpdateAnimation();
                            }
                        });
                    });

                    frame.MdxTransparentSortMs = MeasureDurationMs(() => PlanVisibleMdxPasses(frame));

                    frame.MdxOpaqueSubmissionMs = MeasureDurationMs(() =>
                    {
                        var gpuBatchRenderers = new HashSet<IGpuInstancedModelRenderer>();
                        var immediateBatchRenderers = new HashSet<IModelRenderer>();

                        try
                        {
                            (frame.OpaqueBatchedMdxCount, frame.OpaqueUnbatchedMdxCount) =
                            WorldObjectPassCoordinator.ExecutePlannedOpaqueMdx(
                                frame.ObjectPasses,
                                frame.Visibility,
                                visible =>
                                {
                                    IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                                    if (renderer == null)
                                        return;

                                    if (!_renderDiagPrinted)
                                    {
                                        ViewerLog.Info(ViewerLog.Category.Mdx,
                                            $"[M2-WORLD-DIAG] opaqueUnbatched key=\"{visible.Instance.ModelKey}\" renderer={renderer?.GetType().Name} hasTransparent={renderer?.HasTransparentWorldPass} requiresUnbatched={renderer?.RequiresUnbatchedWorldRender} bounds=({visible.Instance.BoundsMin.X:F1},{visible.Instance.BoundsMin.Y:F1},{visible.Instance.BoundsMin.Z:F1})-({visible.Instance.BoundsMax.X:F1},{visible.Instance.BoundsMax.Y:F1},{visible.Instance.BoundsMax.Z:F1}) pos={visible.Instance.Transform.Translation}");
                                    }

                                    renderer.RenderWithTransform(visible.Instance.Transform, view, proj, RenderPass.Opaque, visible.OpaqueFade,
                                        fogColor, objectFogStart, objectFogEnd, cameraPos,
                                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                                    MdxRenderedCount++;
                                },
                                visible =>
                                {
                                    IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
                                    if (renderer == null)
                                        return;

                                    if (renderer is IGpuInstancedModelRenderer gpuRenderer
                                        && gpuRenderer.SupportsGpuInstancedOpaque
                                        && visible.OpaqueFade >= 0.999f)
                                    {
                                        if (gpuBatchRenderers.Add(gpuRenderer))
                                        {
                                            gpuRenderer.BeginGpuInstanceBatch(
                                                view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                                        }

                                        gpuRenderer.QueueGpuInstance(visible.Instance.Transform, visible.OpaqueFade);
                                    }
                                    else
                                    {
                                        if (immediateBatchRenderers.Add(renderer))
                                        {
                                            renderer.BeginBatch(
                                                view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                                                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                                        }

                                        renderer.RenderInstance(visible.Instance.Transform, RenderPass.Opaque, visible.OpaqueFade);
                                    }

                                    MdxRenderedCount++;
                                });
                        }
                        finally
                        {
                            foreach (IGpuInstancedModelRenderer gpuRenderer in gpuBatchRenderers)
                                gpuRenderer.EndGpuInstanceBatch();
                        }
                    });

                    if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Mdx, $"MDX opaque: {MdxRenderedCount} drawn, {MdxCulledCount} culled");
                },
                () =>
                {
                    // ── PASS 2: LIQUID ──────────────────────────────────────────────
                    // Render liquid after opaque geometry has established the depth buffer,
                    // but before transparent MDX layers so reflective/translucent model
                    // surfaces are composited on top instead of being overpainted by water.
                    _gl.Disable(EnableCap.Blend);
                    _gl.DepthMask(true);
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Lequal);
                    frame.LiquidMs = MeasureDurationMs(() => _terrainManager.RenderLiquid(view, proj, cameraPos));
                },
                () =>
                {
                    // ── PASS 3: TRANSPARENT (back-to-front, frustum-culled) ─────────
                    // Render transparent/blended object layers sorted by distance to camera.
                    // Depth test ON but depth write OFF so transparent objects don't
                    // occlude each other incorrectly.
                    MeasureDurationMs(() =>
                    {
                        _gl.Enable(EnableCap.DepthTest);
                        _gl.DepthFunc(DepthFunction.Lequal);

                        var transparentObjectSort = new List<(bool IsWmo, int Index, float DistanceSq)>(
                            frame.Visibility.VisibleWmos.Count + frame.ObjectPasses.TransparentVisibleMdxRoutes.Count);

                        for (int i = 0; i < frame.Visibility.VisibleWmos.Count; i++)
                            transparentObjectSort.Add((true, i, frame.Visibility.VisibleWmos[i].CenterDistanceSq));

                        for (int i = 0; i < frame.ObjectPasses.TransparentVisibleMdxRoutes.Count; i++)
                        {
                            var route = frame.ObjectPasses.TransparentVisibleMdxRoutes[i];
                            var visible = frame.Visibility.VisibleMdx[route.VisibleMdxIndex];
                            transparentObjectSort.Add((false, route.VisibleMdxIndex, visible.CenterDistanceSq));
                        }

                        transparentObjectSort.Sort((left, right) => right.DistanceSq.CompareTo(left.DistanceSq));

                        frame.TransparentBatchedMdxCount = 0;
                        frame.TransparentUnbatchedMdxCount = 0;
                        frame.WmoTransparentSubmissionMs = 0;
                        frame.MdxTransparentSubmissionMs = 0;

                        foreach (var entry in transparentObjectSort)
                        {
                            if (entry.IsWmo)
                            {
                                var visibleWmo = frame.Visibility.VisibleWmos[entry.Index];
                                WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visibleWmo.Instance.ModelKey);
                                if (renderer == null)
                                    continue;

                                double wmoTransparentMs = MeasureDurationMs(() =>
                                {
                                    renderer.RenderWithTransform(visibleWmo.Instance.Transform, view, proj, WmoRenderPass.Transparent,
                                        fogColor, objectFogStart, objectFogEnd, cameraPos,
                                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                                });
                                frame.WmoTransparentSubmissionMs += wmoTransparentMs;
                                AccumulateWmoRenderStats(frame, renderer.LastRenderStats);
                                continue;
                            }

                            var visibleMdx = frame.Visibility.VisibleMdx[entry.Index];
                            IModelRenderer? mdxRenderer = ResolveVisibleMdxRenderer(frame, visibleMdx.Instance.ModelKey);
                            if (mdxRenderer == null)
                                continue;

                            double mdxTransparentMs = MeasureDurationMs(() =>
                            {
                                mdxRenderer.RenderWithTransform(visibleMdx.Instance.Transform, view, proj, RenderPass.Transparent, visibleMdx.TransparentFade,
                                    fogColor, objectFogStart, objectFogEnd, cameraPos,
                                    lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                            });
                            frame.MdxTransparentSubmissionMs += mdxTransparentMs;
                            frame.TransparentUnbatchedMdxCount++;
                        }

                        foreach (WmoRenderer renderer in _worldFrameWmoRenderers)
                            renderer.EndWorldFrame();
                        _worldFrameWmoRenderers.Clear();
                    });
                    if (!_renderDiagPrinted) _renderDiagPrinted = true;
                },
                () =>
                {
                    overlayElapsedMs = MeasureDurationMs(() =>
                    {
                        objectWireframeEnabled = _assets.ObjectWireframeEnabled || _wireframeRevealEnabled;
                        if (_assets.ObjectWireframeEnabled)
                        {
                            objectWireframeMs = MeasureDurationMs(() =>
                            {
                                (objectWireframePreparedCount, objectWireframeSubmittedCount) =
                                    RenderVisibleObjectWireframeOverlay(frame, view, proj, cameraPos, fogColor, fogStart, fogEnd, lighting);
                            });
                        }
                        else if (_wireframeRevealEnabled)
                        {
                            objectWireframeMs = MeasureDurationMs(() =>
                            {
                                (objectWireframePreparedCount, objectWireframeSubmittedCount) =
                                    RenderWireframeReveal(view, proj, cameraPos, fogColor, fogStart, fogEnd, lighting);
                            });
                        }

                        // Reset GL state before bounding boxes
                        _gl.Disable(EnableCap.Blend);
                        _gl.DepthMask(true);
                        _gl.Enable(EnableCap.DepthTest);
                        _gl.DepthFunc(DepthFunction.Lequal);
                        _gl.UseProgram(0);
                        _gl.BindVertexArray(0);

                        // 4. Debug bounding boxes for camera-admitted placements.
                        //
                        // The visibility pass has already paid the placement admission cost.
                        // Walking _mdxInstances/_wmoInstances here made the debug overlay rescan
                        // every loaded placement on every frame, including placements that could
                        // never produce a box. On full-map scenes that scan was the dominant
                        // overlay cost. Keep selected bounds independent, but consume the
                        // visibility frame for the global debug boxes.
                        if ((_showSelectedObjectBounds || _showBoundingBoxes || _showPm4ObjectBounds) && _bbRenderer != null)
                        {
                // Depth test ON so boxes behind terrain/objects are hidden,
                // depth write OFF so box lines don't occlude models
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                _gl.DepthMask(false);
                _bbRenderer.BeginBatch();

                float selectedBoundsTime = (float)(Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency);
                Vector3 selectedBoundsInnerColor = Pm4ColorSelectedBounds;
                Vector3 selectedBoundsAccentA = Pm4ColorSelection;       // saturated yellow
                Vector3 selectedBoundsAccentB = Pm4ColorHighlight;       // saturated teal

                selectionBoundsMs = MeasureDurationMs(() =>
                {
                    if (_showSelectedObjectBounds)
                    {
                        if (SelectedInstance is ObjectInstance selectedInstance && !ShouldHideObjectInstanceByUniqueId(selectedInstance))
                        {
                            _bbRenderer.BatchHighlightedBoxMinMax(
                                selectedInstance.BoundsMin,
                                selectedInstance.BoundsMax,
                                selectedBoundsTime,
                                selectedBoundsInnerColor,
                                selectedBoundsAccentA,
                                selectedBoundsAccentB);
                            selectionBoundsPreparedCount++;
                        }

                        if (_showPm4Overlay
                            && _selectedPm4ObjectKey.HasValue
                            && _pm4ObjectLookup.TryGetValue(_selectedPm4ObjectKey.Value, out Pm4OverlayObject? selectedPm4Object))
                        {
                            Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
                            bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
                                || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                                || _pm4OverlayScale != Vector3.One;
                            Matrix4x4 objectTransform = BuildPm4ObjectTransform(_selectedPm4ObjectKey.Value, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            Vector3 boundsMin = selectedPm4Object.BoundsMin;
                            Vector3 boundsMax = selectedPm4Object.BoundsMax;
                            if (applyObjectTransform)
                                TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                            _bbRenderer.BatchHighlightedBoxMinMax(
                                boundsMin,
                                boundsMax,
                                selectedBoundsTime,
                                selectedBoundsInnerColor,
                                selectedBoundsAccentA,
                                selectedBoundsAccentB);
                            selectionBoundsPreparedCount++;
                        }
                    }

                    if (_showBoundingBoxes)
                    {
                        var adapter = _terrainManager.Adapter;
                        if (!_renderDiagPrinted)
                            ViewerLog.Debug(ViewerLog.Category.Terrain, $"BB render: {adapter.MddfPlacements.Count} MDDF + {adapter.ModfPlacements.Count} MODF markers");

                        // MDDF bounding boxes (light pastel magenta)
                        foreach (VisibleMdxInstance visible in frame.Visibility.VisibleMdx)
                        {
                            ObjectInstance inst = visible.Instance;
                            _bbRenderer.BatchBoxMinMax(inst.BoundsMin, inst.BoundsMax, Pm4ColorMddfBounds);
                            selectionBoundsPreparedCount++;
                        }
                        // MODF bounding boxes (light pastel cyan)
                        foreach (VisibleWmoInstance visible in frame.Visibility.VisibleWmos)
                        {
                            ObjectInstance inst = visible.Instance;
                            _bbRenderer.BatchBoxMinMax(inst.BoundsMin, inst.BoundsMax, Pm4ColorModfBounds);
                            selectionBoundsPreparedCount++;
                        }
                    }
                });

                if (_showPm4ObjectBounds && _showPm4Overlay && _pm4TileObjects.Count > 0)
                {
                    pm4BoundsMs += MeasureDurationMs(() =>
                    {
                    Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
                    bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
                        || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                        || _pm4OverlayScale != Vector3.One;

                    foreach (var (tileKey, objects) in _pm4TileObjects)
                    {
                        if (!ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                            continue;

                        foreach (Pm4OverlayObject obj in objects)
                        {
                            if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                                continue;

                            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                            Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            if (!ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out _))
                                continue;

                            Vector3 boundsMin = obj.BoundsMin;
                            Vector3 boundsMax = obj.BoundsMax;
                            if (applyObjectTransform)
                                TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                            Vector3 boxColor = Pm4ColorObjectBounds;  // light pastel — container
                            if (_highlightedPm4ObjectKeys.Contains(objectKey))
                                boxColor = Pm4ColorHighlight;        // saturated — search hit
                            if (_selectedPm4ObjectGroupKey.HasValue
                                && IsPm4ObjectInGroup(_selectedPm4ObjectGroupKey.Value, objectKey))
                                boxColor = Pm4ColorSelection;        // saturated — selection

                            _bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, boxColor);
                            pm4BoundsPreparedCount++;
                        }
                    }
                    });
                }

                // CK24-level bounding boxes: one merged box per CK24 object across all sub-objects.
                if (_showPm4Ck24Bounds && _showPm4Overlay && _pm4TileObjects.Count > 0)
                {
                    pm4BoundsMs += MeasureDurationMs(() =>
                    {
                    Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
                    bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
                        || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                        || _pm4OverlayScale != Vector3.One;

                    // Group objects by (tileX, tileY, Ck24) to merge sub-objects into one box per CK24.
                    var ck24Groups = new Dictionary<(int tileX, int tileY, uint ck24), (Vector3 min, Vector3 max, byte ck24Type, int count)>();

                    foreach (var (tileKey, objects) in _pm4TileObjects)
                    {
                        if (!ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                            continue;

                        foreach (Pm4OverlayObject obj in objects)
                        {
                            if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                                continue;

                            var ck24Key = (tileKey.tileX, tileKey.tileY, obj.Ck24);
                            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                            Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            if (!ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out _))
                                continue;

                            Vector3 boundsMin = obj.BoundsMin;
                            Vector3 boundsMax = obj.BoundsMax;
                            if (applyObjectTransform)
                                TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                            if (ck24Groups.TryGetValue(ck24Key, out var existing))
                            {
                                ck24Groups[ck24Key] = (
                                    Vector3.Min(existing.min, boundsMin),
                                    Vector3.Max(existing.max, boundsMax),
                                    existing.ck24Type,
                                    existing.count + 1);
                            }
                            else
                            {
                                ck24Groups[ck24Key] = (boundsMin, boundsMax, obj.Ck24Type, 1);
                            }
                        }
                    }

                    // Render one box per CK24 object.
                    foreach (var ((tileX, tileY, ck24), (boundsMin, boundsMax, ck24Type, count)) in ck24Groups)
                    {
                        // Light pastel — CK24 container color, varied by ck24Type for at-a-glance discrimination
                        Vector3 boxColor = ck24Type switch
                        {
                            0x00 => new Vector3(0.65f, 0.65f, 0.75f),  // nav mesh: light pastel blue-gray
                            0x40 or 0x41 => new Vector3(1.00f, 0.95f, 0.65f),  // M2: light pastel yellow
                            0x42 or 0x43 => new Vector3(0.65f, 0.95f, 0.95f),  // WMO: light pastel cyan
                            0xC0 or 0xC1 or 0xC2 or 0xC3 => new Vector3(1.00f, 0.75f, 0.60f),  // M2 exterior: light pastel orange
                            _ => new Vector3(0.80f, 0.80f, 0.80f)  // unknown: light pastel gray
                        };

                        // Highlight the selected object's CK24 group.
                        if (_selectedPm4ObjectKey.HasValue
                            && _selectedPm4ObjectKey.Value.tileX == tileX
                            && _selectedPm4ObjectKey.Value.tileY == tileY
                            && _selectedPm4ObjectKey.Value.ck24 == ck24)
                        {
                            boxColor = Pm4ColorSelectedBounds;  // light pastel white — clear container
                        }

                        _bbRenderer.BatchBoxMinMax(boundsMin, boundsMax, boxColor);
                        pm4BoundsPreparedCount++;
                    }
                    });
                }

                _bbRenderer.FlushBatch(view, proj);

                _gl.DepthMask(true);
            }

            // 5+6. Batched overlay rendering (POI pins + taxi paths) — single draw call
            if (_bbRenderer != null)
            {
                _bbRenderer.BeginBatch();
                _bbRenderer.BeginSolidBatch();

                _pm4VisibleObjectCount = 0;
                _pm4VisibleLineCount = 0;
                _pm4VisibleTriangleCount = 0;
                _pm4VisiblePositionRefCount = 0;

                pm4GeometryPrepareMs = MeasureDurationMs(() =>
                {
                if (_showPm4Overlay && _pm4TileObjects.Count > 0)
                {
                    Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
                    bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
                        || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
                        || _pm4OverlayScale != Vector3.One;

                    foreach (var (tileKey, objects) in _pm4TileObjects)
                    {
                        if (!ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                            continue;

                        if (_showPm4PositionRefs
                            && _pm4TilePositionRefs.TryGetValue(tileKey, out List<Vector3>? positionRefs)
                            && positionRefs.Count > 0)
                        {
                            for (int i = 0; i < positionRefs.Count; i++)
                            {
                                Vector3 marker = applyPm4Transform ? ApplyPm4OverlayTransform(positionRefs[i], pm4Transform) : positionRefs[i];
                                _bbRenderer.BatchPin(marker, 16f, 3f, Pm4ColorMprl);
                            }

                            _pm4VisiblePositionRefCount += positionRefs.Count;
                        }

                        foreach (Pm4OverlayObject obj in objects)
                        {
                            if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                                continue;

                            var objectKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                            Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                            Matrix4x4 geometryTransform = BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform);

                            if (!ShouldRenderPm4Object(obj, objectTransform, applyObjectTransform, cameraPos, out Vector3 transformedCenter))
                                continue;

                            _pm4VisibleObjectCount++;
                            Vector3 pm4Color = GetPm4ObjectColor(tileKey, obj);
                            if (_highlightedPm4ObjectKeys.Contains(objectKey))
                                pm4Color = Pm4ColorHighlight;  // saturated teal — search hit
                            if (_selectedPm4ObjectGroupKey.HasValue
                                && IsPm4ObjectInGroup(_selectedPm4ObjectGroupKey.Value, objectKey))
                                pm4Color = Pm4ColorSelection;  // saturated yellow — selection

                            if (_showPm4SolidOverlay && obj.Triangles.Count > 0)
                            {
                                for (int i = 0; i < obj.Triangles.Count; i++)
                                {
                                    Pm4Triangle tri = obj.Triangles[i];
                                    Vector3 a = ApplyPm4OverlayTransform(tri.A, geometryTransform);
                                    Vector3 b = ApplyPm4OverlayTransform(tri.B, geometryTransform);
                                    Vector3 c = ApplyPm4OverlayTransform(tri.C, geometryTransform);
                                    _bbRenderer.BatchTriangle(a, b, c, pm4Color, 0.20f);
                                }
                                _pm4VisibleTriangleCount += obj.Triangles.Count;
                            }

                            for (int i = 0; i < obj.Lines.Count; i++)
                            {
                                Pm4LineSegment line = obj.Lines[i];
                                Vector3 from = ApplyPm4OverlayTransform(line.From, geometryTransform);
                                Vector3 to = ApplyPm4OverlayTransform(line.To, geometryTransform);
                                _bbRenderer.BatchLine(from, to, pm4Color);
                            }

                            _pm4VisibleLineCount += obj.Lines.Count;

                            if (_showPm4ObjectCentroids)
                            {
                                // Centroid is a per-object marker, not mesh — use dedicated dark pastel
                                _bbRenderer.BatchPin(transformedCenter, 22f, 4f, Pm4ColorCentroid);
                            }
                        }
                    }
                }
                });
                pm4GeometryPreparedCount = _pm4VisibleLineCount + _pm4VisibleTriangleCount;
                pm4GeometrySubmittedCount = pm4GeometryPreparedCount;

                if (_showPm4Overlay)
                {
                    bool pm4IgnoreDepth = _pm4OverlayIgnoreDepth;

                    if (_showPm4SolidOverlay && _pm4VisibleTriangleCount > 0)
                    {
                        _gl.Enable(EnableCap.Blend);
                        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                        if (pm4IgnoreDepth)
                        {
                            _gl.Disable(EnableCap.DepthTest);
                        }
                        else
                        {
                            _gl.Enable(EnableCap.DepthTest);
                            _gl.DepthFunc(DepthFunction.Lequal);
                        }

                        _gl.DepthMask(false);
                        _gl.Disable(EnableCap.CullFace);
                        pm4GeometrySubmitMs += MeasureDurationMs(() => _bbRenderer.FlushSolidBatch(view, proj));
                        _gl.Enable(EnableCap.CullFace);
                        _gl.Disable(EnableCap.Blend);
                    }

                    // MSCN/MSPV node markers — solid filled cubes. Bright saturated colors
                    // (cyan/magenta) so they pop against the pastel mesh.
                    pm4NodesMs += MeasureDurationMs(() =>
                    {
                    if (_pm4RenderNodesAsCubes && (_showPm4MscnNodes || _showPm4MspvNodes))
                    {
                        if (_showPm4MscnNodes)
                            EnsurePm4MscnData();
                        if (_showPm4MspvNodes)
                            EnsurePm4MspvData();

                        if (_showPm4SolidOverlay || _pm4VisibleTriangleCount == 0)
                        {
                            _gl.Enable(EnableCap.Blend);
                            _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                            if (pm4IgnoreDepth)
                                _gl.Disable(EnableCap.DepthTest);
                            else
                            {
                                _gl.Enable(EnableCap.DepthTest);
                                _gl.DepthFunc(DepthFunction.Lequal);
                            }
                            _gl.DepthMask(false);
                            _gl.Disable(EnableCap.CullFace);

                            int mscnDrawn = 0;
                            if (_showPm4MscnNodes && _pm4TileMscnPoints.Count > 0)
                            {
                                int limit = 15000;
                                foreach (var kv in _pm4TileMscnPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && mscnDrawn < limit; i++)
                                    {
                                        _bbRenderer.BatchSolidCube(pts[i], _pm4MscnCubeSize, Pm4ColorMscn, _pm4MscnCubeAlpha);
                                        mscnDrawn++;
                                    }
                                }
                            }

                            int mspvDrawn = 0;
                            if (_showPm4MspvNodes && _pm4TileMspvPoints.Count > 0)
                            {
                                int limit = 8000;
                                foreach (var kv in _pm4TileMspvPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && mspvDrawn < limit; i++)
                                    {
                                        _bbRenderer.BatchSolidCube(pts[i], _pm4MspvCubeSize, Pm4ColorMspv, _pm4MspvCubeAlpha);
                                        mspvDrawn++;
                                    }
                                }
                            }

                            if (mscnDrawn > 0 || mspvDrawn > 0)
                            {
                                _bbRenderer.FlushSolidBatch(view, proj);
                                _pm4VisiblePositionRefCount += mscnDrawn + mspvDrawn;
                                pm4NodesPreparedCount += mscnDrawn + mspvDrawn;
                            }
                            _gl.Enable(EnableCap.CullFace);
                            _gl.Disable(EnableCap.Blend);
                        }
                    }
                    });

                    bool hasPm4LineGeometry = _pm4VisibleLineCount > 0
                        || _pm4VisiblePositionRefCount > 0
                        || (_showPm4ObjectCentroids && _pm4VisibleObjectCount > 0);
                    if (hasPm4LineGeometry)
                    {
                        _gl.LineWidth(_pm4WireframeLineWidth);
                        if (pm4IgnoreDepth)
                        {
                            _gl.Disable(EnableCap.DepthTest);
                        }
                        else
                        {
                            _gl.Enable(EnableCap.DepthTest);
                            _gl.DepthFunc(DepthFunction.Lequal);
                        }

                        // MSCN node overlay (legacy wireframe pin mode)
                        if (!_pm4RenderNodesAsCubes && _showPm4MscnNodes && _pm4TileMscnPoints.Count > 0)
                        {
                            pm4NodesMs += MeasureDurationMs(() =>
                            {
                                EnsurePm4MscnData();
                                int limit = 15000, drawn = 0;
                                foreach (var kv in _pm4TileMscnPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && drawn < limit; i++)
                                    {
                                        _bbRenderer.BatchPin(pts[i], _pm4MscnCubeSize * 2.0f, _pm4MscnCubeSize * 0.5f, Pm4ColorMscn);
                                        drawn++;
                                    }
                                }
                                _pm4VisiblePositionRefCount += drawn;
                                pm4NodesPreparedCount += drawn;
                            });
                        }
                        if (!_pm4RenderNodesAsCubes && _showPm4MspvNodes && _pm4TileMspvPoints.Count > 0)
                        {
                            pm4NodesMs += MeasureDurationMs(() =>
                            {
                                EnsurePm4MspvData();
                                int limit = 8000, drawn = 0;
                                foreach (var kv in _pm4TileMspvPoints)
                                {
                                    var pts = kv.Value;
                                    if (pts == null) continue;
                                    for (int i = 0; i < pts.Count && drawn < limit; i++)
                                    {
                                        _bbRenderer.BatchPin(pts[i], _pm4MspvCubeSize * 2.0f, _pm4MspvCubeSize * 0.5f, Pm4ColorMspv);
                                        drawn++;
                                    }
                                }
                                _pm4VisiblePositionRefCount += drawn;
                                pm4NodesPreparedCount += drawn;
                            });
                        }

                        _gl.DepthMask(false);
                        pm4GeometrySubmitMs += MeasureDurationMs(() => _bbRenderer.FlushBatch(view, proj));
                        _gl.LineWidth(1.0f);
                    }

                    // Reset default state and clear PM4 primitives so other overlays use their normal pass.
                    _gl.Enable(EnableCap.DepthTest);
                    _gl.DepthFunc(DepthFunction.Lequal);
                    _gl.DepthMask(true);
                    _gl.Disable(EnableCap.Blend);

                    _bbRenderer.BeginBatch();
                    _bbRenderer.BeginSolidBatch();
                }

                poiTaxiMs = MeasureDurationMs(() =>
                {
                // POI pin markers (magenta)
                if (_showPoi && _poiLoader != null && _poiLoader.Entries.Count > 0)
                {
                    var poiColor = new Vector3(1f, 0f, 1f);
                    foreach (var poi in _poiLoader.Entries)
                    {
                        _bbRenderer.BatchPin(poi.Position, 56f, 9f, poiColor);
                        poiTaxiPreparedCount++;
                    }
                }

                // Taxi paths — filtered by selection
                if (_showTaxi && _taxiLoader != null)
                {
                    var nodeColor = new Vector3(1f, 1f, 0f);
                    var lineColor = new Vector3(0f, 1f, 1f);
                    var routeHandleColor = new Vector3(1f, 0.65f, 0f);
                    var selectedRouteColor = new Vector3(1f, 1f, 1f);
                    var nodeBoxColor = new Vector3(1f, 0.92f, 0.35f);
                    var routeBoxColor = new Vector3(1f, 0.78f, 0.28f);
                    int visibleRouteCount = _taxiLoader.Routes.Count(IsTaxiRouteVisible);
                    bool showRouteHandles = _selectedTaxiNodeId >= 0 || _selectedTaxiRouteId >= 0 || visibleRouteCount <= 32;

                    foreach (var node in _taxiLoader.Nodes)
                    {
                        if (!IsTaxiNodeVisible(node)) continue;
                        _bbRenderer.BatchPin(node.Position, 64f, 12f, nodeColor);
                        poiTaxiPreparedCount++;
                        _bbRenderer.BatchBoxMinMax(
                            node.Position - new Vector3(36f, 36f, 18f),
                            node.Position + new Vector3(36f, 36f, 96f),
                            nodeBoxColor);
                        poiTaxiPreparedCount++;
                    }

                    foreach (var route in _taxiLoader.Routes)
                    {
                        if (!IsTaxiRouteVisible(route)) continue;
                        Vector3 routeColor = route.PathId == _selectedTaxiRouteId ? selectedRouteColor : lineColor;
                        for (int i = 0; i < route.Waypoints.Count - 1; i++)
                        {
                            _bbRenderer.BatchLine(route.Waypoints[i], route.Waypoints[i + 1], routeColor);
                            poiTaxiPreparedCount++;
                        }

                        if (showRouteHandles && TryGetTaxiRouteSelectionPoint(route, out Vector3 selectionPoint))
                        {
                            float pinHeight = route.PathId == _selectedTaxiRouteId ? 64f : 52f;
                            float headSize = route.PathId == _selectedTaxiRouteId ? 12f : 10f;
                            _bbRenderer.BatchPin(selectionPoint, pinHeight, headSize,
                                route.PathId == _selectedTaxiRouteId ? selectedRouteColor : routeHandleColor);
                            poiTaxiPreparedCount++;
                            _bbRenderer.BatchBoxMinMax(
                                selectionPoint - new Vector3(34f, 34f, 20f),
                                selectionPoint + new Vector3(34f, 34f, 72f),
                                route.PathId == _selectedTaxiRouteId ? selectedRouteColor : routeBoxColor);
                            poiTaxiPreparedCount++;
                        }
                    }
                }
                });

                // AreaTriggers (green wireframe shapes for portals and event markers)
                areaTriggersMs = MeasureDurationMs(() =>
                {
                if (_showAreaTriggers && _areaTriggerLoader != null && _areaTriggerLoader.Count > 0)
                {
                    var triggerColor = new Vector3(0f, 1f, 0f); // Green
                    foreach (var trigger in _areaTriggerLoader.Triggers)
                    {
                        if (trigger.IsSphere && trigger.Radius > 0f)
                        {
                            // Render sphere triggers as simple wireframe circles (3 orthogonal rings)
                            int segments = 16;
                            float r = trigger.Radius;
                            var c = trigger.Position;
                            
                            // XY plane circle
                            for (int i = 0; i < segments; i++)
                            {
                                float a1 = (i / (float)segments) * MathF.PI * 2f;
                                float a2 = ((i + 1) / (float)segments) * MathF.PI * 2f;
                                var p1 = c + new Vector3(MathF.Cos(a1) * r, MathF.Sin(a1) * r, 0f);
                                var p2 = c + new Vector3(MathF.Cos(a2) * r, MathF.Sin(a2) * r, 0f);
                                _bbRenderer.BatchLine(p1, p2, triggerColor);
                            }
                            
                            // XZ plane circle
                            for (int i = 0; i < segments; i++)
                            {
                                float a1 = (i / (float)segments) * MathF.PI * 2f;
                                float a2 = ((i + 1) / (float)segments) * MathF.PI * 2f;
                                var p1 = c + new Vector3(MathF.Cos(a1) * r, 0f, MathF.Sin(a1) * r);
                                var p2 = c + new Vector3(MathF.Cos(a2) * r, 0f, MathF.Sin(a2) * r);
                                _bbRenderer.BatchLine(p1, p2, triggerColor);
                            }
                            
                            // YZ plane circle
                            for (int i = 0; i < segments; i++)
                            {
                                float a1 = (i / (float)segments) * MathF.PI * 2f;
                                float a2 = ((i + 1) / (float)segments) * MathF.PI * 2f;
                                var p1 = c + new Vector3(0f, MathF.Cos(a1) * r, MathF.Sin(a1) * r);
                                var p2 = c + new Vector3(0f, MathF.Cos(a2) * r, MathF.Sin(a2) * r);
                                _bbRenderer.BatchLine(p1, p2, triggerColor);
                            }
                            areaTriggerPreparedCount += segments * 3;
                        }
                        else if (trigger.BoxLength > 0f && trigger.BoxWidth > 0f && trigger.BoxHeight > 0f)
                        {
                            // Render box triggers as wireframe boxes (12 edges)
                            float halfL = trigger.BoxLength / 2f;
                            float halfW = trigger.BoxWidth / 2f;
                            float h = trigger.BoxHeight;
                            var c = trigger.Position;
                            
                            // 8 corners of the box
                            var v0 = c + new Vector3(-halfL, -halfW, 0f);
                            var v1 = c + new Vector3( halfL, -halfW, 0f);
                            var v2 = c + new Vector3( halfL,  halfW, 0f);
                            var v3 = c + new Vector3(-halfL,  halfW, 0f);
                            var v4 = c + new Vector3(-halfL, -halfW, h);
                            var v5 = c + new Vector3( halfL, -halfW, h);
                            var v6 = c + new Vector3( halfL,  halfW, h);
                            var v7 = c + new Vector3(-halfL,  halfW, h);
                            
                            // Bottom face
                            _bbRenderer.BatchLine(v0, v1, triggerColor);
                            _bbRenderer.BatchLine(v1, v2, triggerColor);
                            _bbRenderer.BatchLine(v2, v3, triggerColor);
                            _bbRenderer.BatchLine(v3, v0, triggerColor);
                            
                            // Top face
                            _bbRenderer.BatchLine(v4, v5, triggerColor);
                            _bbRenderer.BatchLine(v5, v6, triggerColor);
                            _bbRenderer.BatchLine(v6, v7, triggerColor);
                            _bbRenderer.BatchLine(v7, v4, triggerColor);
                            
                            // Vertical edges
                            _bbRenderer.BatchLine(v0, v4, triggerColor);
                            _bbRenderer.BatchLine(v1, v5, triggerColor);
                            _bbRenderer.BatchLine(v2, v6, triggerColor);
                            _bbRenderer.BatchLine(v3, v7, triggerColor);
                            areaTriggerPreparedCount += 12;
                        }
                    }
                }
                });

                if (_showLitLights && _litLoader != null && _litLoader.HasData)
                {
                    int highlightedLightIndex = _selectedLitLightIndex >= 0
                        ? _selectedLitLightIndex
                        : _lastLitSample?.DominantLightIndex ?? -1;

                    for (int lightIndex = 0; lightIndex < _litLoader.Lights.Count; lightIndex++)
                    {
                        LitLoader.LitLight light = _litLoader.Lights[lightIndex];
                        if (!light.HasMeaningfulPosition)
                            continue;

                        Vector3 lightColor = _litLoader.EvaluateOverlayColor(light, lighting.GameTime);
                        bool isHighlighted = lightIndex == highlightedLightIndex;
                        float pinHeight = isHighlighted ? 60f : 36f;
                        float headSize = isHighlighted ? 8f : 5f;
                        _bbRenderer.BatchPin(light.Position, pinHeight, headSize,
                            isHighlighted ? new Vector3(1f, 1f, 1f) : lightColor);

                        float footprintRadius = Math.Max(light.Radius, 6f);
                        float footprintHeight = Math.Max(8f, Math.Min(light.Dropoff, 80f));
                        var min = new Vector3(light.Position.X - footprintRadius, light.Position.Y - footprintRadius, light.Position.Z - footprintHeight * 0.25f);
                        var max = new Vector3(light.Position.X + footprintRadius, light.Position.Y + footprintRadius, light.Position.Z + footprintHeight * 0.25f);
                        _bbRenderer.BatchBoxMinMax(min, max,
                            isHighlighted ? new Vector3(1f, 1f, 1f) : lightColor);
                    }
                }

                _gl.LineWidth(5.0f);
                _bbRenderer.FlushBatch(view, proj);
                _gl.LineWidth(1.0f);
            }
                    });

                    frame.SetOverlayOwner(
                        WorldOverlayOwners.ObjectWireframe,
                        objectWireframeMs,
                        objectWireframeEnabled,
                        objectWireframePreparedCount,
                        objectWireframeSubmittedCount,
                        objectWireframeEnabled ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.SelectionBounds,
                        selectionBoundsMs,
                        _bbRenderer != null && (_showSelectedObjectBounds || _showBoundingBoxes),
                        selectionBoundsPreparedCount,
                        selectionBoundsPreparedCount,
                        _bbRenderer != null && (_showSelectedObjectBounds || _showBoundingBoxes) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4Bounds,
                        pm4BoundsMs,
                        _bbRenderer != null && _showPm4Overlay && (_showPm4ObjectBounds || _showPm4Ck24Bounds),
                        pm4BoundsPreparedCount,
                        pm4BoundsPreparedCount,
                        _bbRenderer != null && _showPm4Overlay && (_showPm4ObjectBounds || _showPm4Ck24Bounds) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4GeometryPrepare,
                        pm4GeometryPrepareMs,
                        _bbRenderer != null && _showPm4Overlay,
                        pm4GeometryPreparedCount,
                        0,
                        _bbRenderer != null && _showPm4Overlay ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4GeometrySubmit,
                        pm4GeometrySubmitMs,
                        _bbRenderer != null && _showPm4Overlay,
                        pm4GeometryPreparedCount,
                        pm4GeometrySubmittedCount,
                        _bbRenderer != null && _showPm4Overlay ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.Pm4Nodes,
                        pm4NodesMs,
                        _bbRenderer != null && _showPm4Overlay && (_showPm4MscnNodes || _showPm4MspvNodes),
                        pm4NodesPreparedCount,
                        pm4NodesPreparedCount,
                        _bbRenderer != null && _showPm4Overlay && (_showPm4MscnNodes || _showPm4MspvNodes) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.PoiTaxi,
                        poiTaxiMs,
                        _bbRenderer != null && (_showPoi || _showTaxi),
                        poiTaxiPreparedCount,
                        poiTaxiPreparedCount,
                        _bbRenderer != null && (_showPoi || _showTaxi) ? "not_cached" : "disabled");
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.AreaTriggers,
                        areaTriggersMs,
                        _bbRenderer != null && _showAreaTriggers,
                        areaTriggerPreparedCount,
                        areaTriggerPreparedCount,
                        _bbRenderer != null && _showAreaTriggers ? "not_cached" : "disabled");

                    double accountedOverlayMs = frame.OverlayOwnerDurationSum;
                    double otherOverlayMs = Math.Max(0, overlayElapsedMs - accountedOverlayMs);
                    frame.SetOverlayOwner(
                        WorldOverlayOwners.OtherOverlay,
                        otherOverlayMs,
                        otherOverlayMs > 0,
                        cacheStatus: otherOverlayMs > 0 ? "not_applicable" : "disabled");
                    frame.OverlayMs = frame.OverlayOwnerDurationSum;
                }));

        if (!continuedPastTerrain)
        {
            FinalizeRenderFrameStats(frame, frameTimer);
            return;
        }

        if (!_doodadsVisible && !_renderDiagPrinted)
            _renderDiagPrinted = true;

        FinalizeRenderFrameStats(frame, frameTimer);
    }

    private void RenderSkyboxBackdrop(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        bool renderedActiveClientSky = false;
        if (_skyDome.NightVisibility > 0.001f
            && TryGetQueuedMdx(_activeLightSkyboxModelKey ?? string.Empty) is { } lightSkyboxRenderer)
        {
            lightSkyboxRenderer.UpdateAnimation();
            lightSkyboxRenderer.RenderBackdrop(Matrix4x4.CreateTranslation(cameraPos), view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            renderedActiveClientSky = true;
        }

        if (_skyboxInstances.Count == 0)
            return;

        ObjectInstance? nearestSkybox = null;
        float nearestDistSq = float.MaxValue;
        foreach (var inst in _skyboxInstances)
        {
            float distSq = Vector3.DistanceSquared(cameraPos, inst.PlacementPosition);
            if (distSq >= nearestDistSq)
                continue;

            nearestDistSq = distSq;
            nearestSkybox = inst;
        }

        if (!nearestSkybox.HasValue)
            return;

        var skybox = nearestSkybox.Value;
        if (renderedActiveClientSky
            && string.Equals(skybox.ModelKey, _activeLightSkyboxModelKey, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        var renderer = TryGetQueuedMdx(skybox.ModelKey);
        if (renderer == null)
            return;

        renderer.UpdateAnimation();
        renderer.RenderBackdrop(CreateSkyboxBackdropTransform(skybox.Transform, cameraPos), view, proj,
            fogColor, fogStart, fogEnd, cameraPos,
            lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
    }

    private static Matrix4x4 CreateSkyboxBackdropTransform(Matrix4x4 placementTransform, Vector3 cameraPos)
    {
        placementTransform.M41 = cameraPos.X;
        placementTransform.M42 = cameraPos.Y;
        placementTransform.M43 = cameraPos.Z;
        return placementTransform;
    }

    internal static bool IsSkyboxModelPath(string modelPath)
    {
        return WorldSkyboxBackdropClassifier.IsBackdropModelPath(modelPath);
    }

    private void UpdateActiveSkyboxModel()
    {
        string? sourcePath = _lightService?.ActiveSkyboxModelPath;
        sourcePath = ResolveClientSkyboxPath(sourcePath);
        if (string.IsNullOrWhiteSpace(sourcePath))
            sourcePath = ResolveClientStarsFallback();

        if (string.Equals(sourcePath, _activeLightSkyboxSourcePath, StringComparison.OrdinalIgnoreCase))
            return;

        _activeLightSkyboxSourcePath = sourcePath;
        _activeLightSkyboxModelKey = string.IsNullOrWhiteSpace(sourcePath)
            ? null
            : WorldAssetManager.NormalizeKey(sourcePath);

        if (string.IsNullOrWhiteSpace(_activeLightSkyboxModelKey))
            return;

        _assets.PrioritizeMdxLoad(_activeLightSkyboxModelKey);
        ViewerLog.Info(
            ViewerLog.Category.Mdx,
            $"[Sky] Active client sky model: {sourcePath} (source={(_lightService?.ActiveSkyboxModelPath is null ? "client-stars fallback" : "LightSkybox DBC")})");
    }

    private string? ResolveClientSkyboxPath(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath) || _dataSource == null)
            return null;

        if (_dataSource.FileExists(sourcePath))
            return sourcePath;

        if (!string.IsNullOrWhiteSpace(Path.GetExtension(sourcePath)))
            return null;

        foreach (string extension in new[] { ".m2", ".mdx", ".mdl" })
        {
            string candidate = sourcePath + extension;
            if (_dataSource.FileExists(candidate))
                return candidate;
        }

        return null;
    }

    private string? ResolveClientStarsFallback()
    {
        if (_clientStarsProbeComplete)
            return _clientStarsFallbackModelPath;

        _clientStarsProbeComplete = true;
        if (_dataSource == null)
            return null;

        string[] candidates =
        [
            @"Environments\Stars\Stars.m2",
            @"Environments\Stars\Stars.mdx",
            @"Environments\Stars\Stars.mdl",
        ];
        foreach (string path in candidates)
        {
            if (!_dataSource.FileExists(path))
                continue;

            _clientStarsFallbackModelPath = path;
            ViewerLog.Info(ViewerLog.Category.Mdx, $"[Sky] Discovered client stars fallback: {path}");
            return path;
        }

        // Some extracted clients retain a World prefix around the same asset.
        foreach (string path in new[]
        {
            @"World\Environments\Stars\Stars.m2",
            @"World\Environments\Stars\Stars.mdx",
            @"World\Environments\Stars\Stars.mdl",
        })
        {
            if (_dataSource.FileExists(path))
            {
                _clientStarsFallbackModelPath = path;
                ViewerLog.Info(ViewerLog.Category.Mdx, $"[Sky] Discovered client stars fallback: {path}");
                return path;
            }
        }

        ViewerLog.Debug(ViewerLog.Category.Mdx, "[Sky] Client stars fallback was not present in the data source file list.");
        return null;
    }

    public void ToggleWireframe()
    {
        bool enable = !IsWireframe;
        SetTerrainWireframeEnabled(enable);
        SetObjectWireframeEnabled(enable);
    }

    public void SetTerrainWireframeEnabled(bool enabled)
    {
        if (_terrainManager.IsWireframe == enabled)
            return;

        _terrainManager.ToggleWireframe();
    }

    public void SetObjectWireframeEnabled(bool enabled)
    {
        if (_assets.ObjectWireframeEnabled == enabled && !_wireframeRevealEnabled)
            return;

        _wireframeRevealEnabled = false;
        _assets.SetObjectWireframeEnabled(enabled);
        ClearWireframeReveal();
    }

    public bool IsWireframe => TerrainWireframeEnabled || ObjectWireframeEnabled;

    public void UpdateWireframeReveal(Matrix4x4 view, Matrix4x4 proj,
        float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight)
    {
        if (!_wireframeRevealEnabled)
        {
            ClearWireframeReveal();
            return;
        }

        if (_instancesDirty)
            RebuildInstanceLists();

        _wireframeRevealWmoIndices.Clear();
        _wireframeRevealMdxIndices.Clear();

        if (_wmosVisible)
            PopulateWireframeRevealHits(_wmoInstances, _wireframeRevealWmoIndices,
                view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight);
        if (_doodadsVisible)
            PopulateWireframeRevealHits(_mdxInstances, _wireframeRevealMdxIndices,
                view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight);
    }

    public void UpdateHoveredAssetInfo(Matrix4x4 view, Matrix4x4 proj,
        float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight)
    {
        float safeViewportWidth = Math.Max(viewportWidth, 1f);
        float safeViewportHeight = Math.Max(viewportHeight, 1f);
        float ndcX = (mouseViewportX / safeViewportWidth) * 2f - 1f;
        float ndcY = 1f - (mouseViewportY / safeViewportHeight) * 2f;
        var (rayOrigin, rayDir) = ScreenToRay(ndcX, ndcY, view, proj);

        bool hasSceneRayHit = TryBuildHoveredSceneInfoByRay(rayOrigin, rayDir, out HoveredAssetInfo sceneRayInfo, out float sceneRayDistance);
        HoveredAssetInfo pm4RayInfo = default;
        float pm4RayDistance = float.MaxValue;
        bool hasPm4RayHit = _showPm4Overlay
            && TryBuildHoveredPm4InfoByRay(rayOrigin, rayDir, out pm4RayInfo, out pm4RayDistance);

        if (hasSceneRayHit || hasPm4RayHit)
        {
            const float rayDistanceEpsilon = 0.01f;
            if (hasPm4RayHit && (!hasSceneRayHit || _pm4OverlayIgnoreDepth || pm4RayDistance < sceneRayDistance - rayDistanceEpsilon))
            {
                _hoveredAssetInfo = pm4RayInfo.WithPreciseRayHit();
                return;
            }

            if (hasSceneRayHit)
            {
                _hoveredAssetInfo = sceneRayInfo.WithPreciseRayHit();
                return;
            }
        }

        bool hasSceneBrushHit = TryBuildHoveredSceneInfo(
            view,
            proj,
            mouseViewportX,
            mouseViewportY,
            viewportWidth,
            viewportHeight,
            out HoveredAssetInfo sceneBrushInfo,
            out float sceneBrushDistanceSq,
            out float sceneBrushDepth);
        HoveredAssetInfo pm4BrushInfo = default;
        int hoveredPm4Count = 0;
        float pm4BrushDistanceSq = float.MaxValue;
        float pm4BrushDepth = float.MaxValue;
        bool hasPm4BrushHit = _showPm4Overlay
            && TryBuildHoveredPm4Info(
                view,
                proj,
                mouseViewportX,
                mouseViewportY,
                viewportWidth,
                viewportHeight,
                out pm4BrushInfo,
                out hoveredPm4Count,
                out pm4BrushDistanceSq,
                out pm4BrushDepth);

        if (hasPm4BrushHit && (!hasSceneBrushHit || ShouldPreferPm4HoverBrush(pm4BrushDistanceSq, pm4BrushDepth, sceneBrushDistanceSq, sceneBrushDepth)))
        {
            _hoveredAssetInfo = new HoveredAssetInfo(
                pm4BrushInfo.AssetKind,
                pm4BrushInfo.DisplayName,
                pm4BrushInfo.SourcePath,
                pm4BrushInfo.DetailLine,
                pm4BrushInfo.WorldPosition,
                Math.Max(0, hoveredPm4Count - 1),
                pm4BrushInfo.Pm4ObjectKey,
                pm4BrushInfo.SceneObjectType,
                pm4BrushInfo.SceneObjectIndex,
                pm4BrushInfo.WlBodyKey);
            return;
        }

        if (hasSceneBrushHit)
        {
            _hoveredAssetInfo = sceneBrushInfo;
            return;
        }

        _hoveredAssetInfo = null;
    }

    public void ClearWireframeReveal()
    {
        _wireframeRevealWmoIndices.Clear();
        _wireframeRevealMdxIndices.Clear();
    }

    public void ClearHoveredAssetInfo()
    {
        _hoveredAssetInfo = null;
    }

    private bool TryBuildHoveredSceneInfo(
        Matrix4x4 view,
        Matrix4x4 proj,
        float mouseViewportX,
        float mouseViewportY,
        float viewportWidth,
        float viewportHeight,
        out HoveredAssetInfo info,
        out float bestDistanceSq,
        out float bestDepth)
    {
        info = default;
        float currentBestDistanceSq = float.MaxValue;
        float currentBestDepth = float.MaxValue;
        bestDistanceSq = float.MaxValue;
        bestDepth = float.MaxValue;

        HoveredAssetInfo? bestInfo = null;
        int hitCount = 0;
        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer;

        void ConsiderCandidate(HoveredAssetInfo candidateInfo, float distanceSq, float depth)
        {
            if (!IsHoverPickPositionAllowed(candidateInfo.WorldPosition))
                return;

            hitCount++;

            const float distanceEpsilon = 0.01f;
            if (!bestInfo.HasValue
                || distanceSq < currentBestDistanceSq - distanceEpsilon
                || (MathF.Abs(distanceSq - currentBestDistanceSq) <= distanceEpsilon && depth < currentBestDepth))
            {
                bestInfo = candidateInfo;
                currentBestDistanceSq = distanceSq;
                currentBestDepth = depth;
            }
        }

        if (_wmosVisible)
        {
            for (int i = 0; i < _wmoInstances.Count; i++)
            {
                ObjectInstance inst = _wmoInstances[i];
                if (ShouldHideObjectInstanceByUniqueId(inst))
                    continue;

                if (!TryMeasureHoverInfoHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredObjectInfo("WMO", inst, ObjectType.Wmo, i), distanceSq, depth);
            }
        }

        if (_doodadsVisible)
        {
            for (int i = 0; i < _mdxInstances.Count; i++)
            {
                ObjectInstance inst = _mdxInstances[i];
                if (ShouldHideObjectInstanceByUniqueId(inst))
                    continue;

                if (!TryMeasureHoverInfoHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredObjectInfo("MDX", inst, ObjectType.Mdx, i), distanceSq, depth);
            }
        }

        if (_showWlLiquids && _wlLoader != null)
        {
            for (int i = 0; i < _wlLoader.Bodies.Count; i++)
            {
                WlLiquidBody body = _wlLoader.Bodies[i];
                if (liquidRenderer != null && !liquidRenderer.IsWlBodyVisible(body.BodyKey))
                    continue;

                if (!TryMeasureHoverInfoHit(body.BoundsMin, body.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredWlLiquidInfo(body), distanceSq, depth);
            }
        }

        if (!bestInfo.HasValue)
            return false;

        HoveredAssetInfo bestCandidate = bestInfo.Value;
        bestDistanceSq = currentBestDistanceSq;
        bestDepth = currentBestDepth;
        info = new HoveredAssetInfo(
            bestCandidate.AssetKind,
            bestCandidate.DisplayName,
            bestCandidate.SourcePath,
            bestCandidate.DetailLine,
            bestCandidate.WorldPosition,
            Math.Max(0, hitCount - 1),
            bestCandidate.Pm4ObjectKey,
            bestCandidate.SceneObjectType,
            bestCandidate.SceneObjectIndex,
            bestCandidate.WlBodyKey);
        return true;
    }

    private bool TryBuildHoveredSceneInfoByRay(Vector3 rayOrigin, Vector3 rayDir, out HoveredAssetInfo info, out float distance)
    {
        info = default;
        float currentDistance = float.MaxValue;
        distance = float.MaxValue;

        HoveredAssetInfo? bestInfo = null;
        LiquidRenderer? liquidRenderer = _terrainManager?.LiquidRenderer;

        void ConsiderCandidate(HoveredAssetInfo candidateInfo, float candidateDistance)
        {
            if (!IsHoverPickDistanceAllowed(candidateDistance))
                return;

            if (candidateDistance < currentDistance)
            {
                bestInfo = candidateInfo;
                currentDistance = candidateDistance;
            }
        }

        if (_wmosVisible)
        {
            Vector3 padding = new(2f, 2f, 2f);
            for (int i = 0; i < _wmoInstances.Count; i++)
            {
                ObjectInstance inst = _wmoInstances[i];
                if (ShouldHideObjectInstanceByUniqueId(inst))
                    continue;

                if (!TryRayIntersectInstanceBounds(rayOrigin, rayDir, inst, padding, out float t))
                    continue;

                ConsiderCandidate(BuildHoveredObjectInfo("WMO", inst, ObjectType.Wmo, i), t);
            }
        }

        if (_doodadsVisible)
        {
            Vector3 padding = new(1f, 1f, 1f);
            for (int i = 0; i < _mdxInstances.Count; i++)
            {
                ObjectInstance inst = _mdxInstances[i];
                if (ShouldHideObjectInstanceByUniqueId(inst))
                    continue;

                if (!TryRayIntersectInstanceBounds(rayOrigin, rayDir, inst, padding, out float t))
                    continue;

                ConsiderCandidate(BuildHoveredObjectInfo("MDX", inst, ObjectType.Mdx, i), t);
            }
        }

        if (_showWlLiquids && _wlLoader != null)
        {
            Vector3 padding = new(2f, 2f, 1f);
            for (int i = 0; i < _wlLoader.Bodies.Count; i++)
            {
                WlLiquidBody body = _wlLoader.Bodies[i];
                if (liquidRenderer != null && !liquidRenderer.IsWlBodyVisible(body.BodyKey))
                    continue;

                float t = RayAABBIntersect(rayOrigin, rayDir, body.BoundsMin - padding, body.BoundsMax + padding);
                if (t < 0f)
                    continue;

                ConsiderCandidate(BuildHoveredWlLiquidInfo(body), t);
            }
        }

        if (!bestInfo.HasValue)
            return false;

        info = bestInfo.Value;
        distance = currentDistance;
        return true;
    }

    private bool TryBuildHoveredPm4InfoByRay(Vector3 rayOrigin, Vector3 rayDir, out HoveredAssetInfo info, out float distance)
    {
        info = default;
        distance = float.MaxValue;

        if (!TryPickPm4ObjectByRay(rayOrigin, rayDir, out var objectKey, out _, out float hitDistance) || !objectKey.HasValue)
            return false;

        if (!IsHoverPickDistanceAllowed(hitDistance))
            return false;

        if (!_pm4ObjectLookup.TryGetValue(objectKey.Value, out Pm4OverlayObject? obj))
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey.Value, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
        Vector3 center = applyObjectTransform ? ApplyPm4OverlayTransform(obj.Center, objectTransform) : obj.Center;

        info = BuildHoveredPm4Info(obj, center, objectKey.Value);
        distance = hitDistance;
        return true;
    }

    private bool ShouldPreferPm4HoverBrush(float pm4DistanceSq, float pm4Depth, float sceneDistanceSq, float sceneDepth)
    {
        if (_pm4OverlayIgnoreDepth)
            return true;

        const float depthEpsilon = 0.0025f;
        if (sceneDepth + depthEpsilon < pm4Depth)
            return false;

        if (pm4Depth + depthEpsilon < sceneDepth)
            return true;

        return pm4DistanceSq <= sceneDistanceSq;
    }

    public void ToggleObjects() => _objectsVisible = !_objectsVisible;
    public void ToggleWmos() => _wmosVisible = !_wmosVisible;
    public void ToggleDoodads() => _doodadsVisible = !_doodadsVisible;
    public bool ObjectFogEnabled
    {
        get => _objectFogEnabled;
        set => _objectFogEnabled = value;
    }

    public int SubObjectCount => 3;

    public string GetSubObjectName(int index) => index switch
    {
        0 => $"Terrain ({_terrainManager.LoadedChunkCount} chunks)",
        1 => $"WMOs ({_wmoInstances.Count} instances, {UniqueWmoModels} unique)",
        2 => $"Doodads ({_mdxInstances.Count} instances, {UniqueMdxModels} unique)",
        _ => ""
    };

    public bool GetSubObjectVisible(int index) => index switch
    {
        0 => true,
        1 => _wmosVisible,
        2 => _doodadsVisible,
        _ => false
    };

    public void SetSubObjectVisible(int index, bool visible)
    {
        switch (index)
        {
            case 1: _wmosVisible = visible; break;
            case 2: _doodadsVisible = visible; break;
        }
    }

    /// <summary>
    /// Select the nearest object whose AABB is hit by a ray from camera.
    /// Call with screen-space mouse coords to pick objects.
    /// </summary>
    public void SelectObjectByRay(Vector3 rayOrigin, Vector3 rayDir)
    {
        if (TryPickSceneObjectByRay(rayOrigin, rayDir, out ObjectType bestType, out int bestIndex, out _))
        {
            _selectedObjectType = bestType;
            _selectedObjectIndex = bestIndex;
            if (TryGetSceneObjectByIndex(bestType, bestIndex, out ObjectInstance selectedInstance))
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(bestType, selectedInstance);
            return;
        }

        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
        _selectedSceneObjectKey = null;
    }

    public bool SelectSceneObject(ObjectType objectType, int objectIndex)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        switch (objectType)
        {
            case ObjectType.Wmo when objectIndex >= 0 && objectIndex < _wmoInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(objectType, _wmoInstances[objectIndex]);
                return true;
            case ObjectType.Mdx when objectIndex >= 0 && objectIndex < _mdxInstances.Count:
                _selectedObjectType = objectType;
                _selectedObjectIndex = objectIndex;
                _selectedSceneObjectKey = CreateSelectedSceneObjectKey(objectType, _mdxInstances[objectIndex]);
                return true;
            default:
                return false;
        }
    }

    public bool TryPickSceneObjectByRay(Vector3 rayOrigin, Vector3 rayDir, out ObjectType objectType, out int objectIndex, out float distance)
    {
        var hits = new List<SceneObjectPickHit>();
        CollectSceneObjectPickHits(rayOrigin, rayDir, hits, logHits: true);

        if (hits.Count == 0)
        {
            objectType = ObjectType.None;
            objectIndex = -1;
            distance = float.MaxValue;
            return false;
        }

        SceneObjectPickHit bestHit = hits[0];
        objectType = bestHit.ObjectType;
        objectIndex = bestHit.ObjectIndex;
        distance = bestHit.Distance;
        return true;
    }

    public bool TryPickSceneObjectsByRay(Vector3 rayOrigin, Vector3 rayDir, List<SceneObjectPickHit> hits)
    {
        return TryPickSceneObjectsByRay(rayOrigin, rayDir, hits, null, null);
    }

    public bool TryPickSceneObjectsByRay(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey,
        Vector3? clickedWorldPoint)
    {
        ArgumentNullException.ThrowIfNull(hits);
        CollectSceneObjectPickHits(rayOrigin, rayDir, hits, logHits: false, clickedChunkKey, clickedWorldPoint);
        return hits.Count > 0;
    }

    private void CollectSceneObjectPickHits(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        bool logHits,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey = null,
        Vector3? clickedWorldPoint = null)
    {
        hits.Clear();

        if (_instancesDirty)
            RebuildInstanceLists();

        AppendSceneObjectPickHits(rayOrigin, rayDir, hits, _wmoInstances, ObjectType.Wmo, new Vector3(2f, 2f, 2f), clickedChunkKey, clickedWorldPoint);
        AppendSceneObjectPickHits(rayOrigin, rayDir, hits, _mdxInstances, ObjectType.Mdx, new Vector3(1f, 1f, 1f), clickedChunkKey, clickedWorldPoint);

        if (clickedChunkKey.HasValue && hits.Any(static hit => hit.SharesClickedChunk))
            hits.RemoveAll(static hit => !hit.SharesClickedChunk);

        hits.Sort(static (left, right) =>
        {
            int clickedChunkCompare = right.SharesClickedChunk.CompareTo(left.SharesClickedChunk);
            if (clickedChunkCompare != 0)
                return clickedChunkCompare;

            int chunkDistanceCompare = left.ChunkGridDistance.CompareTo(right.ChunkGridDistance);
            if (chunkDistanceCompare != 0)
                return chunkDistanceCompare;

            int centroidCompare = left.SelectionPointDistanceSq.CompareTo(right.SelectionPointDistanceSq);
            if (centroidCompare != 0)
                return centroidCompare;

            return left.Distance.CompareTo(right.Distance);
        });

        if (!logHits || hits.Count == 0)
            return;

        ViewerLog.Debug(ViewerLog.Category.Terrain, $"[ObjectPick] Ray hit {hits.Count} objects:");
        foreach (SceneObjectPickHit hit in hits.Take(5))
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  {hit.KindLabel}[{hit.ObjectIndex}] {hit.ModelName} @ dist={hit.Distance:F1}");
        if (hits.Count > 5)
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"  ... and {hits.Count - 5} more");
    }

    private void AppendSceneObjectPickHits(
        Vector3 rayOrigin,
        Vector3 rayDir,
        List<SceneObjectPickHit> hits,
        List<ObjectInstance> instances,
        ObjectType objectType,
        Vector3 padding,
        (int tileX, int tileY, int chunkX, int chunkY)? clickedChunkKey,
        Vector3? clickedWorldPoint)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            ObjectInstance instance = instances[i];
            if (ShouldHideObjectInstanceByUniqueId(instance))
                continue;

            if (!TryRayIntersectInstanceBounds(rayOrigin, rayDir, instance, padding, out float distance) || !IsHoverPickDistanceAllowed(distance))
                continue;

            Vector3 selectionPoint = GetSceneObjectSelectionPoint(instance);
            bool sharesClickedChunk = clickedChunkKey.HasValue
                && TryGetSceneObjectChunkKey(instance, out var instanceChunkKey)
                && instanceChunkKey == clickedChunkKey.Value;
            int chunkGridDistance = clickedChunkKey.HasValue && TryGetSceneObjectChunkKey(instance, out instanceChunkKey)
                ? Math.Abs(instanceChunkKey.tileX - clickedChunkKey.Value.tileX)
                    + Math.Abs(instanceChunkKey.tileY - clickedChunkKey.Value.tileY)
                    + Math.Abs(instanceChunkKey.chunkX - clickedChunkKey.Value.chunkX)
                    + Math.Abs(instanceChunkKey.chunkY - clickedChunkKey.Value.chunkY)
                : int.MaxValue;
            float selectionPointDistanceSq = clickedWorldPoint.HasValue
                ? Vector3.DistanceSquared(selectionPoint, clickedWorldPoint.Value)
                : float.MaxValue;

            hits.Add(new SceneObjectPickHit(
                objectType,
                i,
                distance,
                instance.ModelName,
                instance.ModelPath,
                instance.UniqueId,
                instance.PlacementPosition,
                instance.BoundsMin,
                instance.BoundsMax,
                selectionPoint,
                selectionPointDistanceSq,
                sharesClickedChunk,
                chunkGridDistance));
        }
    }

    private static Vector3 GetSceneObjectSelectionPoint(in ObjectInstance instance)
    {
        if (instance.BoundsResolved)
        {
            Vector3 boundsCenter = (instance.BoundsMin + instance.BoundsMax) * 0.5f;
            if (float.IsFinite(boundsCenter.X) && float.IsFinite(boundsCenter.Y) && float.IsFinite(boundsCenter.Z))
                return boundsCenter;
        }

        return instance.PlacementPosition;
    }

    private static bool TryGetSceneObjectChunkKey(in ObjectInstance instance, out (int tileX, int tileY, int chunkX, int chunkY) key)
    {
        Vector3 selectionPoint = GetSceneObjectSelectionPoint(instance);
        return TryGetTerrainChunkKey(selectionPoint.X, selectionPoint.Y, out key);
    }

    private static bool TryGetTerrainChunkKey(float worldX, float worldY, out (int tileX, int tileY, int chunkX, int chunkY) key)
    {
        key = default;

        float dx = WoWConstants.MapOrigin - worldX;
        float dy = WoWConstants.MapOrigin - worldY;
        if (float.IsNaN(dx) || float.IsNaN(dy) || float.IsInfinity(dx) || float.IsInfinity(dy))
            return false;

        int tileX = (int)MathF.Floor(dx / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor(dy / WoWConstants.ChunkSize);
        if (tileX < 0 || tileX >= WoWConstants.TilesPerMapEdge || tileY < 0 || tileY >= WoWConstants.TilesPerMapEdge)
            return false;

        float localX = dx - tileX * WoWConstants.ChunkSize;
        float localY = dy - tileY * WoWConstants.ChunkSize;
        float chunkSize = WoWConstants.ChunkSize / WoWConstants.ChunksPerTileEdge;

        int chunkY = Math.Clamp((int)MathF.Floor(localX / chunkSize), 0, WoWConstants.ChunksPerTileEdge - 1);
        int chunkX = Math.Clamp((int)MathF.Floor(localY / chunkSize), 0, WoWConstants.ChunksPerTileEdge - 1);

        key = (tileX, tileY, chunkX, chunkY);
        return true;
    }

    public bool SelectPm4ObjectByRay(Vector3 rayOrigin, Vector3 rayDir)
    {
        if (TryPickPm4ObjectByRay(rayOrigin, rayDir, out var bestKey, out var bestGroupKey, out _))
        {
            _selectedPm4ObjectKey = bestKey;
            _selectedPm4ObjectGroupKey = bestGroupKey;
            return true;
        }

        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
        return false;
    }

    public bool TryPickPm4ObjectByRay(
        Vector3 rayOrigin,
        Vector3 rayDir,
        out (int tileX, int tileY, uint ck24, int objectPart)? objectKey,
        out (int tileX, int tileY, uint ck24)? objectGroupKey,
        out float distance)
    {
        objectKey = null;
        objectGroupKey = null;
        distance = float.MaxValue;

        bool profile = Pm4Profiling.Enabled;
        System.Diagnostics.Stopwatch sw = profile ? s_pm4PickSw : null;
        long beforeTicks = profile ? sw.ElapsedTicks : 0;
        int aabbHits = 0;

        if (!_showPm4Overlay || _pm4TileObjects.Count == 0)
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Vector3 padding = new(2f, 2f, 2f);
        float bestT = float.MaxValue;

        // Single pass: test every object's AABB directly (simple and reliable)
        foreach (var (tileKey, objects) in _pm4TileObjects)
        {
            if (!ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                continue;

            foreach (Pm4OverlayObject obj in objects)
            {
                if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                    continue;

                var candidateKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(candidateKey, applyPm4Transform, pm4Transform, out bool applyObjTransform);

                Vector3 bmin = obj.BoundsMin, bmax = obj.BoundsMax;
                if (applyObjTransform)
                    TransformBounds(bmin, bmax, objectTransform, out bmin, out bmax);

                float t = RayAABBIntersect(rayOrigin, rayDir, bmin - padding, bmax + padding);
                if (t >= 0f)
                    aabbHits++;
                if (t >= 0f && t < bestT && IsHoverPickDistanceAllowed(t))
                {
                    bestT = t;
                    objectKey = candidateKey;
                    objectGroupKey = ResolvePm4ObjectGroupKey(candidateKey);
                }
            }
        }

        distance = bestT;
        bool hit = objectKey.HasValue;

        if (profile)
        {
            long afterTicks = sw.ElapsedTicks;
            double elapsedMs = (afterTicks - beforeTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            s_pm4PickCallCount++;
            s_pm4PickAabbHitCount += aabbHits;
            s_pm4PickTotalMs += elapsedMs;
            if (elapsedMs > s_pm4PickMaxMs) s_pm4PickMaxMs = elapsedMs;
            s_pm4PickReportCount++;
            if (elapsedMs >= 50.0 || s_pm4PickReportCount >= 200)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[PM4-PROFILE] TryPickPm4ObjectByRay: call={s_pm4PickCallCount} last={elapsedMs:0.0}ms max={s_pm4PickMaxMs:0.0}ms avg={s_pm4PickTotalMs / s_pm4PickCallCount:0.0}ms aabbHits(last)={aabbHits} totalHits={s_pm4PickAabbHitCount} hit={hit}");
                s_pm4PickReportCount = 0;
            }
        }

        return hit;
    }

    public void ClearSelection()
    {
        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
        _selectedSceneObjectKey = null;
    }

    public void ClearPm4ObjectSelection()
    {
        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
    }

    public bool SelectPm4ObjectGroupKey(uint regionId, ushort ck24ObjectId)
    {
        foreach (var kv in _pm4ObjectLookup)
        {
            var (tx, ty, ck24, part) = kv.Key;
            if (kv.Value.MshdRegionId == regionId && (ushort)(ck24 & 0xFFFF) == ck24ObjectId)
            {
                var key = (tx, ty, ck24, part);
                _selectedPm4ObjectKey = key;
                _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(key);
                return true;
            }
        }
        return false;
    }

    private float ComputeEffectiveHoveredAssetMaxDistance()
    {
        if (!_limitHoveredAssetRange)
            return float.MaxValue;

        if (!_useDynamicHoveredAssetRange)
            return _hoveredAssetMaxDistance;

        float fogDrivenDistance = Math.Clamp(_lastHoverPickFogEnd * 0.4f, 533.33f, MaxWorldObjectViewDistance);
        return Math.Min(_hoveredAssetMaxDistance, fogDrivenDistance);
    }

    private bool IsHoverPickDistanceAllowed(float distance)
    {
        if (!_limitHoveredAssetRange)
            return true;

        return distance <= ComputeEffectiveHoveredAssetMaxDistance();
    }

    private bool IsHoverPickPositionAllowed(Vector3 worldPosition)
    {
        if (!_limitHoveredAssetRange)
            return true;

        Vector3 cameraPosition = GetPm4LoadAnchorCameraPosition();
        return Vector3.Distance(cameraPosition, worldPosition) <= ComputeEffectiveHoveredAssetMaxDistance();
    }

    public bool SelectPm4Object((int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        if (!_pm4ObjectLookup.ContainsKey(objectKey))
            return false;

        _selectedPm4ObjectKey = objectKey;
        _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(objectKey);
        return true;
    }

    public bool TryGetPm4ObjectGroupKey(
        (int tileX, int tileY, uint ck24, int objectPart) objectKey,
        out (int tileX, int tileY, uint ck24) groupKey)
    {
        if (!_pm4ObjectLookup.ContainsKey(objectKey))
        {
            groupKey = default;
            return false;
        }

        groupKey = ResolvePm4ObjectGroupKey(objectKey);
        return true;
    }

    public void SetHighlightedPm4Objects(IEnumerable<(int tileX, int tileY, uint ck24, int objectPart)> objectKeys)
    {
        _highlightedPm4ObjectKeys.Clear();
        foreach (var objectKey in objectKeys)
        {
            if (_pm4ObjectLookup.ContainsKey(objectKey))
                _highlightedPm4ObjectKeys.Add(objectKey);
        }
    }

    public bool TryGetPm4ObjectDebugInfo((int tileX, int tileY, uint ck24, int objectPart) objectKey, out Pm4ObjectDebugInfo info)
    {
        info = default;
        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;
        Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

        Vector3 center = applyObjectTransform ? ApplyPm4OverlayTransform(obj.Center, objectTransform) : obj.Center;
        Vector3 boundsMin = obj.BoundsMin;
        Vector3 boundsMax = obj.BoundsMax;
        if (applyObjectTransform)
            TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

        float nearestPositionRefDistance = float.NaN;
        if (_pm4TilePositionRefs.TryGetValue((objectKey.tileX, objectKey.tileY), out List<Vector3>? positionRefs)
            && positionRefs.Count > 0)
        {
            nearestPositionRefDistance = NearestPointDistance(center, positionRefs, applyPm4Transform, pm4Transform);
        }

        info = new Pm4ObjectDebugInfo(
            obj.Ck24,
            obj.Ck24Type,
            obj.Ck24ObjectId,
            obj.ObjectPartId,
            obj.LinkGroupObjectId,
            obj.LinkedPositionRefCount,
            obj.LinkedPositionRefSummary,
            objectKey.tileX,
            objectKey.tileY,
            obj.MshdField00,
            obj.MshdRegionId,
            obj.MshdField08,
            obj.SurfaceCount,
            obj.DominantGroupKey,
            obj.DominantAttributeMask,
            obj.DominantMscnRefIndex,
            obj.AverageSurfaceHeight,
            boundsMin,
            boundsMax,
            center,
            nearestPositionRefDistance,
            obj.PlanarTransform.SwapPlanarAxes,
            obj.PlanarTransform.InvertU,
            obj.PlanarTransform.InvertV,
            obj.PlanarTransform.InvertsWinding,
            obj.DistinctTypeFlags);

        return true;
    }

    public bool TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo info)
    {
        info = default;
        if (!_selectedPm4ObjectKey.HasValue)
            return false;

        return TryGetPm4ObjectDebugInfo(_selectedPm4ObjectKey.Value, out info);
    }

    public bool TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo info)
    {
        info = default;
        if (!_selectedPm4ObjectKey.HasValue)
            return false;

        bool profile = Pm4Profiling.Enabled;
        long researchStartTicks = profile ? s_pm4ResearchSw.ElapsedTicks : 0;

        var objectKey = _selectedPm4ObjectKey.Value;
        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        if (string.IsNullOrWhiteSpace(obj.SourcePath))
            return false;

        if (!TryGetPm4ResearchContext(obj.SourcePath, out Pm4ResearchContext? context) || context == null)
            return false;

        List<Pm4ResearchHypothesisMatch> allMatches = context.HypothesisReport.Objects
            .Where(hypothesis => hypothesis.Ck24 == obj.Ck24)
            .Select(hypothesis => new Pm4ResearchHypothesisMatch(
                hypothesis.Family,
                hypothesis.FamilyObjectIndex,
                hypothesis.SurfaceCount,
                hypothesis.TotalIndexCount,
                hypothesis.MscnRefIndices.Count,
                hypothesis.GroupKeys.Count,
                hypothesis.MslkGroupObjectIds.Count,
                hypothesis.DominantLinkGroupObjectId,
                hypothesis.MprlFootprint.LinkedRefCount,
                hypothesis.MprlFootprint.LinkedInBoundsCount,
                hypothesis.PlacementComparison.CoordinateMode,
                hypothesis.PlacementComparison.PlanarTransform,
                hypothesis.PlacementComparison.FrameYawDegrees,
                hypothesis.PlacementComparison.MprlHeadingMeanDegrees,
                hypothesis.PlacementComparison.HeadingDeltaDegrees,
                ComputePm4ResearchMatchScore(obj, hypothesis)))
            .OrderBy(match => match.SimilarityScore)
            .ThenBy(match => match.Family)
            .ThenBy(match => match.FamilyObjectIndex)
            .ToList();

        int invalidRefIndexCount = context.DecodeAudit.ReferenceAudits
            .Where(static audit => audit.Name == "MSLK.RefIndex->MSUR")
            .Select(static audit => audit.InvalidCount)
            .FirstOrDefault();

        // Extract raw MSHD header fields
        string? mshdRawFields = null;
        IReadOnlyList<string>? mslkRawEntries = null;
        if (context.RawDocument != null)
        {
            var knownMshd = context.RawDocument.KnownChunks.Mshd;
            if (knownMshd is not null)
            {
                mshdRawFields = $"MSHD: F00={knownMshd.Field00} F04={knownMshd.Field04} F08={knownMshd.Field08} F0C={knownMshd.Field0C} F10={knownMshd.Field10} F14={knownMshd.Field14} F18={knownMshd.Field18} F1C={knownMshd.Field1C}";
            }

            // Collect MSLK entries referencing the selected object's surfaces by CK24
            var mslkLines = new List<string>();
            foreach (MslkEntry mslk in context.RawDocument.KnownChunks.Mslk)
            {
                if (mslk.RefIndex >= 0 && (uint)mslk.RefIndex < (uint)context.RawDocument.KnownChunks.Msur.Count
                    && context.RawDocument.KnownChunks.Msur[mslk.RefIndex].Ck24 == obj.Ck24)
                {
                    mslkLines.Add($"MSLK: TypeFlags=0x{mslk.TypeFlags:X2} Subtype=0x{mslk.Subtype:X2} Padding=0x{mslk.Padding:X4} GroupObjectId=0x{mslk.GroupObjectId:X8} MspiFirstIndex={mslk.MspiFirstIndex} MspiIndexCount={mslk.MspiIndexCount} LinkId=0x{mslk.LinkId:X8} RefIndex={mslk.RefIndex} SystemFlag=0x{mslk.SystemFlag:X4}");
                }
            }
            if (mslkLines.Count > 0)
                mslkRawEntries = mslkLines;
        }

        info = new Pm4SelectedObjectResearchInfo(
            obj.SourcePath,
            context.Snapshot.Version,
            context.Snapshot.MslkCount,
            context.Snapshot.MsurCount,
            context.Snapshot.MscnCount,
            context.Snapshot.MprlCount,
            invalidRefIndexCount,
            context.HypothesisReport.TotalHypothesisCount,
            allMatches.Count,
            context.Snapshot.Diagnostics.Count,
            context.Snapshot.Diagnostics.Take(3).ToList(),
            allMatches.Take(8).ToList(),
            mshdRawFields,
            mslkRawEntries);

        if (profile)
        {
            long afterTicks = s_pm4ResearchSw.ElapsedTicks;
            double elapsedMs = (afterTicks - researchStartTicks) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            s_pm4ResearchCallCount++;
            s_pm4ResearchTotalMs += elapsedMs;
            if (elapsedMs > s_pm4ResearchMaxMs) s_pm4ResearchMaxMs = elapsedMs;
            s_pm4ResearchReportCount++;
            int mslkLines = mslkRawEntries?.Count ?? 0;
            int matchesCount = allMatches.Count;
            if (elapsedMs >= 50.0 || s_pm4ResearchReportCount >= 200)
            {
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[PM4-PROFILE] TryGetSelectedPm4ObjectResearchInfo: call={s_pm4ResearchCallCount} last={elapsedMs:0.0}ms max={s_pm4ResearchMaxMs:0.0}ms avg={s_pm4ResearchTotalMs / s_pm4ResearchCallCount:0.0}ms matches={matchesCount} mslkLines={mslkLines} mslkTotal={context.RawDocument?.KnownChunks.Mslk.Count ?? 0}");
                s_pm4ResearchReportCount = 0;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes MSLK linking statistics across all loaded PM4 research contexts.
    /// Exposed as a plain record so the viewer never needs the internal context type.
    /// </summary>
    public Pm4MslkLinkingStats GetPm4MslkLinkingStats()
    {
        int totalFiles = 0;
        int totalMslkEntries = 0;
        int anchorOnlyLinks = 0;
        int pathWindowLinks = 0;
        int totalComponents = 0;
        int componentsWithLinks = 0;
        int componentsWithoutLinks = 0;
        int refIndexMismatches = 0;

        foreach ((string _, Pm4ResearchContext context) in _pm4ResearchBySourcePath)
        {
            if (context.RawDocument == null)
                continue;

            totalFiles++;
            var chunks = context.RawDocument.KnownChunks;
            totalMslkEntries += chunks.Mslk.Count;

            foreach (var link in chunks.Mslk)
            {
                if (link.MspiFirstIndex < 0)
                    anchorOnlyLinks++;
                else
                    pathWindowLinks++;
            }

            var linksBySurface = new Dictionary<int, List<MslkEntry>>();
            foreach (var link in chunks.Mslk)
            {
                if (link.RefIndex >= 0 && link.RefIndex < chunks.Msur.Count)
                {
                    if (!linksBySurface.TryGetValue(link.RefIndex, out var bucket))
                        linksBySurface[link.RefIndex] = bucket = new List<MslkEntry>();
                    bucket.Add(link);
                }
                else
                {
                    refIndexMismatches++;
                }
            }

            var surfacesByCk24 = new Dictionary<uint, List<int>>();
            for (int i = 0; i < chunks.Msur.Count; i++)
            {
                uint ck24 = chunks.Msur[i].Ck24;
                if (!surfacesByCk24.TryGetValue(ck24, out var bucket))
                    surfacesByCk24[ck24] = bucket = new List<int>();
                bucket.Add(i);
            }

            foreach ((uint _, List<int> surfaceIndices) in surfacesByCk24)
            {
                totalComponents++;
                bool hasLink = false;
                foreach (int si in surfaceIndices)
                {
                    if (linksBySurface.ContainsKey(si))
                    {
                        hasLink = true;
                        break;
                    }
                }
                if (hasLink) componentsWithLinks++;
                else componentsWithoutLinks++;
            }
        }

        return new Pm4MslkLinkingStats(
            totalFiles,
            totalMslkEntries,
            anchorOnlyLinks,
            pathWindowLinks,
            totalComponents,
            componentsWithLinks,
            componentsWithoutLinks,
            refIndexMismatches);
    }

    private bool TryGetPm4ResearchContext(string sourcePath, out Pm4ResearchContext? context)
    {
        if (_pm4ResearchBySourcePath.TryGetValue(sourcePath, out context))
            return true;

        if (_pm4ResearchUnavailablePaths.Contains(sourcePath) || _dataSource == null)
        {
            context = null;
            return false;
        }

        byte[]? bytes = _dataSource.ReadFile(sourcePath);
        if (bytes == null || bytes.Length == 0)
        {
            _pm4ResearchUnavailablePaths.Add(sourcePath);
            context = null;
            return false;
        }

        try
        {
            Pm4File researchFile = CorePm4DocumentReader.Read(bytes, sourcePath);
            context = new Pm4ResearchContext(
                sourcePath,
                CorePm4ResearchSnapshotBuilder.CreateSnapshot(researchFile),
                CorePm4ResearchAuditAnalyzer.Analyze(researchFile),
                CorePm4ResearchHierarchyAnalyzer.Analyze(researchFile),
                researchFile);
            _pm4ResearchBySourcePath[sourcePath] = context;
            return true;
        }
        catch (Exception ex)
        {
            _pm4ResearchUnavailablePaths.Add(sourcePath);
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4 Research] Failed to analyze '{sourcePath}': {ex.Message}");
            context = null;
            return false;
        }
    }

    private byte[]? ReadPm4FileForTile((int tileX, int tileY) tileKey)
    {
        if (_dataSource == null)
            return null;

        // Collect unique source paths for objects on this tile
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            if (objectKey.tileX == tileKey.tileX && objectKey.tileY == tileKey.tileY
                && !string.IsNullOrWhiteSpace(obj.SourcePath)
                && seenPaths.Add(obj.SourcePath))
            {
                byte[]? bytes = _dataSource.ReadFile(obj.SourcePath);
                if (bytes != null && bytes.Length > 0)
                    return bytes;
            }
        }

        return null;
    }

    /// <summary>
    /// Lazily populate <see cref="_pm4TileMscnPoints"/> from the staged PM4 files.
    /// MSCN = scene-graph connector anchors. One Vector3 per MSUR surface (placed via MSUR.MscnRefIndex).
    /// </summary>
    private void EnsurePm4MscnData()
    {
        if (_pm4TileObjects.Count == 0)
            return;
        foreach (var tileKey in _pm4TileObjects.Keys.ToList())
        {
            if (_pm4TileMscnPoints.ContainsKey(tileKey))
                continue;
            var bytes = ReadPm4FileForTile(tileKey);
            if (bytes == null) continue;
            var pm4 = CorePm4DocumentReader.Read(bytes, $"tile_{tileKey.tileX}_{tileKey.tileY}.pm4");
            if (pm4.KnownChunks.Mscn.Count == 0) continue;
            var pts = new List<Vector3>(pm4.KnownChunks.Mscn.Count);
            foreach (var p in pm4.KnownChunks.Mscn)
                pts.Add(new Vector3(WoWConstants.MapOrigin - p.X, WoWConstants.MapOrigin - p.Y, p.Z));
            _pm4TileMscnPoints[tileKey] = pts;
        }
    }

    /// <summary>
    /// Lazily populate <see cref="_pm4TileMspvPoints"/> from the staged PM4 files.
    /// MSPV = path-vertex positions reached via MSPI from MSLK link records. Only present when surfaces are connected.
    /// </summary>
    private void EnsurePm4MspvData()
    {
        if (_pm4TileObjects.Count == 0)
            return;
        foreach (var tileKey in _pm4TileObjects.Keys.ToList())
        {
            if (_pm4TileMspvPoints.ContainsKey(tileKey))
                continue;
            var bytes = ReadPm4FileForTile(tileKey);
            if (bytes == null) continue;
            var pm4 = CorePm4DocumentReader.Read(bytes, $"tile_{tileKey.tileX}_{tileKey.tileY}.pm4");
            if (pm4.KnownChunks.Mspv.Count == 0) continue;
            var pts = new List<Vector3>(pm4.KnownChunks.Mspv.Count);
            foreach (var p in pm4.KnownChunks.Mspv)
                pts.Add(new Vector3(WoWConstants.MapOrigin - p.X, WoWConstants.MapOrigin - p.Y, p.Z));
            _pm4TileMspvPoints[tileKey] = pts;
        }
    }

    private static float ComputePm4ResearchMatchScore(Pm4OverlayObject obj, CorePm4ObjectHypothesis hypothesis)
    {
        float score = 0f;
        score += Math.Abs(hypothesis.SurfaceCount - obj.SurfaceCount) * 3f;
        score += Math.Abs(hypothesis.TotalIndexCount - obj.TotalIndexCount) * 0.125f;
        score += Math.Abs(hypothesis.MprlFootprint.LinkedRefCount - obj.LinkedPositionRefCount) * 4f;

        if (obj.LinkGroupObjectId != 0)
        {
            bool hasExactGroupObjectId = hypothesis.MslkGroupObjectIds.Contains(obj.LinkGroupObjectId);
            score += hasExactGroupObjectId ? -8f : 8f;
            if (hypothesis.DominantLinkGroupObjectId == obj.LinkGroupObjectId)
                score -= 4f;
        }

        return Math.Max(0f, score);
    }

    private static float NearestPointDistance(Vector3 point, IReadOnlyList<Vector3> candidates, bool applyPm4Transform, in Matrix4x4 pm4Transform)
    {
        float best = float.MaxValue;
        for (int i = 0; i < candidates.Count; i++)
        {
            Vector3 candidate = applyPm4Transform ? ApplyPm4OverlayTransform(candidates[i], pm4Transform) : candidates[i];
            float dist = Vector3.Distance(point, candidate);
            if (dist < best)
                best = dist;
        }

        return best;
    }

    /// <summary>
    /// Ray-AABB slab intersection test. Returns distance along ray, or -1 if no hit.
    /// </summary>
    private static float RayAABBIntersect(Vector3 origin, Vector3 dir, Vector3 bmin, Vector3 bmax)
    {
        float tmin = float.NegativeInfinity;
        float tmax = float.PositiveInfinity;

        for (int i = 0; i < 3; i++)
        {
            float o = i == 0 ? origin.X : i == 1 ? origin.Y : origin.Z;
            float d = i == 0 ? dir.X : i == 1 ? dir.Y : dir.Z;
            float lo = i == 0 ? bmin.X : i == 1 ? bmin.Y : bmin.Z;
            float hi = i == 0 ? bmax.X : i == 1 ? bmax.Y : bmax.Z;

            if (MathF.Abs(d) < 1e-8f)
            {
                if (o < lo || o > hi) return -1;
            }
            else
            {
                float t1 = (lo - o) / d;
                float t2 = (hi - o) / d;
                if (t1 > t2) (t1, t2) = (t2, t1);
                tmin = MathF.Max(tmin, t1);
                tmax = MathF.Min(tmax, t2);
                if (tmin > tmax) return -1;
            }
        }

        return tmin >= 0 ? tmin : tmax >= 0 ? tmax : -1;
    }

    private static bool TryRayIntersectInstanceBounds(Vector3 origin, Vector3 dir, in ObjectInstance instance, Vector3 padding, out float distance)
    {
        if (instance.BoundsResolved
            && Matrix4x4.Invert(instance.Transform, out Matrix4x4 inverseTransform))
        {
            Vector3 localOrigin = Vector3.Transform(origin, inverseTransform);
            Vector3 localDirection = Vector3.TransformNormal(dir, inverseTransform);

            if (localDirection.LengthSquared() > 1e-10f)
            {
                float localT = RayAABBIntersect(
                    localOrigin,
                    localDirection,
                    instance.LocalBoundsMin - padding,
                    instance.LocalBoundsMax + padding);

                if (localT >= 0f)
                {
                    Vector3 localHit = localOrigin + (localDirection * localT);
                    Vector3 worldHit = Vector3.Transform(localHit, instance.Transform);
                    distance = Vector3.Distance(origin, worldHit);
                    return true;
                }
            }
        }

        distance = RayAABBIntersect(origin, dir, instance.BoundsMin - padding, instance.BoundsMax + padding);
        return distance >= 0f;
    }

    private void PopulateWireframeRevealHits(List<ObjectInstance> instances, List<int> hitIndices,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight)
    {
        for (int i = 0; i < instances.Count; i++)
        {
            if (ShouldRevealInstance(instances[i], view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight))
                hitIndices.Add(i);
        }
    }

    private static bool ShouldRevealInstance(ObjectInstance inst, Matrix4x4 view, Matrix4x4 proj,
        float mouseViewportX, float mouseViewportY, float viewportWidth, float viewportHeight)
    {
        return TryMeasureHoverBrushHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out _, out _);
    }

    private static bool TryMeasureHoverInfoHit(Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight, out float distanceSq, out float depth)
    {
        return TryMeasureScreenBrushHit(
            boundsMin,
            boundsMax,
            view,
            proj,
            mouseViewportX,
            mouseViewportY,
            viewportWidth,
            viewportHeight,
            HoverInfoBrushPixels,
            HoverInfoMaxScreenRadius,
            out distanceSq,
            out depth);
    }

    private static bool TryMeasureHoverBrushHit(Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight, out float distanceSq, out float depth)
    {
        return TryMeasureScreenBrushHit(
            boundsMin,
            boundsMax,
            view,
            proj,
            mouseViewportX,
            mouseViewportY,
            viewportWidth,
            viewportHeight,
            WireframeRevealBrushPixels,
            WireframeRevealMaxScreenRadius,
            out distanceSq,
            out depth);
    }

    private static bool TryMeasureScreenBrushHit(Vector3 boundsMin, Vector3 boundsMax,
        Matrix4x4 view, Matrix4x4 proj, float mouseViewportX, float mouseViewportY,
        float viewportWidth, float viewportHeight, float brushPixels, float maxScreenRadius,
        out float distanceSq, out float depth)
    {
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        if (!TryProjectToViewport(center, view, proj, viewportWidth, viewportHeight, out float sx, out float sy, out depth))
        {
            distanceSq = 0f;
            return false;
        }

        float dx = sx - mouseViewportX;
        float dy = sy - mouseViewportY;
        distanceSq = dx * dx + dy * dy;

        float worldRadius = MathF.Max((boundsMax - boundsMin).Length() * 0.5f, 4f);
        float projectedRadius = EstimateProjectedRadius(worldRadius, depth, proj, viewportHeight);
        float revealRadius = MathF.Min(brushPixels + projectedRadius, maxScreenRadius);
        return distanceSq <= revealRadius * revealRadius;
    }

    private static HoveredAssetInfo BuildHoveredObjectInfo(string assetKind, in ObjectInstance inst, ObjectType objectType, int objectIndex)
    {
        return new HoveredAssetInfo(
            assetKind,
            inst.ModelName,
            inst.ModelPath,
            $"UniqueId: {inst.UniqueId}",
            inst.PlacementPosition,
            0,
            null,
            objectType,
                objectIndex,
                null);
    }

    private static HoveredAssetInfo BuildHoveredWlLiquidInfo(WlLiquidBody body)
    {
        Vector3 worldPosition = (body.BoundsMin + body.BoundsMax) * 0.5f;
        return new HoveredAssetInfo(
            "WL liquid",
            body.Name,
            body.SourcePath,
            $"{body.FileType} • {body.GroupLabel} • {body.BlockCount} blocks • Z {body.MinHeight:F1}..{body.MaxHeight:F1}",
            worldPosition,
            0,
                null,
                ObjectType.None,
                -1,
                body.BodyKey);
    }

    private static HoveredAssetInfo BuildHoveredPm4Info(Pm4OverlayObject obj, Vector3 worldPosition, (int tileX, int tileY, uint ck24, int objectPart) objectKey)
    {
        return new HoveredAssetInfo(
            "PM4",
            $"CK24 0x{obj.Ck24:X6} part={obj.ObjectPartId}",
            obj.SourcePath,
            $"type=0x{obj.Ck24Type:X2} obj={obj.Ck24ObjectId} region={obj.MshdRegionId} mslk=0x{obj.LinkGroupObjectId:X8} surfaces={obj.SurfaceCount}",
            worldPosition,
            0,
                objectKey,
                ObjectType.None,
                -1,
                null);
    }

    private bool TryBuildHoveredPm4Info(
        Matrix4x4 view,
        Matrix4x4 proj,
        float mouseViewportX,
        float mouseViewportY,
        float viewportWidth,
        float viewportHeight,
        out HoveredAssetInfo info,
        out int hitCount,
        out float bestDistanceSq,
        out float bestDepth)
    {
        info = default;
        hitCount = 0;
        bestDistanceSq = float.MaxValue;
        bestDepth = float.MaxValue;
        HoveredAssetInfo? bestInfo = null;
        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;

        foreach (KeyValuePair<(int tileX, int tileY), List<Pm4OverlayObject>> tileEntry in _pm4TileObjects)
        {
            List<Pm4OverlayObject> objects = tileEntry.Value;
            for (int i = 0; i < objects.Count; i++)
            {
                Pm4OverlayObject obj = objects[i];
                if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                    continue;

                var objectKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

                Vector3 boundsMin = obj.BoundsMin;
                Vector3 boundsMax = obj.BoundsMax;
                Vector3 center = obj.Center;
                if (applyObjectTransform)
                {
                    TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);
                    center = ApplyPm4OverlayTransform(obj.Center, objectTransform);
                }

                if (!TryMeasureHoverInfoHit(boundsMin, boundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                if (!IsHoverPickPositionAllowed(center))
                    continue;

                hitCount++;
                const float distanceEpsilon = 0.01f;
                if (!bestInfo.HasValue
                    || distanceSq < bestDistanceSq - distanceEpsilon
                    || (MathF.Abs(distanceSq - bestDistanceSq) <= distanceEpsilon && depth < bestDepth))
                {
                    bestDistanceSq = distanceSq;
                    bestDepth = depth;
                    bestInfo = BuildHoveredPm4Info(obj, center, objectKey);
                }
            }
        }

        if (!bestInfo.HasValue)
            return false;

        info = bestInfo.Value;
        return true;
    }

    internal bool TryBuildPm4ObjectMatch((int tileX, int tileY, uint ck24, int objectPart) objectKey, int maxMatchesPerObject, out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        EnsurePm4OverlayMatchesCameraWindow(GetPm4LoadAnchorCameraPosition());

        if (_instancesDirty)
            RebuildInstanceLists();

        if (!_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
            return false;

        Pm4ObjectMatchState pm4Object = BuildPm4ObjectMatchState(objectKey.tileX, objectKey.tileY, objectKey, obj);
        List<Pm4PlacementMatchState> placements = BuildPm4PlacementMatchStates();
        List<Pm4AssetProfileState> assetProfiles = BuildPm4AssetProfileStates(placements);
        objectMatch = BuildPm4ObjectMatchObject(pm4Object, placements, assetProfiles, Math.Max(1, maxMatchesPerObject));
        return true;
    }

    private static bool TryProjectToViewport(Vector3 worldPos, Matrix4x4 view, Matrix4x4 proj,
        float viewportWidth, float viewportHeight, out float sx, out float sy, out float depth)
    {
        var viewSpace = Vector4.Transform(new Vector4(worldPos, 1f), view);
        depth = MathF.Abs(viewSpace.Z);
        if (depth < 0.001f)
        {
            sx = sy = 0f;
            return false;
        }

        var clip = Vector4.Transform(new Vector4(worldPos, 1f), view * proj);
        if (clip.W <= 0f)
        {
            sx = sy = 0f;
            return false;
        }

        float ndcX = clip.X / clip.W;
        float ndcY = clip.Y / clip.W;
        sx = (ndcX * 0.5f + 0.5f) * viewportWidth;
        sy = (1f - (ndcY * 0.5f + 0.5f)) * viewportHeight;
        return true;
    }

    private static float EstimateProjectedRadius(float worldRadius, float depth, Matrix4x4 proj, float viewportHeight)
    {
        float yScale = MathF.Abs(proj.M22);
        if (yScale < 0.0001f)
            return 0f;

        return MathF.Min((worldRadius * yScale / depth) * (viewportHeight * 0.5f), WireframeRevealMaxScreenRadius);
    }

    private (int PreparedPrimitiveCount, int SubmittedPrimitiveCount) RenderWireframeReveal(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (_wireframeRevealWmoIndices.Count == 0 && _wireframeRevealMdxIndices.Count == 0)
            return (0, 0);

        int preparedPrimitiveCount = _wireframeRevealWmoIndices.Count + _wireframeRevealMdxIndices.Count;
        int submittedPrimitiveCount = 0;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.Blend);

        foreach (int idx in _wireframeRevealWmoIndices)
        {
            if ((uint)idx >= (uint)_wmoInstances.Count)
                continue;

            var inst = _wmoInstances[idx];
            var renderer = TryGetQueuedWmo(inst.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(inst.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            submittedPrimitiveCount++;
        }

        foreach (int idx in _wireframeRevealMdxIndices)
        {
            if ((uint)idx >= (uint)_mdxInstances.Count)
                continue;

            var inst = _mdxInstances[idx];
            var renderer = TryGetQueuedMdx(inst.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(inst.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            submittedPrimitiveCount++;
        }

        _gl.DepthMask(true);
        _gl.DepthFunc(DepthFunction.Lequal);
        return (preparedPrimitiveCount, submittedPrimitiveCount);
    }

    private (int PreparedPrimitiveCount, int SubmittedPrimitiveCount) RenderVisibleObjectWireframeOverlay(WorldRenderFrame frame, Matrix4x4 view, Matrix4x4 proj,
        Vector3 cameraPos, Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (frame.Visibility.VisibleWmos.Count == 0 && frame.Visibility.VisibleMdx.Count == 0)
            return (0, 0);

        int preparedPrimitiveCount = frame.Visibility.VisibleWmos.Count + frame.Visibility.VisibleMdx.Count;
        int submittedPrimitiveCount = 0;

        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(false);
        _gl.Disable(EnableCap.Blend);

        foreach (VisibleWmoInstance visible in frame.Visibility.VisibleWmos)
        {
            WmoRenderer? renderer = ResolveVisibleWmoRenderer(frame, visible.Instance.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(visible.Instance.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            submittedPrimitiveCount++;
        }

        foreach (VisibleMdxInstance visible in frame.Visibility.VisibleMdx)
        {
            IModelRenderer? renderer = ResolveVisibleMdxRenderer(frame, visible.Instance.ModelKey);
            if (renderer == null)
                continue;

            renderer.RenderWireframeOverlay(visible.Instance.Transform, view, proj,
                fogColor, fogStart, fogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            submittedPrimitiveCount++;
        }

        _gl.DepthMask(true);
        _gl.DepthFunc(DepthFunction.Lequal);
        return (preparedPrimitiveCount, submittedPrimitiveCount);
    }

    /// <summary>
    /// Build a world-space ray from normalized device coordinates using view/proj matrices.
    /// </summary>
    public static (Vector3 origin, Vector3 dir) ScreenToRay(float ndcX, float ndcY, Matrix4x4 view, Matrix4x4 proj)
    {
        Matrix4x4.Invert(proj, out var invProj);
        Matrix4x4.Invert(view, out var invView);

        // Near point in clip space → world
        var nearClip = new Vector4(ndcX, ndcY, -1f, 1f);
        var nearView = Vector4.Transform(nearClip, invProj);
        nearView /= nearView.W;
        var nearWorld = Vector4.Transform(nearView, invView);

        // Far point in clip space → world
        var farClip = new Vector4(ndcX, ndcY, 1f, 1f);
        var farView = Vector4.Transform(farClip, invProj);
        farView /= farView.W;
        var farWorld = Vector4.Transform(farView, invView);

        var origin = new Vector3(nearWorld.X, nearWorld.Y, nearWorld.Z);
        var farPt = new Vector3(farWorld.X, farWorld.Y, farWorld.Z);
        var dir = Vector3.Normalize(farPt - origin);
        return (origin, dir);
    }

    private Vector3 GetPm4ObjectColor((int tileX, int tileY) tileKey, Pm4OverlayObject obj)
    {
        return _pm4ColorMode switch
        {
            Pm4OverlayColorMode.Ck24ObjectId => ColorFromSeed(obj.Ck24ObjectId),
            Pm4OverlayColorMode.Ck24Key => ColorFromSeed(obj.Ck24),
            Pm4OverlayColorMode.Tile => ColorFromSeed((uint)HashCode.Combine(tileKey.tileX, tileKey.tileY)),
            Pm4OverlayColorMode.MshdRegionId => ColorFromSeed(obj.MshdRegionId),
            Pm4OverlayColorMode.GroupKey => ColorFromSeed(obj.DominantGroupKey),
            Pm4OverlayColorMode.AttributeMask => ColorFromSeed(obj.DominantAttributeMask),
            Pm4OverlayColorMode.Height => ColorFromHeight(obj.Center.Z),
            Pm4OverlayColorMode.TypeFlags => BlendTypeFlagColors(obj.DistinctTypeFlags),
            Pm4OverlayColorMode.Ck24TypeVsTypeFlags => GetCk24TypeVsTypeFlagsColor(obj.Ck24Type, obj.DistinctTypeFlags),
            _ => GetPm4TypeColor(obj.Ck24Type)
        };
    }


            public bool TryGetSelectedPm4ObjectGraphInfo(out Pm4SelectedObjectGraphInfo info)
            {
                info = default;
                if (!_selectedPm4ObjectKey.HasValue || !_selectedPm4ObjectGroupKey.HasValue)
                    return false;

                var selectedObjectKey = _selectedPm4ObjectKey.Value;
                var selectedGroupKey = _selectedPm4ObjectGroupKey.Value;
                if (!_pm4ObjectLookup.TryGetValue(selectedObjectKey, out Pm4OverlayObject? selectedObject))
                    return false;

                // Cache by (selection, group, split flags). Rebuilding this on every ImGui
                // frame walked the full _pm4TileObjects dictionary plus a 3-level GroupBy LINQ
                // chain — for a multi-instance container that was 30s per click.
                if (_pm4GraphInfoCacheValue.HasValue
                    && _pm4GraphInfoCacheKey.HasValue
                    && _pm4GraphInfoCacheKey.Value.key == selectedObjectKey
                    && _pm4GraphInfoCacheKey.Value.group == selectedGroupKey
                    && _pm4GraphInfoCacheSplitByMscnRef == _pm4SplitCk24ByMscnRef
                    && _pm4GraphInfoCacheSplitByConnectivity == _pm4SplitCk24ByConnectivity)
                {
                    info = _pm4GraphInfoCacheValue.Value;
                    return true;
                }


                var groupObjects = new List<((int tileX, int tileY, uint ck24, int objectPart) key, Pm4OverlayObject obj)>();
                if (_pm4GroupToObjectKeys.TryGetValue(selectedGroupKey, out var objectKeys))
                {
                    foreach (var objectKey in objectKeys)
                    {
                        if (_pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? obj))
                            groupObjects.Add((objectKey, obj));
                    }
                }

                if (groupObjects.Count == 0)
                    return false;

                List<Pm4SelectedObjectGraphLinkNode> linkGroups = groupObjects
                    .GroupBy(static entry => entry.obj.LinkGroupObjectId)
                    .OrderBy(static group => group.Key)
                    .Select(linkGroup =>
                    {
                        var linkEntries = linkGroup
                            .OrderBy(static entry => entry.obj.DominantMscnRefIndex)
                            .ThenBy(static entry => entry.key.objectPart)
                            .ThenBy(static entry => entry.key.tileX)
                            .ThenBy(static entry => entry.key.tileY)
                            .ToList();

                        List<Pm4SelectedObjectGraphMscnRefNode> mscnRefGroups = linkEntries
                            .GroupBy(static entry => entry.obj.DominantMscnRefIndex)
                            .OrderBy(static group => group.Key)
                            .Select(mscnRefGroup =>
                            {
                                var mscnRefEntries = mscnRefGroup
                                    .OrderBy(static entry => entry.key.objectPart)
                                    .ThenBy(static entry => entry.key.tileX)
                                    .ThenBy(static entry => entry.key.tileY)
                                    .ToList();

                                List<Pm4SelectedObjectGraphPartNode> parts = mscnRefEntries
                                    .Select(entry => new Pm4SelectedObjectGraphPartNode(
                                        entry.key.tileX,
                                        entry.key.tileY,
                                        entry.obj.ObjectPartId,
                                        entry.obj.SurfaceCount,
                                        entry.obj.TotalIndexCount,
                                        entry.obj.Lines.Count,
                                        entry.obj.Triangles.Count,
                                        entry.obj.DominantGroupKey,
                                        entry.obj.DominantAttributeMask,
                                        entry.obj.DominantMscnRefIndex,
                                        entry.key == selectedObjectKey))
                                    .ToList();

                                return new Pm4SelectedObjectGraphMscnRefNode(
                                    mscnRefGroup.Key,
                                    parts.Count,
                                    mscnRefEntries.Sum(static entry => entry.obj.SurfaceCount),
                                    mscnRefEntries.Sum(static entry => entry.obj.TotalIndexCount),
                                    mscnRefEntries.Select(static entry => entry.obj.DominantAttributeMask).Distinct().OrderBy(static value => value).ToList(),
                                    mscnRefEntries.Select(static entry => entry.obj.DominantGroupKey).Distinct().OrderBy(static value => value).ToList(),
                                    parts);
                            })
                            .ToList();

                        Pm4OverlayObject linkSeed = linkEntries[0].obj;
                        return new Pm4SelectedObjectGraphLinkNode(
                            linkGroup.Key,
                            linkEntries.Count,
                            linkEntries.Sum(static entry => entry.obj.SurfaceCount),
                            linkEntries.Sum(static entry => entry.obj.TotalIndexCount),
                            linkSeed.LinkedPositionRefCount,
                            linkSeed.LinkedPositionRefSummary,
                            mscnRefGroups.Select(static group => group.MscnRefIndex).ToList(),
                            linkEntries.Select(static entry => entry.obj.DominantAttributeMask).Distinct().OrderBy(static value => value).ToList(),
                            linkEntries.Select(static entry => entry.obj.DominantGroupKey).Distinct().OrderBy(static value => value).ToList(),
                            mscnRefGroups);
                    })
                    .ToList();

                info = new Pm4SelectedObjectGraphInfo(
                    selectedObjectKey.tileX,
                    selectedObjectKey.tileY,
                    selectedObject.Ck24,
                    selectedObject.Ck24Type,
                    selectedObject.Ck24ObjectId,
                    selectedObject.ObjectPartId,
                    _pm4SplitCk24ByMscnRef,
                    _pm4SplitCk24ByConnectivity,
                    groupObjects.Select(static entry => (entry.key.tileX, entry.key.tileY)).Distinct().Count(),
                    linkGroups.Count,
                    linkGroups.Sum(static group => group.MscnRefGroups.Count),
                    groupObjects.Count,
                    groupObjects.Sum(static entry => entry.obj.SurfaceCount),
                    groupObjects.Sum(static entry => entry.obj.TotalIndexCount),
                    groupObjects.Select(static entry => entry.obj.DominantAttributeMask).Distinct().Count(),
                    groupObjects.Select(static entry => entry.obj.DominantGroupKey).Distinct().Count(),
                    linkGroups,
                    BuildTypeBuckets(groupObjects, linkGroups));

                _pm4GraphInfoCacheKey = (selectedObjectKey, selectedGroupKey);
                _pm4GraphInfoCacheValue = info;
                _pm4GraphInfoCacheSplitByMscnRef = _pm4SplitCk24ByMscnRef;
                _pm4GraphInfoCacheSplitByConnectivity = _pm4SplitCk24ByConnectivity;

                return true;
            }

            private static IReadOnlyList<Pm4SelectedObjectGraphTypeBucket> BuildTypeBuckets(
                List<((int tileX, int tileY, uint ck24, int objectPart) key, Pm4OverlayObject obj)> groupObjects,
                List<Pm4SelectedObjectGraphLinkNode> linkGroups)
            {
                var linkGroupByObjectId = linkGroups.ToDictionary(lg => lg.LinkGroupObjectId);

                var objectsByType = groupObjects
                    .GroupBy(entry => entry.obj.Ck24Type)
                    .OrderBy(g => g.Key);

                var typeBuckets = new List<Pm4SelectedObjectGraphTypeBucket>();
                foreach (var typeGroup in objectsByType)
                {
                    byte ck24Type = typeGroup.Key;
                    var typeLinkGroupIds = typeGroup
                        .Select(e => e.obj.LinkGroupObjectId)
                        .Distinct()
                        .OrderBy(id => id);

                    var typeLinkGroups = new List<Pm4SelectedObjectGraphLinkNode>();
                    foreach (uint linkGroupId in typeLinkGroupIds)
                    {
                        if (linkGroupByObjectId.TryGetValue(linkGroupId, out var linkGroupNode))
                            typeLinkGroups.Add(linkGroupNode);
                    }

                    string typeLabel = ck24Type switch
                    {
                        0x03 => "M2 top",
                        0x10 => "interior WMO floor",
                        0x12 => "exterior WMO solid",
                        _ => $"0x{ck24Type:X2}",
                    };

                    typeBuckets.Add(new Pm4SelectedObjectGraphTypeBucket(
                        ck24Type,
                        typeLabel,
                        typeLinkGroups.Count,
                        typeGroup.Sum(e => e.obj.SurfaceCount),
                        typeLinkGroups));
                }

                return typeBuckets;
            }

            public Pm4ColorLegendInfo GetPm4ColorLegend(int maxEntries = 32)
            {
                maxEntries = Math.Max(1, maxEntries);

                if (_pm4ColorMode == Pm4OverlayColorMode.Height)
                {
                    float minZ = float.IsFinite(_pm4MinObjectZ) ? _pm4MinObjectZ : 0f;
                    float maxZ = float.IsFinite(_pm4MaxObjectZ) ? _pm4MaxObjectZ : minZ;
                    float midZ = minZ + ((maxZ - minZ) * 0.5f);
                    var entries = new List<Pm4ColorLegendEntry>
                    {
                        new($"low ({minZ:F1})", ColorFromHeight(minZ), 0, false),
                        new($"mid ({midZ:F1})", ColorFromHeight(midZ), 0, false),
                        new($"high ({maxZ:F1})", ColorFromHeight(maxZ), 0, false)
                    };

                    return new Pm4ColorLegendInfo(
                        _pm4ColorMode,
                        isContinuous: true,
                        "Continuous gradient by PM4 object center height.",
                        entries.Count,
                        entries);
                }

                if (_pm4ColorMode == Pm4OverlayColorMode.TypeFlags)
                {
                    var bitCounts = new Dictionary<uint, int>();
                    foreach (((int tileX, int tileY) _, Pm4OverlayObject obj) in EnumerateVisiblePm4OverlayObjects())
                    {
                        uint mask = obj.DistinctTypeFlags;
                        for (int bit = 1; bit < 32; bit++)
                        {
                            if ((mask & (1u << bit)) != 0)
                            {
                                uint key = (uint)bit;
                                bitCounts.TryGetValue(key, out int existing);
                                bitCounts[key] = existing + 1;
                            }
                        }
                    }
                    List<Pm4ColorLegendEntry> typeFlagEntries = bitCounts
                        .OrderByDescending(static entry => entry.Value)
                        .Take(maxEntries)
                        .Select(entry => new Pm4ColorLegendEntry(
                            FormatPm4LegendLabel(_pm4ColorMode, entry.Key),
                            GetTypeFlagColor((byte)entry.Key),
                            entry.Value,
                            false))
                        .ToList();
                    return new Pm4ColorLegendInfo(
                        _pm4ColorMode,
                        isContinuous: false,
                        "Each swatch is one MSLK.TypeFlags value present in visible objects.",
                        bitCounts.Count,
                        typeFlagEntries);
                }

                if (_pm4ColorMode == Pm4OverlayColorMode.Ck24TypeVsTypeFlags)
                {
                    int matchCount = 0;
                    int noTypeFlagsCount = 0;
                    int untypedCount = 0;
                    int mismatchCount = 0;
                    foreach (((int tileX, int tileY) _, Pm4OverlayObject obj) in EnumerateVisiblePm4OverlayObjects())
                    {
                        uint mask = obj.DistinctTypeFlags;
                        if (mask == 0) { noTypeFlagsCount++; continue; }
                        if (obj.Ck24Type == 0) { untypedCount++; continue; }
                        if ((mask & (1u << obj.Ck24Type)) != 0) { matchCount++; }
                        else { mismatchCount++; }
                    }
                    var entries = new List<Pm4ColorLegendEntry>();
                    if (matchCount > 0)
                        entries.Add(new Pm4ColorLegendEntry("CK24Type matches TypeFlag", new Vector3(0.10f, 0.85f, 0.20f), matchCount, false));
                    if (untypedCount > 0)
                        entries.Add(new Pm4ColorLegendEntry("CK24Type=0 (untyped carrier)", new Vector3(1.00f, 0.95f, 0.10f), untypedCount, false));
                    if (mismatchCount > 0)
                        entries.Add(new Pm4ColorLegendEntry("CK24Type != any TypeFlag", new Vector3(1.00f, 0.15f, 0.15f), mismatchCount, false));
                    if (noTypeFlagsCount > 0)
                        entries.Add(new Pm4ColorLegendEntry("No TypeFlags data", new Vector3(0.25f, 0.25f, 0.25f), noTypeFlagsCount, false));
                    return new Pm4ColorLegendInfo(
                        _pm4ColorMode,
                        isContinuous: false,
                        "Green = CK24 high byte matches a TypeFlag. Red = no match (anomaly). Yellow = CK24Type=0 carrier object.",
                        entries.Count,
                        entries);
                }

                if (_pm4ColorMode == Pm4OverlayColorMode.Tile)
                {
                    var counts = new Dictionary<(int tileX, int tileY), int>();
                    foreach (((int tileX, int tileY) tileKey, _) in EnumerateVisiblePm4OverlayObjects())
                    {
                        counts.TryGetValue(tileKey, out int existing);
                        counts[tileKey] = existing + 1;
                    }

                    bool hasSelection = _selectedPm4ObjectKey.HasValue;
                    (int tileX, int tileY) selectedTile = hasSelection
                        ? (_selectedPm4ObjectKey!.Value.tileX, _selectedPm4ObjectKey.Value.tileY)
                        : default;
                    List<Pm4ColorLegendEntry> entries = counts
                        .OrderBy(static entry => entry.Key.tileX)
                        .ThenBy(static entry => entry.Key.tileY)
                        .Take(maxEntries)
                        .Select(entry => new Pm4ColorLegendEntry(
                            $"tile ({entry.Key.tileX}, {entry.Key.tileY})",
                            ColorFromSeed((uint)HashCode.Combine(entry.Key.tileX, entry.Key.tileY)),
                            entry.Value,
                            hasSelection && entry.Key == selectedTile))
                        .ToList();

                    return new Pm4ColorLegendInfo(
                        _pm4ColorMode,
                        isContinuous: false,
                        "Each swatch identifies one loaded PM4 tile bucket.",
                        counts.Count,
                        entries);
                }

                var categoricalCounts = new Dictionary<uint, int>();
                foreach (((int tileX, int tileY) _, Pm4OverlayObject obj) in EnumerateVisiblePm4OverlayObjects())
                {
                    uint key = GetPm4LegendValue(_pm4ColorMode, obj);
                    categoricalCounts.TryGetValue(key, out int existing);
                    categoricalCounts[key] = existing + 1;
                }

                uint? selectedValue = TryGetSelectedPm4LegendValue();
                List<Pm4ColorLegendEntry> categoricalEntries = categoricalCounts
                    .OrderBy(static entry => entry.Key)
                    .Take(maxEntries)
                    .Select(entry => new Pm4ColorLegendEntry(
                        FormatPm4LegendLabel(_pm4ColorMode, entry.Key),
                        GetPm4LegendColor(_pm4ColorMode, entry.Key),
                        entry.Value,
                        selectedValue.HasValue && selectedValue.Value == entry.Key))
                    .ToList();

                return new Pm4ColorLegendInfo(
                    _pm4ColorMode,
                    isContinuous: false,
                    "Categorical colors are viewer-identification buckets, not closed PM4 semantics.",
                    categoricalCounts.Count,
                    categoricalEntries);
            }

            /// <summary>
            /// Returns the raw tile→objects dictionary for the full scene outliner.
            /// Uses the existing tuple-based key pattern from GetPm4ObjectHierarchy.
            /// </summary>
            public IReadOnlyDictionary<(int tileX, int tileY), IReadOnlyList<(uint ck24, int objectPart, uint ck24ObjectId, uint mshdRegionId, uint linkGroupObjectId, byte groupKey, byte attributeMask, uint mscnRefIndex, int surfaceCount, int totalIndexCount, float avgHeight, Vector3 boundsMin, Vector3 boundsMax, int linkedPositionRefCount)>> GetPm4TileObjectSummaries()
            {
                var result = new Dictionary<(int tileX, int tileY), IReadOnlyList<(uint, int, uint, uint, uint, byte, byte, uint, int, int, float, Vector3, Vector3, int)>>();
                foreach (var kv in _pm4TileObjects)
                {
                    var list = new List<(uint, int, uint, uint, uint, byte, byte, uint, int, int, float, Vector3, Vector3, int)>();
                    foreach (var obj in kv.Value)
                    {
                        list.Add((obj.Ck24, obj.ObjectPartId, obj.Ck24ObjectId, obj.MshdRegionId,
                            obj.LinkGroupObjectId, obj.DominantGroupKey, obj.DominantAttributeMask,
                            obj.DominantMscnRefIndex, obj.SurfaceCount, obj.TotalIndexCount,
                            obj.AverageSurfaceHeight, obj.BoundsMin, obj.BoundsMax,
                            obj.LinkedPositionRefCount));
                    }
                    result[(kv.Key.tileX, kv.Key.tileY)] = list;
                }
                return result;
            }

            /// <summary>
            /// Select a PM4 object by its tile/CK24/part key. Returns true if found.
            /// </summary>
            public bool SelectPm4ObjectByKey(int tileX, int tileY, uint ck24, int objectPart)
            {
                var key = (tileX, tileY, ck24, objectPart);
                if (!_pm4ObjectLookup.ContainsKey(key))
                    return false;

                _selectedPm4ObjectKey = key;
                _selectedPm4ObjectGroupKey = ResolvePm4ObjectGroupKey(key);
                return true;
            }

            public IReadOnlyList<(int tileX, int tileY, uint ck24, uint ck24ObjectId, uint mshdRegionId, uint linkGroupObjectId, byte groupKey, int objectPartId)> GetPm4ObjectHierarchy()
            {
                var list = new List<(int, int, uint, uint, uint, uint, byte, int)>();
                foreach (var tileEntry in _pm4TileObjects)
                    foreach (var obj in tileEntry.Value)
                        list.Add((tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.Ck24ObjectId, obj.MshdRegionId, obj.LinkGroupObjectId, obj.DominantGroupKey, obj.ObjectPartId));
                return list;
            }

            public IReadOnlyList<(int tileX, int tileY, uint ck24, int objectPart)> GetVisiblePm4ObjectsForRegion(uint regionId)
            {
                var keys = new List<(int tileX, int tileY, uint ck24, int objectPart)>();
                foreach (((int tileX, int tileY) tileKey, Pm4OverlayObject obj) in EnumerateVisiblePm4OverlayObjects())
                {
                    if (obj.MshdRegionId != regionId)
                        continue;

                    keys.Add((tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId));
                }

                return keys;
            }

            public Pm4VisibleOverlaySummaryInfo GetPm4VisibleOverlaySummary(int maxRegions = 10, int maxTypeBucketsPerRegion = 3)
            {
                maxRegions = Math.Max(1, maxRegions);
                maxTypeBucketsPerRegion = Math.Max(1, maxTypeBucketsPerRegion);

                var objectsByRegion = new Dictionary<uint, List<((int tileX, int tileY, uint ck24, int objectPart) key, Pm4ObjectDebugInfo debug)>>();
                int visibleObjectCount = 0;
                var visibleTiles = new HashSet<(int tileX, int tileY)>();

                foreach (((int tileX, int tileY, uint ck24, int objectPart) key, _, Pm4ObjectDebugInfo debug) in EnumerateVisiblePm4OverlayDebugObjects())
                {
                    visibleObjectCount++;
                    visibleTiles.Add((key.tileX, key.tileY));
                    if (!objectsByRegion.TryGetValue(debug.MshdRegionId, out var entries))
                    {
                        entries = new List<((int tileX, int tileY, uint ck24, int objectPart), Pm4ObjectDebugInfo)>();
                        objectsByRegion[debug.MshdRegionId] = entries;
                    }

                    entries.Add((key, debug));
                }

                uint? selectedRegionId = null;
                if (_selectedPm4ObjectKey.HasValue
                    && _pm4ObjectLookup.TryGetValue(_selectedPm4ObjectKey.Value, out Pm4OverlayObject? selectedObject))
                {
                    selectedRegionId = selectedObject.MshdRegionId;
                }

                List<Pm4VisibleRegionSummary> regions = objectsByRegion
                    .Select(entry =>
                    {
                        Dictionary<byte, int> typeCounts = entry.Value
                            .GroupBy(static regionEntry => regionEntry.debug.Ck24Type)
                            .ToDictionary(static group => group.Key, static group => group.Count());
                        int objectCount = entry.Value.Count;
                        float averageHeight = objectCount > 0
                            ? entry.Value.Average(static regionEntry => regionEntry.debug.Center.Z)
                            : 0f;
                        return new Pm4VisibleRegionSummary(
                            entry.Key,
                            objectCount,
                            entry.Value.Select(static regionEntry => (regionEntry.key.tileX, regionEntry.key.tileY)).Distinct().Count(),
                            entry.Value.Select(static regionEntry => regionEntry.debug.Ck24).Distinct().Count(),
                            entry.Value.Select(static regionEntry => regionEntry.debug.LinkGroupObjectId).Distinct().Count(),
                            averageHeight,
                            selectedRegionId.HasValue && selectedRegionId.Value == entry.Key,
                            BuildPm4VisibleTypeBuckets(typeCounts, maxTypeBucketsPerRegion));
                    })
                    .OrderByDescending(static entry => entry.ObjectCount)
                    .ThenBy(static entry => entry.RegionId)
                    .Take(maxRegions)
                    .ToList();

                return new Pm4VisibleOverlaySummaryInfo(
                    visibleObjectCount,
                    visibleTiles.Count,
                    objectsByRegion.Count,
                    selectedRegionId,
                    regions);
            }

            public bool TryGetSelectedPm4RegionInfo(out Pm4SelectedObjectRegionInfo info, int maxPeers = 18, int maxTypeBuckets = 4)
            {
                info = default;
                maxPeers = Math.Max(1, maxPeers);
                maxTypeBuckets = Math.Max(1, maxTypeBuckets);

                if (!_selectedPm4ObjectKey.HasValue
                    || !_pm4ObjectLookup.TryGetValue(_selectedPm4ObjectKey.Value, out Pm4OverlayObject? selectedObject)
                    || !TryGetPm4ObjectDebugInfo(_selectedPm4ObjectKey.Value, out Pm4ObjectDebugInfo selectedDebug))
                {
                    return false;
                }

                uint regionId = selectedObject.MshdRegionId;
                var peers = new List<Pm4RegionPeerSummary>();
                var typeCounts = new Dictionary<byte, int>();
                var uniqueTiles = new HashSet<(int tileX, int tileY)>();
                var uniqueCk24 = new HashSet<uint>();
                var uniqueLinkGroups = new HashSet<uint>();
                var uniqueMscnRefs = new HashSet<uint>();
                int sameCk24Count = 0;
                int sameLinkGroupCount = 0;
                int sameMscnRefCount = 0;
                int totalSurfaceCount = 0;
                float totalHeight = 0f;

                foreach (((int tileX, int tileY, uint ck24, int objectPart) key, _, Pm4ObjectDebugInfo debug) in EnumerateVisiblePm4OverlayDebugObjects())
                {
                    if (debug.MshdRegionId != regionId)
                        continue;

                    uniqueTiles.Add((key.tileX, key.tileY));
                    uniqueCk24.Add(debug.Ck24);
                    uniqueLinkGroups.Add(debug.LinkGroupObjectId);
                    uniqueMscnRefs.Add(debug.DominantMscnRefIndex);
                    totalSurfaceCount += debug.SurfaceCount;
                    totalHeight += debug.Center.Z;
                    if (typeCounts.TryGetValue(debug.Ck24Type, out int existingTypeCount))
                        typeCounts[debug.Ck24Type] = existingTypeCount + 1;
                    else
                        typeCounts[debug.Ck24Type] = 1;

                    bool sameCk24 = debug.Ck24 == selectedDebug.Ck24;
                    bool sameLinkGroup = debug.LinkGroupObjectId == selectedDebug.LinkGroupObjectId;
                    bool sameMscnRefIndex = debug.DominantMscnRefIndex == selectedDebug.DominantMscnRefIndex;
                    if (sameCk24)
                        sameCk24Count++;
                    if (sameLinkGroup)
                        sameLinkGroupCount++;
                    if (sameMscnRefIndex)
                        sameMscnRefCount++;

                    peers.Add(new Pm4RegionPeerSummary(
                        key,
                        debug.Ck24Type,
                        debug.Ck24ObjectId,
                        debug.SurfaceCount,
                        debug.LinkGroupObjectId,
                        debug.DominantMscnRefIndex,
                        debug.Center,
                        key == _selectedPm4ObjectKey.Value,
                        sameCk24,
                        sameLinkGroup,
                        sameMscnRefIndex));
                }

                if (peers.Count == 0)
                    return false;

                var selectedKey = _selectedPm4ObjectKey.Value;
                IReadOnlyList<Pm4RegionPeerSummary> peerList = peers
                    .OrderByDescending(static peer => peer.IsSelected)
                    .ThenByDescending(static peer => peer.SameCk24)
                    .ThenByDescending(static peer => peer.SameLinkGroup)
                    .ThenByDescending(static peer => peer.SameMscnRefIndex)
                    .ThenBy(peer => Math.Abs(peer.ObjectKey.tileX - selectedKey.tileX) + Math.Abs(peer.ObjectKey.tileY - selectedKey.tileY))
                    .ThenBy(static peer => peer.ObjectKey.objectPart)
                    .Take(maxPeers)
                    .ToList();

                info = new Pm4SelectedObjectRegionInfo(
                    regionId,
                    peers.Count,
                    uniqueTiles.Count,
                    uniqueCk24.Count,
                    uniqueLinkGroups.Count,
                    uniqueMscnRefs.Count,
                    sameCk24Count,
                    sameLinkGroupCount,
                    sameMscnRefCount,
                    (float)totalSurfaceCount / peers.Count,
                    totalHeight / peers.Count,
                    BuildPm4VisibleTypeBuckets(typeCounts, maxTypeBuckets),
                    peerList);
                return true;
            }

            private IEnumerable<((int tileX, int tileY) tileKey, Pm4OverlayObject obj)> EnumerateVisiblePm4OverlayObjects()
            {
                foreach (KeyValuePair<(int tileX, int tileY), List<Pm4OverlayObject>> tileEntry in _pm4TileObjects)
                {
                    List<Pm4OverlayObject> objects = tileEntry.Value;
                    for (int i = 0; i < objects.Count; i++)
                    {
                        Pm4OverlayObject obj = objects[i];
                        if (ShouldRenderPm4ObjectType(obj.Ck24Type))
                            yield return (tileEntry.Key, obj);
                    }
                }
            }

            private IEnumerable<((int tileX, int tileY, uint ck24, int objectPart) key, Pm4OverlayObject obj, Pm4ObjectDebugInfo debug)> EnumerateVisiblePm4OverlayDebugObjects()
            {
                foreach (((int tileX, int tileY) tileKey, Pm4OverlayObject obj) in EnumerateVisiblePm4OverlayObjects())
                {
                    var key = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                    if (TryGetPm4ObjectDebugInfo(key, out Pm4ObjectDebugInfo debug))
                        yield return (key, obj, debug);
                }
            }

            private static IReadOnlyList<Pm4VisibleTypeBucket> BuildPm4VisibleTypeBuckets(
                IReadOnlyDictionary<byte, int> typeCounts,
                int maxTypeBuckets)
            {
                return typeCounts
                    .OrderByDescending(static entry => entry.Value)
                    .ThenBy(static entry => entry.Key)
                    .Take(Math.Max(1, maxTypeBuckets))
                    .Select(static entry => new Pm4VisibleTypeBucket(entry.Key, entry.Value))
                    .ToList();
            }

    private static uint GetPm4LegendValue(Pm4OverlayColorMode mode, Pm4OverlayObject obj)
    {
        return mode switch
        {
            Pm4OverlayColorMode.Ck24ObjectId => obj.Ck24ObjectId,
            Pm4OverlayColorMode.Ck24Key      => obj.Ck24,
            Pm4OverlayColorMode.MshdRegionId => obj.MshdRegionId,
            Pm4OverlayColorMode.GroupKey     => obj.DominantGroupKey,
            Pm4OverlayColorMode.AttributeMask=> obj.DominantAttributeMask,
            Pm4OverlayColorMode.TypeFlags    => obj.DistinctTypeFlags,
            _ => obj.Ck24Type   // fallback = Ck24Type
        };
    }

            private uint? TryGetSelectedPm4LegendValue()
            {
                if (!_selectedPm4ObjectKey.HasValue || !_pm4ObjectLookup.TryGetValue(_selectedPm4ObjectKey.Value, out Pm4OverlayObject? selectedObject))
                    return null;

                return _pm4ColorMode switch
                {
                    Pm4OverlayColorMode.Tile => null,
                    Pm4OverlayColorMode.Height => null,
                    _ => GetPm4LegendValue(_pm4ColorMode, selectedObject)
                };
            }

    private string FormatPm4LegendLabel(Pm4OverlayColorMode mode, uint value)
    {
        return mode switch
        {
            Pm4OverlayColorMode.Ck24Type       => $"CK24 type 0x{value:X2}",
            Pm4OverlayColorMode.Ck24ObjectId   => $"CK24 obj {value} (0x{value:X4})",
            Pm4OverlayColorMode.Ck24Key        => $"CK24 0x{value:X6}",
            Pm4OverlayColorMode.MshdRegionId   => $"MSHD region {value}",
            Pm4OverlayColorMode.GroupKey       => $"GroupKey 0x{value:X2}",
            Pm4OverlayColorMode.AttributeMask  => FormatAttributeMaskLabel((byte)value),
            Pm4OverlayColorMode.TypeFlags      => ((byte)value) switch
            {
                0x03 => "TypeFlags 0x03 — M2 top surfaces",
                0x10 => "TypeFlags 0x10 — interior WMO floors",
                0x12 => "TypeFlags 0x12 — exterior WMO solids",
                _ => $"TypeFlags 0x{value:X2} — unknown",
            },
            Pm4OverlayColorMode.Ck24TypeVsTypeFlags => value switch
            {
                0 => "CK24Type matches TypeFlag",
                1 => "No TypeFlags data",
                2 => "CK24Type=0 carrier",
                3 => "CK24Type != TypeFlag (anomaly)",
                _ => $"unknown ({value})",
            },
            _ => value.ToString(CultureInfo.InvariantCulture)
        };
    }

    private Vector3 GetPm4LegendColor(Pm4OverlayColorMode mode, uint value)
    {
        return mode switch
        {
            Pm4OverlayColorMode.Ck24ObjectId  => ColorFromSeed(value),
            Pm4OverlayColorMode.Ck24Key       => ColorFromSeed(value),
            Pm4OverlayColorMode.MshdRegionId  => ColorFromSeed(value),
            Pm4OverlayColorMode.GroupKey      => ColorFromSeed(value),
            Pm4OverlayColorMode.AttributeMask => ColorFromSeed(value),
            Pm4OverlayColorMode.TypeFlags     => GetTypeFlagColor((byte)value),
            Pm4OverlayColorMode.Ck24TypeVsTypeFlags => value switch
            {
                0 => new Vector3(0.10f, 0.85f, 0.20f),
                1 => new Vector3(0.25f, 0.25f, 0.25f),
                2 => new Vector3(1.00f, 0.95f, 0.10f),
                _ => new Vector3(1.00f, 0.15f, 0.15f),
            },
            _ => GetPm4TypeColor((byte)value)  // fallback (Ck24Type uses GetPm4TypeColor)
        };
    }
    private Vector3 ColorFromHeight(float z)
    {
        float denom = _pm4MaxObjectZ - _pm4MinObjectZ;
        float t = denom > 0.001f ? Math.Clamp((z - _pm4MinObjectZ) / denom, 0f, 1f) : 0.5f;
        return Vector3.Lerp(new Vector3(0.45f, 0.55f, 0.80f), new Vector3(0.80f, 0.50f, 0.45f), t);
    }

    // ──────────────────────────────────────────────────────────────────────
    // PM4 color system: light pastels for containers, dark pastels for mesh,
    // saturated colors reserved for markers/highlights/selection.
    // See wow-viewer/docs/architecture/pm4-color-palette.md (to be written).
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>Light pastel — PM4 object bounds (per-object, sub-merged).</summary>
    private static readonly Vector3 Pm4ColorObjectBounds = new(1.00f, 0.75f, 0.80f);

    /// <summary>Light pastel — PM4 CK24 bounds (merged across sub-objects).</summary>
    private static readonly Vector3 Pm4ColorCk24Bounds = new(0.70f, 1.00f, 0.80f);

    /// <summary>Light pastel — selected bounds inner fill.</summary>
    private static readonly Vector3 Pm4ColorSelectedBounds = new(1.00f, 1.00f, 0.95f);

    /// <summary>Light pastel — MDDF (M2) instance bounds.</summary>
    private static readonly Vector3 Pm4ColorMddfBounds = new(1.00f, 0.70f, 1.00f);

    /// <summary>Light pastel — MODF (WMO) instance bounds.</summary>
    private static readonly Vector3 Pm4ColorModfBounds = new(0.75f, 0.95f, 1.00f);

    /// <summary>Dark pastel — PM4 centroid pin (per-object center marker).</summary>
    private static readonly Vector3 Pm4ColorCentroid = new(0.70f, 0.55f, 0.50f);

    /// <summary>Saturated — MSCN scene-graph connector anchor. One cube per MSUR surface (placed via MSUR.MscnRefIndex). Bright cyan, distinct from everything else. See wow-viewer/docs/architecture/pm4-chunk-semantics.md.</summary>
    private static readonly Vector3 Pm4ColorMscn = new(0.10f, 0.95f, 1.00f);

    /// <summary>Saturated — MSPV path-vertex position. One cube per MSPI index reached from an MSLK link's path-vertex chain. Only present when surfaces are connected via MSLK. Bright magenta, distinct from MSCN and from pastel mesh. See wow-viewer/docs/architecture/pm4-chunk-semantics.md.</summary>
    private static readonly Vector3 Pm4ColorMspv = new(1.00f, 0.20f, 0.80f);

    /// <summary>Medium pastel — MPRL position reference pin.</summary>
    private static readonly Vector3 Pm4ColorMprl = new(0.40f, 0.80f, 0.85f);

    // Saturated signals (NOT in the pastel family — reserved for interactive signals)

    /// <summary>Saturated — search/highlight match (eyecatching on pastel mesh).</summary>
    private static readonly Vector3 Pm4ColorHighlight = new(0.20f, 1.00f, 0.95f);

    /// <summary>Saturated — group selection (THE selection signal — must be unmistakable).</summary>
    private static readonly Vector3 Pm4ColorSelection = new(1.00f, 0.95f, 0.20f);

    private static Vector3 ColorFromSeed(uint seed)
    {
        uint golden = seed * 2654435761u;
        float hue = (golden & 0x00FFFFFF) / 16777215f;
        return HsvToRgb(hue, 0.75f, 0.95f);
    }

    private static Vector3 HsvToRgb(float h, float s, float v)
    {
        h = h - MathF.Floor(h);
        float c = v * s;
        float x = c * (1f - MathF.Abs((h * 6f) % 2f - 1f));
        float m = v - c;

        float r;
        float g;
        float b;
        int sector = (int)(h * 6f);
        switch (sector)
        {
            case 0:
                r = c; g = x; b = 0f;
                break;
            case 1:
                r = x; g = c; b = 0f;
                break;
            case 2:
                r = 0f; g = c; b = x;
                break;
            case 3:
                r = 0f; g = x; b = c;
                break;
            case 4:
                r = x; g = 0f; b = c;
                break;
            default:
                r = c; g = 0f; b = x;
                break;
        }

        return new Vector3(r + m, g + m, b + m);
    }

    private static Vector3 GetPm4TypeColor(byte ck24Type)
    {
        // Dark pastels for mesh — Ck24Type is a coarse container classifier
        return ck24Type switch
        {
            0x40 => new Vector3(0.85f, 0.55f, 0.30f),   // dark pastel orange
            0x80 => new Vector3(0.80f, 0.40f, 0.25f),   // dark pastel burnt orange
            _    => new Vector3(0.80f, 0.50f, 0.30f)    // dark pastel amber
        };
    }

    private static Vector3 GetTypeFlagColor(byte typeFlag)
    {
        // Dark pastels for known TypeFlags (mesh). Unknown flags use a desaturated seed.
        return typeFlag switch
        {
            0x03 => new Vector3(0.30f, 0.65f, 0.45f),   // dark pastel green   (M2 top)
            0x10 => new Vector3(0.30f, 0.55f, 0.65f),   // dark pastel teal    (interior floor)
            0x12 => new Vector3(0.80f, 0.45f, 0.45f),   // dark pastel rose    (exterior solid)
            _    => HsvToRgb((typeFlag * 0.19f) % 1.0f, 0.45f, 0.75f),  // desaturated for unknown
        };
    }

    private static Vector3 BlendTypeFlagColors(uint typeFlagsMask)
    {
        if (typeFlagsMask == 0)
            return new Vector3(0.25f, 0.25f, 0.25f); // gray — no TypeFlags bits set anywhere on this object

        Vector3 sum = Vector3.Zero;
        int count = 0;
        for (int bit = 1; bit < 32; bit++)
        {
            if ((typeFlagsMask & (1u << bit)) == 0) continue;
            sum += GetTypeFlagColor((byte)(1u << bit));
            count++;
        }
        return sum / count; // equal-weight additive blend — bit count stays visible
    }

    private static Vector3 GetCk24TypeVsTypeFlagsColor(byte ck24Type, uint typeFlagsMask)
    {
        // These are the *signals* for the Ck24Type vs TypeFlags diagnostic, not mesh colors.
        // Use saturated tones so the diagnostic stands out against the pastel mesh.
        if (typeFlagsMask == 0)
            return new Vector3(0.55f, 0.55f, 0.55f);  // pastel gray — no TypeFlags data

        // Ck24Type of 0 with non-zero TypeFlags = untyped container carrying classified surfaces
        if (ck24Type == 0)
            return Pm4ColorSelection;  // saturated yellow

        // Check if Ck24Type matches any set TypeFlag
        if ((typeFlagsMask & (1u << ck24Type)) != 0)
            return new Vector3(0.20f, 0.75f, 0.30f);  // green — match

        // Ck24Type != 0, TypeFlags present, but no match = anomaly
        return new Vector3(0.85f, 0.20f, 0.20f);       // red — mismatch
    }

    private static string FormatAttributeMaskLabel(byte value)
    {
        if (value == 0) return "AttrMask 0x00 (none)";
        List<string> bits = [];
        if ((value & 0x01) != 0) bits.Add("bit0");
        if ((value & 0x02) != 0) bits.Add("bit1");
        if ((value & 0x04) != 0) bits.Add("bit2");
        if ((value & 0x08) != 0) bits.Add("bit3");
        if ((value & 0x10) != 0) bits.Add("bit4");
        if ((value & 0x20) != 0) bits.Add("bit5");
        if ((value & 0x40) != 0) bits.Add("bit6");
        if ((value & 0x80) != 0) bits.Add("bit7");
        return $"AttrMask 0x{value:X2} ({string.Join("|", bits)})";
    }

    private static byte PickPrimaryTypeFlag(uint mask)
    {
        // Prefer known flags in priority order; fall back to lowest set bit
        if ((mask & (1u << 0x12)) != 0) return 0x12;  // exterior solid
        if ((mask & (1u << 0x10)) != 0) return 0x10;  // interior floor
        if ((mask & (1u << 0x03)) != 0) return 0x03;  // M2 top
        // Pick the lowest set bit for unknown flags
        for (int bit = 1; bit < 32; bit++)
        {
            if ((mask & (1u << bit)) != 0)
                return (byte)bit;
        }
        return 0;
    }

    private bool ShouldRenderPm4ObjectType(byte ck24Type)
    {
        return ck24Type switch
        {
            0x40 => _showPm4Type40,
            0x80 => _showPm4Type80,
            _ => _showPm4TypeOther
        };
    }

    private Matrix4x4 BuildPm4OverlayTransformMatrix()
    {
        float rotX = _pm4OverlayRotationDegrees.X * MathF.PI / 180f;
        float rotY = _pm4OverlayRotationDegrees.Y * MathF.PI / 180f;
        float rotZ = _pm4OverlayRotationDegrees.Z * MathF.PI / 180f;
        return Matrix4x4.CreateScale(_pm4OverlayScale)
            * Matrix4x4.CreateRotationX(rotX)
            * Matrix4x4.CreateRotationY(rotY)
            * Matrix4x4.CreateRotationZ(rotZ)
            * Matrix4x4.CreateTranslation(_pm4OverlayTranslation);
    }

    private static Vector3 ApplyPm4OverlayTransform(Vector3 position, in Matrix4x4 transform)
    {
        return Vector3.Transform(position, transform);
    }

    private static Matrix4x4 BuildPm4GeometryTransform(Pm4OverlayObject obj, in Matrix4x4 objectTransform, bool applyObjectTransform)
    {
        return applyObjectTransform
            ? obj.BaseTransform * objectTransform
            : obj.BaseTransform;
    }

    internal static Matrix4x4 BuildPm4BaseTransform(Vector3 placementAnchor, float baseRotationRadians)
    {
        Matrix4x4 transform = Matrix4x4.Identity;
        if (MathF.Abs(baseRotationRadians) > 1e-6f)
            transform *= Matrix4x4.CreateRotationZ(baseRotationRadians);

        transform *= Matrix4x4.CreateTranslation(placementAnchor);
        return transform;
    }

    private List<CorePm4CorrelationObjectState> BuildPm4CorrelationObjectStates()
    {
        bool applyPm4Transform = !IsNearZeroVector(_pm4OverlayTranslation)
            || !IsNearZeroVector(_pm4OverlayRotationDegrees)
            || !IsNearOneVector(_pm4OverlayScale);
        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        var inputs = new List<CorePm4CorrelationGeometryInput>(_pm4ObjectLookup.Count);

        foreach (var tileEntry in _pm4TileObjects)
        {
            foreach (Pm4OverlayObject obj in tileEntry.Value)
            {
                var objectKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, obj.Ck24, obj.ObjectPartId);
                var groupKey = ResolvePm4ObjectGroupKey(objectKey);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(objectKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);
                Matrix4x4 geometryTransform = BuildPm4GeometryTransform(obj, objectTransform, applyObjectTransform);
                inputs.Add(new CorePm4CorrelationGeometryInput(
                    tileEntry.Key.tileX,
                    tileEntry.Key.tileY,
                    new CorePm4ObjectGroupKey(groupKey.tileX, groupKey.tileY, groupKey.ck24),
                    new CorePm4CorrelationObjectDescriptor(
                        obj.Ck24,
                        obj.Ck24Type,
                        obj.ObjectPartId,
                        obj.LinkGroupObjectId,
                        obj.SurfaceCount,
                        obj.LinkedPositionRefCount,
                        obj.DominantGroupKey,
                        obj.DominantAttributeMask,
                        obj.DominantMscnRefIndex,
                        obj.AverageSurfaceHeight),
                    obj.Lines.Select(static line => new CorePm4GeometryLineSegment(line.From, line.To)).ToList(),
                    obj.Triangles.Select(static triangle => new CorePm4GeometryTriangle(triangle.A, triangle.B, triangle.C)).ToList(),
                    geometryTransform));
            }
        }

        return CorePm4CorrelationMath.BuildObjectStatesFromGeometry(inputs).ToList();
    }

    private bool ShouldRenderPm4Object(
        Pm4OverlayObject obj,
        in Matrix4x4 objectTransform,
        bool applyObjectTransform,
        in Vector3 cameraPos,
        out Vector3 transformedCenter)
    {
        Vector3 boundsMin = obj.BoundsMin;
        Vector3 boundsMax = obj.BoundsMax;
        transformedCenter = obj.Center;

        if (applyObjectTransform)
        {
            TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);
            transformedCenter = ApplyPm4OverlayTransform(obj.Center, objectTransform);
        }

        float distSq = Vector3.DistanceSquared(cameraPos, transformedCenter);
        if (distSq > NoCullRadiusSq && !_frustumCuller.TestAABB(boundsMin, boundsMax))
            return false;

        return true;
    }

    private static bool IsNearZeroVector(Vector3 value)
    {
        return value.LengthSquared() < 0.0001f;
    }

    private static bool IsNearOneVector(Vector3 value)
    {
        return MathF.Abs(value.X - 1f) < 0.0001f
            && MathF.Abs(value.Y - 1f) < 0.0001f
            && MathF.Abs(value.Z - 1f) < 0.0001f;
    }

    private static Vector3 SanitizeScale(Vector3 scale)
    {
        const float minAbsScale = 0.0001f;

        float SanitizeComponent(float component)
        {
            if (MathF.Abs(component) >= minAbsScale)
                return component;

            return component < 0f ? -minAbsScale : minAbsScale;
        }

        return new Vector3(
            SanitizeComponent(scale.X),
            SanitizeComponent(scale.Y),
            SanitizeComponent(scale.Z));
    }

    private void RebuildPm4ObjectGroupBounds()
    {
        _pm4ObjectGroupBounds.Clear();

        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            var groupKey = ResolvePm4ObjectGroupKey(objectKey);
            if (_pm4ObjectGroupBounds.TryGetValue(groupKey, out var existingBounds))
            {
                _pm4ObjectGroupBounds[groupKey] = (
                    Vector3.Min(existingBounds.min, obj.BoundsMin),
                    Vector3.Max(existingBounds.max, obj.BoundsMax));
            }
            else
            {
                _pm4ObjectGroupBounds[groupKey] = (obj.BoundsMin, obj.BoundsMax);
            }
        }
    }

    public bool TryGetPm4ObjectGroupBounds((int tileX, int tileY, uint ck24) groupKey, out Vector3 min, out Vector3 max)
    {
        if (_pm4ObjectGroupBounds.TryGetValue(groupKey, out var b))
        {
            min = b.min;
            max = b.max;
            return true;
        }
        min = default;
        max = default;
        return false;
    }

    public IReadOnlyList<Pm4SurfaceGroupCluster> GetPm4SurfaceGroupClusters(int tileX, int tileY, uint ck24)
    {
        var clusterByGroupKey = new Dictionary<byte, (Vector3 min, Vector3 max, int count)>();

        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            if (objectKey.tileX != tileX || objectKey.tileY != tileY || objectKey.ck24 != ck24)
                continue;

            byte gk = obj.DominantGroupKey;
            if (clusterByGroupKey.TryGetValue(gk, out var existing))
            {
                clusterByGroupKey[gk] = (
                    Vector3.Min(existing.min, obj.BoundsMin),
                    Vector3.Max(existing.max, obj.BoundsMax),
                    existing.count + obj.SurfaceCount);
            }
            else
            {
                clusterByGroupKey[gk] = (obj.BoundsMin, obj.BoundsMax, obj.SurfaceCount);
            }
        }

        var results = new List<Pm4SurfaceGroupCluster>(clusterByGroupKey.Count);
        foreach (var kv in clusterByGroupKey.OrderBy(static kv => kv.Key))
        {
            results.Add(new Pm4SurfaceGroupCluster(kv.Key, kv.Value.min, kv.Value.max, kv.Value.count));
        }
        return results;
    }

    private void RebuildPm4TileCk24Bounds()
    {
        _pm4TileCk24Bounds.Clear();

        foreach (var (objectKey, obj) in _pm4ObjectLookup)
        {
            var tileCk24Key = (objectKey.tileX, objectKey.tileY, objectKey.ck24);
            if (_pm4TileCk24Bounds.TryGetValue(tileCk24Key, out var existingBounds))
            {
                _pm4TileCk24Bounds[tileCk24Key] = (
                    Vector3.Min(existingBounds.min, obj.BoundsMin),
                    Vector3.Max(existingBounds.max, obj.BoundsMax));
            }
            else
            {
                _pm4TileCk24Bounds[tileCk24Key] = (obj.BoundsMin, obj.BoundsMax);
            }
        }
    }

    private bool TryComputePm4ObjectGroupPivot(
        (int tileX, int tileY, uint ck24) groupKey,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out Vector3 pivot)
    {
        if (_pm4ObjectGroupBounds.TryGetValue(groupKey, out var groupBounds))
        {
            pivot = (groupBounds.min + groupBounds.max) * 0.5f;
            if (applyPm4Transform)
                pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            return true;
        }

        pivot = Vector3.Zero;
        return false;
    }

    private bool TryComputePm4TileCk24Pivot(
        (int tileX, int tileY, uint ck24) tileCk24Key,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out Vector3 pivot)
    {
        if (_pm4TileCk24Bounds.TryGetValue(tileCk24Key, out var rawBounds))
        {
            pivot = (rawBounds.min + rawBounds.max) * 0.5f;
            if (applyPm4Transform)
                pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            return true;
        }

        pivot = Vector3.Zero;
        return false;
    }

    private Matrix4x4 BuildPm4ObjectTransform((int tileX, int tileY, uint ck24, int objectPart) objectKey,
        bool applyPm4Transform,
        in Matrix4x4 pm4Transform,
        out bool applyObjectTransform)
    {
        applyObjectTransform = false;
        Matrix4x4 transform = Matrix4x4.Identity;

        if (applyPm4Transform)
        {
            transform = pm4Transform;
            applyObjectTransform = true;
        }

        var objectGroupKey = ResolvePm4ObjectGroupKey(objectKey);
        var tileCk24Key = (objectKey.tileX, objectKey.tileY, objectKey.ck24);
        bool hasLayerTranslation = _pm4TileCk24Translations.TryGetValue(tileCk24Key, out Vector3 layerTranslation)
            && !IsNearZeroVector(layerTranslation);
        bool hasLayerRotation = _pm4TileCk24RotationsDegrees.TryGetValue(tileCk24Key, out Vector3 layerRotationDegrees)
            && !IsNearZeroVector(layerRotationDegrees);
        bool hasLayerScale = _pm4TileCk24Scales.TryGetValue(tileCk24Key, out Vector3 layerScale)
            && !IsNearOneVector(layerScale);

        bool hasGlobalFlip = _pm4FlipAllObjectsY;
        bool hasObjectTranslation = _pm4ObjectTranslations.TryGetValue(objectGroupKey, out Vector3 objectTranslation)
            && !IsNearZeroVector(objectTranslation);
        bool hasObjectRotation = _pm4ObjectRotationsDegrees.TryGetValue(objectGroupKey, out Vector3 objectRotationDegrees)
            && !IsNearZeroVector(objectRotationDegrees);
        bool hasObjectScale = _pm4ObjectScales.TryGetValue(objectGroupKey, out Vector3 objectScale)
            && !IsNearOneVector(objectScale);

        if (hasLayerRotation || hasLayerScale)
        {
            Vector3 pivot = Vector3.Zero;
            if (!TryComputePm4TileCk24Pivot(tileCk24Key, applyPm4Transform, pm4Transform, out pivot)
                && _pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? objectInfo))
            {
                pivot = objectInfo.Center;
                if (applyPm4Transform)
                    pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            }

            Matrix4x4 layerRotationScale = Matrix4x4.Identity;
            if (hasLayerScale)
                layerRotationScale *= Matrix4x4.CreateScale(SanitizeScale(layerScale));

            if (hasLayerRotation)
            {
                float layerRotX = layerRotationDegrees.X * MathF.PI / 180f;
                float layerRotY = layerRotationDegrees.Y * MathF.PI / 180f;
                float layerRotZ = layerRotationDegrees.Z * MathF.PI / 180f;
                layerRotationScale *= Matrix4x4.CreateRotationX(layerRotX)
                    * Matrix4x4.CreateRotationY(layerRotY)
                    * Matrix4x4.CreateRotationZ(layerRotZ);
            }

            Matrix4x4 layerPivotTransform = Matrix4x4.CreateTranslation(-pivot)
                * layerRotationScale
                * Matrix4x4.CreateTranslation(pivot);
            transform = applyObjectTransform
                ? transform * layerPivotTransform
                : layerPivotTransform;
            applyObjectTransform = true;
        }

        if (hasLayerTranslation)
        {
            Matrix4x4 layerTranslationMatrix = Matrix4x4.CreateTranslation(layerTranslation);
            transform = applyObjectTransform
                ? transform * layerTranslationMatrix
                : layerTranslationMatrix;
            applyObjectTransform = true;
        }

        if (hasGlobalFlip || hasObjectRotation || hasObjectScale)
        {
            Vector3 pivot = Vector3.Zero;
            if (!TryComputePm4ObjectGroupPivot(objectGroupKey, applyPm4Transform, pm4Transform, out pivot)
                && _pm4ObjectLookup.TryGetValue(objectKey, out Pm4OverlayObject? objectInfo))
            {
                pivot = objectInfo.Center;
                if (applyPm4Transform)
                    pivot = ApplyPm4OverlayTransform(pivot, pm4Transform);
            }

            Matrix4x4 rotationScale = Matrix4x4.Identity;
            if (hasGlobalFlip)
            {
                rotationScale *= Matrix4x4.CreateScale(1f, -1f, 1f);
            }

            if (hasObjectScale)
                rotationScale *= Matrix4x4.CreateScale(SanitizeScale(objectScale));

            if (hasObjectRotation)
            {
                float objectRotX = objectRotationDegrees.X * MathF.PI / 180f;
                float objectRotY = objectRotationDegrees.Y * MathF.PI / 180f;
                float objectRotZ = objectRotationDegrees.Z * MathF.PI / 180f;
                rotationScale *= Matrix4x4.CreateRotationX(objectRotX)
                    * Matrix4x4.CreateRotationY(objectRotY)
                    * Matrix4x4.CreateRotationZ(objectRotZ);
            }

            Matrix4x4 objectPivotTransform = Matrix4x4.CreateTranslation(-pivot)
                * rotationScale
                * Matrix4x4.CreateTranslation(pivot);
            transform = applyObjectTransform
                ? transform * objectPivotTransform
                : objectPivotTransform;
            applyObjectTransform = true;
        }

        if (hasObjectTranslation)
        {
            Matrix4x4 objectTranslationMatrix = Matrix4x4.CreateTranslation(objectTranslation);
            transform = applyObjectTransform
                ? transform * objectTranslationMatrix
                : objectTranslationMatrix;
            applyObjectTransform = true;
        }

        return transform;
    }

    private static void TransformBounds(Vector3 boundsMin, Vector3 boundsMax, in Matrix4x4 transform,
        out Vector3 transformedMin, out Vector3 transformedMax)
    {
        transformedMin = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        transformedMax = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        Span<Vector3> corners = stackalloc Vector3[8];
        corners[0] = new Vector3(boundsMin.X, boundsMin.Y, boundsMin.Z);
        corners[1] = new Vector3(boundsMax.X, boundsMin.Y, boundsMin.Z);
        corners[2] = new Vector3(boundsMin.X, boundsMax.Y, boundsMin.Z);
        corners[3] = new Vector3(boundsMax.X, boundsMax.Y, boundsMin.Z);
        corners[4] = new Vector3(boundsMin.X, boundsMin.Y, boundsMax.Z);
        corners[5] = new Vector3(boundsMax.X, boundsMin.Y, boundsMax.Z);
        corners[6] = new Vector3(boundsMin.X, boundsMax.Y, boundsMax.Z);
        corners[7] = new Vector3(boundsMax.X, boundsMax.Y, boundsMax.Z);

        for (int i = 0; i < corners.Length; i++)
        {
            Vector3 transformed = Vector3.Transform(corners[i], transform);
            transformedMin = Vector3.Min(transformedMin, transformed);
            transformedMax = Vector3.Max(transformedMax, transformed);
        }
    }

    public void Dispose()
    {
        ReleasePm4LoadCancellation(cancelPendingLoad: true);
        _terrainManager.OnTileLoaded -= OnTileLoaded;
        _terrainManager.OnTileUnloaded -= OnTileUnloaded;
        _terrainManager.Dispose();
        _wdlTerrain?.Dispose();
        _assets.Dispose();
        _bbRenderer?.Dispose();
        _skyDome.Dispose();
        _mdxInstances.Clear();
        _skyboxInstances.Clear();
        _wmoInstances.Clear();
        _tileMdxInstances.Clear();
        _tileSkyboxInstances.Clear();
        _tileWmoInstances.Clear();
        _tileMdxVisibilityBuckets.Clear();
        _tileWmoVisibilityBuckets.Clear();
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _pm4TileObjects.Clear();
        _pm4TileMscnPoints.Clear();
        _pm4TileMspvPoints.Clear();
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ResearchBySourcePath.Clear();
        _pm4ResearchUnavailablePaths.Clear();
        _pm4ObjectLookup.Clear();
        _highlightedPm4ObjectKeys.Clear();
        _pm4MergedObjectGroupKeys.Clear();
        _pm4GroupToObjectKeys.Clear();
        _pm4ObjectGroupBounds.Clear();
        _pm4TileCk24Bounds.Clear();
        _pm4ObjectTranslations.Clear();
        _pm4ObjectRotationsDegrees.Clear();
        _pm4ObjectScales.Clear();
        _pm4TileCk24Translations.Clear();
        _pm4TileCk24RotationsDegrees.Clear();
        _pm4TileCk24Scales.Clear();
    }

    private sealed class Pm4OverlayAsyncLoadResult
    {
        public Pm4OverlayAsyncLoadResult(
            int requestId,
            Pm4OverlayCacheData? cacheData,
            (int minTileX, int minTileY, int maxTileX, int maxTileY)? loadedCameraWindow,
            IReadOnlyCollection<(int tileX, int tileY)> knownMapTiles,
            IReadOnlyCollection<(int tileX, int tileY)> coveredMapTiles,
            (int tileX, int tileY, uint ck24, int objectPart)? selectedObjectKey,
            double loadElapsedMs,
            string statusMessage,
            bool cancelled)
        {
            RequestId = requestId;
            CacheData = cacheData;
            LoadedCameraWindow = loadedCameraWindow;
            KnownMapTiles = knownMapTiles;
            CoveredMapTiles = coveredMapTiles;
            SelectedObjectKey = selectedObjectKey;
            LoadElapsedMs = loadElapsedMs;
            StatusMessage = statusMessage;
            Cancelled = cancelled;
        }

        public int RequestId { get; }
        public Pm4OverlayCacheData? CacheData { get; }
        public (int minTileX, int minTileY, int maxTileX, int maxTileY)? LoadedCameraWindow { get; }
        public IReadOnlyCollection<(int tileX, int tileY)> KnownMapTiles { get; }
        public IReadOnlyCollection<(int tileX, int tileY)> CoveredMapTiles { get; }
        public (int tileX, int tileY, uint ck24, int objectPart)? SelectedObjectKey { get; }
        public double LoadElapsedMs { get; }
        public string StatusMessage { get; }
        public bool Cancelled { get; }
    }
}

internal sealed class Pm4OverlayObject
{
    public static Pm4OverlayObject FromCachedLocalized(
        string sourcePath,
        uint mshdField00,
        uint mshdRegionId,
        uint mshdField08,
        uint ck24,
        byte ck24Type,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        Pm4LinkedPositionRefSummary linkedPositionRefSummary,
        List<Pm4LineSegment> localizedLines,
        List<Pm4Triangle> localizedTriangles,
        int surfaceCount,
        int totalIndexCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMscnRefIndex,
        float averageSurfaceHeight,
        Vector3 placementAnchor,
        float baseRotationRadians,
        Pm4PlanarTransform planarTransform,
        Vector3 boundsMin,
        Vector3 boundsMax,
        IReadOnlyList<Pm4ConnectorKey> connectorKeys)
    {
        return new Pm4OverlayObject(
            sourcePath,
            mshdField00,
            mshdRegionId,
            mshdField08,
            ck24,
            ck24Type,
            objectPartId,
            linkGroupObjectId,
            linkedPositionRefCount,
            linkedPositionRefSummary,
            localizedLines,
            localizedTriangles,
            surfaceCount,
            totalIndexCount,
            dominantGroupKey,
            dominantAttributeMask,
            dominantMscnRefIndex,
            averageSurfaceHeight,
            placementAnchor,
            baseRotationRadians,
            planarTransform,
            connectorKeys,
            boundsMin,
            boundsMax,
            geometryIsLocalized: true);
    }

    public Pm4OverlayObject(
        string sourcePath,
        uint mshdField00,
        uint mshdRegionId,
        uint mshdField08,
        uint ck24,
        byte ck24Type,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        Pm4LinkedPositionRefSummary linkedPositionRefSummary,
        List<Pm4LineSegment> lines,
        List<Pm4Triangle> triangles,
        int surfaceCount,
        int totalIndexCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMscnRefIndex,
        float averageSurfaceHeight,
        Vector3 placementAnchor,
        float baseRotationRadians,
        Pm4PlanarTransform planarTransform,
        IReadOnlyList<Pm4ConnectorKey> connectorKeys)
        : this(
            sourcePath,
            mshdField00,
            mshdRegionId,
            mshdField08,
            ck24,
            ck24Type,
            objectPartId,
            linkGroupObjectId,
            linkedPositionRefCount,
            linkedPositionRefSummary,
            lines,
            triangles,
            surfaceCount,
            totalIndexCount,
            dominantGroupKey,
            dominantAttributeMask,
            dominantMscnRefIndex,
            averageSurfaceHeight,
            placementAnchor,
            baseRotationRadians,
            planarTransform,
            connectorKeys,
            default,
            default,
            geometryIsLocalized: false)
    {
    }

    private Pm4OverlayObject(
        string sourcePath,
        uint mshdField00,
        uint mshdRegionId,
        uint mshdField08,
        uint ck24,
        byte ck24Type,
        int objectPartId,
        uint linkGroupObjectId,
        int linkedPositionRefCount,
        Pm4LinkedPositionRefSummary linkedPositionRefSummary,
        List<Pm4LineSegment> lines,
        List<Pm4Triangle> triangles,
        int surfaceCount,
        int totalIndexCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMscnRefIndex,
        float averageSurfaceHeight,
        Vector3 placementAnchor,
        float baseRotationRadians,
        Pm4PlanarTransform planarTransform,
        IReadOnlyList<Pm4ConnectorKey> connectorKeys,
        Vector3 cachedBoundsMin,
        Vector3 cachedBoundsMax,
        bool geometryIsLocalized)
    {
        SourcePath = sourcePath;
        MshdField00 = mshdField00;
        MshdRegionId = mshdRegionId;
        MshdField08 = mshdField08;
        Ck24 = ck24;
        Ck24Type = ck24Type;
        ObjectPartId = objectPartId;
        LinkGroupObjectId = linkGroupObjectId;
        LinkedPositionRefCount = linkedPositionRefCount;
        LinkedPositionRefSummary = linkedPositionRefSummary;
        Lines = lines;
        Triangles = triangles;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMscnRefIndex = dominantMscnRefIndex;
        AverageSurfaceHeight = averageSurfaceHeight;
        PlanarTransform = planarTransform;
        ConnectorKeys = connectorKeys;
        if (geometryIsLocalized)
        {
            BoundsMin = cachedBoundsMin;
            BoundsMax = cachedBoundsMax;
        }
        else
        {
            (BoundsMin, BoundsMax) = ComputeBounds(lines, triangles);
        }

        Center = (BoundsMin + BoundsMax) * 0.5f;
        PlacementAnchor = IsFiniteVector(placementAnchor) ? placementAnchor : Center;
        BaseRotationRadians = float.IsFinite(baseRotationRadians) ? baseRotationRadians : 0f;
        BaseTransform = WorldScene.BuildPm4BaseTransform(PlacementAnchor, BaseRotationRadians);
        if (geometryIsLocalized)
        {
            Lines = lines;
            Triangles = triangles;
        }
        else
        {
            if (!Matrix4x4.Invert(BaseTransform, out Matrix4x4 inverseBaseTransform))
                inverseBaseTransform = Matrix4x4.CreateTranslation(-PlacementAnchor);

            Lines = LocalizeLines(lines, inverseBaseTransform);
            Triangles = LocalizeTriangles(triangles, inverseBaseTransform);
        }
    }

    public string SourcePath { get; }
    public uint MshdField00 { get; }
    public uint MshdRegionId { get; }
    public uint MshdField08 { get; }
    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);

    // Byte-decomposed view of the 24-bit Ck24. The 32-bit MSUR._0x1C
    // (a.k.a. PackedParams) is interpreted as [0xAA type] [0xBB high]
    // [0xCC low] [0x00 pad] per the user's session-derived model
    // (spec 058). The low byte of the 32-bit word is observed to be
    // zero in our data; treat it as a padding trailer, not identity.
    // Ck24ObjectId above is the lossy flattening of these two bytes
    // into a single 16-bit ID. These two byte fields are pure getters
    // - no new state - so the change is purely additive.
    public byte Ck24HighByte => (byte)((Ck24 >> 8) & 0xFF);
    public byte Ck24LowByte => (byte)(Ck24 & 0xFF);

    public int ObjectPartId { get; }
    public uint LinkGroupObjectId { get; }
    public int LinkedPositionRefCount { get; }
    public Pm4LinkedPositionRefSummary LinkedPositionRefSummary { get; }
    public List<Pm4LineSegment> Lines { get; }
    public List<Pm4Triangle> Triangles { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMscnRefIndex { get; }
    public float AverageSurfaceHeight { get; }
    public Pm4PlanarTransform PlanarTransform { get; }
    public IReadOnlyList<Pm4ConnectorKey> ConnectorKeys { get; }
    public Matrix4x4 BaseTransform { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
    public uint DistinctTypeFlags { get; set; }
    public Vector3 Center { get; }
    public Vector3 PlacementAnchor { get; }
    public float BaseRotationRadians { get; }

    private static List<Pm4LineSegment> LocalizeLines(List<Pm4LineSegment> lines, in Matrix4x4 inverseBaseTransform)
    {
        var localized = new List<Pm4LineSegment>(lines.Count);
        for (int i = 0; i < lines.Count; i++)
        {
            Pm4LineSegment line = lines[i];
            localized.Add(new Pm4LineSegment(
                Vector3.Transform(line.From, inverseBaseTransform),
                Vector3.Transform(line.To, inverseBaseTransform)));
        }

        return localized;
    }

    private static List<Pm4Triangle> LocalizeTriangles(List<Pm4Triangle> triangles, in Matrix4x4 inverseBaseTransform)
    {
        var localized = new List<Pm4Triangle>(triangles.Count);
        for (int i = 0; i < triangles.Count; i++)
        {
            Pm4Triangle tri = triangles[i];
            localized.Add(new Pm4Triangle(
                Vector3.Transform(tri.A, inverseBaseTransform),
                Vector3.Transform(tri.B, inverseBaseTransform),
                Vector3.Transform(tri.C, inverseBaseTransform)));
        }

        return localized;
    }

    private static bool IsFiniteVector(Vector3 value)
    {
        return float.IsFinite(value.X)
            && float.IsFinite(value.Y)
            && float.IsFinite(value.Z);
    }

    private static (Vector3 min, Vector3 max) ComputeBounds(List<Pm4LineSegment> lines, List<Pm4Triangle> triangles)
    {
        Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);
        bool hasData = false;

        for (int i = 0; i < lines.Count; i++)
        {
            min = Vector3.Min(min, lines[i].From);
            min = Vector3.Min(min, lines[i].To);
            max = Vector3.Max(max, lines[i].From);
            max = Vector3.Max(max, lines[i].To);
            hasData = true;
        }

        for (int i = 0; i < triangles.Count; i++)
        {
            min = Vector3.Min(min, triangles[i].A);
            min = Vector3.Min(min, triangles[i].B);
            min = Vector3.Min(min, triangles[i].C);
            max = Vector3.Max(max, triangles[i].A);
            max = Vector3.Max(max, triangles[i].B);
            max = Vector3.Max(max, triangles[i].C);
            hasData = true;
        }

        if (!hasData)
            return (Vector3.Zero, Vector3.Zero);

        return (min, max);
    }
}

internal sealed class Pm4ResearchContext
{
    public Pm4ResearchContext(
        string sourcePath,
        CorePm4ExplorationSnapshot snapshot,
        CorePm4DecodeAuditReport decodeAudit,
        CorePm4TileObjectHypothesisReport hypothesisReport,
        Pm4File? rawDocument = null)
    {
        SourcePath = sourcePath;
        Snapshot = snapshot;
        DecodeAudit = decodeAudit;
        HypothesisReport = hypothesisReport;
        RawDocument = rawDocument;
    }

    public string SourcePath { get; }
    public CorePm4ExplorationSnapshot Snapshot { get; }
    public CorePm4DecodeAuditReport DecodeAudit { get; }
    public CorePm4TileObjectHypothesisReport HypothesisReport { get; }
    public Pm4File? RawDocument { get; }
}

internal sealed record Pm4WmoCorrelationSummary(
    int WmoPlacementCount,
    int WmoMeshResolvedCount,
    int Pm4ObjectCount,
    int PlacementsWithCandidates,
    int PlacementsWithNearCandidates,
    int MaxMatchesPerPlacement);

internal sealed record Pm4WmoCorrelationAdtPlacementInfo(
    bool Found,
    ushort Flags,
    Vector3 RawBoundsMin,
    Vector3 RawBoundsMax);

internal sealed record Pm4WmoCorrelationMeshInfo(
    bool Available,
    int Version,
    int GroupCount,
    int VertexCount,
    int IndexCount,
    int TriangleCount,
    int BatchCount,
    Vector3 LocalBoundsMin,
    Vector3 LocalBoundsMax,
    int FootprintSampleCount,
    int WorldFootprintHullPointCount,
    float WorldFootprintArea);

internal sealed record Pm4WmoCorrelationMatch(
    int TileX,
    int TileY,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int ObjectPartId,
    uint LinkGroupObjectId,
    int SurfaceCount,
    int LinkedPositionRefCount,
    byte DominantGroupKey,
    byte DominantAttributeMask,
    uint DominantMscnRefIndex,
    float AverageSurfaceHeight,
    bool SameTile,
    float PlanarGap,
    float VerticalGap,
    float CenterDistance,
    float PlanarOverlapRatio,
    float VolumeOverlapRatio,
    float FootprintOverlapRatio,
    float FootprintAreaRatio,
    float FootprintDistance,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center);

internal sealed record Pm4WmoCorrelationPlacement(
    int TileX,
    int TileY,
    int UniqueId,
    string ModelName,
    string ModelPath,
    string ModelKey,
    Vector3 PlacementPosition,
    Vector3 PlacementRotation,
    float PlacementScale,
    Pm4WmoCorrelationAdtPlacementInfo AdtPlacement,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    Pm4WmoCorrelationMeshInfo WmoMesh,
    int Pm4CandidateCount,
    int Pm4NearCandidateCount,
    IReadOnlyList<Pm4WmoCorrelationMatch> Pm4Matches);

internal sealed record Pm4WmoCorrelationReport(
    DateTime GeneratedAtUtc,
    string Pm4Status,
    Pm4WmoCorrelationSummary Summary,
    IReadOnlyList<Pm4WmoCorrelationPlacement> Placements);

internal sealed record Pm4ObjectMatchSummary(
    int Pm4ObjectCount,
    int WmoPlacementCount,
    int M2PlacementCount,
    int ObjectsWithCandidates,
    int ObjectsWithNearCandidates,
    int MaxMatchesPerObject);

internal struct Pm4TileBuildDiagnostics
{
    public int TotalMsurCount;
    public int DroppedShortIndexCount;
    public int DroppedOutOfRangeMsviCount;
    public int DroppedEmptyComponentCount;
    public int DroppedLongEdgeLines;
    public int DroppedEmptyFile;

    /// <summary>MSPV/MSPI wall faces emitted for this tile. Zero when wall rendering is off.</summary>
    public int WallFaceCount;
}

internal sealed record Pm4ObjectMatchCandidate(
    int TileX,
    int TileY,
    string Kind,
    int UniqueId,
    string ModelName,
    string ModelPath,
    string ModelKey,
    bool SameTile,
    bool AssetResolved,
    string EvidenceSource,
    ushort PlacementFlags,
    Vector3 PlacementPosition,
    Vector3 PlacementRotation,
    float PlacementScale,
    float AnchorPlanarGap,
    float PlanarGap,
    float VerticalGap,
    float CenterDistance,
    float PlanarOverlapRatio,
    float VolumeOverlapRatio,
    float FootprintOverlapRatio,
    float FootprintAreaRatio,
    float FootprintDistance,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    Vector3 Center,
    int MeshGroupCount,
    int MeshVertexCount,
    int MeshTriangleCount,
    int FootprintSampleCount,
    float WorldFootprintArea);

internal sealed record Pm4ObjectMatchObject(
    int TileX,
    int TileY,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int ObjectPartId,
    uint LinkGroupObjectId,
    int SurfaceCount,
    int LinkedPositionRefCount,
    byte DominantGroupKey,
    byte DominantAttributeMask,
    uint DominantMscnRefIndex,
    float AverageSurfaceHeight,
    Pm4LinkedPositionRefSummary LinkedPositionRefSummary,
    Vector3 PlacementAnchor,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center,
    int CandidateCount,
    int NearCandidateCount,
    int WmoCandidateCount,
    int M2CandidateCount,
    IReadOnlyList<Pm4ObjectMatchCandidate> Candidates);

internal sealed record Pm4ObjectMatchReport(
    DateTime GeneratedAtUtc,
    string MapName,
    string Pm4Status,
    Pm4ObjectMatchSummary Summary,
    IReadOnlyList<Pm4ObjectMatchObject> Objects);

internal readonly record struct Pm4PlacementMatchState(
    int TileX,
    int TileY,
    string Kind,
    int UniqueId,
    string ModelName,
    string ModelPath,
    string ModelKey,
    string AssetProfileKey,
    bool AssetResolved,
    string EvidenceSource,
    ushort PlacementFlags,
    Vector3 PlacementPosition,
    Vector3 PlacementRotation,
    float PlacementScale,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    IReadOnlyList<Vector2> FootprintHull,
    float FootprintArea,
    int MeshGroupCount,
    int MeshVertexCount,
    int MeshTriangleCount,
    int FootprintSampleCount,
    float WorldFootprintArea,
    IReadOnlyList<Pm4PlacementGeometryVariant> GeometryVariants)
{
    public Vector3 Center => (WorldBoundsMin + WorldBoundsMax) * 0.5f;

    public bool SameTile(int tileX, int tileY) => TileX == tileX && TileY == tileY;
}

internal readonly record struct Pm4PlacementGeometryVariant(
    string AssetProfileKey,
    string EvidenceSource,
    Vector3 WorldBoundsMin,
    Vector3 WorldBoundsMax,
    IReadOnlyList<Vector2> FootprintHull,
    float FootprintArea,
    int MeshGroupCount,
    int MeshVertexCount,
    int MeshTriangleCount,
    int FootprintSampleCount,
    float WorldFootprintArea,
    Pm4ShapeSignature ShapeSignature,
    byte? CorrelatedGroupKey)
{
    public Vector3 Center => (WorldBoundsMin + WorldBoundsMax) * 0.5f;
}

internal readonly record struct Pm4PlacementMatchEvaluation(
    Pm4PlacementMatchState Placement,
    float AnchorPlanarGap,
    CorePm4CorrelationMetrics Metrics);

internal readonly record struct Pm4AssetProfileState(
    string AssetProfileKey,
    string Kind,
    string ModelName,
    string ModelPath,
    string ModelKey,
    string EvidenceSource,
    byte? CorrelatedGroupKey,
    int MeshGroupCount,
    int MeshVertexCount,
    int MeshTriangleCount,
    int FootprintSampleCount,
    Pm4ShapeSignature ShapeSignature);

internal readonly record struct Pm4AssetProfileMatchEvaluation(
    Pm4AssetProfileState Profile,
    CorePm4CorrelationMetrics Metrics);

internal readonly record struct Pm4ObjectMatchState(
    int TileX,
    int TileY,
    (int tileX, int tileY, uint ck24, int objectPart) ObjectKey,
    Pm4OverlayObject Object,
    Vector3 PlacementAnchor,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 Center,
    IReadOnlyList<Vector2> FootprintHull,
    float FootprintArea,
    Pm4ShapeSignature ShapeSignature);

public readonly record struct Pm4SurfaceGroupCluster(
    byte GroupKey,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    int SurfaceCount);

internal readonly record struct Pm4ShapeSignature(
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    IReadOnlyList<Vector2> FootprintHull,
    float FootprintArea);

public readonly record struct Pm4OfflineObjExportSummary(
    string OutputDirectory,
    string ManifestPath,
    int SourceFileCount,
    int ExportedTileCount,
    int ExportedObjectCount,
    int ZeroObjectFileCount,
    int DecodeFailedCount,
    int ReadFailedCount);

internal readonly struct Pm4LineSegment
{
    public Pm4LineSegment(Vector3 from, Vector3 to)
    {
        From = from;
        To = to;
    }

    public Vector3 From { get; }
    public Vector3 To { get; }
}

internal readonly struct Pm4Triangle
{
    public Pm4Triangle(Vector3 a, Vector3 b, Vector3 c)
    {
        A = a;
        B = b;
        C = c;
    }

    public Vector3 A { get; }
    public Vector3 B { get; }
    public Vector3 C { get; }
}

public readonly struct Pm4OverlayTileStats
{
    public Pm4OverlayTileStats(int tileX, int tileY, int objectCount, int lineCount, int triangleCount)
    {
        TileX = tileX;
        TileY = tileY;
        ObjectCount = objectCount;
        LineCount = lineCount;
        TriangleCount = triangleCount;
    }

    public int TileX { get; }
    public int TileY { get; }
    public int ObjectCount { get; }
    public int LineCount { get; }
    public int TriangleCount { get; }
}

/// <summary>
/// Lightweight placement instance — just a model key and world transform.
/// The actual renderer is looked up from WorldAssetManager at render time.
/// </summary>
public enum ObjectType { None, Wmo, Mdx }

public enum UniqueIdVisibilityScope
{
    PerMap,
    CameraTile
}

public readonly struct UniqueIdArchaeologyLayer
{
    public UniqueIdArchaeologyLayer(int layerNumber, int minUniqueId, int maxUniqueId, int placementCount, int wmoCount, int mdxCount)
    {
        LayerNumber = layerNumber;
        MinUniqueId = minUniqueId;
        MaxUniqueId = maxUniqueId;
        PlacementCount = placementCount;
        WmoCount = wmoCount;
        MdxCount = mdxCount;
    }

    public int LayerNumber { get; }
    public int MinUniqueId { get; }
    public int MaxUniqueId { get; }
    public int PlacementCount { get; }
    public int WmoCount { get; }
    public int MdxCount { get; }
}

public readonly struct HoveredAssetInfo
{
    public HoveredAssetInfo(
        string assetKind,
        string displayName,
        string sourcePath,
        string detailLine,
        Vector3 worldPosition,
        int additionalHitCount,
        (int tileX, int tileY, uint ck24, int objectPart)? pm4ObjectKey,
        ObjectType sceneObjectType = ObjectType.None,
        int sceneObjectIndex = -1,
        string? wlBodyKey = null,
        bool isPreciseRayHit = false)
    {
        AssetKind = assetKind ?? string.Empty;
        DisplayName = displayName ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
        DetailLine = detailLine ?? string.Empty;
        WorldPosition = worldPosition;
        AdditionalHitCount = Math.Max(0, additionalHitCount);
        Pm4ObjectKey = pm4ObjectKey;
        SceneObjectType = sceneObjectType;
        SceneObjectIndex = sceneObjectIndex;
        WlBodyKey = wlBodyKey ?? string.Empty;
        IsPreciseRayHit = isPreciseRayHit;
    }

    public string AssetKind { get; }
    public string DisplayName { get; }
    public string SourcePath { get; }
    public string DetailLine { get; }
    public Vector3 WorldPosition { get; }
    public int AdditionalHitCount { get; }
    public (int tileX, int tileY, uint ck24, int objectPart)? Pm4ObjectKey { get; }
    public ObjectType SceneObjectType { get; }
    public int SceneObjectIndex { get; }
    public string WlBodyKey { get; }
    public bool IsPreciseRayHit { get; }
    public bool HasSceneObject => SceneObjectType is ObjectType.Mdx or ObjectType.Wmo && SceneObjectIndex >= 0;

    public HoveredAssetInfo WithPreciseRayHit() => new(
        AssetKind, DisplayName, SourcePath, DetailLine, WorldPosition, AdditionalHitCount, Pm4ObjectKey,
        SceneObjectType, SceneObjectIndex, WlBodyKey, isPreciseRayHit: true);
}

public readonly record struct SceneObjectPickHit(
    ObjectType ObjectType,
    int ObjectIndex,
    float Distance,
    string ModelName,
    string ModelPath,
    int UniqueId,
    Vector3 PlacementPosition,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 SelectionPoint,
    float SelectionPointDistanceSq,
    bool SharesClickedChunk,
    int ChunkGridDistance)
{
    public string KindLabel => ObjectType == ObjectType.Wmo ? "WMO" : "MDX";
}
