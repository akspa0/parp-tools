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
using static WoWViewer.Terrain.Pm4OverlayScene;

namespace WoWViewer.Terrain;

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

/// <summary>One node of the PM4 outliner: a placed object, or an unresolved object group.</summary>
public sealed record Pm4OutlineObject(
    (int tileX, int tileY, uint ck24, int objectPart) Key,
    uint Ck24,
    byte Ck24Type,
    float PlacementZ,
    int SurfaceCount,
    System.Numerics.Vector3 BoundsMin,
    System.Numerics.Vector3 BoundsMax,
    System.Numerics.Vector3 Center,
    string? AssetName,
    int? UniqueId,
    float? MatchDelta);

/// <summary>A tile's worth of PM4 objects inside one MSHD region.</summary>
public sealed record Pm4OutlineTile(int TileX, int TileY, IReadOnlyList<Pm4OutlineObject> Objects);

/// <summary>One MSHD region, the top level of the PM4 outliner.</summary>
public sealed record Pm4OutlineRegion(
    uint RegionId,
    IReadOnlyList<Pm4OutlineTile> Tiles,
    int ObjectCount,
    int NamedObjectCount);

/// <summary>One surface class in the loaded scene, split by whether its objects carry a height.</summary>
public sealed record Pm4SceneClassFact(byte SurfaceClass, int WithHeight, int WithoutHeight);

/// <summary>Live PM4 measurements over the loaded scene.</summary>
public sealed record Pm4SceneFacts(
    int Objects,
    int ObjectsWithHeight,
    int ObjectsWithoutHeight,
    int ObjectsResolvedToAsset,
    int ClassHeightDisagreements,
    IReadOnlyList<Pm4SceneClassFact> Classes);
