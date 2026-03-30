using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using MdxViewer.DataSources;
using MdxViewer.Logging;
using MdxViewer.Population;
using MdxViewer.Rendering;
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
using CorePm4MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using CorePm4MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using CorePm4ObjectGroupKey = WowViewer.Core.PM4.Models.Pm4ObjectGroupKey;
using CorePm4CoordinateModeResolution = WowViewer.Core.PM4.Models.Pm4CoordinateModeResolution;
using CorePm4PlacementSolution = WowViewer.Core.PM4.Models.Pm4PlacementSolution;
using Pm4PlanarTransform = WowViewer.Core.PM4.Models.Pm4PlanarTransform;
using CorePm4PlacementContract = WowViewer.Core.PM4.Services.Pm4PlacementContract;
using CorePm4PlacementMath = WowViewer.Core.PM4.Services.Pm4PlacementMath;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using CorePm4DecodeAuditReport = WowViewer.Core.PM4.Models.Pm4DecodeAuditReport;
using CorePm4ExplorationSnapshot = WowViewer.Core.PM4.Models.Pm4ExplorationSnapshot;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using CorePm4ObjectHypothesis = WowViewer.Core.PM4.Models.Pm4ObjectHypothesis;
using MprlEntry = WowViewer.Core.PM4.Models.Pm4MprlEntry;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using MsurEntry = WowViewer.Core.PM4.Models.Pm4MsurEntry;
using Pm4File = WowViewer.Core.PM4.Research.Pm4ResearchDocument;
using CorePm4ReferenceAudit = WowViewer.Core.PM4.Models.Pm4ReferenceAudit;
using CorePm4ResearchAuditAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchAuditAnalyzer;
using CorePm4ResearchHierarchyAnalyzer = WowViewer.Core.PM4.Research.Pm4ResearchHierarchyAnalyzer;
using CorePm4ResearchSnapshotBuilder = WowViewer.Core.PM4.Research.Pm4ResearchSnapshotBuilder;
using CorePm4TileObjectHypothesisReport = WowViewer.Core.PM4.Models.Pm4TileObjectHypothesisReport;

namespace MdxViewer.Terrain;

public enum Pm4OverlayColorMode
{
    Ck24Type,
    Ck24ObjectId,
    Ck24Key,
    Tile,
    GroupKey,
    AttributeMask,
    Height
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
        int surfaceCount,
        byte dominantGroupKey,
        byte dominantAttributeMask,
        uint dominantMdosIndex,
        float averageSurfaceHeight,
        Vector3 boundsMin,
        Vector3 boundsMax,
        Vector3 center,
        float nearestPositionRefDistance,
        bool swapPlanarAxes,
        bool invertU,
        bool invertV,
        bool invertsWinding)
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
        SurfaceCount = surfaceCount;
        DominantGroupKey = dominantGroupKey;
        DominantAttributeMask = dominantAttributeMask;
        DominantMdosIndex = dominantMdosIndex;
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
    public int SurfaceCount { get; }
    public byte DominantGroupKey { get; }
    public byte DominantAttributeMask { get; }
    public uint DominantMdosIndex { get; }
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
        int mdosCount,
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
        MdosCount = mdosCount;
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
    public int MdosCount { get; }
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
        IReadOnlyList<Pm4ResearchHypothesisMatch> topMatches)
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
        uint dominantMdosIndex,
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
        DominantMdosIndex = dominantMdosIndex;
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
    public uint DominantMdosIndex { get; }
    public bool IsSelected { get; }
}

public readonly struct Pm4SelectedObjectGraphMdosNode
{
    public Pm4SelectedObjectGraphMdosNode(
        uint mdosIndex,
        int partCount,
        int surfaceCount,
        int totalIndexCount,
        IReadOnlyList<byte> attributeMasks,
        IReadOnlyList<byte> groupKeys,
        IReadOnlyList<Pm4SelectedObjectGraphPartNode> parts)
    {
        MdosIndex = mdosIndex;
        PartCount = partCount;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        AttributeMasks = attributeMasks;
        GroupKeys = groupKeys;
        Parts = parts;
    }

    public uint MdosIndex { get; }
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
        IReadOnlyList<uint> mdosIndices,
        IReadOnlyList<byte> attributeMasks,
        IReadOnlyList<byte> groupKeys,
        IReadOnlyList<Pm4SelectedObjectGraphMdosNode> mdosGroups)
    {
        LinkGroupObjectId = linkGroupObjectId;
        PartCount = partCount;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        LinkedPositionRefCount = linkedPositionRefCount;
        LinkedPositionRefSummary = linkedPositionRefSummary;
        MdosIndices = mdosIndices;
        AttributeMasks = attributeMasks;
        GroupKeys = groupKeys;
        MdosGroups = mdosGroups;
    }

    public uint LinkGroupObjectId { get; }
    public int PartCount { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public int LinkedPositionRefCount { get; }
    public Pm4LinkedPositionRefSummary LinkedPositionRefSummary { get; }
    public IReadOnlyList<uint> MdosIndices { get; }
    public IReadOnlyList<byte> AttributeMasks { get; }
    public IReadOnlyList<byte> GroupKeys { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphMdosNode> MdosGroups { get; }
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
        bool splitByMdos,
        bool splitByConnectivity,
        int tileCount,
        int linkGroupCount,
        int mdosGroupCount,
        int partCount,
        int surfaceCount,
        int totalIndexCount,
        int attributeMaskCount,
        int groupKeyCount,
        IReadOnlyList<Pm4SelectedObjectGraphLinkNode> linkGroups)
    {
        SelectedTileX = selectedTileX;
        SelectedTileY = selectedTileY;
        Ck24 = ck24;
        Ck24Type = ck24Type;
        Ck24ObjectId = ck24ObjectId;
        SelectedObjectPartId = selectedObjectPartId;
        SplitByMdos = splitByMdos;
        SplitByConnectivity = splitByConnectivity;
        TileCount = tileCount;
        LinkGroupCount = linkGroupCount;
        MdosGroupCount = mdosGroupCount;
        PartCount = partCount;
        SurfaceCount = surfaceCount;
        TotalIndexCount = totalIndexCount;
        AttributeMaskCount = attributeMaskCount;
        GroupKeyCount = groupKeyCount;
        LinkGroups = linkGroups;
    }

    public int SelectedTileX { get; }
    public int SelectedTileY { get; }
    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId { get; }
    public int SelectedObjectPartId { get; }
    public bool SplitByMdos { get; }
    public bool SplitByConnectivity { get; }
    public int TileCount { get; }
    public int LinkGroupCount { get; }
    public int MdosGroupCount { get; }
    public int PartCount { get; }
    public int SurfaceCount { get; }
    public int TotalIndexCount { get; }
    public int AttributeMaskCount { get; }
    public int GroupKeyCount { get; }
    public IReadOnlyList<Pm4SelectedObjectGraphLinkNode> LinkGroups { get; }
}

internal readonly record struct Pm4ConnectorKey(int X, int Y, int Z);

/// <summary>
/// Combines terrain (WDT/ADT), WMO placements (MODF), and MDX placements (MDDF)
/// into a single world scene — the same way the game client renders a map.
/// 
/// Uses <see cref="WorldAssetManager"/> to ensure each model is loaded exactly once.
/// Instances are lightweight structs holding only a model key + transform.
/// </summary>
public class WorldScene : ISceneRenderer
{
    private static float? JsonFiniteOrNull(float value) => float.IsFinite(value) ? value : null;

    private static float DecodeRawMprlPackedAngleRadians(MprlEntry positionRef)
    {
        return positionRef.Unk04 * (2f * MathF.PI / 65536f);
    }

    private readonly GL _gl;
    private readonly TerrainManager _terrainManager;
    private readonly WorldAssetManager _assets;
    private readonly Pm4OverlayCacheService? _pm4OverlayCacheService;

    // Lightweight instance lists — just a key + transform, no renderer reference
    // These are rebuilt from _tileMdxInstances/_tileWmoInstances when tiles change
    private List<ObjectInstance> _mdxInstances = new();
    private List<ObjectInstance> _skyboxInstances = new();
    private List<ObjectInstance> _wmoInstances = new();

    // Per-tile instance storage for lazy load/unload
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileMdxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileSkyboxInstances = new();
    private readonly Dictionary<(int, int), List<ObjectInstance>> _tileWmoInstances = new();
    private readonly List<ObjectInstance> _externalMdxInstances = new();
    private readonly List<ObjectInstance> _externalSkyboxInstances = new();
    private readonly List<ObjectInstance> _externalWmoInstances = new();
    private readonly List<ObjectInstance> _taxiActorInstances = new();
    private bool _instancesDirty = false;

    private bool _objectsVisible = true;
    private bool _wmosVisible = true;
    private bool _doodadsVisible = true;
        private bool _objectFogEnabled = false;

    // Frustum culling
    private readonly FrustumCuller _frustumCuller = new();
    private const float DoodadCullDistance = 5000f; // Max distance for small doodads; raised to prevent long-range tree pop-out
    private const float DoodadCullDistanceSq = DoodadCullDistance * DoodadCullDistance;
    private const float DoodadSmallThreshold = 10f; // AABB diagonal below this = "small" (relaxed — only cull tiny objects)
    private const float FadeStartFraction = 0.80f;  // Fade begins at 80% of cull distance
    private const float WmoCullDistance = 4000f;     // Minimum WMO instance cull distance; actual range expands with fog distance
    private const float WmoFadeStartFraction = 0.85f;
    private const float NoCullRadius = 512f;         // Objects within this radius are never frustum-culled
    private const float NoCullRadiusSq = NoCullRadius * NoCullRadius;
    private const float HoverInfoBrushPixels = 32f;
    private const float HoverInfoMaxScreenRadius = 96f;
    private const float WireframeRevealBrushPixels = 96f;
    private const float WireframeRevealMaxScreenRadius = 220f;

    private readonly struct VisibleMdxInstance
    {
        public VisibleMdxInstance(ObjectInstance instance, MdxRenderer renderer, float centerDistanceSq, float opaqueFade, float transparentFade)
        {
            Instance = instance;
            Renderer = renderer;
            CenterDistanceSq = centerDistanceSq;
            OpaqueFade = opaqueFade;
            TransparentFade = transparentFade;
        }

        public ObjectInstance Instance { get; }
        public MdxRenderer Renderer { get; }
        public float CenterDistanceSq { get; }
        public float OpaqueFade { get; }
        public float TransparentFade { get; }
    }

    // Scratch collections reused every frame to avoid hot-path allocations.
    private readonly HashSet<string> _updatedMdxRenderers = new();
    private readonly List<VisibleMdxInstance> _visibleMdxInstances = new();
    private readonly List<(int visibleIdx, float distSq)> _transparentSortScratch = new();
    private readonly List<int> _wireframeRevealWmoIndices = new();
    private readonly List<int> _wireframeRevealMdxIndices = new();
    private HoveredAssetInfo? _hoveredAssetInfo;
    private bool _wireframeRevealEnabled;

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
    private bool _pm4OverlayIgnoreDepth;
    private bool _pm4FlipAllObjectsY;
    private bool _showPm4PositionRefs;
    private bool _showPm4ObjectCentroids;
    private bool _pm4SplitCk24ByConnectivity;
    private bool _showPm4Type40 = true;
    private bool _showPm4Type80 = true;
    private bool _showPm4TypeOther = true;
    private bool _pm4SplitCk24ByMdos;
    private Pm4OverlayColorMode _pm4ColorMode = Pm4OverlayColorMode.Ck24Type;
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
    private float _pm4MinObjectZ;
    private float _pm4MaxObjectZ;
    private int _pm4CameraTileRadius = Pm4MinCameraTileRadius;
    private double _pm4AverageLoadMs = -1.0;
    private (int minTileX, int minTileY, int maxTileX, int maxTileY)? _pm4LoadedCameraWindow;
    private readonly HashSet<(int tileX, int tileY)> _pm4KnownMapTiles = new();
    private readonly HashSet<(int tileX, int tileY)> _pm4CoveredMapTiles = new();
    private Task<Pm4OverlayAsyncLoadResult>? _pm4LoadTask;
    private CancellationTokenSource? _pm4LoadCancellation;
    private int _pm4LoadRequestId;
    private Vector3 _lastRenderedCameraPosition;
    private bool _hasLastRenderedCameraPosition;
    private readonly Dictionary<(int tileX, int tileY), List<Pm4OverlayObject>> _pm4TileObjects = new();
    private readonly Dictionary<(int tileX, int tileY), Pm4OverlayTileStats> _pm4TileStats = new();
    private readonly Dictionary<(int tileX, int tileY), List<Vector3>> _pm4TilePositionRefs = new();
    private readonly Dictionary<string, Pm4ResearchContext> _pm4ResearchBySourcePath = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _pm4ResearchUnavailablePaths = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4OverlayObject> _pm4ObjectLookup = new();
    private readonly HashSet<(int tileX, int tileY, uint ck24, int objectPart)> _highlightedPm4ObjectKeys = new();
    private readonly Dictionary<(int tileX, int tileY, uint ck24), (int tileX, int tileY, uint ck24)> _pm4MergedObjectGroupKeys = new();
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

    // Bounding box debug rendering
    private bool _showBoundingBoxes = false;
    private BoundingBoxRenderer? _bbRenderer;
    public bool ShowBoundingBoxes { get => _showBoundingBoxes; set => _showBoundingBoxes = value; }

    // Object selection
    private ObjectType _selectedObjectType = ObjectType.None;
    private int _selectedObjectIndex = -1;
    public ObjectType SelectedObjectType => _selectedObjectType;
    public int SelectedObjectIndex => _selectedObjectIndex;
    public bool WireframeRevealEnabled => _wireframeRevealEnabled;
    public HoveredAssetInfo? HoveredAssetInfo => _hoveredAssetInfo;
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

    public void ApplyTextureSamplingSettings()
    {
        _terrainManager.Renderer.ApplyTextureSamplingSettings();
        _assets.ApplyTextureSamplingSettings();
    }
    public int Pm4LineCount => _pm4LineCount;
    public int Pm4TriangleCount => _pm4TriangleCount;
    public int Pm4RejectedLongEdges => _pm4RejectedLongEdges;
    public int Pm4VisibleObjectCount => _pm4VisibleObjectCount;
    public int Pm4VisibleLineCount => _pm4VisibleLineCount;
    public int Pm4VisibleTriangleCount => _pm4VisibleTriangleCount;
    public int Pm4PositionRefCount => _pm4PositionRefCount;
    public int Pm4VisiblePositionRefCount => _pm4VisiblePositionRefCount;
    public bool ShowPm4SolidOverlay { get => _showPm4SolidOverlay; set => _showPm4SolidOverlay = value; }
    public bool ShowPm4ObjectBounds { get => _showPm4ObjectBounds; set => _showPm4ObjectBounds = value; }
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
    public bool Pm4SplitCk24ByConnectivity { get => _pm4SplitCk24ByConnectivity; set => _pm4SplitCk24ByConnectivity = value; }
    public bool ShowPm4Type40 { get => _showPm4Type40; set => _showPm4Type40 = value; }
    public bool ShowPm4Type80 { get => _showPm4Type80; set => _showPm4Type80 = value; }
    public bool ShowPm4TypeOther { get => _showPm4TypeOther; set => _showPm4TypeOther = value; }
    public bool Pm4SplitCk24ByMdos { get => _pm4SplitCk24ByMdos; set => _pm4SplitCk24ByMdos = value; }
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
        return (objectKey.tileX, objectKey.tileY, groupKey);
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

        if (!SelectedPm4RawCk24.HasValue)
            return false;

        var tileCk24Key = SelectedPm4TileCk24Key.Value;
        foreach (var objectKey in _pm4ObjectLookup.Keys)
        {
            if (objectKey.tileX != tileCk24Key.tileX
                || objectKey.tileY != tileCk24Key.tileY
                || objectKey.ck24 != tileCk24Key.ck24)
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
                            objectPartId = obj.ObjectPartId,
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
                            dominantMdosIndex = obj.DominantMdosIndex,
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
                    _pm4SplitCk24ByMdos,
                    _pm4SplitCk24ByConnectivity,
                    ref remainingLineBudget,
                    ref remainingTriangleBudget,
                    ref rejectedLongEdges);

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
            splitCk24ByMdos = _pm4SplitCk24ByMdos,
            splitCk24ByConnectivity = _pm4SplitCk24ByConnectivity,
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
                            candidate.candidate.Object.DominantMdosIndex,
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
                    dominantMdosIndex = match.DominantMdosIndex,
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

        int objectsWithCandidates = 0;
        int objectsWithNearCandidates = 0;
        List<Pm4ObjectMatchObject> reports = new(pm4Objects.Count);

        foreach (Pm4ObjectMatchState pm4Object in pm4Objects)
        {
            Pm4ObjectMatchObject report = BuildPm4ObjectMatchObject(pm4Object, placements, resolvedMaxMatches);
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
        objectMatch = BuildPm4ObjectMatchObject(pm4Object, placements, Math.Max(1, maxMatchesPerObject));
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
        return new Pm4ObjectMatchState(tileX, tileY, objectKey, obj, placementAnchor, boundsMin, boundsMax, center, footprintHull, footprintArea);
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
                int meshGroupCount = 0;
                int meshVertexCount = 0;
                int meshTriangleCount = 0;
                int footprintSampleCount = 0;
                float worldFootprintArea = footprintArea;
                string evidenceSource = "modf-bounds";

                if (hasMeshSummary)
                {
                    TransformBounds(meshSummary.BoundsMin, meshSummary.BoundsMax, instance.Transform, out worldBoundsMin, out worldBoundsMax);
                    footprintHull = CorePm4CorrelationMath.BuildTransformedFootprintHull(meshSummary.FootprintSampleVertices, instance.Transform);
                    footprintArea = CorePm4CorrelationMath.ComputeFootprintArea(footprintHull);
                    meshGroupCount = meshSummary.GroupCount;
                    meshVertexCount = meshSummary.VertexCount;
                    meshTriangleCount = meshSummary.TriangleCount;
                    footprintSampleCount = meshSummary.FootprintSampleCount;
                    worldFootprintArea = footprintArea;
                    evidenceSource = "wmo-mesh";
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
                    worldFootprintArea));
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
                states.Add(new Pm4PlacementMatchState(
                    tileEntry.Key.Item1,
                    tileEntry.Key.Item2,
                    "m2",
                    instance.UniqueId,
                    instance.ModelName,
                    instance.ModelPath,
                    instance.ModelKey,
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
                    footprintArea));
            }
        }

        return states;
    }

    private static Pm4ObjectMatchObject BuildPm4ObjectMatchObject(
        Pm4ObjectMatchState pm4Object,
        IReadOnlyList<Pm4PlacementMatchState> placements,
        int maxMatchesPerObject)
    {
        List<Pm4PlacementMatchEvaluation> rankedCandidates = placements
            .Select(placement => new
            {
                Placement = placement,
                Metrics = CorePm4CorrelationMath.EvaluateMetrics(
                    pm4Object.BoundsMin,
                    pm4Object.BoundsMax,
                    pm4Object.Center,
                    pm4Object.FootprintHull,
                    pm4Object.FootprintArea,
                    placement.WorldBoundsMin,
                    placement.WorldBoundsMax,
                    placement.Center,
                    placement.FootprintHull,
                    placement.FootprintArea),
                AnchorPlanarGap = ComputePm4ObjectAnchorPlanarGap(pm4Object.PlacementAnchor, placement.PlacementPosition),
            })
            .OrderBy(candidate => GetPm4ObjectMatchEvidenceRank(pm4Object, candidate.Placement))
            .ThenByDescending(candidate => candidate.Placement.SameTile(pm4Object.TileX, pm4Object.TileY))
            .ThenBy(candidate => pm4Object.Object.LinkedPositionRefCount > 0 ? candidate.AnchorPlanarGap : float.MaxValue)
            .ThenByDescending(candidate => candidate.Metrics.PlanarOverlapRatio)
            .ThenByDescending(candidate => candidate.Metrics.VolumeOverlapRatio)
            .ThenBy(candidate => candidate.Metrics.PlanarGap)
            .ThenBy(candidate => candidate.Metrics.VerticalGap)
            .ThenBy(candidate => candidate.Metrics.CenterDistance)
            .ThenByDescending(candidate => candidate.Metrics.FootprintOverlapRatio)
            .ThenBy(candidate => candidate.Metrics.FootprintDistance)
            .Select(candidate => new Pm4PlacementMatchEvaluation(
                candidate.Placement,
                candidate.AnchorPlanarGap,
                candidate.Metrics))
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
            pm4Object.Object.DominantMdosIndex,
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

    private static int GetPm4ObjectMatchEvidenceRank(Pm4ObjectMatchState pm4Object, Pm4PlacementMatchState placement)
    {
        bool zeroOrRootObject = pm4Object.Object.Ck24 == 0 || pm4Object.Object.LinkGroupObjectId == 0;
        if (zeroOrRootObject)
        {
            if (pm4Object.Object.LinkedPositionRefCount > 0)
                return string.Equals(placement.Kind, "m2", StringComparison.OrdinalIgnoreCase) ? 0 : 1;

            return 0;
        }

        if (placement.Kind == "wmo" && string.Equals(placement.EvidenceSource, "wmo-mesh", StringComparison.OrdinalIgnoreCase))
            return 0;

        if (placement.Kind == "wmo")
            return 1;

        return 2;
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
    public ObjectInstance? SelectedInstance => _selectedObjectType switch
    {
        ObjectType.Wmo when _selectedObjectIndex >= 0 && _selectedObjectIndex < _wmoInstances.Count => _wmoInstances[_selectedObjectIndex],
        ObjectType.Mdx when _selectedObjectIndex >= 0 && _selectedObjectIndex < _mdxInstances.Count => _mdxInstances[_selectedObjectIndex],
        _ => null
    };

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

    // Taxi selection: -1 = show all (or none if !_showTaxi)
    private int _selectedTaxiNodeId = -1;
    private int _selectedTaxiRouteId = -1;
    private readonly Dictionary<int, string> _taxiActorModelOverrideByPath = new();
    private readonly Dictionary<int, float> _taxiActorTravelByPath = new();
    private long _lastTaxiActorTick;
    private bool _taxiActorClockInitialized;
    private bool _showTaxiActors = true;
    private float _taxiActorSpeedMultiplier = 1.0f;
    private const float TaxiActorBaseUnitsPerSecond = 650f;
    private const float TaxiActorHoverOffset = 12f;
    public int SelectedTaxiNodeId { get => _selectedTaxiNodeId; set { _selectedTaxiNodeId = value; _selectedTaxiRouteId = -1; } }
    public int SelectedTaxiRouteId { get => _selectedTaxiRouteId; set { _selectedTaxiRouteId = value; _selectedTaxiNodeId = -1; } }
    public void ClearTaxiSelection() { _selectedTaxiNodeId = -1; _selectedTaxiRouteId = -1; }
    public bool ShowTaxiActors { get => _showTaxiActors; set => _showTaxiActors = value; }
    public float TaxiActorSpeedMultiplier
    {
        get => _taxiActorSpeedMultiplier;
        set => _taxiActorSpeedMultiplier = Math.Max(0f, value);
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
            return null;

        TaxiPathLoader.TaxiNode? mountNode = ResolveTaxiActorNode(route);
        return string.IsNullOrWhiteSpace(mountNode?.MountModelPath)
            ? null
            : mountNode.MountModelPath.Replace('/', '\\');
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

        if (result.KnownMapTiles.Count > 0)
            _pm4KnownMapTiles.UnionWith(result.KnownMapTiles);

        if (result.CacheData != null)
        {
            bool replaceExisting = !_pm4LoadedCameraWindow.HasValue || _pm4TileObjects.Count == 0;
            if (replaceExisting)
                ClearPm4OverlayRuntimeState();

            MergePm4OverlayFromCache(result.CacheData);
            if (result.CoveredMapTiles.Count > 0)
                _pm4CoveredMapTiles.UnionWith(result.CoveredMapTiles);
            if (result.LoadedCameraWindow.HasValue)
                ExpandPm4LoadedCameraWindow(result.LoadedCameraWindow.Value);
            RestoreSelectedPm4Object(result.SelectedObjectKey);
            UpdatePm4AdaptiveWindow(result.LoadElapsedMs);
        }

        _pm4Status = result.StatusMessage;
        ViewerLog.Important(ViewerLog.Category.Terrain, "[PM4] " + _pm4Status);
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
        string? currentPath,
        bool emitLog)
    {
        if (requestId != _pm4LoadRequestId)
            return;

        string currentFileSuffix = string.IsNullOrWhiteSpace(currentPath)
            ? string.Empty
            : $", file={Path.GetFileName(currentPath)}";
        string status =
            $"PM4 loading: {phase} {processedFiles}/{totalFiles} files, loaded={loadedFiles}, objects={objectCount}, lines={lineCount}, tris={triangleCount}, readFail={readFailed}, decodeFail={decodeFailed}, zero={zeroObjectFiles}{currentFileSuffix}";
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
                _pm4SplitCk24ByMdos,
                _pm4SplitCk24ByConnectivity);
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

            _pm4Status = $"PM4 loading: cache miss, decoding {loadCandidates.Count} camera-window files...";

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
                        ReportPm4LoadProgress(requestId, "reading", processedFiles, loadCandidates.Count, loadedFiles, objectCount, lineCount, triangleCount, readFailed, decodeFailed, zeroObjectFiles, pm4Path, emitLog);
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

                try
                {
                    Pm4File pm4 = CorePm4DocumentReader.Read(bytes, pm4Path);
                    int rejectedLongEdges = 0;
                    List<Pm4OverlayObject> objects = BuildPm4TileObjects(
                        pm4,
                        pm4Path,
                        effectiveTileX,
                        effectiveTileY,
                        _pm4SplitCk24ByMdos,
                        _pm4SplitCk24ByConnectivity,
                        ref remainingLineBudget,
                        ref remainingTriangleBudget,
                        ref rejectedLongEdges);
                    if (objects.Count == 0)
                    {
                        tileSatisfiedCounts[(effectiveTileX, effectiveTileY)]++;
                        zeroObjectFiles++;
                        ViewerLog.Debug(ViewerLog.Category.Terrain,
                            $"[PM4] Parsed '{pm4Path}' (version={pm4.Version}, surfaces={pm4.KnownChunks.Msur.Count}, meshVerts={pm4.KnownChunks.Msvt.Count}, meshIndices={pm4.KnownChunks.Msvi.Count}, links={pm4.KnownChunks.Mslk.Count}, refs={pm4.KnownChunks.Mprl.Count}) but produced 0 overlay objects.");
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
                    ReportPm4LoadProgress(requestId, "decoding", processedFiles, loadCandidates.Count, loadedFiles, objectCount, lineCount, triangleCount, readFailed, decodeFailed, zeroObjectFiles, pm4Path, emitLog);
                    lastStatusReportMs = elapsedMs;
                    if (emitLog)
                        lastLogReportMs = elapsedMs;
                }
            }

            if (loadedFiles == 0)
            {
                return new Pm4OverlayAsyncLoadResult(
                    requestId,
                    null,
                    cameraWindow,
                    knownMapTiles,
                    [],
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

            HashSet<(int tileX, int tileY)> coveredMapTiles = tileCandidateCounts
                .Where(entry => tileSatisfiedCounts[entry.Key] >= entry.Value)
                .Select(static entry => entry.Key)
                .ToHashSet();

            return new Pm4OverlayAsyncLoadResult(
                requestId,
                cacheData,
                cameraWindow,
                knownMapTiles,
                coveredMapTiles,
                selectedObjectKey,
                loadStopwatch.Elapsed.TotalMilliseconds,
                $"PM4 ready: {loadedFiles}/{loadCandidates.Count} camera-window files decoded and cached for ({cameraWindow.minTileX}..{cameraWindow.maxTileX}, {cameraWindow.minTileY}..{cameraWindow.maxTileY}), avg {_pm4AverageLoadMs:0} ms, next radius {_pm4CameraTileRadius}, from {mapPm4CandidateCount} map files, {objectCount} objects, {lineCount} lines, {triangleCount} triangles, {positionRefCount} refs, {rejectedLongEdgesTotal} long edges rejected, {loadStopwatch.ElapsedMilliseconds} ms.",
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
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ResearchBySourcePath.Clear();
        _pm4ResearchUnavailablePaths.Clear();
        _pm4ObjectLookup.Clear();
        _pm4MergedObjectGroupKeys.Clear();
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
                    obj.DominantMdosIndex,
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
                    obj.DominantMdosIndex,
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
                    cachedObject.DominantMdosIndex,
                    cachedObject.AverageSurfaceHeight,
                    cachedObject.PlacementAnchor,
                    cachedObject.BaseRotationRadians,
                    cachedObject.PlanarTransform,
                    cachedObject.BoundsMin,
                    cachedObject.BoundsMax,
                    cachedObject.ConnectorKeys.ToList());
                objects.Add(restored);
                _pm4ObjectLookup[(tile.TileX, tile.TileY, restored.Ck24, restored.ObjectPartId)] = restored;
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

        RebuildPm4MergedObjectGroups();
        RebuildPm4ObjectGroupBounds();
        RebuildPm4TileCk24Bounds();
    }

    private void RecalculatePm4OverlayRuntimeTotals()
    {
        _pm4LoadedFiles = _pm4TileObjects.Count;
        _pm4ObjectCount = 0;
        _pm4LineCount = 0;
        _pm4TriangleCount = 0;
        _pm4PositionRefCount = 0;
        _pm4RejectedLongEdges = 0;
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
                obj.DominantMdosIndex,
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
        bool splitCk24ByMdos,
        bool splitCk24ByConnectivity,
        ref int remainingLineBudget,
        ref int remainingTriangleBudget,
        ref int rejectedLongEdges)
    {
        var objects = new List<Pm4OverlayObject>();
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<MprlEntry> positionRefs = pm4.KnownChunks.Mprl;

        if (remainingLineBudget <= 0 || meshVertices.Count == 0)
            return objects;

        List<Pm4OverlaySeedGroup> seedGroups = BuildPm4OverlaySeedGroups(pm4);
        if (seedGroups.Count == 0)
            return objects;

        Pm4AxisConvention fileAxisConvention = DetectPm4AxisConvention(pm4);
        bool fallbackTileLocalCoordinates = IsLikelyTileLocal(meshVertices);
        int tileLineBudget = Math.Min(Pm4MaxLinesPerTile, remainingLineBudget);
        int tileTriangleBudget = Math.Min(Pm4MaxTrianglesPerTile, remainingTriangleBudget);

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
                bool allowNestedSeedSplits = !seedGroup.RequiresConnectivitySeedSplit;
                List<List<MsurEntry>> anchorGroups = splitCk24ByMdos && allowNestedSeedSplits
                    ? SplitSurfaceGroupByMdos(linkedSurfaces)
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
                        List<Pm4Triangle> triangles = tileTriangleBudget > 0
                            ? BuildCk24ObjectTriangles(pm4, component, tileX, tileY, linkedUseTileLocalCoordinates, ck24AxisConvention, linkedPlanarTransform, linkedWorldPivot, linkedWorldYawCorrection, tileTriangleBudget)
                            : new List<Pm4Triangle>();

                        if (lines.Count == 0 && triangles.Count == 0)
                            continue;

                        byte dominantGroupKey = SelectDominantSurfaceValue(component, static surface => surface.GroupKey);
                        byte dominantAttributeMask = SelectDominantSurfaceValue(component, static surface => surface.AttributeMask);
                        uint dominantMdosIndex = SelectDominantSurfaceValue(component, static surface => surface.MdosIndex);
                        float averageSurfaceHeight = component.Count > 0 ? component.Average(static surface => surface.Height) : 0f;
                        int totalIndexCount = component.Sum(static surface => surface.IndexCount);

                        objects.Add(new Pm4OverlayObject(
                            sourcePath,
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
                            dominantMdosIndex,
                            averageSurfaceHeight,
                            linkedPlacementAnchor,
                            linkedRendererFrameRotationRadians,
                            linkedPlanarTransform,
                            linkedConnectorKeys));

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
        return objects;
    }

    private static CorePm4CoordinateModeResolution ResolveCk24CoordinateModeResolution(
        Pm4File pm4,
        IReadOnlyList<MsurEntry> surfaces,
        IReadOnlyList<MprlEntry> anchorPositionRefs,
        int tileX,
        int tileY,
        Pm4AxisConvention axisConvention,
        bool fallbackTileLocalCoordinates)
    {
        return CorePm4PlacementMath.ResolveCoordinateMode(
            pm4.KnownChunks.Msvt,
            pm4.KnownChunks.Msvi,
            ConvertToCorePm4Surfaces(surfaces),
            ConvertToCorePm4PositionRefs(pm4.KnownChunks.Mprl),
            anchorPositionRefs.Count > 0 ? ConvertToCorePm4PositionRefs(anchorPositionRefs) : null,
            tileX,
            tileY,
            ToCoreAxisConvention(axisConvention),
            ToCoreCoordinateMode(fallbackTileLocalCoordinates));
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
        int triangleBudget)
    {
        IReadOnlyList<Vector3> meshVertices = pm4.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = pm4.KnownChunks.Msvi;
        var triangles = new List<Pm4Triangle>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            MsurEntry surface = surfaces[s];
            if (triangles.Count >= triangleBudget)
                break;

            int firstIndex = (int)surface.MsviFirstIndex;
            int surfaceIndexCount = surface.IndexCount;
            if (surfaceIndexCount < 3 || firstIndex < 0 || firstIndex >= meshIndices.Count)
                continue;

            int endExclusive = Math.Min(firstIndex + surfaceIndexCount, meshIndices.Count);
            int indexCount = endExclusive - firstIndex;
            if (indexCount < 3)
                continue;

            // Most PM4 surfaces are listed as loops; use a fan from the first vertex.
            int i0 = (int)meshIndices[firstIndex];
            if ((uint)i0 >= (uint)meshVertices.Count)
                continue;

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

    private static List<List<MsurEntry>> SplitSurfaceGroupByMdos(IReadOnlyList<MsurEntry> surfaces)
    {
        if (surfaces.Count <= 1)
            return new List<List<MsurEntry>> { surfaces.ToList() };

        var groups = surfaces
            .GroupBy(static surface => surface.MdosIndex)
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
        return CorePm4PlacementMath.ResolvePlacementSolution(
            pm4.KnownChunks.Msvt,
            pm4.KnownChunks.Msvi,
            ConvertToCorePm4Surfaces(surfaceList),
            ConvertToCorePm4PositionRefs(pm4.KnownChunks.Mprl),
            anchorPositionRefs is not null ? ConvertToCorePm4PositionRefs(anchorPositionRefs) : null,
            tileX,
            tileY,
            ToCoreCoordinateMode(useTileLocalCoordinates),
            ToCoreAxisConvention(axisConvention));
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
        _taxiActorClockInitialized = false;
    }

    private void UpdateTaxiActorInstances()
    {
        _taxiActorInstances.Clear();

        bool hasTaxiSelection = _selectedTaxiNodeId >= 0 || _selectedTaxiRouteId >= 0;
        if (!_showTaxi || !_showTaxiActors || _taxiLoader == null || !hasTaxiSelection)
        {
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

            string? actorModelPath = GetTaxiActorModelOverride(route.PathId);
            float scale = 1.0f;

            if (string.IsNullOrWhiteSpace(actorModelPath))
            {
                TaxiPathLoader.TaxiNode? mountNode = ResolveTaxiActorNode(route);
                if (mountNode == null || string.IsNullOrWhiteSpace(mountNode.MountModelPath))
                    continue;

                actorModelPath = mountNode.MountModelPath;
                scale = mountNode.MountScale > 0.01f ? mountNode.MountScale : 1.0f;
            }

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

            float yawRadians = actorDirection.LengthSquared() > 0.0001f
                ? MathF.Atan2(actorDirection.X, actorDirection.Y) + MathF.PI
                : 0f;
            string modelPath = actorModelPath.Replace('/', '\\');
            string key = WorldAssetManager.NormalizeKey(modelPath);
            _assets.QueueMdxLoad(key);

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

    private void LazyLoadAreaTriggers()
    {
        _areaTriggerLoadAttempted = true;
        if (_dbcProvider == null || _dbdDir == null || _dbcBuild == null || _mapId < 0) return;
        _areaTriggerLoader = new AreaTriggerLoader();
        _areaTriggerLoader.Load(_dbcProvider, _dbdDir, _dbcBuild, _mapId);
    }

    /// <summary>
    /// Load Light.dbc and LightData.dbc for zone-based lighting.
    /// </summary>
    public void LoadLighting(DBCD.Providers.IDBCProvider dbcProvider, string dbdDir, string build, int mapId)
    {
        _lightService = new LightService();
        _lightService.Load(dbcProvider, dbdDir, build, mapId);
    }

    public WorldScene(GL gl, string wdtPath, IDataSource? dataSource,
        ReplaceableTextureResolver? texResolver = null,
        string? buildVersion = null,
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
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
        Action<string>? onStatus = null)
    {
        _gl = gl;
        _dataSource = dataSource;
        _dbcBuild = buildVersion;
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

        if (adapter.IsWmoBased && adapter.ModfPlacements.Count > 0)
        {
            // WMO-only maps: pre-load placements + models
            var manifest = _assets.BuildManifest(
                adapter.MdxModelNames, adapter.WmoModelNames,
                adapter.MddfPlacements, adapter.ModfPlacements);
            _assets.LoadManifest(manifest);
            BuildInstances(adapter);

            var p = adapter.ModfPlacements[0];
            var bbCenter = (p.BoundsMin + p.BoundsMax) * 0.5f;
            var bbExtent = p.BoundsMax - p.BoundsMin;
            float dist = MathF.Max(bbExtent.Length() * 0.5f, 100f);
            _wmoCameraOverride = bbCenter + new Vector3(dist, 0, bbExtent.Z * 0.3f);
            ViewerLog.Info(ViewerLog.Category.Terrain, $"WMO-only map, camera at BB center: ({bbCenter.X:F1}, {bbCenter.Y:F1}, {bbCenter.Z:F1}), dist={dist:F0}");

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
                _wdlTerrain = new WdlTerrainRenderer(_gl);
                if (!_wdlTerrain.Load(_dataSource, _terrainManager.MapName))
                {
                    _wdlTerrain.Dispose();
                    _wdlTerrain = null;
                }
            }

            _terrainManager.OnTileLoaded += OnTileLoaded;
            _terrainManager.OnTileUnloaded += OnTileUnloaded;

            // Hide WDL for all ADT-backed tiles in the map so WDL only fills gaps
            // where no detailed ADT tile exists (developer/empty tiles).
            if (_wdlTerrain != null)
            {
                foreach (int tileIdx in adapter.ExistingTiles)
                {
                    int tx = tileIdx / 64;
                    int ty = tileIdx % 64;
                    _wdlTerrain.HideTile(tx, ty);
                }
            }
            onStatus?.Invoke("World loaded (tiles stream as you move).");
        }
        
        // Auto-load WL liquids if enabled
        if (_showWlLiquids && !_wlLoadAttempted)
        {
            LazyLoadWlLiquids();
        }
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
                UniqueId = p.UniqueId
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

            // Get MOHD local bounds from the loaded WMO model and transform to world space.
            // The WMO is a container — its internal geometry has its own local bounding box
            // (MOHD bounds) around the WMO's local origin. We transform that through the
            // placement matrix to get the correct world-space AABB.
            // Falls back to MODF file bounds if model isn't loaded.
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
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
                UniqueId = p.UniqueId
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
        foreach (var p in result.MddfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= mdxNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(mdxNames[p.NameIndex]);
            _assets.QueueMdxLoad(key);
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
                UniqueId = p.UniqueId
            };

            if (IsSkyboxModelPath(modelPath))
                tileSkyboxes.Add(instance);
            else
                tileMdx.Add(instance);
        }

        // Build WMO instances for this tile
        var tileWmo = new List<ObjectInstance>();
        foreach (var p in result.ModfPlacements)
        {
            if (p.NameIndex < 0 || p.NameIndex >= wmoNames.Count) continue;
            string key = WorldAssetManager.NormalizeKey(wmoNames[p.NameIndex]);
            _assets.QueueWmoLoad(key);
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

            // Get MOHD local bounds and transform to world space
            Vector3 localMin, localMax, worldMin, worldMax;
            if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
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
                UniqueId = p.UniqueId
            });
        }

        _tileMdxInstances[(tileX, tileY)] = tileMdx;
        _tileSkyboxInstances[(tileX, tileY)] = tileSkyboxes;
        _tileWmoInstances[(tileX, tileY)] = tileWmo;
        _instancesDirty = true;

        // Hide WDL low-res tile now that detailed ADT is loaded
        _wdlTerrain?.HideTile(tileX, tileY);

        if (tileMdx.Count > 0 || tileSkyboxes.Count > 0 || tileWmo.Count > 0)
            ViewerLog.Info(ViewerLog.Category.Terrain, $"Tile ({tileX},{tileY}) loaded: {tileMdx.Count} MDX, {tileSkyboxes.Count} skybox, {tileWmo.Count} WMO instances");
    }

    /// <summary>
    /// Called by TerrainManager when a tile leaves the AOI.
    /// </summary>
    private void OnTileUnloaded(int tileX, int tileY)
    {
        _tileMdxInstances.Remove((tileX, tileY));
        _tileSkyboxInstances.Remove((tileX, tileY));
        _tileWmoInstances.Remove((tileX, tileY));
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

        _instancesDirty = false;
    }

    private MdxRenderer? TryGetQueuedMdx(string modelKey)
    {
        if (_assets.TryGetLoadedMdx(modelKey, out var renderer))
            return renderer;

        _assets.PrioritizeMdxLoad(modelKey);
        return null;
    }

    private WmoRenderer? TryGetQueuedWmo(string modelKey)
    {
        if (_assets.TryGetLoadedWmo(modelKey, out var renderer))
            return renderer;

        _assets.PrioritizeWmoLoad(modelKey);
        return null;
    }

    private void ProcessDeferredAssetLoads()
    {
        int processed = _assets.ProcessPendingLoads(maxLoads: 6, maxBudgetMs: 4.0);
        if (processed <= 0)
            return;

        bool boundsChanged = false;
        foreach (var list in _tileMdxInstances.Values)
            boundsChanged |= RefreshMdxInstanceBounds(list);
        foreach (var list in _tileSkyboxInstances.Values)
            boundsChanged |= RefreshMdxInstanceBounds(list);
        foreach (var list in _tileWmoInstances.Values)
            boundsChanged |= RefreshWmoInstanceBounds(list);

        boundsChanged |= RefreshMdxInstanceBounds(_externalMdxInstances);
        boundsChanged |= RefreshMdxInstanceBounds(_externalSkyboxInstances);
        boundsChanged |= RefreshWmoInstanceBounds(_externalWmoInstances);

        if (boundsChanged)
        {
            _instancesDirty = true;
            RebuildInstanceLists();
        }
    }

    private bool RefreshMdxInstanceBounds(List<ObjectInstance> instances)
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
            changed = true;
        }

        return changed;
    }

    private bool RefreshWmoInstanceBounds(List<ObjectInstance> instances)
    {
        bool changed = false;

        for (int i = 0; i < instances.Count; i++)
        {
            var inst = instances[i];
            if (inst.BoundsResolved)
                continue;

            if (!_assets.TryGetWmoBounds(inst.ModelKey, out var localMin, out var localMax))
                continue;

            TransformBounds(localMin, localMax, inst.Transform, out var worldMin, out var worldMax);
            inst.LocalBoundsMin = localMin;
            inst.LocalBoundsMax = localMax;
            inst.BoundsMin = worldMin;
            inst.BoundsMax = worldMax;
            inst.BoundsResolved = true;
            instances[i] = inst;
            changed = true;
        }

        return changed;
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

            if (isWmo)
            {
                _assets.QueueWmoLoad(key);

                var transform = Matrix4x4.CreateRotationZ(finalYawRadians)
                    * Matrix4x4.CreateTranslation(pos);

                Vector3 localMin, localMax, worldMin, worldMax;
                if (_assets.TryGetWmoBounds(key, out localMin, out localMax))
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
                    UniqueId = spawn.SpawnId
                });
            }
            else
            {
                _assets.QueueMdxLoad(key);

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
                    UniqueId = spawn.SpawnId
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

    private static float ComputeWmoCullDistance(float fogEnd)
    {
        return MathF.Max(WmoCullDistance, fogEnd + 1200f);
    }

    private void CollectVisibleMdxInstances(
        List<ObjectInstance> instances,
        Vector3 cameraPos,
        float mdxFadeStart,
        float mdxFadeStartSq,
        float mdxFadeRange,
        bool cullSmallDoodadsOnly)
    {
        foreach (var inst in instances)
        {
            float boundsDistSq = DistanceSquaredPointToAabb(cameraPos, inst.BoundsMin, inst.BoundsMax);
            float noCullDistanceSq = ComputeNoCullDistanceSq(inst.BoundsMin, inst.BoundsMax);
            if (boundsDistSq > noCullDistanceSq && !_frustumCuller.TestAABB(inst.BoundsMin, inst.BoundsMax))
            {
                MdxCulledCount++;
                continue;
            }

            float diag = (inst.BoundsMax - inst.BoundsMin).Length();
            bool useSmallDoodadCull = cullSmallDoodadsOnly && diag < DoodadSmallThreshold;
            if (useSmallDoodadCull && boundsDistSq > DoodadCullDistanceSq)
            {
                MdxCulledCount++;
                continue;
            }

            var renderer = TryGetQueuedMdx(inst.ModelKey);
            if (renderer == null)
                continue;

            float opaqueFade = 1.0f;
            if ((!cullSmallDoodadsOnly || useSmallDoodadCull) && boundsDistSq > mdxFadeStartSq)
            {
                float boundsDist = MathF.Sqrt(boundsDistSq);
                opaqueFade = MathF.Max(0f, 1.0f - (boundsDist - mdxFadeStart) / mdxFadeRange);
            }

            float centerDistanceSq = Vector3.DistanceSquared(cameraPos, inst.Transform.Translation);
            float transparentFade = 1.0f;
            if ((!cullSmallDoodadsOnly || useSmallDoodadCull) && centerDistanceSq > mdxFadeStartSq)
            {
                float centerDistance = MathF.Sqrt(centerDistanceSq);
                transparentFade = MathF.Max(0f, 1.0f - (centerDistance - mdxFadeStart) / mdxFadeRange);
            }

            _visibleMdxInstances.Add(new VisibleMdxInstance(inst, renderer, centerDistanceSq, opaqueFade, transparentFade));
        }
    }

    // ── ISceneRenderer ──────────────────────────────────────────────────

    private bool _renderDiagPrinted = false;
    public void Render(Matrix4x4 view, Matrix4x4 proj)
    {
        TryFinalizePm4OverlayLoad();

        // Rebuild flat instance lists if tiles changed
        if (_instancesDirty)
            RebuildInstanceLists();

        ProcessDeferredAssetLoads();
    UpdateTaxiActorInstances();

        // Extract camera position for sky dome
        Matrix4x4.Invert(view, out var viewInvSky);
        var camPos = new Vector3(viewInvSky.M41, viewInvSky.M42, viewInvSky.M43);
        var lighting = _terrainManager.Lighting;
        Vector3 fogColor;
        float fogStart;
        float fogEnd;

        // 0. Render sky dome (before terrain, no depth write)
        // Update DBC lighting early so sky colors are available
        _lightService?.Update(camPos);
        if (_lightService != null && _lightService.ActiveLightId >= 0)
        {
            // Override sky dome colors from DBC Light data
            _skyDome.ZenithColor = _lightService.SkyTopColor;
            _skyDome.HorizonColor = _lightService.FogColor;
            _skyDome.SkyFogColor = _lightService.FogColor;
            fogColor = _lightService.FogColor;
            fogEnd = _lightService.FogEnd > 10f ? _lightService.FogEnd : lighting.FogEnd;
            fogStart = fogEnd * 0.25f;
        }
        else
        {
            _skyDome.UpdateFromLighting(_terrainManager.Lighting.GameTime);
            fogColor = lighting.FogColor;
            fogStart = lighting.FogStart;
            fogEnd = lighting.FogEnd;
        }
        _skyDome.Render(view, proj, camPos);

        var (objectFogStart, objectFogEnd) = ComputeObjectFogRange(fogStart, fogEnd, _objectFogEnabled);

        // Also set clear color to horizon color so any gaps match the sky
        _gl.ClearColor(_skyDome.HorizonColor.X, _skyDome.HorizonColor.Y, _skyDome.HorizonColor.Z, 1f);

        RenderSkyboxBackdrop(view, proj, camPos, fogColor, fogStart, fogEnd, lighting);

        // 0. Render WDL low-res terrain (far background — hidden tiles replaced by detailed ADTs)
        if (ShowWdlTerrain && _wdlTerrain != null)
            _wdlTerrain.Render(view, proj, camPos, _terrainManager.Lighting, _frustumCuller);

        // 1. Render terrain (with frustum culling)
        _terrainManager.Render(view, proj, camPos, _frustumCuller);

        // Reset GL state after terrain
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.DepthMask(true);
        _gl.Disable(EnableCap.Blend);
        _gl.Enable(EnableCap.DepthTest);
        _gl.UseProgram(0); // unbind terrain shader

        if (!_objectsVisible) return;

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
        var cameraPos = new Vector3(viewInv.M41, viewInv.M42, viewInv.M43);
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

        // 2a. WMO opaque pass (with frustum + distance culling + fade)
        WmoRenderedCount = 0;
        WmoCulledCount = 0;
        if (_wmosVisible)
        {
            float wmoCullDistance = ComputeWmoCullDistance(fogEnd);
            float wmoCullDistanceSq = wmoCullDistance * wmoCullDistance;
            float wmoFadeStart = wmoCullDistance * WmoFadeStartFraction;
            float wmoFadeStartSq = wmoFadeStart * wmoFadeStart;
            float wmoFadeRange = wmoCullDistance - wmoFadeStart;

            // State is constant for this pass; set once to avoid per-instance churn.
            _gl.Disable(EnableCap.Blend);
            _gl.DepthMask(true);

            foreach (var inst in _wmoInstances)
            {
                float wmoDistSq = DistanceSquaredPointToAabb(cameraPos, inst.BoundsMin, inst.BoundsMax);
                float wmoNoCullDistanceSq = ComputeNoCullDistanceSq(inst.BoundsMin, inst.BoundsMax);
                // Skip frustum cull for nearby objects to prevent pop-in when turning
                if (wmoDistSq > wmoNoCullDistanceSq && !_frustumCuller.TestAABB(inst.BoundsMin, inst.BoundsMax))
                { WmoCulledCount++; continue; }
                // Distance cull + fade for WMOs
                if (wmoDistSq > wmoCullDistanceSq)
                { WmoCulledCount++; continue; }

                float wmoFade = 1.0f;
                if (wmoDistSq > wmoFadeStartSq)
                {
                    float wmoDist = MathF.Sqrt(wmoDistSq);
                    wmoFade = 1.0f - (wmoDist - wmoFadeStart) / wmoFadeRange;
                }

                var renderer = TryGetQueuedWmo(inst.ModelKey);
                if (renderer == null) continue;
                renderer.RenderWithTransform(inst.Transform, view, proj,
                    fogColor, objectFogStart, objectFogEnd, cameraPos,
                    lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                WmoRenderedCount++;
            }
            if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Wmo, $"WMO render: {WmoRenderedCount} drawn, {WmoCulledCount} culled");
        }

        // 3a. MDX opaque pass (with frustum + distance culling + fade)
        MdxRenderedCount = 0;
        MdxCulledCount = 0;
        float mdxFadeStart = DoodadCullDistance * FadeStartFraction;
        float mdxFadeStartSq = mdxFadeStart * mdxFadeStart;
        float mdxFadeRange = DoodadCullDistance - mdxFadeStart;

        // Advance animation once per unique MDX renderer before any render passes
        if (_doodadsVisible)
        {
            _updatedMdxRenderers.Clear();
            foreach (var inst in _mdxInstances)
            {
                if (_updatedMdxRenderers.Add(inst.ModelKey))
                {
                    _assets.TryGetLoadedMdx(inst.ModelKey, out var r);
                    r?.UpdateAnimation();
                }
            }

            foreach (var inst in _taxiActorInstances)
            {
                if (_updatedMdxRenderers.Add(inst.ModelKey))
                {
                    _assets.TryGetLoadedMdx(inst.ModelKey, out var r);
                    r?.UpdateAnimation();
                }
            }
        }

        _visibleMdxInstances.Clear();
        MdxRenderer? batchRenderer = null;
        if (_doodadsVisible)
        {
            CollectVisibleMdxInstances(_mdxInstances, cameraPos, mdxFadeStart, mdxFadeStartSq, mdxFadeRange, cullSmallDoodadsOnly: true);
            CollectVisibleMdxInstances(_taxiActorInstances, cameraPos, mdxFadeStart, mdxFadeStartSq, mdxFadeRange, cullSmallDoodadsOnly: false);

            if (_visibleMdxInstances.Count > 0)
            {
                batchRenderer = _visibleMdxInstances[0].Renderer;
                batchRenderer.BeginBatch(view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                    lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
            }

            foreach (var visible in _visibleMdxInstances)
            {
                if (visible.Renderer.RequiresUnbatchedWorldRender)
                {
                    visible.Renderer.RenderWithTransform(visible.Instance.Transform, view, proj, RenderPass.Opaque, visible.OpaqueFade,
                        fogColor, objectFogStart, objectFogEnd, cameraPos,
                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                }
                else
                {
                    visible.Renderer.RenderInstance(visible.Instance.Transform, RenderPass.Opaque, visible.OpaqueFade);
                }

                MdxRenderedCount++;
            }

            if (!_renderDiagPrinted) ViewerLog.Info(ViewerLog.Category.Mdx, $"MDX opaque: {MdxRenderedCount} drawn, {MdxCulledCount} culled");
        }

        // ── PASS 2: LIQUID ──────────────────────────────────────────────
        // Render liquid after opaque geometry has established the depth buffer,
        // but before transparent MDX layers so reflective/translucent model
        // surfaces are composited on top instead of being overpainted by water.
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _terrainManager.RenderLiquid(view, proj, cameraPos);

        // ── PASS 3: TRANSPARENT (back-to-front, frustum-culled) ─────────
        // Render transparent/blended layers sorted by distance to camera.
        // Depth test ON but depth write OFF so transparent objects don't
        // occlude each other incorrectly.
        if (_doodadsVisible)
        {
            batchRenderer?.BeginBatch(view, proj, fogColor, objectFogStart, objectFogEnd, cameraPos,
                lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);

            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Lequal);

            // Sort visible instances back-to-front by distance to camera.
            _transparentSortScratch.Clear();
            for (int i = 0; i < _visibleMdxInstances.Count; i++)
            {
                _transparentSortScratch.Add((i, _visibleMdxInstances[i].CenterDistanceSq));
            }

            _transparentSortScratch.Sort((a, b) => b.distSq.CompareTo(a.distSq));

            foreach (var (visibleIdx, _) in _transparentSortScratch)
            {
                var visible = _visibleMdxInstances[visibleIdx];
                if (visible.Renderer.RequiresUnbatchedWorldRender)
                {
                    visible.Renderer.RenderWithTransform(visible.Instance.Transform, view, proj, RenderPass.Transparent, visible.TransparentFade,
                        fogColor, objectFogStart, objectFogEnd, cameraPos,
                        lighting.LightDirection, lighting.LightColor, lighting.AmbientColor);
                }
                else
                {
                    visible.Renderer.RenderInstance(visible.Instance.Transform, RenderPass.Transparent, visible.TransparentFade);
                }
            }
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }
        else
        {
            if (!_renderDiagPrinted) _renderDiagPrinted = true;
        }

        if (_wireframeRevealEnabled)
            RenderWireframeReveal(view, proj, cameraPos, fogColor, fogStart, fogEnd, lighting);

        // Reset GL state before bounding boxes
        _gl.Disable(EnableCap.Blend);
        _gl.DepthMask(true);
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthFunc(DepthFunction.Lequal);
        _gl.UseProgram(0);
        _gl.BindVertexArray(0);

        // 4. Debug bounding boxes for all placements
        if ((_showBoundingBoxes || _showPm4ObjectBounds) && _bbRenderer != null)
        {
            // Depth test ON so boxes behind terrain/objects are hidden,
            // depth write OFF so box lines don't occlude models
            _gl.Enable(EnableCap.DepthTest);
            _gl.DepthFunc(DepthFunction.Lequal);
            _gl.DepthMask(false);

            if (_showBoundingBoxes)
            {
                var adapter = _terrainManager.Adapter;
                if (!_renderDiagPrinted)
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"BB render: {adapter.MddfPlacements.Count} MDDF + {adapter.ModfPlacements.Count} MODF markers");

                // Draw selected object highlight first (thicker visual via slightly larger box)
                if (SelectedInstance is ObjectInstance sel)
                {
                    _bbRenderer.DrawBoxMinMax(sel.BoundsMin, sel.BoundsMax, view, proj, new Vector3(1f, 1f, 1f)); // white highlight
                }

                // MDDF bounding boxes (magenta)
                foreach (var inst in _mdxInstances)
                    _bbRenderer.DrawBoxMinMax(inst.BoundsMin, inst.BoundsMax, view, proj, new Vector3(1f, 0f, 1f));
                // MODF bounding boxes (cyan)
                foreach (var inst in _wmoInstances)
                    _bbRenderer.DrawBoxMinMax(inst.BoundsMin, inst.BoundsMax, view, proj, new Vector3(0f, 1f, 1f));
            }

            if (_showPm4ObjectBounds && _showPm4Overlay && _pm4TileObjects.Count > 0)
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

                        Vector3 boxColor = GetPm4ObjectColor(tileKey, obj) * 0.75f + new Vector3(0.20f, 0.20f, 0.20f);
                        if (_highlightedPm4ObjectKeys.Contains(objectKey))
                            boxColor = new Vector3(0.20f, 1.00f, 0.95f);
                        if (_selectedPm4ObjectGroupKey.HasValue
                            && IsPm4ObjectInGroup(_selectedPm4ObjectGroupKey.Value, objectKey))
                            boxColor = new Vector3(1.0f, 0.9f, 0.2f);
                        if (_selectedPm4ObjectKey.HasValue && _selectedPm4ObjectKey.Value == objectKey)
                            boxColor = new Vector3(1.0f, 1.0f, 1.0f);

                        _bbRenderer.DrawBoxMinMax(boundsMin, boundsMax, view, proj, boxColor);
                    }
                }
            }

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
                            _bbRenderer.BatchPin(marker, 16f, 3f, new Vector3(0.20f, 0.90f, 1.00f));
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
                            pm4Color = new Vector3(0.20f, 1.00f, 0.95f);
                        if (_selectedPm4ObjectGroupKey.HasValue
                            && IsPm4ObjectInGroup(_selectedPm4ObjectGroupKey.Value, objectKey))
                            pm4Color = new Vector3(1.0f, 1.0f, 0.2f);

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
                            _bbRenderer.BatchPin(transformedCenter, 22f, 4f, pm4Color);
                        }
                    }
                }
            }

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
                    _bbRenderer.FlushSolidBatch(view, proj);
                    _gl.Enable(EnableCap.CullFace);
                    _gl.Disable(EnableCap.Blend);
                }

                bool hasPm4LineGeometry = _pm4VisibleLineCount > 0
                    || _pm4VisiblePositionRefCount > 0
                    || (_showPm4ObjectCentroids && _pm4VisibleObjectCount > 0);
                if (hasPm4LineGeometry)
                {
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
                    _bbRenderer.FlushBatch(view, proj);
                }

                // Reset default state and clear PM4 primitives so other overlays use their normal pass.
                _gl.Enable(EnableCap.DepthTest);
                _gl.DepthFunc(DepthFunction.Lequal);
                _gl.DepthMask(true);
                _gl.Disable(EnableCap.Blend);

                _bbRenderer.BeginBatch();
                _bbRenderer.BeginSolidBatch();
            }

            // POI pin markers (magenta)
            if (_showPoi && _poiLoader != null && _poiLoader.Entries.Count > 0)
            {
                var poiColor = new Vector3(1f, 0f, 1f);
                foreach (var poi in _poiLoader.Entries)
                    _bbRenderer.BatchPin(poi.Position, 40f, 6f, poiColor);
            }

            // Taxi paths — filtered by selection
            if (_showTaxi && _taxiLoader != null)
            {
                var nodeColor = new Vector3(1f, 1f, 0f);
                var lineColor = new Vector3(0f, 1f, 1f);
                var routeHandleColor = new Vector3(1f, 0.65f, 0f);
                var selectedRouteColor = new Vector3(1f, 1f, 1f);
                int visibleRouteCount = _taxiLoader.Routes.Count(IsTaxiRouteVisible);
                bool showRouteHandles = _selectedTaxiNodeId >= 0 || _selectedTaxiRouteId >= 0 || visibleRouteCount <= 32;

                foreach (var node in _taxiLoader.Nodes)
                {
                    if (!IsTaxiNodeVisible(node)) continue;
                    _bbRenderer.BatchPin(node.Position, 50f, 8f, nodeColor);
                }

                foreach (var route in _taxiLoader.Routes)
                {
                    if (!IsTaxiRouteVisible(route)) continue;
                    Vector3 routeColor = route.PathId == _selectedTaxiRouteId ? selectedRouteColor : lineColor;
                    for (int i = 0; i < route.Waypoints.Count - 1; i++)
                        _bbRenderer.BatchLine(route.Waypoints[i], route.Waypoints[i + 1], routeColor);

                    if (showRouteHandles && TryGetTaxiRouteSelectionPoint(route, out Vector3 selectionPoint))
                    {
                        float pinHeight = route.PathId == _selectedTaxiRouteId ? 42f : 30f;
                        float headSize = route.PathId == _selectedTaxiRouteId ? 6f : 4f;
                        _bbRenderer.BatchPin(selectionPoint, pinHeight, headSize,
                            route.PathId == _selectedTaxiRouteId ? selectedRouteColor : routeHandleColor);
                    }
                }
            }

            // AreaTriggers (green wireframe shapes for portals and event markers)
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
                    }
                }
            }

            _bbRenderer.FlushBatch(view, proj);
        }
    }

    private void RenderSkyboxBackdrop(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
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
        if (string.IsNullOrWhiteSpace(modelPath))
            return false;

        string normalized = modelPath.Replace('\\', '/').ToLowerInvariant();
        if (!normalized.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) &&
            !normalized.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
            return false;

        if (normalized.Contains("skylight"))
            return false;

        return normalized.Contains("environments/stars/") ||
               normalized.Contains("/skybox/") ||
               normalized.Contains("skybox") ||
               normalized.Contains("skybowl");
    }

    public void ToggleWireframe()
    {
        _wireframeRevealEnabled = !_wireframeRevealEnabled;
        _terrainManager.ToggleWireframe();
        if (!_wireframeRevealEnabled)
            ClearWireframeReveal();
    }

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
        if (_showPm4Overlay)
        {
            if (TryBuildHoveredPm4Info(view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out HoveredAssetInfo hoveredPm4Info, out int hoveredPm4Count))
            {
                _hoveredAssetInfo = new HoveredAssetInfo(
                    hoveredPm4Info.AssetKind,
                    hoveredPm4Info.DisplayName,
                    hoveredPm4Info.SourcePath,
                    hoveredPm4Info.DetailLine,
                    hoveredPm4Info.WorldPosition,
                    Math.Max(0, hoveredPm4Count - 1),
                    hoveredPm4Info.Pm4ObjectKey);
                return;
            }
        }

        HoveredAssetInfo? bestInfo = null;
        float bestDistanceSq = float.MaxValue;
        float bestDepth = float.MaxValue;
        int hitCount = 0;

        void ConsiderCandidate(HoveredAssetInfo info, float distanceSq, float depth)
        {
            hitCount++;

            const float distanceEpsilon = 0.01f;
            if (!bestInfo.HasValue
                || distanceSq < bestDistanceSq - distanceEpsilon
                || (MathF.Abs(distanceSq - bestDistanceSq) <= distanceEpsilon && depth < bestDepth))
            {
                bestInfo = info;
                bestDistanceSq = distanceSq;
                bestDepth = depth;
            }
        }

        for (int i = 0; i < _wmoInstances.Count; i++)
        {
            ObjectInstance inst = _wmoInstances[i];
            if (!TryMeasureHoverInfoHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                continue;

            ConsiderCandidate(BuildHoveredObjectInfo("WMO", inst), distanceSq, depth);
        }

        for (int i = 0; i < _mdxInstances.Count; i++)
        {
            ObjectInstance inst = _mdxInstances[i];
            if (!TryMeasureHoverInfoHit(inst.BoundsMin, inst.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                continue;

            ConsiderCandidate(BuildHoveredObjectInfo("MDX", inst), distanceSq, depth);
        }

        if (_showWlLiquids && _wlLoader != null)
        {
            for (int i = 0; i < _wlLoader.Bodies.Count; i++)
            {
                WlLiquidBody body = _wlLoader.Bodies[i];
                if (!TryMeasureHoverInfoHit(body.BoundsMin, body.BoundsMax, view, proj, mouseViewportX, mouseViewportY, viewportWidth, viewportHeight, out float distanceSq, out float depth))
                    continue;

                ConsiderCandidate(BuildHoveredWlLiquidInfo(body), distanceSq, depth);
            }
        }

        if (bestInfo.HasValue)
        {
            HoveredAssetInfo info = bestInfo.Value;
            _hoveredAssetInfo = new HoveredAssetInfo(
                info.AssetKind,
                info.DisplayName,
                info.SourcePath,
                info.DetailLine,
                info.WorldPosition,
                Math.Max(0, hitCount - 1),
                info.Pm4ObjectKey);
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
            return;
        }

        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
    }

    public bool TryPickSceneObjectByRay(Vector3 rayOrigin, Vector3 rayDir, out ObjectType objectType, out int objectIndex, out float distance)
    {
        if (_instancesDirty)
            RebuildInstanceLists();

        float bestT = float.MaxValue;
        ObjectType bestType = ObjectType.None;
        int bestIndex = -1;

        var hits = new List<(string type, int index, float dist, string name)>();

        // Test WMO bounding boxes
        for (int i = 0; i < _wmoInstances.Count; i++)
        {
            // Slightly inflate AABBs to make selection more forgiving for thin geometry.
            Vector3 pad = new(2f, 2f, 2f);
            float t = RayAABBIntersect(rayOrigin, rayDir, _wmoInstances[i].BoundsMin - pad, _wmoInstances[i].BoundsMax + pad);
            if (t >= 0)
            {
                hits.Add(("WMO", i, t, _wmoInstances[i].ModelName));
                if (t < bestT) { bestT = t; bestType = ObjectType.Wmo; bestIndex = i; }
            }
        }

        // Test MDX bounding boxes
        for (int i = 0; i < _mdxInstances.Count; i++)
        {
            Vector3 pad = new(1f, 1f, 1f);
            float t = RayAABBIntersect(rayOrigin, rayDir, _mdxInstances[i].BoundsMin - pad, _mdxInstances[i].BoundsMax + pad);
            if (t >= 0)
            {
                hits.Add(("MDX", i, t, _mdxInstances[i].ModelName));
                if (t < bestT) { bestT = t; bestType = ObjectType.Mdx; bestIndex = i; }
            }
        }

        // Debug: log all hits sorted by distance
        if (hits.Count > 0)
        {
            var sorted = hits.OrderBy(h => h.dist).ToList();
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"[ObjectPick] Ray hit {hits.Count} objects:");
            foreach (var h in sorted.Take(5))
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"  {h.type}[{h.index}] {h.name} @ dist={h.dist:F1}");
            if (sorted.Count > 5)
                ViewerLog.Debug(ViewerLog.Category.Terrain, $"  ... and {sorted.Count - 5} more");
        }

        objectType = bestType;
        objectIndex = bestIndex;
        distance = bestT;
        return bestType != ObjectType.None && bestIndex >= 0;
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
        if (!_showPm4Overlay || _pm4TileObjects.Count == 0)
        {
            objectKey = null;
            objectGroupKey = null;
            distance = float.MaxValue;
            return false;
        }

        Matrix4x4 pm4Transform = BuildPm4OverlayTransformMatrix();
        bool applyPm4Transform = _pm4OverlayTranslation != Vector3.Zero
            || _pm4OverlayRotationDegrees.LengthSquared() > 0.0001f
            || _pm4OverlayScale != Vector3.One;

        float bestT = float.MaxValue;
        (int tileX, int tileY, uint ck24, int objectPart)? bestKey = null;
        (int tileX, int tileY, uint ck24)? bestGroupKey = null;

        foreach (var (tileKey, objects) in _pm4TileObjects)
        {
            if (!ShouldRenderPm4Tile(tileKey.tileX, tileKey.tileY))
                continue;

            foreach (Pm4OverlayObject obj in objects)
            {
                if (!ShouldRenderPm4ObjectType(obj.Ck24Type))
                    continue;

                var candidateKey = (tileKey.tileX, tileKey.tileY, obj.Ck24, obj.ObjectPartId);
                Matrix4x4 objectTransform = BuildPm4ObjectTransform(candidateKey, applyPm4Transform, pm4Transform, out bool applyObjectTransform);

                Vector3 boundsMin = obj.BoundsMin;
                Vector3 boundsMax = obj.BoundsMax;
                if (applyObjectTransform)
                    TransformBounds(boundsMin, boundsMax, objectTransform, out boundsMin, out boundsMax);

                Vector3 padding = new(2f, 2f, 2f);
                float t = RayAABBIntersect(rayOrigin, rayDir, boundsMin - padding, boundsMax + padding);
                if (t >= 0f && t < bestT)
                {
                    bestT = t;
                    bestKey = candidateKey;
                    bestGroupKey = ResolvePm4ObjectGroupKey(candidateKey);
                }
            }
        }

        objectKey = bestKey;
        objectGroupKey = bestGroupKey;
        distance = bestT;
        return bestKey.HasValue;
    }

    public void ClearSelection()
    {
        _selectedObjectType = ObjectType.None;
        _selectedObjectIndex = -1;
    }

    public void ClearPm4ObjectSelection()
    {
        _selectedPm4ObjectKey = null;
        _selectedPm4ObjectGroupKey = null;
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
            obj.SurfaceCount,
            obj.DominantGroupKey,
            obj.DominantAttributeMask,
            obj.DominantMdosIndex,
            obj.AverageSurfaceHeight,
            boundsMin,
            boundsMax,
            center,
            nearestPositionRefDistance,
            obj.PlanarTransform.SwapPlanarAxes,
            obj.PlanarTransform.InvertU,
            obj.PlanarTransform.InvertV,
            obj.PlanarTransform.InvertsWinding);

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
                hypothesis.MdosIndices.Count,
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
            allMatches.Take(8).ToList());

        return true;
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
                CorePm4ResearchHierarchyAnalyzer.Analyze(researchFile));
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

    private static HoveredAssetInfo BuildHoveredObjectInfo(string assetKind, in ObjectInstance inst)
    {
        return new HoveredAssetInfo(
            assetKind,
            inst.ModelName,
            inst.ModelPath,
            $"UniqueId: {inst.UniqueId}",
            inst.PlacementPosition,
            0,
            null);
    }

    private static HoveredAssetInfo BuildHoveredWlLiquidInfo(WlLiquidBody body)
    {
        Vector3 worldPosition = (body.BoundsMin + body.BoundsMax) * 0.5f;
        return new HoveredAssetInfo(
            "WL liquid",
            body.Name,
            body.SourcePath,
            $"{body.FileType} • {body.BlockCount} blocks",
            worldPosition,
            0,
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
        out int hitCount)
    {
        info = default;
        hitCount = 0;

        float bestDistanceSq = float.MaxValue;
        float bestDepth = float.MaxValue;
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

                hitCount++;
                const float distanceEpsilon = 0.01f;
                if (!bestInfo.HasValue
                    || distanceSq < bestDistanceSq - distanceEpsilon
                    || (MathF.Abs(distanceSq - bestDistanceSq) <= distanceEpsilon && depth < bestDepth))
                {
                    bestDistanceSq = distanceSq;
                    bestDepth = depth;
                    bestInfo = new HoveredAssetInfo(
                        "PM4",
                        $"CK24 0x{obj.Ck24:X6} part={obj.ObjectPartId}",
                        obj.SourcePath,
                        $"type=0x{obj.Ck24Type:X2} obj={obj.Ck24ObjectId} mslk=0x{obj.LinkGroupObjectId:X8} surfaces={obj.SurfaceCount}",
                        center,
                        0,
                        objectKey);
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
        objectMatch = BuildPm4ObjectMatchObject(pm4Object, placements, Math.Max(1, maxMatchesPerObject));
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

    private void RenderWireframeReveal(Matrix4x4 view, Matrix4x4 proj, Vector3 cameraPos,
        Vector3 fogColor, float fogStart, float fogEnd, TerrainLighting lighting)
    {
        if (_wireframeRevealWmoIndices.Count == 0 && _wireframeRevealMdxIndices.Count == 0)
            return;

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
        }

        _gl.DepthMask(true);
        _gl.DepthFunc(DepthFunction.Lequal);
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
            Pm4OverlayColorMode.GroupKey => ColorFromSeed(obj.DominantGroupKey),
            Pm4OverlayColorMode.AttributeMask => ColorFromSeed(obj.DominantAttributeMask),
            Pm4OverlayColorMode.Height => ColorFromHeight(obj.Center.Z),
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

                var groupObjects = new List<((int tileX, int tileY, uint ck24, int objectPart) key, Pm4OverlayObject obj)>();
                foreach (KeyValuePair<(int tileX, int tileY), List<Pm4OverlayObject>> tileEntry in _pm4TileObjects)
                {
                    List<Pm4OverlayObject> objects = tileEntry.Value;
                    for (int i = 0; i < objects.Count; i++)
                    {
                        Pm4OverlayObject candidate = objects[i];
                        var candidateKey = (tileEntry.Key.tileX, tileEntry.Key.tileY, candidate.Ck24, candidate.ObjectPartId);
                        if (ResolvePm4ObjectGroupKey(candidateKey) == selectedGroupKey)
                            groupObjects.Add((candidateKey, candidate));
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
                            .OrderBy(static entry => entry.obj.DominantMdosIndex)
                            .ThenBy(static entry => entry.key.objectPart)
                            .ThenBy(static entry => entry.key.tileX)
                            .ThenBy(static entry => entry.key.tileY)
                            .ToList();

                        List<Pm4SelectedObjectGraphMdosNode> mdosGroups = linkEntries
                            .GroupBy(static entry => entry.obj.DominantMdosIndex)
                            .OrderBy(static group => group.Key)
                            .Select(mdosGroup =>
                            {
                                var mdosEntries = mdosGroup
                                    .OrderBy(static entry => entry.key.objectPart)
                                    .ThenBy(static entry => entry.key.tileX)
                                    .ThenBy(static entry => entry.key.tileY)
                                    .ToList();

                                List<Pm4SelectedObjectGraphPartNode> parts = mdosEntries
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
                                        entry.obj.DominantMdosIndex,
                                        entry.key == selectedObjectKey))
                                    .ToList();

                                return new Pm4SelectedObjectGraphMdosNode(
                                    mdosGroup.Key,
                                    parts.Count,
                                    mdosEntries.Sum(static entry => entry.obj.SurfaceCount),
                                    mdosEntries.Sum(static entry => entry.obj.TotalIndexCount),
                                    mdosEntries.Select(static entry => entry.obj.DominantAttributeMask).Distinct().OrderBy(static value => value).ToList(),
                                    mdosEntries.Select(static entry => entry.obj.DominantGroupKey).Distinct().OrderBy(static value => value).ToList(),
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
                            mdosGroups.Select(static group => group.MdosIndex).ToList(),
                            linkEntries.Select(static entry => entry.obj.DominantAttributeMask).Distinct().OrderBy(static value => value).ToList(),
                            linkEntries.Select(static entry => entry.obj.DominantGroupKey).Distinct().OrderBy(static value => value).ToList(),
                            mdosGroups);
                    })
                    .ToList();

                info = new Pm4SelectedObjectGraphInfo(
                    selectedObjectKey.tileX,
                    selectedObjectKey.tileY,
                    selectedObject.Ck24,
                    selectedObject.Ck24Type,
                    selectedObject.Ck24ObjectId,
                    selectedObject.ObjectPartId,
                    _pm4SplitCk24ByMdos,
                    _pm4SplitCk24ByConnectivity,
                    groupObjects.Select(static entry => (entry.key.tileX, entry.key.tileY)).Distinct().Count(),
                    linkGroups.Count,
                    linkGroups.Sum(static group => group.MdosGroups.Count),
                    groupObjects.Count,
                    groupObjects.Sum(static entry => entry.obj.SurfaceCount),
                    groupObjects.Sum(static entry => entry.obj.TotalIndexCount),
                    groupObjects.Select(static entry => entry.obj.DominantAttributeMask).Distinct().Count(),
                    groupObjects.Select(static entry => entry.obj.DominantGroupKey).Distinct().Count(),
                    linkGroups);

                return true;
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

            private static uint GetPm4LegendValue(Pm4OverlayColorMode mode, Pm4OverlayObject obj)
            {
                return mode switch
                {
                    Pm4OverlayColorMode.Ck24ObjectId => obj.Ck24ObjectId,
                    Pm4OverlayColorMode.Ck24Key => obj.Ck24,
                    Pm4OverlayColorMode.GroupKey => obj.DominantGroupKey,
                    Pm4OverlayColorMode.AttributeMask => obj.DominantAttributeMask,
                    _ => obj.Ck24Type
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
                    Pm4OverlayColorMode.Ck24Type => $"CK24 type 0x{value:X2}",
                    Pm4OverlayColorMode.Ck24ObjectId => $"CK24 obj {value} (0x{value:X4})",
                    Pm4OverlayColorMode.Ck24Key => $"CK24 0x{value:X6}",
                    Pm4OverlayColorMode.GroupKey => $"GroupKey 0x{value:X2}",
                    Pm4OverlayColorMode.AttributeMask => $"AttrMask 0x{value:X2}",
                    _ => value.ToString(CultureInfo.InvariantCulture)
                };
            }

            private Vector3 GetPm4LegendColor(Pm4OverlayColorMode mode, uint value)
            {
                return mode switch
                {
                    Pm4OverlayColorMode.Ck24ObjectId => ColorFromSeed(value),
                    Pm4OverlayColorMode.Ck24Key => ColorFromSeed(value),
                    Pm4OverlayColorMode.GroupKey => ColorFromSeed(value),
                    Pm4OverlayColorMode.AttributeMask => ColorFromSeed(value),
                    _ => GetPm4TypeColor((byte)value)
                };
            }
    private Vector3 ColorFromHeight(float z)
    {
        float denom = _pm4MaxObjectZ - _pm4MinObjectZ;
        float t = denom > 0.001f ? Math.Clamp((z - _pm4MinObjectZ) / denom, 0f, 1f) : 0.5f;
        return Vector3.Lerp(new Vector3(0.15f, 0.45f, 1.00f), new Vector3(1.00f, 0.35f, 0.15f), t);
    }

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
        return ck24Type switch
        {
            0x40 => new Vector3(1.0f, 0.55f, 0.10f),
            0x80 => new Vector3(0.95f, 0.32f, 0.08f),
            _ => new Vector3(0.95f, 0.48f, 0.08f)
        };
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
                        obj.DominantMdosIndex,
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
        _externalMdxInstances.Clear();
        _externalSkyboxInstances.Clear();
        _externalWmoInstances.Clear();
        _pm4TileObjects.Clear();
        _pm4TileStats.Clear();
        _pm4TilePositionRefs.Clear();
        _pm4ResearchBySourcePath.Clear();
        _pm4ResearchUnavailablePaths.Clear();
        _pm4ObjectLookup.Clear();
        _highlightedPm4ObjectKeys.Clear();
        _pm4MergedObjectGroupKeys.Clear();
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
        uint dominantMdosIndex,
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
            dominantMdosIndex,
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
        uint dominantMdosIndex,
        float averageSurfaceHeight,
        Vector3 placementAnchor,
        float baseRotationRadians,
        Pm4PlanarTransform planarTransform,
        IReadOnlyList<Pm4ConnectorKey> connectorKeys)
        : this(
            sourcePath,
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
            dominantMdosIndex,
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
        uint dominantMdosIndex,
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
        DominantMdosIndex = dominantMdosIndex;
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
    public uint Ck24 { get; }
    public byte Ck24Type { get; }
    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);
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
    public uint DominantMdosIndex { get; }
    public float AverageSurfaceHeight { get; }
    public Pm4PlanarTransform PlanarTransform { get; }
    public IReadOnlyList<Pm4ConnectorKey> ConnectorKeys { get; }
    public Matrix4x4 BaseTransform { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
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
        CorePm4TileObjectHypothesisReport hypothesisReport)
    {
        SourcePath = sourcePath;
        Snapshot = snapshot;
        DecodeAudit = decodeAudit;
        HypothesisReport = hypothesisReport;
    }

    public string SourcePath { get; }
    public CorePm4ExplorationSnapshot Snapshot { get; }
    public CorePm4DecodeAuditReport DecodeAudit { get; }
    public CorePm4TileObjectHypothesisReport HypothesisReport { get; }
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
    uint DominantMdosIndex,
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
    uint DominantMdosIndex,
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
    float WorldFootprintArea)
{
    public Vector3 Center => (WorldBoundsMin + WorldBoundsMax) * 0.5f;

    public bool SameTile(int tileX, int tileY) => TileX == tileX && TileY == tileY;
}

internal readonly record struct Pm4PlacementMatchEvaluation(
    Pm4PlacementMatchState Placement,
    float AnchorPlanarGap,
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
public struct ObjectInstance
{
    public string ModelKey;
    public Matrix4x4 Transform;
    /// <summary>World-space AABB (local bounds transformed through placement matrix).</summary>
    public Vector3 BoundsMin;
    /// <summary>World-space AABB (local bounds transformed through placement matrix).</summary>
    public Vector3 BoundsMax;
    /// <summary>Model-local bounding box min (MOHD for WMO, model extents for MDX). Zero if unavailable.</summary>
    public Vector3 LocalBoundsMin;
    /// <summary>Model-local bounding box max (MOHD for WMO, model extents for MDX). Zero if unavailable.</summary>
    public Vector3 LocalBoundsMax;
    /// <summary>Display name (filename) for UI.</summary>
    public string ModelName;
    /// <summary>Renderer-space position from placement.</summary>
    public Vector3 PlacementPosition;
    /// <summary>Rotation in degrees from placement.</summary>
    public Vector3 PlacementRotation;
    /// <summary>Scale from placement (1.0 = default).</summary>
    public float PlacementScale;
    /// <summary>Full model path for diagnostics.</summary>
    public string ModelPath;
    /// <summary>UniqueId from MODF/MDDF placement (for dedup and display).</summary>
    public int UniqueId;
    /// <summary>True once bounds were derived from the loaded model instead of a temporary placement fallback.</summary>
    public bool BoundsResolved;
}

public enum ObjectType { None, Wmo, Mdx }

public readonly struct HoveredAssetInfo
{
    public HoveredAssetInfo(string assetKind, string displayName, string sourcePath, string detailLine, Vector3 worldPosition, int additionalHitCount, (int tileX, int tileY, uint ck24, int objectPart)? pm4ObjectKey)
    {
        AssetKind = assetKind ?? string.Empty;
        DisplayName = displayName ?? string.Empty;
        SourcePath = sourcePath ?? string.Empty;
        DetailLine = detailLine ?? string.Empty;
        WorldPosition = worldPosition;
        AdditionalHitCount = Math.Max(0, additionalHitCount);
        Pm4ObjectKey = pm4ObjectKey;
    }

    public string AssetKind { get; }
    public string DisplayName { get; }
    public string SourcePath { get; }
    public string DetailLine { get; }
    public Vector3 WorldPosition { get; }
    public int AdditionalHitCount { get; }
    public (int tileX, int tileY, uint ck24, int objectPart)? Pm4ObjectKey { get; }
}
