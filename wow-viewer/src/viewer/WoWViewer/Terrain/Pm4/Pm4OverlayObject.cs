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
        BaseTransform = Pm4OverlayScene.BuildPm4BaseTransform(PlacementAnchor, BaseRotationRadians);
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
