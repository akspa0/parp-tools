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
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;

namespace WoWViewer.Terrain;

/// <summary>Pure static PM4 overlay helpers moved verbatim from <c>WorldScene</c> (Epic 251 U-01 E1).</summary>
internal static class Pm4OverlayColors
{

            internal static IReadOnlyList<Pm4SelectedObjectGraphTypeBucket> BuildTypeBuckets(
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

            internal static IReadOnlyList<Pm4VisibleTypeBucket> BuildPm4VisibleTypeBuckets(
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

    internal static uint GetPm4LegendValue(Pm4OverlayColorMode mode, Pm4OverlayObject obj)
    {
        return mode switch
        {
            Pm4OverlayColorMode.PlacementZ   => obj.Ck24,
            Pm4OverlayColorMode.Population    => obj.Ck24 == 0 ? 0u : 1u,
            Pm4OverlayColorMode.MshdRegionId  => obj.MshdRegionId,
            Pm4OverlayColorMode.SurfaceCount  => (uint)obj.SurfaceCount,
            Pm4OverlayColorMode.GroupKey      => obj.DominantGroupKey,
            Pm4OverlayColorMode.TypeFlags     => obj.DistinctTypeFlags,
            _ => obj.Ck24
        };
    }

    internal static Vector3 ColorFromSeed(uint seed)
    {
        uint golden = seed * 2654435761u;
        float hue = (golden & 0x00FFFFFF) / 16777215f;
        return HsvToRgb(hue, 0.75f, 0.95f);
    }

    internal static Vector3 HsvToRgb(float h, float s, float v)
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

    internal static Vector3 GetPm4TypeColor(byte ck24Type)
    {
        // Dark pastels for mesh — Ck24Type is a coarse container classifier
        return ck24Type switch
        {
            0x40 => new Vector3(0.85f, 0.55f, 0.30f),   // dark pastel orange
            0x80 => new Vector3(0.80f, 0.40f, 0.25f),   // dark pastel burnt orange
            _    => new Vector3(0.80f, 0.50f, 0.30f)    // dark pastel amber
        };
    }

    internal static Vector3 GetTypeFlagColor(byte typeFlag)
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

    internal static Vector3 BlendTypeFlagColors(uint typeFlagsMask)
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

    internal static Vector3 GetCk24TypeVsTypeFlagsColor(byte ck24Type, uint typeFlagsMask)
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
}
