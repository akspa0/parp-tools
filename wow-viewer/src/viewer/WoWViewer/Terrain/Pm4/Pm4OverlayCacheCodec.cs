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
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

/// <summary>Pure static PM4 overlay helpers moved verbatim from <c>WorldScene</c> (Epic 251 U-01 E1).</summary>
internal static class Pm4OverlayCacheCodec
{

    internal static bool ShouldSuppressPm4FinalStatusLog(string status)
    {
        return status.StartsWith("PM4: no files intersect camera window", StringComparison.Ordinal)
            || status.StartsWith("PM4: 0/", StringComparison.Ordinal)
            || status.Contains("none decoded into overlay data", StringComparison.Ordinal);
    }

    internal static (int minTileX, int minTileY, int maxTileX, int maxTileY) GetPm4CameraWindow(Vector3 cameraPos, int tileRadius)
    {
        GetPm4CameraTile(cameraPos, out int centerTileX, out int centerTileY);
        int minTileX = Math.Max(0, centerTileX - tileRadius);
        int minTileY = Math.Max(0, centerTileY - tileRadius);
        int maxTileX = Math.Min(63, centerTileX + tileRadius);
        int maxTileY = Math.Min(63, centerTileY + tileRadius);
        return (minTileX, minTileY, maxTileX, maxTileY);
    }

    internal static bool IsPm4TileInsideCameraWindow(
        int tileX,
        int tileY,
        (int minTileX, int minTileY, int maxTileX, int maxTileY) cameraWindow)
    {
        return tileX >= cameraWindow.minTileX
            && tileX <= cameraWindow.maxTileX
            && tileY >= cameraWindow.minTileY
            && tileY <= cameraWindow.maxTileY;
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
    internal static int ApplyCachedTilesToTileDictionaries(
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
    /// Spec 054: read a per-file entry from disk and gate on the file
    /// stamp. A stamp mismatch (file content changed since the entry
    /// was written) is treated as a miss and the on-disk entry is
    /// deleted so the next read is also a miss.
    /// </summary>
    internal static bool TryReadPerFileDiskCache(
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
    internal static List<CorePm4CachedObject> BuildCachedObjectsForDiskWrite(
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

    internal static bool IsMapPm4Path(string path, string mapName)
    {
        string normalized = path.Replace('\\', '/');
        string fileName = Path.GetFileName(normalized);
        if (fileName.StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase))
            return true;

        string mapSegment = "/" + mapName + "/";
        return normalized.Contains(mapSegment, StringComparison.OrdinalIgnoreCase);
    }
}
