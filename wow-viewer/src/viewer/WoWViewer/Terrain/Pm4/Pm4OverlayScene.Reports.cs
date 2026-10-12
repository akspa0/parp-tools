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
using static WoWViewer.Terrain.Pm4OverlayMatching;
using static WoWViewer.Terrain.Pm4OverlayCacheCodec;
using static WoWViewer.Terrain.Pm4OverlayGeometry;
using static WoWViewer.Terrain.Pm4OverlayCoordinates;
using static WoWViewer.Terrain.Pm4OverlayColors;

namespace WoWViewer.Terrain;

// Read-only reports: interchange JSON, OBJ export, WMO correlation, match states, legends, summaries.
public sealed partial class Pm4OverlayScene
{

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
            .Where(tileEntry => IsTileWithinPm4MatchRadius((tileEntry.Key.Item1, tileEntry.Key.Item2)))
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

    /// <summary>The camera-anchor tile, or null when the camera is outside the 64x64 grid.</summary>
    internal (int tileX, int tileY)? GetPm4CameraTile()
    {
        Vector3 anchor = GetPm4LoadAnchorCameraPosition();
        int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - anchor.X) / WoWConstants.ChunkSize);
        int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - anchor.Y) / WoWConstants.ChunkSize);
        if ((uint)tileX >= 64u || (uint)tileY >= 64u)
            return null;

        return (tileX, tileY);
    }

    private bool IsTileWithinPm4MatchRadius((int tileX, int tileY) candidate)
    {
        (int tileX, int tileY)? center = GetPm4CameraTile();
        return !center.HasValue
            || (Math.Abs(candidate.tileX - center.Value.tileX) <= Pm4MatchCameraTileRadius
                && Math.Abs(candidate.tileY - center.Value.tileY) <= Pm4MatchCameraTileRadius);
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

        foreach (var tileEntry in _pm4TileObjects.Where(tileEntry => IsTileWithinPm4MatchRadius(tileEntry.Key)))
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

    internal static void GetPm4CameraTile(Vector3 cameraPos, out int tileX, out int tileY)
    {
        // PM4 filenames and terrain AOI both operate on ADT tile coordinates (64x64 grid).
        // WoWConstants.TileSize is the larger WDL tile span, which collapses camera-window
        // PM4 loads into a tiny corner of the map. Use the ADT tile span instead.
        float camTileX = (WoWConstants.MapOrigin - cameraPos.X) / WoWConstants.ChunkSize;
        float camTileY = (WoWConstants.MapOrigin - cameraPos.Y) / WoWConstants.ChunkSize;
        tileX = Math.Clamp((int)MathF.Floor(camTileX), 0, 63);
        tileY = Math.Clamp((int)MathF.Floor(camTileY), 0, 63);
    }

    internal Vector3 GetPm4ObjectColor((int tileX, int tileY) tileKey, Pm4OverlayObject obj)
    {
        return _pm4ColorMode switch
        {
            Pm4OverlayColorMode.PlacementZ => ColorFromSeed(obj.Ck24),
            Pm4OverlayColorMode.Population => obj.Ck24 == 0
                ? new Vector3(0.95f, 0.55f, 0.20f)
                : new Vector3(0.30f, 0.70f, 0.95f),
            Pm4OverlayColorMode.Tile => ColorFromSeed((uint)HashCode.Combine(tileKey.tileX, tileKey.tileY)),
            Pm4OverlayColorMode.MshdRegionId => ColorFromSeed(obj.MshdRegionId),
            Pm4OverlayColorMode.SurfaceCount => ColorFromSeed((uint)obj.SurfaceCount),
            Pm4OverlayColorMode.GroupKey => ColorFromSeed(obj.DominantGroupKey),
            Pm4OverlayColorMode.Height => ColorFromHeight(obj.Center.Z),
            Pm4OverlayColorMode.TypeFlags => BlendTypeFlagColors(obj.DistinctTypeFlags),
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
                        if (ShouldRenderPm4Object(obj))
                            yield return (tileEntry.Key, obj);
                    }
                }
            }

            /// <summary>
            /// Builds the PM4 scene graph as Region -> Tile -> Object, for an outliner view.
            /// </summary>
            /// <remarks>
            /// Objects are named by the placed asset that produced them wherever that can be
            /// resolved. The key is <c>MSUR._0x1C</c> read as a float, which equals the producing
            /// placement's Z (measured 2026-08-23 at 93.58% against a 2.10% control), combined with
            /// the placement standing inside the object's horizontal footprint.
            ///
            /// <para>The stored <c>Ck24</c> is the top 24 bits of that float, so the low mantissa
            /// byte is not available here and the reconstructed value carries roughly 0.003%
            /// relative error - about 0.001 units at typical heights. The match tolerance below is
            /// sized for that, and an unresolved object is labelled by its value rather than being
            /// hidden.</para>
            /// </remarks>
            /// <summary>
            /// Live measurements over the loaded PM4 objects, so the corpus findings can be checked
            /// against whatever is on screen instead of taken on trust.
            /// </summary>
            /// <remarks>
            /// The one worth watching is the agreement between two unrelated fields:
            /// <c>MSUR._0x00 == 0x03</c> marks a doodad surface, and <c>MSUR._0x1C == 0</c> means no
            /// placement height was recorded. Corpus-wide they pick out the SAME objects, with a
            /// single exception in 1,929. Any object where they disagree is either a genuine edge
            /// case or a decode error, and is worth looking at directly - so it is counted here
            /// rather than averaged away.
            /// </remarks>
            public Pm4SceneFacts BuildPm4SceneFacts()
            {
                var classCounts = new Dictionary<byte, (int withHeight, int withoutHeight)>();
                int objects = 0, withHeight = 0, resolved = 0, disagreements = 0;

                foreach (((int tileX, int tileY, uint ck24, int objectPart) key, Pm4OverlayObject obj, Pm4ObjectDebugInfo debug)
                    in EnumerateVisiblePm4OverlayDebugObjects())
                {
                    objects++;
                    bool hasHeight = debug.Ck24 != 0;
                    if (hasHeight)
                        withHeight++;

                    byte cls = debug.DominantGroupKey;
                    (int w, int wo) = classCounts.GetValueOrDefault(cls);
                    classCounts[cls] = hasHeight ? (w + 1, wo) : (w, wo + 1);

                    // 0x03 should never carry a height; every other class should always carry one.
                    if ((cls == 0x03) == hasHeight)
                        disagreements++;

                    if (TryResolvePm4Asset(debug.Ck24, debug.BoundsMin, debug.BoundsMax, out _, out _, out _))
                        resolved++;
                }

                List<Pm4SceneClassFact> classes = classCounts
                    .Select(kv => new Pm4SceneClassFact(kv.Key, kv.Value.withHeight, kv.Value.withoutHeight))
                    .OrderByDescending(static c => c.WithHeight + c.WithoutHeight)
                    .ToList();

                return new Pm4SceneFacts(objects, withHeight, objects - withHeight, resolved, disagreements, classes);
            }

            public IReadOnlyList<Pm4OutlineRegion> BuildPm4Outline(float zTolerance = 0.05f)
            {
                var byRegion = new Dictionary<uint, Dictionary<(int tileX, int tileY), List<Pm4OutlineObject>>>();

                foreach (((int tileX, int tileY, uint ck24, int objectPart) key, Pm4OverlayObject obj, Pm4ObjectDebugInfo debug)
                    in EnumerateVisiblePm4OverlayDebugObjects())
                {
                    bool ok = TryResolvePm4Asset(
                        debug.Ck24, debug.BoundsMin, debug.BoundsMax,
                        out string? assetName, out int resolvedUniqueId, out float placementZ, zTolerance);
                    int? uniqueId = ok ? resolvedUniqueId : null;
                    float? bestDelta = ok ? 0f : null;

                    if (!byRegion.TryGetValue(debug.MshdRegionId, out var tiles))
                    {
                        tiles = [];
                        byRegion[debug.MshdRegionId] = tiles;
                    }

                    var tileKey = (key.tileX, key.tileY);
                    if (!tiles.TryGetValue(tileKey, out var list))
                    {
                        list = [];
                        tiles[tileKey] = list;
                    }

                    list.Add(new Pm4OutlineObject(
                        key,
                        debug.Ck24,
                        debug.Ck24Type,
                        placementZ,
                        debug.SurfaceCount,
                        debug.BoundsMin,
                        debug.BoundsMax,
                        debug.Center,
                        assetName,
                        uniqueId,
                        bestDelta));
                }

                return byRegion
                    .Select(r => new Pm4OutlineRegion(
                        r.Key,
                        r.Value
                            .OrderBy(static t => t.Key.tileX).ThenBy(static t => t.Key.tileY)
                            .Select(t => new Pm4OutlineTile(
                                t.Key.tileX,
                                t.Key.tileY,
                                t.Value.OrderByDescending(static o => o.SurfaceCount).ToList()))
                            .ToList(),
                        r.Value.Sum(static t => t.Value.Count),
                        r.Value.Sum(static t => t.Value.Count(static o => o.AssetName is not null))))
                    .OrderByDescending(static r => r.ObjectCount)
                    .ThenBy(static r => r.RegionId)
                    .ToList();
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
            Pm4OverlayColorMode.PlacementZ     => value == 0
                ? "no placement height (doodad collision)"
                : $"placement Z {BitConverter.UInt32BitsToSingle(value << 8):F2}",
            Pm4OverlayColorMode.Population     => value == 0 ? "doodad collision (no placement height)" : "placed object (has placement height)",
            Pm4OverlayColorMode.MshdRegionId   => $"MSHD region {value}",
            Pm4OverlayColorMode.SurfaceCount   => $"{value} surfaces",
            Pm4OverlayColorMode.GroupKey       => $"MSUR._0x00 = 0x{value:X2} (unmeasured; 9 values corpus-wide)",
            Pm4OverlayColorMode.TypeFlags      => ((byte)value) switch
            {
                0x03 => "TypeFlags 0x03 — M2 top surfaces",
                0x10 => "TypeFlags 0x10 — interior WMO floors",
                0x12 => "TypeFlags 0x12 — exterior WMO solids",
                _ => $"TypeFlags 0x{value:X2} — unknown",
            },
            _ => value.ToString(CultureInfo.InvariantCulture)
        };
    }

    private Vector3 GetPm4LegendColor(Pm4OverlayColorMode mode, uint value)
    {
        return mode switch
        {
            Pm4OverlayColorMode.PlacementZ    => ColorFromSeed(value),
            Pm4OverlayColorMode.Population    => value == 0
                ? new Vector3(0.95f, 0.55f, 0.20f)
                : new Vector3(0.30f, 0.70f, 0.95f),
            Pm4OverlayColorMode.MshdRegionId  => ColorFromSeed(value),
            Pm4OverlayColorMode.SurfaceCount  => ColorFromSeed(value),
            Pm4OverlayColorMode.GroupKey      => ColorFromSeed(value),
            Pm4OverlayColorMode.TypeFlags     => GetTypeFlagColor((byte)value),
            _ => GetPm4TypeColor((byte)value)  // fallback (Ck24Type uses GetPm4TypeColor)
        };
    }
    private Vector3 ColorFromHeight(float z)
    {
        float denom = _pm4MaxObjectZ - _pm4MinObjectZ;
        float t = denom > 0.001f ? Math.Clamp((z - _pm4MinObjectZ) / denom, 0f, 1f) : 0.5f;
        return Vector3.Lerp(new Vector3(0.45f, 0.55f, 0.80f), new Vector3(0.80f, 0.50f, 0.45f), t);
    }
}
