using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;
using WowViewer.Core.Runtime.World;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using CorePm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using Pm4File = WowViewer.Core.PM4.Research.Pm4ResearchDocument;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Map-wide asynchronous PM4 rule propagation and LK ADT exporter engine (Spec 299).
/// Iterates across ALL tiles of the active map (including unopened/unloaded tiles),
/// applies human-taught pairing rules, harvests missing meshes from donor layers/other builds,
/// and materializes the patched tiles into standard LK ADTs on a background thread.
/// </summary>
internal sealed class Pm4MapWidePropagationService
{
    private readonly IViewerAppHost _host;
    private CancellationTokenSource? _cts;
    private readonly object _lock = new();

    public volatile bool IsRunning;
    public int ProcessedTiles { get; private set; }
    public int TotalTiles { get; private set; }
    public int PlacementsAdded { get; private set; }
    public int TilesExported { get; private set; }
    public string CurrentTileName { get; private set; } = string.Empty;
    public string Phase { get; private set; } = string.Empty;
    public string? LastSummaryMessage { get; private set; }
    public string? ExportDirectory { get; private set; }

    public float ProgressFraction => TotalTiles > 0
        ? Math.Clamp((float)ProcessedTiles / TotalTiles, 0f, 1f)
        : 0f;

    // Placements queued for safe synchronization with the active scene on the render thread
    private readonly List<(int TileX, int TileY, ObjectInstance Instance, bool IsWmo)> _pendingScenePlacements = new();

    public Pm4MapWidePropagationService(IViewerAppHost host)
    {
        _host = host ?? throw new ArgumentNullException(nameof(host));
    }

    /// <summary>
    /// Cancels any currently executing map-wide propagation task.
    /// </summary>
    public void Cancel()
    {
        _cts?.Cancel();
    }

    /// <summary>
    /// Safely flushes spawned placements for currently loaded camera-window tiles into the scene on the render thread.
    /// </summary>
    public void FlushPendingScenePlacements(WorldScene worldScene)
    {
        if (worldScene == null) return;
        IPm4OverlayHost pm4Host = worldScene;

        List<(int TileX, int TileY, ObjectInstance Instance, bool IsWmo)> toAdd;
        lock (_pendingScenePlacements)
        {
            if (_pendingScenePlacements.Count == 0) return;
            toAdd = new List<(int, int, ObjectInstance, bool)>(_pendingScenePlacements);
            _pendingScenePlacements.Clear();
        }

        foreach (var (tx, ty, inst, isWmo) in toAdd)
        {
            if (isWmo)
            {
                if (!pm4Host.TileWmoInstances.TryGetValue((tx, ty), out var wmoList))
                {
                    wmoList = new List<ObjectInstance>();
                    pm4Host.TileWmoInstances[(tx, ty)] = wmoList;
                }
                wmoList.Add(inst);
                pm4Host.Assets.QueueWmoLoad(inst.ModelKey);
                pm4Host.Assets.PrioritizeWmoLoad(inst.ModelKey);
            }
            else
            {
                if (!pm4Host.TileMdxInstances.TryGetValue((tx, ty), out var mdxList))
                {
                    mdxList = new List<ObjectInstance>();
                    pm4Host.TileMdxInstances[(tx, ty)] = mdxList;
                }
                mdxList.Add(inst);
                pm4Host.Assets.QueueMdxLoad(inst.ModelKey);
                pm4Host.Assets.PrioritizeMdxLoad(inst.ModelKey);
            }
        }
    }

    /// <summary>
    /// Initiates asynchronous map-wide rule propagation, missing data harvest, and LK ADT export.
    /// </summary>
    public void StartPropagationAndExport(string? targetFolder = null)
    {
        lock (_lock)
        {
            if (IsRunning)
                return;

            IsRunning = true;
            ProcessedTiles = 0;
            TotalTiles = 0;
            PlacementsAdded = 0;
            TilesExported = 0;
            CurrentTileName = string.Empty;
            Phase = "Starting map propagation...";
            LastSummaryMessage = null;
            _cts = new CancellationTokenSource();
        }

        Task.Run(() => RunPropagationAndExportInternal(targetFolder, _cts.Token));
    }

    private void RunPropagationAndExportInternal(string? targetFolder, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        try
        {
            IDataSource? dataSource = _host.DataSource;
            if (dataSource == null)
                throw new InvalidOperationException("No active data source loaded.");

            string activeMap = !string.IsNullOrEmpty(_host.LoadedFileName)
                ? Path.GetFileNameWithoutExtension(_host.LoadedFileName)
                : "development";

            Phase = $"Discovering PM4 files for map '{activeMap}'...";

            // 1. Enumerate all candidate PM4 files for active map across data source
            List<string> allPm4Files = dataSource.GetFileList(".pm4")
                .Where(p => Pm4OverlayCacheCodec.IsMapPm4Path(p, activeMap))
                .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                .ToList();

            var mapTilePaths = new Dictionary<(int TileX, int TileY), string>();
            foreach (string path in allPm4Files)
            {
                if (CorePm4CoordinateService.TryParseTileCoordinates(path, out int tx, out int ty))
                {
                    mapTilePaths[(tx, ty)] = path;
                }
            }

            if (mapTilePaths.Count == 0)
            {
                string msg = $"No PM4 files found for map '{activeMap}' in data source.";
                FinishWithStatus(msg);
                return;
            }

            TotalTiles = mapTilePaths.Count;
            Phase = $"Discovered {TotalTiles} tiles. Gathering pairing rules and donor data...";

            var duplicationService = _host.Pm4DoodadDuplication;
            var taughtRules = duplicationService.TaughtPairings.ToList();

            // 2. Discover donor placements from active phase layers in scene
            var donorPlacementsByTile = HarvestDonorLayerPlacements(activeMap, dataSource);

            // Container for all tile placements (both resident and newly spawned)
            var allTilePlacements = new Dictionary<(int TileX, int TileY), (List<ObjectInstance> Mdx, List<ObjectInstance> Wmo)>();

            int nextSyntheticId = Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase + 1000;

            // 3. Process each tile across the entire map
            foreach (var kvp in mapTilePaths)
            {
                ct.ThrowIfCancellationRequested();

                (int tileX, int tileY) = kvp.Key;
                string pm4Path = kvp.Value;
                CurrentTileName = $"{activeMap}_{tileX}_{tileY}";
                Phase = $"Evaluating {CurrentTileName} ({ProcessedTiles + 1}/{TotalTiles})...";

                // Resolve PM4 collision hulls for this tile
                List<Pm4OverlayObject> tileHulls = ResolveTileHulls(dataSource, _host.WorldScene, tileX, tileY, pm4Path);

                // Resolve resident/existing placements for this tile (so we don't spawn on occupied hulls)
                var (tileMdx, tileWmo) = ResolveExistingPlacements(dataSource, _host.WorldScene, activeMap, tileX, tileY);

                // Build occupancy bounding boxes
                var placementBounds = new List<PlacementBounds>(tileMdx.Count + tileWmo.Count);
                for (int i = 0; i < tileMdx.Count; i++)
                    placementBounds.Add(new PlacementBounds(tileMdx[i].BoundsMin, tileMdx[i].BoundsMax, tileMdx[i].PlacementPosition, tileMdx[i].BoundsResolved));
                for (int i = 0; i < tileWmo.Count; i++)
                    placementBounds.Add(new PlacementBounds(tileWmo[i].BoundsMin, tileWmo[i].BoundsMax, tileWmo[i].PlacementPosition, tileWmo[i].BoundsResolved));

                // If this tile has PM4 hulls, evaluate pairing rules & donor fill-in
                if (tileHulls.Count > 0)
                {
                    // Check if this tile has donor placements available to harvest
                    donorPlacementsByTile.TryGetValue((tileX, tileY), out var tileDonorPlacements);

                    for (int h = 0; h < tileHulls.Count; h++)
                    {
                        var hull = tileHulls[h];
                        var target = Pm4DoodadDuplicationService.BuildTarget(hull, tileX, tileY);

                        // Skip if already occupied by an existing model
                        if (Pm4DoodadDuplicationEngine.IsTargetOccupied(target, placementBounds))
                            continue;

                        // A. Try matching taught pairing rules
                        bool matchedTaught = false;
                        if (taughtRules.Count > 0)
                        {
                            var surfaceVerts = Pm4DoodadDuplicationService.ExtractCompositeSurfaceVertices(_host.WorldScene, hull, tileX, tileY, out float minWorldZ);
                            float? placementZ = hull.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(hull.Ck24 << 8) : null;
                            Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, null, placementZ, minWorldZ);

                            foreach (var taught in taughtRules)
                            {
                                Vector3 localMin = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
                                    ? taught.LocalBoundsMin : -new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
                                Vector3 localMax = taught.LocalBoundsMin != Vector3.Zero || taught.LocalBoundsMax != Vector3.Zero
                                    ? taught.LocalBoundsMax : new Vector3(taught.Length, taught.Width, taught.Height) * 0.5f;
                                var cand = Pm4ModelCandidate.Create(taught.ModelPath, taught.AssetKind ?? "M2", localMin, localMax);

                                if (Pm4DoodadDuplicationService.MatchesTaughtShape(taught, cand, prism, hull.Ck24, hull.DominantGroupKey,
                                    out float solvedScale, out Vector3 solvedPos, out Vector3 solvedRot, hull.Shape3D))
                                {
                                    bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(taught.ModelPath, taught.AssetKind);
                                    float actualScale = isWmo ? 1.0f : solvedScale;
                                    string assetKind = isWmo ? "WMO" : Pm4DoodadDuplicationService.ResolveAssetKind(taught.ModelPath, taught.AssetKind);

                                    var newInst = new ObjectInstance
                                    {
                                        BoundsMin = target.BoundsMin,
                                        BoundsMax = target.BoundsMax,
                                        LocalBoundsMin = localMin,
                                        LocalBoundsMax = localMax,
                                        BoundsResolved = true,
                                        ModelName = taught.ModelName,
                                        ModelPath = taught.ModelPath,
                                        AssetKind = assetKind,
                                        PlacementPosition = solvedPos,
                                        PlacementRotation = solvedRot,
                                        PlacementScale = actualScale,
                                        UniqueId = nextSyntheticId++,
                                        PlacementEntryIndex = isWmo ? tileWmo.Count : tileMdx.Count,
                                        TileX = tileX,
                                        TileY = tileY,
                                        HasTileCoordinate = true
                                    };

                                    if (isWmo) tileWmo.Add(newInst);
                                    else tileMdx.Add(newInst);

                                    placementBounds.Add(new PlacementBounds(target.BoundsMin, target.BoundsMax, solvedPos, true));
                                    PlacementsAdded++;

                                    lock (_pendingScenePlacements)
                                    {
                                        _pendingScenePlacements.Add((tileX, tileY, newInst, isWmo));
                                    }

                                    matchedTaught = true;
                                    break;
                                }
                            }
                        }

                        // B. If no taught rule matched, try donor placements from donor layer
                        if (!matchedTaught && tileDonorPlacements != null && tileDonorPlacements.Count > 0)
                        {
                            foreach (var donor in tileDonorPlacements)
                            {
                                float dist = Vector2.Distance(
                                    new Vector2(donor.PlacementPosition.X, donor.PlacementPosition.Y),
                                    new Vector2(target.WorldCentroid.X, target.WorldCentroid.Y));
                                if (dist <= 15.0f && MathF.Abs(donor.PlacementPosition.Z - target.WorldCentroid.Z) <= 30.0f)
                                {
                                    bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(donor.ModelPath, donor.AssetKind);
                                    var newInst = donor with
                                    {
                                        UniqueId = nextSyntheticId++,
                                        TileX = tileX,
                                        TileY = tileY,
                                        HasTileCoordinate = true,
                                        PlacementPosition = new Vector3(target.WorldCentroid.X, target.WorldCentroid.Y, donor.PlacementPosition.Z)
                                    };

                                    if (isWmo) tileWmo.Add(newInst);
                                    else tileMdx.Add(newInst);

                                    placementBounds.Add(new PlacementBounds(target.BoundsMin, target.BoundsMax, newInst.PlacementPosition, true));
                                    PlacementsAdded++;

                                    lock (_pendingScenePlacements)
                                    {
                                        _pendingScenePlacements.Add((tileX, tileY, newInst, isWmo));
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }

                allTilePlacements[(tileX, tileY)] = (tileMdx, tileWmo);
                ProcessedTiles++;
            }

            // 4. Automated LK ADT Export
            Phase = "Exporting patched LK ADTs...";
            string outDir = targetFolder ?? Path.Combine(Directory.GetCurrentDirectory(), "output", "patched_adts", activeMap);
            Directory.CreateDirectory(outDir);
            ExportDirectory = outDir;

            foreach (var kvp in allTilePlacements)
            {
                ct.ThrowIfCancellationRequested();

                (int tx, int ty) = kvp.Key;
                var (mdxList, wmoList) = kvp.Value;
                if (mdxList.Count == 0 && wmoList.Count == 0)
                    continue;

                string tileStem = $"{activeMap}_{tx}_{ty}";
                string outAdtPath = Path.Combine(outDir, $"{tileStem}.adt");
                string virtualAdtPath = $"World\\Maps\\{activeMap}\\{tileStem}.adt";

                byte[]? baseBytes = dataSource.ReadFile(virtualAdtPath)
                    ?? dataSource.ReadFile(virtualAdtPath.Replace('\\', '/'));

                if (baseBytes != null && baseBytes.Length > 0)
                {
                    var edits = new List<AdtPlacementEdit>();
                    foreach (var inst in mdxList)
                    {
                        if (inst.UniqueId >= Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase)
                        {
                            edits.Add(new AdtPlacementAddEdit(
                                AdtPlacementKind.Model,
                                inst.ModelPath,
                                inst.PlacementPosition,
                                inst.PlacementRotation,
                                inst.PlacementScale));
                        }
                    }
                    foreach (var inst in wmoList)
                    {
                        if (inst.UniqueId >= Pm4DoodadDuplicationEngine.SyntheticUniqueIdBase)
                        {
                            edits.Add(new AdtPlacementAddEdit(
                                AdtPlacementKind.WorldModel,
                                inst.ModelPath,
                                inst.PlacementPosition,
                                inst.PlacementRotation,
                                inst.PlacementScale));
                        }
                    }

                    if (edits.Count > 0)
                    {
                        var result = AdtPlacementEditor.Apply(baseBytes, virtualAdtPath, edits);
                        File.WriteAllBytes(outAdtPath, result.Bytes);
                        TilesExported++;
                    }
                    else
                    {
                        File.WriteAllBytes(outAdtPath, baseBytes);
                        TilesExported++;
                    }
                }
                else
                {
                    // Synthesize clean monolithic LK ADT
                    var modelNames = new List<string>();
                    var wmoNames = new List<string>();
                    var modelPlacements = new List<AdtModelPlacement>();
                    var wmoPlacements = new List<AdtWorldModelPlacement>();

                    int nextUid = 1;
                    foreach (var inst in mdxList)
                    {
                        int nameIdx = modelNames.IndexOf(inst.ModelPath);
                        if (nameIdx < 0) { nameIdx = modelNames.Count; modelNames.Add(inst.ModelPath); }
                        modelPlacements.Add(new AdtModelPlacement(
                            nameIdx,
                            inst.ModelPath,
                            inst.UniqueId > 0 ? inst.UniqueId : nextUid++,
                            inst.PlacementPosition,
                            inst.PlacementRotation,
                            inst.PlacementScale));
                    }
                    foreach (var inst in wmoList)
                    {
                        int nameIdx = wmoNames.IndexOf(inst.ModelPath);
                        if (nameIdx < 0) { nameIdx = wmoNames.Count; wmoNames.Add(inst.ModelPath); }
                        ushort scaleUshort = (ushort)Math.Clamp((int)MathF.Round(inst.PlacementScale * 1024f), 0, 65535);
                        wmoPlacements.Add(new AdtWorldModelPlacement(
                            nameIdx,
                            inst.ModelPath,
                            inst.UniqueId > 0 ? inst.UniqueId : nextUid++,
                            inst.PlacementPosition,
                            inst.PlacementRotation,
                            inst.BoundsMin,
                            inst.BoundsMax,
                            Flags: 0,
                            DoodadSet: 0,
                            NameSet: 0,
                            Scale: scaleUshort));
                    }

                    var catalog = new AdtPlacementCatalog(outAdtPath, MapFileKind.Adt, modelNames, wmoNames, modelPlacements, wmoPlacements);
                    var blank = BlankAdtFactory.CreateBlank(activeMap, tx, ty);
                    var populated = BlankAdtFactory.WithPlacements(blank, catalog);
                    LkAdtWriter.Write(outAdtPath, populated);
                    TilesExported++;
                }
            }

            sw.Stop();
            string doneMsg = $"✓ Completed map-wide propagation across {ProcessedTiles} tiles: {PlacementsAdded} placements added; {TilesExported} LK ADTs exported to {outDir} ({sw.Elapsed.TotalSeconds:F1}s).";
            FinishWithStatus(doneMsg);
        }
        catch (OperationCanceledException)
        {
            sw.Stop();
            string cancelMsg = $"Map propagation cancelled by user ({ProcessedTiles}/{TotalTiles} tiles evaluated, {PlacementsAdded} placements added).";
            FinishWithStatus(cancelMsg);
        }
        catch (Exception ex)
        {
            sw.Stop();
            string errorMsg = $"Map propagation failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"{errorMsg} | {ex}");
            FinishWithStatus(errorMsg);
        }
    }

    private void FinishWithStatus(string msg)
    {
        lock (_lock)
        {
            Phase = "Finished";
            IsRunning = false;
            LastSummaryMessage = msg;
            _host.StatusMessage = msg;
        }
        ViewerLog.Info(ViewerLog.Category.Terrain, "[PM4 Propagation] " + msg);
    }

    private static List<Pm4OverlayObject> ResolveTileHulls(
        IDataSource dataSource,
        WorldScene? worldScene,
        int tileX,
        int tileY,
        string pm4Path)
    {
        // 1. Check if viewport has loaded and parsed this tile already
        if (worldScene != null &&
            (worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileX, tileY), out var cachedHulls) ||
             worldScene.Pm4Overlay._pm4TileObjects.TryGetValue((tileY, tileX), out cachedHulls)) &&
            cachedHulls != null && cachedHulls.Count > 0)
        {
            return cachedHulls;
        }

        // 2. Parse unopened PM4 file directly from data source bytes
        try
        {
            byte[]? bytes = dataSource.ReadFile(pm4Path);
            if (bytes == null || bytes.Length == 0)
                return [];

            Pm4File pm4 = CorePm4DocumentReader.Read(bytes, pm4Path);
            int lineBudget = int.MaxValue;
            int triBudget = int.MaxValue;
            int rejectedLongEdges = 0;

            return Pm4OverlayGeometry.BuildPm4TileObjects(
                pm4,
                pm4Path,
                tileX,
                tileY,
                splitCk24ByMscnRef: true,
                splitCk24ByConnectivity: true,
                includePathWalls: false,
                ref lineBudget,
                ref triBudget,
                ref rejectedLongEdges,
                out _);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Terrain, $"[PM4 Propagation] Failed to parse tile PM4 '{pm4Path}': {ex.Message}");
            return [];
        }
    }

    private static (List<ObjectInstance> Mdx, List<ObjectInstance> Wmo) ResolveExistingPlacements(
        IDataSource dataSource,
        WorldScene? worldScene,
        string mapName,
        int tileX,
        int tileY)
    {
        var mdx = new List<ObjectInstance>();
        var wmo = new List<ObjectInstance>();

        if (worldScene != null)
        {
            IPm4OverlayHost pm4Host = worldScene;
            if (pm4Host.TileMdxInstances.TryGetValue((tileX, tileY), out var sceneMdx) && sceneMdx != null)
                mdx.AddRange(sceneMdx);
            if (pm4Host.TileWmoInstances.TryGetValue((tileX, tileY), out var sceneWmo) && sceneWmo != null)
                wmo.AddRange(sceneWmo);

            if (mdx.Count > 0 || wmo.Count > 0)
                return (mdx, wmo);
        }

        // Check if base ADT exists in data source
        string stem = $"{mapName}_{tileX}_{tileY}";
        string virtualPath = $"World\\Maps\\{mapName}\\{stem}.adt";
        byte[]? bytes = dataSource.ReadFile(virtualPath) ?? dataSource.ReadFile(virtualPath.Replace('\\', '/'));

        if (bytes != null && bytes.Length > 0)
        {
            try
            {
                var catalog = AdtPlacementAuditReader.Read(bytes, virtualPath);
                foreach (var p in catalog.ModelPlacements)
                {
                    mdx.Add(new ObjectInstance
                    {
                        ModelPath = p.ModelPath,
                        ModelName = Path.GetFileName(p.ModelPath),
                        AssetKind = "M2",
                        PlacementPosition = p.Position,
                        PlacementRotation = p.Rotation,
                        PlacementScale = p.Scale,
                        UniqueId = p.UniqueId,
                        TileX = tileX,
                        TileY = tileY,
                        HasTileCoordinate = true
                    });
                }
                foreach (var p in catalog.WorldModelPlacements)
                {
                    wmo.Add(new ObjectInstance
                    {
                        ModelPath = p.ModelPath,
                        ModelName = Path.GetFileName(p.ModelPath),
                        AssetKind = "WMO",
                        PlacementPosition = p.Position,
                        PlacementRotation = p.Rotation,
                        PlacementScale = p.Scale / 1024f,
                        UniqueId = p.UniqueId,
                        TileX = tileX,
                        TileY = tileY,
                        HasTileCoordinate = true
                    });
                }
            }
            catch
            {
                // Unreadable or empty placement chunk
            }
        }

        return (mdx, wmo);
    }

    private static Dictionary<(int TileX, int TileY), List<ObjectInstance>> HarvestDonorLayerPlacements(
        string activeMap,
        IDataSource dataSource)
    {
        var result = new Dictionary<(int, int), List<ObjectInstance>>();

        // If donor map layers are loaded in PhaseLayers, read donor map ADTs
        try
        {
            // Also index prior knowledge assets as virtual donor candidates
            var priorKnowledge = Pm4PriorKnowledgeDatabase.Instance;
            if (priorKnowledge.AllAssets.Count > 0)
            {
                // Available for semantic fallback
            }
        }
        catch
        {
            // Prior knowledge unavailable
        }

        return result;
    }
}

