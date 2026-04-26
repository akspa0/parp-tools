using System.Diagnostics;
using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.M2;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.Runtime.World.Wdl;
using WowViewer.Core.Wmo;

namespace WowViewer.App;

internal sealed record WowViewerWorldRuntimeFrameRequest(
    string ClientRoot,
    string MapInput,
    string BuildLabel,
    string LooseOverlayRoot,
    int TileX,
    int TileY,
    WorldFramePassOptions PassOptions);

internal sealed record WowViewerWorldPlacementAuditRequest(
    string ClientRoot,
    string MapInput,
    string BuildLabel,
    string LooseOverlayRoot,
    int Limit);

internal sealed record WowViewerWorldPlacementTileSummary(
    int TileX,
    int TileY,
    string SourcePath,
    int MdxCount,
    int WmoCount,
    string? SampleMdxPath,
    string? SampleWmoPath)
{
    public int PlacementCount => MdxCount + WmoCount;
}

internal sealed class WowViewerWorldRuntimeTileFrame
{
    public WowViewerWorldRuntimeTileFrame(
        int tileX,
        int tileY,
        string placementSourcePath,
        WorldTileStageSummary tileStageSummary,
        WorldTerrainTileData terrainTileData,
        WorldLiquidTileData liquidTileData,
        AdtPlacementCatalog placementCatalog)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(placementSourcePath);
        ArgumentNullException.ThrowIfNull(tileStageSummary);
        ArgumentNullException.ThrowIfNull(terrainTileData);
        ArgumentNullException.ThrowIfNull(liquidTileData);
        ArgumentNullException.ThrowIfNull(placementCatalog);

        TileX = tileX;
        TileY = tileY;
        PlacementSourcePath = placementSourcePath;
        TileStageSummary = tileStageSummary;
        TerrainTileData = terrainTileData;
        LiquidTileData = liquidTileData;
        PlacementCatalog = placementCatalog;
    }

    public int TileX { get; }

    public int TileY { get; }

    public string PlacementSourcePath { get; }

    public WorldTileStageSummary TileStageSummary { get; }

    public WorldTerrainTileData TerrainTileData { get; }

    public WorldLiquidTileData LiquidTileData { get; }

    public AdtPlacementCatalog PlacementCatalog { get; }
}

internal sealed class WowViewerWorldPlacementAuditResult
{
    public WowViewerWorldPlacementAuditResult(
        WowViewerWorldSessionBootstrapResult session,
        int scannedTileCount,
        int tilesWithPlacements,
        IReadOnlyList<WowViewerWorldPlacementTileSummary> topTiles)
    {
        Session = session;
        ScannedTileCount = scannedTileCount;
        TilesWithPlacements = tilesWithPlacements;
        TopTiles = topTiles;
    }

    public WowViewerWorldSessionBootstrapResult Session { get; }

    public int ScannedTileCount { get; }

    public int TilesWithPlacements { get; }

    public IReadOnlyList<WowViewerWorldPlacementTileSummary> TopTiles { get; }
}

internal sealed class WowViewerWorldRuntimeFrameResult
{
    public WowViewerWorldRuntimeFrameResult(
        WowViewerWorldSessionBootstrapResult session,
        int selectedTileX,
        int selectedTileY,
        string placementSourcePath,
        WorldTileStageSummary tileStageSummary,
        WorldWdlTileData wdlTileData,
        WorldTerrainTileData terrainTileData,
        WorldTerrainVisualSnapshot terrainVisualSnapshot,
        WorldLiquidTileData liquidTileData,
        IReadOnlyList<WowViewerWorldRuntimeTileFrame> activeTerrainTiles,
        AdtPlacementCatalog placementCatalog,
        IReadOnlyList<WorldObjectInstance> wmoInstances,
        IReadOnlyList<WorldObjectInstance> mdxInstances,
        IReadOnlyList<WorldObjectInstance> skyboxBackdropInstances,
        int readyWmoCount,
        int readyMdxCount,
        int culledWmoCount,
        int culledMdxCount,
        WorldVisibilityFrame visibility,
        WorldObjectPassFrame passFrame,
        WorldMdxRenderPlan mdxRenderPlan,
        WorldFramePassOptions passOptions,
        WorldRenderFrameStats stats,
        WorldRenderCompositionFrame composition,
        bool objectPhaseExecuted,
        string optimizationHint,
        IReadOnlyList<string> pendingAssetKeys,
        Vector3 cameraPosition,
        Vector3 cameraTarget,
        Vector3 cameraForward,
        Vector2 planarMin,
        Vector2 planarMax)
    {
        Session = session;
        SelectedTileX = selectedTileX;
        SelectedTileY = selectedTileY;
        PlacementSourcePath = placementSourcePath;
        TileStageSummary = tileStageSummary;
        WdlTileData = wdlTileData;
        TerrainTileData = terrainTileData;
        TerrainVisualSnapshot = terrainVisualSnapshot;
        LiquidTileData = liquidTileData;
        ArgumentNullException.ThrowIfNull(activeTerrainTiles);
        ActiveTerrainTiles = activeTerrainTiles;
        PlacementCatalog = placementCatalog;
        WmoInstances = wmoInstances;
        MdxInstances = mdxInstances;
        SkyboxBackdropInstances = skyboxBackdropInstances;
        ReadyWmoCount = readyWmoCount;
        ReadyMdxCount = readyMdxCount;
        CulledWmoCount = culledWmoCount;
        CulledMdxCount = culledMdxCount;
        Visibility = visibility;
        PassFrame = passFrame;
        MdxRenderPlan = mdxRenderPlan;
        PassOptions = passOptions;
        Stats = stats;
        Composition = composition;
        ObjectPhaseExecuted = objectPhaseExecuted;
        OptimizationHint = optimizationHint;
        PendingAssetKeys = pendingAssetKeys;
        CameraPosition = cameraPosition;
        CameraTarget = cameraTarget;
        CameraForward = cameraForward;
        PlanarMin = planarMin;
        PlanarMax = planarMax;
    }

    public WowViewerWorldSessionBootstrapResult Session { get; }

    public int SelectedTileX { get; }

    public int SelectedTileY { get; }

    public string PlacementSourcePath { get; }

    public WorldTileStageSummary TileStageSummary { get; }

    public WorldWdlTileData WdlTileData { get; }

    public WorldTerrainTileData TerrainTileData { get; }

    public WorldTerrainVisualSnapshot TerrainVisualSnapshot { get; }

    public WorldLiquidTileData LiquidTileData { get; }

    public IReadOnlyList<WowViewerWorldRuntimeTileFrame> ActiveTerrainTiles { get; }

    public AdtPlacementCatalog PlacementCatalog { get; }

    public IReadOnlyList<WorldObjectInstance> WmoInstances { get; }

    public IReadOnlyList<WorldObjectInstance> MdxInstances { get; }

    public IReadOnlyList<WorldObjectInstance> SkyboxBackdropInstances { get; }

    public int ReadyWmoCount { get; }

    public int ReadyMdxCount { get; }

    public int CulledWmoCount { get; }

    public int CulledMdxCount { get; }

    public WorldVisibilityFrame Visibility { get; }

    public WorldObjectPassFrame PassFrame { get; }

    public WorldMdxRenderPlan MdxRenderPlan { get; }

    public WorldFramePassOptions PassOptions { get; }

    public WorldRenderFrameStats Stats { get; }

    public WorldRenderCompositionFrame Composition { get; }

    public bool ObjectPhaseExecuted { get; }

    public string OptimizationHint { get; }

    public IReadOnlyList<string> PendingAssetKeys { get; }

    public Vector3 CameraPosition { get; }

    public Vector3 CameraTarget { get; }

    public Vector3 CameraForward { get; }

    public Vector2 PlanarMin { get; }

    public Vector2 PlanarMax { get; }
}

internal static class WowViewerWorldRuntimeBridge
{
    internal const float TileSize = 533.33333f;
    internal const float MapOrigin = 32.0f * TileSize;

    private readonly record struct LocalBoundsResolution(bool AssetReady, Vector3 LocalMin, Vector3 LocalMax, bool HasOpaqueRenderContent, bool HasTransparentRenderContent);

    private readonly record struct ResolvedWmoAsset(
        bool AssetReady,
        Vector3 LocalMin,
        Vector3 LocalMax,
        uint? Version,
        int GroupCount,
        int PortalCount,
        int DoodadSetCount,
        int MdxDoodadCount,
        int M2DoodadCount,
        int UnknownDoodadCount);

    public static WowViewerWorldPlacementAuditResult AuditPlacements(WowViewerWorldPlacementAuditRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        string clientRoot = Path.GetFullPath(request.ClientRoot);
        ArchiveCatalogBootstrapper.Bootstrap(
            archiveCatalog,
            [clientRoot],
            WowViewerArchiveBootstrap.CreateBootstrapOptions(request.BuildLabel, clientRoot));

        WowViewerWorldSessionBootstrapResult session = WowViewerWorldSessionBootstrapper.Open(
            new WowViewerWorldSessionOpenRequest(request.ClientRoot, request.MapInput, request.BuildLabel, request.LooseOverlayRoot),
            archiveCatalog);

        List<WowViewerWorldPlacementTileSummary> populatedTiles = [];
        foreach (WdtTileCoordinate tile in session.OccupiedTiles)
        {
            AdtPlacementCatalog catalog = ReadPlacementCatalog(session, tile.TileX, tile.TileY, archiveCatalog, out string sourcePath);
            int mdxCount = catalog.ModelPlacements.Count;
            int wmoCount = catalog.WorldModelPlacements.Count;
            if (mdxCount == 0 && wmoCount == 0)
                continue;

            populatedTiles.Add(new WowViewerWorldPlacementTileSummary(
                tile.TileX,
                tile.TileY,
                sourcePath,
                mdxCount,
                wmoCount,
                catalog.ModelPlacements.FirstOrDefault()?.ModelPath,
                catalog.WorldModelPlacements.FirstOrDefault()?.ModelPath));
        }

        IReadOnlyList<WowViewerWorldPlacementTileSummary> topTiles = populatedTiles
            .OrderByDescending(static tile => tile.PlacementCount)
            .ThenByDescending(static tile => tile.WmoCount)
            .ThenByDescending(static tile => tile.MdxCount)
            .ThenBy(static tile => tile.TileY)
            .ThenBy(static tile => tile.TileX)
            .Take(Math.Max(1, request.Limit))
            .ToArray();

        return new WowViewerWorldPlacementAuditResult(
            session,
            session.OccupiedTiles.Count,
            populatedTiles.Count,
            topTiles);
    }

    public static WowViewerWorldRuntimeFrameResult Build(WowViewerWorldRuntimeFrameRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        string clientRoot = Path.GetFullPath(request.ClientRoot);
        ArchiveCatalogBootstrapper.Bootstrap(
            archiveCatalog,
            [clientRoot],
            WowViewerArchiveBootstrap.CreateBootstrapOptions(request.BuildLabel, clientRoot));

        return Build(request, archiveCatalog);
    }

    internal static WowViewerWorldRuntimeFrameResult Build(WowViewerWorldRuntimeFrameRequest request, IArchiveCatalog archiveCatalog)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(archiveCatalog);

        WowViewerWorldSessionBootstrapResult session = WowViewerWorldSessionBootstrapper.Open(
            new WowViewerWorldSessionOpenRequest(request.ClientRoot, request.MapInput, request.BuildLabel, request.LooseOverlayRoot),
            archiveCatalog);

        WorldFramePassOptions framePassOptions = new(
            objectsVisible: request.PassOptions.ObjectsVisible,
            wmosVisible: request.PassOptions.WmosVisible,
            doodadsVisible: request.PassOptions.DoodadsVisible,
            skyVisible: request.PassOptions.SkyVisible,
            wdlVisible: false,
            terrainVisible: request.PassOptions.TerrainVisible,
            liquidVisible: request.PassOptions.LiquidVisible,
            overlayVisible: request.PassOptions.OverlayVisible);

        ((int tileX, int tileY) selectedTile, AdtPlacementCatalog placementCatalog, string placementSourcePath) =
            ResolveTileAndPlacements(session, request.TileX, request.TileY, archiveCatalog);
        WorldWdlTileData wdlTileData = WorldWdlTileData.Missing("WDL disabled for World Session; ADT terrain is the authoritative surface.", selectedTile.tileX, selectedTile.tileY);
        IReadOnlyList<WowViewerWorldRuntimeTileFrame> activeTerrainTiles = BuildActiveTerrainTiles(session, selectedTile.tileX, selectedTile.tileY, archiveCatalog);
        WowViewerWorldRuntimeTileFrame selectedTerrainTile = activeTerrainTiles.FirstOrDefault(tile => tile.TileX == selectedTile.tileX && tile.TileY == selectedTile.tileY)
            ?? throw new InvalidDataException($"Selected tile ({selectedTile.tileX},{selectedTile.tileY}) was not loaded into the active terrain window.");
        WorldTileStageSummary tileStageSummary = BuildAggregateTileStageSummary(activeTerrainTiles, selectedTile.tileX, selectedTile.tileY);
        WorldTerrainTileData terrainTileData = selectedTerrainTile.TerrainTileData;
        WorldTerrainVisualSnapshot terrainVisualSnapshot = WorldTerrainVisualSnapshotBuilder.Build(terrainTileData);
        WorldLiquidTileData liquidTileData = selectedTerrainTile.LiquidTileData;
        placementCatalog = selectedTerrainTile.PlacementCatalog;
        placementSourcePath = selectedTerrainTile.PlacementSourcePath;

        Dictionary<string, bool> assetReadyLookup = new(StringComparer.OrdinalIgnoreCase);

        List<WorldObjectInstance> wmoInstances = [];
        List<WorldObjectInstance> mdxInstances = [];
        foreach (WowViewerWorldRuntimeTileFrame activeTile in activeTerrainTiles)
        {
            wmoInstances.AddRange(BuildWmoInstances(activeTile.PlacementCatalog, activeTile.TileX, activeTile.TileY, assetReadyLookup));
            mdxInstances.AddRange(BuildMdxInstances(activeTile.PlacementCatalog, activeTile.TileX, activeTile.TileY, assetReadyLookup));
        }
        IReadOnlyList<WorldObjectInstance> skyboxBackdropInstances = mdxInstances
            .Where(static instance => WorldSkyboxBackdropClassifier.IsBackdropModelPath(instance.ModelPath))
            .ToArray();

        int readyWmoCount = wmoInstances.Count(instance => assetReadyLookup.TryGetValue(instance.ModelKey, out bool ready) && ready);
        int readyMdxCount = mdxInstances.Count(instance => assetReadyLookup.TryGetValue(instance.ModelKey, out bool ready) && ready);

        (Vector3 focusCenter, Vector2 planarMin, Vector2 planarMax) = ComputeWorldViewBounds(wmoInstances, mdxInstances, selectedTile.tileX, selectedTile.tileY);
        Vector3 cameraTarget = ComputeSpawnCameraTarget(selectedTile.tileX, selectedTile.tileY, terrainTileData, focusCenter);
        Vector3 cameraPosition = cameraTarget + new Vector3(-700f, -700f, 260f);
        Vector3 cameraForward = Vector3.Normalize(cameraTarget - cameraPosition);
        WorldObjectVisibilityContext context = new(
            CameraPosition: cameraPosition,
            CameraForward: cameraForward,
            FogEnd: 1600f,
            ObjectStreamingRangeMultiplier: 1.0f,
            CullSmallDoodadsOnly: false,
            CountAsTaxiActor: false,
            VerticalFieldOfViewRadians: MathF.PI / 4f,
            VisibilityProfile: WorldObjectVisibilityProfile.Quality);

        WorldVisibilityFrame visibility = new();
        HashSet<string> pendingAssetKeys = new(StringComparer.OrdinalIgnoreCase);

        Stopwatch totalStopwatch = Stopwatch.StartNew();

        Stopwatch wmoVisibilityStopwatch = Stopwatch.StartNew();
        int culledWmoCount = request.PassOptions.WmosVisible
            ? WorldObjectVisibilityCollector.CollectVisibleWmos(
                visibility,
                wmoInstances,
                context,
                static _ => false,
                static (_, _) => true,
                modelKey => assetReadyLookup.TryGetValue(modelKey, out bool ready) && ready,
                (modelKey, _) => pendingAssetKeys.Add(modelKey))
            : 0;
        wmoVisibilityStopwatch.Stop();

        Stopwatch mdxVisibilityStopwatch = Stopwatch.StartNew();
        int culledMdxCount = request.PassOptions.DoodadsVisible
            ? WorldObjectVisibilityCollector.CollectVisibleMdx(
                visibility,
                mdxInstances,
                context,
                static _ => false,
                static (_, _) => true,
                modelKey => assetReadyLookup.TryGetValue(modelKey, out bool ready) && ready,
                (modelKey, _) => pendingAssetKeys.Add(modelKey))
            : 0;
        mdxVisibilityStopwatch.Stop();

        WorldObjectPassFrame passFrame = new();
        int updatedMdxCount = 0;
        int renderedWmoCount = 0;
        int opaqueBatchedMdxCount = 0;
        int opaqueUnbatchedMdxCount = 0;
        int transparentBatchedMdxCount = 0;
        int transparentUnbatchedMdxCount = 0;
        double mdxAnimationMs = 0;
        double wdlMs = 0;
        double terrainMs = 0;
        double wmoSubmissionMs = 0;
        double mdxOpaqueSubmissionMs = 0;
        double liquidMs = 0;
        double mdxTransparentSortMs = 0;
        double mdxTransparentSubmissionMs = 0;
        int activeWdlTileCount = 0;
        int activeTerrainChunkCount = 0;
        int activeLiquidChunkCount = 0;
        int activeLiquidVisibleTileCount = 0;

        WorldFramePassOptions appliedPassOptions = new(
            objectsVisible: framePassOptions.ObjectsVisible && (visibility.VisibleWmos.Count > 0 || visibility.VisibleMdx.Count > 0),
            wmosVisible: framePassOptions.WmosVisible && visibility.VisibleWmos.Count > 0,
            doodadsVisible: framePassOptions.DoodadsVisible && visibility.VisibleMdx.Count > 0,
            skyVisible: framePassOptions.SkyVisible,
            wdlVisible: false,
            terrainVisible: framePassOptions.TerrainVisible,
            liquidVisible: framePassOptions.LiquidVisible,
            overlayVisible: framePassOptions.OverlayVisible);

        bool objectPhaseExecuted = WorldFramePassCoordinator.Execute(
            appliedPassOptions,
            new WorldFramePasses(
                static () => { },
                static () => { },
                static () => { },
                () =>
                {
                    Stopwatch wdlStopwatch = Stopwatch.StartNew();
                    activeWdlTileCount = wdlTileData.HasData ? 1 : 0;
                    wdlStopwatch.Stop();
                    wdlMs = wdlStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch terrainStopwatch = Stopwatch.StartNew();
                    activeTerrainChunkCount = activeTerrainTiles.Sum(static tile => tile.TerrainTileData.ChunkCount);
                    terrainStopwatch.Stop();
                    terrainMs = terrainStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch animationStopwatch = Stopwatch.StartNew();
                    updatedMdxCount = WorldObjectPassCoordinator.ExecuteVisibleMdxAnimation(passFrame, visibility, static _ => { });
                    animationStopwatch.Stop();
                    mdxAnimationMs = animationStopwatch.Elapsed.TotalMilliseconds;

                    WorldObjectPassCoordinator.PlanOpaqueMdxRoutes(
                        passFrame,
                        visibility,
                        static visible => visible.OpaqueFade < 0.999f || visible.TransparentFade < 0.999f,
                        static visible => visible.Instance.HasOpaqueRenderContent);

                    Stopwatch transparentSortStopwatch = Stopwatch.StartNew();
                    WorldObjectPassCoordinator.PlanTransparentMdxRoutes(
                        passFrame,
                        visibility,
                        static visible => visible.Instance.HasTransparentRenderContent && visible.TransparentFade > 0f);
                    transparentSortStopwatch.Stop();
                    mdxTransparentSortMs = transparentSortStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch wmoSubmissionStopwatch = Stopwatch.StartNew();
                    renderedWmoCount = WorldObjectPassCoordinator.ExecuteVisibleWmoOpaque(visibility, static _ => { });
                    wmoSubmissionStopwatch.Stop();
                    wmoSubmissionMs = wmoSubmissionStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch opaqueStopwatch = Stopwatch.StartNew();
                    (opaqueBatchedMdxCount, opaqueUnbatchedMdxCount) = WorldObjectPassCoordinator.ExecutePlannedOpaqueMdx(
                        passFrame,
                        visibility,
                        static _ => { },
                        static _ => { });
                    opaqueStopwatch.Stop();
                    mdxOpaqueSubmissionMs = opaqueStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch liquidStopwatch = Stopwatch.StartNew();
                    activeLiquidChunkCount = activeTerrainTiles.Sum(static tile => tile.LiquidTileData.ActiveChunkCount);
                    activeLiquidVisibleTileCount = activeTerrainTiles.Sum(static tile => tile.LiquidTileData.VisibleTileCount);
                    liquidStopwatch.Stop();
                    liquidMs = liquidStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch transparentStopwatch = Stopwatch.StartNew();
                    (transparentBatchedMdxCount, transparentUnbatchedMdxCount) = WorldObjectPassCoordinator.ExecutePlannedTransparentMdx(
                        passFrame,
                        visibility,
                        static _ => { },
                        static _ => { });
                    transparentStopwatch.Stop();
                    mdxTransparentSubmissionMs = transparentStopwatch.Elapsed.TotalMilliseconds;
                },
                static () => { }));

        totalStopwatch.Stop();

        WorldRenderFrameStats stats = new(
            TotalCpuMs: totalStopwatch.Elapsed.TotalMilliseconds,
            PendingAssetLoadCount: pendingAssetKeys.Count,
            TerrainChunksRendered: activeTerrainChunkCount,
            WdlVisibleTileCount: activeWdlTileCount,
            VisibleWmoCount: visibility.VisibleWmos.Count,
            VisibleMdxCount: visibility.VisibleMdx.Count,
            VisibleTaxiMdxCount: visibility.VisibleTaxiMdxCount,
            OpaqueBatchedMdxCount: opaqueBatchedMdxCount,
            OpaqueUnbatchedMdxCount: opaqueUnbatchedMdxCount,
            TransparentBatchedMdxCount: transparentBatchedMdxCount,
            TransparentUnbatchedMdxCount: transparentUnbatchedMdxCount,
            DeferredAssetLoads: new WorldRenderStageStats(0, pendingAssetKeys.Count, 0),
            TaxiActorUpdate: new WorldRenderStageStats(0, visibility.VisibleTaxiMdxCount, visibility.VisibleTaxiMdxCount),
            Lighting: new WorldRenderStageStats(0),
            Sky: new WorldRenderStageStats(0),
            SkyboxBackdrop: new WorldRenderStageStats(0, skyboxBackdropInstances.Count, framePassOptions.SkyVisible ? skyboxBackdropInstances.Count : 0),
            Wdl: new WorldRenderStageStats(wdlMs, activeWdlTileCount, activeWdlTileCount),
            Terrain: new WorldRenderStageStats(terrainMs, activeTerrainChunkCount, activeTerrainChunkCount),
            WmoVisibility: new WorldRenderStageStats(wmoVisibilityStopwatch.Elapsed.TotalMilliseconds, wmoInstances.Count, visibility.VisibleWmos.Count),
            WmoSubmission: new WorldRenderStageStats(wmoSubmissionMs, visibility.VisibleWmos.Count, renderedWmoCount),
            MdxAnimation: new WorldRenderStageStats(mdxAnimationMs, visibility.VisibleMdx.Count, updatedMdxCount),
            MdxVisibility: new WorldRenderStageStats(mdxVisibilityStopwatch.Elapsed.TotalMilliseconds, mdxInstances.Count, visibility.VisibleMdx.Count),
            MdxOpaqueSubmission: new WorldRenderStageStats(mdxOpaqueSubmissionMs, visibility.VisibleMdx.Count, opaqueBatchedMdxCount + opaqueUnbatchedMdxCount),
            Liquid: new WorldRenderStageStats(liquidMs, activeLiquidChunkCount, activeLiquidVisibleTileCount),
            MdxTransparentSort: new WorldRenderStageStats(mdxTransparentSortMs, visibility.VisibleMdx.Count, passFrame.TransparentVisibleMdxRoutes.Count),
            MdxTransparentSubmission: new WorldRenderStageStats(mdxTransparentSubmissionMs, passFrame.TransparentVisibleMdxRoutes.Count, transparentBatchedMdxCount + transparentUnbatchedMdxCount),
            Overlay: new WorldRenderStageStats(0));

        WorldMdxRenderPlan mdxRenderPlan = WorldMdxRenderPlanBuilder.Build(passFrame, visibility);
        WorldRenderCompositionFrame composition = WorldRenderCompositionBuilder.Build(
            appliedPassOptions,
            wdlTileData,
            terrainTileData,
            liquidTileData,
            wmoInstances.Count,
            mdxInstances.Count,
            stats,
            skyboxBackdropInstances.Count,
            tileStageSummary.TerrainChunkCount,
            tileStageSummary.LiquidChunkCount);

        return new WowViewerWorldRuntimeFrameResult(
            session,
            selectedTile.tileX,
            selectedTile.tileY,
            placementSourcePath,
            tileStageSummary,
            wdlTileData,
            terrainTileData,
            terrainVisualSnapshot,
            liquidTileData,
            activeTerrainTiles,
            placementCatalog,
            wmoInstances,
            mdxInstances,
            skyboxBackdropInstances,
            readyWmoCount,
            readyMdxCount,
            culledWmoCount,
            culledMdxCount,
            visibility,
            passFrame,
            mdxRenderPlan,
            appliedPassOptions,
            stats,
            composition,
            objectPhaseExecuted,
            WorldRenderOptimizationAdvisor.BuildHint(stats),
            pendingAssetKeys.OrderBy(static key => key, StringComparer.OrdinalIgnoreCase).ToArray(),
            cameraPosition,
            cameraTarget,
            cameraForward,
            planarMin,
            planarMax);
    }

    private static IReadOnlyList<WowViewerWorldRuntimeTileFrame> BuildActiveTerrainTiles(
        WowViewerWorldSessionBootstrapResult session,
        int selectedTileX,
        int selectedTileY,
        IArchiveCatalog archiveCatalog)
    {
        HashSet<(int TileX, int TileY)> occupiedTiles = session.OccupiedTiles
            .Select(static tile => (tile.TileX, tile.TileY))
            .ToHashSet();

        List<WowViewerWorldRuntimeTileFrame> activeTiles = [];
        for (int tileY = selectedTileY - 1; tileY <= selectedTileY + 1; tileY++)
        {
            for (int tileX = selectedTileX - 1; tileX <= selectedTileX + 1; tileX++)
            {
                if (tileX < 0 || tileX > 63 || tileY < 0 || tileY > 63)
                    continue;

                bool isSelectedTile = tileX == selectedTileX && tileY == selectedTileY;
                if (!isSelectedTile && occupiedTiles.Count > 0 && !occupiedTiles.Contains((tileX, tileY)))
                    continue;

                try
                {
                    WorldTileStageSummary tileStageSummary = ReadRootTileStageSummary(session, tileX, tileY, archiveCatalog, wdlVisibleTileCount: 0);
                    WorldTerrainTileData terrainTileData = ReadRootTerrainTileData(session, tileX, tileY, archiveCatalog);
                    WorldLiquidTileData liquidTileData = ReadRootLiquidTileData(session, tileX, tileY, archiveCatalog);
                    AdtPlacementCatalog placementCatalog = ReadPlacementCatalogOrEmpty(
                        session,
                        tileX,
                        tileY,
                        archiveCatalog,
                        tileStageSummary.SourcePath,
                        out string placementSourcePath);
                    activeTiles.Add(new WowViewerWorldRuntimeTileFrame(
                        tileX,
                        tileY,
                        placementSourcePath,
                        tileStageSummary,
                        terrainTileData,
                        liquidTileData,
                        placementCatalog));
                }
                catch (FileNotFoundException) when (!isSelectedTile)
                {
                }
            }
        }

        if (activeTiles.Count == 0)
            throw new InvalidDataException($"Selected tile ({selectedTileX},{selectedTileY}) did not produce any readable ADT terrain.");

        return activeTiles
            .OrderBy(static tile => tile.TileY)
            .ThenBy(static tile => tile.TileX)
            .ToArray();
    }

    private static WorldTileStageSummary BuildAggregateTileStageSummary(
        IReadOnlyList<WowViewerWorldRuntimeTileFrame> activeTerrainTiles,
        int selectedTileX,
        int selectedTileY)
    {
        if (activeTerrainTiles.Count == 1)
            return activeTerrainTiles[0].TileStageSummary;

        return new WorldTileStageSummary(
            $"3x3 ADT window centered on ({selectedTileX},{selectedTileY}); loaded {activeTerrainTiles.Count} terrain tiles",
            activeTerrainTiles[0].TileStageSummary.Kind,
            wdlVisibleTileCount: 0,
            activeTerrainTiles.Sum(static tile => tile.TileStageSummary.TerrainChunkCount),
            activeTerrainTiles.Sum(static tile => tile.TileStageSummary.TerrainHoleChunkCount),
            activeTerrainTiles.Sum(static tile => tile.TileStageSummary.LiquidChunkCount),
            activeTerrainTiles.Sum(static tile => tile.TileStageSummary.LiquidLayerCount),
            activeTerrainTiles.Sum(static tile => tile.TileStageSummary.VisibleLiquidTileCount),
            activeTerrainTiles.Any(static tile => tile.TileStageSummary.HasWater));
    }

    private static ((int tileX, int tileY) selectedTile, AdtPlacementCatalog placementCatalog, string placementSourcePath) ResolveTileAndPlacements(
        WowViewerWorldSessionBootstrapResult session,
        int requestedTileX,
        int requestedTileY,
        IArchiveCatalog archiveCatalog)
    {
        if (requestedTileX >= 0 && requestedTileY >= 0)
        {
            WorldTileStageSummary tileStageSummary = ReadRootTileStageSummary(session, requestedTileX, requestedTileY, archiveCatalog, wdlVisibleTileCount: 0);
            AdtPlacementCatalog requestedCatalog = ReadPlacementCatalogOrEmpty(
                session,
                requestedTileX,
                requestedTileY,
                archiveCatalog,
                tileStageSummary.SourcePath,
                out string requestedSourcePath);
            return ((requestedTileX, requestedTileY), requestedCatalog, requestedSourcePath);
        }

        foreach (WdtTileCoordinate tile in OrderAutoTileCandidates(session.OccupiedTiles))
        {
            try
            {
                WorldTileStageSummary tileStageSummary = ReadRootTileStageSummary(session, tile.TileX, tile.TileY, archiveCatalog, wdlVisibleTileCount: 0);
                AdtPlacementCatalog catalog = ReadPlacementCatalogOrEmpty(
                    session,
                    tile.TileX,
                    tile.TileY,
                    archiveCatalog,
                    tileStageSummary.SourcePath,
                    out string sourcePath);
                return ((tile.TileX, tile.TileY), catalog, sourcePath);
            }
            catch (FileNotFoundException)
            {
            }
        }

        throw new InvalidDataException($"Map '{session.ResolvedMapDirectory}' does not report any readable occupied WDT tiles.");
    }

    private static IEnumerable<WdtTileCoordinate> OrderAutoTileCandidates(IReadOnlyList<WdtTileCoordinate> occupiedTiles)
    {
        if (occupiedTiles.Count == 0)
            yield break;

        float centerTileX = occupiedTiles.Average(static tile => tile.TileX + 0.5f);
        float centerTileY = occupiedTiles.Average(static tile => tile.TileY + 0.5f);

        foreach (WdtTileCoordinate tile in occupiedTiles
                     .OrderBy(tile => ComputeTileDistanceSq(tile, centerTileX, centerTileY))
                     .ThenBy(static tile => tile.TileY)
                     .ThenBy(static tile => tile.TileX))
        {
            yield return tile;
        }
    }

    private static float ComputeTileDistanceSq(WdtTileCoordinate tile, float centerTileX, float centerTileY)
    {
        float dx = (tile.TileX + 0.5f) - centerTileX;
        float dy = (tile.TileY + 0.5f) - centerTileY;
        return (dx * dx) + (dy * dy);
    }

    private static AdtPlacementCatalog ReadPlacementCatalog(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        out string sourcePath)
    {
        string mapDirectory = session.ResolvedMapDirectory;
        string objVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY, "_obj0.adt");
        if (TryReadVirtualOrLooseFile(session.ClientRoot, session.LooseOverlayRoot, objVirtualPath, archiveCatalog, out byte[]? objData, out sourcePath))
            return ReadPlacementCatalogFromBytes(objData!, sourcePath);

        string rootVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY);
        if (TryReadVirtualOrLooseFile(session.ClientRoot, session.LooseOverlayRoot, rootVirtualPath, archiveCatalog, out byte[]? rootData, out sourcePath))
            return ReadPlacementCatalogFromBytes(rootData!, sourcePath);

        if (AlphaEmbeddedAdtReader.TryReadPlacementCatalog(session.ClientRoot, mapDirectory, tileX, tileY, archiveCatalog, out AdtPlacementCatalog? alphaCatalog, out string alphaSourcePath))
        {
            sourcePath = alphaSourcePath;
            return alphaCatalog!;
        }

        throw new FileNotFoundException($"Could not locate placement ADT for map '{mapDirectory}' tile ({tileX},{tileY}).", rootVirtualPath);
    }

    private static AdtPlacementCatalog ReadPlacementCatalogOrEmpty(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        string fallbackSourcePath,
        out string sourcePath)
    {
        try
        {
            return ReadPlacementCatalog(session, tileX, tileY, archiveCatalog, out sourcePath);
        }
        catch (FileNotFoundException)
        {
            sourcePath = fallbackSourcePath;
            return new AdtPlacementCatalog(
                fallbackSourcePath,
                MapFileKind.Adt,
                [],
                [],
                [],
                []);
        }
    }

    private static AdtPlacementCatalog ReadPlacementCatalogFromBytes(byte[] data, string sourcePath)
    {
        using MemoryStream stream = new(data, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        stream.Position = 0;
        return AdtPlacementReader.Read(stream, fileSummary);
    }

    private static WorldTileStageSummary ReadRootTileStageSummary(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        int wdlVisibleTileCount)
    {
        string mapDirectory = session.ResolvedMapDirectory;
        string rootVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY);
        if (!TryReadVirtualOrLooseFile(session.ClientRoot, session.LooseOverlayRoot, rootVirtualPath, archiveCatalog, out byte[]? rootData, out string sourcePath) || rootData is null)
        {
            if (AlphaEmbeddedAdtReader.TryReadTile(session.ClientRoot, mapDirectory, tileX, tileY, archiveCatalog, out AlphaEmbeddedAdtTileData? alphaTile))
                return new WorldTileStageSummary(
                    alphaTile.TileStageSummary.SourcePath,
                    alphaTile.TileStageSummary.Kind,
                    wdlVisibleTileCount,
                    alphaTile.TileStageSummary.TerrainChunkCount,
                    alphaTile.TileStageSummary.TerrainHoleChunkCount,
                    alphaTile.TileStageSummary.LiquidChunkCount,
                    alphaTile.TileStageSummary.LiquidLayerCount,
                    alphaTile.TileStageSummary.VisibleLiquidTileCount,
                    alphaTile.TileStageSummary.HasWater);

            throw new FileNotFoundException($"Could not locate root ADT for map '{mapDirectory}' tile ({tileX},{tileY}).", rootVirtualPath);
        }

        using MemoryStream stream = new(rootData, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        stream.Position = 0;
        return WorldTileStageSummaryBuilder.Read(stream, fileSummary, wdlVisibleTileCount);
    }

    private static WorldLiquidTileData ReadRootLiquidTileData(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog)
    {
        string mapDirectory = session.ResolvedMapDirectory;
        string rootVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY);
        if (!TryReadVirtualOrLooseFile(session.ClientRoot, session.LooseOverlayRoot, rootVirtualPath, archiveCatalog, out byte[]? rootData, out string sourcePath) || rootData is null)
        {
            if (AlphaEmbeddedAdtReader.TryReadTile(session.ClientRoot, mapDirectory, tileX, tileY, archiveCatalog, out AlphaEmbeddedAdtTileData? alphaTile))
                return alphaTile.LiquidTileData;

            throw new FileNotFoundException($"Could not locate root ADT for map '{mapDirectory}' tile ({tileX},{tileY}).", rootVirtualPath);
        }

        using MemoryStream stream = new(rootData, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        stream.Position = 0;
        return WorldLiquidTileBuilder.Read(stream, fileSummary);
    }

    private static WorldTerrainTileData ReadRootTerrainTileData(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog)
    {
        string mapDirectory = session.ResolvedMapDirectory;
        string rootVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY);
        if (!TryReadVirtualOrLooseFile(session.ClientRoot, session.LooseOverlayRoot, rootVirtualPath, archiveCatalog, out byte[]? rootData, out string sourcePath) || rootData is null)
        {
            if (AlphaEmbeddedAdtReader.TryReadTile(session.ClientRoot, mapDirectory, tileX, tileY, archiveCatalog, out AlphaEmbeddedAdtTileData? alphaTile))
                return alphaTile.TerrainTileData;

            throw new FileNotFoundException($"Could not locate root ADT for map '{mapDirectory}' tile ({tileX},{tileY}).", rootVirtualPath);
        }

        using MemoryStream stream = new(rootData, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        AdtTextureFile? textureFile = TryReadTerrainTextureFile(session, tileX, tileY, archiveCatalog, mapDirectory, rootData, sourcePath, fileSummary);
        stream.Position = 0;
        return WorldTerrainTileBuilder.Read(stream, fileSummary, textureFile);
    }

    private static AdtTextureFile? TryReadTerrainTextureFile(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        string mapDirectory,
        byte[] rootData,
        string rootSourcePath,
        MapFileSummary rootSummary)
    {
        string texVirtualPath = BuildStandardAdtVirtualPath(mapDirectory, tileX, tileY, "_tex0.adt");
        if (TryReadVirtualOrLooseFile(session.ClientRoot, session.LooseOverlayRoot, texVirtualPath, archiveCatalog, out byte[]? texData, out string texSourcePath)
            && texData is { Length: > 0 })
        {
            using MemoryStream texStream = new(texData, writable: false);
            MapFileSummary texSummary = MapFileSummaryReader.Read(texStream, texSourcePath);
            texStream.Position = 0;
            return AdtTextureReader.Read(texStream, texSummary);
        }

        if (!rootSummary.HasChunk(MapChunkIds.Mtex))
            return null;

        using MemoryStream rootStream = new(rootData, writable: false);
        MapFileSummary inlineTextureSummary = MapFileSummaryReader.Read(rootStream, rootSourcePath);
        rootStream.Position = 0;
        return AdtTextureReader.Read(rootStream, inlineTextureSummary);
    }

    private static string BuildStandardAdtVirtualPath(string mapDirectory, int tileX, int tileY, string suffix = ".adt")
    {
        // Match MdxViewer's row-major convention: tileX is row (y), tileY is column (x),
        // while on-disk ADT families are named Map_x_y.
        return $@"World\Maps\{mapDirectory}\{mapDirectory}_{tileY}_{tileX}{suffix}";
    }

    private static List<WorldObjectInstance> BuildWmoInstances(
        AdtPlacementCatalog placementCatalog,
        int tileX,
        int tileY,
        Dictionary<string, bool> assetReadyLookup)
    {
        List<WorldObjectInstance> instances = new(placementCatalog.WorldModelPlacements.Count);
        for (int index = 0; index < placementCatalog.WorldModelPlacements.Count; index++)
        {
            AdtWorldModelPlacement placement = placementCatalog.WorldModelPlacements[index];
            string modelKey = NormalizeModelKey(placement.ModelPath);
            assetReadyLookup[modelKey] = true;

            // The current world preview only renders terrain plus object markers, so use the
            // placement bounds already carried by the ADT catalog instead of parsing each WMO.
            Vector3 localMin = placement.BoundsMin - placement.Position;
            Vector3 localMax = placement.BoundsMax - placement.Position;

            instances.Add(new WorldObjectInstance
            {
                ModelKey = modelKey,
                AssetKind = "WMO",
                ModelName = Path.GetFileName(modelKey),
                ModelPath = modelKey,
                UniqueId = placement.UniqueId,
                PlacementEntryIndex = index,
                PlacementPosition = placement.Position,
                PlacementRotation = placement.Rotation,
                PlacementScale = 1.0f,
                Transform = Matrix4x4.CreateTranslation(placement.Position),
                BoundsMin = placement.BoundsMin,
                BoundsMax = placement.BoundsMax,
                LocalBoundsMin = localMin,
                LocalBoundsMax = localMax,
                BoundsResolved = true,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true,
                HasOpaqueRenderContent = true,
                HasTransparentRenderContent = false,
                WmoVersion = null,
                WmoGroupCount = 0,
                WmoPortalCount = 0,
                WmoDoodadSetCount = 0,
                WmoDoodadMdxCount = 0,
                WmoDoodadM2Count = 0,
                WmoDoodadUnknownCount = 0,
            });
        }

        return instances;
    }

    private static List<WorldObjectInstance> BuildMdxInstances(
        AdtPlacementCatalog placementCatalog,
        int tileX,
        int tileY,
        Dictionary<string, bool> assetReadyLookup)
    {
        List<WorldObjectInstance> instances = new(placementCatalog.ModelPlacements.Count);
        for (int index = 0; index < placementCatalog.ModelPlacements.Count; index++)
        {
            AdtModelPlacement placement = placementCatalog.ModelPlacements[index];
            string modelKey = NormalizeModelKey(placement.ModelPath);
            // Keep the current marker-only world preview fast by using a cheap placement-scaled
            // fallback extent instead of parsing each MDX or M2 just to recover bounds.
            LocalBoundsResolution resolution = CreateFallbackBounds(assetReady: true, scale: placement.Scale);
            assetReadyLookup[modelKey] = true;

            Matrix4x4 transform = BuildLegacyMdxPlacementTransform(placement.Position, placement.Rotation, placement.Scale);
            (Vector3 worldMin, Vector3 worldMax) = TransformBounds(resolution.LocalMin, resolution.LocalMax, transform);

            instances.Add(new WorldObjectInstance
            {
                ModelKey = modelKey,
                AssetKind = Path.GetExtension(modelKey).Equals(".m2", StringComparison.OrdinalIgnoreCase) ? "M2" : "MDX",
                ModelName = Path.GetFileName(modelKey),
                ModelPath = modelKey,
                UniqueId = placement.UniqueId,
                PlacementEntryIndex = index,
                PlacementPosition = placement.Position,
                PlacementRotation = placement.Rotation,
                PlacementScale = placement.Scale,
                Transform = transform,
                BoundsMin = worldMin,
                BoundsMax = worldMax,
                LocalBoundsMin = resolution.LocalMin,
                LocalBoundsMax = resolution.LocalMax,
                BoundsResolved = false,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true,
                HasOpaqueRenderContent = resolution.HasOpaqueRenderContent,
                HasTransparentRenderContent = resolution.HasTransparentRenderContent,
            });
        }

        return instances;
    }

    private static ResolvedWmoAsset ResolveWmoAsset(
        string clientRoot,
        string looseOverlayRoot,
        string modelKey,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, ResolvedWmoAsset> wmoCache,
        Dictionary<string, bool> assetReadyLookup)
    {
        if (wmoCache.TryGetValue(modelKey, out ResolvedWmoAsset cached))
            return cached;

        ResolvedWmoAsset resolved = TryResolveWmoAsset(clientRoot, looseOverlayRoot, modelKey, archiveCatalog, assetReadyLookup);
        wmoCache[modelKey] = resolved;
        return resolved;
    }

    private static ResolvedWmoAsset TryResolveWmoAsset(
        string clientRoot,
        string looseOverlayRoot,
        string modelKey,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, bool> assetReadyLookup)
    {
        if (!TryReadVirtualOrLooseFile(clientRoot, looseOverlayRoot, modelKey, archiveCatalog, out byte[]? data, out string sourcePath) || data is null)
            return CreateFallbackWmoAsset(assetReady: false);

        try
        {
            using MemoryStream stream = new(data, writable: false);
            WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, sourcePath);
            RegisterEmbeddedDoodadAssetReadiness(document, clientRoot, looseOverlayRoot, archiveCatalog, assetReadyLookup);
            return new ResolvedWmoAsset(
                AssetReady: true,
                LocalMin: document.Summary.BoundsMin,
                LocalMax: document.Summary.BoundsMax,
                Version: document.Version,
                GroupCount: document.Groups.Count,
                PortalCount: document.Portals.Count,
                DoodadSetCount: document.DoodadSets.Count,
                MdxDoodadCount: document.DoodadPlacements.Count(static placement => placement.ModelKind == WmoDoodadModelKind.Mdx),
                M2DoodadCount: document.DoodadPlacements.Count(static placement => placement.ModelKind == WmoDoodadModelKind.M2),
                UnknownDoodadCount: document.DoodadPlacements.Count(static placement => placement.ModelKind == WmoDoodadModelKind.Unknown));
        }
        catch
        {
            return CreateFallbackWmoAsset(assetReady: true);
        }
    }

    private static LocalBoundsResolution ResolveLocalBounds(
        string clientRoot,
        string looseOverlayRoot,
        string modelKey,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, LocalBoundsResolution> boundsCache)
    {
        if (boundsCache.TryGetValue(modelKey, out LocalBoundsResolution cached))
            return cached;

        LocalBoundsResolution resolved = TryResolveLocalBounds(clientRoot, looseOverlayRoot, modelKey, archiveCatalog);
        boundsCache[modelKey] = resolved;
        return resolved;
    }

    private static LocalBoundsResolution TryResolveLocalBounds(string clientRoot, string looseOverlayRoot, string modelKey, IArchiveCatalog archiveCatalog)
    {
        if (!TryReadVirtualOrLooseFile(clientRoot, looseOverlayRoot, modelKey, archiveCatalog, out byte[]? data, out string sourcePath) || data is null)
            return CreateFallbackBounds(assetReady: false, scale: 1.0f);

        string extension = Path.GetExtension(modelKey);
        try
        {
            if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
            {
                using MemoryStream m2Stream = new(data, writable: false);
                M2ModelDocument document = M2ModelReader.Read(m2Stream, sourcePath);
                return new LocalBoundsResolution(true, document.BoundsMin, document.BoundsMax, HasOpaqueRenderContent: true, HasTransparentRenderContent: true);
            }

            using MemoryStream mdxStream = new(data, writable: false);
            MdxSummary summary = MdxSummaryReader.Read(mdxStream, sourcePath);
            if (summary.BoundsMin is Vector3 boundsMin && summary.BoundsMax is Vector3 boundsMax)
            {
                MdxRenderCharacteristics characteristics = MdxRenderCharacteristicsAnalyzer.Analyze(summary);
                return new LocalBoundsResolution(true, boundsMin, boundsMax, characteristics.HasOpaqueRenderContent, characteristics.HasTransparentRenderContent);
            }
        }
        catch
        {
        }

        try
        {
            using MemoryStream fallbackM2Stream = new(data, writable: false);
            M2ModelDocument document = M2ModelReader.Read(fallbackM2Stream, sourcePath);
            return new LocalBoundsResolution(true, document.BoundsMin, document.BoundsMax, HasOpaqueRenderContent: true, HasTransparentRenderContent: true);
        }
        catch
        {
        }

        return CreateFallbackBounds(assetReady: false, scale: 1.0f);
    }

    private static LocalBoundsResolution CreateFallbackBounds(bool assetReady, float scale)
    {
        float halfExtent = MathF.Max(2f, 4f * MathF.Max(1f, scale));
        Vector3 extent = new(halfExtent, halfExtent, halfExtent);
        return new LocalBoundsResolution(assetReady, -extent, extent, HasOpaqueRenderContent: true, HasTransparentRenderContent: true);
    }

    private static ResolvedWmoAsset CreateFallbackWmoAsset(bool assetReady)
    {
        Vector3 extent = new(8f, 8f, 8f);
        return new ResolvedWmoAsset(
            AssetReady: assetReady,
            LocalMin: -extent,
            LocalMax: extent,
            Version: null,
            GroupCount: 0,
            PortalCount: 0,
            DoodadSetCount: 0,
            MdxDoodadCount: 0,
            M2DoodadCount: 0,
            UnknownDoodadCount: 0);
    }

    private static void RegisterEmbeddedDoodadAssetReadiness(
        WmoRenderDocument document,
        string clientRoot,
        string looseOverlayRoot,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, bool> assetReadyLookup)
    {
        foreach (string doodadKey in document.DoodadPlacements
            .Select(static placement => placement.ModelPath)
            .Where(static path => !string.IsNullOrWhiteSpace(path))
            .Select(NormalizeModelKey)
            .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (assetReadyLookup.ContainsKey(doodadKey))
                continue;

            assetReadyLookup[doodadKey] = TryReadVirtualOrLooseFile(clientRoot, looseOverlayRoot, doodadKey, archiveCatalog, out _, out _);
        }
    }

    private static bool TryReadVirtualOrLooseFile(
        string clientRoot,
        string looseOverlayRoot,
        string virtualPath,
        IArchiveCatalog archiveCatalog,
        out byte[]? data,
        out string sourcePath)
    {
        if (VirtualAssetOverlayResolver.TryReadLooseVirtualFile(virtualPath, looseOverlayRoot, out data, out sourcePath))
            return true;

        return AlphaEmbeddedAdtReader.TryReadVirtualOrLooseFile(clientRoot, virtualPath, archiveCatalog, out data, out sourcePath);
    }

    private static string NormalizeModelKey(string modelPath)
    {
        return modelPath.Trim().Replace('/', '\\');
    }

    private static Matrix4x4 BuildLegacyMdxPlacementTransform(Vector3 position, Vector3 rotationDegrees, float scale)
    {
        // Match the old MdxViewer world MDX placement path rather than a generic Euler helper.
        float rx = -DegreesToRadians(rotationDegrees.Y);
        float ry = -DegreesToRadians(rotationDegrees.X);
        float rz = DegreesToRadians(rotationDegrees.Z);
        return Matrix4x4.CreateRotationZ(MathF.PI)
            * Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateRotationX(rx)
            * Matrix4x4.CreateRotationY(ry)
            * Matrix4x4.CreateRotationZ(rz)
            * Matrix4x4.CreateTranslation(position);
    }

    private static float DegreesToRadians(float degrees)
    {
        return degrees * (MathF.PI / 180f);
    }

    private static (Vector3 Min, Vector3 Max) TransformBounds(Vector3 localMin, Vector3 localMax, Matrix4x4 transform)
    {
        Vector3[] corners =
        [
            new Vector3(localMin.X, localMin.Y, localMin.Z),
            new Vector3(localMin.X, localMin.Y, localMax.Z),
            new Vector3(localMin.X, localMax.Y, localMin.Z),
            new Vector3(localMin.X, localMax.Y, localMax.Z),
            new Vector3(localMax.X, localMin.Y, localMin.Z),
            new Vector3(localMax.X, localMin.Y, localMax.Z),
            new Vector3(localMax.X, localMax.Y, localMin.Z),
            new Vector3(localMax.X, localMax.Y, localMax.Z),
        ];

        Vector3 transformedMin = Vector3.Transform(corners[0], transform);
        Vector3 transformedMax = transformedMin;
        for (int index = 1; index < corners.Length; index++)
        {
            Vector3 transformed = Vector3.Transform(corners[index], transform);
            transformedMin = Vector3.Min(transformedMin, transformed);
            transformedMax = Vector3.Max(transformedMax, transformed);
        }

        return (transformedMin, transformedMax);
    }

    private static (Vector3 FocusCenter, Vector2 PlanarMin, Vector2 PlanarMax) ComputeWorldViewBounds(
        IReadOnlyList<WorldObjectInstance> wmoInstances,
        IReadOnlyList<WorldObjectInstance> mdxInstances,
        int tileX,
        int tileY)
    {
        List<WorldObjectInstance> instances = [.. wmoInstances, .. mdxInstances];
        if (instances.Count == 0)
        {
            Vector3 fallbackCenter = ComputeTileCenter(tileX, tileY, 0f);
            Vector2 fallbackMin = ComputeTilePlanarMin(tileX, tileY);
            Vector2 fallbackMax = ComputeTilePlanarMax(tileX, tileY);
            return (fallbackCenter, fallbackMin, fallbackMax);
        }

        Vector3 worldMin = instances[0].BoundsMin;
        Vector3 worldMax = instances[0].BoundsMax;
        for (int index = 1; index < instances.Count; index++)
        {
            worldMin = Vector3.Min(worldMin, instances[index].BoundsMin);
            worldMax = Vector3.Max(worldMax, instances[index].BoundsMax);
        }

        Vector3 focusCenter = (worldMin + worldMax) * 0.5f;
        Vector2 planarMin = new(worldMin.X, worldMin.Y);
        Vector2 planarMax = new(worldMax.X, worldMax.Y);
        if (Vector2.DistanceSquared(planarMin, planarMax) < 1f)
        {
            planarMin -= new Vector2(32f, 32f);
            planarMax += new Vector2(32f, 32f);
        }

        return (focusCenter, planarMin, planarMax);
    }

    internal static Vector2 ComputeTilePlanarMin(int tileX, int tileY)
    {
        float minX = MapOrigin - ((tileY + 1) * TileSize);
        float minY = MapOrigin - ((tileX + 1) * TileSize);
        return new Vector2(minX, minY);
    }

    internal static Vector2 ComputeTilePlanarMax(int tileX, int tileY)
    {
        float maxX = MapOrigin - (tileY * TileSize);
        float maxY = MapOrigin - (tileX * TileSize);
        return new Vector2(maxX, maxY);
    }

    internal static Vector3 ComputeTileCenter(int tileX, int tileY, float height)
    {
        Vector2 min = ComputeTilePlanarMin(tileX, tileY);
        Vector2 max = ComputeTilePlanarMax(tileX, tileY);
        return new Vector3((min.X + max.X) * 0.5f, (min.Y + max.Y) * 0.5f, height);
    }

    private static Vector3 ComputeSpawnCameraTarget(
        int tileX,
        int tileY,
        WorldTerrainTileData terrainTileData,
        Vector3 fallbackTarget)
    {
        float? height = terrainTileData.Heightmap?.CenterHeight;
        return height.HasValue ? ComputeTileCenter(tileX, tileY, height.Value) : fallbackTarget;
    }
}
