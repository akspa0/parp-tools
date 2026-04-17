using System.Diagnostics;
using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.Maps;
using WowViewer.Core.Mdx;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.App;

internal sealed record WowViewerWorldRuntimeFrameRequest(
    string ClientRoot,
    string MapInput,
    string BuildLabel,
    int TileX,
    int TileY,
    WorldFramePassOptions PassOptions);

internal sealed class WowViewerWorldRuntimeFrameResult
{
    public WowViewerWorldRuntimeFrameResult(
        WowViewerWorldSessionBootstrapResult session,
        int selectedTileX,
        int selectedTileY,
        string placementSourcePath,
        WorldTileStageSummary tileStageSummary,
        AdtPlacementCatalog placementCatalog,
        IReadOnlyList<WorldObjectInstance> wmoInstances,
        IReadOnlyList<WorldObjectInstance> mdxInstances,
        int readyWmoCount,
        int readyMdxCount,
        int culledWmoCount,
        int culledMdxCount,
        WorldVisibilityFrame visibility,
        WorldObjectPassFrame passFrame,
        WorldFramePassOptions passOptions,
        WorldRenderFrameStats stats,
        bool objectPhaseExecuted,
        string optimizationHint,
        IReadOnlyList<string> pendingAssetKeys,
        Vector3 cameraPosition,
        Vector3 cameraForward,
        Vector2 planarMin,
        Vector2 planarMax)
    {
        Session = session;
        SelectedTileX = selectedTileX;
        SelectedTileY = selectedTileY;
        PlacementSourcePath = placementSourcePath;
        TileStageSummary = tileStageSummary;
        PlacementCatalog = placementCatalog;
        WmoInstances = wmoInstances;
        MdxInstances = mdxInstances;
        ReadyWmoCount = readyWmoCount;
        ReadyMdxCount = readyMdxCount;
        CulledWmoCount = culledWmoCount;
        CulledMdxCount = culledMdxCount;
        Visibility = visibility;
        PassFrame = passFrame;
        PassOptions = passOptions;
        Stats = stats;
        ObjectPhaseExecuted = objectPhaseExecuted;
        OptimizationHint = optimizationHint;
        PendingAssetKeys = pendingAssetKeys;
        CameraPosition = cameraPosition;
        CameraForward = cameraForward;
        PlanarMin = planarMin;
        PlanarMax = planarMax;
    }

    public WowViewerWorldSessionBootstrapResult Session { get; }

    public int SelectedTileX { get; }

    public int SelectedTileY { get; }

    public string PlacementSourcePath { get; }

    public WorldTileStageSummary TileStageSummary { get; }

    public AdtPlacementCatalog PlacementCatalog { get; }

    public IReadOnlyList<WorldObjectInstance> WmoInstances { get; }

    public IReadOnlyList<WorldObjectInstance> MdxInstances { get; }

    public int ReadyWmoCount { get; }

    public int ReadyMdxCount { get; }

    public int CulledWmoCount { get; }

    public int CulledMdxCount { get; }

    public WorldVisibilityFrame Visibility { get; }

    public WorldObjectPassFrame PassFrame { get; }

    public WorldFramePassOptions PassOptions { get; }

    public WorldRenderFrameStats Stats { get; }

    public bool ObjectPhaseExecuted { get; }

    public string OptimizationHint { get; }

    public IReadOnlyList<string> PendingAssetKeys { get; }

    public Vector3 CameraPosition { get; }

    public Vector3 CameraForward { get; }

    public Vector2 PlanarMin { get; }

    public Vector2 PlanarMax { get; }
}

internal static class WowViewerWorldRuntimeBridge
{
    private readonly record struct LocalBoundsResolution(bool AssetReady, Vector3 LocalMin, Vector3 LocalMax);

    public static WowViewerWorldRuntimeFrameResult Build(WowViewerWorldRuntimeFrameRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);

        WowViewerWorldSessionBootstrapResult session = WowViewerWorldSessionBootstrapper.Open(
            new WowViewerWorldSessionOpenRequest(request.ClientRoot, request.MapInput, request.BuildLabel));

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapper.Bootstrap(archiveCatalog, [session.ClientRoot], new ArchiveCatalogBootstrapOptions());

        ((int tileX, int tileY) selectedTile, AdtPlacementCatalog placementCatalog, string placementSourcePath) =
            ResolveTileAndPlacements(session, request.TileX, request.TileY, archiveCatalog);
        WorldTileStageSummary tileStageSummary = ReadRootTileStageSummary(session, selectedTile.tileX, selectedTile.tileY, archiveCatalog);

        Dictionary<string, LocalBoundsResolution> boundsCache = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, bool> assetReadyLookup = new(StringComparer.OrdinalIgnoreCase);

        List<WorldObjectInstance> wmoInstances = BuildWmoInstances(placementCatalog, selectedTile.tileX, selectedTile.tileY, archiveCatalog, assetReadyLookup);
        List<WorldObjectInstance> mdxInstances = BuildMdxInstances(session.ClientRoot, placementCatalog, selectedTile.tileX, selectedTile.tileY, archiveCatalog, boundsCache, assetReadyLookup);

        int readyWmoCount = wmoInstances.Count(instance => assetReadyLookup.TryGetValue(instance.ModelKey, out bool ready) && ready);
        int readyMdxCount = mdxInstances.Count(instance => assetReadyLookup.TryGetValue(instance.ModelKey, out bool ready) && ready);

        (Vector3 focusCenter, Vector2 planarMin, Vector2 planarMax) = ComputeWorldViewBounds(wmoInstances, mdxInstances, selectedTile.tileX, selectedTile.tileY);
        Vector3 cameraPosition = focusCenter + new Vector3(-700f, -700f, 260f);
        Vector3 cameraForward = Vector3.Normalize(focusCenter - cameraPosition);
        WorldObjectVisibilityContext context = new(
            CameraPosition: cameraPosition,
            CameraForward: cameraForward,
            FogEnd: 1600f,
            ObjectStreamingRangeMultiplier: 1.0f,
            CullSmallDoodadsOnly: false,
            CountAsTaxiActor: false,
            VerticalFieldOfViewRadians: MathF.PI / 3f,
            VisibilityProfile: WorldObjectVisibilityProfile.Balanced);

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
            objectsVisible: request.PassOptions.ObjectsVisible && (visibility.VisibleWmos.Count > 0 || visibility.VisibleMdx.Count > 0),
            wmosVisible: request.PassOptions.WmosVisible && visibility.VisibleWmos.Count > 0,
            doodadsVisible: request.PassOptions.DoodadsVisible && visibility.VisibleMdx.Count > 0,
            skyVisible: request.PassOptions.SkyVisible,
            wdlVisible: request.PassOptions.WdlVisible,
            terrainVisible: request.PassOptions.TerrainVisible,
            liquidVisible: request.PassOptions.LiquidVisible,
            overlayVisible: request.PassOptions.OverlayVisible);

        bool objectPhaseExecuted = WorldFramePassCoordinator.Execute(
            appliedPassOptions,
            new WorldFramePasses(
                static () => { },
                static () => { },
                static () => { },
                () =>
                {
                    Stopwatch wdlStopwatch = Stopwatch.StartNew();
                    activeWdlTileCount = tileStageSummary.WdlVisibleTileCount;
                    wdlStopwatch.Stop();
                    wdlMs = wdlStopwatch.Elapsed.TotalMilliseconds;
                },
                () =>
                {
                    Stopwatch terrainStopwatch = Stopwatch.StartNew();
                    activeTerrainChunkCount = tileStageSummary.TerrainChunkCount;
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
                        static visible => visible.OpaqueFade < 0.999f || visible.TransparentFade < 0.999f);

                    Stopwatch transparentSortStopwatch = Stopwatch.StartNew();
                    WorldObjectPassCoordinator.PlanTransparentMdxRoutes(
                        passFrame,
                        visibility,
                        static visible => visible.TransparentFade > 0f);
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
                    activeLiquidChunkCount = tileStageSummary.LiquidChunkCount;
                    activeLiquidVisibleTileCount = tileStageSummary.VisibleLiquidTileCount;
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
            SkyboxBackdrop: new WorldRenderStageStats(0),
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

        return new WowViewerWorldRuntimeFrameResult(
            session,
            selectedTile.tileX,
            selectedTile.tileY,
            placementSourcePath,
            tileStageSummary,
            placementCatalog,
            wmoInstances,
            mdxInstances,
            readyWmoCount,
            readyMdxCount,
            culledWmoCount,
            culledMdxCount,
            visibility,
            passFrame,
            appliedPassOptions,
            stats,
            objectPhaseExecuted,
            WorldRenderOptimizationAdvisor.BuildHint(stats),
            pendingAssetKeys.OrderBy(static key => key, StringComparer.OrdinalIgnoreCase).ToArray(),
            cameraPosition,
            cameraForward,
            planarMin,
            planarMax);
    }

    private static ((int tileX, int tileY) selectedTile, AdtPlacementCatalog placementCatalog, string placementSourcePath) ResolveTileAndPlacements(
        WowViewerWorldSessionBootstrapResult session,
        int requestedTileX,
        int requestedTileY,
        IArchiveCatalog archiveCatalog)
    {
        if (requestedTileX >= 0 && requestedTileY >= 0)
        {
            AdtPlacementCatalog requestedCatalog = ReadPlacementCatalog(session, requestedTileX, requestedTileY, archiveCatalog, out string requestedSourcePath);
            return ((requestedTileX, requestedTileY), requestedCatalog, requestedSourcePath);
        }

        (int tileX, int tileY)? bestTile = null;
        AdtPlacementCatalog? bestCatalog = null;
        string? bestSourcePath = null;
        int bestPlacementCount = -1;

        foreach (WdtTileCoordinate tile in session.OccupiedTiles)
        {
            AdtPlacementCatalog catalog = ReadPlacementCatalog(session, tile.TileX, tile.TileY, archiveCatalog, out string sourcePath);
            int placementCount = catalog.ModelPlacements.Count + catalog.WorldModelPlacements.Count;
            if (placementCount <= bestPlacementCount)
                continue;

            bestTile = (tile.TileX, tile.TileY);
            bestCatalog = catalog;
            bestSourcePath = sourcePath;
            bestPlacementCount = placementCount;
        }

        if (bestTile.HasValue && bestCatalog is not null && bestSourcePath is not null)
            return (bestTile.Value, bestCatalog, bestSourcePath);

        throw new InvalidDataException($"Map '{session.ResolvedMapDirectory}' does not report any occupied WDT tiles.");
    }

    private static AdtPlacementCatalog ReadPlacementCatalog(
        WowViewerWorldSessionBootstrapResult session,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        out string sourcePath)
    {
        string mapDirectory = session.ResolvedMapDirectory;
        string objVirtualPath = $@"World\Maps\{mapDirectory}\{mapDirectory}_{tileX}_{tileY}_obj0.adt";
        if (TryReadVirtualOrLooseFile(session.ClientRoot, objVirtualPath, archiveCatalog, out byte[]? objData, out sourcePath))
            return ReadPlacementCatalogFromBytes(objData!, sourcePath);

        string rootVirtualPath = $@"World\Maps\{mapDirectory}\{mapDirectory}_{tileX}_{tileY}.adt";
        if (TryReadVirtualOrLooseFile(session.ClientRoot, rootVirtualPath, archiveCatalog, out byte[]? rootData, out sourcePath))
            return ReadPlacementCatalogFromBytes(rootData!, sourcePath);

        throw new FileNotFoundException($"Could not locate placement ADT for map '{mapDirectory}' tile ({tileX},{tileY}).", rootVirtualPath);
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
        IArchiveCatalog archiveCatalog)
    {
        string mapDirectory = session.ResolvedMapDirectory;
        string rootVirtualPath = $@"World\Maps\{mapDirectory}\{mapDirectory}_{tileX}_{tileY}.adt";
        if (!TryReadVirtualOrLooseFile(session.ClientRoot, rootVirtualPath, archiveCatalog, out byte[]? rootData, out string sourcePath) || rootData is null)
            throw new FileNotFoundException($"Could not locate root ADT for map '{mapDirectory}' tile ({tileX},{tileY}).", rootVirtualPath);

        using MemoryStream stream = new(rootData, writable: false);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, sourcePath);
        stream.Position = 0;
        return WorldTileStageSummaryBuilder.Read(stream, fileSummary);
    }

    private static List<WorldObjectInstance> BuildWmoInstances(
        AdtPlacementCatalog placementCatalog,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, bool> assetReadyLookup)
    {
        List<WorldObjectInstance> instances = new(placementCatalog.WorldModelPlacements.Count);
        for (int index = 0; index < placementCatalog.WorldModelPlacements.Count; index++)
        {
            AdtWorldModelPlacement placement = placementCatalog.WorldModelPlacements[index];
            string modelKey = NormalizeModelKey(placement.ModelPath);
            bool assetReady = archiveCatalog.FileExists(modelKey) || archiveCatalog.FileExists(modelKey.Replace('\\', '/'));
            assetReadyLookup[modelKey] = assetReady;

            Vector3 localMin = placement.BoundsMin - placement.Position;
            Vector3 localMax = placement.BoundsMax - placement.Position;

            instances.Add(new WorldObjectInstance
            {
                ModelKey = modelKey,
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
            });
        }

        return instances;
    }

    private static List<WorldObjectInstance> BuildMdxInstances(
        string clientRoot,
        AdtPlacementCatalog placementCatalog,
        int tileX,
        int tileY,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, LocalBoundsResolution> boundsCache,
        Dictionary<string, bool> assetReadyLookup)
    {
        List<WorldObjectInstance> instances = new(placementCatalog.ModelPlacements.Count);
        for (int index = 0; index < placementCatalog.ModelPlacements.Count; index++)
        {
            AdtModelPlacement placement = placementCatalog.ModelPlacements[index];
            string modelKey = NormalizeModelKey(placement.ModelPath);
            LocalBoundsResolution resolution = ResolveLocalBounds(clientRoot, modelKey, archiveCatalog, boundsCache);
            assetReadyLookup[modelKey] = resolution.AssetReady;

            Matrix4x4 transform = BuildPlacementTransform(placement.Position, placement.Rotation, placement.Scale);
            (Vector3 worldMin, Vector3 worldMax) = TransformBounds(resolution.LocalMin, resolution.LocalMax, transform);

            instances.Add(new WorldObjectInstance
            {
                ModelKey = modelKey,
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
                BoundsResolved = resolution.AssetReady,
                TileX = tileX,
                TileY = tileY,
                HasTileCoordinate = true,
            });
        }

        return instances;
    }

    private static LocalBoundsResolution ResolveLocalBounds(
        string clientRoot,
        string modelKey,
        IArchiveCatalog archiveCatalog,
        Dictionary<string, LocalBoundsResolution> boundsCache)
    {
        if (boundsCache.TryGetValue(modelKey, out LocalBoundsResolution cached))
            return cached;

        LocalBoundsResolution resolved = TryResolveLocalBounds(clientRoot, modelKey, archiveCatalog);
        boundsCache[modelKey] = resolved;
        return resolved;
    }

    private static LocalBoundsResolution TryResolveLocalBounds(string clientRoot, string modelKey, IArchiveCatalog archiveCatalog)
    {
        if (!TryReadVirtualOrLooseFile(clientRoot, modelKey, archiveCatalog, out byte[]? data, out string sourcePath) || data is null)
            return CreateFallbackBounds(assetReady: false, scale: 1.0f);

        string extension = Path.GetExtension(modelKey);
        try
        {
            if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
            {
                using MemoryStream m2Stream = new(data, writable: false);
                M2ModelDocument document = M2ModelReader.Read(m2Stream, sourcePath);
                return new LocalBoundsResolution(true, document.BoundsMin, document.BoundsMax);
            }

            using MemoryStream mdxStream = new(data, writable: false);
            MdxSummary summary = MdxSummaryReader.Read(mdxStream, sourcePath);
            if (summary.BoundsMin is Vector3 boundsMin && summary.BoundsMax is Vector3 boundsMax)
                return new LocalBoundsResolution(true, boundsMin, boundsMax);
        }
        catch
        {
        }

        try
        {
            using MemoryStream fallbackM2Stream = new(data, writable: false);
            M2ModelDocument document = M2ModelReader.Read(fallbackM2Stream, sourcePath);
            return new LocalBoundsResolution(true, document.BoundsMin, document.BoundsMax);
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
        return new LocalBoundsResolution(assetReady, -extent, extent);
    }

    private static bool TryReadVirtualOrLooseFile(
        string clientRoot,
        string virtualPath,
        IArchiveCatalog archiveCatalog,
        out byte[]? data,
        out string sourcePath)
    {
        string normalizedPath = NormalizeModelKey(virtualPath);
        string loosePath = Path.Combine(clientRoot, normalizedPath.Replace('\\', Path.DirectorySeparatorChar));
        if (File.Exists(loosePath))
        {
            data = File.ReadAllBytes(loosePath);
            sourcePath = Path.GetFullPath(loosePath);
            return true;
        }

        data = archiveCatalog.ReadFile(normalizedPath) ?? archiveCatalog.ReadFile(normalizedPath.Replace('\\', '/'));
        sourcePath = normalizedPath;
        return data is { Length: > 0 };
    }

    private static string NormalizeModelKey(string modelPath)
    {
        return modelPath.Trim().Replace('/', '\\');
    }

    private static Matrix4x4 BuildPlacementTransform(Vector3 position, Vector3 rotationDegrees, float scale)
    {
        float pitch = DegreesToRadians(rotationDegrees.X);
        float yaw = DegreesToRadians(rotationDegrees.Y);
        float roll = DegreesToRadians(rotationDegrees.Z);
        return Matrix4x4.CreateScale(scale)
            * Matrix4x4.CreateFromYawPitchRoll(yaw, pitch, roll)
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
            Vector3 fallbackCenter = Vector3.Zero;
            Vector2 fallbackMin = new(tileX * 533.3333f, tileY * 533.3333f);
            Vector2 fallbackMax = fallbackMin + new Vector2(533.3333f, 533.3333f);
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
}