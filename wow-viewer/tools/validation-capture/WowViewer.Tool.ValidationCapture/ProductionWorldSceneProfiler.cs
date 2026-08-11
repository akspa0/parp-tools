using System.Diagnostics;
using System.Numerics;
using System.Text.Json;
using Silk.NET.OpenGL;
using WoWViewer.DataSources;
using WoWViewer.Terrain;
using WowViewer.Core.Renderer.Headless;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Tools.ValidationCapture;

/// <summary>
/// Executes the production WorldScene renderer in a hidden OpenGL context. This is intentionally
/// separate from the terrain-only native capture adapter: WMO, MDX, streaming, graph traversal,
/// and the production pass coordinator must be on the measured path.
/// </summary>
internal static class ProductionWorldSceneProfiler
{
    public static string ValidateWdtInput(
        string clientRoot,
        string wdtInput,
        string? listfilePath,
        string? looseOverlayRoot,
        int? targetTileX,
        int? targetTileY)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(wdtInput);

        string[] overlays = string.IsNullOrWhiteSpace(looseOverlayRoot) ? [] : [looseOverlayRoot];
        using var dataSource = new MpqDataSource(clientRoot, listfilePath, overlays);
        ResolvedWdt resolved = ResolveWdt(wdtInput, dataSource);
        ValidateTargetTile(resolved, dataSource, targetTileX, targetTileY);
        string target = targetTileX.HasValue ? $", tile {targetTileX}_{targetTileY}" : string.Empty;
        return $"{(resolved.IsLocalFile ? "local" : "client-archive")} WDT '{resolved.Source}' ({resolved.Bytes.Length:N0} bytes){target}";
    }

    public static WorldRenderDiagnosticReport Run(
        string clientRoot,
        string wdtInput,
        string outputPath,
        string? buildLabel,
        string? listfilePath,
        string? looseOverlayRoot,
        int resolution,
        int warmupFrameCount,
        int measuredFrameCount,
        bool loadAllTiles,
        int? targetTileX,
        int? targetTileY)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(wdtInput);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(resolution, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupFrameCount);
        ArgumentOutOfRangeException.ThrowIfLessThan(measuredFrameCount, 1);

        string[] overlays = string.IsNullOrWhiteSpace(looseOverlayRoot) ? [] : [looseOverlayRoot];
        string? outputDirectory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        PublishProgress(outputPath, "initializing-data-source", 0, warmupFrameCount, 0, measuredFrameCount, null);
        Console.WriteLine("[Profile] Initializing MPQ data source...");
        using var dataSource = new MpqDataSource(clientRoot, listfilePath, overlays);
        ResolvedWdt resolvedWdt = ResolveWdt(wdtInput, dataSource);
        ValidateTargetTile(resolvedWdt, dataSource, targetTileX, targetTileY);
        PublishProgress(outputPath, "creating-hidden-gl-context", 0, warmupFrameCount, 0, measuredFrameCount, null);
        Console.WriteLine("[Profile] Creating hidden OpenGL context...");
        using var context = new HeadlessContext(resolution, resolution);
        using var surface = new RenderSurface(context.GL, resolution, resolution);

        WorldScene? scene = null;
        Stopwatch initializationTimer = Stopwatch.StartNew();
        try
        {
            PublishProgress(outputPath, "constructing-world-scene", 0, warmupFrameCount, 0, measuredFrameCount, null);
            Console.WriteLine("[Profile] Constructing production WorldScene...");
            scene = CreateWorldScene(context.GL, resolvedWdt, dataSource, buildLabel);
            if (loadAllTiles && !scene.IsWmoBased)
            {
                PublishProgress(outputPath, "loading-all-terrain-tiles", 0, warmupFrameCount, 0, measuredFrameCount, null);
                Console.WriteLine("[Profile] Loading all terrain tiles (explicit --load-all-tiles request)...");
                scene.Terrain.LoadAllTiles();
            }
            initializationTimer.Stop();

            Vector3 cameraPosition = ResolveCameraPosition(scene, targetTileX, targetTileY);
            Vector3 forward = BuildCameraForward(cameraPosition);
            Matrix4x4 view = Matrix4x4.CreateLookAt(cameraPosition, cameraPosition + forward, Vector3.UnitZ);
            Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 3f, 1f, 0.1f, 20000f);

            for (int frame = 0; frame < warmupFrameCount; frame++)
            {
                PublishProgress(outputPath, "warming-up", frame, warmupFrameCount, 0, measuredFrameCount, null);
                Console.WriteLine($"[Profile] Warmup {frame + 1}/{warmupFrameCount}...");
                RenderFrame(context, surface, scene, cameraPosition, forward, view, projection);
            }

            var frames = new List<WorldRenderDiagnosticFrame>(measuredFrameCount);
            for (int frame = 0; frame < measuredFrameCount; frame++)
            {
                PublishProgress(outputPath, "measuring", warmupFrameCount, warmupFrameCount, frame, measuredFrameCount, null);
                Console.WriteLine($"[Profile] Measured frame {frame + 1}/{measuredFrameCount}...");
                RenderFrame(context, surface, scene, cameraPosition, forward, view, projection);
                frames.Add(new WorldRenderDiagnosticFrame(frame, scene.LastRenderFrameStats));
            }

            MpqDataSourceStats sourceStats = dataSource.GetStatsSnapshot();
            WorldRenderDiagnosticWorkload workload = new(
                initializationTimer.Elapsed.TotalMilliseconds,
                scene.PendingAssetLoadCount,
                scene.Terrain.PendingTerrainLoadCount,
                scene.PendingDeferredWmoDoodadLoadCount,
                scene.UniqueMdxModels,
                scene.UniqueWmoModels,
                scene.MdxInstanceCount,
                scene.WmoInstanceCount,
                scene.IsHierarchicalSceneTraversalActive,
                targetTileX,
                targetTileY,
                sourceStats.ReadRequests,
                sourceStats.ReadCacheHits,
                sourceStats.ReadCacheMisses,
                sourceStats.AverageUncachedReadMs);
            WorldRenderDiagnosticReport report = WorldRenderDiagnostics.Build(
                "headless-production-world-scene",
                warmupFrameCount,
                frames,
                workload);

            File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine("[Profile] Completed and wrote final diagnostic JSON.");
            return report;
        }
        finally
        {
            scene?.Dispose();
        }
    }

    private static void RenderFrame(
        HeadlessContext context,
        RenderSurface surface,
        WorldScene scene,
        Vector3 cameraPosition,
        Vector3 forward,
        Matrix4x4 view,
        Matrix4x4 projection)
    {
        scene.Terrain.UpdateAOI(cameraPosition, forward);
        surface.Clear(0.04f, 0.05f, 0.06f, 1f);
        scene.Render(view, projection);
        context.RenderSingleFrame();
    }

    private static ResolvedWdt ResolveWdt(string wdtInput, MpqDataSource dataSource)
    {
        if (File.Exists(wdtInput))
            return new ResolvedWdt(wdtInput, File.ReadAllBytes(wdtInput), IsLocalFile: true);

        string virtualPath = wdtInput.Replace('/', '\\').TrimStart('\\');
        byte[]? bytes = dataSource.ReadFile(virtualPath) ?? dataSource.ReadFile(virtualPath.Replace('\\', '/'));
        if (bytes is null || bytes.Length == 0)
        {
            throw new FileNotFoundException(
                "WDT input was neither a local file nor an asset in the configured client. " +
                "Use a client-backed path such as World\\Maps\\Azeroth\\Azeroth.wdt.",
                wdtInput);
        }

        return new ResolvedWdt(virtualPath, bytes, IsLocalFile: false);
    }

    private static WorldScene CreateWorldScene(GL gl, ResolvedWdt wdt, MpqDataSource dataSource, string? buildLabel)
    {
        byte[] wdtBytes = wdt.Bytes;
        if (IsAlphaWdt(wdtBytes))
        {
            if (!wdt.IsLocalFile)
            {
                throw new NotSupportedException(
                    "Archive-backed Alpha WDT input is not supported because AlphaTerrainAdapter requires a local WDT path. " +
                    "Use a local Alpha WDT file for that client family.");
            }

            return new WorldScene(gl, wdt.Source, dataSource, buildVersion: buildLabel);
        }

        string mapName = Path.GetFileNameWithoutExtension(wdt.Source);
        var adapter = new StandardTerrainAdapter(wdtBytes, mapName, dataSource, buildLabel);
        var terrain = new TerrainManager(gl, adapter, mapName, dataSource);
        return new WorldScene(gl, terrain, dataSource, buildVersion: buildLabel);
    }

    private static void ValidateTargetTile(ResolvedWdt wdt, MpqDataSource dataSource, int? tileX, int? tileY)
    {
        if (!tileX.HasValue)
            return;

        if (wdt.IsLocalFile && IsAlphaWdt(wdt.Bytes))
            return;

        string mapName = Path.GetFileNameWithoutExtension(wdt.Source);
        string? mapDirectory = Path.GetDirectoryName(wdt.Source);
        if (string.IsNullOrWhiteSpace(mapDirectory))
            throw new InvalidOperationException($"Cannot derive a map directory from WDT input '{wdt.Source}'.");

        string rootAdtPath = Path.Combine(mapDirectory, $"{mapName}_{tileY}_{tileX}.adt").Replace('/', '\\');
        if (!dataSource.FileExists(rootAdtPath))
            throw new FileNotFoundException($"Target tile {tileX}_{tileY} is not present in map '{mapName}' in the configured client.", rootAdtPath);
    }

    private static Vector3 ResolveCameraPosition(WorldScene scene, int? tileX, int? tileY)
    {
        if (!tileX.HasValue)
            return scene.WmoCameraOverride ?? scene.Terrain.GetInitialCameraPosition();

        const float tileSize = 533.33333f;
        const float mapOrigin = 32f * tileSize;
        float x = mapOrigin - ((tileX.Value + 0.5f) * tileSize);
        float y = mapOrigin - ((tileY!.Value + 0.5f) * tileSize);
        return new Vector3(x, y, 200f);
    }

    private static bool IsAlphaWdt(byte[] bytes)
    {
        for (int offset = 0; offset + 8 <= bytes.Length;)
        {
            int size = BitConverter.ToInt32(bytes, offset + 4);
            if (size < 0 || offset + 8 + size > bytes.Length)
                break;

            bool isMphd = bytes[offset] == (byte)'D' && bytes[offset + 1] == (byte)'H'
                && bytes[offset + 2] == (byte)'P' && bytes[offset + 3] == (byte)'M';
            if (isMphd && size >= 16)
            {
                int mdnmOffset = BitConverter.ToInt32(bytes, offset + 12);
                return mdnmOffset > 1000 && mdnmOffset < bytes.Length;
            }

            offset += 8 + size;
        }

        return false;
    }

    private static Vector3 BuildCameraForward(Vector3 cameraPosition)
    {
        Vector3 target = new(cameraPosition.X + 96f, cameraPosition.Y + 96f, cameraPosition.Z - 48f);
        Vector3 forward = target - cameraPosition;
        return forward.LengthSquared() > 0.0001f ? Vector3.Normalize(forward) : -Vector3.UnitZ;
    }

    private static void PublishProgress(
        string outputPath,
        string phase,
        int completedWarmupFrames,
        int plannedWarmupFrames,
        int completedMeasuredFrames,
        int plannedMeasuredFrames,
        string? detail)
    {
        var progress = new
        {
            schema = "world-render-diagnostic-progress-v1",
            status = "running",
            phase,
            completed_warmup_frames = completedWarmupFrames,
            planned_warmup_frames = plannedWarmupFrames,
            completed_measured_frames = completedMeasuredFrames,
            planned_measured_frames = plannedMeasuredFrames,
            updated_utc = DateTimeOffset.UtcNow,
            detail,
        };
        File.WriteAllText(outputPath, JsonSerializer.Serialize(progress, new JsonSerializerOptions { WriteIndented = true }));
    }

    private sealed record ResolvedWdt(string Source, byte[] Bytes, bool IsLocalFile);
}
