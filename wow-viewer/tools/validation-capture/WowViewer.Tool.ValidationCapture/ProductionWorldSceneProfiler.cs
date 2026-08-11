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
    public static WorldRenderDiagnosticReport Run(
        string clientRoot,
        string wdtPath,
        string outputPath,
        string? buildLabel,
        string? listfilePath,
        string? looseOverlayRoot,
        int resolution,
        int warmupFrameCount,
        int measuredFrameCount,
        bool loadAllTiles)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(wdtPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentOutOfRangeException.ThrowIfLessThan(resolution, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(warmupFrameCount);
        ArgumentOutOfRangeException.ThrowIfLessThan(measuredFrameCount, 1);

        if (!File.Exists(wdtPath))
            throw new FileNotFoundException("The production renderer profile requires a local WDT file.", wdtPath);

        string[] overlays = string.IsNullOrWhiteSpace(looseOverlayRoot) ? [] : [looseOverlayRoot];
        using var dataSource = new MpqDataSource(clientRoot, listfilePath, overlays);
        using var context = new HeadlessContext(resolution, resolution);
        using var surface = new RenderSurface(context.GL, resolution, resolution);

        WorldScene? scene = null;
        Stopwatch initializationTimer = Stopwatch.StartNew();
        try
        {
            scene = CreateWorldScene(context.GL, wdtPath, dataSource, buildLabel);
            if (loadAllTiles && !scene.IsWmoBased)
                scene.Terrain.LoadAllTiles();
            initializationTimer.Stop();

            Vector3 cameraPosition = scene.WmoCameraOverride ?? scene.Terrain.GetInitialCameraPosition();
            Vector3 forward = BuildCameraForward(cameraPosition);
            Matrix4x4 view = Matrix4x4.CreateLookAt(cameraPosition, cameraPosition + forward, Vector3.UnitZ);
            Matrix4x4 projection = Matrix4x4.CreatePerspectiveFieldOfView(MathF.PI / 3f, 1f, 0.1f, 20000f);

            for (int frame = 0; frame < warmupFrameCount; frame++)
                RenderFrame(context, surface, scene, cameraPosition, forward, view, projection);

            var frames = new List<WorldRenderDiagnosticFrame>(measuredFrameCount);
            for (int frame = 0; frame < measuredFrameCount; frame++)
            {
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
                sourceStats.ReadRequests,
                sourceStats.ReadCacheHits,
                sourceStats.ReadCacheMisses,
                sourceStats.AverageUncachedReadMs);
            WorldRenderDiagnosticReport report = WorldRenderDiagnostics.Build(
                "headless-production-world-scene",
                warmupFrameCount,
                frames,
                workload);

            string? outputDirectory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
            if (!string.IsNullOrWhiteSpace(outputDirectory))
                Directory.CreateDirectory(outputDirectory);
            File.WriteAllText(outputPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
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

    private static WorldScene CreateWorldScene(GL gl, string wdtPath, MpqDataSource dataSource, string? buildLabel)
    {
        byte[] wdtBytes = File.ReadAllBytes(wdtPath);
        if (IsAlphaWdt(wdtBytes))
            return new WorldScene(gl, wdtPath, dataSource, buildVersion: buildLabel);

        string mapName = Path.GetFileNameWithoutExtension(wdtPath);
        var adapter = new StandardTerrainAdapter(wdtBytes, mapName, dataSource, buildLabel);
        var terrain = new TerrainManager(gl, adapter, mapName, dataSource);
        return new WorldScene(gl, terrain, dataSource, buildVersion: buildLabel);
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
}
