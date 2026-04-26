using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Liquid;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Terrain;
using WowViewer.Core.Runtime.World.Wdl;

namespace WowViewer.App;

internal readonly record struct WowViewerWorldCompositionLayerSnapshot(
    string DisplayName,
    bool Enabled,
    bool Ready,
    int SourceCount,
    int SubmittedCount,
    string Note);

internal sealed class WowViewerWorldDiagnosticsSnapshot
{
    public static readonly WowViewerWorldDiagnosticsSnapshot Empty = new();

    public string ClientRoot { get; private init; } = string.Empty;

    public string RequestedMapInput { get; private init; } = string.Empty;

    public string ResolvedMapDirectory { get; private init; } = string.Empty;

    public TimeSpan LoadDuration { get; private init; }

    public bool LoadedFromArchive { get; private init; }

    public string WdtKindText { get; private init; } = string.Empty;

    public string WdtVersionText { get; private init; } = "n/a";

    public int WdtChunkCount { get; private init; }

    public WdtSummary? WdtSummary { get; private init; }

    public IReadOnlyList<WdtTileCoordinate> OccupiedTiles { get; private init; } = Array.Empty<WdtTileCoordinate>();

    public bool HasRuntime { get; private init; }

    public int SelectedTileX { get; private init; } = -1;

    public int SelectedTileY { get; private init; } = -1;

    public string PlacementSourcePath { get; private init; } = string.Empty;

    public int ActiveTerrainTileCount { get; private init; }

    public IReadOnlyList<string> ActiveTerrainTileSample { get; private init; } = Array.Empty<string>();

    public Vector3 CameraPosition { get; private init; }

    public Vector3 CameraForward { get; private init; }

    public bool ObjectPhaseExecuted { get; private init; }

    public WorldFramePassOptions PassOptions { get; private init; }

    public double TotalCpuMs { get; private init; }

    public WowViewerWorldLoadPipelineDiagnostics? LoadPipeline { get; private init; }

    public IReadOnlyList<WowViewerWorldCompositionLayerSnapshot> CompositionLayers { get; private init; } = Array.Empty<WowViewerWorldCompositionLayerSnapshot>();

    public string WmoVersionSummary { get; private init; } = string.Empty;

    public int EmbeddedWmoMdxCount { get; private init; }

    public int EmbeddedWmoM2Count { get; private init; }

    public int EmbeddedWmoUnknownCount { get; private init; }

    public IReadOnlyList<string> SkyboxBackdropSamplePaths { get; private init; } = Array.Empty<string>();

    public int VisibleTaxiDoodadCount { get; private init; }

    public int WdlVisibleTileCount { get; private init; }

    public WorldWdlTileData? WdlTileData { get; private init; }

    public WorldTerrainTileData? TerrainTileData { get; private init; }

    public WorldTileStageSummary? TileStageSummary { get; private init; }

    public WorldLiquidTileData? LiquidTileData { get; private init; }

    public int TerrainChunksRendered { get; private init; }

    public int TerrainPreviewWidth { get; private init; }

    public int TerrainPreviewHeight { get; private init; }

    public int TerrainPreviewSampledPixelCount { get; private init; }

    public string TerrainVisualHash { get; private init; } = string.Empty;

    public int WmoSubmittedCount { get; private init; }

    public int MdxAnimatedSubmittedCount { get; private init; }

    public int MdxOpaqueSubmittedCount { get; private init; }

    public int MdxTransparentSubmittedCount { get; private init; }

    public int OpaqueRouteCount { get; private init; }

    public int TransparentRouteCount { get; private init; }

    public string OptimizationHint { get; private init; } = string.Empty;

    public bool HasSession => !string.IsNullOrWhiteSpace(ClientRoot) || !string.IsNullOrWhiteSpace(ResolvedMapDirectory);

    public static WowViewerWorldDiagnosticsSnapshot FromRuntimeFrame(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);

        WowViewerWorldSessionBootstrapResult session = runtimeFrame.Session;
        WdtSummary summary = session.WdtSummary;
        string wmoVersionSummary = string.Join(", ",
            runtimeFrame.WmoInstances
                .Where(static instance => instance.WmoVersion.HasValue)
                .Select(static instance => instance.WmoVersion!.Value)
                .Distinct()
                .OrderBy(static version => version));

        return new WowViewerWorldDiagnosticsSnapshot
        {
            ClientRoot = session.ClientRoot,
            RequestedMapInput = session.RequestedMapInput,
            ResolvedMapDirectory = session.ResolvedMapDirectory,
            LoadDuration = session.LoadDuration,
            LoadedFromArchive = session.LoadedFromArchive,
            WdtKindText = session.FileSummary.Kind.ToString(),
            WdtVersionText = session.FileSummary.Version?.ToString() ?? "n/a",
            WdtChunkCount = session.FileSummary.ChunkCount,
            WdtSummary = summary,
            OccupiedTiles = session.OccupiedTiles.ToArray(),
            HasRuntime = true,
            SelectedTileX = runtimeFrame.SelectedTileX,
            SelectedTileY = runtimeFrame.SelectedTileY,
            PlacementSourcePath = runtimeFrame.PlacementSourcePath,
            ActiveTerrainTileCount = runtimeFrame.ActiveTerrainTiles.Count,
            ActiveTerrainTileSample = runtimeFrame.ActiveTerrainTiles
                .Take(12)
                .Select(static tile => $"({tile.TileX},{tile.TileY})")
                .ToArray(),
            CameraPosition = runtimeFrame.CameraPosition,
            CameraForward = runtimeFrame.CameraForward,
            ObjectPhaseExecuted = runtimeFrame.ObjectPhaseExecuted,
            PassOptions = runtimeFrame.PassOptions,
            TotalCpuMs = runtimeFrame.Stats.TotalCpuMs,
            LoadPipeline = runtimeFrame.LoadPipeline,
            CompositionLayers = runtimeFrame.Composition.Layers
                .Select(static layer => new WowViewerWorldCompositionLayerSnapshot(
                    layer.DisplayName,
                    layer.Enabled,
                    layer.Ready,
                    layer.SourceCount,
                    layer.SubmittedCount,
                    layer.Note))
                .ToArray(),
            WmoVersionSummary = wmoVersionSummary,
            EmbeddedWmoMdxCount = runtimeFrame.WmoInstances.Sum(static instance => instance.WmoDoodadMdxCount),
            EmbeddedWmoM2Count = runtimeFrame.WmoInstances.Sum(static instance => instance.WmoDoodadM2Count),
            EmbeddedWmoUnknownCount = runtimeFrame.WmoInstances.Sum(static instance => instance.WmoDoodadUnknownCount),
            SkyboxBackdropSamplePaths = runtimeFrame.SkyboxBackdropInstances.Take(6).Select(static instance => instance.ModelPath).ToArray(),
            VisibleTaxiDoodadCount = runtimeFrame.Visibility.VisibleTaxiMdxCount,
            WdlVisibleTileCount = runtimeFrame.Stats.WdlVisibleTileCount,
            WdlTileData = runtimeFrame.WdlTileData,
            TerrainTileData = runtimeFrame.TerrainTileData,
            TileStageSummary = runtimeFrame.TileStageSummary,
            LiquidTileData = runtimeFrame.LiquidTileData,
            TerrainChunksRendered = runtimeFrame.Stats.TerrainChunksRendered,
            TerrainPreviewWidth = runtimeFrame.TerrainVisualSnapshot.Width,
            TerrainPreviewHeight = runtimeFrame.TerrainVisualSnapshot.Height,
            TerrainPreviewSampledPixelCount = runtimeFrame.TerrainVisualSnapshot.SampledPixelCount,
            TerrainVisualHash = runtimeFrame.TerrainVisualSnapshot.VisualHash,
            WmoSubmittedCount = runtimeFrame.Stats.WmoSubmission.SubmittedCount,
            MdxAnimatedSubmittedCount = runtimeFrame.Stats.MdxAnimation.SubmittedCount,
            MdxOpaqueSubmittedCount = runtimeFrame.Stats.MdxOpaqueSubmission.SubmittedCount,
            MdxTransparentSubmittedCount = runtimeFrame.Stats.MdxTransparentSubmission.SubmittedCount,
            OpaqueRouteCount = runtimeFrame.PassFrame.OpaqueVisibleMdxRoutes.Count,
            TransparentRouteCount = runtimeFrame.PassFrame.TransparentVisibleMdxRoutes.Count,
            OptimizationHint = runtimeFrame.OptimizationHint,
        };
    }
}