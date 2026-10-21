using System.Diagnostics;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// Shared content helpers retained for navigator, inspector and diagnostics callers.
internal sealed partial class ViewerChromeService
{
    private static string FormatBytes(long bytes)
    {
        const double kib = 1024.0;
        const double mib = kib * 1024.0;
        const double gib = mib * 1024.0;

        if (bytes >= gib)
            return $"{bytes / gib:0.00} GiB";
        if (bytes >= mib)
            return $"{bytes / mib:0.0} MiB";
        if (bytes >= kib)
            return $"{bytes / kib:0.0} KiB";
        return $"{bytes} B";
    }


    internal void DrawRuntimeStatsPanelContent()
    {
        using Process process = Process.GetCurrentProcess();
        long managedHeap = GC.GetTotalMemory(forceFullCollection: false);
        long totalAllocated = GC.GetTotalAllocatedBytes(precise: false);
        ImGui.Text($"Process memory: working={FormatBytes(process.WorkingSet64)}  private={FormatBytes(process.PrivateMemorySize64)}");
        ImGui.Text($"Managed heap: live={FormatBytes(managedHeap)}  allocated={FormatBytes(totalAllocated)}  GC={GC.CollectionCount(0)}/{GC.CollectionCount(1)}/{GC.CollectionCount(2)}");

        int tiles = _terrainManager?.LoadedTileCount ?? _vlmTerrainManager?.LoadedTileCount ?? 0;
        int chunks = _terrainManager?.LoadedChunkCount ?? _vlmTerrainManager?.LoadedChunkCount ?? 0;
        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer != null)
            ImGui.Text($"Tiles: {tiles}  Chunks: {terrainRenderer.ChunksRendered}/{chunks}");
        else if (_terrainManager != null || _vlmTerrainManager != null)
            ImGui.Text($"Tiles: {tiles}  Chunks: {chunks}");

        if (_worldScene == null)
        {
            if (_terrainManager == null && _vlmTerrainManager == null)
                ImGui.TextDisabled("Load a world or terrain scene to view runtime stats.");
            return;
        }

        ImGui.Text($"WMO: {_worldScene.WmoRenderedCount}/{_worldScene.WmoInstanceCount}  MDX: {_worldScene.MdxRenderedCount}/{_worldScene.MdxInstanceCount}");
        ImGui.Text($"Asset queue: {_worldScene.Assets.PendingAssetLoadCount}  WMO ok/fail: {_worldScene.Assets.WmoModelsLoaded}/{_worldScene.Assets.WmoModelsFailed}  MDX ok/fail: {_worldScene.Assets.MdxModelsLoaded}/{_worldScene.Assets.MdxModelsFailed}");

        var renderStats = _worldScene.LastRenderFrameStats;
        LiquidRenderer? renderStatsLiquidRenderer = _terrainManager?.LiquidRenderer;
        ImGui.TextDisabled("World render CPU only. UI/layout/input/swap are not included.");
        ImGui.Text($"World CPU: {renderStats.TotalCpuMs:0.00} ms  Pending asset loads: {renderStats.PendingAssetLoadCount}");
        ImGui.TextDisabled("Frame timing over time and hitch detection live on Utilities > Perf.");

        ImGui.Text($"Visible WMO: {renderStats.VisibleWmoCount}  Visible MDX: {renderStats.VisibleMdxCount}  Taxi actors: {renderStats.VisibleTaxiMdxCount}");
        ImGui.Text($"Object stream range: {_worldScene.ObjectStreamingRangeMultiplier:0.00}x");
        ImGui.Text($"Object detail: {_worldScene.ObjectVisibilityProfile}");
        var graphDiagnostics = _worldScene.SceneGraphTraversalDiagnostics;
        ImGui.Text($"ADT graph: {(_worldScene.IsHierarchicalSceneTraversalActive ? "active" : "inactive")}  roots={_worldScene.SceneGraphResidentAdtCount}  external={(_worldScene.SceneGraphHasExternalRoot ? "yes" : "no")}");
        // "skipped" requires per-kind attribution, which recursively walks every rejected subtree.
        // That is off on production frames, so it is opt-in here rather than a permanent frame cost.
        bool detailedGraphDiagnostics = _worldScene.SceneGraphDetailedDiagnosticsEnabled;
        if (ImGui.Checkbox("Detailed graph attribution (costs frame time)", ref detailedGraphDiagnostics))
            _worldScene.SceneGraphDetailedDiagnosticsEnabled = detailedGraphDiagnostics;
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Collecting skipped-descendant counts walks every culled subtree each frame.\nLeave off unless you are reading the breakdown.");

        string skippedText = graphDiagnostics.DetailedCollectionEnabled
            ? graphDiagnostics.SkippedDescendantCount.ToString()
            : "off";
        ImGui.Text($"Graph visited/tested/rejected/skipped: {graphDiagnostics.VisitedNodeCount}/{graphDiagnostics.IndividuallyTestedNodeCount}/{graphDiagnostics.RejectedNodeCount}/{skippedText}");
        ImGui.Text($"AOI camera tile: ({_worldScene.Terrain.CameraTileX},{_worldScene.Terrain.CameraTileY})  loaded={_worldScene.Terrain.LoadedTileCount}  detailed/retained={_worldScene.Terrain.EffectiveDetailedTileCount}/{_worldScene.Terrain.EffectiveRetainedTileCount}");
        if (_worldScene.Terrain.TileUnloadEventCount > 0)
            ImGui.Text($"Last ADT unload: ({_worldScene.Terrain.LastUnloadedTileX},{_worldScene.Terrain.LastUnloadedTileY})  WMO placements={_worldScene.LastUnloadedWmoInstanceCount}");
        ImGui.Text($"Terrain chunks rendered/culled: {renderStats.TerrainChunksRendered}/{renderStats.TerrainChunksCulled}  WDL visible/hidden: {renderStats.WdlVisibleTileCount}/{renderStats.WdlHiddenTileCount}");
        if (terrainRenderer != null)
            ImGui.Text($"Terrain draw/uniform/tex-bind: {terrainRenderer.LastFrameDrawCalls}/{terrainRenderer.LastFrameUniform1Calls}/{terrainRenderer.LastFrameBindTextureCalls}");
        ImGui.Text($"Deferred/taxi/light: {renderStats.DeferredAssetLoads.DurationMs:0.00} / {renderStats.TaxiActorUpdate.DurationMs:0.00} / {renderStats.Lighting.DurationMs:0.00} ms");
        ImGui.Text($"WDL/terrain/liquid: {renderStats.Wdl.DurationMs:0.00} / {renderStats.Terrain.DurationMs:0.00} / {renderStats.Liquid.DurationMs:0.00} ms");
        if (renderStatsLiquidRenderer != null)
            ImGui.Text($"Liquid visible: {renderStatsLiquidRenderer.LastVisibleTerrainMeshCount}/{renderStatsLiquidRenderer.MeshCount}  WL: {renderStatsLiquidRenderer.LastVisibleWlMeshCount}/{renderStatsLiquidRenderer.WlMeshCount}");
        ImGui.Text($"WMO vis/opaque/trans: {renderStats.WmoVisibility.DurationMs:0.00} / {renderStats.WmoSubmission.DurationMs:0.00} / {renderStats.WmoTransparentSubmission.DurationMs:0.00} ms");
        ImGui.Text($"WMO draws batch/fallback/liquid/doodad: {renderStats.WmoBatchDrawCallCount}/{renderStats.WmoGroupFallbackDrawCallCount}/{renderStats.WmoLiquidDrawCallCount}/{renderStats.WmoDoodadSubmissionCount}  instances={renderStats.WmoOpaqueBatchInstanceCount} groups={renderStats.WmoVisibleGroupSubmissionCount}");
        ImGui.Text($"MDX anim/vis/opaque: {renderStats.MdxAnimation.DurationMs:0.00} / {renderStats.MdxVisibility.DurationMs:0.00} / {renderStats.MdxOpaqueSubmission.DurationMs:0.00} ms");
        ImGui.Text($"MDX sort/trans: {renderStats.MdxTransparentSort.DurationMs:0.00} / {renderStats.MdxTransparentSubmission.DurationMs:0.00} ms");
        ImGui.Text($"Models opaque inst/hoisted/unbatched: {renderStats.OpaqueModelSubmission.Instanced}/{renderStats.OpaqueModelSubmission.StateHoisted}/{renderStats.OpaqueModelSubmission.Unbatched}"
            + $"  transparent: {renderStats.TransparentModelSubmission.Unbatched}  draw calls: {renderStats.OpaqueModelSubmission.DrawCalls}+{renderStats.TransparentModelSubmission.DrawCalls}");
        ImGui.Text($"Sky/backdrop/overlay: {renderStats.Sky.DurationMs:0.00} / {renderStats.SkyboxBackdrop.DurationMs:0.00} / {renderStats.Overlay.DurationMs:0.00} ms");
        ImGui.TextWrapped(_worldScene.RendererOptimizationHint);

        var assetReadStats = _worldScene.Assets.GetReadStats();
        ImGui.Separator();
        ImGui.Text($"Asset I/O req/cache: {assetReadStats.ReadRequests}/{assetReadStats.FileCacheHits}  resolved-cache: {assetReadStats.ResolvedPathCacheHits}  probes hit/miss: {assetReadStats.PathProbeResolutions}/{assetReadStats.PathProbeMisses}");
        ImGui.Text($"Asset raw cache: {assetReadStats.FileCacheCount} files / {FormatBytes(assetReadStats.FileCacheBytes)}");

        // Spec 153 Phase 5. DeferredLoadBudget records these and, until now, nothing read them.
        // OversizedAdmissionCount is documented in that class as "the honest measure of the residual
        // the off-thread decode still owes": every one of these is a frame that paid a full
        // synchronous load because the policy guarantees progress even when the load cannot fit.
        DeferredLoadBudget loadBudget = _worldScene.Assets.LoadBudget;
        ImGui.Text($"Deferred loads: oversized admissions {loadBudget.OversizedAdmissionCount}"
            + $"  budget deferrals {loadBudget.BudgetDeferralCount}"
            + $"  worst single load {loadBudget.WorstObservedLoadMs:0.0} ms");
        ImGui.Text($"  predicted cost: MDX {loadBudget.PredictedCostMs(DeferredLoadKind.Mdx):0.0} ms"
            + $"  WMO {loadBudget.PredictedCostMs(DeferredLoadKind.Wmo):0.0} ms");
        if (loadBudget.OversizedAdmissionCount > 0)
        {
            ImGui.TextWrapped(
                "Oversized admissions are synchronous loads larger than the whole frame budget, "
                + "admitted anyway so the asset eventually appears. They are the DeferredAssetLoads "
                + "hitches. The budget cannot subdivide one load - only moving decode off the render "
                + "thread can.");
        }
        ImGui.Text($"Asset misses: failed retry suppress={_worldScene.Assets.SuppressedFailedMdxRetryCount}  known missing M2 skins={_worldScene.Assets.KnownMissingM2SkinCount}  duplicate skin logs={_worldScene.Assets.SuppressedMissingM2SkinLogCount}");

        // Spec 256 P0: where one model load spends its time, and the whole-list skin scan on its own.
        ModelLoadPhaseStats phases = _worldScene.Assets.LoadPhases;
        ImGui.TextWrapped($"Load phases avg/max ms (count): read {phases.AverageMs(ModelLoadPhase.ModelRead):0.0}/{phases.MaxMs(ModelLoadPhase.ModelRead):0}"
            + $"  skin list {phases.AverageMs(ModelLoadPhase.SkinCandidates):0.0}/{phases.MaxMs(ModelLoadPhase.SkinCandidates):0}"
            + $"  skin read {phases.AverageMs(ModelLoadPhase.SkinRead):0.0}/{phases.MaxMs(ModelLoadPhase.SkinRead):0}");
        ImGui.TextWrapped($"  M2 native parse {phases.AverageMs(ModelLoadPhase.NativeParse):0.0}/{phases.MaxMs(ModelLoadPhase.NativeParse):0}"
            + $"  adapter parse {phases.AverageMs(ModelLoadPhase.AdapterParse):0.0}/{phases.MaxMs(ModelLoadPhase.AdapterParse):0}"
            + $"  GPU create {phases.AverageMs(ModelLoadPhase.GpuCreate):0.0}/{phases.MaxMs(ModelLoadPhase.GpuCreate):0} ({phases.Count(ModelLoadPhase.GpuCreate)})");
        ImGui.TextWrapped($"  WMO parse {phases.AverageMs(ModelLoadPhase.WmoParse):0.0}/{phases.MaxMs(ModelLoadPhase.WmoParse):0}"
            + $"  WMO GPU create {phases.AverageMs(ModelLoadPhase.WmoGpuCreate):0.0}/{phases.MaxMs(ModelLoadPhase.WmoGpuCreate):0} ({phases.Count(ModelLoadPhase.WmoGpuCreate)})");
        ImGui.TextWrapped($"  WMO doodad load {phases.AverageMs(ModelLoadPhase.WmoDoodadLoad):0.0}/{phases.MaxMs(ModelLoadPhase.WmoDoodadLoad):0} ({phases.Count(ModelLoadPhase.WmoDoodadLoad)})"
            + $"  shared doodad models {_worldScene.Assets.SharedWmoDoodadModelCount}, hits {_worldScene.Assets.SharedWmoDoodadModelHits}");
        ImGui.TextWrapped($"  .skin list scans: {phases.Count(ModelLoadPhase.SkinListScan)}  avg {phases.AverageMs(ModelLoadPhase.SkinListScan):0.0}"
            + $"  max {phases.MaxMs(ModelLoadPhase.SkinListScan):0} ms  total {phases.TotalMs(ModelLoadPhase.SkinListScan) / 1000.0:0.0} s");
        var streamed = M2TextureStreamer.Stats;
        ImGui.TextWrapped($"  M2 textures streamed: queued {streamed.Queued}  decoded {streamed.Decoded} (avg {streamed.AvgDecodeMs:0.0} ms)"
            + $"  shared {streamed.SharedHits}  uploaded {streamed.Uploaded} (avg {streamed.AvgUploadMs:0.0} ms)  pending {M2TextureStreamer.PendingCount}");
        if (ImGui.SmallButton("Reset load phases"))
            phases.Reset();

        // Why opaque models were not instanced this frame (gates already recorded per frame).
        WorldModelSubmissionStats opaqueGates = renderStats.OpaqueModelSubmission;
        ImGui.TextWrapped($"Opaque model gates: batching off {opaqueGates.GatedBatchingDisabled}  route needs unbatched {opaqueGates.GatedRouteRequiresUnbatchedRender}"
            + $"  no GPU instancing {opaqueGates.GatedGpuInstancingUnsupported}  fade {opaqueGates.GatedOpaqueFadeBelowThreshold}  no renderer {opaqueGates.GatedRendererUnavailable}");

        if (_dataSource is CascDataSource cascDataSource)
        {
            (long prefetchCompleted, long cacheHits, long cachedBytes) = cascDataSource.Stats;
            ImGui.TextWrapped($"CASC prefetch done {prefetchCompleted}  cache hits {cacheHits}  cached {FormatBytes(cachedBytes)}");
        }

        if (_dataSource is MpqDataSource mpqDataSource)
        {
            var mpqStats = mpqDataSource.GetStatsSnapshot();
            ImGui.Text($"MPQ I/O read cache/miss: {mpqStats.ReadCacheHits}/{mpqStats.ReadCacheMisses}  loose/alpha/mpq/miss: {mpqStats.ReadLooseHits}/{mpqStats.ReadAlphaHits}/{mpqStats.ReadMpqHits}/{mpqStats.ReadMisses}  uncached avg: {mpqStats.AverageUncachedReadMs:0.00} ms");
            ImGui.Text($"MPQ raw cache: {mpqStats.ReadCacheEntryCount} files / {FormatBytes(mpqStats.ReadCacheBytes)}  prefetch queue: {mpqStats.PrefetchQueueDepth}");
            ImGui.Text($"MPQ prefetch enq/done/dup/cache: {mpqStats.PrefetchEnqueued}/{mpqStats.PrefetchCompleted}/{mpqStats.PrefetchDuplicateSkips}/{mpqStats.PrefetchCacheSkips}  queue avg: {mpqStats.AveragePrefetchQueueMs:0.00} ms  read avg: {mpqStats.AveragePrefetchReadMs:0.00} ms");
        }
    }

    internal static void DrawTimeOfDayControl(TerrainLighting lighting)
    {
        bool automatic = lighting.IsAutomaticTimeOfDay;
        if (ImGui.Checkbox("Automatic 24-minute cycle", ref automatic))
            lighting.SetAutomaticTimeOfDay(automatic);

        float gameTime = lighting.GameTime;
        if (ImGui.SliderFloat("Time of Day", ref gameTime, 0f, 1f, "%.2f"))
        {
            lighting.GameTime = gameTime;
            lighting.HasManualGameTimeOverride = true;
        }

        ImGui.TextDisabled(lighting.IsAutomaticTimeOfDay
            ? "Live Alpha 0.5.3 clock: one game day per 24 real minutes."
            : "Time frozen at the selected value; enable the cycle to resume.");
    }
}
