using System.Numerics;
using WoWViewer.Rendering;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;
using WorldModelRenderPath = WowViewer.Core.Runtime.World.Passes.WorldModelRenderPath;
using WorldModelSubmissionTally = WowViewer.Core.Runtime.World.Passes.WorldModelSubmissionTally;
using WorldObjectPassCoordinator = WowViewer.Core.Runtime.World.Passes.WorldObjectPassCoordinator;
using WorldObjectPassFrame = WowViewer.Core.Runtime.World.Passes.WorldObjectPassFrame;
using VisibleMdxInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleMdxEntry;
using VisibleWmoInstance = WowViewer.Core.Runtime.World.Visibility.WorldVisibleWmoEntry;

namespace WoWViewer.Terrain;

// Moved from WorldScene (Spec 255 W0): formerly a private nested class; body unchanged.
// Per-frame render scratch and counters; WorldScene owns the single instance (_renderFrame).
internal sealed class WorldRenderFrame
{
    public WorldVisibilityFrame Visibility { get; } = new();
    public WorldObjectPassFrame ObjectPasses { get; } = new();
    public Dictionary<string, WmoRenderer> VisibleWmoRendererCache { get; } = new(StringComparer.OrdinalIgnoreCase);
    public Dictionary<string, IModelRenderer> VisibleMdxRendererCache { get; } = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Applied render path per visible model key, resolved once per frame from the asset
    /// manager's route decisions. Spec 201 FR-002: attribution uses the applied route, because
    /// a model whose primary route failed and fell back drew on the fallback.
    /// </summary>
    public Dictionary<string, WorldModelRenderPath> VisibleMdxRenderPathCache { get; } = new(StringComparer.OrdinalIgnoreCase);

    // Opaque-pass scratch, reused across frames and cleared in place. These were allocated fresh
    // every frame inside the batching pass — the pass that exists to reduce work. Named scratch,
    // not cache: they are rebuilt every frame by design.
    public List<WorldObjectPassCoordinator.WorldWmoOpaqueBatchCandidate> WmoBatchCandidateScratch { get; } = [];
    public Dictionary<IModelRenderer, List<Matrix4x4>> WmoDoodadBatchGroupScratch { get; } = [];
    public List<WmoOpaqueDoodadBatchItem> WmoDoodadUnbatchedScratch { get; } = [];
    public HashSet<IModelRenderer> UpdatedRendererScratch { get; } = [];
    public HashSet<IGpuInstancedModelRenderer> GpuBatchRendererScratch { get; } = [];
    public HashSet<IModelRenderer> ImmediateBatchRendererScratch { get; } = [];

    // Distinct model keys submitted per pass. Per-model instancing cannot collapse opaque
    // draws below this count, so it is the floor the instanced counter converges on
    // (spec 202 research R2).
    public HashSet<string> OpaqueSubmittedModelKeyScratch { get; } = new(StringComparer.OrdinalIgnoreCase);
    public HashSet<string> TransparentSubmittedModelKeyScratch { get; } = new(StringComparer.OrdinalIgnoreCase);
    public List<(bool IsWmo, int Index, float DistanceSq)> TransparentSortScratch { get; } = [];
    public List<VisibleWmoInstance> WmoInstanceBatchScratch { get; } = [];

    // The batch-group values are themselves per-frame lists; pool them so clearing the group
    // dictionary does not drop a list per renderer per frame onto the heap.
    private readonly Stack<List<Matrix4x4>> _matrixListPool = new();

    public List<Matrix4x4> RentMatrixList()
        => _matrixListPool.Count > 0 ? _matrixListPool.Pop() : [];

    private void ReturnMatrixLists()
    {
        foreach (List<Matrix4x4> list in WmoDoodadBatchGroupScratch.Values)
        {
            list.Clear();
            _matrixListPool.Push(list);
        }

        WmoDoodadBatchGroupScratch.Clear();
    }

    private void ResetOpaquePassScratch()
    {
        WmoBatchCandidateScratch.Clear();
        ReturnMatrixLists();
        WmoDoodadUnbatchedScratch.Clear();
        UpdatedRendererScratch.Clear();
        GpuBatchRendererScratch.Clear();
        ImmediateBatchRendererScratch.Clear();
        OpaqueSubmittedModelKeyScratch.Clear();
        TransparentSubmittedModelKeyScratch.Clear();
        TransparentSortScratch.Clear();
        WmoInstanceBatchScratch.Clear();
    }

    public List<VisibleWmoInstance> VisibleWmoInstances => Visibility.VisibleWmos;
    public List<VisibleMdxInstance> VisibleMdxInstances => Visibility.VisibleMdx;
    public int VisibleTaxiMdxCount
    {
        get => Visibility.VisibleTaxiMdxCount;
        set => Visibility.VisibleTaxiMdxCount = value;
    }

    public int OpaqueBatchedMdxCount { get; set; }
    public int OpaqueUnbatchedMdxCount { get; set; }
    public int TransparentBatchedMdxCount { get; set; }
    public int TransparentUnbatchedMdxCount { get; set; }

    /// <summary>
    /// Specs 201 and 202 Phase 0. Decomposes the four aggregate counters above by render path
    /// and by the gate that stopped each instance short of GPU instancing, and carries the
    /// draw calls the pass actually issued. The aggregates are kept so the decomposition can
    /// be proved to sum to them (spec 201 FR-005).
    /// </summary>
    public WorldModelSubmissionTally OpaqueModelSubmission;
    public WorldModelSubmissionTally TransparentModelSubmission;
    public int WmoDrawCallCount { get; set; }
    public int WmoBatchDrawCallCount { get; set; }
    public int WmoOpaqueBatchInstanceCount { get; set; }
    public int WmoGroupFallbackDrawCallCount { get; set; }
    public int WmoLiquidDrawCallCount { get; set; }
    public int WmoDoodadSubmissionCount { get; set; }
    public int WmoVisibleGroupSubmissionCount { get; set; }

    /// <summary>
    /// Spec 151 admission accounting: which rule admitted each WMO placement and each group.
    /// Accumulated across every placement and submission pass in the frame.
    /// </summary>
    public WmoAdmissionTally WmoAdmission;

    public double DeferredAssetLoadMs { get; set; }
    public double TaxiActorUpdateMs { get; set; }
    public double LightingMs { get; set; }
    public double SkyMs { get; set; }
    public double SkyboxBackdropMs { get; set; }
    public double WdlMs { get; set; }
    public double TerrainMs { get; set; }
    public double WmoVisibilityMs { get; set; }
    public double WmoSubmissionMs { get; set; }
    public double WmoTransparentSubmissionMs { get; set; }
    public double MdxAnimationMs { get; set; }
    public double MdxVisibilityMs { get; set; }
    public double MdxOpaqueSubmissionMs { get; set; }
    public double LiquidMs { get; set; }
    public double MdxTransparentSortMs { get; set; }
    public double MdxTransparentSubmissionMs { get; set; }
    public double OverlayMs { get; set; }
    public double SceneMaintenanceMs { get; set; }
    public double PrepareObjectPhaseMs { get; set; }
    public List<WorldOverlayOwnerFrameStats> OverlayOwners { get; } = new(WorldOverlayOwners.All.Count);

    public void Reset()
    {
        Visibility.Reset();
        ObjectPasses.Reset();
        VisibleWmoRendererCache.Clear();
        VisibleMdxRendererCache.Clear();
        VisibleMdxRenderPathCache.Clear();
        ResetOpaquePassScratch();
        OpaqueModelSubmission.Reset();
        TransparentModelSubmission.Reset();
        OpaqueBatchedMdxCount = 0;
        OpaqueUnbatchedMdxCount = 0;
        TransparentBatchedMdxCount = 0;
        TransparentUnbatchedMdxCount = 0;
        WmoDrawCallCount = 0;
        WmoBatchDrawCallCount = 0;
        WmoOpaqueBatchInstanceCount = 0;
        WmoGroupFallbackDrawCallCount = 0;
        WmoLiquidDrawCallCount = 0;
        WmoDoodadSubmissionCount = 0;
        WmoVisibleGroupSubmissionCount = 0;
        WmoAdmission.Reset();
        DeferredAssetLoadMs = 0;
        TaxiActorUpdateMs = 0;
        LightingMs = 0;
        SkyMs = 0;
        SkyboxBackdropMs = 0;
        WdlMs = 0;
        TerrainMs = 0;
        WmoVisibilityMs = 0;
        WmoSubmissionMs = 0;
        WmoTransparentSubmissionMs = 0;
        MdxAnimationMs = 0;
        MdxVisibilityMs = 0;
        MdxOpaqueSubmissionMs = 0;
        LiquidMs = 0;
        MdxTransparentSortMs = 0;
        MdxTransparentSubmissionMs = 0;
        OverlayMs = 0;
        SceneMaintenanceMs = 0;
        PrepareObjectPhaseMs = 0;
        OverlayOwners.Clear();
        foreach (string ownerId in WorldOverlayOwners.All)
            OverlayOwners.Add(WorldOverlayOwnerFrameStats.Disabled(ownerId));
    }

    public void SetOverlayOwner(
        string ownerId,
        double durationMs,
        bool enabled,
        int preparedPrimitiveCount = 0,
        int submittedPrimitiveCount = 0,
        string cacheStatus = "not_cached",
        int deferredCount = 0)
    {
        WorldOverlayOwnerFrameStats stats = new(
            ownerId,
            Math.Max(0, durationMs),
            enabled,
            Math.Max(0, preparedPrimitiveCount),
            Math.Max(0, submittedPrimitiveCount),
            cacheStatus,
            Math.Max(0, deferredCount));

        for (int i = 0; i < OverlayOwners.Count; i++)
        {
            if (string.Equals(OverlayOwners[i].OwnerId, ownerId, StringComparison.Ordinal))
            {
                OverlayOwners[i] = stats;
                return;
            }
        }

        throw new InvalidOperationException($"Unknown world overlay owner '{ownerId}'.");
    }

    public double OverlayOwnerDurationSum => OverlayOwners.Sum(static owner => owner.DurationMs);

    public WorldRenderFrameStats ToStats(
        double totalCpuMs,
        int pendingAssetLoadCount,
        int terrainChunksRendered,
        int terrainChunksCulled,
        int wdlVisibleTileCount,
        int wdlHiddenTileCount)
    {
        int visibleMdxCount = Math.Max(0, VisibleMdxInstances.Count - VisibleTaxiMdxCount);
        return new WorldRenderFrameStats(
            totalCpuMs,
            pendingAssetLoadCount,
            terrainChunksRendered,
            terrainChunksCulled,
            wdlVisibleTileCount,
            wdlHiddenTileCount,
            VisibleWmoInstances.Count,
            visibleMdxCount,
            VisibleTaxiMdxCount,
            OpaqueBatchedMdxCount,
            OpaqueUnbatchedMdxCount,
            TransparentBatchedMdxCount,
            TransparentUnbatchedMdxCount,
            WmoDrawCallCount,
            WmoBatchDrawCallCount,
            WmoOpaqueBatchInstanceCount,
            WmoGroupFallbackDrawCallCount,
            WmoLiquidDrawCallCount,
            WmoDoodadSubmissionCount,
            WmoVisibleGroupSubmissionCount,
            new WorldRenderStageStats(DeferredAssetLoadMs),
            new WorldRenderStageStats(TaxiActorUpdateMs),
            new WorldRenderStageStats(LightingMs),
            new WorldRenderStageStats(SkyMs),
            new WorldRenderStageStats(SkyboxBackdropMs),
            new WorldRenderStageStats(WdlMs, wdlVisibleTileCount),
            new WorldRenderStageStats(TerrainMs, terrainChunksRendered),
            new WorldRenderStageStats(WmoVisibilityMs, VisibleWmoInstances.Count),
            new WorldRenderStageStats(WmoSubmissionMs, VisibleWmoInstances.Count, VisibleWmoInstances.Count),
            new WorldRenderStageStats(WmoTransparentSubmissionMs, VisibleWmoInstances.Count, WmoDrawCallCount),
            new WorldRenderStageStats(MdxAnimationMs),
            new WorldRenderStageStats(MdxVisibilityMs, VisibleMdxInstances.Count),
            new WorldRenderStageStats(MdxOpaqueSubmissionMs, VisibleMdxInstances.Count, OpaqueBatchedMdxCount + OpaqueUnbatchedMdxCount),
            new WorldRenderStageStats(LiquidMs),
            new WorldRenderStageStats(MdxTransparentSortMs, ObjectPasses.TransparentVisibleMdxRoutes.Count),
            new WorldRenderStageStats(MdxTransparentSubmissionMs, ObjectPasses.TransparentVisibleMdxRoutes.Count, TransparentBatchedMdxCount + TransparentUnbatchedMdxCount),
            new WorldRenderStageStats(OverlayMs),
            new WorldRenderStageStats(SceneMaintenanceMs),
            new WorldRenderStageStats(PrepareObjectPhaseMs))
        {
            OverlayOwners = OverlayOwners.ToArray(),
            WmoAdmission = WmoAdmission.ToStats(),
            OpaqueModelSubmission = OpaqueModelSubmission.ToStats(),
            TransparentModelSubmission = TransparentModelSubmission.ToStats(),
        };
    }
}
