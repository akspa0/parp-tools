using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.Core.Runtime.World;

public readonly record struct WorldRenderStageStats(double DurationMs, int VisibleCount = 0, int SubmittedCount = 0);

public readonly record struct WorldRenderFrameStats(
    double TotalCpuMs,
    int PendingAssetLoadCount,
    int TerrainChunksRendered,
    int TerrainChunksCulled,
    int WdlVisibleTileCount,
    int WdlHiddenTileCount,
    int VisibleWmoCount,
    int VisibleMdxCount,
    int VisibleTaxiMdxCount,
    int OpaqueBatchedMdxCount,
    int OpaqueUnbatchedMdxCount,
    int TransparentBatchedMdxCount,
    int TransparentUnbatchedMdxCount,
    int WmoDrawCallCount,
    int WmoBatchDrawCallCount,
    int WmoOpaqueBatchInstanceCount,
    int WmoGroupFallbackDrawCallCount,
    int WmoLiquidDrawCallCount,
    int WmoDoodadSubmissionCount,
    int WmoVisibleGroupSubmissionCount,
    WorldRenderStageStats DeferredAssetLoads,
    WorldRenderStageStats TaxiActorUpdate,
    WorldRenderStageStats Lighting,
    WorldRenderStageStats Sky,
    WorldRenderStageStats SkyboxBackdrop,
    WorldRenderStageStats Wdl,
    WorldRenderStageStats Terrain,
    WorldRenderStageStats WmoVisibility,
    WorldRenderStageStats WmoSubmission,
    WorldRenderStageStats WmoTransparentSubmission,
    WorldRenderStageStats MdxAnimation,
    WorldRenderStageStats MdxVisibility,
    WorldRenderStageStats MdxOpaqueSubmission,
    WorldRenderStageStats Liquid,
    WorldRenderStageStats MdxTransparentSort,
    WorldRenderStageStats MdxTransparentSubmission,
    WorldRenderStageStats Overlay,
    WorldRenderStageStats SceneMaintenance,
    WorldRenderStageStats PrepareObjectPhase)
{
    public IReadOnlyList<WorldOverlayOwnerFrameStats> OverlayOwners { get; init; } =
        Array.Empty<WorldOverlayOwnerFrameStats>();

    /// <summary>
    /// Which rules admitted this frame's WMO placements and groups. Spec 151 instrumentation:
    /// submission counts alone cannot say why a scene admitted the geometry it did.
    /// </summary>
    /// <remarks>
    /// Left to the struct default rather than initialised from <see cref="WmoAdmissionStats.Empty"/>:
    /// the frame history asserts allocation-free recording, and a static property initialiser puts a
    /// lazy class-constructor check on a path that runs every frame.
    /// </remarks>
    public WmoAdmissionStats WmoAdmission { get; init; }

    /// <summary>
    /// Opaque model submission broken out by render path and by the gate that stopped each
    /// instance short of GPU instancing, plus the draw calls the pass actually issued.
    /// </summary>
    /// <remarks>
    /// Specs 201 and 202 Phase 0. <see cref="OpaqueBatchedMdxCount"/> and
    /// <see cref="OpaqueUnbatchedMdxCount"/> are kept alongside this so the decomposition can be
    /// proved to sum to the pre-change totals (spec 201 FR-005). They are the weaker numbers:
    /// they label M2-routed models <c>MDX</c>, and their "batched" adds GPU-instanced draws to
    /// state-hoisted ones, which do not reduce draw calls at all.
    /// </remarks>
    public WorldModelSubmissionStats OpaqueModelSubmission { get; init; }

    /// <summary>
    /// Transparent model submission, same decomposition as <see cref="OpaqueModelSubmission"/>.
    /// </summary>
    public WorldModelSubmissionStats TransparentModelSubmission { get; init; }

    public static WorldRenderFrameStats Empty { get; } = new(
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0),
        new WorldRenderStageStats(0));
}
