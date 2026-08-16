namespace WowViewer.Core.Runtime.World.Passes;

/// <summary>
/// The declared mapping from each <see cref="WorldFramePasses"/> member to the render stages that
/// account for its cost.
/// <para>
/// This exists because <c>PrepareObjectPhase</c> was declared as a pass and given no stage timer.
/// Its entire cost — including a ~212 ms periodic stall — therefore landed in the unaccounted pass
/// gap, invisible to the stage table by construction. Nothing failed; there was simply nothing that
/// could notice.
/// </para>
/// <para>
/// A test asserts that this table covers every member of <see cref="WorldFramePasses"/> and that no
/// pass maps to an empty stage set, so adding a twelfth pass without a timer breaks the build rather
/// than silently reopening the hole.
/// </para>
/// </summary>
public static class WorldFramePassInstrumentation
{
    /// <summary>
    /// Stages recorded by each pass. A pass may own more than one stage (the MDX opaque pass records
    /// visibility, animation, transparent planning and submission separately), but it must own at
    /// least one.
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyList<WorldRenderStage>> StagesByPass { get; } =
        new Dictionary<string, IReadOnlyList<WorldRenderStage>>(StringComparer.Ordinal)
        {
            [nameof(WorldFramePasses.RenderLighting)] = new[] { WorldRenderStage.Lighting },
            [nameof(WorldFramePasses.RenderSky)] = new[] { WorldRenderStage.Sky },
            [nameof(WorldFramePasses.RenderSkyboxBackdrop)] = new[] { WorldRenderStage.SkyboxBackdrop },
            [nameof(WorldFramePasses.RenderWdl)] = new[] { WorldRenderStage.Wdl },
            [nameof(WorldFramePasses.RenderTerrain)] = new[] { WorldRenderStage.Terrain },
            [nameof(WorldFramePasses.PrepareObjectPhase)] = new[] { WorldRenderStage.PrepareObjectPhase },
            [nameof(WorldFramePasses.RenderWmoOpaque)] = new[]
            {
                WorldRenderStage.WmoVisibility,
                WorldRenderStage.WmoSubmission,
            },
            [nameof(WorldFramePasses.RenderMdxOpaque)] = new[]
            {
                WorldRenderStage.MdxVisibility,
                WorldRenderStage.MdxAnimation,
                WorldRenderStage.MdxTransparentSort,
                WorldRenderStage.MdxOpaqueSubmission,
            },
            [nameof(WorldFramePasses.RenderLiquid)] = new[] { WorldRenderStage.Liquid },
            [nameof(WorldFramePasses.RenderMdxTransparent)] = new[]
            {
                WorldRenderStage.WmoTransparentSubmission,
                WorldRenderStage.MdxTransparentSubmission,
            },
            [nameof(WorldFramePasses.RenderOverlay)] = new[] { WorldRenderStage.Overlay },
        };

    /// <summary>
    /// Stages timed before the pass coordinator runs, so they belong to no pass. Listed explicitly so
    /// the coverage test can assert that every stage is accounted for by something.
    /// </summary>
    public static IReadOnlyList<WorldRenderStage> PrePassStages { get; } = new[]
    {
        WorldRenderStage.SceneMaintenance,
        WorldRenderStage.DeferredAssetLoads,
        WorldRenderStage.TaxiActorUpdate,
    };
}
