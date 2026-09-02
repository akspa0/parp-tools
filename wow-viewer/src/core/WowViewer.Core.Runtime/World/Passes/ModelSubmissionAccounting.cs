namespace WowViewer.Core.Runtime.World.Passes;

/// <summary>
/// Which render path actually drew a model instance. Mirrors the viewer's <c>M2RouteType</c>,
/// which is the axis the project already uses to distinguish M2 loaders from the legacy MDX one.
/// </summary>
/// <remarks>
/// Spec 201 FR-001/FR-006. The frame panel reported every one of these as <c>MDX</c>, so
/// "100% unbatched" named no path and could not be acted on. Attribution uses the <b>applied</b>
/// route (FR-002): a model whose primary route failed and fell back drew on the fallback, and
/// that is what the metric has to say.
/// <para>
/// Declared here rather than reusing the viewer's enum because <c>WowViewer.Core.Runtime</c>
/// cannot reference the viewer. The viewer maps <c>M2RouteType</c> onto this at the submission
/// site; the two must stay in step.
/// </para>
/// </remarks>
public enum WorldModelRenderPath
{
    /// <summary>No route decision was recorded for the model key.</summary>
    Unknown = 0,

    /// <summary>Direct M2 adapter plus an external .skin.</summary>
    AdapterSkin,

    /// <summary>M2 adapter fed by embedded root-profile geometry (no external .skin).</summary>
    AdapterEmbeddedProfile,

    /// <summary>Native static M2 renderer fed by an embedded legacy root profile.</summary>
    NativeEmbeddedProfile,

    /// <summary>Byte-level M2-to-MDX conversion fallback.</summary>
    ConversionFallback,

    /// <summary>Legacy MDX loader — a genuine non-M2 model.</summary>
    MdxDirect,
}

/// <summary>
/// How one model instance reached the GPU this frame.
/// </summary>
/// <remarks>
/// Spec 202 research R1. <see cref="Instanced"/> and <see cref="StateHoisted"/> were counted
/// identically as "batched", and only the first reduces draw calls: state hoisting still issues
/// one draw per instance, it just skips the per-draw uniform setup. A scene reported as batched
/// could therefore be issuing exactly as many draws as an unbatched one, which made every
/// before/after on this panel unreadable. Contract C3 forbids summing them.
/// </remarks>
public enum WorldModelSubmissionOutcome
{
    /// <summary>Queued into a GPU instance batch — one draw call for the whole batch.</summary>
    Instanced = 0,

    /// <summary>Submitted through a hoisted batch state — still one draw call per instance.</summary>
    StateHoisted,

    /// <summary>Submitted per instance with its own full state setup.</summary>
    Unbatched,
}

/// <summary>
/// The gate that stopped an instance short of GPU instancing, or <see cref="None"/> when nothing
/// did.
/// </summary>
/// <remarks>
/// Spec 202 research R3 establishes that three independent gates force an instance off the
/// instanced path, and they are not equivalent — they have different fixes and unknown relative
/// sizes. Naming the gate per instance is what turns "100% unbatched" into a work item.
/// </remarks>
public enum WorldModelBatchGate
{
    /// <summary>Instance reached GPU instancing; no gate applied.</summary>
    None = 0,

    /// <summary>No renderer resolved for the model key, so nothing was submitted at all.</summary>
    RendererUnavailable,

    /// <summary>The opaque batching toggle is off (spec 153 US3), so every instance is routed unbatched.</summary>
    BatchingDisabled,

    /// <summary>The renderer declares <c>RequiresUnbatchedWorldRender</c> — its route has no batch path.</summary>
    RouteRequiresUnbatchedRender,

    /// <summary>The renderer batches state but does not implement GPU instancing.</summary>
    GpuInstancingUnsupported,

    /// <summary>
    /// The instance is distance-faded, and the instanced path admits only <c>OpaqueFade &gt;= 0.999</c>.
    /// Distance-faded doodads are exactly the dense population, so this gate is a live suspect for
    /// being the largest of the three.
    /// </summary>
    OpaqueFadeBelowInstancingThreshold,

    /// <summary>The pass has no batch path at all — transparent submission is always per instance today.</summary>
    PassHasNoBatchPath,
}

/// <summary>One render path's submission outcomes for a single pass.</summary>
public struct WorldModelPathTally
{
    public int Instanced;
    public int StateHoisted;
    public int Unbatched;

    public readonly int Total => Instanced + StateHoisted + Unbatched;

    public void Add(in WorldModelPathTally other)
    {
        Instanced += other.Instanced;
        StateHoisted += other.StateHoisted;
        Unbatched += other.Unbatched;
    }

    public readonly WorldModelPathStats ToStats() => new()
    {
        Instanced = Instanced,
        StateHoisted = StateHoisted,
        Unbatched = Unbatched,
    };
}

/// <summary>
/// Allocation-free per-frame accumulator for model submission accounting. One
/// <see cref="Record"/> call per submitted instance, plus <see cref="RecordDrawCalls"/> for the
/// draw calls the pass actually issued.
/// </summary>
/// <remarks>
/// Specs 201 and 202 Phase 0. Deliberately records no rendering decision of its own — this type
/// exists so the numbers mean something before anything is optimised on the strength of them.
/// </remarks>
public struct WorldModelSubmissionTally
{
    public WorldModelPathTally Unknown;
    public WorldModelPathTally AdapterSkin;
    public WorldModelPathTally AdapterEmbeddedProfile;
    public WorldModelPathTally NativeEmbeddedProfile;
    public WorldModelPathTally ConversionFallback;
    public WorldModelPathTally MdxDirect;

    /// <summary>
    /// Instances that were instanced despite being distance-faded (spec 207 US1).
    /// </summary>
    /// <remarks>
    /// This population used to be forced unbatched by an <c>OpaqueFade &gt;= 0.999</c> gate, one draw
    /// call each. Its size is the measure of what removing that gate was worth, so it is reported
    /// separately from the fully-opaque instanced count rather than folded into it.
    /// </remarks>
    public int FadedInstanced;

    public int GatedRendererUnavailable;
    public int GatedBatchingDisabled;
    public int GatedRouteRequiresUnbatchedRender;
    public int GatedGpuInstancingUnsupported;
    public int GatedOpaqueFadeBelowThreshold;
    public int GatedPassHasNoBatchPath;

    /// <summary>
    /// Draw calls the pass actually issued for models, counted at the GL call sites rather than
    /// inferred from instance counts. A model draws once per geoset/section, not once per
    /// instance, so no arithmetic over the instance counters can produce this number.
    /// </summary>
    public int DrawCalls;

    /// <summary>
    /// Distinct model keys submitted this pass. Per-model instancing cannot go below this, so it
    /// is the floor the instanced count is converging on (spec 202 research R2).
    /// </summary>
    public int DistinctModelCount;

    /// <summary>Record that an instanced submission was distance-faded.</summary>
    public void RecordFadedInstanced() => FadedInstanced++;

    public void Record(WorldModelRenderPath path, WorldModelSubmissionOutcome outcome, WorldModelBatchGate gate)
    {
        // A struct member cannot return one of its own fields by reference, so the path is
        // selected here and the outcome applied through a by-ref local.
        switch (path)
        {
            case WorldModelRenderPath.AdapterSkin:
                Apply(ref AdapterSkin, outcome);
                break;
            case WorldModelRenderPath.AdapterEmbeddedProfile:
                Apply(ref AdapterEmbeddedProfile, outcome);
                break;
            case WorldModelRenderPath.NativeEmbeddedProfile:
                Apply(ref NativeEmbeddedProfile, outcome);
                break;
            case WorldModelRenderPath.ConversionFallback:
                Apply(ref ConversionFallback, outcome);
                break;
            case WorldModelRenderPath.MdxDirect:
                Apply(ref MdxDirect, outcome);
                break;
            default:
                Apply(ref Unknown, outcome);
                break;
        }

        switch (gate)
        {
            case WorldModelBatchGate.None:
                break;
            case WorldModelBatchGate.RendererUnavailable:
                GatedRendererUnavailable++;
                break;
            case WorldModelBatchGate.BatchingDisabled:
                GatedBatchingDisabled++;
                break;
            case WorldModelBatchGate.RouteRequiresUnbatchedRender:
                GatedRouteRequiresUnbatchedRender++;
                break;
            case WorldModelBatchGate.GpuInstancingUnsupported:
                GatedGpuInstancingUnsupported++;
                break;
            case WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold:
                GatedOpaqueFadeBelowThreshold++;
                break;
            case WorldModelBatchGate.PassHasNoBatchPath:
                GatedPassHasNoBatchPath++;
                break;
        }

        static void Apply(ref WorldModelPathTally tally, WorldModelSubmissionOutcome outcome)
        {
            switch (outcome)
            {
                case WorldModelSubmissionOutcome.Instanced:
                    tally.Instanced++;
                    break;
                case WorldModelSubmissionOutcome.StateHoisted:
                    tally.StateHoisted++;
                    break;
                case WorldModelSubmissionOutcome.Unbatched:
                    tally.Unbatched++;
                    break;
            }
        }
    }

    /// <summary>Records an instance that resolved no renderer and therefore never reached the GPU.</summary>
    public void RecordRendererUnavailable() => GatedRendererUnavailable++;

    public void RecordDrawCalls(int drawCalls)
    {
        if (drawCalls > 0)
            DrawCalls += drawCalls;
    }

    public void Add(in WorldModelSubmissionTally other)
    {
        Unknown.Add(other.Unknown);
        AdapterSkin.Add(other.AdapterSkin);
        AdapterEmbeddedProfile.Add(other.AdapterEmbeddedProfile);
        NativeEmbeddedProfile.Add(other.NativeEmbeddedProfile);
        ConversionFallback.Add(other.ConversionFallback);
        MdxDirect.Add(other.MdxDirect);

        GatedRendererUnavailable += other.GatedRendererUnavailable;
        GatedBatchingDisabled += other.GatedBatchingDisabled;
        GatedRouteRequiresUnbatchedRender += other.GatedRouteRequiresUnbatchedRender;
        GatedGpuInstancingUnsupported += other.GatedGpuInstancingUnsupported;
        FadedInstanced += other.FadedInstanced;
        GatedOpaqueFadeBelowThreshold += other.GatedOpaqueFadeBelowThreshold;
        GatedPassHasNoBatchPath += other.GatedPassHasNoBatchPath;

        DrawCalls += other.DrawCalls;
        DistinctModelCount += other.DistinctModelCount;
    }

    public void Reset() => this = default;

    public readonly WorldModelSubmissionStats ToStats() => new()
    {
        Unknown = Unknown.ToStats(),
        AdapterSkin = AdapterSkin.ToStats(),
        AdapterEmbeddedProfile = AdapterEmbeddedProfile.ToStats(),
        NativeEmbeddedProfile = NativeEmbeddedProfile.ToStats(),
        ConversionFallback = ConversionFallback.ToStats(),
        MdxDirect = MdxDirect.ToStats(),
        GatedRendererUnavailable = GatedRendererUnavailable,
        GatedBatchingDisabled = GatedBatchingDisabled,
        GatedRouteRequiresUnbatchedRender = GatedRouteRequiresUnbatchedRender,
        GatedGpuInstancingUnsupported = GatedGpuInstancingUnsupported,
        FadedInstanced = FadedInstanced,
        GatedOpaqueFadeBelowThreshold = GatedOpaqueFadeBelowThreshold,
        GatedPassHasNoBatchPath = GatedPassHasNoBatchPath,
        DrawCalls = DrawCalls,
        DistinctModelCount = DistinctModelCount,
    };
}

/// <summary>One render path's submission outcomes for a single pass, as reported.</summary>
public readonly record struct WorldModelPathStats
{
    public int Instanced { get; init; }
    public int StateHoisted { get; init; }
    public int Unbatched { get; init; }

    public int Total => Instanced + StateHoisted + Unbatched;

    /// <summary>
    /// What the old aggregate counter called "batched" — instanced plus state-hoisted. Exposed
    /// only so the decomposition can be proved to sum to the pre-change totals (spec 201 FR-005).
    /// Do not report this as a performance figure: half of it does not reduce draw calls.
    /// </summary>
    public int LegacyBatchedEquivalent => Instanced + StateHoisted;
}

/// <summary>
/// One pass's model submission accounting, broken out by render path and by the gate that stopped
/// each instance short of GPU instancing.
/// </summary>
public readonly record struct WorldModelSubmissionStats
{
    public WorldModelPathStats Unknown { get; init; }
    public WorldModelPathStats AdapterSkin { get; init; }
    public WorldModelPathStats AdapterEmbeddedProfile { get; init; }
    public WorldModelPathStats NativeEmbeddedProfile { get; init; }
    public WorldModelPathStats ConversionFallback { get; init; }
    public WorldModelPathStats MdxDirect { get; init; }

    public int GatedRendererUnavailable { get; init; }
    public int GatedBatchingDisabled { get; init; }
    public int GatedRouteRequiresUnbatchedRender { get; init; }
    public int GatedGpuInstancingUnsupported { get; init; }
    public int FadedInstanced { get; init; }

    public int GatedOpaqueFadeBelowThreshold { get; init; }
    public int GatedPassHasNoBatchPath { get; init; }

    public int DrawCalls { get; init; }
    public int DistinctModelCount { get; init; }

    public static WorldModelSubmissionStats Empty { get; } = default;

    public int Instanced
        => Unknown.Instanced + AdapterSkin.Instanced + AdapterEmbeddedProfile.Instanced
           + NativeEmbeddedProfile.Instanced + ConversionFallback.Instanced + MdxDirect.Instanced;

    public int StateHoisted
        => Unknown.StateHoisted + AdapterSkin.StateHoisted + AdapterEmbeddedProfile.StateHoisted
           + NativeEmbeddedProfile.StateHoisted + ConversionFallback.StateHoisted + MdxDirect.StateHoisted;

    public int Unbatched
        => Unknown.Unbatched + AdapterSkin.Unbatched + AdapterEmbeddedProfile.Unbatched
           + NativeEmbeddedProfile.Unbatched + ConversionFallback.Unbatched + MdxDirect.Unbatched;

    public int Total => Instanced + StateHoisted + Unbatched;

    /// <summary>
    /// The pre-decomposition "batched" count. Spec 201 FR-005 requires the per-path counts to sum
    /// exactly to the aggregate totals, which is what proves this is a decomposition rather than a
    /// redefinition.
    /// </summary>
    public int LegacyBatchedEquivalent => Instanced + StateHoisted;

    /// <summary>
    /// Draw calls per submitted instance. At 1.0 batching is buying nothing regardless of what the
    /// batched counter says; well under 1.0 means instancing is actually collapsing draws.
    /// </summary>
    public double DrawCallsPerInstance => Total == 0 ? 0d : (double)DrawCalls / Total;

    /// <summary>
    /// The gate that blocked the most instances, which is the one worth fixing first. Returns
    /// <see cref="WorldModelBatchGate.None"/> when nothing was blocked.
    /// </summary>
    public WorldModelBatchGate DominantGate
    {
        get
        {
            WorldModelBatchGate gate = WorldModelBatchGate.None;
            int best = 0;
            Consider(WorldModelBatchGate.RendererUnavailable, GatedRendererUnavailable);
            Consider(WorldModelBatchGate.BatchingDisabled, GatedBatchingDisabled);
            Consider(WorldModelBatchGate.RouteRequiresUnbatchedRender, GatedRouteRequiresUnbatchedRender);
            Consider(WorldModelBatchGate.GpuInstancingUnsupported, GatedGpuInstancingUnsupported);
            Consider(WorldModelBatchGate.OpaqueFadeBelowInstancingThreshold, GatedOpaqueFadeBelowThreshold);
            Consider(WorldModelBatchGate.PassHasNoBatchPath, GatedPassHasNoBatchPath);
            return gate;

            void Consider(WorldModelBatchGate candidate, int count)
            {
                if (count <= best)
                    return;

                best = count;
                gate = candidate;
            }
        }
    }

    public WorldModelPathStats ForPath(WorldModelRenderPath path) => path switch
    {
        WorldModelRenderPath.AdapterSkin => AdapterSkin,
        WorldModelRenderPath.AdapterEmbeddedProfile => AdapterEmbeddedProfile,
        WorldModelRenderPath.NativeEmbeddedProfile => NativeEmbeddedProfile,
        WorldModelRenderPath.ConversionFallback => ConversionFallback,
        WorldModelRenderPath.MdxDirect => MdxDirect,
        _ => Unknown,
    };
}
