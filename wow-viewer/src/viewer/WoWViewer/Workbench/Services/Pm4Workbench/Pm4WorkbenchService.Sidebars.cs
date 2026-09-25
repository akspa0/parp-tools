using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// Pm4WorkbenchService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class Pm4WorkbenchService
{

    // Snapshot state for the frame-history view. Snapshot() allocates, so it is refreshed on a
    // bounded cadence rather than every UI frame — a diagnostic that churns every frame would be
    // the same mistake this view exists to expose.
    private WorldRenderFrameHistorySnapshot? _frameHistorySnapshot;
    private double _frameHistoryNextRefreshSeconds;
    private float _frameHistoryHitchThresholdMs = 33.3f;
    private readonly float[] _frameHistoryPlotBuffer = new float[240];
    private double _frameHistoryInjectStallMs = 250;
    private bool _frameHistoryPaused;

    private static double SumStageMaxima(WorldRenderFrameHistorySnapshot snapshot)
    {
        double sum = 0;
        foreach (WorldRenderTimingDistribution dist in snapshot.Stages.Values)
            sum += dist.MaxMs;
        return sum;
    }

    /// <summary>
    /// Frame timing over time, with hitches marked and attributed to a stage.
    /// <para>
    /// The single-frame "World CPU" readout above cannot show a periodic hitch: by the time you
    /// read it the spike is gone. This is the surface that makes the gallop observable.
    /// </para>
    /// </summary>
    private void DrawFrameHistoryContent()
    {
        if (_worldScene == null)
            return;

        if (!ImGui.CollapsingHeader("Frame history (hitch detection)", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        WorldRenderFrameHistory history = _worldScene.FrameHistory;

        // Freeze so the numbers can be read and screenshotted. Recording continues regardless.
        ImGui.Checkbox("Pause updates", ref _frameHistoryPaused);
        ImGui.SameLine();
        ImGui.TextDisabled("(recording continues)");

        double now = ImGui.GetTime();
        if (_frameHistorySnapshot is null || (!_frameHistoryPaused && now >= _frameHistoryNextRefreshSeconds))
        {
            _frameHistorySnapshot = history.Snapshot(_frameHistoryHitchThresholdMs);
            _frameHistoryNextRefreshSeconds = now + 0.25;
        }

        WorldRenderFrameHistorySnapshot snapshot = _frameHistorySnapshot;

        ImGui.SetNextItemWidth(140f);
        ImGui.SliderFloat("Hitch threshold (ms)", ref _frameHistoryHitchThresholdMs, 8f, 100f, "%.1f");

        if (snapshot.FrameCount == 0)
        {
            ImGui.TextDisabled("No frames recorded yet.");
            return;
        }

        // A stationary window cannot demonstrate movement-induced behavior. Say so rather than
        // letting a quiet result be read as "no hitching".
        if (!snapshot.CanDemonstrateMovementBehavior)
        {
            ImGui.TextColored(new Vector4(1f, 0.75f, 0.2f, 1f),
                "Camera stationary for this window - cannot show movement hitching. Move the camera.");
        }

        WorldRenderTimingDistribution total = snapshot.Total;
        ImGui.Text($"Frames {snapshot.FrameCount}  median {total.MedianMs:0.00}  p95 {total.P95Ms:0.00}  p99 {total.P99Ms:0.00}  max {total.MaxMs:0.00} ms");

        Vector4 hitchColor = total.OverThresholdCount > 0
            ? new Vector4(1f, 0.4f, 0.35f, 1f)
            : new Vector4(0.5f, 0.9f, 0.5f, 1f);
        ImGui.TextColored(hitchColor,
            $"Hitches over {_frameHistoryHitchThresholdMs:0.0} ms: {total.OverThresholdCount} of {snapshot.FrameCount} frames");

        // Total minus the sum of every stage timer. If this is large, the cost is somewhere no timer
        // covers, and the stage table below is not where to look.
        WorldRenderTimingDistribution unaccounted = snapshot.Unaccounted;
        bool unaccountedDominates = unaccounted.MaxMs > 1.0 && unaccounted.MaxMs > total.MaxMs * 0.5;
        ImGui.TextColored(
            unaccountedDominates ? new Vector4(1f, 0.55f, 0.2f, 1f) : new Vector4(0.7f, 0.7f, 0.7f, 1f),
            $"UNACCOUNTED (not covered by any stage timer): median {unaccounted.MedianMs:0.00}  p99 {unaccounted.P99Ms:0.00}  max {unaccounted.MaxMs:0.00} ms");
        if (unaccountedDominates)
        {
            ImGui.TextWrapped(
                "Most of the hitch is NOT in any instrumented stage. The stage table below cannot "
                + "explain it - the work is happening between or outside the timers.");
        }

        // Submission efficiency. If most instances are unbatched, the cost is one draw call per
        // object and no amount of allocation tuning will touch it - instancing is the fix.
        if (ImGui.TreeNode("Submission efficiency (this frame)###FrameHistoryBatching"))
        {
            WorldRenderFrameStats live = _worldScene.LastRenderFrameStats;
            int mdxOpaqueTotal = live.OpaqueBatchedMdxCount + live.OpaqueUnbatchedMdxCount;
            int mdxTransparentTotal = live.TransparentBatchedMdxCount + live.TransparentUnbatchedMdxCount;

            // Specs 201 and 202 Phase 0. The aggregate pair above conflates two things at once:
            // it labels M2-routed models MDX, and its "batched" adds GPU-instanced draws (one draw
            // per batch) to state-hoisted ones (still one draw per instance). Both totals are kept
            // so the decomposition below can be checked to sum to them, but the decomposition is
            // what to read.
            DrawModelSubmissionCounters("Opaque models", live.OpaqueModelSubmission);
            ImGui.Spacing();
            DrawModelSubmissionCounters("Transparent models", live.TransparentModelSubmission);

            ImGui.Spacing();
            ImGui.Text($"WMO draw calls:  total {live.WmoDrawCallCount,6}  batched {live.WmoBatchDrawCallCount,6}  group-fallback {live.WmoGroupFallbackDrawCallCount,6}");
            ImGui.Text($"WMO doodad submissions: {live.WmoDoodadSubmissionCount}   visible groups: {live.WmoVisibleGroupSubmissionCount}");
            SceneLightingFrameStats lightStats = live.SceneLighting;
            ImGui.Text($"WMO placements:  batched {lightStats.WmoPlacementsBatched,6}  lit per-placement {lightStats.WmoPlacementsLitFallback,6}  (self-lit {lightStats.WmoPlacementsSelfLit})");
            ImGui.Text($"Scene lights:  kept {lightStats.LightsKept,6} of {lightStats.LightsCollected,6}  queries {lightStats.QueryCount,6}  candidates tested {lightStats.CandidatesTested,8}");

            ImGui.Spacing();
            ImGui.TextDisabled($"Pre-decomposition aggregate (spec 201 FR-005 sum check): opaque batched {live.OpaqueBatchedMdxCount}"
                + $"/unbatched {live.OpaqueUnbatchedMdxCount} of {mdxOpaqueTotal}"
                + $", transparent batched {live.TransparentBatchedMdxCount}/unbatched {live.TransparentUnbatchedMdxCount} of {mdxTransparentTotal}.");

            // Same-flight before/after for Spec 153 US3. Off reproduces the recorded 100%-unbatched
            // baseline exactly, so the comparison does not depend on reflying the route.
            bool mdxBatching = _worldScene.MdxOpaqueBatchingEnabled;
            if (ImGui.Checkbox("Opaque MDX batching (Spec 153 US3)", ref mdxBatching))
                _worldScene.MdxOpaqueBatchingEnabled = mdxBatching;
            ImGui.TextDisabled("Off = per-instance state setup per draw (the recorded baseline).");

            bool worldDoodadAnimation = _worldScene.WorldDoodadAnimationEnabled;
            if (ImGui.Checkbox("Animate world doodads", ref worldDoodadAnimation))
                _worldScene.WorldDoodadAnimationEnabled = worldDoodadAnimation;
            ImGui.TextDisabled("Off = only WMO doodads auto-animate. Watch the MdxAnimation stage timer.");

            // Spec 202 Phase 3. This is the only switch here that changes the number of draw
            // calls rather than the setup cost per draw, so it is the one to flip when comparing.
            if (MdxRenderer.GpuInstancingShaderAvailable)
            {
                bool gpuInstancing = MdxRenderer.GpuInstancingEnabled;
                if (ImGui.Checkbox("GPU instancing for opaque models (Spec 202)", ref gpuInstancing))
                    MdxRenderer.GpuInstancingEnabled = gpuInstancing;
                ImGui.TextDisabled("On = one draw call per model instead of one per instance.");
            }
            else
            {
                ImGui.TextColored(new Vector4(1f, 0.55f, 0.2f, 1f),
                    "GPU instancing unavailable: the instanced vertex shader did not compile on this driver.");
            }
            ImGui.TreePop();
        }

        // Spec 151. WmoSubmission is the measured owner of the remaining hitching, and submission
        // counts alone cannot say why a scene admitted the geometry it did. This names the rule.
        if (ImGui.TreeNode("WMO admission (this frame)###FrameHistoryWmoAdmission"))
        {
            DrawWmoAdmissionCounters(_worldScene.LastRenderFrameStats.WmoAdmission);
            ImGui.TreePop();
        }

        // Localise the unaccounted time to a region of Render(). Peaks are what matter: hitches are
        // transient, so a live reading will almost never land on the bad frame.
        // Default-open: when unaccounted dominates, this is the only node that says where.
        ImGui.SetNextItemOpen(unaccountedDominates, ImGuiCond.Once);
        if (ImGui.TreeNode("Unaccounted breakdown (peaks)###FrameHistoryRegions"))
        {
            ImGui.TextDisabled("Untimed time inside Render(), split by region. Peak since reset.");
            ImGui.Text($"{"Prologue (setup before passes)",-32} now {_worldScene.RenderPrologueMs,7:0.00}  peak {_worldScene.RenderProloguePeakMs,8:0.00} ms");
            ImGui.Text($"{"Pass gap (between stage timers)",-32} now {_worldScene.RenderPassGapMs,7:0.00}  peak {_worldScene.RenderPassGapPeakMs,8:0.00} ms");
            ImGui.Text($"{"Epilogue (after passes)",-32} now {_worldScene.RenderEpilogueMs,7:0.00}  peak {_worldScene.RenderEpiloguePeakMs,8:0.00} ms");

            // PrepareObjectPhase now has its own stage timer (it appears in the stage table), so it
            // is no longer part of the pass gap. These sub-probes attribute its internals, which is
            // what names the periodic stall.
            ImGui.Separator();
            ImGui.TextDisabled("Inside PrepareObjectPhase (now timed; also in the stage table)");
            ImGui.Text($"{"  PrepareObjectPhase total",-32} now {_worldScene.ObjectPhasePrepareMs,7:0.00}  peak {_worldScene.ObjectPhasePreparePeakMs,8:0.00} ms");
            ImGui.Text($"{"    AudioRuntime.Update",-32} now {_worldScene.AudioRuntimeUpdateMs,7:0.00}  peak {_worldScene.AudioRuntimeUpdatePeakMs,8:0.00} ms");
            ImGui.Text($"{"    PM4 overlay window",-32} now {_worldScene.Pm4OverlayWindowMs,7:0.00}  peak {_worldScene.Pm4OverlayWindowPeakMs,8:0.00} ms");
            // The residual is the rest of the pass: render-diag scan, camera extraction,
            // GetChunkInfoAt, frustum update, GL state. If the stall lives here, neither sub-probe
            // will match it and the pass must be subdivided further rather than a fix guessed.
            double objectPhaseResidualNow = Math.Max(0,
                _worldScene.ObjectPhasePrepareMs - _worldScene.AudioRuntimeUpdateMs - _worldScene.Pm4OverlayWindowMs);
            ImGui.Text($"{"    other (unprobed remainder)",-32} now {objectPhaseResidualNow,7:0.00}       (peak: subdivide)");
            if (ImGui.Button("Reset region peaks"))
                _worldScene.ResetRenderRegionPeaks();
            ImGui.TreePop();
        }

        // The real per-frame series, copied without allocating. This is live every UI frame even
        // though the statistics above refresh on a cadence.
        int plotCount = history.CopyRecentTotalMs(_frameHistoryPlotBuffer);
        if (plotCount > 0)
        {
            ImGui.PlotLines(
                "##FrameHistoryPlot",
                ref _frameHistoryPlotBuffer[0],
                plotCount,
                0,
                $"frame ms (last {plotCount})",
                0f,
                (float)Math.Max(total.MaxMs * 1.1, _frameHistoryHitchThresholdMs * 1.5),
                new Vector2(0, 60f));
        }

        // Stable ImGui IDs via ###: the visible label carries a changing count, and ImGui derives the
        // widget ID from the label, so a bare interpolated label collapses the node whenever the
        // count ticks.
        if (snapshot.Hitches.Count > 0
            && ImGui.TreeNode($"Recent hitches ({snapshot.Hitches.Count})###FrameHistoryHitches"))
        {
            int shown = 0;
            for (int i = snapshot.Hitches.Count - 1; i >= 0 && shown < 12; i--, shown++)
            {
                WorldRenderHitch hitch = snapshot.Hitches[i];
                if (hitch.IsDominatedByUnaccountedTime)
                {
                    ImGui.TextColored(new Vector4(1f, 0.55f, 0.2f, 1f),
                        $"frame {hitch.FrameIndex}: {hitch.TotalCpuMs:0.0} ms  <- UNACCOUNTED ({hitch.UnaccountedMs:0.0} ms, no timer)");
                }
                else
                {
                    ImGui.Text($"frame {hitch.FrameIndex}: {hitch.TotalCpuMs:0.0} ms  <- {hitch.DominantStage} ({hitch.DominantStageMs:0.0} ms)");
                }
            }
            ImGui.TreePop();
        }

        // Sorted by MAX, not p99. A stage that fires rarely but costs 300 ms has a near-zero p99, so
        // p99 ordering hides precisely the thing a hitch hunt is looking for.
        if (ImGui.TreeNode("Stage cost (max, worst first)###FrameHistoryStages"))
        {
            int shown = 0;
            foreach ((WorldRenderStage stage, WorldRenderTimingDistribution dist) in snapshot.StagesByMaxDescending())
            {
                if (dist.MaxMs <= 0.0001 || shown++ >= 12)
                    continue;
                ImGui.Text($"{stage,-26} median {dist.MedianMs,6:0.00}  p99 {dist.P99Ms,6:0.00}  max {dist.MaxMs,7:0.00} ms");
            }
            ImGui.TextDisabled($"{"SUM of stage maxima",-26} {SumStageMaxima(snapshot),29:0.00} ms  vs total max {total.MaxMs:0.00} ms");
            ImGui.TreePop();
        }

        if (ImGui.TreeNode("Detector self-check"))
        {
            ImGui.TextWrapped(
                "Stall one frame by a known amount. If it does not appear above at that magnitude, "
                + "this view is not trustworthy and no measurement taken from it counts.");
            ImGui.SetNextItemWidth(120f);
            float stallMs = (float)_frameHistoryInjectStallMs;
            if (ImGui.SliderFloat("Stall (ms)", ref stallMs, 50f, 1000f, "%.0f"))
                _frameHistoryInjectStallMs = stallMs;
            if (ImGui.Button("Inject stall into next frame"))
                _worldScene.DebugInjectStallMs = _frameHistoryInjectStallMs;
            ImGui.SameLine();
            if (ImGui.Button("Reset history"))
            {
                history.Clear();
                _frameHistorySnapshot = null;
            }
            ImGui.TextDisabled($"Recorder overhead: {snapshot.RecorderOverheadMsPerFrame * 1000.0:0.00} us/frame");
            ImGui.TreePop();
        }
    }

    /// <summary>
    /// Specs 201 and 202 Phase 0 instrumentation. Reports one pass's model submissions by render
    /// path and by the gate that stopped each instance short of GPU instancing, plus the draw
    /// calls the pass actually issued. Diagnostic only; it changes no submission decision.
    /// </summary>
    private static void DrawModelSubmissionCounters(string label, WorldModelSubmissionStats stats)
    {
        ImGui.Text($"{label}: instanced {stats.Instanced,6}  state-hoisted {stats.StateHoisted,6}  unbatched {stats.Unbatched,6}   (total {stats.Total})");

        // The number the batching work is actually trying to move. Instances are not draw calls:
        // a model draws once per geoset or section, so this is counted at the GL call sites.
        ImGui.Text($"  draw calls {stats.DrawCalls,6}  per instance {stats.DrawCallsPerInstance,6:0.00}"
            + $"   distinct models {stats.DistinctModelCount,5} (per-model instancing floor)");

        if (stats.StateHoisted > 0)
        {
            // Spec 202 research R1: the defect that made every previous before/after unreadable.
            ImGui.TextDisabled("  state-hoisted still issues one draw per instance - it saves setup, not draws.");
        }

        if (stats.Total > 0)
        {
            ImGui.Text("  by render path:");
            DrawPathRow("AdapterSkin", stats.AdapterSkin);
            DrawPathRow("AdapterEmbeddedProfile", stats.AdapterEmbeddedProfile);
            DrawPathRow("NativeEmbeddedProfile", stats.NativeEmbeddedProfile);
            DrawPathRow("ConversionFallback", stats.ConversionFallback);
            DrawPathRow("MdxDirect", stats.MdxDirect);
            DrawPathRow("Unknown (no route decision)", stats.Unknown);
        }

        if (stats.FadedInstanced > 0)
        {
            // Spec 207 US1: this population used to be forced unbatched by the fade gate, one draw
            // call each. Its size is the measure of what removing that gate was worth.
            ImGui.TextColored(new Vector4(0.4f, 0.9f, 0.5f, 1f),
                $"  of which distance-faded          {stats.FadedInstanced,6}  (previously 1 draw call EACH)");
        }

        ImGui.Text("  blocked from instancing by:");
        DrawGateRow("batching toggle off", stats.GatedBatchingDisabled);
        DrawGateRow("route requires unbatched render", stats.GatedRouteRequiresUnbatchedRender);
        DrawGateRow("renderer has no GPU instancing", stats.GatedGpuInstancingUnsupported);
        DrawGateRow("distance fade below 0.999", stats.GatedOpaqueFadeBelowThreshold);
        DrawGateRow("pass has no batch path", stats.GatedPassHasNoBatchPath);
        if (stats.GatedRendererUnavailable > 0)
        {
            ImGui.TextColored(new Vector4(1f, 0.55f, 0.2f, 1f),
                $"    no renderer resolved            {stats.GatedRendererUnavailable,6}  (never reached the GPU)");
        }

        if (stats.DominantGate != WorldModelBatchGate.None)
            ImGui.TextDisabled($"  largest gate: {stats.DominantGate}");

        static void DrawPathRow(string name, WorldModelPathStats path)
        {
            if (path.Total == 0)
                return;

            ImGui.Text($"    {name,-30} instanced {path.Instanced,6}  hoisted {path.StateHoisted,6}  unbatched {path.Unbatched,6}");
        }

        static void DrawGateRow(string name, int count)
        {
            if (count == 0)
                return;

            ImGui.Text($"    {name,-30} {count,6}");
        }
    }

    /// <summary>
    /// Spec 151 group-admission instrumentation. Reports which rule admitted or rejected WMO
    /// placements and groups. This is diagnostic only; it changes no admission decision.
    /// </summary>
    private static void DrawWmoAdmissionCounters(WmoAdmissionStats admission)
    {
        ImGui.TextDisabled("Two layers: which placements entered the visible set, then which groups");
        ImGui.TextDisabled("inside them were submitted and on whose authority.");
        ImGui.Separator();

        ImGui.Text($"Placements: considered {admission.PlacementsConsidered,6}  admitted {admission.PlacementsAdmitted,6}");
        ImGui.Text($"  rejected  hidden {admission.PlacementsRejectedHidden,5}  off-frustum+cone {admission.PlacementsRejectedOffFrustumAndCone,5}"
            + $"  distance {admission.PlacementsRejectedDistance,5}");
        ImGui.Text($"            max-view {admission.PlacementsRejectedMaxViewDistance,5}  projected-size {admission.PlacementsRejectedProjectedSize,5}"
            + $"  not-resident {admission.PlacementsRejectedAssetNotReady,5}");

        ImGui.Separator();
        ImGui.Text($"Groups: considered {admission.GroupsConsidered,6}  admitted {admission.GroupsAdmitted,6}  rejected {admission.GroupsRejected,6}");
        ImGui.Text($"Placement evaluations: {admission.GroupPlacementEvaluations,5}"
            + $"   mean admitted/placement {admission.MeanGroupsAdmittedPerPlacement,8:0.0}"
            + $"   worst {admission.MaxGroupsAdmittedInOnePlacement,6}");
        if (!string.IsNullOrEmpty(admission.WorstPlacementModelKey))
            ImGui.TextDisabled($"  worst placement: {admission.WorstPlacementModelKey}");

        ImGui.Separator();
        ImGui.Text("Admitted by rule:");
        DrawRule("runtime visibility disabled", admission.AdmittedByRuntimeVisibilityDisabled);
        DrawRule("placement transform invalid", admission.AdmittedByPlacementTransformInvalid);
        DrawRule("portal conservative fallback", admission.AdmittedByPortalFallback);
        DrawRule("portal traversal only", admission.AdmittedByPortal);
        DrawRule("frustum union only", admission.AdmittedByFrustum);
        DrawRule("portal + frustum", admission.AdmittedByPortalAndFrustum);
        DrawRule("gpu-instanced shell", admission.AdmittedByGpuInstancedShell);

        if (admission.PortalFallbackEvaluations > 0)
        {
            ImGui.TextColored(new Vector4(1f, 0.55f, 0.2f, 1f),
                $"Portal fallback fired on {admission.PortalFallbackEvaluations} of {admission.GroupPlacementEvaluations} evaluations"
                + $" (first reason: {admission.FirstPortalFallbackReason ?? "unknown"})");
            ImGui.TextWrapped(
                "A conservative fallback admits every group in the placement. While it fires, portal "
                + "culling is not reducing anything and the reason above is the thing to fix.");
        }

        if (admission.DominantGroupAdmissionRule == WmoGroupAdmissionRule.Frustum
            || admission.DominantGroupAdmissionRule == WmoGroupAdmissionRule.PortalAndFrustum)
        {
            ImGui.TextColored(new Vector4(1f, 0.55f, 0.2f, 1f),
                "Most groups are admitted by the post-portal frustum union, which never rejects.");
        }

        void DrawRule(string label, int count)
        {
            double share = admission.GroupsAdmitted == 0 ? 0d : 100.0 * count / admission.GroupsAdmitted;
            if (count == 0)
                ImGui.TextDisabled($"  {label,-30} {count,8}");
            else
                ImGui.Text($"  {label,-30} {count,8}   ({share:0.0}%)");
        }
    }
}
