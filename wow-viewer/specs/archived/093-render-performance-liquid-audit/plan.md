# Implementation Plan: 093 Render Performance And WMO Liquid Audit

**Branch**: `093-render-performance-liquid-audit`

**Date**: 2026-07-06

**Spec**: `spec.md`

## Technical Context

- Active app: `wow-viewer/src/viewer/WoWViewer`.
- Shared runtime stats contracts: `wow-viewer/src/core/WowViewer.Core.Runtime/World`.
- WMO renderer owner: `wow-viewer/src/viewer/WoWViewer/Rendering/WmoRenderer.cs`.
- Runtime Stats panel owner: `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- Current facts from source audit:
  - WMO placements are rendered per visible placement, then per visible group/material batch inside `WmoRenderer.RenderWithTransform`.
  - MDX "batched" path is shared-shader submission, not true GPU instancing.
  - Only the first compatible MDX renderer seeds `BeginBatch`; each MDX renderer still owns its own VAOs, materials, textures, and draw calls.
  - WMO transparent work and WMO MLIQ liquid rendering were previously timed inside the MDX transparent submission bucket.
  - WMO MLIQ GL state already enables alpha blending and disables depth writes, so "opaque liquid" likely points to flat shader/material/order behavior rather than basic blend-state absence.

## Constitution Check

- `gillijimproject_refactor` is read-only reference only.
- New code remains in `wow-viewer`.
- First slice is diagnostic-only; no batching architecture rewrite and no liquid visual change.
- Validation is `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, with focused tests where touched.

## Phase 1: Make Runtime Stats Honest

1. Add WMO render counters for batch draws, fallback draws, liquid draws, doodad submissions, and visible group submissions.
2. Accumulate WMO counters into `WorldRenderFrameStats`.
3. Split WMO transparent submission time out of MDX transparent submission time.
4. Show the new counters in Runtime Stats.
5. Build and focused-test the stats/advisor path.

## Phase 2: Real Dense-Map Capture

1. Load staged `4_0_0_11927` Stormwind or a dense city equivalent.
2. Capture Runtime Stats with overlays off and normal object visibility.
3. Capture Runtime Stats with WMO hidden, then MDX hidden, then overlays hidden.
4. Record whether WMO draw pressure, MDX submission, terrain, overlay, asset loading, or memory dominates.
5. Decide the next implementation slice from the largest measured cost.

## Phase 3: WMO Liquid Visual Audit

1. Pick a WMO with visible MLIQ.
2. Capture WMO liquid draw count and WMO transparent timing.
3. Inspect whether liquid data type/color/alpha is plausible.
4. Compare current flat-color shader against native-client research notes for water shader families.
5. Implement only one visual correction per slice, starting with material alpha/color/order before texture/ripple/refraction.

## Validation

- Build: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- Focused tests: `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldRenderOptimizationAdvisorTests`.
- Manual: Runtime Stats screenshot or copied values from dense map frame.
