# Active Context — wow-viewer

Last updated: 2026-09-06

**Start with [Spec status](../specs/STATUS.md), then the owning Spec Kit artifacts.** This file is
the current operational dashboard, not a historical plan.

## Active — Spec 223 acceptance remediation

- **Operator-reported failures:** Fog Start/Fog End rubberband; WMO-only maps do not render their
  global WMO; Archaeology > Playback & Capture lacks visible record/path-playback/apply actions; and
  the visible camera rig is not the requested interactive 3D spatial HUD.
- **Source repairs built:** every fog editor uses `WorldScene` user override state; WMO-only global
  MODF placements enter the external WMO admission collector without changing readers; Playback &
  Capture hosts canonical Capture Automation / Camera Path controls and **Apply playback to next
  capture**; scene-only recording does not change UI-chrome state; recording cleanup stops only
  archaeology playback it started. The decorative camera rig hides with Tab.
- **Source evidence:** isolated viewer Debug build passed with 0 errors; focused
  WDT/WMO visibility-policy tests passed 29/29. Neither proves slider input, GL WMO rendering,
  ffmpeg capture/output, path capture, layout quality, or interactive spatial UI.
- **Next bounded action — operator-owned:** run the retest matrix in
  [Spec 223 Phase 6 evidence](../specs/223-ui-consolidation-audit/evidence/phase6-validation.md)
  with an actual client and `ffmpeg`. Record client root, exact build, map, ffmpeg version, output
  path, fog result, WMO result, and produced video result. Do not close the operator gate from a
  build or unit test.

## Spec 212 — Phases 1–2 source complete; interactive spatial UI still open

- `CameraHudTransform` / `CameraSpaceProjection` own normalized camera basis and depth-clamped
  placement. `CameraHudRig` renders before ImGui, shares Tab/settings visibility, and loads the
  committed reticle, compass, bezel, gimbal, and brackets. Focused Core tests passed 3/3; the
  isolated viewer Debug build passed with 0 errors.
- Gates 1–2 live camera-lock/Tab/mesh proof are operator-owned. The rig remains decorative geometry,
  not a spatial workbench: it has no panel surface, hit testing, pointer routing, Museum profile, or
  time widget. Phase 3 owns pointer interception and Phase 4 owns Museum/time.

## Other active constraints

- Reader boundaries: preserve existing MPQ, ADT, WMO, M2/MDX readers; do not modify
  `AlphaWdtWriter.cs` without a separately proven compatibility regression.
- Runtime proof is operator-owned. Never infer visual, FPS, audio, input, ffmpeg, or real-client
  behavior from source/build/test evidence.
- Keep shared contracts in Core and UI in the viewer. New specs/plans/tasks belong under `specs/`,
  not this memory bank.
- Preserve unrelated worktree changes. Stage named files only; no broad reset/stage operations.

## Handoff

- **Current target:** Spec 223 operator retest.
- **Completed slice:** source remediation plus documentation reconciliation; see
  [Phase 6 evidence](../specs/223-ui-consolidation-audit/evidence/phase6-validation.md).
- **Unproven gap:** live fog, WMO-only rendering, video capture, and UI acceptance.
- **Explicitly out of scope:** claiming Phase 6 accepted, altering format readers, or calling the
  decorative rig an interactive 3D HUD.

Older continuity detail is archived under [memory-bank/archive](archive/README.md).
