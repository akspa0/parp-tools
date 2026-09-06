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

## Operator directive 2026-09-06 (third) — WoW-style shell, keybind profiles, reconstruction editor

- **Repeated request now in writing:** the operator has asked over and over for a **WoW-interface-
  style shell** that de-complicates the UI; nothing ever landed. Captured in
  [Spec 229](../specs/229-wow-shell-keybind-profiles/spec.md) together with the chosen simplification
  mechanism: **contextual keybind profiles** (Noggit/Noggit-Red style — active surface swaps the
  binding set and visible action row atomically).
- **Reconstruction editor direction:** [Spec 230](../specs/230-reconstruction-editor/spec.md) —
  Rosetta-indexed object placement (copy/paste asset paths into new maps/positions), a **New Map
  Generator** UI wrapping the Spec 192 generator, and save targets Alpha WDT / LK ADT now with
  Cata/MoP split gated on the Spec 197 writer. Editing model = reconstruction from existing data,
  NOT sculpting/hand-painting.
- **Sequencing:** 227 inventory v2 → 229 shell/keybinds + 228 extraction → 230 editor composition.
  230 composes existing tooling (192/190/197/220/222) — no forks.

## Operator directive 2026-09-06 (second) — UI re-audit + source decomposition

- **UI verdict:** only Viewer/Quick/Inspector are sane; Editor tabs are nonsensical; weak-signal
  amplifiers are duplicated; Archaeology styling mismatches; Inspector repeats data 3–5×; multiple
  minimaps with only the tiny one teleporting reliably. Mandate captured in
  [Spec 227](../specs/227-ui-reaudit/spec.md): full top-to-bottom audit (inventory v2), ONE sidebar
  design system with dropdown sub-categories, dedupe with teeth. AGENTS.md §11 binds the standard.
- **Source decomposition:** WorldScene.cs (~16.9k lines) + ViewerApp.cs (~16.7k + partials) are
  unworkable god classes. AGENTS.md §10 now BINDS: **no new members in `WorldScene`/`ViewerApp`**,
  ~2,000-line file budget, owned-service extraction with receipts. Owner:
  [Spec 228](../specs/228-source-decomposition/spec.md). Sequencing follows the Spec 227 audit so
  services are extracted to their post-audit shape.
- **Wireframe evidence narrowed (Spec 226):** terrain wireframe renders ONLY on multiple-textured
  tiles (suspect: array-tile shader lacks the wireframe pass) and MDX/M2/WMO wireframes are fully
  dead (suspect: batching paths never execute the wireframe pass). Undiagnosed; before/after
  captures are the required receipts.

## Operator correction 2026-09-06 — Spec 212 scope; governance rules now binding

- The decorative reticle/compass/bezel HUD elements were **unrequested scope** and are struck from
  Spec 212; the meshes are unused and `CameraHudRig.Enabled` defaults OFF. The real assignment is
  **ImGui panels composited onto 3D surfaces mounted to the camera frame** (offscreen ImGui texture
  → camera-space quad → composite before chrome → hit testing). Camera-space contracts survive as
  the correct foundation.
- **Governance is now binding** (`AGENTS.md` §9, owner [Spec 224](../specs/224-speckit-governance/spec.md)):
  scope freeze, receipts before any `[x]`, write containment, spec-sync, monthly `speckit-cleanup`
  (skill installed at `C:\Users\akspa\.roo\skills\speckit-cleanup\`). Ledger: last cleanup
  2026-09-06, next due 2026-10-01.
- Rules approval by the operator is Spec 224 Gate 1; the first full cleanup run is Phase 2.

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
