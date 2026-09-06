# Progress — wow-viewer

Last updated: 2026-09-06

## 2026-09-06 — WoW-style shell + keybind profiles + reconstruction editor specced (Specs 229/230)

- **Spec 229** ([spec](../specs/229-wow-shell-keybind-profiles/spec.md)): the operator's REPEATEDLY
  unimplemented request for a **WoW-interface-style shell** is finally in writing, paired with
  **contextual keybind profiles** (Noggit/Noggit-Red reference) — active surface swaps binding set +
  visible action row; registry binds to canonical services; conflicts reported; bindings persisted.
- **Spec 230** ([spec](../specs/230-reconstruction-editor/spec.md)): reconstruction-first editing —
  Rosetta-indexed object placement through the staged pipeline, **New Map Generator** UI wrapping the
  Spec 192 CLI, save targets Alpha WDT/LK ADT (Cata/MoP gated on Spec 197's slot-aware writer).
- Both sequence after Spec 227's inventory v2; they compose existing tooling, no forks. STATUS.md
  registered; sequencing note added to activeContext.

## 2026-09-06 — UI re-audit + god-class freeze specced and bound (Specs 227/228)

- **Operator verdict recorded verbatim:** Editor tabs nonsensical, duplicate weak-signal amplifiers,
  Archaeology styling mismatch, Inspector data repeated 3–5×, multiple minimaps (only the tiny one
  teleports reliably), app unapproachable. **Spec 227** authors the full re-audit (inventory v2 with
  screenshots), one sidebar design system with dropdown sub-categories, and dedupe-with-teeth on the
  named duplicates. AGENTS.md §11 binds SharedUiWidgets-only styling and inventory registration.
- **God-class freeze in writing:** WorldScene.cs ~16.9k and ViewerApp.cs ~16.7k(+partials) are the
  context-window bottleneck; the ViewerApp_ partial split failed because partials share one state
  space. AGENTS.md §10 binds: no new god-class members, ~2,000-line file budget, owned-service
  extraction with receipts. **Spec 228** owns the phased, behavior-preserving extraction, sequenced
  after the Spec 227 audit so services land in their post-audit shape.
- **Wireframe evidence narrowed (Spec 226):** terrain wireframe ONLY on multiple-textured tiles
  (suspect: array-tile shader lacks the pass) and MDX/M2/WMO wireframes fully dead (suspect:
  batching paths never execute it). Diagnose with captures before changing anything.
- **Rules reminder now in force:** any new UI feature = inventory row + owned service class + a
  `tasks.md` receipt, or it does not land.

## 2026-09-06 — Slider order corrected, overlays rebalanced, hover occlusion, community credits (v0.5.2.3)

- **Fog sliders:** the "rubberband" the operator fought was mostly **the reversed slider order**
  (Fog End on top per the old Spec 223 T501 requirement) making them adjust the wrong slider.
  Amended: sliders now render **Fog Start above Fog End** in the shared editor, with a dated note in
  Spec 223 T501. The crossing-drag hardening and stale-binary fix remain.
- **Overlays:** neon cells toned from 0.85 → **0.45** mix (glow 0.4 → 0.3); chunk and tile grids
  upgraded to the same `fwidth` anti-aliased treatment and made **more obvious** (0.6→0.75 cyan,
  0.8→0.9 orange). Both terrain shaders.
- **Hover occlusion:** `UpdateWorldSceneHoveredAssetInfo` now terrain-occlusion-tests precise ray
  hits — an object behind a hill is no longer hovered "through the ground". The click path already
  culled hits behind the clicked terrain point; residual gaps there are recorded in Spec 226.
- **About box:** new **"Thanks to..."** banner — Marlamin, schlumpf, Dovah, Pirate the Explorer,
  fean, implave, IS4, Adspartan (Noggit), Skarn (Noggit-Red) — with the WoW Exploration community
  credit and the restoration-not-polish framing (Noggit/Noggit-Red are the preferred fine-tuning
  editors for this tooling's output).
- **Spec 226** ([spec](../specs/226-renderer-polish/spec.md)) authored verbatim from operator
  feedback: wireframe fails by camera angle (suspected fill/line depth-fighting) and MDX lighting
  lacks real specular. Not diagnosed or changed yet — both require before/after captures.
- **Build:** viewer Debug **0 errors**. Runtime visual proof operator-owned.

## 2026-09-06 — Fog rubberband ROOT CAUSE = stale binary; Cells glow/fade + toolbar order shipped (Spec 225 Phase 0)

- **Fog:** the operator's running viewer was **v0.5.2.2 — built before the fog fix existed in
  source** (remediation builds went to isolated output dirs because the running viewer locked the
  normal Debug output). Source path verified correct: override → `ResolveActiveFogRange` → UI reads
  `UserFogStart/End`. Hardened the one real defect: a crossing drag (Start >= End) now adjusts
  Start instead of snapping both sliders to the 200/1500 defaults. Version bumped to **v0.5.2.3**
  (title bar proves the build); normal Debug output build now succeeds 0 errors — close the viewer
  and relaunch via `dotnet run` to get the fix.
- **Cells overlay (Spec 225 T002):** both terrain shaders rewritten — `fwidth`-based line width
  (kills distance moiré), ~1px neon core + soft glow halo, sub-pixel detail fade, and `fogFactor²`
  dissolve so distant cells fade like every other surface.
- **Toolbar (Spec 225 T001):** bottom-bar grid order is now Tiles, Chunks, Cells. The Overhead
  ortho view mode itself is spec'd (225-T101..T103) but not started.
- **Builds:** viewer Debug **0 errors** (normal + isolated output). Runtime visual/slider proof is
  operator-owned.

## 2026-09-06 — GOVERNANCE: operator rejects fabricated scope; rules + cleanup skill land (Spec 224)

- **Trigger:** the operator rejected the decorative reticle/compass/bezel HUD as unrequested scope
  ("I didn't ask for 99.9% of what is there") and demanded systemic rules, receipts, and monthly
  cleanup.
- **Landed:** `AGENTS.md` §9 governance (scope freeze, receipts, write containment, spec-sync,
  context discipline, cleanup ledger); [Spec 224](../specs/224-speckit-governance/spec.md)
  spec/plan/tasks; `speckit-cleanup` skill authored in `.roo/skills/` and installed to the operator
  skill directory. Spec 212 Phase 2 rewritten with dated correction; `CameraHudRig` stripped to
  gimbal+brackets with `Enabled = false` (viewer Debug build 0 errors).
- **Open:** operator approval of the rules (Gate 1); first full cleanup run (Phase 2).

## 2026-09-06 — Spec 223 acceptance remediation documented; runtime proof remains open

- **Operator feedback recorded:** Fog sliders rubberbanded; WMO-only maps omitted their global WMO;
  Archaeology > Playback & Capture did not expose video recording, camera-path playback, or apply
  controls; and the requested interactive 3D spatial UI is absent.
- **Source remediation:** fog controls now read/write `WorldScene` user overrides; WMO-only global
  MODF instances use the external admission list; Archaeology now renders Capture Automation and
  Camera Path controls plus **Apply playback to next capture**. Scene-only recording preserves the
  existing UI-chrome state and stops only playback that it initiated. The decorative rig hides with
  Tab but remains non-interactive.
- **Verification:** isolated viewer Debug build passed with **0 errors**. Focused
  `WdtSummaryReaderTests|WmoAdmissionTallyTests|WorldObjectVisibilityCollectorTests` passed **29/29**.
  The tests do not cover GL-backed `WorldScene` admission, live slider behavior, `ffmpeg`, output
  validity, camera capture, or UI acceptance.
- **Spec 212 Phases 1–2:** `CameraHudTransform` / `CameraSpaceProjection` plus three focused tests
  formalize camera-space placement and HUD-depth clamping. `CameraHudRig` renders before ImGui,
  shares Tab/settings visibility, and loads/renders the authored reticle, compass, visor bezel,
  gimbal, and brackets. Focused tests pass **3/3** and an isolated viewer Debug build passes
  **0 errors**. Gates 1–2 live visual proof remains open; no interactive surface, pointer routing,
  Museum profile, or 3D time control is claimed.
- **Documentation:** [Spec 223 status](../specs/STATUS.md),
  [tasks](../specs/223-ui-consolidation-audit/tasks.md),
  [Phase 6 evidence](../specs/223-ui-consolidation-audit/evidence/phase6-validation.md), and the
  [user guide](../docs/WoWViewer/USERGUIDE.md) now distinguish source repairs from retest evidence.
  Pre-compression history is preserved in [archive](archive/README.md).

## 2026-09-05 — Spec 223 Phase 6 source gate passed; operator acceptance subsequently failed

- Taxi active-ride simulation, streaming camera playback, terrain inspection/pins, workbench route
  consolidation, shared widgets, and Quick-profile preservation had source/build/policy evidence.
- The later operator report supersedes any inference that source-gate success constituted user
  acceptance. See the current Spec 223 remediation entry above for the live validation contract.

## Durable active history

- **Spec 212:** authored OpenSCAD HUD assets are not interactive UI. Camera-space/rig source work is
  complete with visual Gate 1 open; surface/hit-test, pointer-routing, Museum, and time-control tasks
  remain authoritative in
  [Spec 212 tasks](../specs/212-spatial-ui-shell/tasks.md).
- **Spec 222:** Cartography is specified as a multi-map/multi-tile composition workbench; work starts
  with occupied-tile queries. See [Spec 222](../specs/222-map-composition-workbench/tasks.md).
- **Spec 221:** corpus validation found Alpha alpha-leg round-trip drift; object corpus validation and
  alpha-leg repair remain open. See [Phase 0 evidence](../specs/221-converter-validation-harness/evidence/phase0-baseline.md).
- **Specs 151/153/204:** renderer performance ownership and earlier measurement detail are durable in
  their specs; real-client benchmark/capture proof remains required for visual/FPS claims.

Historical session detail before 2026-09-06 is preserved in [memory-bank/archive](archive/README.md).
