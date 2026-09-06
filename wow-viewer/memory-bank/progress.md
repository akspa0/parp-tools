# Progress — wow-viewer

Last updated: 2026-09-06

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
