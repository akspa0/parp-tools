# Active Context — wow-viewer

Last updated: 2026-09-26 · Branch: `v0.6.0-dev`

## Fresh-chat route

1. [Spec status](../specs/STATUS.md) — 7 active epics (248–254), all **triage pending**.
2. This compact handoff.
3. [TRIAGE.md](../specs/TRIAGE.md) — the operator's Want / Drop / Later decisions.
4. Only then one epic's `spec.md` → `plan.md` → `tasks.md`, and only the archived design documents
   its `plan.md` adopts for the selected item.

[Progress ledger](progress.md), [memory archive](archive/README.md) and `specs/archived/` are
on-demand history, never default reading.

## Current lanes — operator P1s (both marked Want 2026-09-23)

| Lane | Code | Owed (operator) | Next agent step |
|---|---|---|---|
| **R-10** modern-data lighting perf (Epic 249) | T002, T005, T006 landed 2026-09-23; **T004's per-placement WMO gate reverted 2026-09-27** (operator saw <1 FPS; T009) | check FPS after the revert on `wow_classic_beta` 1.60.1 `Azeroth` (0 WMO placements batched while lights exist); T003/T007 captures | redesign WMO instancing only after a before/after capture |
| **R-39** CASC minimap streaming cost (Epic 249) | code landed and **reverted** 2026-09-27 (operator: performance worse); code = `c2e9517` | say whether to add per-tile read/decode/upload timing + GC counts, then capture on a CASC client | no minimap change before measurement |
| **Spec 256** modern-data streaming & M2 submission | P0 `77f6f02`, minimap `b15e553`, P1 `4f1e6b4`; capture showed ≈ 57 ms/M2 in texture work → P2a shared textures `c9cf7b6`, P2b off-thread decode `6a326d3`, P2c compressed DXT + authored mips `bad4884`; P2b fallback fix `328866d`; missing textures → error texture, no search `983c8ac`; D WMO doodad loads + shared cache `b09654b`; P3 native M2 instancing `2c609f3`; Perf panel out of PM4 service `93d7461`; A animation: cached key frames + GPU skinning `e437673`; T060/T092 file splits landed (`WorldAssetManager.cs` 1,576 lines, `WmoRenderer.cs` 1,937 lines) | T102 capture (MdxAnimation, entering animated areas) + T042 visual A/B (Utilities > Perf > Frame history > Submission efficiency) | operator captures (T015, T022, T042, T081, T102) |
| **Spec 257** unified video recording & automation | **Complete 2026-10-02**: fixed taxi node stop-crash (null deref), unified `RecordingCoordinatorService` in `Recording/`, CLI video recording automation (`--record-taxi-route`, `--record-camera-path`, `--record-feature-tour`), camera path & feature tour integration | Operator smoke test of manual and automated recording | None (code & spec complete; receipt in `evidence/receipt-spec257.md`) |
| **Spec 258** taxi playlists & marketing showreel tour | **Complete 2026-10-02**: multi-route taxi playlists, continuous multi-segment recording, dynamic showreel telemetry HUD, large centered cinematic zone transition banners, landmark callouts; strict tour-mode viewport gating | Operator smoke test of playlist playback and showreel overlays | None (code & spec complete; receipt in `evidence/receipt-spec258.md`) |
| **Spec 259** ground effects & WMO detail doodads | **Complete 2026-10-02**: dynamic grass/foliage on terrain and WMO surfaces (`MDDL`), client slope ($Z \ge 0.4$) & MCCV rules from recent wowdev.wiki disclosures, 1.60 CASC support, and instanced GPU rendering; UI controls in RenderQuality; receipt in `evidence/receipt-spec259.md` | Operator visual smoke on terrain maps and WMO surfaces | None (code & spec complete) |
| **Spec 260** engine diagnostics & hitch telemetry HUD | **Complete 2026-10-02**: real-time engine pipeline metrics (M2 instancing %, flora count, WMO batches, ADT chunks, RAM & draw calls), rolling hitch detection with dominant stage attribution, flashing crimson/amber alert badge, multi-stage expanded timing panel, UI toggles, and CLI flags; receipt in `evidence/receipt-spec260.md` | Operator smoke test during recording and live viewport preview | None (code & spec complete) |
| **Spec 261** WoW Forever terrain hole fidelity | **Complete 2026-10-04**: verified ADT v18 format conformance, 64-bit high-res hole decoding at MCNK offset 0x14 with Little-Endian mapping (`cellY * 8 + cellX`), 100% seam boundary continuity, real CASC test cases for WoW Forever 1.60.1 / 11.2.7 (Deathknell church crypt and graves); receipt in `evidence/receipt-spec261.md` | Operator visual smoke on WoW Forever (1.60.1) Deathknell/caves/crypts | None (code & spec complete) |
| **U-01** god-class decomposition (Epic 251) | **E1 + E3 + ViewerApp campaign landed 2026-09-25**: PM4 overlay → `Terrain/Pm4/` (`WorldScene.cs` 17,175 → 8,326); `ViewerApp` class 42,973 lines / 29 files → 2,512 / 6 (51 services + 3 static helpers under `Workbench/Services/`) | U01-T003 E1 smoke; U01-T007 / T013 smoke of every moved ViewerApp surface; U01-T009 E4 hover/click smoke; **Spec 255 T018** smoke of W0–W7 | E4 landed 2026-09-26; **Spec 255 W0–W7 landed 2026-09-27** (`WorldScene.cs` 8,275 → 4,922; 7 scene services + PM4 helpers). W8/W9 and E2 `Render()` split wait for R-10's after-capture |

Receipts: [E1](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e1-pm4-extraction-2026-09-25.md) ·
[ViewerApp campaign](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-viewerapp-extraction-2026-09-25.md) ·
[Spec 255 W0–W7](../specs/255-worldscene-decomposition/evidence/w0-w7-extraction-2026-09-27.md) ·
[E4 selection](../specs/251-epic-viewer-ux-and-code-health/evidence/u01-e4-selection-2026-09-26.md).
**Where code lives now:** viewer features are owned services under `src/viewer/WoWViewer/Workbench/Services/<Feature>/`
(namespace `WoWViewer`); they reach app state only through `IViewerAppHost` (implemented in
`ViewerApp_Host.cs`). `ViewerApp.cs` is the shell: fields, composition root, window lifecycle. New behaviour
goes into the owning service; if it needs more app state, add one `IViewerAppHost` member + its one-line
implementation. PM4 overlay code lives in `Terrain/Pm4/` (`WorldScene.Pm4Overlay.X`). World-scene features live in
`Terrain/Scene/<Feature>/` services (`WorldScene.HoverPick`, `.TaxiActors`, `.ObjectFilters`, `.Selection`, `.Atmosphere`,
`.ExternalSpawns`, `.TerrainQueries`); they read scene state only through `IWorldSceneHost` (explicit impls in `WorldScene.cs`).

All other epic items remain untriaged in [TRIAGE.md](../specs/TRIAGE.md); nothing else is scheduled.
Receipt for the 2026-09-23 reconciliation:
[reconciliation README](../specs/archived/reconciliation-2026-09-23/README.md).

**Linux build note (agent sessions):** the viewer builds and tests on Linux with
`-p:EnableWindowsTargeting=true` after `git submodule update --init` of the six `wow-viewer/libs/*`
submodules. 26 tests fail on `HEAD` there for environmental reasons (missing `test_data/`,
Windows-path expectations) — compare against that set, not zero. One Curation test rewrites
`data-harvester/tests/fixtures/spec122_curation_manifest/.../curation_run.json` with the local path;
revert it before staging.

## Release state

`eng/Version.props` = `0.6.0` / `InformationalVersion 0.6.0-alpha4`, release 2026-10-01 (tag
`v0.6.0-alpha4`, all four self-contained binaries). Notes: `docs/releases/v0.6.0-alpha4.md`.

**Shipped unverified in alpha2** (operator verification, tracked as V-tasks in the epics): the M2
texture-wrap fix (every M2 era); `LkAdtWriter` chunk-completeness fixes (17 call sites); no exported
map loaded in a client or Noggit; no DAT Cartography layer composed on screen; new export menu and
sidebar buttons never clicked.

## Measured gaps worth knowing before any work (details: TRIAGE §1)

- No map save pipeline exists (`MapSaveService` absent; `EditorSession.SaveAll()` writes nothing).
- Editor undo reverses only 2 of 5+ operation kinds.
- Zone music is disabled by policy and still reads `ZoneMusic` as a SoundEntries id.
- Modern data: the scene-light gate disables WMO instancing (~5.5 FPS, 16,431 WMO draws pre-R-10); R-10's per-placement gate made it worse (<1 FPS, operator) and was reverted 2026-09-27.
- v22 DAT blends layer 0 only; `AMAP` codec unidentified.
- `WorldScene.cs` 4,922 lines (after Spec 255 W0–W7, 2026-09-27), `ViewerApp` class 2,512 lines in 6 files — no new members (AGENTS.md §10).

## Non-negotiable constraints

- Preserve MPQ/ADT/WMO/M2/MDX readers and `AlphaWdtWriter`; no feature scope rides on a refactor.
- Runtime visual, input, FPS, audio, video and client-data proof are operator-owned.
- Training, harvests, GPU and cloud runs are operator-run; hand over PowerShell-ready commands.
- Preserve unrelated dirty work (`imgui.ini`); stage named files only.
- New scope needs operator-approved wording (AGENTS.md §9.1); receipts per §9.2.

## Handoff

**Immediate:** operator runs R10-T003/T007 captures and the U-01 smoke passes (U01-T003 PM4 overlay;
U01-T007/T013 every moved ViewerApp surface — checklist in the campaign receipt). U01-T009 E4 hover/click smoke and
[Spec 255](../specs/255-worldscene-decomposition/spec.md) T018 (W0–W7 smoke — checklist in its
[receipt](../specs/255-worldscene-decomposition/evidence/w0-w7-extraction-2026-09-27.md)) are owed too.
Next agent-owned U-01 work: Spec 255 W8/W9 and E2, after R-10's after-capture (R10-T007). Do not start any other epic
item before it is marked Want. Superseded dashboard:
[archive/2026-09-23-pre-reconciliation-active-context.md](archive/2026-09-23-pre-reconciliation-active-context.md).
