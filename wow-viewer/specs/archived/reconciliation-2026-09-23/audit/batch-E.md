# Batch E audit — viewer UI/UX shell, sidebars, UI audits, governance, source decomposition, capture automation

Specs: 009, 069, 072, 073, 110, 212, 223, 224, 225, 227, 228, 229, 231, 233.
Code roots checked: `wow-viewer/src/viewer/WoWViewer/` (`ViewerApp*.cs`, `Terrain/WorldScene.cs`,
`UI/SharedUiWidgets.cs`, `Workbench/`, `Capture/`).

**228 line-count baseline (measured 2026-09-23)**, budget = 2000 lines/file (AGENTS.md §10):

| File | Lines | Over budget |
|---|---:|---|
| `Terrain/WorldScene.cs` | 17,153 | 8.6x — single file, **zero partials** (`WorldScene_*.cs` does not exist) |
| `ViewerApp.cs` | 16,746 | 8.4x |
| `ViewerApp_Sidebars.cs` | 5,762 | 2.9x |
| `ViewerApp_Pm4Utilities.cs` | 4,178 | 2.1x |
| `ViewerApp_CaptureAutomation.cs` | 2,244 | 1.1x |
| `ViewerApp_Editor.cs` | 1,928 | under |
| 16 other `ViewerApp_*.cs` partials | 114–1,291 each | under |

Extraction pattern (god-class field + delegation into an owned service, per §10) has **not** been
applied to `WorldScene` or `ViewerApp` themselves — the existing `ViewerApp_*.cs` split is a
same-class partial split (shares one state space, explicitly called a failure to contain growth by
§10's own preamble), not a service extraction. No `src/core/.../World/Selection/` or
`Terrain/Scene/Selection/` directory exists (Spec 228's Core selection service, Phase 3–4).

---

### 009 Full Project Reimplementation Spec
- Stated status: no status line; dated 2026-05-22 design notebook | Tasks: no tasks.md
- Scope: a 28-section "design notebook" sufficient to reimplement the entire wow-viewer project
  from scratch in a new repo — format specs, rendering pipeline, ML pipeline, CLI surfaces, deep
  dives into PM4/WMO/M2/shaders/converters, written as documentation of the **existing** system.
- Verified implemented: N/A — this is a reference document describing already-shipped
  functionality (viewer, core I/O, ML pipeline), not a forward task list. Every "requirement" is a
  description of current behavior at time of writing.
- Partial: none tracked — no FR/US/T ids are meant to be checked off.
- Not implemented: none stated as open work.
- Checkbox accuracy: N/A (no checkboxes).
- Operator gates owed: none — it is not an execution contract.
- Open residue (spec-stated only): none. The document itself says its purpose is "design
  specification sufficient to fully reimplement... from scratch," i.e. a snapshot/reference, not a
  plan with unclosed items.
- Superseded by / overlaps: none directly; downstream specs (069, 223, 227, 231, 228) now own the
  UI portions it describes.
- Disposition: ARCHIVE-COLD
- Proposed epic theme: (none — historical reference, not epic residue)
- Confidence: high

---

### 069 Viewer UI Overhaul — Workbench Panel + Tab System + Archeology Playback
- Stated status: "In progress (Phase 15 — memory bank + spec sync)" | Tasks: no tasks.md (spec +
  plan only)
- Scope: replace sidebars with one Workbench popout, top-tab + bottom-tab (`TopTab`/`BottomTab`)
  navigation, Archeology as a first-class top tab with playback, minimap as a resizable sub-tab
  (FR-001–FR-035).
- Verified implemented: none of the spec's core mechanism exists — grep for `enum TopTab`,
  `enum BottomTab`, `DrawTopTabBar`, `DrawBottomTabBar`, `ArcheologyPlaybackConfig` returns zero
  hits anywhere in `src/viewer/WoWViewer/`.
- Partial: `DrawInteractiveMinimapSurface(...)` does exist and already takes a size parameter
  (`ViewerApp_MinimapAndStatus.cs:413`), matching the spec's FR-033 assumption, but it's used from
  the legacy sidebar/fullscreen paths, not a "Minimap sub-tab."
- Not implemented: FR-001 (`ShellPanelId` removal) — the enum is still live with 120 references
  repo-wide, including `Left`/`Right` lanes (`ViewerApp.cs:70`); FR-017/FR-018 (Archeology tab) —
  `_showUniqueIdArchaeologyWindow` is still a **floating window** (`ViewerApp.cs:816`,
  `ViewerApp_Sidebars.cs:3842: ImGui.Begin("UniqueId Archaeology", ...)`), not a tab.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: none — the mechanism was never built, so there is nothing to witness.
- Open residue (spec-stated only): none carried forward — the entire top/bottom-tab-bar design was
  abandoned in favor of a different pattern (see below); nothing in 069 should re-enter a new epic
  as-is.
- Superseded by / overlaps: the codebase instead grew a `Workbench/` navigator
  (`WorkbenchNavigator.cs`, `WorkbenchTab.cs`, `Workbench/Pages/*`) used by the Editor profile —
  this is Spec 231's shape, not 069's. 223 and 227 are the specs that actually consolidated
  navigation (four-profile tabs), superseding 069's six-top-tab design.
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: UI & Approachability (superseded shape, no residue)
- Confidence: high

---

### 072 Sidebar Resize Cleanup
- Stated status: no status line, hotfix task list | Tasks: 9/9 checked (T001–T009)
- Scope: fix sidebar-resize splitters fighting the cursor, and toolbar being overwritten by
  sidebars, in the 071 tab-mode layout.
- Verified implemented:
  - T001/T002/T003/T004 — current tab-mode `DrawLeftSidebar()` (`ViewerApp_Sidebars.cs:543`) does
    **not** call `DrawFixedSidebarWidthControl`; that call only remains in
    `DrawLegacyLeftSidebar()` (line 602), the dockspace/legacy fallback path — matches "remove from
    DrawLeftSidebar/DrawRightSidebar."
  - Edge splitters computed from absolute mouse delta at `ViewerApp_Sidebars.cs:1978-2037`
    (`_leftSidebarWidth`/`_rightSidebarWidth` via `ClampFixedSidebarWidth`).
  - T006 — `DrawToolbar()` is called after `DrawLeftSidebar()`/`DrawRightSidebar()` in `DrawUI()`
    (`ViewerApp.cs:1829-1848`), matching "toolbar renders on top."
- Partial: **T005 is checked but not accurately implemented.** The spec/task says "Change
  `DrawToolbar` main window to span only the scene viewport width (`viewportX` to
  `viewportWidth`)." The live code (`ViewerApp_Sidebars.cs:344-345`) still sets
  `toolbarWidth = io.DisplaySize.X` — full display width, not viewport-scoped. T006's "renders on
  top if any overlap remains" masks this rather than the width fix actually landing.
- Not implemented: none beyond the T005 gap above.
- Checkbox accuracy: 8 accurate, 1 checked-but-inaccurate (T005 — toolbar still spans full
  display width, not the viewport-only width the task claims).
- Operator gates owed: none stated (hotfix, build-gated only; T008/T009 build+commit).
- Open residue (spec-stated only): T005's actual width scoping, if the operator still wants it —
  but this was never separately re-opened anywhere else, so treat as closed residue absorbed by
  073/227 toolbar work rather than reopening 072 itself.
- Superseded by / overlaps: 073a (toolbar/sidebar dedup), 227 (sidebar standardization).
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: UI & Approachability
- Confidence: high

---

### 073 UI Surface Revamp
- Stated status: no status field; plan only, 4 sub-plans (073a–073d) | Tasks: no tasks.md
  (speckit-tasks never run)
- Scope: dedupe toolbar/left-sidebar controls, surface converter tools in a Tools tab, polish
  tab/sub-tab alignment, polish Model/World/Terrain panel alignment. Explicitly "surface only," no
  deletions.
- Verified implemented: 073b landed, later, under a different name — a **Converters sub-tab**
  exists: `ViewerApp_Sidebars.cs:5658 DrawConvertersSubTabContent()`, reached via
  `ToolsBottomTab.Converters` → `OpenWorkbenchTab(WorkbenchTab.Editor, 3) // Converters (Spec 231)`
  (line 4921) and exposed through `Workbench.Pages.IEditorPageHost.DrawConverters()`
  (`ViewerApp_Editor.cs:94`). The in-code comment attributes it to Spec 231, not 073.
- Partial: 073a/073c/073d (toolbar/sidebar dedup, tab alignment, panel alignment polish) have no
  independently attributable evidence — later specs (072 splitters, 227 SharedUiWidgets pass)
  cover the same ground generically, so it's not possible to say 073 itself was executed as
  written.
- Not implemented: `View > Legacy UI` toggle as named — `_useDockspaceUi` exists and gates the
  legacy path, but there's no menu item literally called "Legacy UI" found.
- Checkbox accuracy: N/A (no tasks.md ever generated for this plan).
- Operator gates owed: none tracked (spec never reached an execution/task phase).
- Open residue (spec-stated only): none worth carrying as 073-owned — its one concretely-landed
  piece (Converters tab) is already owned and dated under 231; the rest was never planned into
  tasks.
- Superseded by / overlaps: 231 (Converters page, "Editor is a mess" reorg), 227 (SharedUiWidgets
  standardization), 072 (toolbar/sidebar splitter fix).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: UI & Approachability
- Confidence: medium (073 was never task-tracked, so "implemented" vs "coincidentally covered by
  231/227" can't be fully disentangled from source alone)

---

### 110 Viewer Stabilization
- Stated status: "Draft" | Tasks: 47/65 checked (checked ~72%; 18 explicitly unchecked)
- Scope: huge multi-US spec — fog/lighting visibility (US1), LIT marker overlay+navigation (US5),
  synthesized terrain-minimap export (US6, the bulk of the FR list), fog-slider/Archeology-tab
  reachability (US7), native M2 rendering instead of MDX-conversion fallback (US2), Tools-menu
  cleanup (US3), WMO/M2 conversion capability documentation (US4). Only the UI-shell-relevant
  slices (US1/US5/US7 controls, US3 Tools menu, and the M2-fallback-in-render-path question) were
  audited in depth for this batch; the minimap-compositor FR-023* family (US6) is a
  rendering/dataset concern, not UI shell, and was not re-verified line-by-line here.
- Verified implemented:
  - US7 fog sliders — `ImGui.SliderFloat("Fog Start", ...)` / `SliderFloat("Fog End", ...)` at
    `ViewerApp_Settings.cs:83-84` (visible grab, not drag-only), matching FR-024.
  - US1 — `UseLitFogOverride` checkbox + fallback fog exist (`WorldScene.cs:3993-4049`,
    `ViewerApp.cs:9427-9429`).
  - US3 (partial) — no "MK Dataset"/"VLM Dataset" menu launchers found anywhere in
    `src/viewer/WoWViewer/*.cs` (only unrelated string literals in `VlmProjectLoader.cs`), matching
    FR-009 even though T035-T039 are unchecked.
- Partial: US3 Tools-menu audit tasks (T035-T039) are unchecked and there's no
  `docs/architecture/viewer-conversion-capability-2026-07-16.md` (US4/FR-012) on disk, matching
  their unchecked state.
- Not implemented (confirmed against FR-008/US2): **`WorldAssetManager.cs` still constructs
  `MdxRenderer` as an M2 rendering fallback via `ConvertM2ToMdx(...)`**
  (`Terrain/WorldAssetManager.cs:1478-1528`, used at lines 1405/1462/1496), directly contradicting
  FR-008 "MUST NOT use M2-to-MDX conversion as a renderer fallback." Matches unchecked
  T028-T032 (native M2 runtime bridge unconditional route not wired).
- Checkbox accuracy: accurate for the parts sampled (US2/US4 unchecked-and-absent; US1/US7
  checked-and-present; US3 unchecked-but-partially-present for the MK/VLM removal specifically).
- Operator gates owed: T019 (two-map visual proof), T034 (native M2 test/build run) — both
  explicitly unchecked and stated as such.
- Open residue (spec-stated only): US2/FR-006–FR-008 (native M2 route, remove MDX-conversion
  fallback — T028-T032); US3/FR-010 (Tools menu inventory + dependency diagnostics — T035-T039);
  US4/FR-011-FR-012 (published WMO v14/v17 and M2→MDX capability tables with fixture evidence —
  T040-T044).
- Superseded by / overlaps: 235 (Legacy MDX/M2 Rendering) is the newer, explicitly-scoped owner of
  the native-M2-vs-MDX-fallback problem (STATUS.md #10) — 110's US2 residue should fold there, not
  into a new UI epic.
- Disposition: FOLD (US1/US5/US7 UI-reachability slices — ARCHIVE-COMPLETE-enough for a UI epic;
  US2 residue routes to 235; US3/US4 residue is small and could fold into a UI/Tools cleanup epic)
- Proposed epic theme: UI & Approachability (US3/US7 residue only); US2 routes to 235 instead
- Confidence: medium (spec is huge; only UI-relevant slices deep-verified per batch scope)

---

### 212 3D Spatial UI Shell
- Stated status: "Phase 1 source-complete; Phase 2 REWRITTEN per operator correction 2026-09-06"
  | Tasks: 6/19 checked (Phases 1-2 partial; Phases 3-4 entirely open)
- Scope: render ImGui panels onto positioned 3D surfaces in the scene so panels stop subtracting
  viewport space (US1/US2); a top-bar-driven "rig" that re-mounts panel sets per task (US3);
  authored OpenSCAD shell geometry (US4); a 2D-mode escape hatch (US5); silhouette-accurate
  selection/hover outlines instead of bounding boxes (US6, independently shippable); a "museum
  profile" minimal camera-locked HUD (US7); 3D-shaped controls e.g. a clock-face time-of-day dial
  (US8).
- Verified implemented: only camera-space math scaffolding —
  `CameraHudTransform`/`CameraSpaceProjection` (Core.Runtime) and `CameraHudRig` in
  `Rendering/CameraHudRig.cs`, wired into `ViewerApp.OnRender()`. Original decorative
  reticle/compass/bezel scope was operator-struck 2026-09-06 (documented in-spec and in 224-T103).
- Partial: `CameraHudRig.Enabled` **defaults to `false`** (`CameraHudRig.cs:24`) and the OpenSCAD
  HUD primitives (T201) are committed but explicitly noted as "unused until a panel-surface design
  consumes them."
- Not implemented: T203-T205 (render ImGui panel to offscreen texture, map onto camera-frame quad,
  composite it — the actual US1/US2 mechanism) — zero code found; Phase 3 `SpatialUiHitTestService`
  (Core.Geometry) — does not exist anywhere in the repo; Phase 4 Museum Profile / 3D clock widget —
  does not exist. **US6 (silhouette selection outlines) has no task entries in tasks.md at all**
  despite being P1 and spec-stated as independently shippable with no dependency on US1-US5;
  current selection highlight is still `Terrain/BoundingBoxRenderer.cs` (a box), confirming the
  residue is real and unaddressed.
- Checkbox accuracy: accurate (nothing claimed beyond the scaffolding is checked).
- Operator gates owed: Gate 1/Gate 2/Gate 3/Gate 4, all unchecked; no operator visual witness
  exists because the underlying mechanism isn't built yet.
- Open residue (spec-stated only): US6 selection-silhouette outlining (FR-024–FR-029, P1,
  independent of the rest) is real, spec-stated, unimplemented, and currently un-tracked in
  tasks.md — worth explicitly re-opening in any successor epic even though the rest of 212 (panels-
  as-3D-objects, museum profile) is a much larger, unstarted, lower-certainty bet.
- Superseded by / overlaps: 231/227 took the conventional (non-spatial) Workbench/tab-standardization
  route instead — the actual shipped UI direction diverges from 212's premise.
- Disposition: FOLD (US6 residue only) / rest is ARCHIVE-COLD (foundational scaffolding with
  `Enabled=false`, no consuming feature, superseded in practice by 227/231's conventional approach)
- Proposed epic theme: UI & Approachability (US6 selection-outline residue); rest not carried
- Confidence: high

---

### 223 Viewer UI Consolidation Audit
- Stated status: "Implementing Phase 6; interactive acceptance pending" | Tasks: 9/29 checked
  (Phase 6 T601-T608 + source gate checked; all of Phase 1-5 T101-T502 unchecked)
- Scope: operator-directed audit+consolidation — catalog every UI surface (US1), one authoritative
  detail surface per object type via a unified Inspector (US2/US2a), WMO doodad-set switching from
  selection (US3), merge Editor+Archaeology then re-split "true editor" work (US4), Quick-tab
  improvements including a default Fog End the operator resets every session.
- Verified implemented:
  - US3 doodad-set switching **is implemented** despite T103 being unchecked:
    `DrawHoveredWmoDoodadSetCombo()` (`ViewerApp_Sidebars.cs:4764`), `ActiveDoodadSet`/
    `GetDoodadSetName()` on `WmoRenderer`, wired to hover (line 271-297) and to a selected-WMO
    section (line 1425-1432) — "unchecked-but-present."
  - T601-T608 (Phase 6, later/different approach than T101-T502): taxi/camera-path fix, terrain
    Inspector blanking fix + click-to-pin, Scene tab absorbed into Inspector, Utilities dispersed,
    MCNK flags integrated into Inspector's ADT section, `WoWViewer.UI.SharedUiWidgets` built
    (`UI/SharedUiWidgets.cs`, 220 lines), Quick-tab/profile state preservation, USERGUIDE update —
    all with a passing Phase 6 source/build gate checked.
  - `surface-inventory.md` exists at the spec root (US1's catalog deliverable).
- Partial: the originally-planned Inspector consolidation-by-object-type (T101-T107, "ADT/WMO/MDX/
  M2/PM4/WL* sections absorbing MCNK Explorer, ADT Chunk Investigation, Model Info, PM4 detail
  pages, WL Liquid Investigation") never happened as specced — Phase 6 took a narrower, different
  shape (Scene tab folded into Inspector, MCNK flags folded in) rather than the full six-section
  Inspector redesign.
- Not implemented: T201-T203 (retire floating windows, merge World Overview/World Maps/Chunk
  Clipboard duplicates, retire Terrain Workbench window); T301-T302 (Editor content into merged
  profile, Cartography under Archaeology); T401 (dedicated Editor profile split-back); T501-T502
  (shared Quick mirror component, single Fog Defaults implementation feeding Quick).
- Checkbox accuracy: 1 unchecked-but-present found (T103, doodad-set switching); everything else
  sampled was accurate. Per 224's 2026-09-11 governance audit this spec was previously found
  "clean" for its checked set.
- Operator gates owed: T609 (operator UI-acceptance retest of fog/WMO-only/Playback&Capture paths)
  and T610 (record acceptance result separately from build evidence) — both explicitly unchecked
  and are the current top STATUS.md item (#7, "223 residual").
- Open residue (spec-stated only): T101-T107 (unified Inspector-by-object-type + Gate A);
  T201-T203 (floating-window retirement, duplicate merges); T301-T302 (Editor/Archaeology merge,
  Cartography placement) + Gate B; T401 (Editor re-split); T501-T502 (Quick-tab single source of
  truth for Fog Defaults, the operator's own named pain point); T609/T610 (operator gate retest).
- Superseded by / overlaps: 227 (re-audit that inherited the same unresolved dedup problem after
  223's Phase 6 shipped a narrower fix); 231 (Editor/Archaeology reorg, doing what 223 T301/T401
  described); 225 (Fog End default-toolbar-order item overlaps T501/T502's Quick-tab ask).
- Disposition: FOLD
- Proposed epic theme: UI & Approachability
- Confidence: high

---

### 224 Agent Governance — Scope Fidelity, Receipts & Spec Hygiene
- Stated status: no status field | Tasks: 4/9 checked (Phase 1 T101-T104 checked, Gate 1
  unchecked; Phase 2 T201/T202 unchecked-but-partial; Phase 3 T301 unchecked)
- Scope: binding process rules for every agent/harness — scope freeze (FR-1), receipt-gated
  checkboxes (FR-2), write containment (FR-3), a `speckit-cleanup` skill + AGENTS.md timestamp
  ledger (FR-4), spec-sync (FR-5), context discipline/archival (FR-6).
- Process-rules-vs-tasks split (per batch instruction): **FR-1 through FR-6 are now literally
  codified in repo-root `AGENTS.md` §9** (confirmed present verbatim: "9.1 Scope freeze" through
  "9.5 Context discipline & monthly cleanup," including the cleanup ledger). T101-T104 (write the
  rules, author the skill, apply the 212 correction, register in STATUS.md) are genuinely done and
  checked. What remains as **open task**, not rule-authoring, is the *execution* of the mechanism:
  Gate 1 (operator approves the rules as binding — still unchecked, though the rules are already
  being enforced in practice by this very audit), and the recurring monthly cleanup (T301).
- Verified implemented: AGENTS.md §9 exists exactly as T101 describes; `speckit-cleanup` skill is
  listed as an available skill in this session's tool list, confirming T102's install.
- Partial: T201/T202 (first cleanup run) are marked unchecked with a "PARTIAL 2026-09-10" note in
  tasks.md, but **AGENTS.md's own ledger says "Last cleanup: 2026-09-11 (224-T201 receipt/symbol
  audit COMPLETE — all 5 active specs... audited)"** — i.e. the cleanup ledger in AGENTS.md is
  further ahead than 224's own tasks.md, which was never updated after the 2026-09-11 run. This is
  a stale-doc gap, not a fabricated claim (the AGENTS.md ledger entry is itself detailed and
  falsifiable), but tasks.md should be resynced.
- Not implemented: T301 recurring monthly run — next due 2026-10-01 per the ledger; not yet due
  relative to today (2026-09-23).
- Checkbox accuracy: T101-T104 accurate; T201/T202 understate actual progress (AGENTS.md ledger is
  ahead of tasks.md — sync gap, not overclaim).
- Operator gates owed: Gate 1 (explicit operator sign-off on the rules as binding) — unchecked,
  though the rules are de facto already governing this and other audit sessions.
- Open residue (spec-stated only): Gate 1 sign-off; T201/T202 tasks.md resync to match the
  2026-09-11 AGENTS.md ledger entry; T301 recurring monthly cadence (mechanical, not a design gap).
- Superseded by / overlaps: none — this is the standing meta-process spec and stays live by
  design; it is infrastructure, not a UI feature, and should not fold into a UI epic.
- Disposition: KEEP-ACTIVE
- Proposed epic theme: Infrastructure & Governance (already its own epic in STATUS.md, item 8)
- Confidence: high

---

### 225 Overhead Orthographic World View with Grid Overlays
- Stated status: "Draft — authored from operator directive, not yet implemented" | Tasks: 2/5
  checked (Phase 0 prep only)
- Scope: an Overhead toolbar toggle (first position before Tiles/Chunks/Cells) that switches to a
  top-down orthographic view reusing either the world renderer with an ortho projection or the
  minimap synthesizer's existing output, with the three grid overlays rendering on top; camera
  state must save/restore around the toggle.
- Verified implemented: T001 (bottom-toolbar grid control reorder to Tiles/Chunks/Cells) and T002
  (Cells-overlay anti-aliasing/glow/fog-fade shader work) — both plausible from the Phase 0 receipt
  notes; not independently re-verified pixel-for-pixel in this pass but the toolbar-order and
  shader-touching claims are consistent with prior specs' shader notes.
- Partial: none — Phase 1 is a clean unstarted block.
- Not implemented: T101 (projection-owner decision), T102 (Overhead toggle + camera save/restore),
  T103 (route grid overlays into the overhead view). Grep for "Overhead" across
  `src/viewer/WoWViewer/*.cs` finds only the unrelated pre-existing Chunk-Manipulator canvas field
  `_chunkManipulatorOverheadZoom` (`ViewerApp_Editor.cs:1024`) — explicitly named in the spec's own
  Context section as *not* the intended mechanism ("is not the renderer and does not use the
  map-wide imagery"). No Overhead toolbar toggle exists anywhere.
- Checkbox accuracy: accurate.
- Operator gates owed: Gate 1 (SC-1 walkthrough) — unchecked, nothing to witness yet.
- Open residue (spec-stated only): the entire Phase 1 (T101-T103, FR-1 through FR-4) — decide
  ortho-projection owner, build the toggle with camera state preservation, wire the three existing
  grid overlays into it.
- Superseded by / overlaps: none found; still the sole owner of this request.
- Disposition: FOLD
- Proposed epic theme: UI & Approachability
- Confidence: high

---

### 227 UI Re-Audit — Sidebar Standardization & Deduplication
- Stated status: "Implementing — T001 source-audit baseline is receipted; full inventory
  reconciliation, operator screenshots, and all source consolidation remain open" | Tasks: 2/17
  checked
- Scope: re-run the Spec 223 surface inventory against the current build with screenshots (US1);
  standardize every sidebar on `SharedUiWidgets` with dropdown/collapsible sub-navigation reachable
  in ≤3 interactions, explicitly including fixing the Archaeology styling mismatch (US2);
  deduplicate weak-signal amplifiers, minimaps (teleport-in-one-only), and 3-5x Inspector
  repetition down to one authoritative surface each (US3); approachability pass (US4).
- Verified implemented: T001/T002 — `surface-inventory-v2.md` (128 lines) and
  `evidence/t001-source-audit.md` / `evidence/t002-spec223-reconciliation.md` exist, giving a real
  dated inventory baseline and a reconciliation against 223's dispositions.
- Partial: none beyond the baseline — no per-surface `SharedUiWidgets` migration, no
  weak-signal/minimap/Inspector dedup, no screenshot evidence set (`evidence/screenshots/` was not
  populated at the depth T003 requires) found.
- Not implemented: T003-T004 (screenshot matrix + inventory-gate receipt); T005-T008 (per-surface
  `SharedUiWidgets` migration loop, US2's whole point); T009-T014 (weak-signal-amplifier,
  minimap-teleport, and Inspector-repetition dedup, US3); T015-T017 (approachability action order,
  USERGUIDE update, final screenshot comparison, US4). AGENTS.md §11 itself still names "a styling
  mismatch (e.g., Archaeology)" as an open defect, confirming this residue is current, not stale.
- Checkbox accuracy: accurate (matches the 224 2026-09-11 governance audit's "227 clean" finding).
- Operator gates owed: T004's inventory-gate receipt, T008/T012/T014/T017's operator
  screenshot/reachability confirmations — all unchecked, nothing witnessed yet beyond the baseline.
- Open residue (spec-stated only): the entire US2 (SharedUiWidgets standardization, ≤3-interaction
  navigation, Archaeology styling fix), US3 (weak-signal/minimap/Inspector dedup, named explicitly
  by the operator), and US4 (approachability) — T003-T017.
- Superseded by / overlaps: 223 (US1 inventory is 227's re-run of 223's own inventory); 231 (took
  over the Editor/Archaeology reorganization piece of US2 in practice — see 231 report); 228 (T004
  of this spec is a hard blocking gate for 228's WorldScene extraction, so 227 stalling stalls 228
  too).
- Disposition: FOLD
- Proposed epic theme: UI & Approachability
- Confidence: high

---

### 228 Source Decomposition — God-Class Split
- Stated status: no status field, `evidence/` dir exists but empty of the referenced receipts |
  Tasks: 0/15 checked
- Scope: establish a source-size baseline (Phase 1); gate on Spec 227 T004's UI-authority decision
  before moving any selection code (Phase 2); build a pure, testable `WorldSceneSelectionService`
  in Core.Runtime with no `WorldScene`/GL/ImGui reference (Phase 3, US3); extract exactly one
  hover/click selection algorithm out of `WorldScene` behind a narrow adapter seam (Phase 4, US1);
  pick exactly one `ViewerApp` feature as the next candidate, but only as a follow-up amendment,
  not implement it yet (Phase 5, US2); record the handoff (Phase 6).
- Verified implemented: nothing. `src/core/WowViewer.Core.Runtime` has no `Selection` directory;
  `src/viewer/WoWViewer/Terrain/Scene/Selection/` does not exist; no
  `WorldSceneSelectionService`/`WorldSceneSelectionAdapter` anywhere in the repo.
- Partial: none.
- Not implemented: everything — T001-T015 all unchecked, explicitly blocked per the tasks.md
  dependency chain (`T003 is externally blocked by Spec 227 T004`), and 227 T004 is itself
  unchecked (see 227 report above), so 228 is correctly stalled, not merely neglected.
- Checkbox accuracy: accurate.
- Operator gates owed: none reachable yet — every checkpoint requires a prior receipt that doesn't
  exist.
- Open residue (spec-stated only): the entire spec — T001/T002 (baseline + selection-boundary
  survey), T003 (227 T004 gate check), T004-T007 (pure Core selection service + tests), T008-T011
  (WorldScene extraction + operator smoke), T012-T013 (next ViewerApp candidate selection, as a
  future amendment only), T014-T015 (handoff + re-audit). The **measured line-count baseline
  itself** (this report's header table) is new evidence 228's own T001 should absorb.
- Superseded by / overlaps: hard-blocked by 227 (T004); §10 of AGENTS.md (God-Class Freeze) is the
  binding rule this spec exists to satisfy — any successor epic must keep both the file-budget rule
  and this spec's "one bounded extraction at a time" discipline rather than attempting a bulk
  rewrite.
- Disposition: KEEP-ACTIVE (blocked, not cold — it is the active owner of a binding AGENTS.md
  §10 constraint and the god-class problem is getting worse, not better, per the line counts above)
- Proposed epic theme: Infrastructure & Governance (blocking gate on 227) / UI & Approachability
  (consumer once unblocked)
- Confidence: high

---

### 229 WoW-Style Shell & Contextual Keybind Profiles
- Stated status: "Draft — authored verbatim from operator directive; not planned" | Tasks: no
  tasks.md, no plan.md (spec only)
- Scope: adopt WoW-interface visual language (action-bar tool rows, panel chrome) skinned over the
  existing four-tab structure (US1); contextual keybind profiles that swap live bindings with the
  active workspace, Noggit/Noggit-Red as the reference (US2); an on-screen key-reference overlay
  (US3). Explicitly sequenced after 227 (action inventory) and 228 (owned services to bind to).
- Verified implemented: nothing. No `ActionBar` type/file anywhere in `src/viewer/WoWViewer/`; no
  keybind-profile registry beyond the single existing `ViewerKeyBindings.cs` (which has no
  `Profile` concept — grep for "Profile" in that file returns nothing).
- Partial: none.
- Not implemented: US1, US2, US3 in full — this spec never reached a plan.md or tasks.md, and its
  own stated dependencies (227's action inventory, 228's owned services) are themselves barely
  started (see above), so nothing here could have been built yet even if attempted.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: none reachable — no implementation exists to gate.
- Open residue (spec-stated only): the entire spec (US1-US3) — but it is explicitly sequenced
  behind 227 and 228, both of which are themselves early-stage; any successor epic should keep 229
  behind those, not treat it as independently startable.
- Superseded by / overlaps: none — still the sole owner of the WoW-shell/keybind-profile idea;
  depends on 227 and 228.
- Disposition: FOLD
- Proposed epic theme: UI & Approachability
- Confidence: high

---

### 231 Editor & Archaeology Workspace UI Overhaul
- Stated status: "Draft (spec + plan + tasks authored 2026-09-07; implementation deferred to a
  fresh session per operator instruction)" | Tasks: 23/33 checked
- Scope: explicitly supersedes 227's unimplemented "sane Editor tabs" portion; converge Editor and
  Archaeology on the Data-I/O-page/Quick-panel pattern; remove named-duplicate surfaces (5x PM4
  export buttons, 2x correlation panel, 2x Clipboard+Save, 4x doodad-set combo, Editor content
  hosted inside Archaeology); remove non-working/never-wired surfaces (weak-signal amplifier, weird
  terrain tools, stale selection tools) as a ViewerApp size-reduction pass.
- Verified implemented: 23 checked tasks including a real 4-page Editor IA and a landed doodad-set
  combo consolidation (T032, confirmed live: `DrawHoveredWmoDoodadSetCombo` +
  `_hoveredWmoDoodadSetComboWmo` fields in `ViewerApp_Sidebars.cs`). Per 224's 2026-09-11
  governance audit, 231 was found clean (0 unreceipted checks; one item, T074, flagged only as
  "receipt not in evidence/" — a documentation gap, not a false claim).
- Partial: T041 (Converters page — map/WMO/M2-MDX converters, round-trip validation) is unchecked
  even though the Converters sub-tab itself exists and is reachable (see 073 report) — the page
  exists but this spec's fuller validation/round-trip scope for it isn't done.
- Not implemented: T061-T064 (remove weak-signal amplifier, weird terrain tools, stale selection
  tools; converge remaining pages on the Data I/O/Quick pattern) — confirmed live:
  `_terrainWeakSignalRestore*` fields are still present in both `ViewerApp.cs` and
  `ViewerApp_Sidebars.cs`, i.e. the removal pass has not happened. T080 records a **known
  regression**: the toolbar's hovered-WMO doodad-set combo (landed under T032) reportedly
  disappeared in a later change — an explicitly tracked, unresolved regression, not a residue gap.
  T050-T052 (final operator navigation smoke, inventory-v3 rows, final build/test receipts) remain
  open — the spec has not reached "Implemented with user gates."
- Checkbox accuracy: accurate (matches the recent governance audit finding).
- Operator gates owed: T050 (US1/US2/US3 navigation smoke), and by extension the whole "final
  acceptance" gate — explicitly still open.
- Open residue (spec-stated only): T041 (Converters page full scope); T061-T064 (removal pass +
  page convergence on the Data I/O/Quick pattern); T080 (doodad-combo regression — fix, don't
  re-plan); T050-T052 (operator smoke, inventory v3, final receipts).
- Superseded by / overlaps: 227 (explicitly named as the spec whose "sane Editor tabs" item this
  one supersedes); 073b (Converters integration, same feature, different attribution — see 073
  report).
- Disposition: FOLD
- Proposed epic theme: UI & Approachability
- Confidence: high

---

### 233 Renderer Marketing Capture Automation
- Stated status: "Draft" | Tasks: 14/29 checked
- Scope: named, versioned feature-tour recipes that warm+run a camera path and reveal scripted UI
  beats with chrome otherwise hidden (US1); a durable per-attempt receipt recording provenance,
  beat outcomes, and frame-time/FPS data so a clean-looking video can't hide a bad benchmark (US2);
  a versioned, path-safe external authoring handoff for a future MCP/ComfyUI pipeline, opt-in and
  failure-explicit (US3); real, receipt-backed video/stills in the README only after operator
  review (final tasks).
- Verified implemented (US1, substantial): real recipe/beat machinery exists in
  `src/core/WowViewer.Core.Runtime/Marketing/`: `FeatureTourRecipe.cs` (163 lines),
  `MarketingTourAttempt.cs` (124 lines, beat lifecycle + `MarketingTourAttemptStartResult`),
  `BuiltinFeatureTourRecipes.cs`, `MarketingCaptureOutputPolicy.cs` — matching STATUS.md's own
  characterization ("path warmup + direct framebuffer capture now has a Feature Tour action and
  timed clean-scene callouts").
- Partial: `AuthoringHandoff.cs` (100 lines) exists but is **types-only** —
  `MarketingCaptureTerminalOutcome`, `AuthoringHandoffProvenance`, `AuthoringHandoffRequest`,
  `AuthoringHandoff` record with `SchemaV1` — no builder/derivation logic from a real receipt yet,
  matching T022 ("extend... to derive a descriptor") being unchecked rather than a false claim of
  completeness.
- Not implemented: **US2 in full** — no `TourAttemptReceipt.cs` exists anywhere in
  `src/core/WowViewer.Core.Runtime/Marketing/` (T016-T020 unchecked: receipt model, serialization
  tests, wiring into `ViewerApp_CaptureAutomation.cs`'s start/stop seam, operator benchmark-witness
  receipt) — this is the FR-006/FR-007 "a video alone MUST NOT imply a successful benchmark"
  guarantee, and it does not exist yet; **US3 transport** — T023 (select the real MCP schema before
  any transport code), T024 (opt-in adapter), T025 (real Comfy workflow witness) all unchecked, and
  correctly gated ("Direct HTTP to ComfyUI is out of scope" until T023 is done); T015 (operator
  tour witness); T026-T029 (README real-asset embed, STATUS.md sync, scoped commit, Patreon
  spin-off) all open.
- Checkbox accuracy: accurate.
- Operator gates owed: T015 (real video/UI-timing witness), T020 (real completed/degraded/failed
  receipt inspection with hitch/FPS evidence), T025 (real Comfy workflow witness) — all explicitly
  unchecked; this is STATUS.md's #0 current top-priority item ("receipt/transport still open").
- Open residue (spec-stated only): T015 (operator witness for US1); T016-T020 (the entire receipt
  model, US2 — currently the biggest concrete gap, since it's the only thing that makes a capture
  trustworthy as a benchmark rather than a pretty video); T021-T025 (authoring-handoff
  builder + MCP schema selection + adapter + real witness, US3); T026-T029 (README, STATUS sync,
  commit, Patreon carve-out).
- Superseded by / overlaps: none — sole active owner; STATUS.md's #0 slot.
- Disposition: KEEP-ACTIVE
- Proposed epic theme: Renderer Performance & Correctness / stand-alone (capture automation isn't
  really a "UI shell" spec despite the batch grouping — it is closer to a renderer-benchmarking
  and release-asset pipeline; recommend NOT folding it into the UI & Approachability epic)
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|---|---|---|---|
| 009 | ARCHIVE-COLD | 0 | (historical reference doc) |
| 069 | ARCHIVE-SUPERSEDED | 0 | UI & Approachability |
| 072 | ARCHIVE-COMPLETE | 0 (1 checkbox inaccuracy noted, T005) | UI & Approachability |
| 073 | ARCHIVE-SUPERSEDED | 0 | UI & Approachability |
| 110 | FOLD | 3 items (US3 Tools audit, US4 conversion-capability docs; US2 native-M2 routes to 235) | UI & Approachability (+ 235 for US2) |
| 212 | FOLD (US6 only) / ARCHIVE-COLD (rest) | 1 (US6 selection-silhouette outlines) | UI & Approachability |
| 223 | FOLD | 6 items (Inspector unification, floating-window retirement, Editor/Archaeology merge+split, Quick/Fog-Defaults single source, operator gate T609/T610) | UI & Approachability |
| 224 | KEEP-ACTIVE | 2 items (Gate 1 sign-off, tasks.md resync + monthly cadence) | Infrastructure & Governance |
| 225 | FOLD | 1 item (entire Phase 1: Overhead toggle) | UI & Approachability |
| 227 | FOLD | 3 items (SharedUiWidgets standardization, weak-signal/minimap/Inspector dedup, approachability pass) | UI & Approachability |
| 228 | KEEP-ACTIVE | 1 item (entire spec, blocked on 227 T004) | Infrastructure & Governance |
| 229 | FOLD | 1 item (entire spec, blocked on 227+228) | UI & Approachability |
| 231 | FOLD | 4 items (Converters page full scope, removal pass, doodad-combo regression, final operator smoke) | UI & Approachability |
| 233 | KEEP-ACTIVE | 4 items (operator witness, receipt model, authoring-handoff/MCP transport, README/commit) | Renderer Performance & Correctness (stand-alone, not UI) |
