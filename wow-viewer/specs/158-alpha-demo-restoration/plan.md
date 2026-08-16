# Implementation Plan: Alpha Demo Restoration — WTF Commands, Camera Follow, and Torchlight

**Branch**: `v0.5.3-dev` (this repo keeps all specs on one branch; no per-feature branch is created)
**Date**: 2026-08-16
**Spec**: [spec.md](./spec.md)

## Summary

Seven independently-shippable capabilities, in priority order: read a real WTF file's statements (US1 —
**already delivered** by Spec 159's `WowViewer.Core.IO.Wtf` library, adopted here rather than rebuilt);
execute worldport/teleport commands found in one (US2); browse WTF files as a clickable index of
explorable waypoints (US3 — the payoff of the WTF lane, turning a corpus of location commands into a
usable frame of reference); a real, measured Alt+P keybind for the existing performance overlay (US4 —
de-risked by Spec 159's direct read of `WTF\DefaultBindings.wtf`); attach the camera to a character
model's bone (US5); equip a torch that casts a real dynamic point light (US6, split into two sub-phases —
M2 attachment-point parsing and dynamic lighting — because they are very different sizes and risks); and
replaying a captured investor demo (US7, explicitly blocked on Spec 159 finding a real source file).

## Technical Context

**Language/Version**: C# / .NET 10 (matches the rest of `wow-viewer`)

**Primary Dependencies**: `WowViewer.Core.IO` (WTF parsing — already built, Spec 159), `WowViewer.Core.Runtime` (M2 bone pose evaluation), `WowViewer.Core.Renderer` (terrain/model shaders), Silk.NET.OpenGL (rendering), Silk.NET.Input (keybinding)

**Storage**: N/A — this feature reads client data and drives runtime viewer state; it does not persist new data of its own (no new on-disk format)

**Testing**: xUnit (`WowViewer.Core.Tests`), plus manual in-viewer verification for the rendering-facing stories (US4, US5) per this project's "test the actual feature in a browser/app before reporting complete" standard — headless unit tests cannot verify a camera visually tracks a bone or a light visibly brightens terrain

**Target Platform**: Desktop viewer (`WoWViewer` / `ParpToolsWoWViewer`), Windows-primary per this session's environment

**Project Type**: Desktop 3D viewer application with a thin CLI inspection tool suite alongside it

**Performance Goals**: No new hard budget stated by the spec; SC and edge cases require that dynamic lighting and camera-follow introduce no perceptible frame-time regression, consistent with this project's existing frame-pacing discipline (Specs 152/153)

**Constraints**: No client data enters the repository (Principle VI / Data Policy); do not route new code through `WowViewer.Core.Anim` (`PathNormalizer` throws on `H:\CLIENTS` paths — tracked, deliberately unbundled follow-up, not this spec's problem to fix); do not rewrite the M2Era100 reader's existing parsing, only extend it (Format Reader/Writer Ownership)

**Scale/Scope**: Six user stories, one new library area (WTF command execution — distinct from Spec 159's WTF *reading*), one new runtime capability (camera follow target), one extension to an existing-but-incomplete reader (M2Era100 attachments), one new rendering subsystem (dynamic point lights)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design — see below.*

| Principle | Check | Status |
|---|---|---|
| I. Repo Independence | All new code lives under `wow-viewer/src/`; no path outside it referenced | PASS |
| II. Library-First | WTF command execution → `WowViewer.Core.IO.Wtf` (extends Spec 159's library) or `WowViewer.Core.Runtime` for the dispatch-to-camera glue; attachment parsing → `WowViewer.Core.IO.M2Era100` (existing reader's namespace); dynamic lighting → `WowViewer.Core.Renderer`. `ViewerApp*` stays orchestration only | PASS |
| III. Real-Data Validation | US1/US3 already validated against real 0.5.3.3368/2.0.0.5610 data via Spec 159. US2/US4/US5 require validation against real staged clients and real loaded models before being called done — planned explicitly per phase below | PASS (pending per-phase execution) |
| Format Reader/Writer Ownership | M2Era100 attachment parsing **fills in** `M2Era100ModelReader.cs`'s already-documented, not-yet-implemented offsets (`M2Era100Constants.cs:143-147`) — this is completing an existing reader, not duplicating one. WTF line classification reuses Spec 159's `WtfLineClassifier` unmodified for statement recognition; this plan adds *execution* of already-classified lines, not a second parser | PASS |
| Terrain Alpha Risk Area | Dynamic lighting (US5) touches shading, but only adds a point-light contribution — it must not alter MCAL decode, alpha packing, or existing shader blending paths. Flagged for explicit regression check against the pre-regression baseline | CONDITIONAL — checked in Phase 5b |
| One Phase at a Time | Eight phases below (0 through 6, with 2b inserted and 5 split into 5a/5b), each with its own validation gate | PASS |
| Bite-Sized Plans | Each phase capped at ≤10 steps | PASS |
| No Client Path Assumptions | All builds referenced via `--archive-root`/configured root, never hardcoded | PASS |
| Core.Anim exclusion | No phase below touches `WowViewer.Core.Anim` | PASS |

## Project Structure

### Documentation (this feature)

```text
specs/158-alpha-demo-restoration/
├── spec.md                  # already written
├── checklists/requirements.md  # already written
├── plan.md                  # this file
├── research.md               # Phase 0 output
├── data-model.md             # Phase 1 output
├── contracts/                 # Phase 1 output
│   ├── wtf-command.md
│   ├── camera-follow-target.md
│   └── dynamic-light.md
└── quickstart.md              # Phase 1 output
```

### Source Code (repository root: `wow-viewer/`)

```text
src/core/WowViewer.Core.IO/
├── Wtf/                                  # EXISTING (Spec 159) — reused, not duplicated
│   ├── WtfModel.cs                       # WtfLine, WtfLineKind (incl. PortCommandCandidate)
│   ├── WtfLineClassifier.cs
│   └── WtfSweeper.cs
├── M2Era100/
│   ├── M2Era100Constants.cs              # EXISTING — attachment offsets already defined
│   ├── M2Era100ModelReader.cs            # EXTEND — parse attachments (currently zero refs)
│   └── M2Era100Attachment.cs             # NEW — attachment record, mirrors legacy MdxAttachment.cs shape

src/core/WowViewer.Core.Runtime/
├── Wtf/
│   └── WtfCommandDispatcher.cs           # NEW — turns a classified PortCommandCandidate/Set line
│                                          #        into a worldport/teleport/cvar action
├── World/
│   └── CameraFollowTarget.cs             # NEW — model+bone reference a camera can track
└── M2/
    └── M2BonePoseAccessor.cs             # NEW (small) — public accessor for a placed M2's current
                                           #                bone world matrices (M2Renderer doesn't
                                           #                expose M2BonePoseState today)

src/core/WowViewer.Core.Renderer/
└── Lighting/
    └── DynamicPointLight.cs              # NEW — position/color/radius + shader-facing light list

src/viewer/WoWViewer/
├── ViewerApp_CameraFollow.cs             # NEW partial — attach/detach camera to a model's bone
├── ViewerApp_WtfCommands.cs              # NEW partial — run a WTF file's commands against the scene
├── ViewerApp_WtfExplorer.cs              # NEW partial — WTF tab: files, waypoints, click-to-travel
├── ViewerKeyBindings.cs                  # EXTEND — Alt+P → ui.toggle_perf_overlay
└── ViewerApp.cs                          # EXTEND — Alt+P edge-detected toggle (mirrors existing
                                           #           Ctrl/Shift modifier-check pattern at :1305-1344)

tools/inspect/WowViewer.Tool.Inspect/
└── WtfCommandSupport.cs                  # EXISTING (Spec 159) — extend with a `wtf run` command
                                           #                        for dry-running command execution
                                           #                        outside the full viewer

tests/WowViewer.Core.Tests/
├── WtfCommandDispatcherTests.cs          # NEW
├── M2Era100AttachmentReaderTests.cs      # NEW
├── CameraFollowTargetTests.cs            # NEW
└── DynamicPointLightTests.cs             # NEW
```

**Structure Decision**: Extend Spec 159's `Wtf/` library rather than create a parallel one (US1 is
already done). New runtime glue lives in `WowViewer.Core.Runtime` (camera/command orchestration is
already that library's job elsewhere — `ViewerApp_CameraPaths.cs`'s per-frame camera drive is the
existing precedent). New rendering work lives in `WowViewer.Core.Renderer` alongside the existing
`TerrainShader.cs`. `ViewerApp` partials stay thin dispatch/UI only, matching every other feature in this
codebase.

## Phases

### Phase 0 — Confirm what Spec 159 already delivers, map what doesn't exist yet

No new code. Read-only confirmation phase, matching this project's "map before building" precedent
(Spec 155 Phase 0).

1. Confirm `WowViewer.Core.IO.Wtf.WtfSweeper`/`WtfLineClassifier` (Spec 159, committed `f0dffdaa`) fully
   satisfies US1's acceptance scenarios as written — it does not need extension for *reading*, only for
   *acting on* what it reads.
2. Confirm Spec 159's real findings still hold: `WTF\DefaultBindings.wtf` packed in 0.5.3.3368 and
   2.0.0.5610's archives, `bind ALT-P TOGGLEPERFORMANCEDISPLAY` present verbatim.
3. Record which staged builds are usable for US2/US4/US5 validation (0.5.3.3368 confirmed has a loadable
   WDT/map + loadable character models; note any build-specific blockers from Spec 154's M2 era-reader
   findings if a chosen validation build's models don't read).
4. Record US6's status as still blocked per Spec 159 — no action, just carry the dependency forward
   explicitly so this plan does not silently imply it's been resolved.

**Exit gate**: a short table in research.md confirming 1-4, no code changes.

---

### Phase 1 (US1) — Adopt, do not rebuild

US1 is delivered. This phase is documentation-only: record in data-model.md/quickstart.md that WTF
*reading* is `WowViewer.Core.IO.Wtf`, unmodified, and point every later phase at it rather than having
any later phase re-implement line classification.

**Exit gate**: quickstart.md shows a real command reading a real file via the existing `wtf sweep` CLI,
citing Spec 159's own committed output as proof — no new test needed since Spec 159's 13 tests already
cover this.

---

### Phase 2 (US2) — Execute worldport and teleport commands

1. Define `WtfCommandDispatcher` (`WowViewer.Core.Runtime/Wtf/`) taking a `WtfLine` of kind
   `PortCommandCandidate` and producing a `WorldportRequest` (map + position) or `TeleportRequest`
   (position only), using the existing `HasMapIdArg` classification Spec 159 already computes — no new
   parsing.
2. Define the outcome type distinguishing success / map-load-failure / not-yet-loaded-and-no-current-map
   (per spec.md's edge cases) so a caller can report failure without repositioning.
3. Add the "load map, then position camera once loaded" glue this project does not have yet (per
   research: `MapDiscoveryService` resolves an ID to a WDT path, but nothing sequences load-then-position
   today) — a new, small coordinator, not a change to `MapDiscoveryService` itself.
4. Wire `ViewerApp_WtfCommands.cs`: given a parsed `WtfLine` list, execute each recognized port command in
   order via the dispatcher above, reporting unrecognized/failed ones per FR-007.
5. Extend `tools/inspect`'s `wtf` command with `wtf run --archive-root ... --build ... --file <path>` so
   commands can be dry-run and inspected outside the full viewer (useful for validating Phase 2 before
   ever touching `ViewerApp`).
6. Unit tests: worldport with unloaded map loads-then-positions; teleport never attempts a map load;
   worldport targeting an unloadable map leaves the camera untouched and reports failure; unrecognized
   command reported, not silently dropped.
7. Real-data validation: hand-write a small worldport/teleport command set (per spec.md's Independent
   Test — this does not require a real demo file) and run it against a real loaded 0.5.3.3368 scene,
   confirming camera position matches expectation.

**Exit gate**: SC-002 met and demonstrated against a real loaded build, not just unit tests.

---

### Phase 2b (US3) — WTF explorer tab and waypoint index

The payoff phase for the WTF lane: turn parsed location commands into a browsable, clickable index of
places worth exploring. Depends on Phase 2's dispatcher; independent of everything after it.

1. Define a `WtfWaypoint` view model: source file, source build, command kind (worldport/teleport),
   target map (when present), position — enough to satisfy FR-018's traceability requirement.
2. Extend the WTF sweep surface to accept a **folder** of collected WTF files spanning builds, not only a
   single build's archive/loose surface (FR-020) — the user may hold a collection covering builds up
   through ~8.x, and it must be usable without staging each client.
3. Build the waypoint index: for each discovered file, extract every `PortCommandCandidate` line as a
   waypoint. Files with zero waypoints stay listed (FR-019) — an empty result is a visible result.
4. `ViewerApp_WtfExplorer.cs`: a tab listing files, their waypoints, and source/build provenance.
5. Selecting a waypoint dispatches it through Phase 2's existing `WtfCommandDispatcher` — no second
   execution path.
6. Real-data validation: point the tab at a real folder/build, confirm listing and traceability; confirm
   a same-map waypoint repositions without a map load and a cross-map waypoint loads first.

**Exit gate**: SC-007 met, demonstrated live in the viewer.

---

### Phase 3 (US4) — Alt+P keybind

Already de-risked: the binding's real target action name (`TOGGLEPERFORMANCEDISPLAY`) and modifier
(`ALT-P`) are both confirmed from real 0.5.3.3368 data (Spec 159), not assumed.

1. Add `ui.toggle_perf_overlay` to `ViewerKeyBindingCatalog` (`ViewerKeyBindings.cs:27-47`).
2. In `ViewerApp.cs`, add an Alt+P edge-detected toggle for `_showPerfWindow`, copying the existing
   modifier-check + edge-detection shape already used for Ctrl/Shift at `:1305-1344` — first verify
   `Key.AltLeft`/`Key.AltRight` exist in the vendored Silk.NET.Input `Key` enum (flagged in spec.md as
   not yet directly confirmed).
3. Unit/manual test: Alt+P toggles the overlay; plain "P" (already bound to `ui.focus_pm4`) is unaffected;
   toggling via the existing toolbar buttons still works identically.

**Exit gate**: SC-003 met, verified manually in the running viewer (this is a UI/keybind change — per
this project's standard, must be exercised in the actual app, not unit-tested alone).

---

### Phase 4 (US5) — Camera attached to a character model's bone

1. Define `CameraFollowTarget` (`WowViewer.Core.Runtime/World/`): a reference to a placed model instance
   + bone index/KeyBoneId, with a method to resolve the bone's current world transform.
2. Add `M2BonePoseAccessor` — the small missing public accessor on the modern M2 pipeline
   (`M2Renderer.cs` does not currently expose `M2BonePoseState.Matrices` externally; the legacy
   `MdxAnimator.BoneMatrices` path is already public and needs no change).
3. Extend `Camera` (`Rendering/Camera.cs`) with an optional follow-target mode: when set, `Position`/
   `Yaw`/`Pitch`/`Roll` are derived from the target's current bone transform each frame instead of from
   WASD input, mirroring the exact per-frame external-drive pattern already used in
   `ViewerApp_CameraPaths.cs:679-688`.
4. `ViewerApp_CameraFollow.cs`: UI/command surface to attach/detach camera-follow for a selected model;
   detaching returns to free-fly from the last followed position (FR-011).
5. Unit tests: follow-target resolves a known bone's transform correctly for both the legacy Mdx and
   modern M2 pipelines; detaching preserves last position; a removed/unloaded followed model falls back
   to free-fly rather than crashing (edge case in spec.md).
6. Real-data validation: attach to an animating character model in a real loaded scene, confirm visual
   tracking through at least one full animation cycle (SC-004) — manual, in-viewer, per this project's
   UI-verification standard.

**Exit gate**: SC-004 met, demonstrated live, not only asserted from unit tests.

---

### Phase 5a (US6, part 1) — M2Era100 attachment-point parsing

The format is already fully documented; this phase completes an existing reader, it does not design a
new format.

1. Define `M2Era100Attachment` (`WowViewer.Core.IO/M2Era100/`) — fields mirroring what the real
   decompiled 1.0.0 client algorithm needs (`test_data/native-research/1.0.0-decomp/feat_has_attachment.c`,
   `feat_attachment_worldtransform.c`): bone index, pivot point, attachment ID/KeyBoneId-equivalent.
2. Extend `M2Era100ModelReader.cs` to parse the attachment table using the already-defined offsets
   (`AttachmentCountOffset=0x104`, `AttachmentOffsetOffset=0x108`, `AttachmentLookupCountOffset=0x10C`,
   `AttachmentLookupOffsetOffset=0x110`, `AttachmentStride=0x30`) — read-only parsing, no behavior change
   to anything the reader already does.
3. Add a `GetAttachmentWorldTransform`-equivalent method, translated directly from the real decompiled
   reference algorithm (cited above), not reinvented — this is a case where the real client's own logic
   is already sitting in this repo as ground truth.
4. Unit tests against a real staged M2Era100 model with known attachment points (a torch-carrying NPC or
   any hand-slot-bearing humanoid model is sufficient — exact model TBD during implementation from what's
   actually staged and loadable per Spec 154).
5. Real-data validation: resolved attachment world transform matches the model's visually-rendered hand
   position in the existing viewer (sanity check by eye, not just numeric assertion).

**Exit gate**: attachment parsing validated against real data; this phase can ship and be useful (e.g.
for other future attachment consumers) independent of Phase 5b ever landing.

---

### Phase 5b (US6, part 2) — Dynamic point-light rendering

The largest, riskiest phase in this spec — genuinely new rendering infrastructure where today there is
none (`TerrainShader.cs` has exactly one static directional light).

1. Define `DynamicPointLight` (`WowViewer.Core.Renderer/Lighting/`): position, color, radius/falloff.
2. Extend the terrain (and relevant object) shaders with a bounded point-light array + count uniform,
   additive with the existing directional+ambient term — must not alter existing MCAL/alpha-blend paths
   per the Terrain Alpha Risk Area constitution gate.
3. A simple first-pass culling/limit (e.g. nearest-N lights, or a fixed small max count) so cost scales
   with nearby lights, not total scene size (spec.md edge case) — a two-phase broad/narrow approach is
   acceptable for a first version; exact algorithm is an implementation detail decided against real
   frame-time measurement, not asserted in advance.
4. Wire a `DynamicPointLight` to a `CameraFollowTarget`'s (or any model's) resolved attachment-point
   transform from Phase 5a, updated every frame.
5. Un-equip/removal path: removing the light source's association removes its contribution the same
   frame (FR-015) — no lingering state.
6. Unit tests: light list uniform packing round-trips (position/color/radius survive); light removal
   clears its slot; combining with existing static LIT lighting doesn't zero out or clobber the ambient/
   directional term (FR-014).
7. Real-data validation, against the Terrain Alpha Risk Area baseline (`343dadfa27df08d384614737b6c5921efe6409c8`):
   confirm no regression in existing Alpha-era and LK 3.3.5 terrain rendering with this feature present
   but inactive (zero lights), before validating the lit-torch case itself.
8. Real-data validation, feature active: attach a torch to a character in a dim real scene (recreating the
   referenced 2001 screenshot's conditions), confirm visible brightening that moves with the character
   (SC-005) — manual, in-viewer.

**Exit gate**: SC-005 met, demonstrated live; Terrain Alpha Risk Area regression check passed and
recorded with its baseline commit cited, per the constitution's explicit requirement for this risk area.

---

### Phase 6 (US7) — Explicitly not started

No code. This phase exists in the plan only to state, in the same place every other phase's status
lives, that it remains blocked on Spec 159 locating a real source file (per spec.md's Scope Note and
Assumptions). Re-evaluate this plan's Phase 6 the moment Spec 159's status changes — no other phase in
this plan needs to change if that happens.

## Constitution Check — Post-Design Re-Check

Re-evaluated after Phase 1 design (research.md, data-model.md, contracts/, quickstart.md). No new
violations introduced: `PortCommandRequest`/`CameraFollowTarget`/`M2Era100Attachment`/`DynamicPointLight`
all live in the library layers the Constitution Check table above already committed to; none of them
duplicate an existing reader (M2Era100 attachment parsing fills a documented gap, per research.md §1 the
WTF line grammar itself is reused unchanged from Spec 159); the Terrain Alpha Risk Area CONDITIONAL item
remains gated by Phase 5b's explicit zero-lights regression step against the cited baseline commit, now
concretely specified in that phase's step list rather than left abstract. PASS, unchanged from the
pre-Phase-0 check.

## Complexity Tracking

*No Constitution Check violations requiring justification.* The one CONDITIONAL item (Terrain Alpha Risk
Area, Phase 5b) is handled by an explicit regression gate within that phase, not a deviation requiring a
simpler-alternative justification.
