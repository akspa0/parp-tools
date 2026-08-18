# Implementation Plan: Skybox Rendering

**Branch**: `v0.5.3-dev` | **Date**: 2026-08-18 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/160-skybox-rendering/spec.md`

## Summary

The viewer already owns every part a working sky needs — a dome, a backdrop render path, a LIT
decoder, a `Light*` DBC resolver, and a `MOSB` parser — but five defects mean none of the authored
data reaches the screen. This plan repairs the existing paths rather than building a new sky
renderer.

Phase 0 captures a frame-cost baseline and lands the provenance scaffold that every later phase
reports through. Phase 1 fixes the per-frame overwrite that makes authored colour invisible, which
is what unblocks visual validation of everything else. Phases 2-5 then take one user story each.
Phase 6 closes on non-regression and documentation.

Phase 0 research split what the spec modelled as one resolution into two independent ones: LIT
authors sky **colours** but no **model name**, so the gradient source and the model source resolve
separately, each provenanced. This is what keeps FR-002's no-mixing rule enforceable on alpha builds
where the model has no declaration at all.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET.OpenGL (render backend), DBCD + WoWDBDefs (`Light*` chain),
existing in-repo LIT, WMO, M2/MDX, and BLP readers. No new dependencies.

**Storage**: N/A — reads client data through the configured runtime data source; no new on-disk
artifacts.

**Testing**: xUnit under `tests/` for source resolution, provenance, band mapping, classification,
and interior selection. **Rendering correctness is validated by automated capture and pixel diff** —
the viewer's startup capture automation (`--capture-shot`/`--capture-no-ui`/`--exit-after-capture`)
and the headless `profile-render` production-scene profiler. Two automation gaps are closed in
Phase 0 first: a `--time-of-day` pin, and camera motion in `profile-render` (whose camera is
currently resolved once and reused for every sampled frame).

**Target Platform**: Windows desktop viewer (`src/viewer/WoWViewer`).

**Project Type**: Desktop application over shared core libraries.

**Performance Goals**: No new hitches attributable to sky work; steady-state `Sky` +
`SkyboxBackdrop` stage cost within the budget set in Phase 0 against a measured baseline.

**Constraints**: Sky asset resolution and loading must not block the render thread (FR-024). Sky
work must be fully skipped when sky rendering is disabled (FR-023). Terrain and object fog behaviour
must not change except where fog colour is already shared with the sky.

**Scale/Scope**: Five bounded defect repairs across two core libraries and the viewer. No new file
formats, no new readers, no new profiler.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Repo Independence | **PASS** | All work inside `wow-viewer/`. No external project or path references. |
| II. Library-First | **PASS with a required constraint** | Sky source resolution, provenance, band mapping, and classification are **library** concerns and land in `src/core/WowViewer.Core.Runtime/` and `src/core/WowViewer.Core.IO/`. `WorldScene.cs` is already ~15k lines; it gets wiring and draw-order only, no resolution logic. Recorded as a gate on every phase, not an aspiration. |
| III. Real-Data Validation | **PASS** | Every user story's acceptance is real-client proof against a configured build, produced by automated capture + pixel diff. Build/test success is still not accepted as rendering proof — a rendered image is. |
| IV. Per-Signal Reporting | **N/A** | No model or training work in this spec. |
| V. Streaming-First Dataset Pipeline | **N/A** | No dataset work in this spec. |
| VI. No Client Path Assumptions | **PASS** | Model discovery probes relative paths *inside* the configured data source. No machine-local root enters source or portable docs. Build identity and root are reported with validation evidence. |
| Format Reader Ownership | **PASS** | No parser is rewritten. `MOSB` already parses; US4 preserves a field the existing reader discards rather than adding a second read path (research R5). |
| Terrain Alpha Risk Area | **PASS with a guard** | MCAL, edge-fix, `_tex0.adt`, alpha packing, and terrain shader blending are untouched. The one adjacency is that sky already shares fog colour, so Phase 6 explicitly re-checks terrain fog against both Alpha-era and LK 3.3.5 terrain. |
| `AlphaWdtWriter` Frozen | **PASS** | Not touched. |
| One Phase at a Time | **PASS** | Each phase below ends in validation; a phase is not done until validated. |
| Bite-Sized Plans | **PASS** | Every phase is ≤ 10 steps, one concern per step, each independently validatable. |
| Spec Docs Are Source of Truth | **PASS** | Phase 6 updates the affected architecture docs in the same commit as the code. |

**No violations. Complexity Tracking section omitted.**

### Post-design re-check (after Phase 1 artifacts)

Re-evaluated after `data-model.md` and `contracts/` were written. The design adds five types
(`SkyProvenance`, `SkyBand`, `SkyGradientSource`, `SkyModelReference`, `SkySourceSelection`), all
under `src/core/WowViewer.Core.Runtime/World/Sky/`, with the viewer consuming rather than
constructing them — this **strengthens** Library-First compliance rather than straining it, since it
moves decisions that currently live inside `WorldScene.cs` out to a library.

Two design choices were checked specifically against the constitution and both hold:

- **No new reader** — US4 preserves a field `WmoSummaryReader` already parses and discards, instead
  of adding a second read path (research R5). Format Reader Ownership: PASS.
- **No new profiler** — the budget uses `WorldRenderStage.Sky` and `SkyboxBackdrop`, already
  instrumented (research R6). PASS.

**Still no violations.**

## Project Structure

### Documentation (this feature)

```text
specs/160-skybox-rendering/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output — R1-R7
├── data-model.md        # Phase 1 output — entities and state
├── quickstart.md        # Phase 1 output — automated capture + diff commands
├── contracts/
│   ├── sky-resolution.md    # Source selection + provenance contract
│   ├── sky-gradient.md      # Band ordering and dome mapping contract
│   └── frame-budget.md      # FR-022 budget, filled from measured baseline
├── checklists/
│   └── requirements.md  # Spec quality checklist (passing)
└── tasks.md             # Created by speckit-tasks, not by this command
```

### Source Code

```text
src/core/WowViewer.Core/
└── Wmo/
    └── WmoSummary.cs                     # US4: carry the MOSB name, not just a bool

src/core/WowViewer.Core.IO/
├── Wmo/
│   └── WmoSummaryReader.cs               # US4: stop discarding the parsed name
└── Lighting/
    └── LightDbcCatalog.cs                # US5: expose declared skybox names

src/core/WowViewer.Core.Runtime/World/
├── Sky/
│   ├── SkyGradientSource.cs              # NEW — resolved band set + provenance
│   ├── SkySourceResolver.cs              # NEW — single-source selection (FR-001/FR-002)
│   ├── SkyProvenance.cs                  # NEW — per-value source record (FR-003)
│   ├── SkyModelReference.cs              # NEW — resolved model + why it was chosen
│   └── SkyDomeVertexBuilder.cs           # unchanged (research R2)
├── WorldSkyboxBackdropClassifier.cs      # US5: declaration-driven, filename fallback
└── WmoCameraVisibility.cs                # US4: unchanged, finally called

src/viewer/WoWViewer/
├── Rendering/
│   ├── SkyDomeRenderer.cs                # US3: N-band gradient; US1: stop clobbering
│   └── ModelRenderer.cs                  # US2: clock-driven skybox animation
├── Terrain/
│   ├── WorldScene.cs                     # wiring + draw order only, no resolution logic
│   └── LightService.cs                   # expose resolved model + provenance
└── ViewerApp_Sidebars.cs                 # provenance readout (FR-003 inspectable)

tests/
└── (unit coverage for resolution, provenance, bands, classification, interior selection)
```

**Structure Decision**: Existing layout, no new projects. The split follows Constitution II — every
decision (which source wins, which bands exist, which model is active, what classified it) is a
library concern under `src/core/`; the viewer keeps only wiring, draw order, and display. This is
the gate that keeps the fix from accreting further into `WorldScene.cs`.

---

## Phases

Each phase ends in validation. Per Constitution "One Phase at a Time", a phase is **not done when
coded — it is done when validated**. Rendering proof is produced by automated capture; commands are
in [quickstart.md](./quickstart.md).

### Phase 0 — Automation gaps, baseline, and provenance scaffold

**Why first**: FR-022 requires a budget measured against a pre-change baseline. Once any sky code
changes, that baseline is unrecoverable. This follows the project's established "prove the detector
before you use it" pattern — and here the detector itself needs two fixes before it can be trusted.

0a. Add a `--time-of-day` startup pin so day-cycle and time-scrub tests are reproducible.
0b. Add camera motion to `profile-render`, whose camera is currently resolved **once** and reused for
    every sampled frame, and emit per-stage p50/p99/max plus `CameraMovedDuringWindow`.

1. Capture the pre-change frame-cost baseline for `WorldRenderStage.Sky` and
   `WorldRenderStage.SkyboxBackdrop` (p50/p99/max), with camera motion so the window is
   movement-valid, on at least one dense and one sparse map.
2. Record the baseline, build identity, configured root, and map set in `contracts/frame-budget.md`,
   and set the FR-022 budget as a delta plus an absolute ceiling.
3. Add `SkyProvenance` — the source, record identity, and authored-vs-fallback flag for one resolved
   value.
4. Add `SkyGradientSource` and `SkyModelReference` as the two independently-resolved results
   (research R1).
5. Add `SkySourceResolver` with single-source selection and no cross-source field blending
   (FR-001/FR-002), preferring map-scoped over global and recording the choice.
6. Unit-test the resolver: LIT-only, DBC-only, both-resolve, neither-resolves; assert no result ever
   draws fields from two sources.
7. Surface provenance in the viewer's diagnostics readout (FR-003).

**Validation**: automation gaps closed; baseline recorded with build identity; resolver unit tests pass including the
no-mixing assertion; provenance visible in the UI for a real map. Nothing rendered has changed yet —
that is expected and is the point.

---

### Phase 1 — US1: authored sky colours reach the screen (P1)

**Why here**: the overwrite defect makes US2-US5 invisible. Nothing else can be visually validated
before this lands.

1. Change `SkyDomeRenderer.UpdateFromLighting` so it no longer unconditionally overwrites resolved
   colours — it supplies values only where no authored value was resolved.
2. Wire the resolved `SkyGradientSource` through `WorldScene`'s lighting step so authored colours
   survive to the draw, with the hardcoded curve as an explicit, reported fallback (FR-004/FR-005).
3. Ensure sky colours follow the world time-of-day clock through the selected source's timed samples
   (FR-006).
4. Keep the existing manual LIT override working as an override, recorded in provenance (research
   R7).
5. Unit-test that a resolved authored value is never replaced by a fallback value.

**Validation**: change an authored sky colour in client data, reload, and confirm the rendered sky
changes (SC-001 — currently 0% of such changes are visible). Scrub time of day and confirm the sky
follows authored samples. Confirm a no-profile map renders the fallback and *says* it is the
fallback. 

---

### Phase 2 — US2: skybox model visible across the whole day (P1)

Depends on Phase 0 (model reference + provenance). Independent of Phase 1 (research R1).

1. Remove the `NightVisibility` gate so a resolved model renders across the full cycle (FR-010).
2. Drive the active skybox model's animation from the world clock by setting `CurrentFrame`, instead
   of accumulating wall-clock deltas (FR-011, research R3).
3. Handle sequence wrap at the midnight boundary at the call site — the two animators clamp
   differently (research R3).
4. Confirm draw order: gradient first, model composited over it, both behind all world geometry, no
   depth write (FR-012).
5. Make selection deterministic when multiple candidates tie on distance (FR-013).
6. Make a missing, unresolvable, or still-loading model degrade to gradient-only, reported once and
   not retried per frame (FR-014, FR-024).
7. Unit-test deterministic selection and once-only failure reporting.

**Validation**: model visible at midday; continuous across a full day sweep with no threshold
pop (SC-002); appearance advances when time is scrubbed; world geometry never occluded; a deliberately
broken model path still renders a sky (SC-007). 

---

### Phase 3 — US3: five-band sky gradient (P2)

Depends on Phase 1 — until authored colour survives, band changes are invisible.

1. Extend the dome fragment shader from a two-colour `mix` to an ordered N-band interpolation over
   `vHeight` (FR-007, research R2).
2. Upload the resolved ordered band set from `SkyGradientSource`; do not change
   `SkyDomeVertexBuilder` (research R2).
3. Define band-to-height mapping against the existing hemisphere convention, including the current
   below-horizon fog blend under `vHeight < 0.15`; record it in `contracts/sky-gradient.md`
   (research R2).
4. Handle sources authoring fewer bands than the full set: use what exists, report the shortfall, do
   not zero-fill (FR-008).
5. Verify transitions are continuous with no banding seam (FR-009).
6. Unit-test band ordering and short-band-set handling.

**Validation**: change one mid-sky band in isolation and confirm the change appears in that band's
region while zenith and horizon hold (SC-003); inspect band boundaries for seams. 

---

### Phase 4 — US4: WMO interior skyboxes (P3)

Depends on Phase 2 — needs a working model path underneath.

1. Preserve the parsed `MOSB` name on `WmoSummary` instead of reducing it to `HasSkybox`
   (FR-015, research R5).
2. Keep `WmoSummaryReader` a single parse — do not add a second read of the WMO root (research R5).
3. Call the existing `WmoCameraVisibility.IsInsideRootOrGroup` where instance transforms and bounds
   are already resident; the caller owns the world→local transform and padding (research R4).
4. Make the interior skybox the active sky while inside and restore the outdoor sky on exit
   (FR-016).
5. Make selection deterministic and stable under nested and overlapping WMOs (FR-017).
6. Fall back to the outdoor sky on an unresolvable declared name, reported (FR-018).
7. Add hysteresis or equivalent so repeated boundary crossings do not strobe the sky (SC-005).
8. Unit-test nested-WMO determinism and unresolvable-name fallback.

**Validation**: enter a WMO that declares a skybox and confirm the sky changes; leave and confirm it
reverts; cross the boundary repeatedly and confirm no flicker; enter a WMO with an unresolvable
declared name and confirm the outdoor sky persists with the reference reported. 

---

### Phase 5 — US5: data-driven classification (P3)

Depends on Phase 2. Lowest visible payoff — it prevents silent wrong behaviour rather than fixing a
visible defect.

1. Build the declared-skybox name set from available sources — `LightSkybox` names and `MOSB` names
   (FR-019).
2. Classify placements against declared names rather than filename keywords.
3. Keep the filename heuristic as an explicitly-reported fallback for builds with no declaration
   source — which per research R1 includes LIT-era alpha builds for outdoor models (FR-020).
4. Report which declaration classified each skybox (FR-021).
5. Unit-test both directions: a declared skybox whose filename lacks the keywords is classified; a
   non-sky asset whose filename contains a keyword is not.

**Validation**: confirm both directions against real client data, and confirm the fallback path
reports itself on a build with no declaration source. 

---

### Phase 6 — Non-regression and documentation

1. Re-capture `Sky` and `SkyboxBackdrop` distributions on the Phase 0 maps with a moving camera and
   compare against the recorded budget (FR-022, SC-008). 
2. Confirm hitch attribution shows no new sky-attributed hitches.
3. Confirm that with sky rendering disabled, sky evaluation and sky draw cost measure zero
   (FR-023, SC-009).
4. Walk the full failure matrix — no profile, missing asset, still-loading asset, unresolvable WMO
   reference — and confirm every case still renders a sky (SC-007).
5. Re-check terrain fog against both Alpha-era and LK 3.3.5 terrain, since sky already shares fog
   colour (Constitution terrain risk-area guard).
6. Update the affected architecture docs in the same commit as the code (Constitution: Spec Docs Are
   Source of Truth).
7. Update `specs/STATUS.md` and the memory bank.

**Validation**: measured frame cost within budget; zero black-sky frames across the failure matrix;
terrain fog unchanged on both eras.

---

## Risks

| Risk | Mitigation |
|---|---|
| Sky work regresses the frame-time specs (151/152/153) currently in flight | Baseline captured before any change; budget is a gate in Phase 6, not an afterthought; both stages already instrumented separately |
| Source-agnostic resolution silently does the wrong thing on an unfamiliar build | Provenance is mandatory on every value (FR-003) — divergence surfaces as a reportable fact rather than a wrong colour |
| Resolution logic accretes into `WorldScene.cs` | Constitution II gate is restated per phase; the viewer gets wiring only |
| Below-horizon bands have no geometry to land on | Called out in research R2; mapping is fixed in `contracts/sky-gradient.md` before shader work |
| Phase 1 makes previously-hidden data bugs suddenly visible | Expected, not a regression — authored data has never reached the screen; treat surprises as newly-visible truth and record them |

## Dependencies

- **Blocks nothing.** No other active spec depends on this one.
- **Shares surface with** specs 151/152/153 (renderer frame time) — same per-frame path, different
  stages. Coordination is the Phase 6 budget gate.
- **Consumes** existing LIT, `Light*` DBC, WMO, and M2/MDX readers. Adds none.

## Next Step

Run **speckit-tasks** to generate `tasks.md` from these phases.
