# Implementation Plan: Legacy MDX & M2 Rendering Correctness (1.0.0 Through 3.0.1) & Fuckported-Asset Compatibility

**Branch**: `235-legacy-mdx-m2-rendering` | **Date**: 2026-09-10 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/235-legacy-mdx-m2-rendering/spec.md`

> **⚠ SUPERSEDED IN SHAPE — read [research.md](research.md) first (2026-09-10).** Phase 0 was partly
> executed after this plan was written and its findings invalidate the six-phase per-era structure
> below: the world-placement loader never reaches the 1.0.0 reader at all, no bounding-box fallback
> exists, and Warcraft.NET's already-wrapped generic reader covers this whole era except for one
> missing embedded-skin walk. The corrected, smaller reframe is at the end of research.md and has
> **not** been applied here yet. Do not execute Phases 2-3 as written.

## Summary

M2/MDX model reading works at two known-good ends — the Alpha `MDLX` route and `MD20 0x108`
(measured at 3.3.0.10958) — and is broken, missing, or refuses everywhere between them, plus two
genuinely new gaps: assets whose chunks were rewritten by non-standard third-party tools
("fuckported") don't render even when an external reference parser can read them, and MDX
light-bearing objects (torches) are missing their visual effect. This plan reconciles and completes
two prior, substantially-overlapping, real-evidence specs (104, 154 — see spec.md's Supersession
notice) rather than re-deriving their findings, and adds the two new pillars.

**The approach stays survey-first**, inherited directly from Spec 154: the premise "the later route
works" was already falsified once by measurement (a `4.0.0.11927` beta crashes), so no reader is
touched until every staged build in range is read and recorded. Layout knowledge for the `0x100`
era already exists in the codebase (`M2Era100Constants.cs`, Ghidra-confirmed) and was never
connected to a bone parser — Phase 2 below is wiring plus verification, not format discovery.

**Critical open question carried into Phase 0, not assumed**: this feature's own spec (FR-014)
flagged that `FormatProfileRegistry` (viewer, `Terrain/` namespace) might not even be on the live
model-load path. Spec 154's plan already names the *actual* dispatch mechanism —
`M2ModelReaderDispatcher` routing to `M2Era100ModelReader` / `M2Era1121ModelReader` / the `0x108`+
`M2ModelReader` — which is architecturally separate from `FormatProfileRegistry`. Phase 0 settles
which mechanism is authoritative before any other phase touches a reader.

## Technical Context

**Language/Version**: C# / .NET 10.

**Primary Dependencies**: No new dependencies. Existing `WowViewer.Core.IO` readers
(`M2ModelReaderDispatcher`, `M2Era100/`, `M2Era1121/`, `M2/M2ModelReader.cs`), `WowViewer.Core`
model documents, the vendored Warcraft.NET library (`libs/ModernWoWTools/Warcraft.NET`,
`WarcraftNetM2Adapter.cs`) as the FR-008/FR-009 parity reference, and optionally Spec 193's Benilla
external oracle for 1.x-specific cross-checks.

**Storage**: Survey records written under `wow-viewer/output/` as workspace artifacts (reused
convention from Spec 154). No client data enters the repository.

**Testing**: xUnit — `tests/WowViewer.Core.Tests`. Real-client reads are operator-run and recorded
as evidence; unit tests cover pure structure validation, survey-record shape, and format-detection
logic.

**Target Platform**: Cross-platform library; survey and diagnostics driven from the existing
`tools/inspect` CLI; rendering proof is the Windows viewer app.

**Project Type**: Format-reader library plus thin CLI/viewer surfaces (existing multi-project C#
solution — no new project).

**Performance Goals**: None — correctness work. Rendering-performance for this era is Spec 136's
separate, non-overlapping concern.

**Constraints**:

- Hard ceiling at 4.0.0 (inherited from Spec 154 SC-008). Nothing at or beyond it is read,
  surveyed, or referenced by this feature.
- The unit of support is the build, not the version word or the expansion (Spec 154's FR-011/FR-012
  discipline) — structurally significant changes land in patch releases without a version-word
  bump; the measured library already proves this (three distinct `3.0.1` builds: 8303/8334/8391).
- `MDLX` (Alpha route) output must be byte-identical after this work.
- The `0x108` route (3.3.0+) must continue reading at least as well as today.
- No model in the staged library may cause an unhandled termination.
- No surface introduced by this feature may route through `WowViewer.Core.Anim` —
  `PathNormalizer.cs` throws `InvalidOperationException` on any path containing the staged client
  library root (a known, separately-tracked constitution contradiction, deliberately not bundled
  here; see Spec 154's plan.md "Recorded hazard").
- AGENTS.md §10: no new members in `WorldScene`/`ViewerApp`; new logic lives in owned service
  classes or the existing `WowViewer.Core.IO`/`Core.Runtime` libraries.

**Scale/Scope**: The staged client library holds roughly a dozen relevant builds across
1.0.0–3.0.1 (per Spec 154's partial enumeration); the surveyed model set is a fixed handful of
representative character/creature/doodad models present across eras, not a full archive sweep. The
fuckported-asset and light-emitter pillars (US4/US5) are scoped to whatever concrete assets the
operator can supply or that are already present in the staged library — this feature does not go
hunting for every fuckported asset that might exist.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1 design.*

| Principle / Rule | Status | Notes |
|---|---|---|
| I. Repo Independence | **PASS** | All work inside `wow-viewer/`. Client roots stay runtime configuration. |
| II. Library-First | **PASS** | Readers, dispatch, and the fuckported/light-effect logic live in `WowViewer.Core.IO`/`Core.Runtime`; viewer and `tools/inspect` stay thin consumers. |
| III. Real-Data Validation | **PASS** | Every phase's exit gate is a staged-client read or render, not a unit test alone (AGENTS.md §6). |
| VI. No Client Path Assumptions | **PASS** | Roots are CLI/runtime arguments throughout. |
| Read-Only Reference Codebase | **PASS** | `gillijimproject_refactor` untouched. |
| **Format Reader/Writer Ownership** | **PASS — load-bearing** | Existing readers are extended, never replaced or duplicated. `M2Era100Constants` already records the `0x100` layout (Ghidra-confirmed); Phase 2 consumes it rather than restating it. FR-014's reconciliation is exactly this rule applied to a resolver that may be a second, competing mechanism. |
| AlphaWdtWriter Frozen | **N/A** | Untouched — no terrain writer work here. |
| One Phase at a Time | **PASS** | Phases below are sequential; each ends in a recorded validation gate. |
| Bite-Sized Plans | **PASS** | Each phase has at most 8 steps. |
| AGENTS.md §4 (working readers) | **PASS** | Spec 235's own Context section is the required evidence of a proven bug before touching `M2Era100ModelReader`/`M2ModelReader`/`FormatProfileRegistry`. |
| AGENTS.md §10 (no god-class growth) | **PASS** | New logic (fuckported-asset check, light-effect wiring) lands in owned classes under `Core.IO`/`Core.Runtime`, not new `ViewerApp_*`/`WorldScene` members. |

**Post-Phase-1-design re-check (2026-09-10)**: data-model.md and the two contracts introduce no new
project, no new dependency, and no new storage mechanism beyond what Spec 154 already established
(workspace-local survey-record JSON). All rows above still PASS; no Complexity Tracking entry
beyond the three already listed is required.

### Recorded hazard (inherited from Spec 154): `Core.Anim.PathNormalizer` rejects the staged library

Unchanged from Spec 154's own finding — see Constraints above. No new surface in this feature may
route through `WowViewer.Core.Anim`.

## Project Structure

### Documentation (this feature)

```text
specs/235-legacy-mdx-m2-rendering/
├── spec.md
├── plan.md              # this file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   ├── survey-record.md          # adapted from Spec 154's contract (US4 rig-projection scope dropped)
│   └── fuckported-asset-parity.md # NEW — US4 contract
├── checklists/
│   └── requirements.md
└── tasks.md              # Phase 2 of speckit — generated by speckit-tasks, not here
```

### Source Code

```text
wow-viewer/
├── src/core/WowViewer.Core.IO/
│   ├── M2Chunked/
│   │   └── M2ModelReaderDispatcher.cs      # candidate real dispatch authority — Phase 0 confirms
│   ├── M2Era100/
│   │   ├── M2Era100Constants.cs            # 1.0.0 layout, Ghidra-confirmed — consumed, not restated
│   │   └── M2Era100ModelReader.cs          # gains the bone parser it never had (D1/D2)
│   ├── M2Era1121/
│   │   └── M2Era1121ModelReader.cs         # existing 1.12.1-shaped `0x100` route
│   ├── M2/
│   │   ├── M2ModelReader.cs                # 0x108+ reference reader; camera-record crash (D3) lives here
│   │   └── M2GeometryReader.cs
│   ├── M2Survey/                           # NEW (reused design from Spec 154) — read-attempt reporting
│   └── M2/{MdxToM2Converter.cs, M2ToMdxConverter.cs}  # FR-015 reconciliation target (one of two copies)
├── src/viewer/WoWViewer/Terrain/
│   ├── FormatProfileRegistry.cs            # the MDX/M2 resolver with the confirmed major>=1 MDX
│   │                                       # fallback bug and major==1 M2 null-return bug — Phase 0
│   │                                       # determines its real relationship to M2ModelReaderDispatcher
│   └── Transfer/M2ToMdxConverter.cs        # FR-015 reconciliation target (the other copy)
├── src/viewer/WoWViewer/Rendering/
│   └── WarcraftNetM2Adapter.cs             # FR-008/FR-009 parity reference entry point
├── src/core/WowViewer.Core/Mdx/
│   ├── MdxLightSummary.cs                  # existing parse-side light-node data — FR-010 foundation
│   └── MdxLightType.cs
├── tools/inspect/WowViewer.Tool.Inspect/   # thin: `m2 survey`, new `m2 fuckport-check` (US4)
└── tests/WowViewer.Core.Tests/             # structure, dispatch, survey-record, light-effect tests
```

**Structure Decision**: Extend the existing layout; no new project. `M2Survey` (namespace under
`Core.IO`) is reused verbatim from Spec 154's design — it must observe every reader without any
reader depending on it. The fuckported-asset check (US4) is new, owned code under `Core.IO`
alongside the existing converters it audits (FR-015). The light-effect wiring (US5) extends the
existing MDX render/effect path in `Core.Runtime`/the viewer's MDX renderer, consuming
`MdxLightSummary`/`MdxLightType` rather than re-parsing.

## Phases

Each phase ends validated (a real staged-client read/render, not just a build), per AGENTS.md §6 and
this project's "One Phase at a Time" discipline. Phase N+1 does not begin until Phase N's validation
is recorded.

### Phase 0 — Reconcile the resolution mechanisms and complete the measurement (US1 foundation)

Resolves FR-014 and Spec 154's still-open research items before any reader code changes.

1. **FR-014**: trace every caller of `FormatProfileRegistry.ResolveMdxProfile`/`ResolveModelProfile`
   and every caller of `M2ModelReaderDispatcher`. Determine: are these on the same load path, does
   one feed the other, or is `FormatProfileRegistry`'s MDX/M2 resolution dead/unused for actual
   model rendering (e.g. used only for a different concern like terrain-context doodad hints)?
   Record the answer with file/line evidence — do not guess.
2. Enumerate every staged build at or below 4.0.0 in the configured client library and record its
   identity (inherited from Spec 154 Phase 0 step 1 — not yet done there either).
3. For each build, record what its representative character/creature/doodad models declare (magic,
   version word) and which route the *actual* authoritative dispatcher (per step 1) selects.
4. Survey the three known `3.0.1` builds (8303, 8334, 8391) **separately** — do not assume they
   agree.
5. Determine whether any staged build lands in `0x102`–`0x106` (Spec 154's open item — this decides
   how much of Phase 3 below is real work against a range with no representative).
6. Resolve the `4.0.0.11927` camera-record-crash contradiction: does the viewer's render path and
   the dispatcher/inspection path reach the same code for the same file? Name the cause.
7. Confirm Warcraft.NET's actual supported version range (FR-008/FR-009 depend on knowing this, not
   assuming it covers everything) and identify at least one concrete fuckported asset (operator-
   supplied or found in the staged library) to validate US4 against.
8. Confirm whether the US5 torch/light gap is a parsing absence or a runtime/effect-wiring absence
   by tracing one concrete torch-bearing MDX model end to end.

**Exit gate**: `research.md` fully answers all 8 items with recorded evidence; no "assumed" entries.
This phase produces no rendering change.

### Phase 1 — US1: survey capability and records

Reuses Spec 154's `M2Survey` design (contracts/survey-record.md) near-verbatim; extends it to also
record MDX builds (Spec 154 only covered M2).

1. Bring over the `SurveyRecord`/`BuildIdentity`/`LayoutSelection`/`SectionOutcome` shapes from
   Spec 154's data-model.md/contracts, extended with an MDX-specific section outcome
   (`lightEffect: NotPresent|Succeeded|Failed`) for US5's later use.
2. Make reading report per-section outcomes instead of aborting the whole document on first failure,
   across whichever dispatcher Phase 0 confirmed authoritative.
3. Convert unhandled failures into reported, positioned failures (never process termination).
4. Add/extend the `m2 survey` CLI command over a configured archive root and the representative
   model set, covering both MDX and M2 assets.
5. Emit the survey record for every build enumerated in Phase 0; commit records as evidence.
6. Add tests for record shape and failure-position reporting on a crafted malformed input.

**Exit gate**: SC-001 met — every build in range has a complete survey record, no "unknown" layout,
no unexplained failure.

### Phase 2 — US2/US3: `0x100`-era geometry and skeleton (the core "objects render" fix)

Reuses Spec 104's embedded-skin-profile mechanism (data-model.md, contracts/m2-format-profile.md —
already Ghidra-confirmed for 1.0.0) and Spec 154's bone-validation rules (D1/D2 fixes).

1. Populate `embeddedSkinProfileCount`/`Offset` from the header's real view count/offset for the
   `0x100`-era route instead of hardcoding zero (Spec 104 Decision 2).
2. Materialize division zero (LOD 0) into render vertices, sections, batches, and texture lookups
   per the already-recorded `M2Era100Constants` layout (Spec 104's confirmed offsets: divisions
   `0x4C`, vertices `0x44`, textures `0x5C`, etc.) — do not re-derive these.
3. Wire bone parsing into `M2Era100ModelReader` consuming the same already-recorded constants (Spec
   154 D1); stop any fallback path from reading `0x100` bones at the `0x108` stride (D2's 20-byte
   drift).
4. Validate every bone (finite pivot, in-range or "no parent" parent, acyclic walk) and every
   submesh/texture-unit range against file bounds; reject and fall back to bounding box on failure,
   never crash (FR-005/FR-006).
5. Route the resulting geometry through the native `M2StaticRenderModel`/`M2Renderer` path — never
   construct `MdxFile`/`MdxRenderer` or an M2-to-MDX compatibility model for this route (Spec 104
   FR-011).
6. Verify against the known-failing 2.0.0.5610 Blood Elf/Night Elf models (previously `bones=0`,
   fail at bone index 10) and at least one 1.0.0 and one 1.12.1 model.
7. Recheck the `0x108`+ route for zero regression (151-bone Blood Elf at 3.3.0.10958 remains intact).

**Exit gate**: SC-002/SC-003/SC-004 met for the `0x100`-era route specifically: geometry renders
with bound textures, or an accurate bounding box; skeletons are complete and finite; no regression
on the known-good route.

### Phase 3 — US2/US3 continued: extend to `0x101`–`0x107` (only for builds Phase 0 proves exist)

Scope is set by Phase 0 step 5 — never claim a sub-range with no staged representative.

1. Establish each in-range build's real layout from its own bytes (evidence-based, per FR-002 —
   never inferred from a neighboring build).
2. Route each to a reader consuming that build's recorded layout, reusing Phase 2's validated
   mechanism rather than a new one where the layout matches.
3. Replace any blanket "unsupported era" refusal with either a successful read or a positioned,
   specific failure (FR-006).
4. Verify per build, never per range — the three `3.0.1` builds (8303/8334/8391) are each verified
   independently (FR-012).
5. Confirm the `0x107` model from 3.0.1.8303 (previously an unhandled "2.x TBC era, not yet
   supported" refusal) now reads or fails specifically.
6. Re-run the Phase 1 survey and diff it against the Phase 1 baseline.

**Exit gate**: every targeted build in `0x101`–`0x107` reads or fails specifically, each with its
own record; SC-002/SC-003 extended to cover this sub-range.

### Phase 4 — US4: fuckported-asset compatibility (NEW — no direct prior art)

1. Using the concrete fuckported asset(s) identified in Phase 0 step 7, diagnose exactly how its
   chunk rewriting diverges from a standard file of its declared format/version.
2. Determine whether the existing reader can be made tolerant of that specific divergence (a
   bounds/validation relaxation) versus needing a distinct repair/normalization step before the
   standard reader runs — prefer the former; only add a repair step if the file is genuinely
   non-standard in a way no relaxation can absorb.
3. If a repair/normalization step is needed, check whether it belongs with the existing
   `WmoV17ToV14Converter` pattern (a precedent for backward-compatibility asset handling) or is
   MDX/M2-specific and belongs beside the converters targeted by FR-015.
4. Render the asset and compare against Warcraft.NET's (or Benilla's, for 1.x) parse of the same
   file as the parity check (FR-008).
5. Ensure an asset neither this reader nor the external reference can parse reports a specific,
   per-asset failure — never a silent drop (FR-009).
6. **FR-015, folded in here as it's mechanically small**: reconcile the two existing MDX↔M2
   converter implementations to one owned copy; delete the redundant one; update all callers.
7. Add regression tests pinning the specific fuckported-asset case(s) validated.

**Exit gate**: SC-006 met — the identified fuckported asset(s) render using the external-reference
parity path; FR-015's converter duplication is resolved.

### Phase 5 — US5: MDX torch/light-emitter effects

Reuses the light-specific portion of Spec 104's Phase 4 (MDX material/effect compatibility repair),
scoped down to the light-node piece relevant here — not the full particle/multi-stage-shader
parity that remains out of scope.

1. Confirm from Phase 0 step 8 whether the gap is parse-side or runtime-wiring-side; if parsing
   already works (`MdxLightSummary`/`MdxLightType`), this phase is runtime/effect-wiring only.
2. Parse/resolve classic `LITE` light-node records and their pivots (after `PIVT`), keeping
   self-illumination material state separate from native light records (Spec 104 Phase 4 steps 1-2,
   if not already covered by Phase 0's findings).
3. Apply static Omni/Ambient `LITE` values to the bounded MDX model-local lighting path only — do
   **not** promote them to global scene lights; that cross-object light-transport contract is
   explicitly a separate, not-yet-approved slice (Spec 104 Phase 4 step 3, carried forward as an
   explicit boundary here too).
4. Render a torch-bearing MDX model and confirm the visual light/glow effect is visible.
5. For a light node type the effect pipeline doesn't yet model, log the gap explicitly (FR-010
   acceptance scenario 2) rather than skipping silently.
6. Add regression tests for the light-node resolution and the explicit-gap logging path.

**Exit gate**: SC-007 met — a torch-bearing MDX model shows its defined light/glow effect; unmodeled
light types are named, not silently dropped.

## Regression Protection

Applies at every phase exit gate, inherited from Spec 154's discipline:

- `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — 0 errors.
- `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — record the current
  pre-existing-failure set as this feature's baseline (do not assume Spec 154's "9 known failures"
  figure is still accurate; re-measure). The failure **set**, not just the count, must stay stable
  outside of this feature's own intended changes.
- `MDLX` (Alpha route) output byte-identical before/after — the 0.5.3 High Elf model is the standing
  check.
- The `0x108` route continues reading the 3.3.0 Blood Elf model with its full bone/geometry set —
  any deliberate change here needs its own evidence.

## Complexity Tracking

| Decision | Why needed | Simpler alternative rejected because |
|---|---|---|
| Reuse `M2Survey` namespace design from Spec 154 rather than inventing a new one | Must observe every reader without any reader depending on it; already designed and evidenced | Re-deriving would violate "consume existing records, don't restate" (spec FR-013) |
| Fold FR-015 (converter dedup) into Phase 4 rather than its own phase | It's mechanically small and naturally surfaces while diagnosing fuckported-asset handling | A standalone phase for a ~1-day dedup would violate "bite-sized" proportionality |
| US4 (fuckported assets) has no reusable prior-art phase design, unlike US1-US3 | Genuinely new scope from today's operator directive; neither Spec 104 nor 154 addressed non-standard chunk rewrites | N/A — no existing design to reuse |
