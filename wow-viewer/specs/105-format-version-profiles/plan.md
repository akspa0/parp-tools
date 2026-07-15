# Implementation Plan: Unified M2 Format Version Profiles

**Branch**: `v0.5.1` (no feature branch — see Structure Decision) | **Date**: 2026-07-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/105-format-version-profiles/spec.md`

## Summary

**Scope expanded 2026-07-15**: make a 1.0.0 M2 **render correctly** — textured, lit, and animated —
across three rendering pillars, resting on a unified version-profile foundation. A live-viewer
screenshot showed animation alone leaves the model a gray, flat blob.

**Pillar sequencing (texture → lighting → animation)** — most-visible-first, and a textured model is the
prerequisite for judging the other two:

- **P-Texture** (visible, independent of the architecture): the 1.0.0 model renders gray because the
  batch→combo→texture *resolution* yields no bindable path — not a parse or render-support gap (reader
  parses filenames at `M2Era100ModelReader.cs:302`; renderer has a full load+replaceable path at
  `M2Renderer.cs:1023`). Root cause must be diagnosed (combo mis-index vs BLP-not-found vs
  replaceable-with-no-skin) before the fix.
- **P-Lighting** (confirm then correct): Lambert already runs (`M2Renderer.cs:854`, gated on
  `Material.IsUnshaded` at `:697`); a gradient is visible. Audit the material-flag(`0x01`=UNLIT)→
  `IsUnshaded` mapping; the "flat" look is mostly the missing texture, so judge only after P-Texture.
- **P-Animation** (needs the architecture): close the three era gaps below.

The **animation** pillar rests on the version-profile foundation and these three era gaps:

1. **Track addressing** — 1.0.0 slices one flat key array per sequence via inclusive interpolation ranges; Wrath+ nests per-sequence arrays and has no ranges. Our contract models only Wrath.
2. **Sequence time base** — 1.0.0 sequences span a shared global timeline (`start`/`end`); Wrath+ are sequence-local (`duration`). Our contract models only Wrath.
3. **Sequence record layout is actively misread** (found during planning, see research R1) — era-100 uses Wrath field offsets under a correct `0x44` stride, so every field from `+0x04` on is shifted 4 bytes. This is a live bug. (It corrupts the *displayed* duration; it is NOT what freezes the pose — that is `bones: null/[]`.)

The technical spine of the animation pillar is **R2's `{Start, Duration}` normalization**: it expresses
both eras' time base in two fields, collapses the sampler to one rule, and reduces to today's exact
behaviour for Wrath (`Start = 0`), making the 3.x/4.x no-regression guarantee structural rather than
empirical.

Scope is the M2 surface only. `FormatProfileRegistry` survives as the ADT/WMO/MDX owner; only its inert
M2 half is deleted. Both era readers drop bones (`M2Era100ModelReader.cs:118` `bones: null`;
`M2Era1121ModelReader.cs:95` `bones: []`) — the era-100 fix is the proving slice; era-1121 shares the
gap and is tracked, not fixed here.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: None new. Core M2 code has zero Warcraft.NET dependency and this feature does not add one.

**Storage**: N/A (in-memory parse; on-disk artifacts are the committed baseline JSON)

**Testing**: xUnit. Precedent: `M2Era100ModelReaderTests`, `M2Era1121ModelReaderTests`.

**Target Platform**: Windows (viewer); core libraries are platform-neutral

**Project Type**: Library-first (`WowViewer.Core*`), consumed by a desktop viewer

**Performance Goals**: No regression. The sampler is per-bone-per-frame; R2 removes a modulo branch and adds one add. FR-006 forbids per-track mutable state, so the client's key cache is deliberately not ported — a stateless bracketing search over an inclusive range is `O(log n)` on key count (typically < 100).

**Constraints**:
- 3.x/4.x M2 behaviour must be **bit-identical** to the pre-change baseline (FR-015/016). Baseline captured **before** any shared type is touched (R7) — ordering is load-bearing.
- Net M2 version-dispatch + profile code must **shrink** (SC-004). Feasibility verified in research R6 (≈ −50 lines).
- Every layout fact cites a Ghidra address or is marked provisional, distinguishable at point of use (FR-011).
- Core must not reference the viewer (Constitution I). Resolved by dependency inversion — research R4.
- SC-001 (1.0.0 model visibly animates) **cannot be self-certified** — AGENTS Rule 0.

**Scale/Scope**: ~5 files in core, 1 deletion in the viewer, 2 test files. Four eras (`0x100` era-100, `0x100` era-1121, `0x108` 3.x, `0x109` 4.x) + MDLX, of which only era-100 changes behaviour.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1.*

| Principle | Status | Notes |
|---|---|---|
| **I. Repo Independence** | **PASS** | Nothing outside `wow-viewer/`. Core→viewer reference explicitly avoided via R4 dependency inversion. |
| **II. Library-First** | **PASS — this feature exists to fix a violation of it** | "One canonical owner per format surface" is today violated for M2 (two schemes). After this, the M2 surface has exactly one owner, in core. Satisfied *per surface*: `FormatProfileRegistry` legitimately remains the ADT/WMO/MDX owner. |
| **III. Real-Data Validation** | **PASS (deferred to user)** | Every gate validates against staged clients. SC-001 is user-run per AGENTS Rule 0. Mock assets are explicitly not sufficient. |
| **IV. Residual Model Chain** | **N/A** | Not an ML feature. |
| **V. Streaming-First Dataset Pipeline** | **N/A** | Not a dataset feature. |
| **VI. No Game Client Path Assumptions** | **PASS (with an amendment in flight)** | Validation uses staged clients only. The Ghidra evidence underpinning this spec was read from a `H:\CLIENTS`-imported program; the user clarified 2026-07-15 that the original distrust (broken clients of unknown origin) no longer applies — it is now a curated SSD staging area fed from WoWArchive. Principle VI is being amended under Governance; this plan does not depend on the outcome, since it reads no client path directly. |
| **Read-Only Reference Codebase** | **PASS** | `gillijimproject_refactor` untouched. |
| **Format Reader/Writer Ownership** | **PASS** | Existing working parsers are not rewritten. Era-100 is corrected against evidence; 3.x/4.x is untouched behaviourally. |
| **Terrain Alpha Risk Area** | **PASS** | Explicitly out of scope (FR-018/020). ADT/WMO profiles are not touched. |
| **AlphaWdtWriter is Frozen** | **PASS** | Not touched. |
| **One Phase at a Time** | **PASS** | Each phase ends with a validation gate, not code. Phase N+1 blocked on Phase N's gate. |
| **Spec Docs Are Source of Truth** | **PASS** | Spec 104's `contracts/m2-format-profile.md` is updated in the same phase that changes the layouts it describes. |
| **Bite-Sized Plans** | **PASS** | Max 10 steps/phase; one concern per step; each independently validatable. |

**Gate result: PASS.** No unjustified violations; Complexity Tracking is therefore empty.

## Project Structure

### Documentation (this feature)

```text
specs/105-format-version-profiles/
├── plan.md              # This file
├── spec.md              # Feature spec
├── research.md          # Phase 0 output — R1..R7 decisions
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output — validation commands (user-run)
├── contracts/
│   └── m2-era-profile.md
├── checklists/
│   └── requirements.md
└── tasks.md             # NOT created by speckit-plan
```

### Source Code

```text
wow-viewer/src/core/
├── WowViewer.Core/M2/
│   ├── M2AnimationBlocks.cs          # MODIFY: track addressing mode + interp-range ref
│   ├── M2SequenceDefinition.cs       # MODIFY: add Start (Duration retained, meaning preserved)
│   └── M2EraProfile.cs               # NEW: the profile contract + evidence citation
├── WowViewer.Core.IO/
│   ├── M2Chunked/M2ModelReaderDispatcher.cs   # MODIFY: deterministic resolution, build hint
│   ├── M2Era100/M2Era100ModelReader.cs        # MODIFY: fix sequence offsets; populate bones (Phase 3)
│   ├── M2Era100/M2Era100Constants.cs          # MODIFY: sequence field offsets + bone/track layout
│   └── M2Era1121/M2Era1121ModelReader.cs      # MODIFY: mark sequence offsets provisional (no fix)
└── WowViewer.Core.Runtime/M2/
    └── M2TrackSampler.cs             # MODIFY: flat+ranges path; unified Start+Duration time rule

wow-viewer/src/viewer/WoWViewer/
├── Terrain/FormatProfileRegistry.cs  # MODIFY: delete M2Profile records + ResolveModelProfile ONLY
├── Rendering/WarcraftNetM2Adapter.cs # MODIFY: remove now-dangling M2 validation call (see R6 note)
├── Rendering/WowViewerM2RuntimeBridge.cs # MODIFY (Phase T): era-100 texture-combo resolution
└── Rendering/M2Renderer.cs           # AUDIT (Phase T/L): shared texture load + IsUnshaded mapping — era-100-scope changes where possible

wow-viewer/tests/WowViewer.Core.Tests/
├── M2Era100ModelReaderTests.cs       # EXTEND (bones, texture bindings)
├── M2TrackSamplerEraTests.cs         # NEW
└── M2ThreeXBaselineRegressionTests.cs # NEW (Phase 0 — animation + rendering baseline)
```

**Structure Decision**: Existing library-first layout; no new projects. Work proceeds on `v0.5.1`
without a feature branch, matching how specs 103/104 were done in this repo — the `.specify` scripts
are PowerShell-only and resolve to the git root (`parp-tools`) rather than `wow-viewer/`, so spec
artifacts are written manually. Recorded in the memory bank.

## Phases

Each phase ends with a **gate**. Phase N+1 does not start until Phase N's gate passes (Constitution:
One Phase at a Time — "done means validated, not coded").

**Execution order**: Phase 0 (baseline) → **Phase T (texture)** → **Phase L (lighting)** → Phase 1
(profile contract) → Phase 2 (track addressing) → Phase 3 (bones) → Phase 4 (era resolution).
The rendering pillars (T, L) run **first** — most visible, and independent of the architecture work
(they touch the renderer and the era-100 material bridge, not the shared `M2TrackDefinition`/
`M2SequenceDefinition` types). The architecture/animation phases (1–4) are unchanged from the
original plan and keep their numbers.

### Phase 0 — Pin the baseline (blocking; nothing else may start)

Ordering is load-bearing (R7): once a shared type or shared renderer path changes, a baseline measures the new code against itself.

1. Name the 3.x/4.x regression set: specific models from staged clients, incl. 4.0.0 data.
2. Add `M2ThreeXBaselineRegressionTests` capturing, per model: sequence `Duration`, per-bone sampled TRS at fixed times across several sequences, section/batch counts.
3. **Capture a 3.x/4.x rendering baseline too** (bound texture count/paths per section, per-batch blend mode, per-material `IsUnshaded`) — Phase T and Phase L may touch the shared `M2Renderer`, so texture/material state must be pinned, not just animation state.
4. Serialize the baseline to a committed JSON artifact.
5. Run and commit **before touching any shared type or shared renderer path**.

**Gate**: baseline committed and green on unmodified code. Any later diff is a real regression.

### Phase T — Texture resolution (P-Texture; most visible, run first)

Diagnose before fixing (FR-T1): guessing caused prior bugs, and the cause is not yet pinned.

1. **Diagnose** on CentaurKhan (staged 1.0.0): log, per batch, the raw texture-combo entry, the resolved lookup value (signed), whether it resolved to a hardcoded filename or a negative replaceable id, and whether the BLP was found. This step *reports*, it does not change behaviour. Classify the cause: combo mis-index vs BLP-not-found vs replaceable-with-no-skin.
2. Fix the classified cause in `BuildEra100Material` / the era-100 texture-binding path (`WowViewerM2RuntimeBridge.cs:220`), scoped to era-100 where possible so the 3.x baseline is untouched.
3. Ensure a negative combo entry routes to the replaceable path with a visible placeholder + log when unresolved — never a silent shared gray (FR-T2, FR-T4).
4. Ensure multi-stage batches bind each stage (FR-T3).
5. Extend `M2Era100ModelReaderTests` / add a bridge test asserting a hardcoded-texture batch yields a bound path and a replaceable batch yields the replaceable id, not gray.

**Gate**: solution builds; 3.x/4.x rendering baseline (Phase 0 step 3) **unchanged**. **USER runs the viewer**: CentaurKhan shows texture detail, not gray (SC-T1). Not self-certifiable.

### Phase L — Lighting (P-Lighting; confirm then correct)

Blocked on Phase T (FR-L3: the flat look is mostly the missing texture; judge only once textured).

1. **Audit** the material-flag→`IsUnshaded` mapping against the Ghidra evidence (flag bit `0x01` = UNLIT); confirm era-100 materials set it correctly rather than defaulting (FR-L1).
2. Fix the mapping only if the audit shows a defect; otherwise record that lighting already conforms and close the pillar.
3. If a fix touches the shared shader / `IsUnshaded` plumbing, gate on the 3.x/4.x rendering baseline (FR-015 extends to lighting).
4. Add/extend a test asserting a non-UNLIT era-100 material yields `IsUnshaded == false` and an UNLIT one yields `true`.

**Gate**: 3.x/4.x rendering baseline unchanged. **USER confirms** a textured 1.0.0 model is directionally lit where materials are not UNLIT, fullbright where they are (SC-L1). Not self-certifiable.

### Phase 1 — Profile contract in core + delete the inert M2 half

1. Add `M2EraProfile` in `WowViewer.Core` — per-era layout facts, each with an evidence citation or explicit provisional marker (FR-011).
2. Populate era table: era-100 (Ghidra-cited), era-1121 (provisional), 3.x/4.x (from working readers).
3. Point `M2ModelReaderDispatcher` at the table; **behaviour unchanged this phase** (trial-parse still present — it dies in Phase 4).
4. Delete the 5 inert `M2Profile` records, `ModelRootMagic`, `ResolveModelProfile` from `FormatProfileRegistry`. **Do not touch its Adt/Wmo/Mdx halves** (FR-018/020).
5. Remove the dangling M2 validation call in `WarcraftNetM2Adapter` — **pending user confirmation of the FR-017 reading (research R6)**.
6. Update spec 104's `contracts/m2-format-profile.md` in this same commit (Constitution: Spec Docs Are Source of Truth).

**Gate**: solution builds; full test suite green incl. Phase 0 baseline; measured net line count **negative** (SC-004).

### Phase 2 — Era-aware track addressing + time base (the shared-type change)

1. Add `Start` to `M2SequenceDefinition`; Wrath readers set `Start = 0` (R2).
2. **Fix era-100 sequence field offsets** against Ghidra: `start@0x04`, `end@0x08`, `movespeed@0x0C`, `flags@0x10`; set `Start = start`, `Duration = end - start` (R1).
3. Mark era-1121 sequence offsets **provisional** with a comment citing R1. **Do not fix by analogy.**
4. Add `M2TrackAddressingMode` + `InterpolationRanges` to `M2TrackDefinition<T>` (R3).
5. Unify `M2TrackSampler.ResolveSampleTime` to `Start + (elapsed mod Duration)` — reduces to today's expression when `Start = 0`.
6. Add the flat+ranges path to `TryReadSequenceSlice`: resolve `interpRanges[sequenceIndex]` → inclusive `[first,last]`; empty ranges ⇒ whole array (FR-003).
7. Reproduce the client's total-count clamp exactly (FR-004; spec Edge Cases) — **do not "fix" it**.
8. Keep the sampler stateless — no key cache, no per-track mutable state (FR-006).
9. Add `M2TrackSamplerEraTests` covering: inclusive bounds, empty ranges, global sequence override, degenerate range (`end ≤ start`), and the cross-range clamp.

**Gate**: Phase 0 baseline **bit-identical** (FR-015). New era tests green. Era-100 sequence durations become plausible (the "3333" pathology gone).

### Phase 3 — Populate era-100 bones

Blocked on Phase 2 (FR-007: wrong poses are worse than none).

1. Add bone/track layout constants to `M2Era100Constants` with Ghidra citations: `M2CompBone 0x6C`, `M2Track 0x1C`.
2. Read the `0x6C` bone array; parse the three `0x1C` tracks + pivot.
3. Replace `bones: null` at `M2Era100ModelReader.cs:118`.
4. Decode the rotation track as a compressed quaternion (`FUN_00720d30`).
5. Extend `M2Era100ModelReaderTests`: bone count/hierarchy, parent indices, pivots, track ranges.
6. **Do not** model per-bone independent animation or cross-animation blending (spec Assumptions) — note them in code where the client diverges.

**Gate**: parser tests green. **USER runs the viewer** against a staged 1.0.0 client (TrollMale, DoomGuard) and confirms visible bone motion → SC-001. Not self-certifiable.

### Phase 4 — Deterministic era resolution

1. Add an optional build hint to the core read API (R4 dependency inversion).
2. Pass the viewer's existing `_dbcBuild` through.
3. Implement + **measure** the header-size structural discriminator against real 1.0.0 and 1.12.1 models (R4 step 2 is currently reasoned, not measured — it stays provisional until this step).
4. Delete the trial-parse from `DetectEra` (FR-012).
5. Fail loudly on unrecognized, naming version and ambiguity (FR-013).
6. Test both models route correctly with the fallback disabled.

**Gate**: SC-005 (zero trial-parse fallbacks); both eras route deterministically; unknown fails loudly.

## Complexity Tracking

> Fill ONLY if Constitution Check has violations that must be justified.

**Empty — Constitution Check passed with no violations.**

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Phase T/L touch the **shared** `M2Renderer` (used by 3.x/4.x too) | **High** | Scope era-100 changes to the era-100 material bridge where possible; Phase 0 step 3 pins a 3.x/4.x rendering baseline; both phases gate on it. |
| P-Texture cause is not yet pinned (gray could be 3 different bugs) | **Medium** | Phase T step 1 is a *diagnosis-only* pass that classifies the cause before any fix (FR-T1). No guessing. |
| Phase 2 changes shared types used by working 3.x/4.x | **High** | R2's `Start = 0` makes Wrath reduce to today's exact expression — no-regression is structural, not empirical. Phase 0 baseline pins it independently. |
| Era-1121 sequence offsets are likely wrong too (R1) | **Medium** | Marked provisional, not fixed by analogy. Needs a 1.12.1 client in Ghidra. **Note: 1.12.1 may currently animate wrongly for the same reason and nobody has reported it** — worth an explicit user check. |
| Header-size discriminator (R4) is reasoned, not measured | **Medium** | Phase 4 step 3 measures it before relying on it; build hint is the proven path meanwhile. |
| FR-017 vs deleting the M2 validation call (R6) | **Low** | Flagged for user confirmation; plan works under either reading (shrink holds either way). |
| SC-001 cannot be self-certified | **Low** | Explicit user gate at Phase 3. Do not claim signoff on a green build. |
