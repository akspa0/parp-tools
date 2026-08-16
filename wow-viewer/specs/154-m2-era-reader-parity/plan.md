# Implementation Plan: M2 Reader Era Parity (1.x – 3.0.1)

**Branch**: `v0.5.3-dev` | **Date**: 2026-08-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/154-m2-era-reader-parity/spec.md`

## Summary

M2 model reading works at two ends — the Alpha `MDLX` route and `MD20 0x108` — and is broken or
refuses between them. Repair the `0x100`–`0x107` range against `0x108` as the reference, and expose
skeleton plus sequence data in one shape so a rig can be compared across routes.

The approach is survey-first. The premise "the later route works" was already falsified once by
measurement (4.0.0.11927 fails), so no reader is touched until every staged build has been read and
recorded. Layout knowledge for the `0x100` era already exists in the codebase and was never connected
to a bone parser; this is wiring plus verification, not format discovery.

## Technical Context

**Language/Version**: C# 13 / .NET 10

**Primary Dependencies**: None new. Existing `WowViewer.Core.IO` readers and `WowViewer.Core` model
documents.

**Storage**: Survey records written under `wow-viewer/output/` as workspace artifacts. No client data
enters the repository.

**Testing**: xUnit — `tests/WowViewer.Core.Tests`. Real-client reads are operator-run and recorded as
evidence; unit tests cover pure structure validation and the survey record shape.

**Target Platform**: Cross-platform library; survey driven from the existing `tools/inspect` CLI.

**Project Type**: Format-reader library plus a thin CLI surface.

**Performance Goals**: None. Correctness work. The survey runs offline over a staged library and has
no latency budget.

**Constraints**:

- Hard ceiling at 4.0.0 (SC-008). No later format is read, surveyed, or referenced.
- The unit of support is the build. Support for one build never implies the adjacent patch.
- `MDLX` output must be byte-identical after this work (FR-008).
- No model may cause an unhandled termination (FR-007).

**Scale/Scope**: The staged library reaches roughly a dozen relevant builds. The surveyed model set is
a fixed handful of character models present across eras, not a full archive sweep.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Repo Independence | **PASS** | All work inside `wow-viewer/`. Client paths stay runtime configuration. |
| II. Library-First | **PASS** | Readers and the survey capability live in `WowViewer.Core.IO`. `tools/inspect` gains a thin command only. |
| III. Real-Data Validation | **PASS** | The entire feature is real-data validation. Every claim records command, configured root, and build identity. |
| IV. Model Architecture | **N/A** | No ML component. |
| V. Streaming-First Dataset | **N/A** | Not a dataset pipeline. |
| VI. No Client Path Assumptions | **PASS** | Roots are CLI arguments. See the `PathNormalizer` hazard below. |
| Read-Only Reference Codebase | **PASS** | `gillijimproject_refactor` untouched. |
| Format Reader Ownership | **PASS — load-bearing** | Existing readers are extended, never replaced. `M2Era100Constants` already records the `0x100` layout; the plan consumes it rather than restating it. No new parser is written for a format that has one. |
| Terrain Alpha Risk Area | **N/A** | No MCAL, terrain, or shader change. |
| AlphaWdtWriter Frozen | **N/A** | Untouched. |
| One Phase at a Time | **PASS** | Phases below are sequential and each ends in validation. |
| Bite-Sized Plans | **PASS** | One concern per step, ≤10 steps per phase. |
| Spec Docs Source of Truth | **PASS** | Spec and plan updated in the same commit as the code they describe. |

### Recorded hazard: `Core.Anim.PathNormalizer` rejects the staged library

`src/core/WowViewer.Core.Anim/PathNormalizer.cs` throws `InvalidOperationException` for any path
containing `H:\CLIENTS`, pinned by `PathNormalizerTests`. The constitution names this as a known
contradiction with Principle VI and an explicitly tracked follow-up (amendment 1.1.0, migration note).

**Consequence for this plan**: no Spec 154 surface may route through `WowViewer.Core.Anim`. The survey
and the rig-comparison projection live in `WowViewer.Core.IO`. This is a routing decision, not a fix —
repairing `PathNormalizer` remains its own tracked item and is **out of scope here**, deliberately not
bundled, consistent with how the constitution scoped it.

## Project Structure

### Documentation (this feature)

```text
specs/154-m2-era-reader-parity/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── survey-record.md
│   └── rig-projection.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (speckit-tasks — not created here)
```

### Source Code

```text
wow-viewer/
├── src/core/WowViewer.Core.IO/
│   ├── M2Chunked/
│   │   └── M2ModelReaderDispatcher.cs      # routing; gains evidence reporting
│   ├── M2Era100/
│   │   ├── M2Era100Constants.cs            # layout already recorded — consumed, not restated
│   │   └── M2Era100ModelReader.cs          # gains the bone parser it never had
│   ├── M2Era1121/
│   │   └── M2Era1121ModelReader.cs         # 0x101 route
│   ├── M2/
│   │   ├── M2ModelReader.cs                # 0x108 reference; camera defect lives here
│   │   └── M2GeometryReader.cs
│   └── M2Survey/                           # NEW — read-attempt reporting + rig projection
├── src/core/WowViewer.Core/M2/             # model documents (skeleton shape)
├── tools/inspect/WowViewer.Tool.Inspect/   # thin: `m2 survey`, `m2 rig-project`
└── tests/WowViewer.Core.Tests/             # structure + record-shape tests
```

**Structure Decision**: Existing layout, extended. One new namespace `M2Survey` under `Core.IO` owns
the read-attempt record and the cross-route rig projection, because both must observe every reader
without any reader depending on them. Deliberately **not** in `Core.Anim` — see the hazard above.

## Phases

Each phase ends validated, not merely coded. Phase N+1 does not begin until Phase N's validation is
recorded in `research.md`.

### Phase 0 — Complete the measurement (research)

Resolves the open unknowns before any code changes. Output: `research.md` sections filled.

1. Enumerate every staged build at or below 4.0.0 and record its identity. No build is skipped for
   looking redundant.
2. For each build, record what its character models declare (magic and version word).
3. Record which route the current dispatcher selects for each, and the exact outcome.
4. Survey the three 3.0.1 builds (8303, 8334, 8391) **separately** and report whether they differ.
5. Determine whether any staged build lands in `0x102`–`0x106`; this decides how much of US3 is real.
6. Document the existing `0x100` disambiguation probe: what it inspects and why it separates the two
   `0x100` layouts.
7. Resolve the 4.0.0.11927 contradiction: establish whether the viewer's render path and the
   dispatcher path reach the same outcome for the same file, and name the cause.

**Exit gate**: every unknown above answered from a recorded read; no "assumed" entries.

### Phase 1 — US1: survey capability and records

1. Define the read-attempt record: per section, attempted / succeeded / failed-with-position.
2. Make reading report per-section outcomes instead of aborting the whole document on first failure.
3. Convert unhandled failures into reported failures across the dispatcher and reader surfaces
   (FR-007).
4. Record, per read, the build identity and the evidence that selected the layout (FR-001, FR-002).
5. Add a thin `m2 survey` command over a configured archive root and a model set.
6. Emit the survey record for every staged build; commit the records as evidence.
7. Add tests for the record shape and for failure-position reporting on a crafted malformed input.

**Exit gate**: SC-001, SC-002, SC-004 met. Every staged build has a complete record and no read
terminates the process.

### Phase 2 — US2: `0x100` skeletons

1. Add bone parsing to the `0x100` reader, consuming the existing recorded constants (FR-012).
2. Populate the model document's skeleton instead of passing none (FR-003).
3. Stop the geometry path from reading `0x100` bones through the `0x108` layout.
4. Validate every bone: finite pivot, in-range parent, acyclic parent walk (FR-004).
5. Distinguish "no bones" from "bones not read" in the document and in the survey record (FR-005).
6. Verify against the 2.0.0.5610 Blood Elf and Night Elf models — the two that fail at bone index 10.
7. Add tests for the structural validators, including a cyclic-parent and an out-of-range-parent case.

**Exit gate**: SC-003 met. Both previously failing models read to completion with complete, finite,
structurally valid skeletons.

### Phase 3 — US3: `0x101`–`0x107`

Scope is set by Phase 0 step 5 — only builds shown to exist are targeted.

1. Establish each target build's layout from its own bytes, recorded as evidence.
2. Route each to a reader that consumes that build's recorded layout.
3. Replace the blanket refusal with either a successful read or a positioned, specific failure
   (FR-006).
4. Verify per build, never per range; record each build separately (FR-011).
5. Confirm the `0x107` Blood Elf model from 3.0.1.8303 reads.
6. Re-run the Phase 1 survey and diff it against the Phase 1 baseline.

**Exit gate**: every targeted build reads or fails specifically, each with its own record. No build
claimed on another's result.

### Phase 4 — US4: cross-route rig projection

Not blocked by Phases 2–3 — `MDLX` and `0x108` both work today. **May be promoted to run immediately
after Phase 1** if an early answer to the driving comparison is wanted; that is a scheduling call for
the operator, not a dependency.

1. Define the projection: bone identity, parent, pivot, plus the sequence table (FR-010).
2. Project from the `MDLX` route.
3. Project from the `MD20` routes.
4. Add a thin command emitting the projection with both build identities attached.
5. Compare structurally — whether the earlier bone set appears within the later one with corresponding
   parents and pivots — never by bone count, which differs 54 against 151.
6. Produce the High Elf / Blood Elf comparison with Night Elf and Human as controls.

**Exit gate**: SC-006 met. A comparison exists that reports correspondence for related rigs and
difference for unrelated ones, with controls run.

## Regression Protection

Applies to every phase, checked at each exit gate:

- `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — 0 errors.
- `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — the core suite carries **9
  known pre-existing failures unrelated to this work**. The failure **set** must stay byte-identical;
  a changed count is a regression until proven otherwise, and comparing counts alone is not sufficient.
- `MDLX` reads produce identical output before and after (FR-008). The 0.5.3 High Elf model is the
  standing check.
- The `0x108` route continues to read the 3.3.0 Blood Elf model with 151 bones and geometry available
  (FR-009). Any deliberate change here needs its own evidence.

## Complexity Tracking

| Decision | Why needed | Simpler alternative rejected because |
|---|---|---|
| New `M2Survey` namespace rather than extending an existing one | The survey must observe every reader without any reader taking a dependency on it | Putting it inside a reader namespace would make one route's reader the owner of all routes' diagnostics |
| Survey capability in `Core.IO`, not `Core.Anim` | `Core.Anim.PathNormalizer` throws on the staged client library | Using `Core.Anim` would require fixing that throw, which the constitution scoped as a separate tracked item |
| Per-section read outcomes rather than whole-document success | FR-006/FR-007 require positioned failures and no unhandled termination; a document that aborts on first failure cannot be surveyed | All-or-nothing reading is what produces the current unhandled exceptions and the uninformative `bones=0` |
