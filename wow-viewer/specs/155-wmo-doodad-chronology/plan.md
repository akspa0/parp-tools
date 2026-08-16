# Implementation Plan: Asset Reference Inventory — Expected vs Catalogued vs Present

**Branch**: `v0.5.3-dev` | **Date**: 2026-08-16 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/155-wmo-doodad-chronology/spec.md`

## Summary

Every world object and model in a build makes claims that assets exist. Collect those claims, resolve
each against what the build actually contains, and compare the result against what the listfiles name.
The disagreement is the deliverable.

Delivered by extending the inspection tooling. Every reader involved already exists — world-object
doodad and texture tables, model texture tables, the archive catalogue that already resolves per-asset
containers. What is added is per-asset reference reporting, corpus-wide sweeping, and the comparison.

The Mt. Hyjal effect objects are a known-true instance of the thing being hunted, so the sweep is gated
on flagging them without being told where to look.

## Technical Context

**Language/Version**: C# 13 / .NET 10

**Primary Dependencies**: None new. Existing format readers and the archive access layer in
`WowViewer.Core.IO`.

**Storage**: Reference ledgers and sweep reports under `wow-viewer/output/`. No client data enters the
repository — records carry paths, outcomes, and provenance only.

**Testing**: xUnit — `tests/WowViewer.Core.Tests`. Pure classification and comparison logic is unit
tested; corpus sweeps are operator-run against staged clients and recorded as evidence.

**Target Platform**: Cross-platform library plus thin CLI surface.

**Project Type**: Analysis capability over existing format readers.

**Performance Goals**: A sweep must complete over a full build without operator babysitting. No latency
target; it is an offline corpus job.

**Constraints**:

- Corpus comes from the archive access layer, never from archive internal listfiles.
- "Named by a listfile" and "readable from the build" are separate facts and are never merged.
- A sweep must not abort on one unreadable asset.
- **"Could not check" must never render as "nothing missing."**

**Scale/Scope**: Largest staged build is roughly 9,700 world objects and 17,300 models. The earliest is
532 world objects and 5,545 models.

## Constitution Check

*GATE: evaluated before Phase 0, re-evaluated after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Repo Independence | **PASS** | All work inside `wow-viewer/`. Client paths remain runtime configuration. |
| II. Library-First | **PASS** | Extraction, resolution, and comparison live in `WowViewer.Core.IO`. The CLI gains thin commands only. |
| III. Real-Data Validation | **PASS** | The feature is real-data validation. Every record names its build and configured root. |
| IV. Model Architecture | **N/A** | No ML component. |
| V. Streaming-First Dataset | **N/A** | Not a dataset pipeline. |
| VI. No Client Path Assumptions | **PASS** | Roots are CLI arguments. See the `PathNormalizer` hazard below. |
| Read-Only Reference Codebase | **PASS** | `gillijimproject_refactor` untouched. |
| Format Reader Ownership | **PASS — load-bearing** | No parser is written. Existing world-object doodad/texture readers, model texture tables, and the archive catalogue are consumed as-is. |
| Terrain Alpha Risk Area | **N/A** | No MCAL, terrain, or shader change. |
| AlphaWdtWriter Frozen | **N/A** | Untouched. |
| One Phase at a Time | **PASS** | Sequential phases, each ending in validation. |
| Bite-Sized Plans | **PASS** | One concern per step, ≤10 steps per phase. |
| Spec Docs Source of Truth | **PASS** | Spec and plan updated in the same commit as the code they describe. |

### Recorded hazard: `Core.Anim.PathNormalizer` rejects the staged library

`src/core/WowViewer.Core.Anim/PathNormalizer.cs` throws for any path containing the staged client
library root, pinned by tests, and the constitution records this as a known contradiction with
Principle VI and a deliberately unbundled follow-up.

**Consequence**: no Spec 155 surface may route through `WowViewer.Core.Anim`. Analysis lives in
`WowViewer.Core.IO`, which already owns the archive access layer. Repairing `PathNormalizer` stays out
of scope, unbundled as the constitution scoped it.

### Recorded dependency: Spec 154 blocks model sweeps on some builds

Spec 154 measured that model reading fails for builds declaring `MD20 0x100`–`0x107` and works for the
Alpha `MDLX` route and `MD20 0x108`. Model texture sweeping is therefore achievable today on the
earliest staged build (5,545 Alpha-route models) and on `0x108` builds, and is **blocked** for roughly
2.x through 3.0.1.

**Consequence**: Phase 2 scopes model sweeps to routes that read, and records blocked builds explicitly
as blocked. This is the plan's single most dangerous failure mode — a build whose models cannot be read
would otherwise report zero missing textures, which reads as "all healthy" and means "never checked."
The contracts make that structurally impossible to express.

## Project Structure

### Documentation (this feature)

```text
specs/155-wmo-doodad-chronology/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── reference-ledger.md
│   └── sweep-report.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (speckit-tasks — not created here)
```

### Source Code

```text
wow-viewer/
├── src/core/WowViewer.Core.IO/
│   ├── Wmo/                       # doodad + texture table readers — consumed as-is
│   ├── Mdx/                       # model texture tables — consumed as-is
│   ├── M2Chunked/                 # model route dispatch — consumed as-is
│   ├── Files/                     # archive catalogue, native archive service, virtual file reader
│   └── AssetReferences/           # NEW — extraction, resolution, comparison
└── tools/inspect/WowViewer.Tool.Inspect/   # thin: reference dump, sweep, compare
```

**Structure Decision**: One new namespace `AssetReferences` under `Core.IO`. It depends on the readers
and the archive layer; nothing depends on it. Deliberately **not** in `Core.Anim` — see the hazard.

## Phases

Each phase ends validated, not merely coded. **Phases 1–3 are the MVP** that answers the driving
question; 4–7 build on it.

### Phase 0 — Locate the control and map what is readable (research)

Nothing is built until the positive control is known to be reachable.

1. Identify which staged build(s) contain the Mt. Hyjal effect objects, and their asset paths.
2. Confirm that build's model route reads today. **If it does not, the control cannot fire and the
   whole plan is gated here** — say so rather than proceeding.
3. Record, per staged build, which model route applies and whether it reads, cross-referencing Spec 154.
4. Confirm the archive catalogue enumerates 532 world objects for the earliest build.
5. Establish how a presence probe distinguishes "absent" from "present but unreadable" — these must not
   collapse.
6. Establish whether the listfile index is usable as the catalogued set per build, and record its
   coverage against what the catalogue enumerates.

**Exit gate**: the control is located and reachable, and every staged build is marked readable or
blocked for models. No "assumed" entries.

### Phase 1 — US1: references from one asset

1. Define the reference record: referencing asset, reference kind, target path, resolution outcome.
2. Extract doodad references from one world object, reusing the existing reader.
3. Extract texture references from one world object, reusing the existing readers.
4. Extract texture references from one model on the Alpha route.
5. Extract texture references from one model on a readable `MD20` route.
6. Resolve one reference to present, absent, or unreadable.
7. Add a thin command reporting one asset's references and their outcomes.
8. **Validate against the Mt. Hyjal objects**: the missing texture reference must appear as absent.
9. Unit-test the resolution outcome mapping, including the unreadable case.

**Exit gate**: pointing the command at the control objects reports the missing texture. SC-003 shape is
in place for a single asset.

### Phase 2 — US2: sweep a whole build

1. Enumerate the world-object corpus from the archive catalogue.
2. Enumerate the model corpus, scoped to routes Phase 0 marked readable.
3. Sweep world objects, accumulating references.
4. Sweep models, accumulating references.
5. Continue past any individual unreadable asset, recording it.
6. Report examined counts per asset kind, plus skipped and unreadable counts.
7. Record blocked model routes as blocked, distinctly from zero findings.
8. Add a thin sweep command over a configured root.
9. **Validate: an untargeted sweep of the control build flags the Mt. Hyjal objects.**

**Exit gate**: SC-001, SC-002, SC-007 met. The sweep finds the control without being aimed at it, and
reports 532 world objects for the earliest build.

### Phase 3 — US3: the three-set comparison

1. Assemble the catalogued set per build from the listfile index.
2. Assemble the present set from the archive layer, independently of the catalogue.
3. Classify each referenced asset as working, catalogue-claims-but-absent, catalogue-gap, or missing.
4. Identify orphans — present, referenced by nothing swept.
5. Report per-category counts.
6. State the sweep's coverage limits alongside the orphan list, so an unswept reference source cannot be
   mistaken for absence of references.
7. Add a thin compare command.
8. Unit-test the classification across all four category combinations.

**Exit gate**: SC-004, SC-005 met. Every referenced asset lands in exactly one category, and nothing is
called missing merely because a listfile omitted it.

### Phase 4 — US4: candidate matches

1. Define candidate evidence: the nature of the difference between a missing path and a present asset.
2. Search present assets and orphans for near matches.
3. Report all candidates for a reference; never choose.
4. Verify every candidate is present in the same build.
5. Report the candidate result for the Mt. Hyjal texture specifically.
6. Unit-test that a candidate from another build or a non-existent path is rejected.

**Exit gate**: SC-006 met.

### Phase 5 — US5: cross-build chronology

1. Assemble per-build asset sets from the sweeps.
2. Derive introduction windows bounded by named builds.
3. Record disappearances as their own fact.
4. Record references missing in one build and resolving in another.
5. State granularity on every claim, and treat patch-adjacent builds as distinct.

**Exit gate**: SC-008 met.

### Phase 6 — US6: repair

1. Keep analysis and repair in separate paths so no analysis can mutate.
2. Apply a repair only for an unambiguous single candidate.
3. Record original, replacement, and evidence.
4. Make every repair reversible to the exact prior state.
5. Leave untouched and report anything ambiguous or candidate-less.
6. Demonstrate reversal by restoring and comparing.

**Exit gate**: SC-009, SC-010 met.

### Phase 7 — US7: conversion capability survey

1. Exercise each conversion operation against real staged data.
2. Record outcome per operation per build, naming what failed where it failed.
3. Compare the record against documented capability and correct overstatement.
4. Scope parity work against recorded defects only.

**Exit gate**: SC-011 met, and no documented conversion capability overstates what exists.

## Regression Protection

Checked at each exit gate:

- `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — 0 errors.
- `dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` — the core suite carries **9
  known pre-existing failures** unrelated to this work. The failure **set** must stay byte-identical;
  compare names, not counts. To baseline, stash **including untracked files** — otherwise new tests are
  left behind and the baseline build fails.
- No existing reader behaviour changes. This feature only consumes them.

## Complexity Tracking

| Decision | Why needed | Simpler alternative rejected because |
|---|---|---|
| New `AssetReferences` namespace | Extraction, resolution and comparison span world objects, models, and the archive layer; no existing owner covers all three | Adding it to a format reader would make one format's reader the owner of cross-format analysis |
| Analysis in `Core.IO`, not `Core.Anim` | `Core.Anim.PathNormalizer` throws on the staged client library | Using `Core.Anim` would require fixing that throw, which the constitution scoped as separate |
| Catalogued and present tracked separately rather than one "exists" flag | The four-way disagreement is the entire deliverable; a single flag collapses it | One flag cannot distinguish a catalogue gap from a missing asset, which is the distinction being measured |
| Blocked model routes recorded explicitly | A build whose models cannot be read would otherwise report zero missing textures, reading as healthy | Omitting blocked builds silently converts "not checked" into "nothing found" — the failure this project keeps hitting |
