# Implementation Plan: PM4-Guided Object Transfer and Museum Placement Repair

**Branch**: `176-object-transfer` | **Date**: 2026-08-25 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/176-object-transfer/spec.md`

**Planning artifacts**: [research.md](research.md), [data-model.md](data-model.md), [quickstart.md](quickstart.md), and [reconciliation.openapi.yaml](contracts/reconciliation.openapi.yaml)

**Dependency gate**: Implementation starts only after Specs 166, 167, 168, 173, and 175 have their focused checkpoints. This feature consumes the editor host/bridge/session and placement-authoring seams; it does not create a parallel editor surface while those owners are still absent.

## Implementation status (2026-08-26)

| Phase | State | Evidence |
|---|---|---|
| 0 Discovery / contract lock | Done | research.md, data-model.md, OpenAPI, this plan |
| 1 Pure proposal engine | Done | `Pm4ReconciliationEngine` + adapter; 81 focused editor tests include residual confidence + `AlreadyAligned` |
| 2 Placement transaction / IDs / name tables | Done | `AdtPlacementEditor` is the sole Core.IO mutation owner; `AdtPlacementWriter` deleted; MODF bounds translate with moves |
| 3 Editor bridge + viewport preview | Partial | Reconcile tab live; review list + bulk accept live; **no in-scene overlay**; preview still on the render thread |
| 4 Save / reload / provenance | Partial | Apply writes loose ADT + `.reconciliation.json` and records undo; real-client reload/independent-reader proof is user-owned |
| 5 Quality gate / handoff | Open | Focused editor tests pass; full-suite / visual / independent-reader gates remain |

**Committed through `6fa1adbb`.** Last landed UX: timestamped project output folder (`output/projects/<map>/<yyyyMMdd_HHmmss>`), residual-derived align confidence, `AlreadyAligned` no-op reporting, shared staged-save queue for authored edits.

**Uncommitted WIP (does not compile):** scene-discerned PM4/`_obj0.adt` pairs from `WorldScene.LoadedPm4Tiles`. `PrefillReconciliationPathsFromScene` was removed while [`ViewerApp_Sidebars.cs`](../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs) still calls it. Do not treat that working-tree slice as shipped.

**Next bounded implementation slice (one concern):** make the Reconcile tab compile with scene-discerned inputs — delete the leftover `PrefillReconciliationPathsFromScene()` call, delete unused `DrawReconciliationPathField`, drop unused path fields, then `dotnet build` the viewer. After that: in-scene overlay, then user-owned visual proof.

## Summary

Deliver a reviewed PM4-guided repair workflow for Museum map placements. The viewer loads a PM4 guide and its paired ADT/WDT placement catalog, reuses the existing PM4 object matcher to produce explainable alignment/substitution/clone proposals, shows those proposals immediately in the viewport, and applies only explicit user decisions through the existing placement and format writers. PM4 files and source game files remain read-only; accepted changes are staged as one undoable editor operation and saved as loose ADT/WDT outputs with a provenance sidecar.

The first implementation is deterministic and evidence-first. It does not train a model, add a second reader, or auto-confirm identity from proximity. Ambiguous and unresolved cases are useful visible results, not failures to hide.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing `WowViewer.Core`, `WowViewer.Core.IO`, `WowViewer.Core.PM4`, `WowViewer.Core.Runtime`, editor host/bridge/session contracts from Specs 166–168, `System.Numerics`, `System.Text.Json`, existing ImGui viewer shell, and existing ADT/WDT readers/writers.

**Storage**: Source PM4/ADT/WDT/client files are read-only inputs. Outputs are loose ADT/WDT files in the configured output directory plus a JSON provenance report. No MPQ/CASC output, database, client asset bytes, or new persistent corpus store is introduced.

**Testing**: Focused xUnit tests in `tests/WowViewer.Core.Tests` and `tests/WowViewer.Core.PM4.Tests`, contract/schema validation where available, source/build checks, and user-owned real-client visual/independent-reader validation from [quickstart.md](quickstart.md).

**Target Platform**: Windows desktop viewer, with runtime-configured client roots and output paths.

**Project Type**: Library-first desktop editor capability hosted by the existing WoW viewer.

**Performance Goals**: Preview computation must be off the render thread, cancellable, and incrementally refreshable for the measured scale of hundreds to roughly 1,000 PM4 objects and thousands of placements. The viewport must continue rendering the last committed/staged state while a preview refreshes. A numeric latency budget is deferred until a real Museum corpus is benchmarked; frame pacing must not depend on a corpus-wide matcher run.

**Constraints**: PM4 is read-only guidance; no game-install writes; no Blizzard-container output; no duplicate parsers; no silent matcher decisions; supported-era refusal; source-fingerprint validation; atomic multi-file commit; unaffected chunk byte preservation; no proprietary client bytes in committed artifacts; preserve Alpha/standard terrain separation; keep `AlphaWdtWriter.cs` frozen.

**Scale/Scope**: One reviewed Museum/PM4 map session initially, including multi-tile batches. Design for the measured PM4 corpus scale (904 PM4 objects mapped to 243 source assets) and normal ADT placement catalogs, without unattended corpus-wide repair.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / constraint | Status | Plan evidence |
|---|---|---|
| I. Repo independence | PASS | All new code/docs/tests stay under `wow-viewer/`; client roots are runtime configuration. |
| II. Library-first / one owner | PASS | Matching/reconciliation contracts live in core; IO uses existing readers/writers; viewer is a thin plugin surface. |
| III. Real-data validation | PASS | Paired PM4/Museum corpus and configured client fingerprint are mandatory gates; mocks only support unit isolation. |
| VI. No client-path assumptions | PASS | No source or portable config hardcodes a client root; quickstart records the operator's configured root. |
| VII. Blizzard containers are inputs | PASS | Only loose ADT/WDT content and JSON provenance are written; MPQ/CASC remain read-only. |
| Read-only legacy/reference tree | PASS | No changes under `gillijimproject_refactor/`. |
| Existing reader/writer ownership | PASS | No parser/serializer is added; any missing full-placement capability is extended in the owning core library. |
| Alpha writer freeze | PASS | `AlphaWdtWriter.cs` remains out of scope unless a proven compatibility defect reopens it. |
| One phase at a time / bite-sized work | PASS | Each phase has a focused checkpoint and no more than six implementation slices. |

No constitution violation requires a complexity exception.

## Project Structure

### Documentation (this feature)

```text
specs/176-object-transfer/
├── plan.md
├── spec.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/
    └── reconciliation.openapi.yaml
```

### Source Code (repository root)

```text
src/core/WowViewer.Core/
└── Maps/
    └── placement transaction/catalog extensions used by the editor contract
src/core/WowViewer.Core.PM4/
└── Reconciliation/
    ├── PM4 guide observation and candidate contracts
    ├── reviewed alignment/substitution/clone proposal service
    └── deterministic residual/evidence calculations
src/core/WowViewer.Core.IO/
└── Maps/
    ├── placement/name-table transaction integration
    └── atomic output preparation using existing writers
src/viewer/WoWViewer/
└── Editor plugin registration and PM4/Museum preview surface
tests/WowViewer.Core.Tests/
└── placement transaction, name-table, ID, and atomic-write tests
tests/WowViewer.Core.PM4.Tests/
└── reconciliation, scoring, ambiguity, transform, and provenance tests
```

**Structure Decision**: Keep pure guide/matching logic in `WowViewer.Core.PM4`, placement and format mutation in `WowViewer.Core`/`WowViewer.Core.IO`, and UI/bridge adapters in the editor host created by Specs 166–167. Do not add reconciliation state to `ViewerApp` or `WorldScene`; the plugin owns its transient preview state and the session owns undo/dirty state.

## Phase 0 — Discovery and contract lock

**Goal:** verify the dependency seams and real-data assumptions before adding a mutation path.

1. Audit the completed 166–175 contracts and identify the actual bridge/session/plugin registration types; record any dependency checkpoint that is not yet available.
2. Build a paired Museum/PM4 fixture catalog using `Pm4CoordinateService.TryGetObj0PathForPm4`, including one existing placement, one missing-object case, one ambiguous case, and one cross-tile case.
3. Verify the available ADT/WDT writer capabilities for rotation, scale, add, delete, name-table merge, and split/Alpha output; map each supported operation to its existing owner and record explicit refusals.
4. Calibrate the scorer/synthesizer signals against the paired fixture without changing thresholds; retain ranked candidates, breakdowns, and negative cases in the research evidence.
5. Lock the preview/apply/undo contract and source-fingerprint policy against the data model and OpenAPI shapes; stop if any action would require a second parser or writer.

**Checkpoint:** reviewed research note names supported eras, writer capabilities, fixture hashes, and all refusals; no code mutation is allowed until this passes.

## Phase 1 — Core reconciliation models and pure proposal engine

**Goal:** generate deterministic, side-effect-free proposals from PM4 observations and placement catalogs.

1. Add immutable PM4 guide, placement snapshot, evidence, candidate, residual, and proposal contracts from `data-model.md` in the owning core projects.
2. Add a reconciliation input adapter that consumes existing PM4 segment/match results and `AdtPlacementCatalog` records without reading files itself.
3. Implement compatible-kind/tile/bounds prefiltering and reuse the existing scorer's explicit status semantics; preserve all candidate breakdowns and rationale.
4. Implement align/substitute/clone proposal construction, including canonical coordinates, target tile validation, deterministic proposal IDs, and explicit unsupported/ambiguous results.
5. Add focused tests for transform parity, score/ambiguity preservation, evidence completeness, and no side effects during preview.

**Checkpoint:** core tests pass for positive, ambiguous, unresolved, unsupported, and stale-input cases; the proposal engine does not write files or depend on viewer types.

## Phase 2 — Placement transaction, ID allocation, and name-table integration

**Goal:** turn accepted proposals into validated, atomic placement edits using existing format owners.

1. Extend the placement edit data contract for rotation/scale/add/substitute/delete only where the target era writer can express the operation; keep unsupported operations as named refusals.
2. Implement deterministic non-colliding ID allocation across target MDDF/MODF rows and report every allocation/remap.
3. Implement model/WMO name-table merge and index remapping through the existing ADT representation and writer; do not add a serializer or alter `AlphaWdtWriter.cs`.
4. Validate all accepted decisions against source hashes, unique IDs, supported era, tile bounds, resolved asset paths, and target writer capabilities before producing output bytes.
5. Stage every target output in memory, then commit the complete output set and JSON provenance report atomically; preserve unchanged chunk bytes and refuse partial application.
6. Add focused tests for name-table collisions, ID exhaustion, multi-file induced failure, read-back, and source/output hash isolation.

**Checkpoint:** core/IO tests prove accepted edits round-trip, ambiguous/rejected edits do not mutate bytes, and an induced multi-target failure leaves every target unchanged.

## Phase 3 — Editor bridge and viewport preview

**Goal:** make the reviewed operation visible and usable in the viewer without coupling core to UI.

1. Register a reconciliation plugin through the existing editor host and declare supported build eras and required data sources.
2. Read live scene/selection/tile context through the 167 bridge only; do not add new `ViewerApp` placement staging fields.
3. Add PM4/Museum overlay rendering for current placement, guide geometry, proposed transform, status, and residual; keep the last committed scene visible while preview computation runs.
4. Add proposal filtering and per-proposal accept/reject controls, with ambiguous/unresolved items visibly disabled from automatic acceptance and an evidence detail view.
5. Submit accepted decisions as a single 168 session operation and refresh only affected scene regions; do not write during preview or on selection changes.
6. Add UI/source checks for plugin availability, fault containment, selection transitions, and no direct editor references from runtime/renderer types.

**Checkpoint:** the viewer can preview a real paired tile, visually show a corrected placement, and stage decisions without writing. The dependency bridge and editor-removal checks pass.

## Phase 4 — Save, reload, and provenance workflow

**Goal:** close the loop from reviewed proposal to independently readable output.

1. Add Validate and Apply commands using the core batch contract and show all refusals before writing.
2. Write the configured loose ADT/WDT outputs and sidecar report; show paths, hashes, remaps, allocations, and read-back results in the editor.
3. Reload the output in a fresh viewer session and compare every accepted placement and name-table index against the report.
4. Exercise undo/redo before unrelated output edits and refuse undo when output fingerprints no longer match the operation.
5. Add the real-data validation record with configured root, build/fingerprint, PM4/Museum hashes, independent-reader result, and user-owned visual result.

**Checkpoint:** the quickstart's alignment, missing clone, ambiguous refusal, save/reload, and undo gates are each evidenced; no runtime/visual claim is inferred from compilation alone.

## Phase 5 — Final quality gate and handoff

1. Run focused PM4 and core tests, then the solution Debug build and test command.
2. Re-check the constitution and dependency boundaries, especially no new serializer/parser and no game-install writes.
3. Review the operation report schema for client-byte leakage and deterministic evidence.
4. Update the spec status, `specs/STATUS.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md` only when implementation checkpoints actually pass.
5. Hand off the bounded user-owned real-client visual/independent-tool proof with exact PowerShell commands and configured build/root details.

## Risks and mitigations

- **PM4 identity remains imperfect:** proposals stay reviewable; confidence never grants mutation rights.
- **Writer cannot express a requested edit:** report unsupported and leave the source unchanged; do not approximate or reopen frozen writers without focused evidence.
- **Museum placement and PM4 are not one-to-one:** model proposals as align/substitute/clone separately, surface competing candidates, and preserve rejection history.
- **Large PM4 corpus stalls rendering:** compute off-thread, cache immutable observations, and update the viewport only from completed preview snapshots.
- **Stale preview overwrites another edit:** source hashes and placement unique IDs are checked again at validation/apply time.

## Post-design Constitution Check

**PASS.** The design adds no external dependency, parser, serializer, Blizzard-container output, client path assumption, or runtime-to-editor reference. Core owns the operation, IO owns format mutation, and the viewer remains an adapter. Real-client visual and independent-reader proof remains an explicit user-owned gate rather than an unsupported claim.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| None | N/A | The plan reuses existing core projects and writer owners. |
