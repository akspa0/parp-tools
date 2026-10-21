# Tasks: Offline WoWMuseum ADT Patcher

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)

**Branch**: `281-museum-adt-patcher`

**Rule**: a task is checked `[x]` only with a receipt (AGENTS §9.2): files changed, exact commands
and exit status, and a criterion→evidence table. Build/test output alone never proves runtime,
format, or determinism behavior.

## P0 — Skeleton and path guard

- [ ] T001 Create `tools/museum-patch/WowViewer.Tool.MuseumPatch/` and register it in
  `WowViewer.slnx`.
- [ ] T002 Implement argument parsing for `--museum`, `--pm4`, `--out`, `--map`, `--dry-run`,
  `--review`, `--overwrite`, `--min-score`, `--jobs`, `--verbose`.
- [ ] T003 Implement `PathGuard`: refuse an output folder that is an input folder; refuse to write
  outside `--out`; refuse to overwrite an input.
- [ ] T004 Return exit code `2` for invalid arguments or unsafe paths.
- [ ] T005 Build the solution with 0 errors.

**Gate P0**: the tool builds; unsafe paths are refused with exit code `2`. Receipt:
`evidence/p0-skeleton.md`.

## P1 — Discovery and pairing

- [ ] T010 Enumerate Museum ADTs (`<map>_<y>_<x>.adt` and `<map>_<y>_<x>_obj0.adt`).
- [ ] T011 Enumerate PM4/PD4 files (`<map>_<y>_<x>.pm4` / `.pd4`).
- [ ] T012 Pair by `(tileX, tileY)` using `Pm4CoordinateService.TryParseTileCoordinates`.
- [ ] T013 Report paired and unpaired tiles before writing anything.
- [ ] T014 Add unit tests for pairing, including a tile with only one input.

**Gate P1**: a real folder pair reports the correct paired/unpaired counts. Receipt:
`evidence/p1-discovery.md`.

## P2 — Dry-run proposals

- [ ] T020 Run the core pipeline per tile: `Pm4ObjectSegmentBuilder` → `AdtPlacementReader` →
  `Pm4AssetMatchScorer` → `Pm4ReconciliationInputAdapter` → `Pm4ReconciliationEngine`.
- [ ] T021 Sort proposals by a stable key.
- [ ] T022 Write `review.json` with every proposal, its source, score, status, and default
  decision.
- [ ] T023 Write no ADTs in dry-run mode.
- [ ] T024 Add unit tests for dry-run output shape and zero ADT writes.

**Gate P2**: a real dry run writes a review file and zero ADTs. Receipt: `evidence/p2-dry-run.md`.

## P3 — Apply and LK output

- [ ] T030 Read `review.json` and map decisions by proposal id.
- [ ] T031 Treat undecided proposals as rejected and report them.
- [ ] T032 Apply accepted proposals with `ReconciliationApplyService.ApplyAccepted`.
- [ ] T033 Write a monolithic 3.3.5 LK v18 ADT with `LkAdtWriter.Write`.
- [ ] T034 Refuse to overwrite an existing output unless `--overwrite` is passed.
- [ ] T035 Add unit tests for accept/reject/undecided handling.

**Gate P3**: a real apply writes one ADT per accepted tile and zero for rejected tiles; the output
opens in the 3.3.5 adapter. Receipt: `evidence/p3-apply.md`.

## P4 — Provenance and determinism

- [ ] T040 Write `<map>_<y>_<x>.provenance.json` with input hashes, every proposal, and the output
  hash.
- [ ] T041 Write `run-summary.json` with paired, unpaired, accepted, rejected, conflicts, failures.
- [ ] T042 Prove two runs on the same inputs produce identical output hashes.
- [ ] T043 Add a determinism test that runs the pipeline twice and compares hashes.

**Gate P4**: two runs produce identical hashes; every proposal has a source, score, decision, and
output hash. Receipt: `evidence/p4-provenance.md`.

## P5 — Viewer removal

- [ ] T050 Remove the PM4 Correlation, Reconcile, and Alignment tabs from the PM4 workbench.
- [ ] T051 Remove the match-curation section from the PM4 Selection tab.
- [ ] T052 Delete `Pm4ObjectMatchStore.cs`.
- [ ] T053 Decide `Pm4WmoGroupMatchService.cs`: keep only if the viewer still needs shape search;
  otherwise delete.
- [ ] T054 Keep PM4 Overlay, Selection, Info, Outliner, and Object Groups working.
- [ ] T055 Build and test the viewer with 0 errors.

**Gate P5**: the viewer has no matching/alignment/reconciliation tabs; PM4 inspection still works.
Receipt: `evidence/p5-viewer-removal.md`.

## P6 — Documentation and release guard

- [ ] T060 Document the tool in `tools/README.md` and `docs/CLI-TOOLS.md`, including the no-data
  constraint.
- [ ] T061 Add a release check that fails if the archive contains any `.adt`, `.pm4`, or `.pd4`.
- [ ] T062 Add a repository check that fails if a data file is committed under the tool folder.
- [ ] T063 Record the operator-facing usage example.

**Gate P6**: the release check passes on a clean archive and fails on a planted data file. Receipt:
`evidence/p6-docs-release.md`.

## Final verification

- [ ] T070 `dotnet build wow-viewer/WowViewer.slnx -c Debug` — 0 errors.
- [ ] T071 `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName~MuseumPatch"` — all pass.
- [ ] T072 Operator dry run on a real Museum ADT folder and PM4 folder.
- [ ] T073 Operator apply on the same inputs and confirm the output ADT opens in 3.3.5 tooling.
- [ ] T074 Operator confirms the release archive contains no data files.
- [ ] T075 Update [spec.md](spec.md) status to `Implemented` and record the final receipt.

**Gate Final**: T070–T075 pass; the receipt is `evidence/final-receipt.md`.
