# Whole-map partition instrument and explicit candidate pools — 2026-10-20

## Result and proof boundary

**Later same-day correction:** the initial instrument phase below preceded the
[whole-map structural measurements](structural-foundations-2026-10-20.md). Short summary-only
passes have now read all 616 PM4s and validated packed target-tile references; file-local references
must be treated as controls, not universal addressing. The research audit now has twelve hypotheses
including addressed graphs. The large raw/Museum partition export remains unrun. Preserve this
receipt as the earlier instrument delivery, not the latest semantic status.

Prepared the operator-directed map-wide zero-bucket investigation instrument and persistent tile
object-pool tooling. **25 focused tests pass; both thin utilities build.** A contained CLI smoke
restricted two synthetic candidates to the selected one and imported 70 real WL-family filenames
(11 recognized aliases), without generating tile assignments. Existing WMO matching, shared
PM4/ADT/WL readers, viewer code and original map files are unchanged. The new strict ADT evidence
wrapper is opt-in and layers above the existing reader.

**No whole-map corpus run occurred. No unknown field was decoded, no correct M2 partition was
established, and no matching-accuracy gain was measured.** The operator-reported ~70% WMO baseline
is preserved, not re-measured. Viewer annotation controls and automatic WL footprint associations
remain open. [Exact operator run and usage](map-context-usage.md).

## Source findings (not format conclusions)

- [Segment builder:42](../../../src/core/WowViewer.Core.PM4/Matching/Pm4ObjectSegmentBuilder.cs#L42)
  reduces referenced-surface type metadata to the first encountered link, losing competing types.
- [Segment builder:59](../../../src/core/WowViewer.Core.PM4/Matching/Pm4ObjectSegmentBuilder.cs#L59)
  seeds zero geometry by raw group/attrMask equality before full relationships are considered.
- [Position-reference collection:267](../../../src/core/WowViewer.Core.PM4/Matching/Pm4ObjectSegmentBuilder.cs#L267)
  reuses a reference number for both surface membership and MPRL indexing when in range. A range
  fit is not evidence that the two reference domains mean the same thing.
- [Adjacency analyzer:94](../../../src/core/WowViewer.Core.PM4/Research/Pm4AdjacencyComponentAnalyzer.cs#L94)
  resets union-find per document and aggregates summaries. That code cannot establish the
  absence of cross-file context in the format.
- [New surface extractor:10](../../../src/core/WowViewer.Core.PM4/Services/Pm4SurfaceCorrelationExtractor.cs#L10)
  receives surface geometry/labels only, and its fingerprint discards link relationships.

These justify measuring lost context rather than declaring M2 data intrinsically unmatchable.
Old comments equating raw fields with height, IDs, types, counts or scene indices remain hypotheses.

## Instrument behavior

Ten explicit partition hypotheses cover global rounded geometry, all/anchor/path/reciprocal
window-reference edges, cross-file geometric support, raw group values under local/map scope, and
raw attrMask/group seeding baselines. All surfaces remain in the graph, including keyed context.
Derived CK24 zero and raw-word zero are counted separately. Every link retains its raw record,
owner list and reference-domain fit; orphan, invalid and anchor-only records survive.

Raw MSLK fields and MSUR attrMask/group/padding/window/raw-word incidence are exported for further
cuts, including attrMask bits. Rival window lengths test the current ownership interpretation.
Normalized candidate shape families keep membership, absolute RMS radius and map centroid. They
do not promise to separate multiple copies merged into one connected pathfinding mesh. Every
decoded chunk and original PM4 chunk payload is retained by the runner for subsequent analysis.

Museum ingestion retains every authored catalog/placement, including unpaired tiles and repeated
references, with named-asset/scale families as independent source evidence. It does not assign
individual PM4 fragments to those placements by nearest origin.

Candidate pools are a separate operator expectation layer. Explicit selectors constrain eligible
records; design-kit/class labels are metadata. No score is altered. No annotation preserves the
exact original database; empty assigned pools do not fall back to unrelated kits. The CLI emits a
normal filtered database and an inspectable sidecar for existing matching tooling.

## Files added/changed

- Core.PM4: [audit models](../../../src/core/WowViewer.Core.PM4/Models/Pm4MapPartitionAuditModels.cs),
  [map audit](../../../src/core/WowViewer.Core.PM4/Research/Pm4MapPartitionAudit.cs),
  [shape diagnostic](../../../src/core/WowViewer.Core.PM4/Research/Pm4MapPartitionShape.cs).
- Core.PM4: [pool models](../../../src/core/WowViewer.Core.PM4/Models/Pm4ObjectPoolModels.cs),
  [filter](../../../src/core/WowViewer.Core.PM4/Matching/Pm4ObjectPoolFilter.cs),
  [manifest store](../../../src/core/WowViewer.Core.PM4/Services/Pm4ObjectPoolManifestStore.cs),
  [filename hints](../../../src/core/WowViewer.Core.PM4/Services/Pm4DesignKitHints.cs).
- Core.IO: [opt-in evidence wrapper](../../../src/core/WowViewer.Core.IO/Maps/AdtPlacementAuditReader.cs).
- Tests: [map audit](../../../tests/WowViewer.Core.PM4.Tests/Pm4MapPartitionAuditTests.cs),
  [pool/hint controls](../../../tests/WowViewer.Core.PM4.Tests/Pm4ObjectPoolTests.cs),
  [placement evidence](../../../tests/WowViewer.Core.PM4.Tests/AdtPlacementAuditReaderTests.cs).
- Evidence runners: [map project](map-partition-audit/MapPartitionAudit.csproj) / [driver](map-partition-audit/Program.cs),
  [pool project](object-pools/ObjectPools.csproj) / [driver](object-pools/Program.cs).
- This spec's requirements/plan/tasks/status, usage, raw logs and smoke manifest; relevant
  active-context/progress/workstream handoff entries.

## Exact verification commands and outcomes

Run from repository root through PowerShell; each native command's exit status was checked.

```powershell
dotnet test 'wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj' -c Debug --no-restore --filter 'FullyQualifiedName~Pm4MapPartitionAuditTests|FullyQualifiedName~Pm4ObjectPoolTests|FullyQualifiedName~Pm4M2SurfaceCorrelationTests|FullyQualifiedName~AdtPlacementAuditReaderTests' --logger 'console;verbosity=normal' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-context-and-pools-tests-2026-10-20.txt'
dotnet build 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/object-pools/ObjectPools.csproj' -c Debug --no-restore 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/object-pools-build-2026-10-20.txt'
dotnet build 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-partition-audit/MapPartitionAudit.csproj' -c Debug --no-restore 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-partition-build-2026-10-20.txt'
```

Exit statuses: **0 / 0 / 0**. [Tests: 25 passed](map-context-and-pools-tests-2026-10-20.txt),
[pool build: 0 errors](object-pools-build-2026-10-20.txt), [map build: 0 errors](map-partition-build-2026-10-20.txt).
Existing dependency vulnerability warnings remain; dependencies/readers were not modified to hide them.
Earlier instrument-only run passed eight tests; combined pre-review run passed 22; final 25 includes
strict placement-evidence controls. These are not three independent map-validation results.

CLI smoke calls (each exit 0; fixture directory created only if absent):

```powershell
$out = 'wow-viewer/output/tmp/pm4-object-pool-smoke-20261020'
$tool = 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/object-pools/bin/Debug/net10.0/ObjectPools.dll'
$manifest = Join-Path $out 'manifest.json'
dotnet $tool init $manifest 'synthetic-context-smoke'
dotnet $tool pool $manifest 'synthetic-pool' 'synthetic-kit' 'props' 'm2' 'world/synthetic-kit'
dotnet $tool assign $manifest 'synthetic-pool' 10 20 10 20 'Synthetic control, not a real development tile assignment.'
dotnet $tool hints $manifest 'wow-viewer/test_data/original_development/World/Maps/development'
dotnet $tool filter $manifest (Join-Path $out 'input-db.json') 'synthetic-context-smoke' 10 20 (Join-Path $out 'filtered-db.json')
```

The generated fixture has two M2 records with identical one-bin histograms, paths
`world/synthetic-kit/prop.m2` and `world/unrelated-kit/prop.m2`. The retained manifest explicitly
names a **synthetic map**, not a real development tile allocation.
[CLI stdout](object-pool-smoke-2026-10-20.txt) records 70 names/11 recognized and `candidates 2 -> 1`.
[Persisted manifest](object-pool-smoke-manifest-2026-10-20.json) is real file-backed annotation output;
only source filenames are real corpus evidence, not inferred zone footprints or confirmed assets.

## Criterion → evidence

| Task | Evidence | Remaining limit |
|---|---|---|
| J01 source audit | Code references above; independent read-only review inspected new code | No format-semantic conclusion from comments |
| J02 provenance/partitions | Synthetic multi-file/local-index, orphan/overlap/anchor/keyed-context tests pass | Whole-map data not run |
| J03 scale-family diagnostic | Translated scaled copies co-family; wrong aspect excluded; absolute scale retained | Simple-shape/clipping ambiguity remains |
| J04 runner | Built thin host; all-file enumeration, complete decoded export, same-byte hashes, conservative completeness gate | Operator runtime/output proof owed under J06 |
| J05 controls/build | Final 25 passing focused tests and two successful builds | Not visual, FPS or real-client proof |
| J06 real map analysis | **Not run; remains unchecked** | Operator owns broad pass |
| J07 continuity | Updated spec/router, usage command, memory-bank dashboard/history | Real analysis begins after operator output |
| K01 persistence | Unit round-trip and CLI-produced manifest | No viewer controls |
| K02 candidate filtering | Unit isolation/empty/union controls; CLI 2→1 with identical geometry bins | No measured match-accuracy gain |
| K03 tool output | Persisted filtered DB/context in project output | Must explicitly use tile-scoped DB in matcher |
| K04 validation/handoff | Logs/commands and limits above | Viewer UI remains K05 |
| K06 WL hints | 70 filenames retained, 11 recognized; ambiguity/custom alias/confirmation controls pass | Exact tile association not established |

## Review corrections and disclosure

New-code review found: trailing-separator output-boundary bypass, diagnostic-only partial parses,
separate fingerprint/parse reads, unresolved/truncated authored records, read failures outside the
manifest handler, hint-confirmation loss on re-import, and output-sidecar partial publication.
Corrected in the new wrappers/hosts and preserved confirmed hints; shared readers remain frozen.
The map runner conservatively blocks completion on any reader diagnostic, without calling all
diagnostics bugs. Final tests/builds ran after these corrections.

The earlier documentation-only Git check was declined when the operator redirected the work;
it was not rerun as that declined action and is not claimed as final verification for this phase.
No full-map runtime, installed-client, viewer UI or performance proof was performed.
