# PM4 structural foundations — whole-map measurements, 2026-10-20

## Result

**Real map-level progress, not another asset-match score:** all 616 PM4 inputs were read in bounded,
summary-only passes. The proposed surface/link windows are exact; the link reference is interpreted
in a **packed target tile**, not necessarily the file holding the link. Reciprocal geometry checks
and paired local-target controls strongly support that interpretation for this corpus.

This invalidates the earlier blanket assumption that every reference is file-local. It does not
establish full PM4 semantics, correct M2 partitions, a source-asset identifier, or improved matching
accuracy. Working readers, viewer and production matchers remain unchanged. Only owned research
code/instrumentation was added/updated. The broad decoded-data/Museum partition export remains
unrun; these small summary passes are a different, measured workload.

## Input identity and validation

- Corpus root: repository-relative `wow-viewer/test_data/original_development/World/Maps/development`.
- 616 PM4 files, 169.08 MiB; 114 zero-byte files; 309 files contain surfaces. The remaining files
  are included, not silently dropped. Exact client/build attribution is not established here.
- 518,092 surfaces, 1,273,335 links; 186,060 derived-zero-CK24 surfaces, all also raw-word zero in
  this corpus. Equality is measured here, not assumed for every build.
- Zero reader-diagnostic files, zero invalid polygon windows and zero invalid proposed link windows
  in the fingerprinted geometry pass.
- Whole-map SHA-256 over sorted `filename:sha256` lines:
  `d6460882861d2ffa0e13db323a8ddf9a8e52705ce8d3c23e74fa1e6a86b8d709`.
  Same-byte hashes and per-file identities are retained in both geometry/address reports.
  The existing window probe was a separate pass, not retrospectively claimed as atomic with these hashes.

## Measured / rejected / unresolved ledger

### 1. MSUR supplies two exact running windows in this corpus

| Candidate relation | Consecutive-window chain fits | Target overruns | Coverage |
|---|---:|---:|---:|
| MSVI start + index count → MSVI (positive control) | 517,783 / 517,783 | 0 | 100% |
| Raw MSUR +0x18 start + byte +0x02 → MSLK | 517,783 / 517,783 | 0 | 100% |
| Same +0x18/+0x02 window aimed at MSCN | Same chain arithmetic | 6,240 | 93.52% |
| +0x18 with index-count length (negative control) | 115,206 / 517,783 (22.25%) | 554 | 94.73% |
| +0x18 with raw group-byte length (negative control) | 43,805 / 517,783 (8.46%) | 1,769 | 96.15% |

The chain test is `start[i] + length[i] == start[i+1]`, not simply “the index is in range”. Sum
of proposed MSLK lengths equals all 1,273,335 link records. Therefore byte +0x02 **acts as the link
window count in this record relationship**. That is evidence stronger than its legacy `attrMask`
name; it does not imply every unknown byte is now understood or that it directly partitions M2s.

Reference-as-owner is strongly contradicted: only 18 of 1,273,335 entries point to the source
surface index under the old local reading. Surface-neighbor interpretation needs the tile address
below; the local lookup alone produced misleading self-links and unresolved references.

### 2. The packed field addresses the target tile

Measured address interpretation for this map:

```text
target tile X = MSLK raw +0x0C field & 0xFF
target tile Y = (MSLK raw +0x0C field >> 8) & 0xFF
target surface = addressed tile's MSUR[raw +0x10 ushort]
```

The current model calls the +0x0C field `LinkId` and +0x10 `RefIndex`. Those names alone were not
evidence. The high half-word and other flags are **not** semantically decoded by this measurement.

| Address reading | Links | Resolved | Missing addressed file | Reference outside present target | Shared boundary | Reciprocal pair |
|---|---:|---:|---:|---:|---:|---:|
| Correct byte order, same tile | 1,257,562 | 1,257,562 | 0 | 0 | 666,830 | **1,257,562** |
| Correct byte order, other tile | 15,773 | 14,659 | 1,114 | **0** | 7,190 | 13,956 |
| Source-file-only control | 1,273,335 | 1,268,782 | 0 | 4,553 | 666,861 | 1,257,578 |
| Swapped bytes, off-diagonal targets | 1,248,771 | 75,691 | 858,637 | 314,443 | 47 | 8,312 |

Same-coordinate/diagonal cases are not useful to discriminate byte order and are reported
separately in raw output. Correct-address same-tile links have **zero self-references** and 100%
pair reciprocity. “Reciprocal” means a B→A edge exists, not that every raw field is mirrored.

Among the 15,773 cross-tile links, raw type 0x01 supplies 7,754 links; 7,211 resolve and 7,190 both
share a boundary and have a back-link. The other path-bearing kinds have different geometric
behavior and weaker reciprocity; they are not relabeled as simple adjacency or asset classes.

### 3. Paired test excludes the “it also fits locally” coincidence

For **10,222 identical cross-tile records where both addressed and local target surfaces exist**:

| Metric | Addressed target | Same-file alternative |
|---|---:|---:|
| Shared boundary | **5,026** | 28 |
| Reciprocal pair | **9,748** | 15 |

The direct raw-type-0x01 subset is 5,042 paired records: 5,026 addressed boundaries versus 13 local;
5,029 addressed reciprocal pairs versus 4 local. This closes the selection-bias gap of looking only
at references that were already out of range locally.

Not all 11,220 locally-in-range/cross-addressed records are paired: 998 point at missing files.
The out-of-range-local subset was 4,553: all point to an 8-neighbor tile in the correct order;
4,437 target files are present and all their surface indices fit. The initial out-of-range-only
probe is supporting evidence, not the full-map address accuracy denominator.

### 4. Raw link kinds separate geometric roles, not established asset identities

Under the **old local-target reading**, which must not be mistaken for corrected map topology:

- Negative path-start links within zero CK24 share a full boundary at 98.93%; within the same
  nonzero key, 99.86%. Raw types 0x01 and 0x11 form this population.
- Nonnegative path-start records have a second geometry window; 587,229 have four path indices,
  11,024 have six, and 629 have five. These total all 598,882 path-bearing link records.
- For 596,536 locally resolved path-bearing records, path vertices touch the source surface in
  371,524 cases and referenced surface in 384,281. Referenced-surface contact falls to 580 for
  hashed wrong targets and 2,657 for offset-17 controls. Full path-edge contact is 167,030 for the
  referenced target versus 148/282 controls.

Thus shared-edge adjacency and links carrying additional geometry are observably different.
This does **not** identify the latter universally as walls, drops, portals, jumps, or per-object
boundaries. The local path-contact numbers need an addressed-target rerun before stronger meaning.
An exact endpoint-edge test misses collinear partial overlaps and differently tessellated edges.

## Provisional structural model (corpus-scoped)

```text
surface polygons ── index/count ──> MSVI ──> MSVT geometry
       │
       └─ raw +0x18 / +0x02 window ──> MSLK connection records
                                           │
                                           ├─ packed tile + surface index ──> neighbour surface, possibly another tile
                                           └─ some kinds: MSPI window ──> MSPV connection geometry
```

This is a **tiled surface-connectivity/connection-geometry layer**, not a demonstrated scene-object
graph. It explains why taking every link's connected component need not produce one placed M2:
navigation relationships can cross source geometry boundaries. The source-object partition remains
an independent question; group/type field aliases are not proof that such a partition is stored.

## Authored path-node layer: separate, still unassigned

The operator reports WoWEdit screenshots of pathing points placed for NPC movement and believes
these may be the only directly editor-visible pathfinding layer. Treat this as a separate authored
route/control-point hypothesis alongside the generated surface graph. The screenshots were not
available for fresh visual inspection in this pass. No assertion is made that MPRL, MSCN or MPRR
corresponds to that editor layer merely because of a legacy name.

Still unknown: which point stream represents authored nodes; how its references connect to the
surface graph; `_0x04`/subtype/remaining flag semantics; MPRR structure; whether any field preserves
source-object ownership for merged scaled M2s; why some target tiles are absent.

Next decisive experiment: audit raw +0x04, subtype and the point/reference chunks **against this
correctly addressed graph**, testing index-domain and geometric relations with wrong-domain
controls, then compare any proposed ownership partition against independently authored Museum
copies/scales. Do not equate a path node, surface or graph component with a placed M2 beforehand.

## Code and verification receipt

Changed this phase:

- New [local geometry/control probe](../../../src/core/WowViewer.Core.PM4/Research/Pm4LinkGeometryProbe.cs),
  [cross-tile address/control probe](../../../src/core/WowViewer.Core.PM4/Research/Pm4CrossTileLinkProbe.cs).
- [Research audit](../../../src/core/WowViewer.Core.PM4/Research/Pm4MapPartitionAudit.cs) and
  [models](../../../src/core/WowViewer.Core.PM4/Models/Pm4MapPartitionAuditModels.cs): retain packed
  target resolution and two addressed graph hypotheses **beside**, not instead of, local controls.
- [Summary runner project](structure-summary/StructureSummary.csproj) / [driver](structure-summary/Program.cs).
- [Geometry controls](../../../tests/WowViewer.Core.PM4.Tests/Pm4LinkGeometryProbeTests.cs),
  [address controls](../../../tests/WowViewer.Core.PM4.Tests/Pm4CrossTileLinkProbeTests.cs),
  [map-audit regression](../../../tests/WowViewer.Core.PM4.Tests/Pm4MapPartitionAuditTests.cs).
- Spec/plan/tasks and continuity docs; raw reports and summary logs linked below.

Exact successful commands from repo root, run through PowerShell with exit checks; all exited **0**:

```powershell
dotnet 'wow-viewer/tools/inspect/WowViewer.Tool.Inspect/bin/Debug/net10.0/WowViewer.Tool.Inspect.dll' pm4 msur-window --input 'wow-viewer/test_data/original_development/World/Maps/development' --output 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-msur-window-2026-10-20.json'
dotnet run --project 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/structure-summary/StructureSummary.csproj' -c Debug --no-restore -- 'wow-viewer/test_data/original_development/World/Maps/development' 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-path-contact-2026-10-20.json'
dotnet run --project 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/structure-summary/StructureSummary.csproj' -c Debug --no-restore -- --cross-tile 'wow-viewer/test_data/original_development/World/Maps/development' 'wow-viewer/specs/270-pm4-object-association-ledger-and-placement/evidence/map-cross-tile-paired-2026-10-20.json'
dotnet test 'wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj' -c Debug --no-restore --filter 'FullyQualifiedName~Pm4CrossTileLinkProbeTests|FullyQualifiedName~Pm4LinkGeometryProbeTests|FullyQualifiedName~Pm4MapPartitionAuditTests|FullyQualifiedName~Pm4ObjectPoolTests|FullyQualifiedName~Pm4M2SurfaceCorrelationTests|FullyQualifiedName~AdtPlacementAuditReaderTests' --logger 'console;verbosity=normal'
```

Stdout was captured with `2>&1 | Tee-Object -FilePath <owning-report.txt>` and each native exit code
checked. Elapsed analysis times: window 7.087 s; initial geometry 3.998 s; path-contact 4.865 s;
initial cross-address 5.232 s; paired cross-address 4.315 s. These are CPU summary measurements,
not viewer FPS/performance claims. The final focused regression run passed **32/32**.

Reports: [window](map-msur-window-2026-10-20.json), [geometry](map-link-geometry-2026-10-20.json),
[path/contact + raw exceptions](map-path-contact-2026-10-20.json), [initial address](map-cross-tile-links-2026-10-20.json),
[paired address](map-cross-tile-paired-2026-10-20.json).
Summaries: [geometry](map-structure-summary-2026-10-20.txt), [path/contact](map-path-contact-summary-2026-10-20.txt),
[out-of-range-only address test](map-external-address-hypothesis-2026-10-20.txt),
[full address](map-cross-tile-summary-2026-10-20.txt), [paired address](map-cross-tile-paired-summary-2026-10-20.txt),
[final tests](structure-foundation-tests-2026-10-20.txt).

One intermediate summary-runner build failed (exit 1) because its new cross-map branch lacked
the models namespace import; tests had passed and no data run occurred. Corrected the import;
successful cross-map/paired runs above followed. Existing dependency warnings remain.

| Criterion | Real evidence | Limit |
|---|---|---|
| L01 corpus/detector | All 616 input identities; positive/negative chain and geometry controls | No source-build attribution |
| L02 structural relationships | Exact window statistics plus whole-map addressed references | Corpus-scoped, not universal semantics |
| L03 geometry/context cross-check | Boundary and reciprocal cross-tile edges, paired same-link wrong-local targets | No new Museum asset-identity proof |
| L04 measured/refuted/unknown ledger | This receipt + updated continuity | Full format/point layer/object partition unresolved |

No training, GPU job, large raw export, map regeneration, visual/client proof or production matcher
change occurred. J06's large partition/Museum export remains distinct from these completed summaries.
