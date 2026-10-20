# Sub-step 2A receipt — source-edge ordinal and point-reference tests

Date: 2026-10-20. Operator continuation: keep investigating PM4 comprehensively. Scope was selected
in plan.md before the probe: raw +0x04 spatial candidates and an independent raw +0x01 edge test.

## Result

**MSLK byte +0x01 (legacy Subtype) is strongly supported as a zero-based source-polygon edge
ordinal in this corpus**, not just an unspecified layer/type. It is valid for all **1,273,335**
connection records, and geometrically selects the correct boundary far above rival ordinals.

Direct raw +0x04 point-index interpretations tested here show no useful boundary-contact signal.
This does not establish a different meaning or rule out untested transforms/indirections. No stream
is identified as the operator-reported authored editor path nodes. No M2 ownership claim follows.

Existing shared readers, viewer and production matching behavior remain unchanged. No large export,
training or GPU/client run occurred.

## Corpus and controls

Both real passes include all **616** PM4 files and 1,273,335 records, no tile sampling. Same-byte
input fingerprints match the established whole-map identity:

```text
d6460882861d2ffa0e13db323a8ddf9a8e52705ce8d3c23e74fa1e6a86b8d709
```

The addressed graph accounts for **1,114 link records pointing to absent files**, not 1,114
distinct missing tiles. Zero out-of-range addressed target-surface indices. Reader diagnostics
are a failure gate; no diagnostic input was silently accepted.

Spatial tests use 0.01-yard tight and 0.25-yard loose thresholds. Point-to-segment tests include
segment interiors, not just equal endpoints. Ordinal alternatives: zero-/one-based source/target
and shifted ordinal; invalid values are counted, never modulo-coerced into validity. The shift
control wraps only after a valid raw ordinal exists. Zero-length edges are excluded from matches.

## Measured source-edge relationship

The first pass counted ordinal validity only where a target exists; the follow-up counts source
ordinal validity even for absent targets. Those denominators are intentionally different.

| Hypothesis | Valid source/target ordinals (follow-up) | Matching other-polygon boundaries |
|---|---:|---:|
| Source zero-based | **1,273,335 / 1,273,335** | **674,013** |
| Source one-based | 946,326 | 96 |
| Source shifted by one (control) | 1,273,335 | 137 |
| Target zero-based | 1,168,949 | 167,939 |
| Target one-based | 917,731 | 110,613 |
| Target shifted by one (control) | 1,168,949 | 146,484 |

For raw type 0x01, **669,791 / 669,956** available addressed targets share the selected source
boundary. Raw type 0x11: **3,954 / 3,954**. These counts equal the full-edge geometry matches for
the direct-neighbor population from the preceding structural probe; the ordinal selects those
boundaries rather than merely pointing somewhere in the polygon.

Raw type 0x01 zero-bucket subset: 246,755 / 246,887; keyed subset: 423,036 / 423,069. Thus the
relationship is not only a keyed-WMO effect. This is a source boundary address, not an M2 partition.

## Path-bearing geometry corroboration

For all **598,882** records with auxiliary geometry windows:

| Selected source-edge test | Zero-based source | Shifted source control |
|---|---:|---:|
| Exact 3D auxiliary boundary edge | **143,325** | 71 |
| Both selected endpoints on auxiliary boundary | **155,083** | 317 |
| All auxiliary points in edge's vertical plane | **536,142** | 3,120 |

The vertical-plane test uses XY line distance only; it is **not** equal-height contact or proof
of traversal. Raw kind 0x04 has 54,662 / 54,724 coplanar windows; 0x0C has 23,385 / 23,411.
0x02 has 398,236 / 451,195; 0x0A has 55,050 / 64,048. Remaining cases and roles stay unresolved.

## Raw +0x04 direct point candidates

| Candidate interpretation | Available records | Source boundary at 0.01 yd | Both source/target boundaries |
|---|---:|---:|---:|
| Source MSCN[raw] | 1,273,335 | 26 | 1 |
| MSCN[raw+1] control | 1,273,335 | 24 | 3 |
| MSCN[raw+17] control | 1,273,157 | 99 | 32 |
| Deterministic shuffled MSCN control | 1,273,335 | 827 | 209 |
| Source MSVI[raw] → MSVT | 1,273,335 | 3,272 | 1,260 |
| MSVI[raw+17] control → MSVT | 1,273,277 | 3,175 | 1,288 |
| Source MSVT[raw] | 1,257,696 | 6,182 | 2,563 |
| Source MSPV[raw] | 1,250,416 | 1,598 | 671 |
| Source MSPI[raw] → MSPV | 1,268,534 | 1,268 | 562 |
| Addressed MSCN[raw] | 1,269,216 | 24 | 1 |
| Source MPRL[raw], raw XYZ | 280,709 | 0 | 0 |
| Source MPRL[raw], tested ZXY permutation | 280,709 | 79 | 10 |

These observations reject a general **boundary-point** interpretation using those particular
indices/frames. They do not prove MSCN or MPRL are not spatial data, do not identify +0x04, and do
not rule out another addressing/encoding. Candidate availability differs and remains explicit.

## Files changed

- [Shared spatial probe](../../../src/core/WowViewer.Core.PM4/Research/Pm4PointReferenceProbe.cs).
- [Five detector controls](../../../tests/WowViewer.Core.PM4.Tests/Pm4PointReferenceProbeTests.cs).
- [Thin runner project](point-reference-probe/PointReferenceProbe.csproj) / [driver](point-reference-probe/Program.cs).
- Plan/tasks and reference/unknown/continuity updates; this receipt and actual reports below.

## Commands and outcomes

Repository root, PS7; native exits explicitly checked. All commands below exited **0**:

```powershell
dotnet test 'wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj' -c Debug --no-restore --filter 'FullyQualifiedName~Pm4PointReferenceProbeTests' --logger 'console;verbosity=normal' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-tests-2026-10-20.txt'
dotnet run --project 'wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj' -c Debug -- 'wow-viewer/test_data/original_development/World/Maps/development' 'wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-2026-10-20.json' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-2026-10-20.txt'
dotnet run --project 'wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj' -c Debug --no-restore -- --edges-only 'wow-viewer/test_data/original_development/World/Maps/development' 'wow-viewer/specs/279-decode-pm4-format/evidence/edge-ordinal-geometry-2026-10-20.json' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/279-decode-pm4-format/evidence/edge-ordinal-geometry-2026-10-20.txt'
```

Initial full point pass: **20.682 s**, four initial detector tests passed. Focused follow-up:
**5.392 s**, final **five** detector tests passed, including differing-height/coplanarity control.
These are CPU summary durations, not viewer performance claims.

Reports: [point pass](point-reference-2026-10-20.json), [point summary](point-reference-summary-2026-10-20.txt),
[initial ordinal strata](subtype-ordinal-summary-2026-10-20.txt),
[edges/auxiliary follow-up](edge-ordinal-geometry-2026-10-20.json),
[follow-up summary](edge-ordinal-geometry-summary-2026-10-20.txt),
[final detector tests](point-reference-tests-2026-10-20.txt).

Checkpoint regression: **40/40 passed**, exit **0**, [actual output](checkpoint-2a-tests-2026-10-20.txt).
Exact command before the named-path Git checkpoint:

```powershell
dotnet test 'wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj' -c Debug --no-restore --filter 'FullyQualifiedName~Pm4PointReferenceProbeTests|FullyQualifiedName~Pm4FieldDomainProbeTests|FullyQualifiedName~Pm4CrossTileLinkProbeTests|FullyQualifiedName~Pm4LinkGeometryProbeTests|FullyQualifiedName~Pm4MapPartitionAuditTests|FullyQualifiedName~Pm4ObjectPoolTests|FullyQualifiedName~Pm4M2SurfaceCorrelationTests|FullyQualifiedName~AdtPlacementAuditReaderTests' --logger 'console;verbosity=normal' 2>&1 | Tee-Object -FilePath 'wow-viewer/specs/279-decode-pm4-format/evidence/checkpoint-2a-tests-2026-10-20.txt'
```

The operator explicitly requested findings be committed while work continues. The index was empty
before staging; only enumerated PM4 artifact/source/doc paths are selected. The unrelated root
UI-state file remains excluded. No broad staging, reset or source-reader rewrite is part of this checkpoint.

## Criterion → evidence / gate

| Sub-step criterion | Evidence | Proof limit |
|---|---|---|
| T020 bounded scope/controls | plan.md selected 2A and conditional focused follow-up | No whole-format scope expansion |
| T021 spatial alternatives | All-map same-byte reports and true/wrong stream/index controls | Raw +0x04 meaning remains unknown |
| Source ordinal test | 100% validity and 674,013 vs 137 shared boundaries | Corpus-scoped; no source-object ownership |
| Auxiliary corroboration | 536,142 vs 3,120 vertical-plane cases with distinct 3D metrics | Does not label drops/jumps/portals or traversability |
| T024 per-question evidence | This receipt, exact commands and real output | MPRR/editor screenshots and overall Gate R2 remain open |

**Sub-step 2A complete.** Next concern can be selected separately; this is not certification of the
whole point/reference layer. No visual/client proof, matching accuracy gain or global semantic rename.
