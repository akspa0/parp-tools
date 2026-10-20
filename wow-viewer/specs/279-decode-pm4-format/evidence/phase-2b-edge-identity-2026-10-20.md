# Phase 2B — Raw +0x04 reciprocal-edge identity (2026-10-20)

## Question

Phase 1 rejected +0x04 as a general reverse-link pointer. Phase 2A showed +0x01 is a source-edge
ordinal. Does +0x04 carry a unique edge identity, or a value shared by reciprocal records?

## Method

[`Pm4EdgeIdentityProbe`](../../../src/core/WowViewer.Core.PM4/Research/Pm4EdgeIdentityProbe.cs)
resolves reverse records **independently of the tested value** (by resolved endpoints), then compares
the raw +0x04 values. It also measures per-file value reuse across distinct `(SourceSurface, Subtype)`
edges and a next-record control.

## Result

Corpus: 616 files, 1,273,335 links.

| Measure | Value |
|---|---|
| Unique reverse pairs | 1,271,518 |
| Multiple reverse links | 0 |
| No reverse link | 1,817 |
| Reused +0x04 values | 889,594 links across 885,143 distinct edges |

Reciprocal records sharing the same +0x04 value, by raw type:

| Raw type | Unique reverse | Same value | Different | Same % |
|---|---|---|---|---|
| 1 | 669,935 | 662,710 | 7,225 | **98.92%** |
| 2 | 450,390 | 211,009 | 239,381 | 46.85% |
| 4 | 54,417 | 817 | 53,600 | 1.50% |
| 10 | 64,026 | 46 | 63,980 | 0.07% |
| 12 | 23,292 | 0 | 23,292 | 0.00% |
| 17 | 3,954 | 3,954 | 0 | **100.00%** |
| 18 | 4,087 | 920 | 3,167 | 22.51% |
| 20 | 393 | 0 | 393 | 0.00% |
| 26 | 836 | 0 | 836 | 0.00% |
| 28 | 188 | 0 | 188 | 0.00% |

Next-record control: 36,326 same vs 1,234,894 different.

## Conclusion

+0x04 is **not a unique edge ID** (889,594 reused values). It is a **shared value on reciprocal
records of type 1 (98.92%) and type 17 (100%)**, and mostly not shared for types 4, 10, 12, 20, 26,
28. The meaning remains **U**: it behaves like a symmetric pair/edge attribute for some connection
kinds, not a global identity. Source: **E-EDGE2**.

## Files changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4EdgeIdentityProbe.cs`
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4EdgeIdentityProbeTests.cs`
- `wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/Program.cs` (`--edge-identity`)
- `wow-viewer/specs/279-decode-pm4-format/evidence/edge-identity-2026-10-20.json`

## Verification

```powershell
dotnet test "wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj" -c Debug --filter "FullyQualifiedName~Pm4EdgeIdentityProbeTests"
# exit 0

dotnet run --project "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --no-build -- --edge-identity "wow-viewer/test_data/development/World/Maps/development" "wow-viewer/specs/279-decode-pm4-format/evidence/edge-identity-2026-10-20.json"
# exit 0; 616 files, 1,273,335 links
```

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| +0x04 is not a unique edge ID | 889,594 reused values across 885,143 edges |
| Reciprocal type-1 records share the value | 662,710 / 669,935 = 98.92% |
| The sharing is type-specific, not universal | Types 4/10/12/20/26/28 ≤ 1.5% |
| Reverse records resolved independently | 0 multiple reverse links; 1,817 no-reverse retained |
