# Phase 3 — M2 per-object partition (negative result, 2026-10-20)

## Question

The M2/WMO partition is solved (`Ck24Type == 0` / `GroupKey == 3`). Can the M2 bucket be split into
individual placed objects by a decoded structural feature? The operator reports that M2 surfaces are
merged into a mesh containing one or many scaled/unscaled copies.

## Method

[`Pm4M2PartitionProbe`](../../../src/core/WowViewer.Core.PM4/Research/Pm4M2PartitionProbe.cs) takes
the M2 surfaces of each tile and tests, against the authored M2 origins:

1. Topological components (surfaces sharing an MSVT vertex).
2. Per-placement MSVT vertex sets (disjoint vs shared).
3. Per-placement MSUR index ranges (disjoint vs overlapping).
4. `MSUR.Padding` (+0x03), `MSUR._0x18`, Normal and Height.
5. `MSLK.GroupObjectId` (+0x04) from each surface's MSLK window.

## Result

Corpus: 170 tiles with M2 surfaces, 112,818 M2 surfaces, 59,327 M2 placements.

| Test | Result |
|---|---|
| Topological components | 22,286 |
| Components covering 0 / 1 / 2+ origins | 14,777 / 4,775 / 2,734 |
| Multi-origin components, same model | 734 |
| Multi-origin components, mixed models | **2,000** |
| Largest component | 1,023 surfaces, 133 origins, 7 model paths |
| Placements with disjoint MSVT vertex sets | 4,289 |
| Placements sharing MSVT vertices | **6,576** |
| Placements with disjoint MSUR index ranges | 1,295 |
| Placements with overlapping MSUR index ranges | **9,570** |
| `MSUR.Padding` (+0x03) | **0 for all 112,818 surfaces** |
| Placements with a single `GroupObjectId` | 1,895 |
| Placements with multiple `GroupObjectId`s | **7,315** |
| `GroupObjectId`s mapping to one model | 21,463 |
| `GroupObjectId`s mapping to multiple models | 4,739 |

## Conclusion

The merge is genuine. No decoded structural feature partitions the M2 bucket into per-object pieces:
surface order, vertex ranges and connection IDs all interleave objects. Per-object M2 ownership is
**U**; it must come from an undecoded field or is lossy in this corpus. This is a negative result and
does not change the working WMO baseline.

## Files changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4M2PartitionProbe.cs` (new)
- `wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/Program.cs` (`--m2-partition`)
- `wow-viewer/docs/research/pm4-format-reference.md` (§3.6, E-M2PART)
- `wow-viewer/specs/279-decode-pm4-format/evidence/m2-partition-v6-2026-10-20.json` (report)

## Verification

```powershell
dotnet build "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --nologo -v q
# exit 0, 0 errors

dotnet run --project "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --no-build -- --m2-partition "wow-viewer/test_data/development/World/Maps/development" "wow-viewer/specs/279-decode-pm4-format/evidence/m2-partition-v6-2026-10-20.json"
# exit 0; 170 files with M2, 112,818 surfaces, 22,286 components, 36.513s
```

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| Topology does not partition M2 | 2,000 mixed-model components; largest covers 133 origins / 7 models |
| Vertex ranges do not partition M2 | 6,576 placements share vertices vs 4,289 disjoint |
| Surface order does not partition M2 | 9,570 overlapping MSUR ranges vs 1,295 disjoint |
| Padding is not a partition key | 0 for all 112,818 M2 surfaces |
| GroupObjectId is not a per-object key | 7,315 placements multi-valued; 4,739 IDs map to multiple models |
