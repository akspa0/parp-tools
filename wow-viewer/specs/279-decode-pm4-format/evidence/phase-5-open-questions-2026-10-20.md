# Phase 5 — Open-question sweep (2026-10-20)

> **Follow-up (same day):** `Pm4OpenQuestionFollowupProbe` resolved the plane frame and the MPRR
> node bounds, and improved the placement-Z match. Results: `_0x10 = -dot(normal, v0)` in raw MSVT
> space (518,080 / 518,092, median error 0); MPRR first two records are ordered spatial bounds in
> 3,171,133 / 3,171,912 runs (99.98%); `_0x1c` matches a placement Z anywhere in the map for
> 231,383 / 332,032 (69.7%); MSHD spans vs raw placement spans r = 0.39–0.58; `MPRL._0x14` is not
> the run index (20,834) or the node level (53,007). Output:
> `open-questions-2b-2026-10-20.json`.

## Question

Resolve as many of the reference's open questions (Q01–Q20) as the corpus supports, using
corpus-wide tests with controls, and record the answers in the reference.

## Method

- New `Pm4FieldSemanticsProbe` (field-level tests) and `Pm4SpatialSemanticsProbe` (spatial tests).
- New runner mode `--open-questions` in `point-reference-probe`.
- Corpus: `wow-viewer/test_data/original_development/World/Maps/development` — 616 PM4 files
  (502 structured), 411 companion `_obj0.adt` files.
- Output: `open-questions-2026-10-20.json` (11,472 lines).

## Results

| Question | Result | Status |
|---|---|---|
| Q01 `MVER` | 502 files at 12304; 114 zero-byte files at 0 | answered |
| Q02 `MSHD` spans | `_0x00`/`_0x08` vs point-stream spans r = 0.13–0.26; vs placement spans r = 0.09–0.18 | open |
| Q03 `MSUR._0x00` | 7 classes: 3 = 184,356 surfaces, all `_0x1c == 0`, mean area 45.9; 16 = 87,243 (mean Z 97.1); 17 = 13,493 (83.9); 18 = 161,359 (81.1); 19 = 71,293 (114.0); 20 = 105; 21 = 243 | answered |
| Q04 normal | 506,626 / 518,092 match the polygon normal within 5°, median angle 0° | answered |
| Q04 plane distance | 9 / 518,092 match `dot(normal, v0)`; median error 7,318 | open |
| Q05 `_0x1c` | 332,032 nonzero; 196,016 (59.0%) match a placement Z in the same file | partial |
| Q06 `MSLK._0x04` | 5,155 groups one model, 20,543 multi-model, 3,466 none | answered: finer group, not a model id |
| Q07 `MSLK._0x00` | bit 0 = no geometry (flags 1, 17); bit 4 = local/same-tile (flags 17/18/20/26/28 have 0 cross-tile); low family 1/2/4/10/12 is cross-tile | partial |
| Q08 `MSLK._0x0c` upper half | 1,273,335 / 1,273,335 are `0xFFFF` | answered |
| Q09 path kinds | 4-index 587,229 (99.86% coplanar); 5-index 629 (63.8%); 6-index 11,024 (83.8%); 0 Z-dominant in all kinds | answered: all are vertical wall quads |
| Q10 `MSCN` contour | 378,614 closer to boundary vs 281,007 closer to interior; median boundary 0.304, interior 0.554 | partial: supported, not decisive |
| Q11 `MPRL._0x14` | 23,976 match the group index, 18,724 the reverse index, 135,888 neither | open |
| Q12 `MPRR` | 3,171,912 runs; first two = min/max of the middle in only 1,186 (0.04%); middle values are valid MPRL indices in 1,966,930 (62%) | partial: bounds encoding is not min/max |
| Q13/Q14 destructible | MDBH 502, MDBI 10, MDBF 30, MDOS 526, MDSF 2,684; all MDSF indices in range | answered: sparse but consistent |
| Q16 unattributed groups | 17,965 groups: 2,478 one model, 14,053 multi-model, 1,434 none | answered: mostly multi-model |
| Q17 chunk presence | 14 core chunks in 502 files; MDBI/MDBF in 1 file | answered |
| Q18 tall surfaces | 32,567 tall, 9,789 non-Z-dominant (30.1%); 485,525 short, 55,700 non-Z-dominant (11.5%) | answered: 2.6× enrichment, 70% still Z-dominant |
| Q19 no-geometry `MSLK` | 674,453 records; 673,910 (99.9%) resolve to a present target surface; 543 absent | answered |
| Q20 `MSUR` as terrain source | 1,930,146 vertices; 180,588 (9.4%) within 1.0 of an `MPRL` sample; median height error 1.08 | partial: sparse |

## Files changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4FieldSemanticsProbe.cs` — new.
- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4SpatialSemanticsProbe.cs` — new.
- `wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/Program.cs` — `--open-questions`.
- `wow-viewer/specs/279-decode-pm4-format/evidence/open-questions-2026-10-20.json` — output.
- `wow-viewer/docs/research/pm4-format-reference.md` — §4.7 correction and §4.13 results table.

## Verification

```powershell
dotnet build wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj -c Debug
dotnet run --project wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj -c Debug -- --open-questions "I:\parp\parp-tools\wow-viewer\test_data\original_development\World\Maps\development" "I:\parp\parp-tools\wow-viewer\specs\279-decode-pm4-format\evidence\open-questions-2026-10-20.json"
```

Exit 0. Console: `Open-questions probe: 616 files, normal match 506626/518092, plane match 9, groups one/multi/none = 5155/20543/3466, MSCN boundary/interior = 378614/281007, 14.001s.`

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| Every question is tested on the whole corpus | `open-questions-2026-10-20.json`, 616 files |
| Answers are counts, not identity claims | JSON `Field` and `Spatial` sections |
| Reference reflects the results | §4.13 table; §4.7 bounds correction |
| No regression in the reader | probe build and run exit 0 |

## Limits

- Q02, Q04 plane distance, Q11, and Q15 remain open.
- Q05, Q10, Q12, and Q20 are partial.
- The `MPRR` bounds encoding is not min/max; the quadtree reading is supported by node sizes and
  run quantisation, not by the first two records.
- No runtime/visual proof is claimed.
