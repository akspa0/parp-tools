# Phase 3 — MPRL/MPRR authored path-node layer (2026-10-20)

## Question

The operator reports WoWEdit screenshots of pathing nodes laid down for NPCs. Which chunk holds
them, and what is MPRR?

## Method

[`Pm4PathNodeProbe`](../../../src/core/WowViewer.Core.PM4/Research/Pm4PathNodeProbe.cs) measures:

1. MPRL point containment in the PM4 mesh (ADT placement space).
2. MPRL raw-field distributions.
3. MPRR value ranges against MPRL and MSCN counts.
4. MPRR byte decomposition and run lengths between `0xFFFF` sentinels.
5. A raw record dump (`--mprr-dump`) of `development_00_00.pm4`.

## Result

Corpus: 616 files.

| Measure | Value |
|---|---|
| MPRL points | 178,588 |
| MPRL inside PM4 mesh | 177,525 (99.4%) |
| MPRR records | 17,150,640 |
| MPRR sentinels (`Value1 == 0xFFFF`) | 3,171,410 (18.5%) |
| MPRR `Value1` in MPRL range | 6,778,712 (39.5%) |
| MPRR `Value2` in MPRL range | 10,695,894 (62.4%) |

MPRL raw fields:

| Field | Observation |
|---|---|
| `Unk02` | **−1 for all 178,588 points** |
| `Unk06` | **32768 (0x8000) for all 178,588 points** |
| `Unk00` | varied (10032, 10033, 13073, …) — candidate node/group id |
| `Unk04` | varied (34, 173, 35, …) — candidate connection id |
| `Unk14` | −1, 0, 1, 2, 3, 4, 5, 6 — small type/flag |
| `Unk16` | 0 (129,701) or 16383 (0x3FFF, 48,887) |

MPRR run lengths between sentinels are **all `4k+3`**: 3 (2,394,347 runs), 7 (585,290), 11 (146,070),
15 (32,366), 19 (5,675), 23 (1,451), 27 (1,360), 31 (788), 35 (686). The ~4:1 decay is a hierarchical
signature.

Raw dump of `development_00_00.pm4` (81,936 MPRR records, 2,493 MPRL, 9,990 MSCN):

```text
[0] v1=0 v2=768 (0x0300)
[1] SENTINEL run=1
[2] v1=0 v2=0
[3] v1=31 v2=31
[4] v1=0 v2=768
[5] SENTINEL run=3
[6] v1=32 v2=0
[7] v1=63 v2=31
[8] v1=0 v2=768
[9] SENTINEL run=3
[10] v1=64 v2=0
[11] v1=95 v2=31
[12..55] v1=256,257,258,283,308,... v2=0
[56] v1=0 v2=768
[57] SENTINEL run=47
```

## Interpretation

- **MPRL is the authored point layer (H):** 99.4% of points lie inside the PM4 mesh, and the
  coordinate mapping already matches terrain height at 89.38%. `Unk02`/`Unk06` are constant, so the
  varying `Unk00`/`Unk04`/`Unk14`/`Unk16` are the per-node payload.
- **MPRR is a spatial grid, not a point graph (M):** only 39.5%/62.4% of its values fall in MPRL
  range, so it is not an edge list over MPRL. The values are linear cell indices (0, 31, 32, 63, 64,
  95, 256, 257, …), each run ends with `(0, 0x0300)`, and the `4k+3` run lengths with 4:1 decay are
  a hierarchical/quadtree signature (H). The exact cell width and node layout are **U**.

## Follow-up: "range records" tests (2026-10-20)

The operator described MPRR as **range records**. The record stride is exactly 4 bytes
(payload/record = 4.000). Candidate range interpretations were tested across the whole map:

| Test | Result |
|---|---|
| First two records form a valid `min <= max` box | 3,171,612 / 3,171,635 (99.999%) |
| First two records form a 32×32 chunk | 75,101 (2.4%) — **refuted** |
| First two records form a valid MSUR range | 2,314,976 (73.0%) |
| MSUR range widths sum vs total MSUR count | 6,823,601 vs 308,892 — **22× overlap, refuted** |
| Middle records are all valid MPRL indices | 1,967,408 (62.0%) |
| Middle records all have `Value2 == 0` | 479 (0.015%) — **refuted** |

The byte decomposition is the strongest remaining signal: `Value1`'s high byte is a small
row/type (0–6) and its low byte is a bitmask-like value (0, 255, 31, 224, 32, 192, 128, 64).
`Value2`'s high byte takes 0, 1, 3, 5, 17, 9, 41, 25 — the same strata as `MSLK.TypeFlags`.
This is consistent with a **run-length encoded bitmap / range table** over a pathfinding grid
(**H**), not a point graph. The exact row width, bit order and region layout are **U**.

## Follow-up: the object partition is MPRL.Unk04 (2026-10-20)

MPRL `Unk04` is a spatial group id. In `development_00_00.pm4`, points with `Unk04 = 573` cluster
around (225, 170) and points with `Unk04 = 577` cluster around (205, 185), all at the same terrain
height (Y = 40.20). `Unk14` correlates with `Unk04` (573 → 3, 577 → 0).

Testing the MPRR runs against those groups:

| Test | Result |
|---|---|
| MPRR runs whose middle MPRL points share one `Unk04` | **2,549,636 / 2,582,056 (98.8%)** |
| MPRR runs whose middle MPRL points share one `Unk00` | **2,581,846 / 2,581,846 (100%)** |

So an MPRR run is a **per-object group of MPRL terrain-intersection points**, keyed by
`MPRL.Unk04` (and `Unk00`). This is the "collect all surfaces for an object within the footprint"
mechanism the operator suspected. The first two records of a run are still **U**; they are not a
32×32 chunk and not a disjoint MSUR partition.

## Follow-up: MPRL.Unk04 is MSLK.GroupObjectId (2026-10-20)

Across the whole map, 176,748 of 178,588 MPRL points (98.97%) carry an `Unk04` that appears as an
`MSLK.GroupObjectId` (+0x04) in the same file. The alternatives are negligible:

| Match | Points |
|---|---|
| `MPRL.Unk04` == `MSLK.GroupObjectId` | **176,748 (98.97%)** |
| `MPRL.Unk04` == `MSUR.Ck24ObjectId` | 111 (0.06%) |
| `MPRL.Unk04` == `MSUR.GroupKey` | 289 (0.16%) |

This closes the loop: the terrain-intersection points, the connection records and the surfaces can
be joined on one object-group id. The id is a **group/model id, not a unique placement id** — it is
reused across placements, which is why earlier tests found 4,739 ids mapping to multiple models.

## Follow-up: ground-surface containment (2026-10-20)

The direct group→surface link is **not** a containment link: only 724 of 176,748 points (0.4%) lie
inside a surface whose MSLK window references their group id. That is expected if the linked
surfaces are the object's connection/wall surfaces while the points are on the ground. The ground
surfaces that actually contain the points split as:

| Containing surface class | MPRL points |
|---|---|
| M2 (`Ck24Type == 0`) | 119,191 (66.7%) |
| WMO (`Ck24Type` 64–67) | 51,821 (29.0%) |
| Other | 11,968 (6.7%) |

So the object partition is: `GroupObjectId` → MPRL terrain-intersection points + MSLK connection
records; the ground surfaces are the MSUR surfaces that contain those points.

## Files changed

- `wow-viewer/src/core/WowViewer.Core.PM4/Research/Pm4PathNodeProbe.cs` (new)
- `wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/Program.cs` (`--path-nodes`, `--mprr-dump`)
- `wow-viewer/docs/research/pm4-format-reference.md` (§3.7, E-PATH)
- `wow-viewer/specs/279-decode-pm4-format/evidence/path-nodes-v2-2026-10-20.json` (report)

## Verification

```powershell
dotnet build "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --nologo -v q
# exit 0, 0 errors

dotnet run --project "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --no-build -- --path-nodes "wow-viewer/test_data/development/World/Maps/development" "wow-viewer/specs/279-decode-pm4-format/evidence/path-nodes-v2-2026-10-20.json"
# exit 0; 616 files, MPRL 178588, MPRR 17150640, inside mesh 177525/178588, 30.959s

dotnet run --project "wow-viewer/specs/279-decode-pm4-format/evidence/point-reference-probe/PointReferenceProbe.csproj" -c Debug --no-build -- --mprr-dump "wow-viewer/test_data/development/World/Maps/development/development_00_00.pm4" 60
# exit 0; 81936 MPRR, 2493 MPRL, 9990 MSCN
```

## Criterion → evidence

| Criterion | Evidence |
|---|---|
| MPRL points lie on the PM4 mesh | 177,525 / 178,588 (99.4%) inside |
| MPRL has constant fields | `Unk02 = −1`, `Unk06 = 0x8000` for all points |
| MPRR is not an MPRL edge list | Only 39.5% / 62.4% of values in MPRL range |
| MPRR has a hierarchical run structure | Run lengths 3, 7, 11, 15, … (`4k+3`) with ~4:1 decay |
| MPRR runs terminate consistently | `(0, 0x0300)` before each `0xFFFF` sentinel |
