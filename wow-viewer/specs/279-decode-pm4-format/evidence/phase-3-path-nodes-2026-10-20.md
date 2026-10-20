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
