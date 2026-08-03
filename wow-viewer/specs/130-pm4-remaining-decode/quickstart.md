# Quickstart: PM4 Remaining Decode

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

PowerShell 7 on Windows. Run from `i:\parp\parp-tools\wow-viewer`.

> **Verification status**: the commands in "Baselines" and "Reference measurements" exist today.
> `pm4 unknowns`, `pm4 cross-tile` and `pm4 export-json` were confirmed on 2026-08-03 to honour
> `--output`; **`pm4 inspect` and `pm4 audit` accept `-o` and silently ignore it** — they only print.
> The commands under "New commands" are the [CLI contract](./contracts/cli-commands.md) for work not
> yet implemented — **re-verify each against the real argument parsing as it lands**. Do not treat
> this file as proof a command runs.

## Setup

```powershell
dotnet build tools\inspect\WowViewer.Tool.Inspect\WowViewer.Tool.Inspect.csproj
$INSPECT = "tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.dll"
$CORPUS  = "test_data\development\World\Maps\development"
```

The corpus is 616 `.pm4` files, 309 non-empty. Confirm before trusting any number:

```powershell
(Get-ChildItem $CORPUS -Filter *.pm4).Count    # expect 616
```

## Baselines — run these first

These are the numbers every success criterion is measured against. **Re-run them, do not
re-estimate them.**

```powershell
New-Item -ItemType Directory -Force output\pm4-decode | Out-Null

dotnet $INSPECT pm4 unknowns   -i $CORPUS -o output\pm4-decode\baseline-unknowns.json
dotnet $INSPECT pm4 cross-tile -i $CORPUS -o output\pm4-decode\baseline-cross-tile.json
```

Expected, from `baseline-unknowns.json` (`relationships[]`):

| relationship | fits | misses |
|---|---|---|
| `MSUR.Msvi window -> MSVI` | 518,092 | 0 |
| `MSVI -> MSVT` | 1,930,146 | 0 |
| `MSLK.Mspi window -> MSPI` | 598,882 | 0 |
| `MSPI -> MSPV` | 2,418,205 | 0 |
| `MSLK.RefIndex -> MSUR` | 1,268,782 | 4,553 |
| **`MSLK.GroupObjectId -> MPRL.Unk04`** | **65,819** | **1,206,977** |
| `MPRR.Value1 -> MPRL` | 6,778,712 | 7,200,518 |
| `MPRR.Value1 -> MSVT` | 8,740,189 | 5,239,041 |
| `MDOS.buildingIndex -> MDBH` | 1 | 24 |

From `baseline-cross-tile.json`: 1,229 distinct CK24 keys, 266 spanning 2+ tiles. **CK24 = 0 spans
291 tiles and is a null sentinel, not an object** — exclude it.

If any of these differ, stop. Either the corpus changed or something upstream broke, and every
comparison downstream is meaningless until it is explained.

```powershell
# Read a specific edge without opening the whole file
$b = Get-Content output\pm4-decode\baseline-unknowns.json | ConvertFrom-Json
$b.relationships | Where-Object { $_.edge -like "*GroupObjectId*" } |
    Format-List edge, status, fits, misses
```

## Reference measurements — `development_00_00.pm4`

Measured 2026-08-03 and used by research.md R7/R8. Cheap to re-check, and they anchor two findings.

```powershell
dotnet $INSPECT pm4 audit -i "$CORPUS\development_00_00.pm4"      # counts (prints only, -o ignored)
dotnet $INSPECT pm4 export-json -i "$CORPUS\development_00_00.pm4" -o output\pm4-decode\doc_0000.json
```

| chunk | entries | bounds min | bounds max |
|---|---|---|---|
| MSPV | 8,778 | (169.60, 31.84, 0.85) | (498.79, 363.85, 134.55) |
| MSPI | 26,458 | — | — |
| MSVT | 6,318 | (168.11, 31.00, −12.08) | (501.55, 450.70, 133.74) |
| MSVI | 15,602 | — | — |
| MSUR | 4,110 | — | — |
| MSCN | 9,990 | (168.84, 31.42, −12.08) | (499.38, 450.40, 133.00) |
| MPRL | 2,493 | **(31.00, 5.00, 168.18)** | **(364.86, 40.20, 499.77)** |
| MPRR | 81,936 | — | — |

Two things these settle:

- **The April 2025 OBJ is fully explained** (R7). Its 15,096 vertices = MSPV + MSVT = 8,778 + 6,318.
  Its 7,382 faces = Σ(MSUR.IndexCount − 2) = 15,602 − 2×4,110 — every surface as a triangle fan.
  MSPI produced no faces. Nothing about the surface mesh was lost.
- **MSPV, MSVT and MSCN share one frame; MPRL is the permuted chunk** (R8). MPRL's third axis
  (168–499) is MSVT's first (168–501). The nesting hazard is real but MPRL-specific.

## New commands

### Phase 1 — prior-art harvest (reading, no commands)

No CLI. Read the sources listed in [plan.md](./plan.md) Phase 1 and write
`prior-art-inventory.md`. They are all on `main` and are read-only extraction inputs:

```powershell
git ls-tree -r --name-only main parpToolbox/src/parpToolbox/Services/PM4/
git show main:PM4Tool/docs/pm4/pm4-analysis-findings.md
git show main:PM4Tool/docs/pm4/pm4-mesh-extraction.md
```

Gate: every extracted hypothesis is listed with its source file and the phase that will test it.
No decode claims — harvesting is not validating.

### Phase 2 — seed the evidence register

```powershell
dotnet $INSPECT pm4 evidence -i $CORPUS --seed -o output\pm4-decode\evidence-register.json
```

Gate: the nine open questions round-trip with status, evidence and confidence intact.

### Phases 3–4 — evaluate grouping rules

```powershell
dotnet $INSPECT pm4 grouping-rules -i $CORPUS `
    -o output\pm4-decode\grouping-comparison.json `
    -r output\pm4-decode\evidence-register.json
```

Gate: `G0` reproduces 65,819 / 1,206,977 exactly. Then compare rules on the *grouping* metric —
which is a different quantity from the baseline edge, see [research.md](./research.md) R1:

```powershell
$g = Get-Content output\pm4-decode\grouping-comparison.json | ConvertFrom-Json
$g.rules | Format-Table ruleId, surfacesGrouped, surfacesUngrouped, objectCount,
    crossTileObjectCount, largestObjectSurfaceCount
```

Check `largestObjectSurfaceCount` every time. A rule that swallows the corpus into one object scores
perfectly on coverage and is worthless.

### Phase 5 — emit the object identity table

```powershell
dotnet $INSPECT pm4 object-identity -i $CORPUS --rule G3 `
    -o output\pm4-decode\object-identity.json
```

Gate — determinism, since Spec 129 caches on it:

```powershell
dotnet $INSPECT pm4 object-identity -i $CORPUS --rule G3 -o output\pm4-decode\object-identity-2.json
(Get-FileHash output\pm4-decode\object-identity.json).Hash -eq
    (Get-FileHash output\pm4-decode\object-identity-2.json).Hash    # expect True
```

Gate — no surface silently dropped:

```powershell
$o = Get-Content output\pm4-decode\object-identity.json | ConvertFrom-Json
$o.coverage    # surfacesAssigned + surfacesUngrouped + surfacesSentinelExcluded == surfacesTotal
```

### Phase 6 — viewer whole-object selection

Run the viewer, load the development map, enable the PM4 overlay, and click a surface of a
multi-surface object.

Gate (SC-002): the whole object highlights; an object spanning tiles highlights in every tile it
occupies; the same click twice gives an identical selection; a surface with undetermined membership
selects alone and is visibly marked ungrouped.

### Phase 7 — connective geometry (MSPV/MSPI and MSCN)

Detector power **first** — this gate exists because the current counters cannot discriminate at all:

```powershell
dotnet $INSPECT pm4 connective-geometry -i $CORPUS --verify-detector
```

Only once that passes:

```powershell
dotnet $INSPECT pm4 connective-geometry -i $CORPUS --source both `
    -o output\pm4-decode\geometry-stream.json `
    -r output\pm4-decode\evidence-register.json
```

Gate: every candidate interpretation, **for both sources**, carries corpus-wide fits and misses; the
window-size histogram is published per TypeFlags family. The histogram is the deliverable — the ~4.04
mean indices per window is not evidence of anything on its own.

Frame resolution is not needed here: MSPV, MSVT and MSCN measurably share one coordinate frame
(research.md R8). MPRL is the permuted chunk.

### Phase 8 — reconstruct and measure against a real asset

Needs a **configured** asset root (Constitution VI — never hardcode it):

```powershell
$ASSETS = "<your configured client/asset root>"

dotnet $INSPECT pm4 reconstruct-object -i $CORPUS `
    --identity output\pm4-decode\object-identity.json `
    --object-id <pm4obj-...> `
    --asset-root $ASSETS `
    -o output\pm4-decode\reconstruction.json
```

Gate (SC-006): one object built both ways; volume difference quantified; `sealednessRatio` reported
against the real asset. `1.0` is a closed manifold.

### Phase 9 — MPRR and the remaining questions

```powershell
dotnet $INSPECT pm4 mprr -i $CORPUS --runs `
    -o output\pm4-decode\mprr.json `
    -r output\pm4-decode\evidence-register.json
```

Gate (SC-004): each of the nine questions is resolved, narrowed with domains eliminated, or
documented as unresolvable with the reason. Eliminations are recorded so the search is not repeated.

## Regression check — run after every phase

`pm4 unknowns` must keep emitting identical numbers. Those figures are cited by the spec, the epic
and the memory bank.

```powershell
dotnet $INSPECT pm4 unknowns -i $CORPUS -o output\pm4-decode\check-unknowns.json
(Get-FileHash output\pm4-decode\baseline-unknowns.json).Hash -eq
    (Get-FileHash output\pm4-decode\check-unknowns.json).Hash     # expect True
```

## Tests

```powershell
dotnet test tests\WowViewer.Core.PM4.Tests\WowViewer.Core.PM4.Tests.csproj
```

Unit tests may use synthetic chunk sets. **No claim about the format is ever validated on synthetic
data** (Constitution III) — claims are validated on the corpus, per file and in total.
