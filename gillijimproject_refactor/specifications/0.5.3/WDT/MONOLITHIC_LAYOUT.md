# 0.5.3 Monolithic WDT↔ADT Layout (Deep-Dive Notes)

## Why this matters
In 0.5.3, the client appears to parse WDT root chunks and then continue into terrain/object chunks commonly associated with ADT content. This is consistent with early monolithic map behavior.

## Confirmed parser token checks (from Ghidra strings)
### Root WDT layer
- `MARE` (`0x0089FBE4`)
- `MAOF` (`0x0089FC00`)
- `MVER` (`0x0089FC2C`, `0x0089FC84`)
- `MAIN` (`0x0089FC54`)
- `MPHD` (`0x0089FC6C`)
- `MDNM` (`0x0089FC9C`)
- `MONM` (`0x0089FCB4`)

### Embedded terrain/object layer (ADT-like within WDT flow)
- `MHDR` (`0x008A2388`)
- `MCIN` (`0x008A236C`)
- `MTEX` (`0x008A2350`)
- `MCNK` (`0x008A126C`)
- `MCLY` (`0x008A1254`)
- `MCRF` (`0x008A12DC`)
- `MDDF` (`0x008A2334`)
- `MODF` (`0x008A2318`)

## Inferred parser stages
1. Open `<map>.wdt` via `wdtFile`/`%s\\%s.wdt` path logic.
2. Validate root chunk sequence (`MVER`, `MPHD`, `MAIN`, names, plus alpha-only `MARE/MAOF`).
3. Enter map-body parse that validates ADT-style chunk tokens.
4. Parse per-chunk terrain blocks (`MCNK` children like `MCLY`, `MCRF`) and object placements (`MDDF`/`MODF`).

## What remains unresolved
- Exact chunk order constraints vs permissive dispatch
- Precise field offsets for `MARE`, `MAOF`, and `MPHD`
- Whether additional MCNK children are handled via non-string token constants
- Definitive proof of reversed FourCC handling in this exact parser path

## Confidence
- **High**: chunk presence and mixed-layer parser behavior
- **Medium**: stage sequencing
- **Low**: field-level layouts for unresolved chunks
