# MDX/M2 0.9.0 Render Path — Part 08 (Runtime Evidence Table Template)

## Scope
Fill-ready table for concrete branch values at `FUN_004349b0` material-bound gate.

## Branch site
- Function: `FUN_004349b0`
- Assert/log literal: `00822668`
- Violation predicate: `numMaterials <= geoShared.materialId`

## Table (to fill during live stepping)
| Asset Case | Build/Binary | Breakpoint Hit Count | geoShared.materialId | numMaterials | Predicate True? | Outcome (Rendered/Invisible) | Notes |
|---|---|---:|---:|---:|---|---|---|
| Kel’Thuzad (problem) | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | Primary failure sample |
| Newer M2-style case | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | Compatibility probe |
| Control (known-good) | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | Baseline |

## Capture checklist
1. Break on `FUN_004349b0` (or exact offset around `0x00434a20..0x00434a44`).
2. Record values immediately before compare/log branch.
3. Record whether branch executed and whether model/geoset renders.
4. Repeat across all three cases.

## Note
- This artifact is intentionally a small template output; populate in a dedicated runtime pass.
