# WMO 0.9.0 Render Path — Part 08 (Runtime Evidence Table Template)

## Scope
Fill-ready table for concrete branch values at WMO group color/light gates and liquid layout gate(s).

## Branch sites (to lock during anchor pass)
- Group color/light gate: `TBD_FUN_GROUP_COLOR_GATE`
- Liquid layout gate: `TBD_FUN_LIQUID_LAYOUT_GATE`

## Table A — Group color/light anomaly
| Asset Case | Build/Binary | Breakpoint Hit Count | groupId | materialId | lightRef/ColorRef | Predicate True? | Outcome (Correct/Weird Color) | Notes |
|---|---|---:|---:|---:|---:|---|---|---|
| Stormwind group (problem) | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | TBD | Primary color anomaly sample |
| Secondary problem case | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | TBD | Validation sample |
| Control (known-good group) | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | TBD | Baseline |

## Table B — Liquid arrangement anomaly
| Asset Case | Build/Binary | Breakpoint Hit Count | liquidType | gridW x gridH | origin(x,y,z) | axis/stride mode | Predicate True? | Outcome (Correct/Misaligned) | Notes |
|---|---|---:|---|---|---|---|---|---|---|
| Stormwind liquid (problem) | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Primary layout failure sample |
| Secondary problem case | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Validation sample |
| Control (known-good liquid) | 0.9.0 / Build 3807 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Baseline |

## Capture checklist
1. Break on both gate functions and record values before branch/compare.
2. Record whether branch executed and final visual outcome.
3. Capture at least one full failing case and one control per table.
4. Preserve exact operand sources for later Part 09 closure.
