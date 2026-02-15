# WMO 0.9.0 Render Path — Part 09 (Ghidra Value Source Map)

## Scope
Ghidra-only closure artifact: exact static provenance of key operands in the two WMO anomaly gates:
- group color/light mismatch gate
- liquid layout/orientation gate

## Target branches
- Group color/light gate function: `TBD_FUN_GROUP_COLOR_GATE`
  - key disassembly window: `TBD_ADDR_START..TBD_ADDR_END`
- Liquid layout gate function: `TBD_FUN_LIQUID_LAYOUT_GATE`
  - key disassembly window: `TBD_ADDR_START..TBD_ADDR_END`

## Operand provenance (static)

### Gate A — Group color/light mismatch
#### Operand A1 — `group.materialId` (or equivalent)
- Source load: `TBD_INSN`
- Base pointer provenance: `TBD_REGISTER <- [TBD_STACK_SLOT or field chain]`
- Compare/log usage:
  - `TBD_ADDR: CMP ...`
  - `TBD_ADDR: PUSH ...`

#### Operand B1 — `material/light/color domain bound`
- Source load: `TBD_INSN`
- Base pointer provenance: `TBD_REGISTER <- [TBD_STACK_SLOT or field chain]`
- Compare/log usage:
  - `TBD_ADDR: CMP ...`
  - `TBD_ADDR: PUSH ...`

### Gate B — Liquid layout/orientation mismatch
#### Operand A2 — `liquid grid extent / index operand`
- Source load: `TBD_INSN`
- Base pointer provenance: `TBD_REGISTER <- [TBD_STACK_SLOT or field chain]`
- Compare/log usage:
  - `TBD_ADDR: CMP ...`
  - `TBD_ADDR: PUSH ...`

#### Operand B2 — `expected stride/orientation/domain operand`
- Source load: `TBD_INSN`
- Base pointer provenance: `TBD_REGISTER <- [TBD_STACK_SLOT or field chain]`
- Compare/log usage:
  - `TBD_ADDR: CMP ...`
  - `TBD_ADDR: PUSH ...`

## Closure requirements
1. Every operand used in the relevant `CMP` must have a proven static source chain.
2. Every log/assert literal used in mismatch branches must be tied to a concrete `PUSH literal` site.
3. Operand maps must match the runtime captures from Part 08 before contract closure.

## Output format rule
For each gate, preserve this tuple in final closure write-up:
`(compare site, operand A source, operand B source, branch polarity, literal binding)`
