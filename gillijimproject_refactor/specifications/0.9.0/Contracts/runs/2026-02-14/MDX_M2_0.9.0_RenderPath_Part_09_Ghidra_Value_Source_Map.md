# MDX/M2 0.9.0 Render Path — Part 09 (Ghidra Value Source Map)

## Scope
Ghidra-only closure artifact: exact static provenance of the two operands in the material-bound branch.

## Target branch
- Function: `FUN_004349b0`
- Literal: `00822668` (`AddGeosetToScene: geoShared->materialId (%d) >= numMaterials (%d)`)
- Key disassembly window: `0x00434a20..0x00434a44`

## Operand provenance (static)
### Operand A — `geoShared.materialId`
- Loaded from `dword ptr [EDI]`
- `EDI` at this point is `dword ptr [EBP + 0x8]` (first stack argument to `FUN_004349b0`)
- Compare/log usage:
  - `00434a20: CMP dword ptr [EDI], ESI`
  - `00434a33: MOV EDX, dword ptr [EDI]`
  - `00434a35: PUSH ESI`
  - `00434a36: PUSH EDX`
  - `00434a37: PUSH 0x822668`

### Operand B — `numMaterials`
- Loaded into `ESI` from `dword ptr [EBP + 0x14]` (fourth stack argument)
- Compare/log usage:
  - `00434a20: CMP [EDI], ESI`
  - `00434a35: PUSH ESI`

## Condition semantics
- Branch path enters log block when compare is **not** `JC`:
  - `00434a22: JC 0x00434a44` (skip log on `[EDI] < ESI`)
- Therefore log executes on:
  - `[EDI] >= ESI`
- Mapped meaning:
  - `geoShared.materialId >= numMaterials`

## Caller-side value sources
### `FUN_004348a0` -> `FUN_004349b0`
- `FUN_004348a0` sets up `FUN_004349b0` arguments from geoset iteration domains.
- First stack arg to `FUN_004349b0` is geoset-shared pointer (domain from model arrays at `+0x48` or `+0x70` paths).
- Fourth stack arg is material-count domain propagated from caller chain.

### `FUN_00434500` -> `FUN_004348a0`
- `FUN_00434500` passes model-data sourced domains (`+0x84`, `+0x80`, `+0xAC`, `+0xA8`) into `FUN_004348a0`.
- This is the static root for `numMaterials` propagation into `FUN_004349b0`.

## Ghidra-only capture checklist (no runtime debugger)
1. Open `FUN_004349b0` and annotate branch at `0x00434a20`.
2. Label first stack arg as `geoSharedPtr` and `[geoSharedPtr]` as `materialId`.
3. Label fourth stack arg as `numMaterials`.
4. Mark branch predicate as `materialId >= numMaterials`.
5. Trace callers (`FUN_004348a0`, `FUN_00434500`, `FUN_004353d0`) and document field provenance comments for each arg.

## Limitation note
- Concrete runtime tuples `(materialId, numMaterials)` require execution-time inspection.
- With current scope constrained to Ghidra static analysis, this artifact provides exact static source mapping and branch truth condition.
