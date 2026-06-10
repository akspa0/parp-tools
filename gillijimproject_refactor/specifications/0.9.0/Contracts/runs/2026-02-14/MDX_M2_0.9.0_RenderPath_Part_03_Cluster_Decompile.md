# MDX/M2 0.9.0 Render Path — Part 03 (0x0042e1xx–0x0042edxx Cluster)

## Scope
Micro-pass decompilation of nearby functions to separate true render-gate logic from utility/accessor stubs.

## Decompilation set
- `FUN_0042e150` (from Part 02 disassembly context)
- `FUN_0042e2a0`
- `FUN_0042e510`
- `FUN_0042e590`
- `FUN_0042e5e0`
- `FUN_0042ec20`
- `FUN_0042ec30`
- `FUN_0042ec60`
- `FUN_0042ed20`
- `FUN_0042ecf0`

## Key findings
### 1) Confirmed render-path control candidates
- `FUN_0042e150` and `FUN_0042e2a0` both:
  - enforce model state checks (`+0x14 == 1` style gate)
  - iterate bounded arrays with index-vs-size assert messaging
  - recurse into linked child structures
- Both functions reference source-path-derived strings (`s_C__build_buildWoW_ENGINE_Source__...`) and assertion format strings (`s_index__0x_08X___array_size__0x_0_0081f484`).

### 2) Accessor/validation helpers
- `FUN_0042e510`, `FUN_0042e590`, `FUN_0042e5e0` are lightweight validation/accessor-style wrappers around model/shared pointers and state checks.

### 3) Non-gate stubs/utilities in sampled subset
- `FUN_0042ecf0` and `FUN_0042ec30` return constant `0xfffffffe` (stub-like).
- `FUN_0042ec20`, `FUN_0042ed20` return type-name strings.
- `FUN_0042ec60` is cleanup/destructor-like over keyframe/geoset animation sections.

## Current confidence update
- Confidence is now **High** that `FUN_0042e150`/`FUN_0042e2a0` belong to renderer-side model traversal/gating code.
- Confidence remains **Medium** on exact binding to `AddGeosetToScene: materialId >= numMaterials`; that specific message has not yet been observed in a decompiled body in this pass.

## Next micro-step
- Continue decompilation in adjacent functions likely to perform geoset enqueue/material binding checks (e.g., `FUN_0042e510` onward call graph neighbors) and stop once the exact material-id assertion literal appears in decompiled pseudocode.
