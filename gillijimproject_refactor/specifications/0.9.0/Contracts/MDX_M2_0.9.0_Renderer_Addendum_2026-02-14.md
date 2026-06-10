# MDX/M2 0.9.0 Renderer Addendum Contract — 2026-02-14

## Purpose
Record renderer-side ground truth discovered in the 2026-02-14 run and state enforceable contract implications for 0.9.0 tooling.

## Renderer anchor set (ground-truth)
- `FUN_0043cb90` (`ModelRender.cpp` assertions at lines `0xaf7`, `0xaf8`, `0xb07`)
  - model/shared validation
  - render-side material attach/creation
  - geoset shared handoff to `FUN_0043cea0`
- `FUN_0043cea0` (`ModelRender.cpp` assertion at line `0xacd` with `geoShared` null gate)
  - geoset shared payload materialization
  - repeated strict index bound checks during stream copies
- Adjacent render helpers in same source file:
  - `FUN_0043a680`, `FUN_0043ae70`, `FUN_0043d3a0`, `FUN_0043b6e0`, `FUN_0043b8b0`

## Contract statements (normative for 0.9.0 profile)
1. Parser-output geoset stream cardinalities MUST remain coherent across arrays consumed by renderer copy paths.
2. Material linkage for geosets MUST be treated as required render input, not advisory metadata.
3. Any parser hard-fail condition (`GEOS` overrun, `SEQS` stride violation) MUST remain hard-fail; do not degrade to warning.
4. Tooling should preflight geoset/material consistency before export/render handoff.

## Literal binding closure
- `00822668`: `AddGeosetToScene: geoShared->materialId (%d) >= numMaterials (%d)`
  - bound to `FUN_004349b0`
  - disassembly proof: `00434a37: PUSH 0x822668`
  - caller context: `FUN_004348a0` geoset iteration path invokes `FUN_004349b0`

Status: renderer subtrack is now closed at both function-level and literal-branch-level.

## Traceability
- See staged run artifacts in:
  - `specifications/0.9.0/Contracts/runs/2026-02-14/`
