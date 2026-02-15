# MDX/M2 0.9.0 Render Path — Part 07 (Material-Gate Operand Mapping)

## Scope
Resolve operand roles for the proven material-bound assertion branch in `FUN_004349b0`.

## Proven branch (from Part 06)
- Function: `FUN_004349b0`
- Literal xref: `00434a37: PUSH 0x822668`
- Message: `AddGeosetToScene: geoShared->materialId (%d) >= numMaterials (%d)`

## Call-flow evidence
### A) Upstream setup in `FUN_00434500`
`FUN_004348a0` is called with stack args sourced from model data:
- `*(modelData + 0x84)`
- `*(modelData + 0x80)`
- `*(modelData + 0xAC)`
- `*(modelData + 0xA8)`
plus render flags in fastcall `EDX`.

### B) Dispatcher in `FUN_004348a0`
For each geoset candidate, `FUN_004349b0` is called with:
- `ECX = model`
- `EDX = render flags`
- stack arg 1 = geoset-shared pointer (computed from `+0x48`/`+0x70` domains)
- stack arg 2 = per-geoset metadata pointer
- stack arg 3 = material lookup/array base
- stack arg 4 = material-count domain (from upstream `+0xA8`)

### C) Gate in `FUN_004349b0`
- Conditional form in pseudocode:
  - `if (param_6 <= *param_3) { log AddGeosetToScene material bound violation }`
- Logging call passes two numbers in message order:
  - first: `*param_3`
  - second: `param_6`

## Operand interpretation (confidence-graded)
- `*param_3` => **geoShared.materialId** (High)
- `param_6` => **numMaterials** (High)

Rationale:
1. Literal text explicitly names `geoShared->materialId` and `numMaterials`.
2. Logged argument order matches message placeholders.
3. `param_6` arrives from `FUN_004348a0` domain fed by model-side collection counts.

## Condition form for contract
- Violation branch triggers when:
  - `geoShared.materialId >= numMaterials`
- Equivalent to implementation compare:
  - `if (numMaterials <= geoShared.materialId)`

## Remaining runtime task
- Capture concrete `(materialId, numMaterials)` pairs per test asset at this branch site.
