# MDX/M2 0.9.0 Render Path — Part 02 (Candidate Functions)

## Scope
Validate previously recorded render-path addresses against current Ghidra program state.

## Inputs
- Prior note reference: `specifications/outputs/053/mdx-alignment/01-overview.md`
- Candidate addresses from that note:
  - `AddGeosetToScene` -> `0x0042e1f0`
  - `ModelAddToScene` -> `0x0042ecf0`

## Validation results
### Candidate A: `0x0042e1f0`
- Disassembler resolved this address inside a larger function beginning at `0x0042e150`.
- `0x0042e150` behavior characteristics:
  - iterates list-like structures (`[esi+0xe0]` count-driven loop)
  - performs recursive traversal (`call 0x0042e150`)
  - includes assertion/logging pattern with format string pointer `0x81f484`
  - requires state/flags checks before traversal (`[ecx+0x14] == 1`, bit tests)
- **Assessment:** strong render-gate candidate function (likely geoset/tree enqueue validation logic).

### Candidate B: `0x0042ecf0`
- Current disassembly at `0x0042ecf0` is a stub:
  - `mov eax, 0xfffffffe`
  - `ret`
- **Assessment:** historical symbol mapping is stale for this binary/load-state, or address provenance came from a different build.

## Delta vs previous assumptions
- Prior address-level label for `AddGeosetToScene` appears offset/inexact: the active containing function starts at `0x0042e150`, not `0x0042e1f0`.
- Prior label for `ModelAddToScene` at `0x0042ecf0` is not valid in this current program state.

## Next micro-step
- Resolve which string literal is at `0x81f484` and whether it corresponds to the `AddGeosetToScene` material-bounds assertion.
- If confirmed, promote `0x0042e150` as the primary render gate function in the contract addendum.
