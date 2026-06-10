# Task 5 - Operand source map for reject gate(s)

## Compare tuple(s)

1. `(compare_site, operandA_source, operandB_source, branch_polarity, literal_binding)`

- `FUN_004349b0` / initial geoset gate
- Operand A source chain:
  - `param_3[6]` -> geoset id
  - scaled index: `geosetId * 0x14`
  - byte load: `*(char *)(param_4 + 3 + geosetId*0x14)`
- Operand B source: immediate `0`
- Branch polarity: submit path only if `A != 0`; reject/suppress if `A == 0`
- Literal binding: inline zero compare in branch condition

2. `(0x007abeab, model+0x3A8 (FormatVersion), 0x5DC, reject if A>B, "File version newer than newest")`
- Source chain for A:
  - `VERS`/profile parse -> `FUN_007abf40` / `FUN_007abdd0`
  - stored at `[model + 0x3A8]`
- Source chain for B: immediate `0x5DC`
- Polarity: reject if version exceeds 1500.

## Notes
- Tuple (1) is the earliest static draw-suppress compare.
- Tuple (2) is a hard parse reject gate (not render-specific) but useful for version provenance completeness.
