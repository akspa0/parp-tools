# WMO 0.9.0 Render Path — Part 01 (Anchors)

## Scope
Establish initial anchor set in Ghidra for WMO group and liquid rendering paths.

## String/literal anchor search set
- `WMO`
- `MOGP`
- `MLIQ`
- `liquid`
- `material`
- `light`
- `group`

## Required function lock targets
- Group render entry: `TBD_FUN_WMO_GROUP_RENDER_ENTRY`
- Group material/color apply path: `TBD_FUN_WMO_GROUP_COLOR_APPLY`
- Group light bind path: `TBD_FUN_WMO_LIGHT_BIND`
- Liquid build path: `TBD_FUN_WMO_LIQUID_BUILD`
- Liquid draw path: `TBD_FUN_WMO_LIQUID_DRAW`

## Output checklist
1. Address + current symbol name for each target.
2. One-line role statement for each function.
3. Adjacent caller/callee map for each target (1 hop).

## Note
Use neutral names until branch behavior is confirmed.
