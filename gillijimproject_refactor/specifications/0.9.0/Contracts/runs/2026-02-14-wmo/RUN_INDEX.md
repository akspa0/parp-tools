# WMO Render Ground-Truth Run Index — 2026-02-14

## Goal
Track incremental, small-scope outputs for WMO render-path reverse engineering focused on:
- group color/light anomalies
- liquid layout/orientation mismatches

## Outputs
1. `WMO_0.9.0_RenderPath_Part_01_Anchors.md`
   - string anchors + symbol visibility baseline for WMO group/liquid paths
2. `WMO_0.9.0_RenderPath_Part_05_Parser_Render_Contract_Crosscheck.md`
   - parser↔renderer invariant cross-check and contract implications
3. `WMO_0.9.0_RenderPath_Part_08_Runtime_Evidence_Table_Template.md`
   - fill-ready runtime table for group color/light and liquid arrangement cases
4. `WMO_0.9.0_RenderPath_Part_09_Ghidra_Value_Source_Map.md`
   - static provenance map for key compare operands in color/light and liquid layout gates

## Current status
- Anchor pass: pending
- Renderer gate decompilation: pending
- Parser/render contract cross-check: seeded
- Runtime evidence table: seeded
- Addendum write-up: seeded (`WMO_0.9.0_Renderer_Addendum_2026-02-14.md`)

## Next write target
- `WMO_0.9.0_RenderPath_Part_10_Runtime_Table_Filled.md`
  - stop condition: concrete gate values captured for all sample WMO cases.
