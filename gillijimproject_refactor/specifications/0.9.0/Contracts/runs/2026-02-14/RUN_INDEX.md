# Render Ground-Truth Run Index — 2026-02-14

## Goal
Track incremental, small-scope outputs for render-path reverse engineering.

## Outputs
1. `MDX_M2_0.9.0_RenderPath_Part_01_Anchors.md`
   - string anchors + symbol visibility baseline
2. `MDX_M2_0.9.0_RenderPath_Part_02_Candidate_Functions.md`
   - address validation for historical function mapping
3. `MDX_M2_0.9.0_RenderPath_Part_03_Cluster_Decompile.md`
   - focused decompile pass over 0x0042e1xx–0x0042edxx cluster
4. `MDX_M2_0.9.0_RenderPath_Part_04_ModelRender_Function_Lock.md`
   - concrete `ModelRender.cpp` function anchors (`FUN_0043cb90`, `FUN_0043cea0`, etc.)
5. `MDX_M2_0.9.0_RenderPath_Part_05_Parser_Render_Contract_Crosscheck.md`
   - parser↔renderer invariant cross-check and contract implications
6. `MDX_M2_0.9.0_RenderPath_Part_06_MaterialGate_Literal_Closure.md`
   - exact binding of `00822668` to `FUN_004349b0` (`00434a37: PUSH 0x822668`)
7. `MDX_M2_0.9.0_RenderPath_Part_07_MaterialGate_Operand_Mapping.md`
   - confirms compare operands: `geoShared.materialId` vs `numMaterials`
8. `MDX_M2_0.9.0_RenderPath_Part_08_Runtime_Evidence_Table_Template.md`
   - fill-ready runtime table for kelthuzad/newer/control assets
9. `MDX_M2_0.9.0_RenderPath_Part_09_Ghidra_Value_Source_Map.md`
   - Ghidra-only static operand provenance for `materialId` and `numMaterials`

## Current status
- Render-path function enumeration: done
- Renderer gate decompilation: in progress
- Parser/render contract cross-check: pending
- Addendum write-up: pending

## Next write target
- `MDX_M2_0.9.0_RenderPath_Part_10_Ghidra_Comments_And_Renames.md`
   - stop condition: branch-site and caller-chain comments/renames applied for fast revisit.
