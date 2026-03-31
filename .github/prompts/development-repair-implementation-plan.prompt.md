---
description: "Implement the next non-GUI development-map repair slice in gillijimproject_refactor so the original 4.x Development dataset can be reconstructed into auditable 3.3.5 outputs with explicit per-tile provenance."
name: "Development Repair Implementation Plan"
argument-hint: "Optional tile, repair phase, failing artifact, or subsystem to prioritize"
agent: "agent"
---

Implement the next non-GUI `development` repair slice in `gillijimproject_refactor` using the original 4.x source dataset as the source of truth and the current repair plan/spec as the architecture contract.

If the ask is primarily about VLM dataset generation, real-map curation, or model-oriented missing-layer reconstruction rather than repair-pipeline implementation, use `vlm-dataset-reconstruction-plan.prompt.md` instead of this repair-pipeline prompt.

## Read First

1. `gillijimproject_refactor/plans/development_repair_plan.md`
2. `gillijimproject_refactor/specifications/development_repair_pipeline_spec.md`
3. `gillijimproject_refactor/memory-bank/data-paths.md`
4. `gillijimproject_refactor/memory-bank/activeContext.md`
5. `gillijimproject_refactor/plans/pm4_support_plan.md`
6. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`

## Goal

Make concrete implementation progress on the repair pipeline that reconstructs the original `development` map into reproducible 3.3.5-compatible outputs.

The intended end state is:

1. all original source tiles are audited and classified
2. usable split tiles are merged and normalized into valid monolithic 3.3.5 ADTs
3. missing or unusable terrain tiles are rebuilt from WDL with explicit synthetic-tile labeling
4. `WL*` files are converted into 3.x-compatible liquid data
5. PM4 evidence is used deliberately and with provenance, not as undocumented magic
6. every tile records what fallback path was used

## Non-Negotiable Architecture

- Treat `test_data/development/World/Maps/development` as the source of truth.
- Treat museum exports and older generated outputs as reference-only, not canonical input.
- Do not run `development-repair` with `test_data/WoWMuseum/335-dev/...` as input; that path is validation/reference-only.
- Do not silently collapse all tile failures into one repair path.
- Do not claim a tile is "fixed" unless the output manifest says exactly which repair route was used.
- Do not hide rollback primitives behind ad hoc one-off shell workflows.
- Reuse existing rollback implementations where they are already the strongest reference, but port or wrap them cleanly in the active toolchain.
- Preserve the distinction between:
  - original terrain preserved
  - original terrain normalized/repaired
  - scan-order recovered terrain
  - WDL-generated terrain
  - PM4-refined terrain

## Required Tile Classes

Every tile must be classified into one of these buckets before repair:

- `healthy-split`
- `index-corrupt`
- `scan-only-root`
- `wdl-rebuild`
- `manual-review`

Use the active `development-analyze` command and/or the same rules it encodes as the front door for this classification.

## Required Starting Files

1. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Services/DevelopmentMapAnalyzer.cs`
2. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/Program.cs`
3. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/WlToLiquidConverter.cs`
4. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs`
5. `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs`
6. `gillijimproject_refactor/WoWRollback/WoWRollback.Cli/Commands/DevelopmentRepairCommand.cs`
7. `gillijimproject_refactor/WoWRollback/WoWRollback.Cli/Commands/RepairMcnkIndicesCommand.cs`
8. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/SplitAdtMerger.cs`
9. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/Services/WdlToAdtGenerator.cs`
10. `gillijimproject_refactor/WoWRollback/WoWRollback.PM4Module/GlobalUniqueIdFixer.cs`

## Required Implementation Order

1. Confirm the current audited tile classes from the original development dataset.
2. Decide what the next minimal production-worthy slice is.
3. Prefer implementing the first active `development-repair` orchestration seam in `wowmapconverter` over adding more isolated rollback-only commands.
4. Wire the first repair command to cover only the stable, high-value phases:
   - split merge
   - `MCNK` index repair
   - WDL generation for missing/unusable terrain
   - `WL*` to `MH2O`
   - per-tile manifest emission
5. Keep PM4 terrain refinement limited to validated `MPRL` usage in the first active slice.
6. Leave CK24/WMO reconstruction and viewer `UniqueID` range filtering as explicit next phases unless they are directly required for the chosen slice.

## Required Provenance Model

Every repaired tile must emit enough metadata to answer:

- what source files existed
- what source files were actually used
- whether split merge ran
- whether chunk indices were repaired
- whether WDL generation ran
- whether PM4 `MPRL` patching ran
- whether `WL*` liquids were converted
- whether minimap-derived `MCCV` painting ran
- whether the tile still needs manual review

## Strong Reuse Guidance

When choosing between active-core code and rollback code:

- prefer active-core code when the behavior already exists there
- prefer rollback code when it is clearly the stronger reference implementation
- if a rollback implementation is required for the active path, either:
  - port the logic into active core, or
  - wrap it behind a clean active command surface with explicit dependencies

Do not leave the real repair path undocumented in a rollback-only corner.

## Validation Rules

- Use the fixed development paths from the memory bank.
- Build the changed project(s).
- If you did not run the new repair command on real `development` data, say so explicitly.
- If you only improved the audit or command wiring, say so explicitly.
- If automated tests were not added or run, say so explicitly.
- Do not claim the reconstructed map is correct from build-only validation.
- Do not generalize from one tile if the command is still single-tile or sample-only.

## Deliverables

Return all items:

1. the exact repair slice implemented
2. files changed and why
3. which phases are now active versus still planned
4. validation status split into build, tests, and real-data execution
5. any tile classes or data-shape problems still blocking broader rollout
6. any plan/spec or memory-bank updates still required

## First Output

Start with:

1. the current state of the active repair pipeline
2. the next minimal implementation slice worth landing
3. which rollback primitive, if any, should be treated as the source of truth for that slice
4. what files you will change first
5. what real-data validation you can perform after the change