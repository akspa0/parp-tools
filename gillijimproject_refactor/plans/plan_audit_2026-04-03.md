# Plan Audit - Apr 03, 2026

## Purpose

Audit the current plans folder and separate:

- implemented work that plans still present as pending
- truly open implementation gaps
- stale prompt-era docs that should be archived or clearly marked as historical

## Scope Audited

Top-level plans:

- mdxviewer_renderer_performance_plan_2026-03-31.md
- unified_format_io_overhaul_prompt_2026-03-23.md
- v0_5_0_new_repo_library_migration_prompt_2026-03-25.md
- v0_5_0_wow_viewer_bootstrap_and_migration_draft_2026-03-25.md
- vlm_dataset_reconstruction_plan_2026-03-31.md
- wow_viewer_bootstrap_layout_plan_2026-03-25.md
- wow_viewer_cli_gui_surface_plan_2026-03-25.md
- wow_viewer_editor_plan_2026-04-03.md
- wow_viewer_format_parity_matrix_2026-03-28.md
- wow_viewer_full_format_ownership_plan_2026-03-28.md
- wow_viewer_m2_runtime_plan_2026-03-31.md
- wow_viewer_pm4_library_plan_2026-03-25.md
- wow_viewer_shared_io_library_plan_2026-03-26.md
- wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md
- wow_viewer_world_runtime_service_plan_2026-03-31.md

viewer_future_port plans:

- alpha_core_sql_scene_liveness_prompt_2026-03-25.md
- enhanced_renderer_architecture_prompt_2026-03-25.md
- enhanced_renderer_plan_set_2026-03-25.md
- enhanced_terrain_first_slice_prompt_2026-03-25.md
- enhanced_terrain_shader_lighting_prompt_2026-03-25.md
- mdxviewer_architecture_plan.md
- pm4_support_plan.md
- shader_family_and_lighting_roadmap_prompt_2026-03-25.md
- v0_4_6_v0_5_0_roadmap_prompt_2026-03-25.md
- v0_5_0_goal_stack_prompt_2026-03-25.md
- viewer_performance_recovery_prompt_2026-03-25.md
- wowrollback_uniqueid_timeline_prompt_2026-03-25.md

## Audit Summary

### Already Landed But Still Easy To Misread As Pending

1. wow-viewer M2 runtime slice 01 is landed.
   - Core/M2, Core.IO/M2, Core.Runtime/M2, M2 foundation tests, and inspect m2 inspect exist.
   - Action: update m2 runtime plan header text that still describes M2 ownership as absent.

2. PM4 library extraction is far beyond an initial seed.
   - Core.PM4 readers, analyzers, inspect verbs, tests, and first MdxViewer consumer hookups are landed.
   - Action: replace first slice partially landed wording with a concise landed-through-slice-N summary.

3. Shared I/O progress is substantial and current.
   - Many landed entries are documented, including WMO flag correlation surfaces and uniqueid-report.
   - Action: keep this file as the operational changelog, but trim old MDX continuation sections into a historical appendix.

4. Renderer performance phase 1 plus part of phase 2 are effectively landed.
   - World render stats and frame-contract extraction have already happened.
   - Action: mark phases 1-2 as landed/partial and re-baseline phases 3+.

5. world-runtime extraction has started.
   - The plan still reads as if slice 01 is purely pending, but miss suppression and early runtime seam work already landed.
   - Action: mark slice 01 as landed or partially landed and move blocker text under current open defects.

### Truly Open Implementation Work

1. Editor foundation is still not implemented.
   - No real saved-map transaction boundary, dirty-map pipeline, object-move persistence, or writer ownership in wow-viewer yet.

2. M2 runtime slices 02-05 are still open.
   - section classification/material routing
   - animation/lighting/effect runtime
   - scene submission and batching
   - consumer cutover and parity harness

3. world runtime extraction slices 02-05 are still open.
   - visible set extraction
   - pass-service extraction
   - WorldScene host thinning
   - first wow-viewer app runtime consumer

4. Full format ownership remains incomplete.
   - ADT write ownership, split ADT _lod, WDL, deeper WMO payload ownership, deep MDX runtime seams, first-party BLP decode/write, broader DB2 ownership.

5. Tool cutover is still incomplete.
   - converter/inspect dual-surface architecture is planned but not fully cut over from legacy tools.

6. VLM dataset reconstruction plan appears mostly open.
   - manifest/provenance contract and curation split tooling still listed as the next slice.

### Stale Or Superseded Docs

1. viewer_future_port/mdxviewer_architecture_plan.md is heavily stale.
   - It lists major systems as missing even though many are now present.

2. viewer_future_port/pm4_support_plan.md is superseded for ownership direction.
   - Useful as historical PM4 coordinate evidence only.

3. viewer_future_port v0.4.6/v0.5.0 prompts are mostly historical.
   - Useful for archaeology, not active queueing.

4. format parity matrix is stale in at least one major row.
   - M2 is shown as none even though M2 slice 01 landed.

## Recommended Build Queue (What Is Left)

Priority 1: editor reality slice

- implement one saved-object transaction path in wow-viewer (object move or PM4-chosen object), with dirty-state and save scope contract.
- keep scope to object persistence first, not full terrain write.

Priority 2: M2 slice 02

- implement section classification and material routing contract in wow-viewer M2 runtime.
- require real-data probe proof beyond build/test.

Priority 3: world runtime slice 02

- extract visible-set contracts from WorldScene into wow-viewer runtime services.

Priority 4: shared I/O ownership gap closure

- choose one concrete family seam from parity matrix with high user impact:
  - split ADT _lod detection+reader, or
  - WDL shared reader, or
  - ADT object write ownership needed by editor save.

Priority 5: tool cutover seam

- wire one dual-surface workflow over shared services end-to-end (converter command + viewer workflow) without app-local parsing.

## Plan Maintenance Actions

1. Update status headers in:
   - wow_viewer_m2_runtime_plan_2026-03-31.md
   - wow_viewer_world_runtime_service_plan_2026-03-31.md
   - mdxviewer_renderer_performance_plan_2026-03-31.md

2. Refresh matrix row statuses in:
   - wow_viewer_format_parity_matrix_2026-03-28.md

3. Keep as active execution plans:
   - wow_viewer_editor_plan_2026-04-03.md
   - wow_viewer_m2_runtime_plan_2026-03-31.md
   - wow_viewer_world_runtime_service_plan_2026-03-31.md
   - wow_viewer_shared_io_library_plan_2026-03-26.md
   - wow_viewer_pm4_library_plan_2026-03-25.md

4. Move to historical/prompt reference status (or add explicit historical banner):
   - viewer_future_port/*.md except any prompt explicitly reused in current workflow
   - unified_format_io_overhaul_prompt_2026-03-23.md
   - v0_5_0_new_repo_library_migration_prompt_2026-03-25.md

## Reality Check

- There is real implementation progress in PM4 and shared I/O and first M2 foundation ownership.
- The editor core transaction/save seam is still missing.
- M2 animation/runtime ownership past foundation is still missing.
- Several plans are now mixed documents (part changelog, part old prompt), which is causing queue confusion.