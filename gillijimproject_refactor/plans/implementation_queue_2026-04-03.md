# Implementation Queue - Apr 03, 2026

Use this as the chat-by-chat execution order.

Each item should be one narrow implementation chat with explicit proof.

## Recently Landed

- former queue item 3, `Editor UI scaffolding slice`, is landed in `gillijimproject_refactor/src/MdxViewer`
- proof captured so far: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- boundary: viewer/editor workspace shell plus task and save-affordance plumbing only; no fake save logic, map write path, or shared writer ownership
- former queue item 1, `Editor Slice 01 - object-delta transaction boundary`, is now landed in `wow-viewer`
- proof captured so far: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter "AdtPlacementReaderTests|AdtPlacementWriterTests"`
- boundary: translation-only moves for existing `MDDF` and `MODF` entries; no add/remove placement support, path-table rebuilds, terrain writes, or dirty-map pipeline yet
- former queue item 2, `MdxViewer aggregated dirty-map/save packaging slice`, is now implementation-landed in `gillijimproject_refactor/src/MdxViewer`
- proof captured so far: `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- boundary: selected existing ADT MDDF/MODF placements can now be staged across selection changes and saved in grouped per-source batches, but proof is still build-only; no add/remove support, terrain writes, or runtime signoff yet

1. Shared I/O seam for broader editor save ownership
- source plans: `wow_viewer_shared_io_library_plan_2026-03-26.md`, `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- target: extend the first placement writer beyond translation-only updates into the next required save operation such as add/remove placement support or path-table rebuilds
- proof: focused tests + real-data roundtrip on the fixed development dataset

2. Real-data validation for grouped MdxViewer save packaging
- source plans: `wow_viewer_editor_plan_2026-04-03.md`
- target: validate the new grouped pending-save queue against the fixed development dataset and capture the exact workflow/result without overstating runtime closure
- proof: real-data selected-placement save workflow against the fixed development dataset

3. M2 Slice 02 - section classification and material routing
- source plans: `wow_viewer_m2_runtime_plan_2026-03-31.md`
- target: typed active-section contract with preserved unresolved flags and material/effect routing metadata
- proof: build/tests + real asset inspect evidence beyond synthetic-only checks

4. M2 Slice 03 - animation/lighting/effect runtime
- source plans: `wow_viewer_m2_runtime_plan_2026-03-31.md`
- target: external anim contract ownership and typed runtime light/effect state
- proof: fixed-asset runtime metadata/evaluation evidence

5. World Runtime Slice 02 - visible-set extraction
- source plans: `wow_viewer_world_runtime_service_plan_2026-03-31.md`
- target: move visible-set contracts/scratch orchestration out of `WorldScene`
- proof: compatibility build + service seam usage in active consumer path

6. World Runtime Slice 03 - pass service extraction
- source plans: `wow_viewer_world_runtime_service_plan_2026-03-31.md`
- target: explicit runtime-owned terrain/WMO/MDX/overlay pass services
- proof: measurable frame-stage ownership shift in counters/contracts

7. Renderer performance Phase 3 follow-through
- source plans: `mdxviewer_renderer_performance_plan_2026-03-31.md`
- target: MDX batching/state-compatible submission with reduced transparent sorting cost
- proof: before/after counters and frame-time deltas on development map

8. WDL shared ownership seam
- source plans: `wow_viewer_format_parity_matrix_2026-03-28.md`, `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- target: first shared WDL reader + inspect surface
- proof: shared tests + real file inspect output

9. split ADT `_lod` seam
- source plans: `wow_viewer_format_parity_matrix_2026-03-28.md`
- target: detection + top-level reader ownership for `_lod.adt`
- proof: detector tests + inspect/converter integration check

10. BLP deep ownership slice
- source plans: `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- target: first-party decode/write seam for core pixel paths
- proof: reader/writer regression + real-data decode checks

11. Tool cutover vertical slice (dual-surface)
- source plans: `wow_viewer_cli_gui_surface_plan_2026-03-25.md`, `wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`
- target: one workflow exposed in both CLI and GUI over the same shared service boundary
- proof: one CLI command plus one UI workflow producing equivalent artifacts

## Deferred / Historical

- `viewer_future_port/*` prompt-era planning docs are reference-only unless a specific prompt is intentionally revived.
- broad v0.5.0 bootstrap/migration prompts are architecture references, not immediate implementation queue items.
