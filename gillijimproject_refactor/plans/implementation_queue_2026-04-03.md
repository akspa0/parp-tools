# Implementation Queue - Apr 03, 2026

Use this as the chat-by-chat execution order.

Each item should be one narrow implementation chat with explicit proof.

1. Editor Slice 01 - object-delta transaction boundary
- source plans: `wow_viewer_editor_plan_2026-04-03.md`
- target: first save-capable object edit transaction in shared wow-viewer seams
- proof: one real-data object move/chosen-object save artifact re-read by shared readers

2. Shared I/O seam for editor save ownership
- source plans: `wow_viewer_shared_io_library_plan_2026-03-26.md`, `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- target: ADT/object persistence seam required by item 1
- proof: focused tests + inspect/converter command on fixed development data

3. Editor UI scaffolding slice
- source plans: `wow_viewer_editor_plan_2026-04-03.md`
- target: viewer/editor workspace split scaffold with dirty-state/save-scope affordance plumbing only
- proof: build + visible UI mode switch with no fake save logic

4. M2 Slice 02 - section classification and material routing
- source plans: `wow_viewer_m2_runtime_plan_2026-03-31.md`
- target: typed active-section contract with preserved unresolved flags and material/effect routing metadata
- proof: build/tests + real asset inspect evidence beyond synthetic-only checks

5. M2 Slice 03 - animation/lighting/effect runtime
- source plans: `wow_viewer_m2_runtime_plan_2026-03-31.md`
- target: external anim contract ownership and typed runtime light/effect state
- proof: fixed-asset runtime metadata/evaluation evidence

6. World Runtime Slice 02 - visible-set extraction
- source plans: `wow_viewer_world_runtime_service_plan_2026-03-31.md`
- target: move visible-set contracts/scratch orchestration out of `WorldScene`
- proof: compatibility build + service seam usage in active consumer path

7. World Runtime Slice 03 - pass service extraction
- source plans: `wow_viewer_world_runtime_service_plan_2026-03-31.md`
- target: explicit runtime-owned terrain/WMO/MDX/overlay pass services
- proof: measurable frame-stage ownership shift in counters/contracts

8. Renderer performance Phase 3 follow-through
- source plans: `mdxviewer_renderer_performance_plan_2026-03-31.md`
- target: MDX batching/state-compatible submission with reduced transparent sorting cost
- proof: before/after counters and frame-time deltas on development map

9. WDL shared ownership seam
- source plans: `wow_viewer_format_parity_matrix_2026-03-28.md`, `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- target: first shared WDL reader + inspect surface
- proof: shared tests + real file inspect output

10. split ADT `_lod` seam
- source plans: `wow_viewer_format_parity_matrix_2026-03-28.md`
- target: detection + top-level reader ownership for `_lod.adt`
- proof: detector tests + inspect/converter integration check

11. BLP deep ownership slice
- source plans: `wow_viewer_full_format_ownership_plan_2026-03-28.md`
- target: first-party decode/write seam for core pixel paths
- proof: reader/writer regression + real-data decode checks

12. Tool cutover vertical slice (dual-surface)
- source plans: `wow_viewer_cli_gui_surface_plan_2026-03-25.md`, `wow_viewer_tool_inventory_and_cutover_plan_2026-03-25.md`
- target: one workflow exposed in both CLI and GUI over the same shared service boundary
- proof: one CLI command plus one UI workflow producing equivalent artifacts

## Deferred / Historical

- `viewer_future_port/*` prompt-era planning docs are reference-only unless a specific prompt is intentionally revived.
- broad v0.5.0 bootstrap/migration prompts are architecture references, not immediate implementation queue items.
