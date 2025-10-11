# Post‑Mortem: Viewer Pack Build and Data Harvesting

- Date: 2025-10-10
- Owner: WoWRollback pipeline
- Scope: Work to produce a self-contained 2D viewer pack that “just works”

## Summary
We attempted to ship a self-contained viewer folder by:
- Copying UI assets (`WoWRollback.Viewer/assets2d/`) into `05_viewer/`.
- Moving all viewer data to `05_viewer/data/` and making the server map `/data/*` → that folder.
- Harvesting LK-only overlays from the ADT conversion outputs into `data/overlays/<map>/...`.

Outcome: The viewer still returned 404 for `/index.html` in some runs and legacy overlays from the Analysis stage continued to be generated (and logged), causing confusion about which data was actually used.

## What Changed
- Viewer data now written under `05_viewer/data/`:
  - `data/index.json`
  - `data/tiles/<map>/<x>_<y>.webp`
  - `data/overlays/<map>/manifest.json`
  - `data/overlays/<map>/coords|m2|wmo|area/{x}_{y}.json`
- HTTP server mapping: `/data/*` → `<packRoot>/data/*`.
- New stage `ViewerBuildStageRunner` builds the pack, harvests overlays (LK-only), and copies UI assets (with a fallback `index.html` if assets are missing).
- Analysis stage still emits legacy overlays under `05_viewer/overlays/...` (not used by the viewer), but their existence and logs caused confusion.

## What Went Wrong
- UI `index.html` 404 persisted:
  - Asset source detection may be wrong in the environment (copy skipped silently before logging was improved), or server was pointed at a folder built before assets were copied.
  - Fallback `index.html` may not have been written due to earlier exceptions aborting the stage.
- Legacy Analysis overlays continued to be produced (e.g., `terrain_complete`, `objects_combined`, `shadow_map`) and logged. These are not consumed by the viewer but appeared under `05_viewer/`, giving the impression nothing changed.

## Current State
- `ViewerPackBuilder` writes tiles, overlays, and `index.json` under `05_viewer/data/`.
- `ViewerServer` maps `/data/*` to `<packRoot>/data/*`.
- `ViewerBuildStageRunner` integrates: tiles, overlays (LK-only), UI assets copy (with fallback).
- Analysis still writes legacy overlays into `05_viewer/overlays/` (unused by viewer).

## Actionable Next Steps
1. Disable legacy overlays in Analysis:
   - Set `GenerateOverlays = false` (or stop writing anything into `05_viewer/overlays/`).
2. Deterministic UI assets copy:
   - Add `--viewer-assets <path>` CLI flag; plumb into stage to avoid guessing.
   - Always log absolute source → destination and the resulting `index.html` path.
3. Always ensure `index.html` exists:
   - If assets copy fails, unconditionally write fallback `index.html` before returning; never 404 at root.
4. Hard-verify pack structure after build:
   - Log existence of: `<packRoot>/index.html`, `<packRoot>/data/index.json`, at least one tile and manifest path.
5. Cleanup legacy debris:
   - Optionally delete `05_viewer/overlays/` after build to avoid confusion.
6. Telemetry in logs:
   - Summarize counts for tiles and overlays per layer; print example URLs.

## Verification Plan
- Filesystem checks:
  - `05_viewer/index.html` exists.
  - `05_viewer/data/index.json` exists.
  - `05_viewer/data/tiles/<map>/0_0.webp` exists for at least one map.
  - `05_viewer/data/overlays/<map>/manifest.json` exists.
- HTTP checks:
  - `/` returns 200 with viewer page (or fallback page if assets missing).
  - `/data/index.json` returns 200.
  - `/data/tiles/<map>/0_0.webp` returns 200.
  - `/data/overlays/<map>/coords/0_0.json` returns 200 (or empty if tile missing).
- Network tab shows the viewer only fetches `/data/...` (not `/overlays/...`).

## Open Decisions
- Listfile resolution: Keep LK listfile only or also allow community listfile as fallback for `resolved` fields.
- Area layer: add a rendering plugin (labels/heatmap) vs. keep as raw JSON for now.

## Files Referenced
- `WoWRollback.Orchestrator/PipelineOrchestrator.cs`
- `WoWRollback.Orchestrator/AnalysisStageRunner.cs`
- `WoWRollback.Orchestrator/ViewerBuildStageRunner.cs`
- `WoWRollback.Core/Services/Viewer/ViewerPackBuilder.cs`
- `WoWRollback.ViewerModule/ViewerServer.cs`
- `WoWRollback.Viewer/assets2d/`
