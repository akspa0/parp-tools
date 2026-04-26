# Progress

This file is intentionally compressed. Keep only recent validated milestones, open boundaries, and the next recommended slice.

## Current Position

- The v10 terrain-AI lane is in early Wave 2.
- Wave 1 is complete.
- The last committed Wave 2 slice landed object-anchored brush mining plus placement-derived object-mask extraction, but the first bounded end-to-end Wave 2 run has not been recorded yet.

## Recent Validated Milestones

### Apr 26, 2026 - v10 Wave 1 library infrastructure landed in `wow-viewer`

- Added the canonical tensor-pack extraction stack in shared libraries:
  - `TerrainTileTensorPack`
  - `AdtTensorPackBuilder`
  - `NpzTileSerializer`
  - `AdtMclqReader`
  - `AdtMtxfReader`
  - `AdtMcrfReader`
  - `WlFile` and `WlFileReader`
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/WowViewer.Core.csproj -c Debug` passed.
  - `dotnet build i:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj -c Debug` passed.
- Boundary:
  - this Wave 1 baseline was immediately followed by a Wave 2 commit that replaced the old "object masks empty" status with placement-derived proxy masks
  - PM4 masks are still empty
  - no trainer or model code exists yet

### Apr 26, 2026 - Wave 2 started with object-anchored 3D brush-pattern extraction

- Commit: `f125fa5` (`feat: Add extraction of object-anchored 3D brush patterns and related functionality`).
- What landed:
  - added `gillijimproject_refactor/src/WoWMapConverter/scripts/v10/mine_mcal_brush_patterns.py`
  - the script mines object-anchored MCAL brush patterns from Wave 1 NPZ tiles plus per-tile placement JSON and emits `object_anchored_brush_dictionary.npz` plus a JSON sidecar
  - `AdtTensorPackBuilder` now populates `ObjectMask257` and `ObjectPreciseMask257` from `MDDF` and `MODF` placements via `AdtPlacementReader`
  - `wowviewer-converter extract-v10-tensors` remains the canonical Wave 1 to Wave 2 extraction entrypoint
  - `TerrainPatchAdtCommand` was updated, and the earlier converter `SeamAudit` compile blocker is no longer present
- Validation:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with warnings only
- Boundary:
  - the new miner is script-level, not yet a `wow-viewer` CLI command
  - the miner still depends on external placement JSON under `--placement-dir`
  - no committed real-data run or artifact path was recorded for the new dictionary output

### Apr 26, 2026 - GPU viewer plan-set workflow was registered

- Added `.github/prompts/wow-viewer-gpu-viewer-plan-set.prompt.md` and the ordered prompt set under `.github/prompts/wow-viewer-gpu-viewer/`.
- Updated `.github/copilot-instructions.md` and `AGENTS.md` so future sessions route world-viewer work through the GPU-first, library-first plan.
- Boundary:
  - workflow registration only
  - no new runtime parity claim

### Apr 25-26, 2026 - `WowViewer.App` host-state extraction advanced, but the hard-reset boundary still stands

- `WowViewerWorldSceneHost` now owns more shell-facing state, including bootstrap-only session metadata before a full runtime frame exists.
- The app can capture a pending-load world UI with `--capture-during-world-load`.
- Boundary:
  - this is host-state progress, not target-architecture closure
  - `WowViewerWorldRuntimeBridge` and `WorldGpuPreviewRenderer` remain transitional bridge code

### Older brush tooling still exists, but it is separate from the new Wave 2 slice

- `wow-viewer` `ml-harvest-brushes` already exists and is validated as a deterministic brush-imprint harvest over exported dataset JSON plus heightmaps.
- Important distinction:
  - `ml-harvest-brushes` is older groundwork
  - the new `mine_mcal_brush_patterns.py` commit is the actual start of Wave 2 work
  - neither surface yet gives a fully wired, validated, canonical `wow-viewer` Wave 2 pipeline

## Open Boundaries

- No validated MCLY combination dictionary run exists yet for the v10 corpus.
- No validated end-to-end real-data run exists yet for the new object-anchored brush dictionary.
- No validated non-object-anchored MCAL brush dictionary or composition vocabulary run exists yet for the v10 corpus.
- The next blocking seam is the placement handoff: either export per-tile placement JSON beside Wave 1 NPZ output or teach the miner to read placements directly from ADT or `_obj0.adt` sources.
- The world-viewer path is still mid-migration and should not be described as final runtime parity.

## Recommended Next Slice

1. Keep Wave 1 NPZ output as the canonical input surface for Wave 2.
2. Wire the missing placement handoff for `mine_mcal_brush_patterns.py`.
3. Run the first bounded real-data object-anchored dictionary build and record its artifact path.
4. After that proof lands, move to MCLY combination mining and then broader MCAL composition mining.
