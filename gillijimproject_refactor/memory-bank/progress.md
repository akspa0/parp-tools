# Progress

This file is intentionally compressed. Keep only recent validated milestones, open boundaries, and the next recommended slice.

## Current Position

- The v10 terrain-AI lane is in early Wave 2.
- Wave 1 is complete.
- The current local continuation has moved Wave 2 beyond the original script-only slice: the canonical miner is now native in `wowviewer-converter mine-v10-brushes`, and bounded terrain-only plus hybrid proofs exist.

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
  - `dotnet build i:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/WowViewer.Core.csproj -c Debug` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/WowViewer.Core.IO.csproj -c Debug` passed

### Apr 26, 2026 - Wave 2 started with object-anchored 3D brush-pattern extraction

- Commit: `f125fa5` (`feat: Add extraction of object-anchored 3D brush patterns and related functionality`)
- What landed:
  - `gillijimproject_refactor/src/WoWMapConverter/scripts/v10/mine_mcal_brush_patterns.py`
  - placement-derived `ObjectMask257` and `ObjectPreciseMask257` via `AdtPlacementReader`
  - `wowviewer-converter extract-v10-tensors` as the canonical Wave 1 to Wave 2 extraction surface

### Apr 27, 2026 - Wave 2 widened and moved into native `wow-viewer` converter ownership

- `wowviewer-converter extract-v10-tensors` now writes matching `*_placements.json` sidecars when placement data exists
- `wowviewer-converter mine-v10-brushes` now owns the anchor-aware miner natively in `WowViewer.Tool.Converter`
- Native miner coverage includes `objects`, `terrain`, and `hybrid`
- `NpzTileSerializer` was fixed to emit standards-compliant NumPy payloads for direct NPZ consumption
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-v10-brushes --input-dir i:/parp/parp-tools/output/build-validation/v10-wave2-wider-corpus/corpus --placement-dir i:/parp/parp-tools/output/build-validation/v10-wave2-wider-corpus/corpus --output-dir i:/parp/parp-tools/output/build-validation/v10-wave2-wider-corpus/native-cli-hybrid-proof --anchor-mode hybrid --terrain-samples-per-tile 32 --dictionary-size 16 --min-occurrences 3 --context-radius 16 --seed 1337` passed
  - native CLI proof wrote `output/build-validation/v10-wave2-wider-corpus/native-cli-hybrid-proof/brush_dictionary.json`
- Wider bounded artifacts:
  - widened corpus: `output/build-validation/v10-wave2-wider-corpus/corpus`
  - widened corpus size: 6 usable root ADT tensor packs and 11,746 placement records
  - terrain-only proof: `output/build-validation/v10-wave2-wider-corpus/terrain-proof/brush_dictionary.json`
  - hybrid proof: `output/build-validation/v10-wave2-wider-corpus/hybrid-proof/brush_dictionary.json`

### Apr 26, 2026 - GPU viewer plan-set workflow was registered

- Added `.github/prompts/wow-viewer-gpu-viewer-plan-set.prompt.md` and updated workflow routing files
- Boundary:
  - workflow registration only
  - no new runtime parity claim

## Open Boundaries

- No validated MCLY combination dictionary run exists yet for the v10 corpus.
- No validated broad-corpus non-object-anchored MCAL composition vocabulary run exists yet for the v10 corpus.
- The next blocking decision is whether to retain or retire the older Python reference miner now that the canonical command is native.
- The world-viewer path is still mid-migration and should not be described as final runtime parity.

## Recommended Next Slice

1. Keep Wave 1 NPZ output as the canonical input surface for Wave 2.
2. Widen the current 6-tile proof into a larger curated or map-wide corpus while keeping terrain-only and hybrid validations separate.
3. Decide whether the older Python reference script should stay or be retired now that `mine-v10-brushes` is native.
4. After that, move to MCLY combination mining and then broader MCAL composition mining.
