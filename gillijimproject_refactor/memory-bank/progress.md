# Progress

This file is intentionally compressed. Keep only recent validated milestones, open boundaries, and the next recommended slice.

## Current Position

- The v10 terrain-AI lane is in early Wave 2.
- Wave 1 is complete.
- The current local continuation has moved beyond the original script-only slice: the canonical anchor-aware brush miner is native, native MCLY/MCAL composition/MCAL brush/height-profile dictionary commands exist, the first bounded Stage 1 corpus command exists, the first bounded Stage 1 trainer baseline has passed on CUDA, and both tile-level and 16x16 chunk-grid minimap-to-MCLY classifier trainers have bounded CPU smokes.

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

### Apr 27, 2026 - Stage 1 minimap-backed v10 training baseline landed in `wow-viewer`

- `extract-v10-tensors` now accepts `--minimap-root` and writes `minimap_rgb_256.npy` when a loose minimap is available
- `NpzTileSerializer` metadata now writes valid JSON through `JsonSerializer`, so Windows source paths and signal lists are safe for downstream consumers
- `wowviewer-converter dataset-build-v10-stage1` now bulk-builds minimap-backed Stage 1 NPZ shards plus a JSON manifest from root ADTs and a minimap root
- `wow-viewer/scripts/train_v10_stage1_minimap2height.py` now exists as the first bounded Stage 1 trainer consuming the v10 NPZ shard contract directly
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed after the new command and metadata fix
  - `dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- dataset-build-v10-stage1 --input-dir i:/parp/parp-tools/gillijimproject_refactor/test_data/original_development/World/Maps/development --output-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --minimap-root i:/parp/parp-tools/datasets/original_development/development --manifest i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus/v10_stage1_manifest.json --limit 64` completed with `Written = 64` and `Skipped = 34`
  - `i:/parp/parp-tools/gillijimproject_refactor/.venv-train/Scripts/python.exe i:/parp/parp-tools/wow-viewer/scripts/train_v10_stage1_minimap2height.py i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus/v10_stage1_manifest.json --output-dir i:/parp/parp-tools/output/ml-training/v10_stage1_gpu_smoke --epochs 3 --batch-size 8 --num-workers 0 --device cuda --no-use-compile` passed on the local RTX 4070 Ti SUPER
  - bounded CUDA smoke metrics after 3 epochs: train loss `0.2355`, val loss `0.1217`, val MAE `2.91m`, val RMSE `23.80m`

### Apr 27, 2026 - Native enriched MCLY combination dictionary mining landed in `wow-viewer`

- `wowviewer-converter mine-v10-mcly` now scans v10 NPZ shards for `mcly_texture_ids`
- `TerrainTileTensorPack` metadata now preserves the tile-level `mcly_texture_names` MTEX table
- The command consumes `mcly_texture_names` when present, keys combinations by texture paths instead of local-only texture IDs, preserves local ID tuple distributions plus example tile/chunk coordinates, and writes both the plan-named `mclay_dictionary.json` and chunk-accurate `mcly_dictionary.json`
- Current biome tags are conservative texture-name token heuristics, not a trained biome classifier
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- dataset-build-v10-stage1 --input-dir i:/parp/parp-tools/gillijimproject_refactor/test_data/original_development/World/Maps/development --output-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --minimap-root i:/parp/parp-tools/datasets/original_development/development --manifest i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus/v10_stage1_manifest.json --limit 64 --overwrite` passed and regenerated shards with `mcly_texture_names`
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-v10-mcly --input-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --output-dir i:/parp/parp-tools/output/build-validation/v10-wave2-mcly-dictionary --min-occurrences 2 --example-limit 12` passed
  - proof output: `output/build-validation/v10-wave2-mcly-dictionary/mclay_dictionary.json`
  - bounded corpus result: `64` shards discovered, `11` tiles read with `mcly_texture_ids`, `1979` chunks counted, `41` raw texture-path keyed combinations, `35` retained combinations, `53` shards skipped as `missing_mcly_texture_ids`
  - retained biome-tag distribution: `grassland=16`, `built=8`, `dirt_path=7`, `desert=2`, `rocky=1`, `unknown=1`
- Test caveat:
  - `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-build` currently fails in this checkout, mostly because tests expect missing `gillijimproject_refactor/test_data/development/World/Maps/development/...` fixtures; one unrelated synthetic M2 footprint assertion also fails.

### Apr 27, 2026 - Native MCAL chunk composition dictionary mining landed in `wow-viewer`

- `wowviewer-converter mine-v10-mcal-compositions` now scans v10 NPZ shards for `mcal_alpha_pack_256`
- The command groups real 64x64x4 chunk-level alpha compositions by active-layer, coverage, gradient, quadrant-balance, and optional height-shape bins
- It writes JSON metadata plus NumPy centroids:
  - `mcal_composition_dictionary.json`
  - `mcal_composition_dictionary.npz`
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed with existing warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-v10-mcal-compositions --input-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --output-dir i:/parp/parp-tools/output/build-validation/v10-wave2-mcal-compositions --dictionary-size 32 --min-occurrences 2 --example-limit 12` passed
  - proof output: `output/build-validation/v10-wave2-mcal-compositions/mcal_composition_dictionary.json`
  - bounded corpus result: `64` shards discovered, `11` tiles read, `2816` chunks read, `545` candidate compositions, `446` raw composition groups, `32` retained compositions

### Apr 27, 2026 - Native MCAL brush-stroke dictionary mining landed in `wow-viewer`

- `wowviewer-converter mine-v10-mcal-brushes` now scans v10 NPZ shards for `mcal_alpha_pack_256`
- The command filters near-uniform fills, clusters real per-layer 64x64 alpha stamps, records coarse shape-family labels, and writes:
  - `mcal_brush_dictionary.json`
  - `mcal_brush_dictionary.npz`
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed with existing warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-v10-mcal-brushes --input-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --output-dir i:/parp/parp-tools/output/build-validation/v10-wave2-mcal-brushes --dictionary-size 32 --min-occurrences 2 --example-limit 12 --seed 1337` passed
  - proof output: `output/build-validation/v10-wave2-mcal-brushes/mcal_brush_dictionary.json`
  - bounded corpus result: `64` shards discovered, `11` tiles read, `11264` layer patches read, `9681` near-uniform patches rejected, `1583` candidate brushes, `32` retained brushes
  - NPZ shape check found `stamps=(32,64,64)`, `brush_ids=(32)`, `frequencies=(32)`, `shape_features=(32,7)`, and feature normalization arrays

### Apr 27, 2026 - Native height profile clustering landed in `wow-viewer`

- `wowviewer-converter mine-v10-height-profiles` now scans v10 NPZ shards for `height_257`
- The command clusters normalized downsampled height profiles and records terrain archetype labels, summary stats, per-tile labels, and representative examples
- It writes JSON metadata plus NumPy centroid payloads:
  - `height_profile_dictionary.json`
  - `height_profile_dictionary.npz`
- The reader accepts both standards-compliant NumPy magic and the older `?NUMPY` compatibility form already tolerated by the brush miner
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed with existing warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-v10-height-profiles --input-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --output-dir i:/parp/parp-tools/output/build-validation/v10-wave2-height-profiles --dictionary-size 8 --min-occurrences 1 --profile-size 17 --example-limit 8 --seed 1337` passed
  - proof output: `output/build-validation/v10-wave2-height-profiles/height_profile_dictionary.json`
  - bounded corpus result: `64` shards discovered, `64` tiles read, `8` retained profiles

### Apr 27, 2026 - Bounded minimap-to-MCLY classifier trainer landed in `wow-viewer`

- `wow-viewer/scripts/train_v10_minimap_to_mclay.py` now trains the first Wave 2 classifier for `minimap_rgb_256 -> retained MCLY palette label`
- The trainer consumes the existing v10 NPZ shard contract plus `mclay_dictionary.json` or `mcly_dictionary.json` from `mine-v10-mcly`
- It writes:
  - `minimap_to_mclay_classifier.pt`
  - `label_index.json`
  - `metrics.json`
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\train_v10_minimap_to_mclay.py` passed
  - `.venv\Scripts\python.exe wow-viewer\scripts\train_v10_minimap_to_mclay.py output\build-validation\v10-stage1-development-corpus\v10_stage1_manifest.json --dictionary output\build-validation\v10-wave2-mcly-dictionary\mclay_dictionary.json --output-dir output\ml-training\v10_minimap_to_mclay_smoke --epochs 2 --batch-size 4 --num-workers 0 --device cpu --no-channels-last` passed
  - smoke output: `output/ml-training/v10_minimap_to_mclay_smoke/minimap_to_mclay_classifier.pt`
  - bounded corpus result: `64` shards discovered, `11` labeled samples, `6` active retained MCLY labels, `8` train samples, `3` validation samples
- Environment note:
  - `gillijimproject_refactor\.venv-train\Scripts\python.exe` currently points at a missing UV-managed Python path in this checkout; the smoke used the workspace `.venv`, which has CPU Torch.

### Apr 27, 2026 - Bounded minimap-to-MCLY chunk-grid classifier trainer landed in `wow-viewer`

- `wow-viewer/scripts/train_v10_minimap_to_mclay_grid.py` now trains the first Wave 2 classifier for `minimap_rgb_256 -> 16x16 retained MCLY palette labels`
- The trainer consumes the existing v10 NPZ shard contract plus `mclay_dictionary.json` or `mcly_dictionary.json` from `mine-v10-mcly`
- It writes:
  - `minimap_to_mclay_grid_classifier.pt`
  - `label_index.json`
  - `metrics.json`
- It preserves dictionary-backed label provenance and uses `ignore_index=-100` for chunks whose texture combination is not retained in the mined dictionary
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\train_v10_minimap_to_mclay_grid.py` passed
  - `.venv\Scripts\python.exe wow-viewer\scripts\train_v10_minimap_to_mclay_grid.py output\build-validation\v10-stage1-development-corpus\v10_stage1_manifest.json --dictionary output\build-validation\v10-wave2-mcly-dictionary\mclay_dictionary.json --output-dir output\ml-training\v10_minimap_to_mclay_grid_smoke --epochs 2 --batch-size 4 --num-workers 0 --device cpu --no-channels-last` passed
  - smoke output: `output/ml-training/v10_minimap_to_mclay_grid_smoke/minimap_to_mclay_grid_classifier.pt`
  - bounded corpus result: `64` shards discovered, `11` labeled samples, `1,973` retained chunk labels, `35` active retained MCLY labels, `8` train samples, `3` validation samples

### Apr 26, 2026 - GPU viewer plan-set workflow was registered

- Added `.github/prompts/wow-viewer-gpu-viewer-plan-set.prompt.md` and updated workflow routing files
- Boundary:
  - workflow registration only
  - no new runtime parity claim

## Open Boundaries

- No validated broad-corpus MCAL brush-stroke vocabulary run exists yet beyond the bounded development Stage 1 proof.
- No validated broad-corpus minimap-to-MCLY classifier or chunk-grid run exists yet beyond the `11` currently labelable development shards.
- MCLY dictionary biome tags are heuristic and should be replaced or validated by the planned minimap-to-biome/palette classifier.
- The next blocking decision is whether to retain or retire the older Python reference miner now that the canonical command is native.
- Stage 1 exists only as a bounded trainer baseline today; Stage 2 refinement and broader experiment orchestration still remain open.
- The world-viewer path is still mid-migration and should not be described as final runtime parity.

## Recommended Next Slice

1. Keep the minimap-backed Wave 1 NPZ output plus `dataset-build-v10-stage1` manifest as the canonical Stage 1 input surface.
2. Widen the current 64-shard development proof into a larger curated or map-wide Stage 1 corpus and run longer CUDA training.
3. Widen the minimap-to-MCLY classifier and native MCAL brush dictionary proof to a broader corpus.
4. Decide whether the older Python reference miner should stay or be retired, then move into Stage 2 refinement.
