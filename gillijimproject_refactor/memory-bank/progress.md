# Progress

This file is intentionally compressed. Keep only recent validated milestones, open boundaries, and the next recommended slice.

## Current Position

- The v10 terrain-AI lane is in Stage 2 refinement.
- Wave 1 is complete.
- Wave 2 pattern-mining infrastructure is complete and native: all dictionary commands, label manifests, and bounded classifier trainers are in place.
- The older Python reference miner has been retired; the canonical path is native.
- Stage 2 refinement is now the active slice.

## Recent Validated Milestones

### Apr 30, 2026 - Full native v10 corpus orchestration path landed

- `wowviewer-converter list-maps` now discovers map names from a staged client archive/catalog and writes a JSON map list for corpus runs
- `wowviewer-converter dataset-build-v10-stage1` now accepts `--client-root`, `--build-key`, `--map-name`, and `--tile-list`, allowing selected native v10 shards to be built per client/map group while reusing one archive catalog and loading archive-backed minimaps
- `wowviewer-converter extract-v10-tensors` also accepts archive-backed minimap metadata for one-off shard extraction, but grouped `dataset-build-v10-stage1` is the broad-corpus path
- `wow-viewer/scripts/build_v10_corpus.py` now:
  - defaults to per-client map discovery instead of the old fixed map list
  - normalizes C# fingerprint JSON before dedup scoring
  - deduplicates to the configured native shard budget
  - builds only selected tiles through batched per-group tile lists, defaulting to `32` selected tiles per Stage 1 batch
  - emits `v10_full_native_stage1_manifest.json`
  - runs trainer-facing curation with `--max-total` capped by `max_tiles` (default `1500`) and no hidden `0.25` shrink unless configured
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\build_v10_corpus.py wow-viewer\scripts\curate_v10_training_shards.py` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug --no-restore` passed
  - `list-maps` smoke passed against staged `3_3_5_12340` and wrote a limited JSON map list
  - grouped `dataset-build-v10-stage1 --tile-list` smoke wrote a one-shard native v10 manifest for `development_0_0`
  - `curate_v10_training_shards.py` accepted that smoke manifest with `--max-total 1500 --max-selected-fraction 0 --max-per-era 0`
- Boundary:
  - The full all-client/all-map corpus has not been executed yet.
  - One archive-backed `AhnQiraj_26_46` extraction smoke timed out during tensor extraction. Broad corpus runs must remain timeout/skip-gated until that slow or unsupported ADT family is isolated.

### Apr 29, 2026 - Minimap tileset decomposition preprocessor landed

- `MinimapTilesetPatternMatcher` now ranks v3 tileset pattern candidates for each minimap grid cell using mean RGB, chroma signature, baked chroma-detail residual, dominant palette, and detail-energy distance
- `wowviewer-converter decompose-minimap-tilesets` now consumes `v10-tileset-patterns.v3` and writes `v10-minimap-tileset-decomposition.v1`
- The command writes `minimap_tileset_decomposition.json`, `best_match_mean.png`, `residual_to_best_mean.png`, and `confidence.png` for inspection before any Stage 2 channel contract changes
- Proof:
  - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter PatternMinerTests` passed with `4` focused tests
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug --no-restore` passed with existing warnings only
  - bounded `64`-pattern v3 tileset library mining passed at `output/build-validation/v10-tileset-pattern-v3-limit64/pattern_library.json`
  - bounded decomposition smoke passed at `output/build-validation/v10-minimap-tileset-decompose-kalimdor-smoke/minimap_tileset_decomposition.json` over an existing Kalimdor minimap capture with `64` cells, `64` pattern candidates, `3` candidates per cell, `9` distinct top candidates, average top score `0.9402`, and preview PNGs
- Boundary:
  - This is a candidate-ranking preprocessor, not a solved alpha-mask or iterative subtraction pipeline.
  - The smoke used an existing app minimap capture rather than a fresh raw client minimap export; use raw/source minimap tiles for the next quality pass.

### Apr 29, 2026 - Color/detail-aware tileset pattern signatures landed

- `PatternMiner` now separates grayscale pattern identity from RGB, luminance-normalized chroma identity, and baked chroma-detail residual identity
- `PatternStamp` now records `MeanRgb`, `RgbStdDev`, `MeanHueDegrees`, `MeanSaturation`, `Colorfulness`, `MeanColorHex`, `DominantColorsHex`, `ColorMipSignature`, `ChromaMipSignature`, `ChromaDetailSignature`, `ChromaDetailEnergy`, `PatternSignatureHash`, `ColorSignatureHash`, `ChromaSignatureHash`, and `ChromaDetailSignatureHash`
- `wowviewer-converter mine-tileset-patterns` now writes `schema_version = v10-tileset-patterns.v3` and prints tint/chroma/detail evidence in the top-pattern summary
- Proof:
  - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter PatternMinerTests` passed with `3` focused tests
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug --no-restore` passed with existing warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-tileset-patterns --input i:/parp/parp-tools/output/ml-training/v10_tileset_database/merged_tileset_index.json --output-dir i:/parp/parp-tools/output/build-validation/v10-tileset-pattern-color-smoke --limit 5 --mip 3` passed
  - smoke output: `output/build-validation/v10-tileset-pattern-color-smoke/pattern_library.json`
  - smoke result: `5` processed, `0` errors, `5` pattern clusters, with `MeanColorHex`, `DominantColorsHex`, `ColorSignatureHash`, `ChromaSignatureHash`, `ChromaDetailSignatureHash`, and `ChromaDetailEnergy` in the JSON
- Boundary:
  - This lands the texture-pattern evidence layer only. The minimap-decomposition pass that uses this library to identify era-specific tileset variants, baked detail layers, and alpha-mask candidates is still open.
  - Do not widen the Stage 2 model channel contract for texture decomposition until that preprocessor has a bounded artifact and proof.

### Apr 29, 2026 - Slim Stage 2 architecture and curation defaults landed

- `wow-viewer/scripts/train_v10_stage2_terrain_synth.py` now defaults to `slim_structured_v1`, `120` epochs, and `--coarse-prior-mode zero`
- The slim model keeps the split-stem surface/structure/liquid routing but reduces width to `679,051` parameters over the current `23` input channels
- The old target-fed `height_17` coarse prior is no longer the default because it leaks the training target into the input; use `--coarse-prior-mode target` only for deliberate refinement-only comparisons
- `--stage1-checkpoint` now fails loud because Stage 1 predicted coarse-prior wiring is not implemented yet
- `wow-viewer/scripts/curate_v10_training_shards.py` now defaults to `--max-selected-fraction 0.25` plus `--max-per-era 128`
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\curate_v10_training_shards.py wow-viewer\scripts\train_v10_stage2_terrain_synth.py` passed
  - model instantiation reported `slim_structured_v1`, `zero`, `23`, and `679,051` parameters
  - full-corpus slim curation passed at `output/ml-training/v10_curated/v10_full_corpus_slim_pattern_manifest.json`
  - full-corpus result: `3,945` candidates, `3,240` valid preselection shards, `717` selected shards, `705` rejected, and all `41` pattern-annotated native v10 rows retained
  - native-dev slim curation passed at `output/ml-training/v10_curated/v10_dev_slim_pattern_manifest.json`
  - native-dev result: `64` candidates, `41` valid preselection shards, `10` selected shards, all selected rows carrying pattern hints
  - bounded CPU trainer smoke passed at `output/ml-training/v10_stage2_slim_arch_smoke/checkpoints/best.pt` with `--max-samples 8 --epochs 1 --device cpu`
- Boundary:
  - The one-epoch CPU smoke emitted a low prediction-variance warning, which is expected for that tiny proof and is not a quality verdict.
  - Relax curation only after a longer slim CUDA run shows real underfitting.

### Apr 29, 2026 - Era-routed v10 tileset harvest landed

- `wowviewer-converter harvest-tileset-blps` now uses each merged tileset entry's `era_tag` to pick the matching staged-local client session before falling back to the remaining client roots
- The shared `V10TilesetArchiveReader` centralizes archive session setup and loose-file fallback for the v10 tileset index and harvest commands
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed with existing warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- harvest-tileset-blps --input i:/parp/parp-tools/output/ml-training/v10_tileset_database/merged_tileset_index.json --output-dir i:/parp/parp-tools/output/build-validation/v10-tileset-harvest-era-routing-smoke-counters --limit 5` passed
  - smoke output: `output/build-validation/v10-tileset-harvest-era-routing-smoke-counters/harvest_manifest.json`
  - result: `5` exports, `5` preferred era-session hits, `0` fallback hits, and `0` errors from staged-local `output/tmp/wowarchive-clients` roots

### Apr 27, 2026 - Full mixed-corpus v10 Stage 2 CUDA run 3 launched

- The canonical long-running Stage 2 trainer was launched from `wow-viewer/scripts/train_v10_stage2_terrain_synth.py` using the proven curated manifest `output/ml-training/v10_curated/v10_v9all_plus_native_dev_balanced_manifest.json`
- The run uses `gillijimproject_refactor/.venv-train` with Torch `2.11.0+cu128` on CUDA and writes to `output/ml-training/v10_stage2_v9cache_native_dev_cuda_full_run3_20260427`
- Launch command:
  - `i:/parp/parp-tools/gillijimproject_refactor/.venv-train/Scripts/python.exe i:/parp/parp-tools/wow-viewer/scripts/train_v10_stage2_terrain_synth.py i:/parp/parp-tools/output/ml-training/v10_curated/v10_v9all_plus_native_dev_balanced_manifest.json --output-dir i:/parp/parp-tools/output/ml-training/v10_stage2_v9cache_native_dev_cuda_full_run3_20260427 --epochs 60 --batch-size 4 --num-workers 4 --device cuda`
- Startup proof:
  - curated manifest exists with `1,262` entries
  - `torch.cuda.is_available()` returned `True`
  - `nvidia-smi` showed a live Python process from the UV-managed interpreter on the RTX 4070 Ti SUPER during launch
  - terminal output reached Torch Inductor startup (`Not enough SMs to use max_autotune_gemm mode`), which is expected startup noise rather than a fatal error
- Boundary:
  - this is a launched full run, not a completed training result yet

### Apr 27, 2026 - Native rectangular prefab-cell clone detection landed in `wow-viewer`

- `wowviewer-converter mine-v10-prefab-cells` now detects repeating square/rectangular chunk sets (cells) that were copy-pasted across maps
- The command enumerates all cell positions for configurable widths/heights via `--cell-sizes` (e.g. `8x8,12x12,16x16`), computes a strict SHA256 fingerprint from per-chunk MCLY tuples, hole bits, quantized height stats, and per-layer alpha coverage signatures, and groups exact matches across tiles
- Relaxed hash now captures per-layer quantized coverage (`L0q0.25,L1q0.50,...`) so cells with similar alpha layer patterns group together even when dominant layer differs
- It retains only cells with frequency >= `--min-occurrences`, averages their height/alpha/hole data into centroids, and writes per-size subdirectories:
  - `prefab_cell_dictionary.json` (schema v2 with `alpha_layer_signatures`)
  - `prefab_cell_dictionary.npz`
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with warnings only
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- mine-v10-prefab-cells --input-dir i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus --output-dir i:/parp/parp-tools/output/build-validation/v10-wave2-prefab-cells-large --cell-sizes 8x8,12x12,16x16 --dictionary-size 128 --min-occurrences 2 --example-limit 8` passed
  - proof output: `output/build-validation/v10-wave2-prefab-cells-large/`
  - bounded corpus result: `64` shards discovered, `11` tiles read, large-cell retained cells:
    - `8x8`: `2` retained cells
    - `12x12`: `1` retained cell
    - `16x16`: `0` retained cells

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
- It can now consume the reusable native `v10-mcly-label-manifest.v1` output directly, instead of recomputing labels from NPZ plus dictionary on every training run
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\train_v10_minimap_to_mclay_grid.py` passed
  - `.venv\Scripts\python.exe wow-viewer\scripts\train_v10_minimap_to_mclay_grid.py output\build-validation\v10-stage1-development-corpus\v10_stage1_manifest.json --dictionary output\build-validation\v10-wave2-mcly-dictionary\mclay_dictionary.json --output-dir output\ml-training\v10_minimap_to_mclay_grid_smoke --epochs 2 --batch-size 4 --num-workers 0 --device cpu --no-channels-last` passed
  - `.venv\Scripts\python.exe wow-viewer\scripts\train_v10_minimap_to_mclay_grid.py output\build-validation\v10-wave2-mcly-labels\v10_mcly_label_manifest.json --output-dir output\ml-training\v10_minimap_to_mclay_grid_manifest_smoke --epochs 2 --batch-size 4 --num-workers 0 --device cpu --no-channels-last` passed
  - smoke output: `output/ml-training/v10_minimap_to_mclay_grid_smoke/minimap_to_mclay_grid_classifier.pt`
  - manifest-driven smoke output: `output/ml-training/v10_minimap_to_mclay_grid_manifest_smoke/minimap_to_mclay_grid_classifier.pt`
  - bounded corpus result: `64` shards discovered, `11` labeled samples, `1,973` retained chunk labels, `35` active retained MCLY labels, `8` train samples, `3` validation samples

### Apr 27, 2026 - Native reusable MCLY label manifest generation landed in `wow-viewer`

- `wowviewer-converter label-v10-mcly` now materializes retained MCLY dictionary labels from Stage 1 NPZ shards plus `mclay_dictionary.json`
- It writes `v10-mcly-label-manifest.v1` with:
  - per-label dictionary provenance and usage counts
  - per-tile dominant retained palette metadata
  - per-tile 16x16 chunk label grids
  - `ignore_index = -100` for chunks whose texture combination was not retained in the mined dictionary
- Proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug --no-restore` passed with existing warnings only after escalation for dotnet first-run sandbox denial
  - `dotnet run --no-build --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- label-v10-mcly --input i:/parp/parp-tools/output/build-validation/v10-stage1-development-corpus/v10_stage1_manifest.json --dictionary i:/parp/parp-tools/output/build-validation/v10-wave2-mcly-dictionary/mclay_dictionary.json --output i:/parp/parp-tools/output/build-validation/v10-wave2-mcly-labels/v10_mcly_label_manifest.json --min-retained-chunks 8` passed after escalation for the same dotnet first-run sandbox denial
  - proof output: `output/build-validation/v10-wave2-mcly-labels/v10_mcly_label_manifest.json`
  - bounded corpus result: `64` shards discovered, `11` shards with `mcly_texture_ids`, `11` labeled samples, `1,973` retained chunks, `35` active retained labels, `53` skipped shards

### Apr 26, 2026 - GPU viewer plan-set workflow was registered

- Added `.github/prompts/wow-viewer-gpu-viewer-plan-set.prompt.md` and updated workflow routing files
- Boundary:
  - workflow registration only
  - no new runtime parity claim

### Apr 27, 2026 - Wave 2 continuation: retired Python reference miner and landed first bounded Stage 2 trainer

- Moved the legacy Python reference miner from `gillijimproject_refactor/src/WoWMapConverter/scripts/v10/mine_mcal_brush_patterns.py` to `.../v10/archived/mine_mcal_brush_patterns.py`
- The canonical anchor-aware brush path is now exclusively `wowviewer-converter mine-v10-brushes`
- `wow-viewer/scripts/train_v10_stage2_terrain_synth.py` now exists as the first bounded Stage 2 trainer
- The trainer consumes the existing v10 NPZ shard contract directly (no new dataset builder needed)
- It predicts multi-resolution height at 17×17, 65×65, and 257×257 using all available ground-truth signals
- It supports signal-dropout augmentation (default 15%) so the model is robust to missing channels at inference time
- Loss stack: full L1 + 0.5×mid L1 + 0.25×coarse L1 + 0.3×gradient + 0.3×mid_residual + 0.3×detail_res
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\train_v10_stage2_terrain_synth.py` passed
  - `.venv\Scripts\python.exe wow-viewer\scripts\train_v10_stage2_terrain_synth.py output\build-validation\v10-stage1-development-corpus\v10_stage1_manifest.json --output-dir output\ml-training\v10_stage2_smoke --epochs 2 --batch-size 2 --num-workers 0 --device cpu --no-channels-last --signal-dropout 0.1 --max-samples 8` passed
  - smoke output: `output/ml-training/v10_stage2_smoke/checkpoints/last.pt`
  - bounded smoke metrics after 2 epochs: train loss `0.8890`, val loss `2.2534`, val MAE `2.90m`, val RMSE `8.24m`

### Apr 27, 2026 - v10 mixed-corpus curation and first CUDA Stage 2 run started

- Added `wow-viewer/scripts/curate_v10_training_shards.py`
- The curation utility accepts native v10 manifests, legacy v9 tensor-cache manifests, NPZ files, and NPZ directories
- It verifies required training arrays (`minimap_rgb_256`, `height_257`, `height_17`), rejects flat or blank shards by default, ranks by quality, and can cap per dataset bucket for balanced first-pass training
- Updated `wow-viewer/scripts/train_v10_stage2_terrain_synth.py` to consume:
  - native `pm4_path_mask` and `pm4_building_footprint_mask`
  - legacy v9 aliases for `hole_mask_16x16`, `object_mask_precise_257`, `pm4_mask_257`, `liquid_mask_257`, and `liquid_height_257`
- Repaired the UV-managed CUDA training environment with `gillijimproject_refactor/scripts/setup_training_env.ps1 -Backend auto -Recreate`
- Proof:
  - `.venv\Scripts\python.exe -m py_compile wow-viewer\scripts\curate_v10_training_shards.py wow-viewer\scripts\train_v10_stage2_terrain_synth.py` passed
  - curation over `output/ml-training/cache/v9_direct_archive_core_devholdout_plus11927_alphafix_companionfix_20260420/cache/v9_tensor_cache_manifest.json` plus `output/build-validation/v10-stage1-development-corpus/v10_stage1_manifest.json` passed
  - curation output: `output/ml-training/v10_curated/v10_v9all_plus_native_dev_balanced_manifest.json`
  - curation report: `3,945` candidates, `3,240` valid preselection shards, `1,262` selected shards, `705` rejected, `22` dataset buckets
  - rejection reasons: `467` missing required arrays, `238` height range below threshold
  - CUDA training output: `output/ml-training/v10_stage2_balanced_cuda_run1`
  - CUDA run 1 trained for `3` epochs over `1,262` selected shards with `1,072` train and `190` validation samples
  - CUDA run 1 best checkpoint: `output/ml-training/v10_stage2_balanced_cuda_run1/checkpoints/best.pt`
  - CUDA run 1 best metrics after epoch 3: val loss `0.3865`, val MAE `73.88m`, val RMSE `104.48m`
  - CUDA run 2 output: `output/ml-training/v10_stage2_v9cache_native_dev_cuda_run2`
  - CUDA run 2 trained for `10` epochs over the same `1,262` selected shards with `1,072` train and `190` validation samples
  - CUDA run 2 best checkpoint: `output/ml-training/v10_stage2_v9cache_native_dev_cuda_run2/checkpoints/best.pt`
  - CUDA run 2 best metrics at epoch `6`: val loss `0.3438`, val MAE `70.38m`, val RMSE `100.47m`
- Boundary:
  - broad all-version coverage currently comes from the legacy v9 cache, not native v10 extraction for every client root
  - native richer v10 signals in this first mixed manifest are limited to the development-map Stage 1 corpus (`41` selected native v10 shards, `11` with MCAL/MCLY texture signals, `41` with PM4 masks)
  - this is a first started training run, not a converged model

## Open Boundaries

- No validated broad-corpus MCAL brush-stroke vocabulary run exists yet beyond the bounded development Stage 1 proof.
- No validated broad-corpus minimap-to-MCLY classifier or chunk-grid run exists yet beyond the `11` currently labelable development shards.
- MCLY dictionary biome tags are heuristic and should be replaced or validated by the planned minimap-to-biome/palette classifier.
- Stage 2 trainer now has mixed-corpus CUDA runs from the proven all-version v9 cache plus native v10 development shards; longer-running CUDA training, broader native v10 corpora, and production-grade experiment management remain open.
- All-version broad training intentionally depends on legacy v9 cache shards for now because that path already harvested the staged archive/client roots correctly; native v10 archive/client-root extraction remains open beyond the development-map proof.
- The world-viewer path is still mid-migration and should not be described as final runtime parity.

## Recommended Next Slice

1. Run `decompose-minimap-tilesets` against raw/source minimap tiles with a broader v3 pattern library, then inspect residual/confidence previews for obvious false positives.
2. Add the first iterative subtraction/alpha-candidate pass only after the per-cell candidate ranking is visually plausible on raw minimaps.
3. Evaluate whether decomposition outputs should become compact Stage 2 conditioning channels before relaxing dataset caps or widening the model.
