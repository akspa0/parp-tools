# WoW Terrain ML Dataset Training Guide

ML Dataset is now the preferred user-facing name for this terrain supervision corpus. Existing `VLM` type and file names in code are legacy implementation names and compatibility seams.

For the specific `MCSH`/missing-object problem statement, also read [gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md](gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md).

This guide details the process of training a Vision Language Model (Qwen2-VL via Unsloth) to understand and reconstruct World of Warcraft terrain data.

## Reality Grounding

Before treating any training run as meaningful, read `docs/ML_DATASET_GROUNDING.md`.

The active claim for this project is not that a GAN can invent plausible terrain. The active claim is that the model is trained against supervision harvested from real client data and deterministic cleanup or analysis passes over that real data.

Practical rules:

- treat `datasets/` as the authoritative harvested corpus surface
- treat GAN as a training-time refinement objective, not a dataset source
- treat the brush channel as active grounded supervision
- treat the prefab channel as experimental and deferred from the trusted training contract

## Model Lines

This repo currently has two different model stories that should be documented separately.

### V9 Native Terrain Line

- active native terrain reconstruction path
- broad main corpus comes from the `wow-viewer` direct shared-reader cache flow
- development-map compatibility cache still supplies the current PM4-rich and object-rich supervision seam
- current best-understood branch uses `train_v9_optimized.py` with a separate non-overlapping development holdout for checkpoint selection
- documented in `docs/V9_Native_Terrain_Training_Guide.md`
- direct cache setup and wrapper boundaries documented in `../../wow-viewer/docs/validation/direct-v9-training-setup.md`

### V7.5.1 Terrain Line

- active grounded terrain-regression path
- multichannel input over harvested dataset roots
- documented in `docs/v75-model-architecture-guide.md`

### V7.6 Paired Reconstruction Line

- separate image-to-height+albedo branch
- meant to learn both geometry and a cleaner terrain-material surface from image input
- should emit a structured predicted dataset rather than loose files
- documented in `docs/v76-model-architecture-guide.md`
- output package contract documented in `docs/v76-output-dataset-spec.md`

## Prerequisites

- **NVIDIA GPU**: RTX 30xx/40xx recommended (8GB+ VRAM).
- **Windows**: (Linux works too but this guide focuses on Windows).
- **CUDA**: Version 12.1 or higher (Unsloth supports up to 12.4/13.0).
- **Python**: 3.10 or 3.11.

For the current native `v9` line on Windows CUDA, prefer the dedicated bootstrap documented here and in `scripts/setup_training_env.ps1`. The current bootstrap also installs `triton-windows` so `torch.compile` remains available in the optimized trainer.

## 1. Environment Setup

We use **Unsloth** for efficient 4-bit LoRA finetuning.

### Create Virtual Environment
```bash
cd src/WoWMapConverter/scripts
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies
Follow the official [Unsloth Installation Guide](https://github.com/unslothai/unsloth?tab=readme-ov-file#installation-instructions) or use the provided setup (condensed):

```bash
# Install PyTorch with CUDA support (check official site for exact command matching your CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth and extras
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```
*Note: Dependencies in `vlm_curate.py` or headers of scripts might list specific versions.*

## 2. Generating the Dataset

Use the C# tool `ml-export` to extract data from your WoW client.

The exported dataset supports three different learning problems:

1. terrain reconstruction: minimap plus WDL plus known-loss channels -> height
2. texture decomposition: minimap -> terrain texture palette plus alpha masks
3. shadow-scar object recovery: minimap plus `MCSH` shadow evidence plus surviving placements -> missing-object candidates

For the separate paired-output V7.6 branch, the harvested dataset can also be converted into cached paired tensors where:

- the source minimap becomes the model input
- the global heightmap becomes the geometry target
- a synthesized terrain albedo becomes the appearance target

That V7.6 cache-and-train path is documented separately because it is not the same contract as the active V7.5.1 multichannel terrain line.

### Fixed-Client Corpus Export

The checked-in wrapper for the current machine-local clients is:

```bash
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1
```

It uses [gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json](gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json) and currently targets the fixed `0.7.0.3694`, `3.0.1.8303`, `3.3.5.12340`, and `4.0.0.11927` roots plus the checked-in `original_development` split-root seam. The wrapper writes dataset roots under `datasets/<label>/<map>/`, passes `--minimap-root` when a client config needs a separate minimap source, and then runs `ml-harvest` for each exported map.

Each harvested dataset root now also contains:

- `ml_dataset_manifest.json`
- `metadata.jsonl`
- `dataset_info.json`

That gives each dataset a root-level HF-friendly metadata surface in addition to the legacy `dataset/*.json` tile payloads.

### Corpus Truth Audit For V7.5 Curation

Use the wow-viewer audit command before treating a corpus as V7.5-ready:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-audit-signals --dataset-root i:/parp/parp-tools/datasets/3_0_1_8303/Northrend --output i:/parp/parp-tools/output/build-validation/ml-audit/northrend_signal_audit.json --limit 32
```

The audit currently reports:

- dedupe groups
- concept clusters
- per-tile retention recommendation (`canonical` or `review-duplicate`)
- liquid semantic class (`visible-surface`, `below-terrain-likely`, `uncertain`, `none`)
- signal coverage counts for minimap, heights, alpha, objects, liquids, and `no_liquid_minimap`

This is the first gate toward the V7.5 canvas-aware curation flow. Do not treat it as final semantic truth yet; it is a bounded audit layer meant to identify duplicate density and suspect liquid supervision before retraining.

### Terrain-Only Minimap Contract In V7.5

V7.5 keeps the same `13`-channel tensor contract, but it changes which RGB minimap surface is preferred.

Current precedence in `train_v7.py` and `infer_v7.py`:

1. `terrain_only_minimap`
2. `no_object_minimap`
3. `no_mccv_minimap`
4. raw exported `image`

`terrain_only_minimap` is generated during dataset export when enough auxiliary masks exist. It starts from the no-MCCV-cleaned minimap when available, then replaces the strongest non-mesh contaminant regions with chunk texture rebake and nearest-chunk base-texture fallback:

- object visibility masks
- PM4 masks
- liquid masks
- stitched alpha masks

Exported `MCSH` shadow maps remain useful as diagnostics, but they are not currently removed as terrain contamination in the active `terrain_only_minimap` path.

This is the main semantic bump from V7.4 to V7.5. The model shape stays stable, but the preferred RGB evidence is more terrain-focused and less polluted by baked lighting, blend overlays, and object occlusion.

### Brush-Imprint Harvest For WoWEdit Archaeology

The next deeper dataset seam is not tile dedupe alone. It is harvesting repeated patch and patch-group terrain imprints that likely reflect hidden 3D brush usage in the original WoWEdit workflow.

Use the wow-viewer harvester:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/datasets/original_development/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/original_development --limit 6 --write-previews
```

Current behavior:

- treats each tile as `16x16` chunks
- treats each chunk as `16x16` patch cells
- scores terrain-shape change on the `257x257` global height lattice
- groups adjacent high-score cells into candidate brush-imprint regions
- writes a separate dataset surface under `brush_imprints/` for later clustering, retrieval, or separate-model work
- assigns deterministic brush archetype IDs and corpus-level archetype summaries under `brush_imprints/brush_archetype_manifest.json` and `brush_imprints/archetypes/*.json`
- also writes a tile-level `brush_mask_path` that can be consumed as a first conditioning channel by the terrain trainer

Important boundary:

- this is first-pass brush-imprint harvesting, not final brush dedupe
- the goal is to isolate the imprints into their own dataset first, then analyze them separately

### First Brush Channel In `train_v7.py`

`train_v7.py` now has a first brush-imprint conditioning seam.

Current behavior:

- `MODEL_INPUT_CHANNELS` is now `13`
- the dataset loader looks for `brush_imprints/brush_imprint_manifest.json` under each dataset root
- when a tile-level `brush_mask_path` exists, it is loaded as an extra binary input channel
- the current terrain model therefore sees one coarse brush-imprint mask, but not yet grouped brush identity or a learned brush embedding

This is intentionally the smallest safe integration step. It proves the terrain model can consume a brush-derived context channel while the separate brush dataset and future brush-specific model are still being built.

### Prefab Status

Prefab harvesting is not part of the current trusted training contract.

The repo still contains prefab tooling and review surfaces, but until that path is validated to the same standard as the brush harvest it should be treated as experimental dataset research rather than active supervision.

If you are explaining what grounds the current terrain model in reality, point to the exported tile corpus and the brush-imprint pass, not the prefab outputs.

### V7.6 Output Packaging Goal

When V7.6 is used on arbitrary image inputs, the result should be a structured predicted dataset, not a loose folder of PNGs and OBJ files.

The intent is to mirror the strengths of the harvested input-dataset packaging:

- stable manifests
- per-sample JSON records
- explicit source provenance
- explicit model provenance

That predicted-output contract is defined in `docs/v76-output-dataset-spec.md`.

### Geometry-First Default Training Strategy

The trainer defaults are now biased toward a long geometry-first phase before GAN activation.

Current default behavior in `train_v7.py`:

- `--adversarial-scale 0.20`
- `--start-gan-epoch 101`
- `--disc-every 2`
- `--early-stop-start-epoch 101`
- `--lr-plateau-patience 8`

This means the model can spend roughly the first `100` epochs learning large-scale terrain structure before adversarial sharpening turns on, and early-stop patience will not count during that warmup period.

Use this longer geometry-first schedule as the new default baseline before concluding that the architecture itself has topped out.

If a run reports a negative validation loss, treat that checkpoint as invalid numeric output rather than as a real improvement. The trainer now keeps the forward pass under AMP but computes the structural loss stack in float32 and refuses to treat negative or non-finite validation loss as a valid `best.pt` candidate.

Validation currently influences training in several concrete ways:

- it chooses `best.pt`
- it drives `ReduceLROnPlateau`
- it drives early-stop patience once that counter is allowed to count
- it now arms best-triggered GAN refinement bursts when `--gan-burst-after-best` is enabled

Important boundary:

- the validation split is a real held-out slice, but it is still drawn from the same audited corpus family
- that means it is useful feedback for overfitting and checkpoint selection, but it is not a full guarantee of performance on truly unseen minimaps or unseen map families

To make preview monitoring less misleading, each epoch now renders a mixed validation preview set by default:

- `2` fixed high-signal validation tiles
- `2` random held-out validation tiles

Each preview epoch also writes a small JSON sidecar next to the PNGs listing the exact tile labels shown in that preview grid.

The preview output now also includes `val_epoch_XXXX_context.png`, which is specifically for object-mask validation. Its columns are:

- raw minimap
- minimap with object-mask overlay
- minimap with object-mask regions blanked out as a diagnostic view
- raw object mask
- liquid mask
- brush mask

This does not mean the model is currently zeroing the minimap input itself during training. The active training path still feeds the raw minimap plus a separate object-mask channel. The context preview is there so you can visually confirm what object-corrupted regions the model was told about.

If you want to change the mix, use:

- `--static-preview-count`
- `--random-preview-count`

### Best-Triggered GAN Refinement Bursts

The active scheduling strategy is now best-triggered refinement instead of a fixed late-epoch GAN phase.

Current practical training guidance:

- treat `100` epochs as the upper bound unless a run is still making real validation progress late
- let early stopping count from the start for best-triggered GAN runs, because validation is now part of the control loop instead of being artificially gated behind a long warmup

New primary control:

- `--gan-burst-after-best`: after any new best checkpoint, automatically arm GAN for this many subsequent epochs

This gives a simpler pattern:

- geometry-only training runs first
- whenever validation sets a new best checkpoint, GAN turns on automatically for a short refinement burst
- if GAN-assisted epochs continue improving validation, the burst rearms again

This directly matches the practical goal: use adversarial pressure only when the model has just demonstrated a meaningful improvement, rather than guessing a fixed calendar epoch to turn GAN on.

Example best-triggered command:

```powershell
$audits = Get-ChildItem '.\output\build-validation\ml-audit\trusted' -Filter '*_signal_audit.json' -File |
	ForEach-Object {
		$j = Get-Content $_.FullName -Raw | ConvertFrom-Json
		[PSCustomObject]@{
			Name = $_.BaseName
			DatasetRoot = [string]$j.dataset_root
			Tiles = [int]$j.tile_count
			Local = [int]$j.coverage.tiles_with_local_heightmap
			Global = [int]$j.coverage.tiles_with_global_heightmap
		}
	}

$eligible = $audits | Where-Object {
	$_.Local -eq $_.Tiles -and
	$_.Global -eq $_.Tiles -and
	$_.Name -ne '301_8303_Kalimdor_signal_audit' -and
	$_.Name -notmatch '__UNTRUSTED_DO_NOT_USE'
}

$outDir = '.\output\ml-training\v7_5_terrain_only_bestburst_20260413'
New-Item -ItemType Directory -Force $outDir | Out-Null

$args = @(
	'.\gillijimproject_refactor\src\WoWMapConverter\scripts\train_v7.py',
	'--profile', 'manual',
	'--epochs', '100',
	'--patience', '12',
	'--output-dir', $outDir,
	'--learning-rate', '8e-5',
	'--disc-learning-rate', '5e-5',
	'--amp-dtype', 'auto',
	'--train-workers', '4',
	'--val-workers', '2',
	'--log-every', '5',
	'--gan-burst-after-best', '2'
)

foreach ($entry in $eligible) { $args += @('--dataset-root', $entry.DatasetRoot) }
C:\Users\akspa\anaconda3\python.exe @args
```

That example means:

- GAN stays off until a best checkpoint appears
- after each new best checkpoint, GAN is armed for the next `2` epochs
- if one of those GAN-assisted epochs also becomes the new best, the `2`-epoch burst is armed again

The older `--start-gan-epoch`, `--gan-cycle-length`, `--gan-cycle-on-epochs`, and `--gan-cooldown-after-best` controls still exist as fallback schedule tools, but they are no longer the preferred strategy.

In practice, if a run already looked good around epoch `30..40` and later spent many epochs with no meaningful validation improvement, do not keep stretching it to `140`. Keep the ceiling near `100` and let patience stop it earlier.

### Syntax
```bash
cd src/WoWMapConverter
dotnet run --project WoWMapConverter.Cli -- ml-export --client "C:\Path\To\WoW" --map "MapName" --out "J:\ml_output\MapName"
```

### Output
The tool generates:
- **`manifest.json`**: List of all tiles.
- **`stitched/`**: Full map atlases for Minimap, Shadows, and Alpha layers (useful for verification).
- **`dataset/`**: Individual tile JSON files containing terrain metadata.
- **`images/`**: Source images (minimap tiles, shadow quilts, alpha masks).

### How to Use the Supervision

- **Terrain model inputs**: `image`, `normalmap`, `wdl_heights`, `liquid_mask`, `objects`, `height_min`, `height_max`
- **Terrain model targets**: `heightmap_local`, `heightmap_global`
- **Texture model inputs**: `image`, optional palette context from `terrain_data.textures` or decoded `tilesets/`
- **Texture model targets**: `alpha_masks`, `alpha_atlas`, and chunk-layer texture assignments from `terrain_data.chunk_layers`
- **Shadow-scar model inputs**: `image`, `shadow_maps`, raw `shadow_bits`, `shadow_analysis`, and surviving `objects`
- **Shadow-scar model targets**: unexplained shadow regions, missing-object candidate masks, and later recovered placement hypotheses

Do not assume one model should learn both jobs well. The current direction is to
keep geometry reconstruction, texture-layer decomposition, and shadow-scar
object recovery as separate models that share the same exported dataset root.

For the development map specifically, the available terrain-texture palette in
the `4.0.0.11927` client is relatively limited. That makes it a useful first
correlation target for the texture-decomposition model because the minimap-to-
tileset mapping problem is narrower than a full cross-expansion palette.

### Curation
Before training, combine separate tile JSONs into a single `train.jsonl` file formatted for the VLM.

```bash
python scripts/vlm_curate.py --input "J:\vlm_output\MapName" --output "J:\vlm_output\MapName_curated"
```

## 3. Training

The training path is now split by problem instead of forcing one model to learn
both terrain geometry and texture-layer decomposition.

### Terrain Model

Use `train_v7.py` for:

- minimap + WDL + known-loss channels -> terrain heights
- outputs: global height, local height, bounds

#### Training Environment Bootstrap (uv)

Use the dedicated uv bootstrap script before training so the venv gets a
hardware-matched torch build instead of falling back to whatever is already
installed in a random workspace venv.

PowerShell:

```powershell
./gillijimproject_refactor/scripts/setup_training_env.ps1 -Backend auto -Recreate
```

Bash:

```bash
./gillijimproject_refactor/scripts/setup_training_env.sh --backend auto --recreate
```

On Windows CUDA environments, the bootstrap scripts now also install `triton-windows`
so `torch.compile` can use the Triton-backed Inductor path instead of failing at
runtime on a missing Triton module.

Then run training through that dedicated interpreter:

```powershell
i:/parp/parp-tools/gillijimproject_refactor/.venv-train/Scripts/python.exe `
	i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py `
	--profile development-map
```

`train_v7.py` now refuses implicit CPU fallback when CUDA is unavailable.
If you intentionally want a CPU-only debug run, pass `--allow-cpu` explicitly.

#### V7.3 Performance Profile (Apr 12, 2026)

`train_v7.py` now has live training telemetry and Tensor Core-oriented defaults:

- live tqdm updates include rolling generator/discriminator loss, learning rate, and VRAM
- per-epoch summary now includes throughput (`steps/s`, `samples/s`)
- CUDA path defaults to AMP + TF32 + cuDNN benchmark (`--no-amp`, `--no-tf32`, `--no-cudnn-benchmark` disable each)
- AMP dtype is selectable (`--amp-dtype auto|bfloat16|float16`), with `auto` preferring `bfloat16` on supported GPUs
- default loader profile is now `train_workers=4`, `val_workers=2`, `prefetch_factor=2`

Measured benchmark on `NVIDIA GeForce RTX 4070 Ti SUPER` over a one-epoch `Northrend` subset (`limit=640`, batch size 4):

- baseline (AMP off, TF32 off): `1.47 steps/s`, `72.15s`
- Tensor Core path (AMP auto + TF32 on): `1.60 steps/s`, `69.10s`
- measured speedup: `+8.8%`

Frequency-loss FFT now runs in float32 even under AMP to avoid NaN instability.

#### Recommended Full Trusted-Corpus Resume Command

```powershell
$base = "i:/parp/parp-tools/output/ml-corpus"
$roots = Get-ChildItem $base -Directory |
	ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
	Where-Object { $_.FullName -notmatch "__UNTRUSTED_DO_NOT_USE" -and (Test-Path (Join-Path $_.FullName "dataset")) }

$args = @(
	"i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py",
	"--profile", "manual",
	"--epochs", "10",
	"--resume", "i:/parp/parp-tools/output/ml-training/v7_3_all_trusted_maps_20260411_235624/checkpoint.pt",
	"--output-dir", "i:/parp/parp-tools/output/ml-training/v7_3_all_trusted_maps_20260411_235624",
	"--amp-dtype", "auto",
	"--train-workers", "4",
	"--val-workers", "2",
	"--log-every", "5"
)
foreach ($r in $roots) { $args += @("--dataset-root", $r.FullName) }
python @args
```

#### Fine-Tune Recipe When Validation Drifts After a Strong Early Best

Observed Apr 12 run behavior on the full trusted corpus:

- best remained at epoch 5 (`val=0.0493`)
- later epochs drifted (`epoch 8: 0.1723`, `epoch 9: 0.1682`, `epoch 10: 0.1758`)
- adversarial pressure and discriminator confidence kept rising while val did not recover

Use the fine-tune controls added to `train_v7.py` to continue from `best.pt` with gentler GAN pressure:

- `--adversarial-scale`: down-weight adversarial contribution
- `--start-gan-epoch`: delay GAN objective for a few epochs
- `--disc-every`: update discriminator less frequently
- `--disc-learning-rate`: lower discriminator learning rate
- resume now restores optimizer/discriminator/scheduler/scaler states unless `--no-resume-optimizer` is passed

Suggested command for the next continuation:

```powershell
$base = "i:/parp/parp-tools/output/ml-corpus"
$roots = Get-ChildItem $base -Directory |
	ForEach-Object { Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue } |
	Where-Object { $_.FullName -notmatch "__UNTRUSTED_DO_NOT_USE" -and (Test-Path (Join-Path $_.FullName "dataset")) }

$out = "i:/parp/parp-tools/output/ml-training/v7_3_all_trusted_maps_finetune_20260412"
$args = @(
	"i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/scripts/train_v7.py",
	"--profile", "manual",
	"--epochs", "14",
	"--resume", "i:/parp/parp-tools/output/ml-training/v7_3_all_trusted_maps_20260411_235624/best.pt",
	"--output-dir", $out,
	"--learning-rate", "5e-5",
	"--disc-learning-rate", "5e-5",
	"--adversarial-scale", "0.35",
	"--start-gan-epoch", "8",
	"--disc-every", "2",
	"--amp-dtype", "auto",
	"--train-workers", "4",
	"--val-workers", "2",
	"--log-every", "5"
)
foreach ($r in $roots) { $args += @("--dataset-root", $r.FullName) }
python @args
```

Keep the old run unchanged for traceability; compare this new run's best checkpoint against the prior `0.0493` baseline.

### Texture Model

Use `train_texture_v1.py` for:

- minimap -> three overlay alpha masks
- minimap -> chunk-slot texture classes for up to four terrain layers

This is the first separate texture-decomposition seam. It is aimed at the
limited palette available in the `4.0.0.11927` development-family terrain data
rather than claiming broad all-expansion closure.

### Shadow-Scar Object Recovery Model

This should be treated as a third model family, not folded back into either the
terrain-height or texture-layer path.

The goal is not to predict all objects from rendered minimaps alone. The goal is
to explain the part of `MCSH` shadow evidence that current placements do not
already explain. That gives a narrower and more defensible target:

- derive current object-footprint masks from `terrain_data.objects`
- compare them against stitched/raw `MCSH` shadows
- isolate persistent unexplained shadow regions as `shadow scar` candidates
- train a model to map minimap appearance plus shadow evidence plus surviving
	object context into missing-object family labels and restored placement
	hypotheses

This is the right place to recover objects that were deleted or moved after the
terrain/shadow state we still observe in the minimap-era data.

The training script uses LoRA (Low-Rank Adaptation) to finetune Qwen2-VL-8B.

### Run Training
```bash
python scripts/train_local.py
```

### Configuration (`train_local.py`)
Edit the top of the file to point to your data:
- `TRAIN_FILE`: Path to your `train.jsonl`.
- `OUTPUT_DIR`: Where to save the model.
- `MAX_STEPS`: Number of training steps (default 60-100 for small tests).

**Key Settings:**
- `dataset_num_proc=1`: Must be set to 1 on Windows to avoid multiprocessing crashes.
- `processing_class`: Must be passed to `SFTTrainer` (fixes VLM detection).

## 4. Export to GGUF

To use the model in **llama.cpp** or **Ollama**, you must export it to GGUF format.

### Option A: Interactive (Post-Training)
`train_local.py` will ask at the end if you want to save to GGUF. Type `y`.

### Option B: Manual Script (Windows Friendly)
If the automated export fails (common on Windows due to build tools), use the dedicated script:

```bash
python scripts/export_gguf.py
```

**Configuration (`export_gguf.py`)**:
- `LORA_PATH`: Path to your trained `lora` directory.
- `LLAMA_CPP_DIR`: Path to your local folder containing `llama-quantize.exe`.

**Pipeline**:
1. Merges LoRA adapters into the base model (16-bit).
2. Downloads `convert_hf_to_gguf.py` from llama.cpp.
3. Converts to F16 GGUF.
4. Quantizes to `q4_k_m` (best balance of speed/quality).

## Troubleshooting

- **"MistralTokenizerType" Error**: Run `pip install --upgrade --force-reinstall git+https://github.com/ggerganov/llama.cpp.git@master#subdirectory=gguf-py` to sync your `gguf` library with the conversion script.
- **Memory Issues**: Reduce `BATCH_SIZE` in `train_local.py`.
- **Large Map Crashes**: The dataset exporter automatically skips stitching images > 16k pixels.
