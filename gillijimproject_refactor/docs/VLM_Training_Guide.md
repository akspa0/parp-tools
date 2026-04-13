# WoW Terrain ML Dataset Training Guide

ML Dataset is now the preferred user-facing name for this terrain supervision corpus. Existing `VLM` type and file names in code are legacy implementation names and compatibility seams.

For the specific `MCSH`/missing-object problem statement, also read [gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md](gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md).

This guide details the process of training a Vision Language Model (Qwen2-VL via Unsloth) to understand and reconstruct World of Warcraft terrain data.

## Prerequisites

- **NVIDIA GPU**: RTX 30xx/40xx recommended (8GB+ VRAM).
- **Windows**: (Linux works too but this guide focuses on Windows).
- **CUDA**: Version 12.1 or higher (Unsloth supports up to 12.4/13.0).
- **Python**: 3.10 or 3.11.

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

### Fixed-Client Corpus Export

The checked-in wrapper for the current machine-local clients is:

```bash
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1
```

It uses [gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json](gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json) and currently targets the fixed `3.0.1.8303`, `3.3.5.12340`, and `4.0.0.11927` roots with a narrow checked-in subset: `Northrend` plus `PVPZone01` through `PVPZone04` from `3.0.1.8303`, `Azeroth` from `3.3.5.12340`, and `LostIsles` from `4.0.0.11927`. The wrapper writes dataset roots under `output/ml-corpus/` and then runs `ml-harvest` for each exported map.

### Corpus Truth Audit For V7.4 Curation

Use the wow-viewer audit command before treating a corpus as V7.4-ready:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-audit-signals --dataset-root i:/parp/parp-tools/output/ml-corpus/301_8303/Northrend --output i:/parp/parp-tools/output/build-validation/ml-audit/northrend_signal_audit.json --limit 32
```

The audit currently reports:

- dedupe groups
- concept clusters
- per-tile retention recommendation (`canonical` or `review-duplicate`)
- liquid semantic class (`visible-surface`, `below-terrain-likely`, `uncertain`, `none`)
- signal coverage counts for minimap, heights, alpha, objects, liquids, and `no_liquid_minimap`

This is the first gate toward the V7.4 canvas-aware curation flow. Do not treat it as final semantic truth yet; it is a bounded audit layer meant to identify duplicate density and suspect liquid supervision before retraining.

### Brush-Imprint Harvest For WoWEdit Archaeology

The next deeper dataset seam is not tile dedupe alone. It is harvesting repeated patch and patch-group terrain imprints that likely reflect hidden 3D brush usage in the original WoWEdit workflow.

Use the wow-viewer harvester:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/output/ml-corpus/400_12304/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/development_40012304 --limit 6 --write-previews
```

Current behavior:

- treats each tile as `16x16` chunks
- treats each chunk as `16x16` patch cells
- scores terrain-shape change on the `257x257` global height lattice
- groups adjacent high-score cells into candidate brush-imprint regions
- writes a separate dataset surface under `brush_imprints/` for later clustering, retrieval, or separate-model work
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
