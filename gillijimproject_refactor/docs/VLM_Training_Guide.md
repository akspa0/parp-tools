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
