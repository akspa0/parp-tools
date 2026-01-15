# WoW Terrain VLM Training Guide

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

Use the C# tool `vlm-export` to extract data from your WoW client.

### Syntax
```bash
cd src/WoWMapConverter
dotnet run --project WoWMapConverter.Cli -- vlm-export --client "C:\Path\To\WoW" --map "MapName" --out "J:\vlm_output\MapName"
```

### Output
The tool generates:
- **`manifest.json`**: List of all tiles.
- **`stitched/`**: Full map atlases for Minimap, Shadows, and Alpha layers (useful for verification).
- **`dataset/`**: Individual tile JSON files containing terrain metadata.
- **`images/`**: Source images (minimap tiles, shadow quilts, alpha masks).

### Curation
Before training, combine separate tile JSONs into a single `train.jsonl` file formatted for the VLM.

```bash
python scripts/vlm_curate.py --input "J:\vlm_output\MapName" --output "J:\vlm_output\MapName_curated"
```

## 3. Training

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
