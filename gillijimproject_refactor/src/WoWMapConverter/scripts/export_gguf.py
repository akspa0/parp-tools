"""
WoW Terrain VLM - GGUF Export Script (Manual Pipeline)
======================================================
1. Merges LoRA adapters into base model (16-bit).
2. Converts HuggingFace model to GGUF (f16) using llama.cpp script.
3. Quantizes to q4_k_m using local llama-quantize executable.

Usage:
    python export_gguf.py
"""

import os
import sys
import requests
import subprocess
from pathlib import Path
from unsloth import FastVisionModel

# ============================================================================
# Configuration
# ============================================================================

LORA_PATH = r"j:\vlm_output\wow_terrain_vlm\lora"
OUTPUT_BASE = r"j:\vlm_output\wow_terrain_vlm"
MERGED_DIR = os.path.join(OUTPUT_BASE, "merged_16bit")
GGUF_DIR = os.path.join(OUTPUT_BASE, "gguf")
GGUF_F16 = os.path.join(GGUF_DIR, "model-f16.gguf")
GGUF_Q4 = os.path.join(GGUF_DIR, "model-q4_k_m.gguf")

# User provided llama.cpp directory
LLAMA_CPP_DIR = r"J:\vlm_output\llamacpp"
QUANTIZE_EXE = os.path.join(LLAMA_CPP_DIR, "llama-quantize.exe")

BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("GGUF Export Pipeline")
    print("="*60)
    
    # Check dependencies
    print("Ensuring latest 'gguf' package is installed from source (FORCE REINSTALL)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-cache-dir", "git+https://github.com/ggerganov/llama.cpp.git@master#subdirectory=gguf-py", "protobuf"])
    except subprocess.CalledProcessError:
        print("Warning: Git install failed. Falling back to pypi upgrade.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gguf", "protobuf"])

    # Ensure output directories exist
    os.makedirs(MERGED_DIR, exist_ok=True)
    os.makedirs(GGUF_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # 1. Merge and Save 16-bit Model
    # ------------------------------------------------------------------------
    print(f"\n[1/3] Merging LoRA adapters into base model...")
    print(f"Loading base: {BASE_MODEL}")
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
    )
    
    print(f"Loading LoRA from: {LORA_PATH}")
    # Explicitly load adapter
    model.load_adapter(LORA_PATH)
    
    # Verify/Force PEFT wrapping
    from peft import PeftModel
    if not isinstance(model, PeftModel):
        print("Model object is not PEFT instance. Wrappping manually...")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    
    print(f"Saving merged 16-bit model to: {MERGED_DIR}")
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print("Merge complete.")

    # ------------------------------------------------------------------------
    # 2. Convert HF to GGUF F16
    # ------------------------------------------------------------------------
    print(f"\n[2/3] Converting to GGUF (F16)...")
    
    # Download conversion script if missing
    convert_script = "convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print("Downloading convert_hf_to_gguf.py from llama.cpp repo...")
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
        try:
            r = requests.get(url)
            with open(convert_script, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Error downloading script: {e}")
            return

    # Run conversion
    cmd = [sys.executable, convert_script, MERGED_DIR, "--outfile", GGUF_F16, "--outtype", "f16"]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"F16 GGUF saved to: {GGUF_F16}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        return

    # ------------------------------------------------------------------------
    # 3. Quantize to Q4_K_M
    # ------------------------------------------------------------------------
    print(f"\n[3/3] Quantizing to Q4_K_M...")
    
    if not os.path.exists(QUANTIZE_EXE):
        print(f"ERROR: llama-quantize.exe not found at {QUANTIZE_EXE}")
        print("Please check the path.")
        return

    cmd = [QUANTIZE_EXE, GGUF_F16, GGUF_Q4, "q4_k_m"]
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print(f"Quantization complete!")
        print(f"Final model: {GGUF_Q4}")
    except subprocess.CalledProcessError as e:
        print(f"Quantization failed: {e}")
        return
        
    print("\nSUCCESS! You can now use the model in llama.cpp / Ollama.")

if __name__ == "__main__":
    main()
