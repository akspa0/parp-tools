"""
WoW Terrain VLM Local Training Script
======================================
Based on official Unsloth Qwen3-VL notebook.

Usage:
    python train_local.py
"""

import torch
from pathlib import Path
import json

# ============================================================================
# Configuration
# ============================================================================

TRAIN_FILE = r"j:\vlm_output\053_curated_full\train.jsonl"
OUTPUT_DIR = r"j:\vlm_output\wow_terrain_vlm"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 100
LEARNING_RATE = 2e-4

MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*60)
    print("WoW Terrain VLM Training")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Load Unsloth
    print("Loading Unsloth...")
    from unsloth import FastVisionModel
    
    # Load model
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    # Add LoRA
    print("Adding LoRA adapters...")
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Load dataset as LIST (not HuggingFace Dataset!)
    print(f"Loading dataset: {TRAIN_FILE}")
    from PIL import Image
    
    converted_dataset = []
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            # Load the actual image
            image_path = sample.get("images", [None])[0]
            if image_path and Path(image_path).exists():
                try:
                    image = Image.open(image_path).convert("RGB")
                    # Reconstruct in the format Unsloth expects
                    messages = sample.get("messages", [])
                    # Inject actual image into the content
                    for msg in messages:
                        if msg["role"] == "user":
                            for content in msg.get("content", []):
                                if content.get("type") == "image":
                                    content["image"] = image
                    converted_dataset.append({"messages": messages})
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}")
    
    print(f"Loaded {len(converted_dataset)} training samples")
    
    # Setup trainer per official notebook
    print("Setting up trainer...")
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    
    FastVisionModel.for_training(model)  # Enable for training!
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
            # MUST have for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=1,  # Avoid crashing issue
            max_seq_length=MAX_SEQ_LENGTH,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )
    
    # Record memory
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    print(f"GPU memory: {start_gpu_memory} GB / {max_memory} GB")
    
    # Train
    print()
    print("="*60)
    print("Starting training...")
    print("="*60)
    
    trainer_stats = trainer.train()
    
    # Stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    print(f"\nTraining complete in {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"Peak memory: {used_memory} GB ({used_memory_for_lora} GB for LoRA)")
    
    # Save
    lora_path = Path(OUTPUT_DIR) / "lora"
    print(f"Saving to: {lora_path}")
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    
    # Save GGUF (optional)
    try:
        print("\n" + "="*60)
        save_gguf = input("Save to GGUF for local inference? (y/n): ").lower() == 'y'
        if save_gguf:
            gguf_path = Path(OUTPUT_DIR) / "gguf"
            print(f"Saving GGUF (q4_k_m) to: {gguf_path}")
            model.save_pretrained_gguf(str(gguf_path), tokenizer, quantization_method="q4_k_m")
            print("GGUF saved successfully.")
    except Exception as e:
        print(f"Error saving GGUF: {e}")
        print("You can try running 'export_gguf.py' separately.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
