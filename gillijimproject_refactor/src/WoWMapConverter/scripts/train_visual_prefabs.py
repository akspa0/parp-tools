"""
WoW Visual Prefab LoRA Trainer
==============================
Fine-tunes Qwen3-VL to recognize specific terrain prefabs (e.g. "HELP ME" easter egg).
"""

import torch
from pathlib import Path
import json

# Configuration
TRAIN_FILE = r"j:\wowDev\parp-tools\gillijimproject_refactor\height_regression.jsonl"
OUTPUT_DIR = r"j:\vlm_output\wow_height_regression_lora"
MAX_SEQ_LENGTH = 2048 # Need longer context for 145 float tokens + image
MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit" # Use latest Qwen2.5-VL if available, else Qwen2-VL

def main():
    print("="*60)
    print("WoW Visual Prefab LoRA Trainer")
    print("Dataset:", TRAIN_FILE)
    print("="*60)
    
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected!")
        return

    from unsloth import FastVisionModel
    
    # 1. Load Model
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    
    # 2. Add LoRA
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
    
    # 3. Load Dataset
    print("Loading dataset...")
    # Modified load logic for local files
    from PIL import Image
    converted_dataset = []
    
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            messages = sample["messages"]
            
            # Find and load image
            img_path = None
            for msg in messages:
                if msg["role"] == "user":
                    for content in msg["content"]:
                        if content["type"] == "image":
                            img_path = content["image"]
                            try:
                                # Replace path with PIL Image object for Unsloth
                                content["image"] = Image.open(img_path).convert("RGB")
                            except Exception as e:
                                print(f"Error loading {img_path}: {e}")
            
            if img_path:
                converted_dataset.append({"messages": messages})

    print(f"Loaded {len(converted_dataset)} samples.")
    if len(converted_dataset) == 0:
        print("No valid samples found. Exiting.")
        return

    # 4. Train
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    
    training_args = SFTConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = 60, # Quick fine-tune for 2 samples
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        save_steps = 20,
        warmup_steps = 10,
        optim = "adamw_8bit",
        seed = 3407,
        dataset_text_field = "", # Unsloth vision handles this
        remove_unused_columns = False,
    )
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = FastVisionModel.get_data_collator(tokenizer),
        train_dataset = converted_dataset,
        args = training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving LoRA to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
