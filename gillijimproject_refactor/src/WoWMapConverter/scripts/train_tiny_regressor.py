import json
import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from tqdm import tqdm

# Configuration
TRAIN_FILE = r"j:\wowDev\parp-tools\gillijimproject_refactor\height_regression.jsonl"
OUTPUT_DIR = r"j:\vlm_output\wow_tiny_vit_regressor"
MODEL_NAME = "google/vit-base-patch16-224" 
NUM_LABELS = 145 

class WoWHeightDataset(Dataset):
    def __init__(self, jsonl_path, processor, cache_in_memory=True):
        self.samples = []
        self.processor = processor
        self.cache_in_memory = cache_in_memory
        self.images_cache = []
        self.labels_cache = []
        
        # Statistics for Normalization
        all_heights = []
        
        print(f"Loading dataset index: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"Parsing {len(lines)} samples...")
        for line in tqdm(lines):
            try:
                entry = json.loads(line)
                msgs = entry.get("messages", [])
                img_path = None
                heights = None
                
                for m in msgs:
                    if m["role"] == "user":
                        for c in m["content"]:
                            if c["type"] == "image":
                                img_path = c["image"]
                    elif m["role"] == "assistant":
                        for c in m["content"]:
                            if c["type"] == "text":
                                heights = json.loads(c["text"])
                                
                if img_path and heights and len(heights) == NUM_LABELS:
                    self.samples.append((img_path, heights))
                    all_heights.append(heights)
            except:
                pass
                
        # Compute Normalization Stats
        print("Computing Normalization Statistics...")
        np_heights = np.array(all_heights)
        self.mean = torch.tensor(np.mean(np_heights, axis=0), dtype=torch.float32)
        self.std = torch.tensor(np.std(np_heights, axis=0), dtype=torch.float32) + 1e-6 # Avoid div/0
        
        print(f"Mean Height: {self.mean.mean().item():.2f}, Std: {self.std.mean().item():.2f}")
        
        # Cache Images in RAM (Optimization)
        if self.cache_in_memory:
            print("Caching images in RAM...")
            for img_path, heights in tqdm(self.samples):
                try:
                    # Pre-load PIL image
                    img = Image.open(img_path).convert("RGB")
                    # Pre-process to tensor? better to keep as PIL to save RAM? 
                    # Actually pre-processing to tensor saves CPU time during training.
                    # ViT Image size is 224x224 usually. Inputs are ~64x64. Upscaling happens.
                    # Tensor size: 3x224x224 floats = 600KB per image.
                    # 13,000 * 0.6 MB = 7.8 GB RAM.
                    # Mmm, that might be too much for 8GB simplified setup.
                    # PIL 64x64 is tiny.
                    # Let's cache PIL images.
                    img.load() # Force load
                    self.images_cache.append(img)
                    
                    # Normalize label immediately
                    lbl = (torch.tensor(heights, dtype=torch.float32) - self.mean) / self.std
                    self.labels_cache.append(lbl)
                except Exception as e:
                    print(f"Error caching {img_path}: {e}")
                    # Handle index mismatch if skipping... actually we should filter self.samples first.
                    # For simplicity, we assume robust load or just crash on bad data.
                    # Re-align
                    if len(self.images_cache) > len(self.labels_cache): self.images_cache.pop()

    def __len__(self):
        if self.cache_in_memory:
            return len(self.images_cache)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache_in_memory:
            image = self.images_cache[idx]
            labels = self.labels_cache[idx]
        else:
            img_path, heights = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            labels = (torch.tensor(heights, dtype=torch.float32) - self.mean) / self.std
            
        # Processor handles augmentation/resizing
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels
        }

def main():
    print("="*60)
    print("WoW Tiny ViT Regressor Training (Optimized)")
    print("="*60)
    
    # 1. Image Processor
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    
    # 2. Dataset
    # Cache in memory for speed (PIL objects are small for 64x64, ~12KB)
    dataset = WoWHeightDataset(TRAIN_FILE, processor, cache_in_memory=True)
    
    # Save normalization stats for inference
    stats = {
        "mean": dataset.mean.tolist(),
        "std": dataset.std.tolist()
    }
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(OUTPUT_DIR) / "normalization_stats.json", 'w') as f:
        json.dump(stats, f)
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 3. Model
    print(f"Loading Model: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="regression",
        ignore_mismatched_sizes=True
    )
    
    # 4. Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=64, # Increased Batch Size
        gradient_accumulation_steps=1,
        learning_rate=1e-4, 
        num_train_epochs=5, # More epochs since fast
        save_steps=500,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4, # Multiprocessing
        dataloader_pin_memory=True, # GPU Speedup
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor, 
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
