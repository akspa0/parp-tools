import argparse
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

# Ensure harvester package is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from harvester.v25.lapnet import V25StageBPredictor
from harvester.v25.losses import V25UnifiedLoss

def main():
    parser = argparse.ArgumentParser(description="V25 real-shard solver/loss demo")
    parser.add_argument("--npz", required=True,
                        help="harvested tile shard NPZ (e.g. output/ZulAman_27_29_v14.npz)")
    args = parser.parse_args()
    npz_path = args.npz
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} does not exist. Please harvest it first.")
        sys.exit(1)
        
    print(f"=== Loading Real Shard: {npz_path} ===")
    shard = np.load(npz_path, allow_pickle=True)
    
    # 1. Load the actual arrays harvested from Zul'Aman tile 27,29
    h_257_real = torch.from_numpy(shard["height_257"]).float().unsqueeze(0)  # (1, 257, 257)
    h_17_real = torch.from_numpy(shard["height_17"]).float().unsqueeze(0)    # (1, 17, 17)
    minimap_real = torch.from_numpy(shard["minimap_rgb_256"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 3, 256, 256)
    alpha_real = torch.from_numpy(shard["mcal_alpha_pack_256"]).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, 4, 256, 256)
    mcly_real = torch.from_numpy(shard["mcly_layer_mask"]).long().permute(2, 0, 1).unsqueeze(0)  # (1, 4, 16, 16)
    
    print(f"Loaded height_257 shape: {h_257_real.shape}")
    print(f"Loaded height_17 shape: {h_17_real.shape}")
    print(f"Loaded minimap shape:    {minimap_real.shape}")
    print(f"Loaded alpha shape:      {alpha_real.shape}")
    print(f"Loaded mcly shape:       {mcly_real.shape}")
    
    # --- Part 1: Run Progressive Solver on Real Data ---
    print("\n=== Running V25StageBPredictor (Progressive Solver) ===")
    model = V25StageBPredictor()
    
    # Downsample height_17 to 33x33 to act as h_33 solver input
    h_33 = F.interpolate(h_17_real.unsqueeze(1), size=(33, 33), mode="bilinear", align_corners=True).squeeze(1)
    
    # Run the model (bilinear upscaling -> residual correction -> Sylvester solver)
    pred_h_257 = model(h_33, minimap_real)
    print(f"Solver output height shape: {pred_h_257.shape}")
    
    # Check L1 error of solver output vs true heightmap
    solve_l1 = F.l1_loss(pred_h_257, h_257_real).item()
    print(f"L1 error (untrained solver vs ground truth): {solve_l1:.4f} meters")
    
    # --- Part 2: Run Unified Loss with CVPR 2026 Enhancements ---
    print("\n=== Running V25UnifiedLoss ===")
    loss_fn = V25UnifiedLoss(use_freq_split=True, freq_cutoff=0.1)
    
    # Setup mock dictionary predictions to feed the loss
    # (We plug in real targets with tiny noise to simulate predictions)
    pred_outputs = {
        "mask_logits": torch.randn(1, 1, 256, 256),
        "h_257": pred_h_257,  # Use progressive solver output
        "placements": {
            "class_logits": torch.randn(1, 8, 16),
            "coords": torch.zeros(1, 8, 3),
            "rotations": torch.zeros(1, 8, 3),
            "exist_logits": torch.randn(1, 8, 1)
        },
        "mtex_logits": torch.randn(1, 64),
        "mcly_logits": torch.randn(1, 4, 16, 16, 4),
        "alpha_256": alpha_real + torch.randn_like(alpha_real) * 0.05  # target + noise
    }
    
    # Setup targets dictionary from real data
    target_outputs = {
        "mask": torch.ones(1, 1, 256, 256),
        "h_257": h_257_real,
        "placements": {
            "class_ids": torch.zeros(1, 8, dtype=torch.long),
            "coords": torch.zeros(1, 8, 3),
            "rotations": torch.zeros(1, 8, 3),
            "exist": torch.zeros(1, 8, dtype=torch.long)
        },
        "mtex_labels": torch.zeros(1, 64, dtype=torch.long),
        "mcly_labels": torch.zeros(1, 16, 16, 4, dtype=torch.long),
        "alpha_256": alpha_real
    }
    
    # Compute the loss (passing minimap triggers TexADiff texture-density weighting)
    losses = loss_fn(pred_outputs, target_outputs, minimap=minimap_real)
    
    print("Computed Loss Terms:")
    print(f"  Total Unified Loss:    {losses['loss'].item():.4f}")
    print(f"  Height Loss (FRAMER):  {losses['height'].item():.4f}")
    print(f"    Low-Frequency (LF):  {losses['height_lf'].item():.4f}")
    print(f"    High-Frequency (HF): {losses['height_hf'].item():.4f}")
    print(f"  Alpha Loss (TexADiff): {losses['alpha'].item():.4f}")
    print(f"    Average Density Wt:  {losses['texture_density_mean'].item():.4f}")

if __name__ == "__main__":
    main()
