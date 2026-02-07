
import torch
from pathlib import Path

cached_dir = Path("cached_v7_6")
files = list(cached_dir.glob("target_height_*.pt"))

if not files:
    print("No files found.")
else:
    # Check first few
    for p in files[:5]:
        t = torch.load(p).float()
        print(f"{p.name}: Min={t.min().item()}, Max={t.max().item()}, Mean={t.mean().item()}")

print("-" * 20)
files_albedo = list(cached_dir.glob("target_albedo_*.pt"))
for p in files_albedo[:5]:
    t = torch.load(p).float()
    print(f"{p.name}: Min={t.min().item()}, Max={t.max().item()}, Mean={t.mean().item()}")
