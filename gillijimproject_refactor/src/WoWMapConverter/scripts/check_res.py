from PIL import Image
from pathlib import Path
import os

paths = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30\images\Azeroth_30_48.png"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30\images\Azeroth_25_25.png")
]

for p in paths:
    if not p.exists():
        print(f"File not found: {p}")
    else:
        with Image.open(p) as img:
            print(f"File: {p.name}")
            print(f"Size: {img.size}")
            print(f"Mode: {img.mode}")
