import os
from PIL import Image
from pathlib import Path
from collections import Counter

root_dir = Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30")

print(f"Scanning {root_dir}...")

res_counts = Counter()
file_samples = {}

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith('.png'):
            path = Path(root) / file
            try:
                with Image.open(path) as img:
                    res = img.size
                    res_counts[res] += 1
                    if res not in file_samples:
                        file_samples[res] = path.relative_to(root_dir)
            except Exception as e:
                pass

print("\n--- Resolution Counts ---")
for res, count in res_counts.items():
    print(f"{res}: {count} files (Example: {file_samples[res]})")
