import json
import os
from pathlib import Path

files = [
    r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth5\dataset\Azeroth_38_39.json",
    r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor\dataset\Kalimdor_32_30.json"
]

for p in files:
    try:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"File: {Path(p).name}")
            print(f" Image: {data.get('image')}")
            # Check if image exists relative to json
            if data.get('image'):
                dataset_dir = Path(p).parent.parent # 053_azeroth5
                img_path = dataset_dir / data.get('image')
                print(f" Image Exists: {img_path.exists()} ({img_path})")
    except Exception as e:
        print(f"Error {p}: {e}")
