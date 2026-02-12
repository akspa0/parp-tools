from PIL import Image
import numpy as np
import json

path_img = r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30\images\Azeroth_32_48_heightmap_global.png"
path_json = r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30\dataset\Azeroth_32_48.json"

img = Image.open(path_img)
print(f"Mode: {img.mode}")
arr = np.array(img)
print(f"Min: {arr.min()}, Max: {arr.max()}")

with open(path_json) as f:
    d = json.load(f)["terrain_data"]
    print(f"JSON Min: {d.get('height_global_min')}")
    print(f"JSON Max: {d.get('height_global_max')}")
