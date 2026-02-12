from PIL import Image
import numpy as np

img_path = "test_data/vlm-datasets/053_azeroth_v20/images/Azeroth_32_48.png"
norm_path = "test_data/vlm-datasets/053_azeroth_v20/images/Azeroth_32_48_normalmap.png"

def check(path):
    print(f"Checking {path}")
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        print(f"  Shape: {arr.shape}")
        print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.2f}")
        print(f"  Sample [128,128]: {arr[128,128]}")
    except Exception as e:
        print(f"  Error: {e}")

check(img_path)
check(norm_path)
