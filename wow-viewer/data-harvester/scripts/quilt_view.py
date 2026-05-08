import numpy as np
from PIL import Image
import sys, json

npz = np.load(sys.argv[1])

def safe_get(key, default=None):
    try:
        return npz[key]
    except KeyError:
        return default

meta_bytes = bytes(npz["metadata.json"])
meta = json.loads(meta_bytes.decode("utf-8"))
tile_name = meta.get("tile_name", "tile")

def norm8(arr):
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    return (arr * 255).astype(np.uint8)

def resize_to(arr, size):
    img = Image.fromarray(arr)
    img = img.resize((size, size), Image.NEAREST)
    return np.array(img, dtype=np.float32)

height = norm8(safe_get("height_257", np.zeros((257,257))))
alpha = safe_get("mcal_alpha_pack_256", np.zeros((256,256,4)))
alpha_rgb = norm8(alpha[:, :, 1:4]) if alpha.shape[-1] >= 4 else np.zeros((256,256,3), dtype=np.uint8)
normals = norm8(safe_get("mcnr_normal_xyz", np.zeros((257,257,3))) * 0.5 + 0.5)
shadow = safe_get("mcsh_shadow_mask_256", np.zeros((256,256)))
minimap = safe_get("minimap_rgb_256", np.zeros((256,256,3), dtype=np.uint8))

# 2x3 grid: height, normals, shadow, minimap, alpha, height+shadow
h, w = 257, 257
grid = Image.new("RGB", (w * 3, h * 2))

grid.paste(Image.fromarray(np.stack([height]*3, axis=-1), "RGB"), (0, 0))
grid.paste(Image.fromarray(normals, "RGB"), (w, 0))
shadow_img = Image.fromarray(np.stack([norm8(shadow)]*3, axis=-1), "RGB")
shadow_img = shadow_img.resize((w, h), Image.NEAREST)
grid.paste(shadow_img, (w*2, 0))

minimap_img = Image.fromarray(minimap, "RGB").resize((w, h), Image.NEAREST)
grid.paste(minimap_img, (0, h))

alpha_img = Image.fromarray(alpha_rgb, "RGB").resize((w, h), Image.NEAREST)
grid.paste(alpha_img, (w, h))

shadow257_arr = np.array(Image.fromarray(norm8(shadow)).resize((w, h), Image.NEAREST), dtype=np.float32) / 255.0
hs = norm8(safe_get("height_257", np.zeros((257,257))) * 0.7 + shadow257_arr * 0.3)
grid.paste(Image.fromarray(np.stack([hs]*3, axis=-1), "RGB"), (w*2, h))

out = sys.argv[2] if len(sys.argv) > 2 else f"{tile_name}_quilt.png"
grid.save(out)
print(f"Saved quilt to {out}")
print(f"  signals: {meta.get('available_signals', [])}")
