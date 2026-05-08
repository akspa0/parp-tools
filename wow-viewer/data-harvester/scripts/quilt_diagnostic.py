import numpy as np, json, sys
from PIL import Image

npz = np.load(sys.argv[1])
meta = json.loads(bytes(npz["metadata.json"]).decode("utf-8"))
tile = meta.get("tile_name", "tile")

def norm8(a):
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi - lo > 1e-8:
        a = (a - lo) / (hi - lo)
    return (a * 255).astype(np.uint8)

# Panel 1: height (grayscale) — 257x257
h = norm8(npz["height_257"])

# Panel 2: normals X — 257x257
nx = norm8(npz["mcnr_normal_xyz"][:,:,0])

# Panel 3: normals Y
ny = norm8(npz["mcnr_normal_xyz"][:,:,1])

# Panel 4: normals Z
nz = norm8(npz["mcnr_normal_xyz"][:,:,2])

# Panel 5: shadow mask — 256x256
sh = norm8(npz["mcsh_shadow_mask_256"])

# Panel 6: alpha layer 1 — 256x256
a1 = norm8(npz["mcal_alpha_pack_256"][:,:,1])
a2 = norm8(npz["mcal_alpha_pack_256"][:,:,2])
a3 = norm8(npz["mcal_alpha_pack_256"][:,:,3])

# Panel 7: texture ID grid — 16x16, each cell colored by ID
import colorsys
ids = npz["mcly_texture_ids"]  # (16, 16, 4)
unique_ids = sorted(set(ids.flatten()) - {-1})
id_colors = {}
for i, uid in enumerate(unique_ids):
    hue = i / max(1, len(unique_ids))
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
    id_colors[uid] = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
id_img = np.zeros((256, 256, 3), dtype=np.uint8)
for cy in range(16):
    for cx in range(16):
        cid = ids[cy, cx, 0]
        color = id_colors.get(cid, (128, 128, 128))
        id_img[cy*16:(cy+1)*16, cx*16:(cx+1)*16] = color

# Panel 8: minimap (ground truth)
mm = npz["minimap_rgb_256"]

# Assemble 3x4 grid
cells = [
    (h, "height"),
    (nx, "norm-X"),
    (ny, "norm-Y"),
    (nz, "norm-Z"),
    (sh, "shadow"),
    (a1, "alpha-L1"),
    (a2, "alpha-L2"),
    (a3, "alpha-L3"),
    (id_img, "tex-ID"),
    (mm, "minimap"),
]

# Lay out 5 rows x 2 cols = 10 panels
rows, cols = 5, 2
pw, ph = 260, 260  # panel size
grid = Image.new("RGB", (pw * cols, ph * rows), (32, 32, 32))

for i, (arr, label) in enumerate(cells):
    r, c = i // cols, i % cols
    arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[-1] != 3:
        arr = arr[:,:,:3]
    # Resize to 256x256 if needed
    img = Image.fromarray(arr, "RGB").resize((256, 256), Image.NEAREST)
    grid.paste(img, (c * pw + 2, r * ph + 2))

out = sys.argv[2] if len(sys.argv) > 2 else f"{tile}_diagnostic.png"
grid.save(out)
print(f"Diagnostic saved: {out}")
