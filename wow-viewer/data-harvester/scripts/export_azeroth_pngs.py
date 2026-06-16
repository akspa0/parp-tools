import numpy as np
import json
from PIL import Image
import os

with open(r'I:\parp\parp-tools\output\weak_signal_azeroth_final\weak_signal_patch_report.json') as f:
    report = json.load(f)

outdir = r'I:\parp\parp-tools\output\azeroth_audio'
os.makedirs(outdir, exist_ok=True)

patched = [t for t in report['tiles'] if t['was_patched']]

for t in patched:
    tx, ty = t['tile_x'], t['tile_y']
    npy_path = os.path.join(
        r'I:\parp\parp-tools\output\weak_signal_azeroth_final\World\Maps\Azeroth\heightmaps',
        f'Azeroth_{tx}_{ty}_before.npy')
    hm = np.load(npy_path)

    # Save as 16-bit grayscale PNG for inspection
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    img = (hm_norm * 65535).astype(np.uint16)
    Image.fromarray(img).save(os.path.join(outdir, f'Azeroth_{tx}_{ty}_raw.png'))

    # Also save amplified version
    ap = npy_path.replace('_before.npy', '_after.npy')
    if os.path.exists(ap):
        hm_a = np.load(ap)
        hm_a_norm = (hm_a - hm_a.min()) / (hm_a.max() - hm_a.min() + 1e-8)
        img_a = (hm_a_norm * 65535).astype(np.uint16)
        Image.fromarray(img_a).save(os.path.join(outdir, f'Azeroth_{tx}_{ty}_amplified.png'))

print(f'Ouput directory: {outdir}/')
print(f'{len(patched)} tiles exported as 16-bit PNGs')
