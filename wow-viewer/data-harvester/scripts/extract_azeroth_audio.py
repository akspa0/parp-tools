import numpy as np
import soundfile as sf
import json
import os

with open(r'I:\parp\parp-tools\output\weak_signal_azeroth_final\weak_signal_patch_report.json') as f:
    report = json.load(f)

outdir = r'I:\parp\parp-tools\output\azeroth_audio'
os.makedirs(outdir, exist_ok=True)

patched = [t for t in report['tiles'] if t['was_patched']]
print(f'Extracting audio from {len(patched)} weak-signal tiles...')

rates = [8000, 11025, 16000, 22050]
for t in patched:
    npy_path = os.path.join(
        r'I:\parp\parp-tools\output\weak_signal_azeroth_final\World\Maps\Azeroth\heightmaps',
        f"Azeroth_{t['tile_x']}_{t['tile_y']}_before.npy")
    try:
        hm = np.load(npy_path)
        raw = hm.flatten().astype(np.float32)
        peak = max(abs(raw.max()), abs(raw.min()), 1e-8)
        raw_norm = raw / peak

        # Write at 11025Hz (most natural for the ~66K samples → ~6s clips)
        outpath = os.path.join(outdir, f"Azeroth_{t['tile_x']}_{t['tile_y']}_11025.wav")
        sf.write(outpath, raw_norm, 11025)
        print(f"  ({t['tile_x']},{t['tile_y']}) ok")
    except Exception as e:
        print(f"  ({t['tile_x']},{t['tile_y']}) skip: {e}")

print(f'\nDone. {outdir}/')
for f in sorted(os.listdir(outdir)):
    size = os.path.getsize(os.path.join(outdir, f))
    print(f'  {f}  ({size} bytes)')
