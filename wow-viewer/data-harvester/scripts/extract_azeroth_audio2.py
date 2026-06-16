import numpy as np
import soundfile as sf
import json
import os

with open(r'I:\parp\parp-tools\output\weak_signal_azeroth_final\weak_signal_patch_report.json') as f:
    report = json.load(f)

outdir = r'I:\parp\parp-tools\output\azeroth_audio'
os.makedirs(outdir, exist_ok=True)

# Sort tiles by Y then X to form a spatial sequence
patched = sorted(
    [t for t in report['tiles'] if t['was_patched']],
    key=lambda t: (t['tile_y'], t['tile_x']))

print(f'Tiles in spatial order: {[(t["tile_x"], t["tile_y"]) for t in patched]}')

# Concatenate all tiles into one continuous audio stream
all_audio = []
for t in patched:
    npy_path = os.path.join(
        r'I:\parp\parp-tools\output\weak_signal_azeroth_final\World\Maps\Azeroth\heightmaps',
        f"Azeroth_{t['tile_x']}_{t['tile_y']}_before.npy")
    hm = np.load(npy_path)
    raw = hm.flatten().astype(np.float32)
    peak = max(abs(raw.max()), abs(raw.min()), 1e-8)
    raw_norm = raw / peak
    all_audio.append(raw_norm)

continuous = np.concatenate(all_audio)
peak_all = max(abs(continuous.max()), abs(continuous.min()), 1e-8)
continuous /= peak_all

full_path = os.path.join(outdir, 'Azeroth_all_weak_signals_11025.wav')
sf.write(full_path, continuous, 11025)
print(f'\nContinuous audio: {len(continuous)} samples @ 11025Hz = {len(continuous)/11025:.1f}s')
print(f'Wrote {full_path}')

# Also try transposed (columns as time) for a few tiles
for tx, ty in [(25,22), (38,56), (25,56)]:
    npy_path = os.path.join(
        r'I:\parp\parp-tools\output\weak_signal_azeroth_final\World\Maps\Azeroth\heightmaps',
        f'Azeroth_{tx}_{ty}_before.npy')
    hm = np.load(npy_path)
    raw_t = hm.T.flatten().astype(np.float32)
    peak = max(abs(raw_t.max()), abs(raw_t.min()), 1e-8)
    raw_t /= peak
    tpath = os.path.join(outdir, f'Azeroth_{tx}_{ty}_transposed_11025.wav')
    sf.write(tpath, raw_t, 11025)
    print(f'Transposed: {tpath}')

print('Done.')
