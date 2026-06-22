import numpy as np
import soundfile as sf
import os
from pathlib import Path
import zarr
import zarr.storage
import pyarrow.parquet as pq

DATASET_ROOT = Path(r'I:\parp\parp-tools\wow-viewer\output\datasets\v18')
outdir = r'I:\parp\parp-tools\output\azeroth_audio'
os.makedirs(outdir, exist_ok=True)

BUILD = '0_5_3_3368'
zarr_path = DATASET_ROOT / f'{BUILD}.zarr'
if not zarr_path.exists():
    raise FileNotFoundError(f'No zarr store at {zarr_path}')

store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
root = zarr.open_group(store=store, mode='r')

index_path = zarr_path / 'index.parquet'
table = pq.read_table(str(index_path))

entries = []
for i in range(table.num_rows):
    row = {col: table.column(col)[i].as_py() for col in table.column_names}
    if row.get('map', '').lower() == 'azeroth':
        entries.append(row)

print(f'Found {len(entries)} Azeroth tiles in {BUILD}.zarr')

tiles_sorted = sorted(entries, key=lambda r: (r['tile_y'], r['tile_x']))
print(f'Tile coords: {[(r["tile_x"], r["tile_y"]) for r in tiles_sorted]}')

SAMPLE_RATE = 11025
all_audio = []
tile_count = 0
skipped = 0

for entry in tiles_sorted:
    tile_id = int(entry['tile_id'])
    tx = entry['tile_x']
    ty = entry['tile_y']

    try:
        hm = root['height_257'][tile_id].astype(np.float32)
    except Exception as e:
        print(f'  ({tx},{ty}) skip: height_257 unavailable ({e})')
        skipped += 1
        continue

    raw = hm.flatten()
    peak = max(abs(raw.max()), abs(raw.min()), 1e-8)
    raw_norm = raw / peak
    all_audio.append(raw_norm)
    tile_count += 1
    print(f'  ({tx},{ty}) ok  shape={hm.shape} range=[{raw.min():.2f},{raw.max():.2f}]')

if not all_audio:
    print('No audio data extracted. Aborting.')
    exit(1)

continuous = np.concatenate(all_audio)
peak_all = max(abs(continuous.max()), abs(continuous.min()), 1e-8)
continuous /= peak_all

full_path = os.path.join(outdir, f'Azeroth_all_tiles_{BUILD}_{SAMPLE_RATE}Hz.wav')
sf.write(full_path, continuous, SAMPLE_RATE)
duration = len(continuous) / SAMPLE_RATE
print(f'\nContinuous audio: {len(continuous)} samples @ {SAMPLE_RATE}Hz = {duration:.1f}s')
print(f'Tiles processed: {tile_count}  skipped: {skipped}')
print(f'Wrote {full_path}')
