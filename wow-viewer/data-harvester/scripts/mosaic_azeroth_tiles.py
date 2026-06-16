import numpy as np, matplotlib.pyplot as plt, json, os

with open(r'I:\parp\parp-tools\output\weak_signal_azeroth_final\weak_signal_patch_report.json') as f:
    report = json.load(f)

base = r'I:\parp\parp-tools\output\weak_signal_azeroth_final\World\Maps\Azeroth\heightmaps'
outdir = r'I:\parp\parp-tools\output\azeroth_audio'
os.makedirs(outdir, exist_ok=True)

patched = [t for t in report['tiles'] if t['was_patched']]

# Cluster 1: (35-40, 21-23) = 6 cols x 3 rows
fig1, axes1 = plt.subplots(3, 6, figsize=(18, 9))
axes1 = axes1.flatten()
c1 = sorted(
    [t for t in patched if 35 <= t['tile_x'] <= 40 and 21 <= t['tile_y'] <= 23],
    key=lambda t: (t['tile_y'], t['tile_x']))
for i, t in enumerate(c1):
    if i >= 18: break
    path = os.path.join(base, f"Azeroth_{t['tile_x']}_{t['tile_y']}_before.npy")
    hm = np.load(path)
    axes1[i].imshow(hm, cmap='gray', aspect='auto')
    axes1[i].set_title(f"({t['tile_x']},{t['tile_y']}) {hm.min():.1f}..{hm.max():.1f}", fontsize=7)
    axes1[i].axis('off')
for i in range(len(c1), 18):
    axes1[i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'cluster1_mosaic.png'), dpi=150)
plt.close()

# Cluster 2: (25-27, 22-23) = 3x2
fig2, axes2 = plt.subplots(2, 3, figsize=(10, 7))
axes2 = axes2.flatten()
c2 = sorted(
    [t for t in patched if 25 <= t['tile_x'] <= 27 and 22 <= t['tile_y'] <= 23],
    key=lambda t: (t['tile_y'], t['tile_x']))
for i, t in enumerate(c2):
    if i >= 6: break
    path = os.path.join(base, f"Azeroth_{t['tile_x']}_{t['tile_y']}_before.npy")
    hm = np.load(path)
    axes2[i].imshow(hm, cmap='gray', aspect='auto')
    axes2[i].set_title(f"({t['tile_x']},{t['tile_y']}) {hm.min():.1f}..{hm.max():.1f}", fontsize=8)
    axes2[i].axis('off')
for i in range(len(c2), 6):
    axes2[i].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'cluster2_mosaic.png'), dpi=150)
plt.close()

# Scattered tiles
scattered = [t for t in patched if t not in c1 and t not in c2]
print(f'Cluster 1: {len(c1)} tiles')
print(f'Cluster 2: {len(c2)} tiles')
print(f'Scattered: {len(scattered)} tiles: {[(t["tile_x"],t["tile_y"]) for t in scattered]}')

# Also save each scattered tile individually
for t in scattered:
    path = os.path.join(base, f"Azeroth_{t['tile_x']}_{t['tile_y']}_before.npy")
    hm = np.load(path)
    plt.figure(figsize=(4, 4))
    plt.imshow(hm, cmap='gray', aspect='auto')
    plt.title(f"({t['tile_x']},{t['tile_y']}) {hm.min():.1f}..{hm.max():.1f}")
    plt.axis('off')
    plt.savefig(os.path.join(outdir, f"scattered_{t['tile_x']}_{t['tile_y']}.png"), dpi=100)
    plt.close()

print('Done')
