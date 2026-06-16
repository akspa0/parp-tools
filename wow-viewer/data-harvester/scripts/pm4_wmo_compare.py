"""Compare PM4 surfaces for OID=9304 against the real WMO collision data."""
from __future__ import annotations
import struct, sys
from collections import defaultdict

# Load WMO collision data via inspect tool subprocess
import subprocess, json

tool = r"I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe"
client = r"I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft"
wmo_path = "world/wmo/azeroth/buildings/duskwoodabandoned_barn/duskwoodabandoned_barn.wmo"

# Dump WMO summary
result = subprocess.run(
    [tool, "wmo", "inspect", "--archive-root", client, "--virtual-path", wmo_path],
    capture_output=True, text=True
)
print("=== WMO COLLISION DATA ===")
for line in result.stdout.split('\n'):
    if any(x in line.lower() for x in ['group', 'mopy', 'movt', 'movi', 'bounds', 'colli', 'mogi']):
        print(f"  {line}")

# Parse PM4 for OID=9304
PM4_PATH = r"I:\parp\parp-tools\wow-viewer\test_data\development\World\Maps\development\development_24_35.pm4"
chunks = {}
with open(PM4_PATH, 'rb') as f:
    magic = f.read(4)
    sz = struct.unpack('<I', f.read(4))[0]
    _ = f.read(sz)
    swapped = magic == b'REVM'
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        fcc = hdr[:4]
        if swapped:
            fcc = fcc[::-1]
        fourcc = fcc.decode('ascii')
        sz = struct.unpack('<I', hdr[4:8])[0]
        chunks[fourcc] = f.read(sz)

# Parse MSUR
msur = []
for i in range(0, len(chunks.get('MSUR', b'')), 32):
    data = chunks['MSUR'][i:i+32]
    if len(data) < 32:
        break
    (gk, ic, am, pad, nx, ny, nz, h, msvi, mscn_ref, pp) = struct.unpack_from('<BBBBffffIII', data)
    ck24 = (pp >> 8) & 0x00FFFFFF
    oid = ck24 & 0xFFFF
    msur.append({
        'oid': oid, 'ck24': ck24, 'type': (pp >> 24) & 0xFF,
        'gk': gk, 'ic': ic, 'am': am,
        'nx': nx, 'ny': ny, 'nz': nz,
        'h': h, 'msvi': msvi, 'mscn_ref': mscn_ref, 'idx': i//32
    })

# Parse MSCN
mscn = []
for i in range(0, len(chunks.get('MSCN', b'')), 12):
    mscn.append(struct.unpack_from('<fff', chunks['MSCN'], i))

# Parse MSVT
msvt = []
for i in range(0, len(chunks.get('MSVT', b'')), 12):
    msvt.append(struct.unpack_from('<fff', chunks['MSVT'], i))

# Parse MSVI
msvi = []
for i in range(0, len(chunks.get('MSVI', b'')), 4):
    msvi.append(struct.unpack_from('<I', chunks['MSVI'], i)[0])

# Filter surfaces for OID=9304
target_surfs = [s for s in msur if s['oid'] == 9304]
print(f"\n=== PM4 Surfaces for OID=9304 ({len(target_surfs)} total) ===")

# Group by attribute mask to understand surface types
by_am = defaultdict(list)
for s in target_surfs:
    by_am[s['am']].append(s)

for am in sorted(by_am.keys()):
    grp = by_am[am]
    norms_x = [s['nx'] for s in grp]
    norms_y = [s['ny'] for s in grp]
    norms_z = [s['nz'] for s in grp]
    heights = [s['h'] for s in grp]
    refs = set(s['mscn_ref'] for s in grp)
    print(f"\n  attr_mask=0x{am:02X}: {len(grp)} surfaces, {len(refs)} MSCN refs")
    print(f"    avg_n=({sum(norms_x)/len(norms_x):.4f},{sum(norms_y)/len(norms_y):.4f},{sum(norms_z)/len(norms_z):.4f})")
    print(f"    h_range=({min(heights):.3f},{max(heights):.3f})")
    print(f"    heights: {sorted(set(round(h,1) for h in heights))}")
    
    # Show first 3 surfaces
    for s in grp[:3]:
        print(f"      [{s['idx']}] ic={s['ic']} gk=0x{s['gk']:02X} "
              f"n=({s['nx']:.3f},{s['ny']:.3f},{s['nz']:.3f}) h={s['h']:.3f} "
              f"mscn={s['mscn_ref']}")

# Surface statistics
print(f"\n  === Surface summary ===")
print(f"  Total surfaces: {len(target_surfs)}")
print(f"  GroupKeys: {sorted(set(s['gk'] for s in target_surfs))}")
print(f"  AttrMasks: {sorted(set(s['am'] for s in target_surfs))}")
print(f"  Unique MSCN refs: {len(set(s['mscn_ref'] for s in target_surfs))}")

# Show MSCN positions
refs = sorted(set(s['mscn_ref'] for s in target_surfs if s['mscn_ref'] < len(mscn)))
print(f"\n  MSCN positions ({len(refs)}):")
for r in refs[:10]:
    print(f"    [{r}]=({mscn[r][0]:.2f},{mscn[r][1]:.2f},{mscn[r][2]:.2f})")

# Check if MSVT vertices ARE the collision mesh
print(f"\n  === MSVT vertices ({len(msvt)}) ===")
print(f"  First 5: {msvt[:5]}")
print(f"  Last 5: {msvt[-5:]}")
