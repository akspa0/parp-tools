"""Dump PM4 per-ObjectId surface data for comparison against real WMO."""
from __future__ import annotations
import struct, sys
from collections import defaultdict

PM4_PATH = r"I:\parp\parp-tools\wow-viewer\test_data\development\World\Maps\development\development_24_35.pm4"

def read_pm4(path):
    chunks = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        size = struct.unpack('<I', f.read(4))[0]
        _ = f.read(size)
        swapped = magic == b'REVM'
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            fcc = hdr[:4]
            if swapped:
                fcc = fcc[::-1]
            fourcc = fcc.decode('ascii', errors='replace')
            sz = struct.unpack('<I', hdr[4:8])[0]
            chunks[fourcc] = f.read(sz)
    return chunks

chunks = read_pm4(PM4_PATH)
msur_raw = chunks.get('MSUR', b'')
mscn_raw = chunks.get('MSCN', b'')
msvt_raw = chunks.get('MSVT', b'')

# Parse MSCN positions
mscn = []
for i in range(0, len(mscn_raw), 12):
    mscn.append(struct.unpack_from('<fff', mscn_raw, i))

# Parse MSVT vertices  
msvt = []
for i in range(0, len(msvt_raw), 12):
    msvt.append(struct.unpack_from('<fff', msvt_raw, i))

# Parse MSUR surfaces
msur = []
for i in range(0, len(msur_raw), 32):
    data = msur_raw[i:i+32]
    if len(data) < 32:
        break
    (gk, ic, am, pad, nx, ny, nz, h, msvi, mscn_ref, pp) = struct.unpack_from('<BBBBffffIII', data)
    ck24 = (pp >> 8) & 0x00FFFFFF
    msur.append({
        'oid': ck24 & 0xFFFF,
        'type': (pp >> 24) & 0xFF,
        'ck24': ck24,
        'gk': gk, 'ic': ic, 'am': am,
        'nx': nx, 'ny': ny, 'nz': nz,
        'h': h, 'msvi': msvi, 'mscn_ref': mscn_ref
    })

# Group by ObjectId
by_oid = defaultdict(list)
for s in msur:
    by_oid[s['oid']].append(s)

print(f"Total surfaces: {len(msur)}")
print(f"Total MSCN: {len(mscn)}")
print(f"Total MSVT: {len(msvt)}")
print()

for oid in sorted(by_oid.keys()):
    if oid == 0:
        continue  # skip zero-CK24 fallback groups
    surfs = by_oid[oid]
    first = surfs[0]
    
    # Get unique MSCN refs for this object
    refs = sorted(set(s['mscn_ref'] for s in surfs if s['mscn_ref'] < len(mscn)))
    positions = [(r, mscn[r]) for r in refs[:10]]  # first 10 positions
    
    # Bounds from positions
    if positions:
        xs = [p[1][0] for p in positions]
        ys = [p[1][1] for p in positions]
        zs = [p[1][2] for p in positions]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)
        span = (bx1-bx0, by1-by0)
    else:
        span = (0, 0)
    
    # Surface summary
    norms_x = [s['nx'] for s in surfs]
    norms_y = [s['ny'] for s in surfs]
    norms_z = [s['nz'] for s in surfs]
    heights = [s['h'] for s in surfs]
    
    print(f"OID={oid:5d} type=0x{first['type']:02X} Ck24=0x{first['ck24']:06X}")
    print(f"  surfaces={len(surfs)} span=({span[0]:.1f},{span[1]:.1f})")
    print(f"  avg_n=({sum(norms_x)/len(norms_x):.4f},{sum(norms_y)/len(norms_y):.4f},{sum(norms_z)/len(norms_z):.4f})")
    print(f"  h_range=({min(heights):.3f},{max(heights):.3f})")
    print(f"  positions ({len(positions)} unique):")
    for r, p in positions[:5]:
        print(f"    [{r}]=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})")
    print()
