"""Spec 237: test whether ACNK header +0x12..+0x21 (16 bytes after a uint16 at +0x10) is a 2-bit 8x8 predominant-texture-layer map (as MCNK's).

For each chunk with 2+ layers, every 8x8 cell's dominant layer is computed from the 64x64 AMAP alphas using the
standard blend (layer i weight = alpha_i * prod(1 - alpha_j, j > i); layer 0 weight = prod(1 - alpha_j)).
The header bits are decoded under both bit orders and both cell orders and compared. Chance agreement for
a chunk with k layers is roughly the share of its most common dominant layer, reported as the control.

Usage: python acnk_predominant_v26.py <corpus dir> [map offset, default 0x12]
"""
import collections
import os
import struct
import sys

corpus = sys.argv[1]
MAP_OFFSET = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x12


def walk(b, start, end):
    p = start
    while p + 8 <= end:
        cid = b[p:p + 4].decode('latin1')[::-1]
        sz = struct.unpack_from('<I', b, p + 4)[0]
        if p + 8 + sz > end:
            break
        yield cid, p + 8, sz
        p += 8 + sz


agree = collections.Counter()
cells = 0
control_hits = 0
for name in sorted(os.listdir(corpus)):
    b = open(os.path.join(corpus, name), 'rb').read()
    for cid, off, sz in walk(b, 0, len(b)):
        if cid != 'ACNK':
            continue
        header = b[off:off + 0x40]
        layers = []
        for sid, soff, ssz in walk(b, off + 0x40, off + sz):
            if sid == 'ALYR':
                amap = None
                for aid, aoff, asz in walk(b, soff + 0x20, soff + ssz):
                    if aid == 'AMAP' and asz == 4096:
                        amap = b[aoff:aoff + 4096]
                layers.append(amap)
        if len(layers) < 2 or any(a is None for a in layers[1:]):
            continue
        bits128 = int.from_bytes(header[MAP_OFFSET:MAP_OFFSET + 16], 'little')
        dominant = []
        for cy in range(8):
            for cx in range(8):
                weights = [0.0] * len(layers)
                for py in range(8):
                    for px in range(8):
                        i = (cy * 8 + py) * 64 + cx * 8 + px
                        remaining = 1.0
                        for li in range(len(layers) - 1, 0, -1):
                            a = layers[li][i] / 255.0
                            weights[li] += a * remaining
                            remaining *= (1 - a)
                        weights[0] += remaining
                dominant.append(max(range(len(layers)), key=lambda k: weights[k]))
        common = collections.Counter(dominant).most_common(1)[0][1]
        control_hits += common
        cells += 64
        for bit_order in ('lsb', 'msb'):
            for cell_order in ('row', 'col'):
                hits = 0
                for cy in range(8):
                    for cx in range(8):
                        cell = cy * 8 + cx if cell_order == 'row' else cx * 8 + cy
                        if bit_order == 'lsb':
                            value = (bits128 >> (cell * 2)) & 3
                        else:
                            value = (bits128 >> (cell * 2)) & 3
                            value = ((value & 1) << 1) | (value >> 1)
                        hits += value == dominant[cy * 8 + cx]
                agree[(bit_order, cell_order)] += hits

print('cells compared', cells)
for key, hits in agree.most_common():
    print(f'  {key}: {hits}/{cells} = {hits / cells:.3f}')
print(f'control (always the chunk\'s most common dominant layer): {control_hits / cells:.3f}')
