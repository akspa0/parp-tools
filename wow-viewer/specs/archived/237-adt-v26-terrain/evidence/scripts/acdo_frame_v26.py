"""Spec 237: establish the ACDO position frame for DAT v26.

For every ACDO: chunk (cx = i % 16, cy = i // 16) of the ACNK that holds it, the three position floats,
and the terrain height (inches) at the chunk centre. Fits each position axis against chunk X / chunk Y
and compares the vertical candidate with terrain height in inches and in yards.

Usage: python acdo_frame_v26.py <corpus dir> [community listfile csv]
"""
import os
import struct
import sys

corpus = sys.argv[1]
listfile_path = sys.argv[2] if len(sys.argv) > 2 else None


def walk(b, start, end):
    p = start
    while p + 8 <= end:
        cid = b[p:p + 4].decode('latin1')[::-1]
        sz = struct.unpack_from('<I', b, p + 4)[0]
        if p + 8 + sz > end:
            break
        yield cid, p + 8, sz
        p += 8 + sz


rows = []
wmo_flag = []
for name in sorted(os.listdir(corpus)):
    path = os.path.join(corpus, name)
    if not os.path.isfile(path):
        continue
    b = open(path, 'rb').read()
    outer = None
    adoo = []
    tx = ty = None
    ci = 0
    for cid, off, sz in walk(b, 0, len(b)):
        if cid == 'ALOC':
            _, tx, ty = struct.unpack_from('<III', b, off)
        elif cid == 'AVTX':
            outer = struct.unpack_from('<%df' % (129 * 129), b, off)
        elif cid == 'ADOO':
            adoo.append(b[off:off + sz].split(b'\0')[0].decode('latin1'))
        elif cid == 'ACNK':
            for sid, soff, ssz in walk(b, off + 0x40, off + sz):
                if sid == 'ACDO':
                    rec = b[soff:soff + ssz]
                    idx = struct.unpack_from('<I', rec, 0)[0]
                    px, py, pz = struct.unpack_from('<3f', rec, 4)
                    cx, cy = ci % 16, ci // 16
                    # chunk centre on the outer grid: 8 cells per chunk
                    col, row = cx * 8 + 4, cy * 8 + 4
                    h = outer[row * 129 + col] if outer else None
                    flag = struct.unpack_from('<I', rec, 0x30)[0]
                    model = adoo[idx] if idx < len(adoo) else '?'
                    rows.append((name, tx, ty, cx, cy, px, py, pz, h, flag, model, ssz))
            ci += 1

print('ACDO records', len(rows))


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx if sxx else 0
    r = sxy / (sxx * syy) ** 0.5 if sxx and syy else 0
    return slope, my - slope * mx, r


for axis, label in ((5, '+0x04'), (6, '+0x08'), (7, '+0x0C')):
    for chunk_axis, clabel in ((3, 'chunkX'), (4, 'chunkY')):
        s, c, r = fit([rw[chunk_axis] for rw in rows], [rw[axis] for rw in rows])
        print(f'{label} vs {clabel}: slope {s:.3f} yd/chunk (33.333 expected), intercept {c:.2f}, r {r:.3f}')

with_h = [rw for rw in rows if rw[8] is not None]
s, c, r = fit([rw[8] for rw in with_h], [rw[6] for rw in with_h])
print(f'+0x08 vs terrain height (inches) at chunk centre: slope {s:.4f} (1/36 = {1/36:.4f}), intercept {c:.2f}, r {r:.3f}')
diffs = sorted(abs(rw[6] - rw[8] / 36.0) for rw in with_h)
print('|+0x08 - height/36| median', round(diffs[len(diffs) // 2], 2), 'p90', round(diffs[int(len(diffs) * 0.9)], 2))

flag_models = {}
for rw in rows:
    ext = os.path.splitext(rw[10])[1].lower()
    flag_models.setdefault((rw[9], rw[11], ext), 0)
    flag_models[(rw[9], rw[11], ext)] += 1
print('(+0x30 value, record size, model extension) -> count', flag_models)
