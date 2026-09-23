"""Spec 237: test the chunk-local inch frame for ACDO positions.

Hypothesis: horizontal fields are inches from the centre of the ACNK that stores the record (a chunk is
1200 in wide, the fields range +-600). For each axis assignment and vertical reference, the object's
vertical field (inches) is compared with terrain height (inches) sampled at the object's location minus the
reference height. A control run shuffles objects across chunks, so the winning error can be compared with
what the same test gives when the frame is wrong.

Usage: python acdo_chunk_frame_v26.py <corpus dir>
"""
import itertools
import os
import random
import struct
import sys

corpus = sys.argv[1]
CHUNK_IN = 1200.0


def walk(b, start, end):
    p = start
    while p + 8 <= end:
        cid = b[p:p + 4].decode('latin1')[::-1]
        sz = struct.unpack_from('<I', b, p + 4)[0]
        if p + 8 + sz > end:
            break
        yield cid, p + 8, sz
        p += 8 + sz


tiles = {}
objects = []  # (tileX, tileY, cx, cy, (f04, f08, f0C))
for name in sorted(os.listdir(corpus)):
    path = os.path.join(corpus, name)
    if not os.path.isfile(path):
        continue
    b = open(path, 'rb').read()
    tx = ty = None
    outer = None
    recs = []
    ci = 0
    for cid, off, sz in walk(b, 0, len(b)):
        if cid == 'ALOC':
            _, tx, ty = struct.unpack_from('<III', b, off)
        elif cid == 'AVTX':
            outer = struct.unpack_from('<%df' % (129 * 129), b, off)
        elif cid == 'ACNK':
            for sid, soff, ssz in walk(b, off + 0x40, off + sz):
                if sid == 'ACDO':
                    recs.append((ci % 16, ci // 16, struct.unpack_from('<3f', b, soff + 4)))
            ci += 1
    if tx is None or outer is None:
        continue
    tiles[(tx, ty)] = outer
    for cx, cy, pos in recs:
        objects.append((tx, ty, cx, cy, pos))


def sample(tx, ty, col, row):
    """Bilinear outer-grid height at fractional (col, row) in tile (tx, ty), following into neighbours."""
    tx += int(col // 128)
    col -= 128 * int(col // 128)
    ty += int(row // 128)
    row -= 128 * int(row // 128)
    grid = tiles.get((tx, ty))
    if grid is None:
        return None
    c0, r0 = min(int(col), 127), min(int(row), 127)
    ax, ay = col - c0, row - r0
    return (grid[r0 * 129 + c0] * (1 - ax) * (1 - ay) + grid[r0 * 129 + c0 + 1] * ax * (1 - ay)
            + grid[(r0 + 1) * 129 + c0] * (1 - ax) * ay + grid[(r0 + 1) * 129 + c0 + 1] * ax * ay)


def evaluate(objs, vert, a, sa, b_, sb, ref):
    errs = []
    for tx, ty, cx, cy, pos in objs:
        centre_col, centre_row = cx * 8 + 4, cy * 8 + 4
        col = centre_col + sa * pos[a] / CHUNK_IN * 8
        row = centre_row + sb * pos[b_] / CHUNK_IN * 8
        h = sample(tx, ty, col, row)
        if h is None:
            continue
        if ref == 'absolute':
            base = 0.0
        elif ref == 'chunk_centre':
            base = sample(tx, ty, centre_col, centre_row)
        else:  # chunk_min over the 9x9 outer vertices
            grid = tiles[(tx, ty)]
            base = min(grid[(cy * 8 + r) * 129 + cx * 8 + c] for r in range(9) for c in range(9))
        errs.append(abs(pos[vert] - (h - base)))
    errs.sort()
    return (errs[len(errs) // 2], errs[int(len(errs) * 0.9)], len(errs)) if errs else None


results = []
for vert in range(3):
    horiz = [k for k in range(3) if k != vert]
    for a, b_ in (horiz, horiz[::-1]):
        for sa, sb in itertools.product((1, -1), repeat=2):
            for ref in ('absolute', 'chunk_centre', 'chunk_min'):
                r = evaluate(objects, vert, a, sa, b_, sb, ref)
                if r:
                    results.append((r[0], r[1], r[2], vert, a, sa, b_, sb, ref))

results.sort()
print('objects', len(objects))
print('best: median |dz| in, p90 in, n, vertical, X, Y, reference')
for r in results[:8]:
    med, p90, n, vert, a, sa, b_, sb, ref = r
    print(f'  median {med:9.2f} in ({med/36:6.2f} yd)  p90 {p90:9.2f}  n {n}  vert=+0x{4+vert*4:02X}  col=+0x{4+a*4:02X}*{sa:+d}  row=+0x{4+b_*4:02X}*{sb:+d}  {ref}')

best = results[0]
random.seed(7)
shuffled = [(o[0], o[1], random.randrange(16), random.randrange(16), o[4]) for o in objects]
control = evaluate(shuffled, best[3], best[4], best[5], best[6], best[7], best[8])
print(f'control (objects moved to random chunks, same hypothesis): median {control[0]:.2f} in, p90 {control[1]:.2f}')
