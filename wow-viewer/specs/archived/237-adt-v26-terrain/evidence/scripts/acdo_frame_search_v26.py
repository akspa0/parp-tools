"""Spec 237: brute-force the ACDO position frame against the terrain surface.

A doodad's vertical position should sit on the terrain under it. For every hypothesis (horizontal fields,
axis flips, unit, origin) the terrain height (AVTX inches, converted to the same unit) is bilinearly sampled
at the hypothesised location and compared with the vertical field. The hypothesis with the lowest median
|error| wins; the runner-up gap shows whether the detector can tell hypotheses apart.

Usage: python acdo_frame_search_v26.py <corpus dir>
"""
import itertools
import os
import struct
import sys

corpus = sys.argv[1]
TILE_YD = 1600.0 / 3.0


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
objects = []
for name in sorted(os.listdir(corpus)):
    path = os.path.join(corpus, name)
    if not os.path.isfile(path):
        continue
    b = open(path, 'rb').read()
    tx = ty = None
    outer = None
    recs = []
    for cid, off, sz in walk(b, 0, len(b)):
        if cid == 'ALOC':
            _, tx, ty = struct.unpack_from('<III', b, off)
        elif cid == 'AVTX':
            outer = struct.unpack_from('<%df' % (129 * 129), b, off)
        elif cid == 'ACNK':
            for sid, soff, ssz in walk(b, off + 0x40, off + sz):
                if sid == 'ACDO':
                    recs.append(struct.unpack_from('<3f', b, soff + 4))
    if tx is None or outer is None:
        continue
    tiles[(tx, ty)] = outer
    for r in recs:
        objects.append((tx, ty, r))

print('tiles', len(tiles), 'objects', len(objects))


def height_at(gx, gy):
    """Terrain height (inches) at fractional global grid position in tile units (gx along tile X columns)."""
    tx, ty = int(gx // 1), int(gy // 1)
    grid = tiles.get((tx, ty))
    if grid is None:
        return None
    fx, fy = (gx - tx) * 128.0, (gy - ty) * 128.0
    c0, r0 = min(int(fx), 127), min(int(fy), 127)
    ax, ay = fx - c0, fy - r0
    h00 = grid[r0 * 129 + c0]
    h01 = grid[r0 * 129 + c0 + 1]
    h10 = grid[(r0 + 1) * 129 + c0]
    h11 = grid[(r0 + 1) * 129 + c0 + 1]
    return (h00 * (1 - ax) * (1 - ay) + h01 * ax * (1 - ay) + h10 * (1 - ax) * ay + h11 * ax * ay)


units = {'yard': 1.0, 'foot': 3.0, 'inch': 36.0}
results = []
for vert in range(3):
    horiz = [k for k in range(3) if k != vert]
    for a, b_ in (horiz, horiz[::-1]):
        for sa, sb in itertools.product((1, -1), repeat=2):
            for unit_name, per_yard in units.items():
                for origin in ('tile_corner', 'tile_centre'):
                    errs = []
                    for tx, ty, pos in objects:
                        u = sa * pos[a] / per_yard / TILE_YD
                        v = sb * pos[b_] / per_yard / TILE_YD
                        if origin == 'tile_centre':
                            u += 0.5
                            v += 0.5
                        h = height_at(tx + u, ty + v)
                        if h is None:
                            continue
                        errs.append(abs(pos[vert] / per_yard - h / 36.0))
                    if len(errs) < len(objects) * 0.5:
                        continue
                    errs.sort()
                    results.append((errs[len(errs) // 2], errs[int(len(errs) * 0.9)], len(errs), vert, a, sa, b_, sb, unit_name, origin))

results.sort()
print('best hypotheses: median |dz| yd, p90, n, vertical field, (tile X field, sign), (tile Y field, sign), unit, origin')
for r in results[:10]:
    med, p90, n, vert, a, sa, b_, sb, unit, origin = r
    print(f'  median {med:8.2f} p90 {p90:8.2f} n {n:5d}  vert=+0x{4 + vert * 4:02X}  X=+0x{4 + a * 4:02X}*{sa:+d}  Y=+0x{4 + b_ * 4:02X}*{sb:+d}  {unit:5s} {origin}')
print('worst kept:', results[-1][:3])
