"""Find the DAT v26 -> ADT mapping: height unit, horizontal scale, orientation and tile offset.
DAT heights are converted with --unit (default 36 = inches per yard). For each candidate scale s
(DAT tiles per ADT tile along an axis) and each of 8 orientations, every non-flat DAT tile is
matched against every ADT position; a mapping is consistent when one global offset explains all
tiles. Detector power: the best mapping must beat the runner-up by a wide margin.
Usage: python align_dat_to_adt.py <dat folder> <azeroth_heights.csv> [unit]"""
import csv, os, struct, sys, statistics, collections

dat_dir, adt_csv = sys.argv[1], sys.argv[2]
unit = float(sys.argv[3]) if len(sys.argv) > 3 else 36.0
SAMPLES = 8  # DAT samples per tile per axis (every 16 outer vertices)

# ADT: grid[(tileX, tileY)][row][col] for rows/cols 0..128 (outer vertices, row along Y)
adt = collections.defaultdict(dict)
with open(adt_csv) as fh:
    for r in csv.DictReader(fh):
        adt[(int(r["tileX"]), int(r["tileY"]))][(int(r["row"]), int(r["col"]))] = float(r["height"])

def adt_height(gx, gy):
    """Global ADT vertex coords (x = tileX*128 + col, y = tileY*128 + row)."""
    tx, col = divmod(gx, 128)
    ty, row = divmod(gy, 128)
    t = adt.get((tx, ty))
    return None if t is None else t.get((row, col))

dat = {}
for name in os.listdir(dat_dir):
    if "." in name:
        continue
    d = open(os.path.join(dat_dir, name), "rb").read()
    pos, aloc, outer = 0, None, None
    while pos + 8 <= len(d):
        cid = d[pos:pos + 4][::-1]
        size = struct.unpack_from("<I", d, pos + 4)[0]
        if cid == b"ALOC":
            aloc = struct.unpack_from("<5I", d, pos + 8)
        elif cid == b"AVTX":
            outer = struct.unpack_from("<16641f", d, pos + 8)
        pos += 8 + size
    if max(outer) - min(outer) < 1e-3:
        continue
    step = 128 // SAMPLES
    grid = [[outer[(r * step) * 129 + c * step] / unit for c in range(SAMPLES)] for r in range(SAMPLES)]
    dat[(aloc[1], aloc[2])] = grid

print(f"non-flat DAT tiles: {len(dat)}; ADT tiles: {len(adt)}; unit divisor {unit}")
xs = [k[0] for k in dat]; ys = [k[1] for k in dat]
print(f"DAT tile range x {min(xs)}..{max(xs)} y {min(ys)}..{max(ys)}")

def orient(r, c, n, o):
    if o & 4: r, c = c, r
    if o & 1: r = n - 1 - r
    if o & 2: c = n - 1 - c
    return r, c

results = []
adt_xs = [k[0] for k in adt]; adt_ys = [k[1] for k in adt]
for s in (1, 2, 4):
    span = 128 // s                      # ADT vertices covered by one DAT tile
    stride = span // SAMPLES             # ADT vertices per DAT sample
    for o in range(8):
        # Anchor on one DAT tile, then score the global offset over all DAT tiles.
        anchor = max(dat, key=lambda k: max(max(row) for row in dat[k]) - min(min(row) for row in dat[k]))
        best_local = []
        for gy0 in range(min(adt_ys) * 128, (max(adt_ys) + 1) * 128 - span + 1, stride * 2):
            for gx0 in range(min(adt_xs) * 128, (max(adt_xs) + 1) * 128 - span + 1, stride * 2):
                err, n = 0.0, 0
                for r in range(SAMPLES):
                    for c in range(SAMPLES):
                        rr, cc = orient(r, c, SAMPLES, o)
                        h = adt_height(gx0 + cc * stride, gy0 + rr * stride)
                        if h is None:
                            n = -1; break
                        err += (h - dat[anchor][r][c]) ** 2; n += 1
                    if n < 0: break
                if n > 0:
                    best_local.append((err / n, gx0, gy0))
        best_local.sort()
        for rmse2, gx0, gy0 in best_local[:3]:
            # global offset: ADT vertex origin for DAT tile (0,0)
            ox, oy = gx0 - anchor[0] * span, gy0 - anchor[1] * span
            errs = []
            for (tx, ty), g in dat.items():
                e, n = 0.0, 0
                for r in range(SAMPLES):
                    for c in range(SAMPLES):
                        rr, cc = orient(r, c, SAMPLES, o)
                        h = adt_height(ox + tx * span + cc * stride, oy + ty * span + rr * stride)
                        if h is None: continue
                        e += (h - g[r][c]) ** 2; n += 1
                if n: errs.append((e / n) ** 0.5)
            if errs:
                results.append((statistics.median(errs), len(errs), s, o, ox, oy))

results.sort()
print("best mappings (median per-tile RMSE yd, tiles scored, scale, orientation, ADT origin x/y):")
for row in results[:6]:
    print("  ", tuple(round(v, 2) if isinstance(v, float) else v for v in row))
