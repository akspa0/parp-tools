"""Spec 237: ANRM component order/sign and ACVT channel statistics for DAT v26.

ANRM: for each terrain tile, normals are computed from the outer AVTX grid (inches, 150 in per cell) by central
differences in the grid frame (column axis a, row axis b, vertical v). Every permutation and sign of the three
int8 components is scored by mean dot product with the computed normal; a control scores ANRM against the
normal grid shifted by one tile row of vertices. ACVT: per-channel value distributions over outer vertices.

Usage: python anrm_acvt_v26.py <corpus dir>
"""
import collections
import itertools
import math
import os
import struct
import sys

corpus = sys.argv[1]
N = 129


def walk(b, start, end):
    p = start
    while p + 8 <= end:
        cid = b[p:p + 4].decode('latin1')[::-1]
        sz = struct.unpack_from('<I', b, p + 4)[0]
        if p + 8 + sz > end:
            break
        yield cid, p + 8, sz
        p += 8 + sz


scores = collections.defaultdict(list)
control = collections.defaultdict(list)
acvt_hist = [collections.Counter() for _ in range(4)]
tiles_used = 0
for name in sorted(os.listdir(corpus)):
    path = os.path.join(corpus, name)
    if not os.path.isfile(path):
        continue
    b = open(path, 'rb').read()
    heights = normals = acvt = None
    for cid, off, sz in walk(b, 0, len(b)):
        if cid == 'AVTX':
            heights = struct.unpack_from('<%df' % (N * N), b, off)
        elif cid == 'ANRM':
            normals = struct.unpack_from('<%db' % (N * N * 3), b, off)
        elif cid == 'ACVT':
            acvt = b[off:off + N * N * 4]
    if acvt:
        for i in range(0, len(acvt), 4 * 37):
            for ch in range(4):
                acvt_hist[ch][acvt[i + ch]] += 1
    if heights is None or normals is None or max(heights) - min(heights) < 1.0:
        continue
    tiles_used += 1
    step = 150.0
    for row in range(1, N - 1, 3):
        for col in range(1, N - 1, 3):
            dha = (heights[row * N + col + 1] - heights[row * N + col - 1]) / (2 * step)
            dhb = (heights[(row + 1) * N + col] - heights[(row - 1) * N + col]) / (2 * step)
            ref = (-dha, -dhb, 1.0)  # (column axis, row axis, vertical)
            length = math.sqrt(ref[0] ** 2 + ref[1] ** 2 + 1)
            if abs(ref[0]) + abs(ref[1]) < 0.05:
                continue  # flat vertices cannot discriminate orders
            ref = tuple(x / length for x in ref)
            for idx, shifted in ((row * N + col, False), (((row + 20) % (N - 2) + 1) * N + col, True)):
                raw = normals[idx * 3: idx * 3 + 3]
                rl = math.sqrt(sum(v * v for v in raw)) or 1.0
                for perm in itertools.permutations(range(3)):
                    for signs in itertools.product((1, -1), repeat=3):
                        vec = [signs[k] * raw[perm[k]] / rl for k in range(3)]
                        d = sum(vec[k] * ref[k] for k in range(3))
                        (control if shifted else scores)[(perm, signs)].append(d)

print('tiles with terrain used', tiles_used)
ranked = sorted(((sum(v) / len(v), k) for k, v in scores.items()), reverse=True)
print('ANRM mapping -> (column, row, vertical) = components[perm] * signs; mean dot with height normals')
for mean, (perm, signs) in ranked[:6]:
    c = control[(perm, signs)]
    print(f'  perm {perm} signs {signs}: mean dot {mean:.4f}  (control, vertex 20 rows away: {sum(c) / len(c):.4f})  n {len(scores[(perm, signs)])}')

raw_len = []
for name in sorted(os.listdir(corpus))[:40]:
    b = open(os.path.join(corpus, name), 'rb').read()
    for cid, off, sz in walk(b, 0, len(b)):
        if cid == 'ANRM':
            v = struct.unpack_from('<%db' % (N * 3), b, off)
            raw_len += [math.sqrt(v[i] ** 2 + v[i + 1] ** 2 + v[i + 2] ** 2) for i in range(0, len(v), 3)]
raw_len.sort()
print('ANRM raw vector length: min %.1f median %.1f max %.1f' % (raw_len[0], raw_len[len(raw_len) // 2], raw_len[-1]))

for ch in range(4):
    h = acvt_hist[ch]
    total = sum(h.values())
    top = h.most_common(5)
    print(f'ACVT byte {ch}: distinct {len(h)}, top {[(v, round(c / total, 3)) for v, c in top]}')
