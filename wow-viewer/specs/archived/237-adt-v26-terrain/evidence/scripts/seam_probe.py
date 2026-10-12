"""Does ALOC (field1, field2) give tile grid adjacency, and which height-grid edge meets which?
Hypothesis: AVTX = outer 129x129 float32 row-major, then inner 128x128.
Detector power: all 16 edge pairings x 2 neighbour axes are scored; a real seam must stand out."""
import os, struct, statistics, itertools
from array import array

d = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
files = [f for f in os.listdir(d) if "." not in f]
N = 129

def chunk(data, want):
    pos = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if cid == want:
            return data[pos + 8:pos + 8 + size]
        pos += 8 + size

tiles = {}
lo, hi = float("inf"), float("-inf")
flat = 0
for f in files:
    data = open(os.path.join(d, f), "rb").read()
    aloc = struct.unpack_from("<5I", chunk(data, "ALOC"))
    outer = array("f", chunk(data, "AVTX")[:N * N * 4])
    mn, mx = min(outer), max(outer)
    lo, hi = min(lo, mn), max(hi, mx)
    isflat = (mx - mn) < 1e-3
    flat += isflat
    tiles[(aloc[1], aloc[2])] = (outer, isflat)

print(f"tiles={len(tiles)} flat={flat} height range={lo:.2f}..{hi:.2f}")

edges = {
    "row0":   lambda o: o[0:N],
    "row128": lambda o: o[(N - 1) * N:N * N],
    "col0":   lambda o: o[0::N],
    "col128": lambda o: o[N - 1::N],
}

def score(step, ea, eb):
    meds = []
    for (a, b), (oa, fa) in tiles.items():
        nb = tiles.get((a + step[0], b + step[1]))
        if nb is None or (fa and nb[1]):
            continue
        xa, xb = edges[ea](oa), edges[eb](nb[0])
        meds.append(statistics.median(abs(p - q) for p, q in zip(xa, xb)))
    return (statistics.median(meds) if meds else None), len(meds)

for step, label in (((1, 0), "neighbour at field1+1"), ((0, 1), "neighbour at field2+1")):
    rows = []
    for ea, eb in itertools.product(edges, edges):
        s, n = score(step, ea, eb)
        if s is not None:
            rows.append((s, ea, eb, n))
    rows.sort()
    print(f"\n{label}: best 3 and worst 1 of {len(rows)} pairings")
    for s, ea, eb, n in rows[:3] + rows[-1:]:
        print(f"  this.{ea:6s} vs next.{eb:6s}  median|dh|={s:10.4f}  pairs={n}")
