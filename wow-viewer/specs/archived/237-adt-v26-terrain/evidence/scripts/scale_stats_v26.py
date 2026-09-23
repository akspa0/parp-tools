"""Height statistics for DAT v26 tiles vs a real ADT, to size a display scale.
Reports per-source: height percentiles and |dh| between adjacent outer vertices (same row).
Usage: python scale_stats_v26.py <dat folder> <adt root> [<adt root> ...]"""
import os, struct, sys, statistics

def pct(values, ps=(5, 50, 95, 99)):
    s = sorted(values)
    return {p: round(s[min(len(s) - 1, int(len(s) * p / 100))], 3) for p in ps}

def dat_tiles(folder):
    for name in os.listdir(folder):
        if "." in name:
            continue
        d = open(os.path.join(folder, name), "rb").read()
        pos, aloc, outer = 0, None, None
        while pos + 8 <= len(d):
            cid = d[pos:pos + 4][::-1]
            size = struct.unpack_from("<I", d, pos + 4)[0]
            if cid == b"ALOC":
                aloc = struct.unpack_from("<5I", d, pos + 8)
            elif cid == b"AVTX":
                outer = struct.unpack_from("<16641f", d, pos + 8)
            pos += 8 + size
        yield aloc, outer

heights, diffs, flat_values = [], [], []
for aloc, outer in dat_tiles(sys.argv[1]):
    if max(outer) - min(outer) < 1e-3:
        flat_values.append(outer[0])
        continue
    heights.extend(outer[::7])
    for r in range(0, 129, 4):
        row = outer[r * 129:(r + 1) * 129]
        diffs.extend(abs(row[c + 1] - row[c]) for c in range(128))

print(f"DAT v26 non-flat: heights {pct(heights)}  |dh| neighbours {pct(diffs)}")
print(f"DAT v26 flat tiles: {len(flat_values)} distinct heights {sorted(set(round(v, 2) for v in flat_values))[:8]}")

for path in sys.argv[2:]:
    d = open(path, "rb").read()
    pos, hs, ds = 0, [], []
    while pos + 8 <= len(d):
        cid = d[pos:pos + 4][::-1]
        size = struct.unpack_from("<I", d, pos + 4)[0]
        if cid == b"MCNK":
            base_z = struct.unpack_from("<f", d, pos + 8 + 0x70)[0]
            sub = pos + 8 + 0x80
            while sub + 8 <= pos + 8 + size:
                scid = d[sub:sub + 4][::-1]
                ssize = struct.unpack_from("<I", d, sub + 4)[0]
                if scid == b"MCVT":
                    v = struct.unpack_from("<145f", d, sub + 8)
                    vals = [base_z + x for x in v]
                    hs.extend(vals)
                    for start in range(0, 145, 17):  # outer rows of 9
                        row = vals[start:start + 9]
                        ds.extend(abs(row[i + 1] - row[i]) for i in range(8))
                sub += 8 + ssize
        pos += 8 + size
    print(f"ADT {os.path.basename(path)}: heights {pct(hs)}  |dh| neighbours {pct(ds)}")
