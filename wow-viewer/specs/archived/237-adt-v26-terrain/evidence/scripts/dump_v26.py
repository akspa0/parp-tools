import os, struct, collections
d = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
files = sorted((f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and "." not in f), key=int)

def chunks(data):
    pos = 0
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        yield cid, data[pos + 8:pos + 8 + size]
        pos += 8 + size

def cstr(b):
    return b.split(b"\0")[0].decode("latin-1")

aloc_rows = []
adst_vals = collections.Counter()
ahdr_vals = collections.Counter()
aoch_nonzero = 0
tex_names = collections.Counter()
doo_names = collections.Counter()
acnk_hdr_idx = []

for i, name in enumerate(files):
    data = open(os.path.join(d, name), "rb").read()
    first_acnk = True
    for cid, p in chunks(data):
        if cid == "AHDR":
            ahdr_vals[struct.unpack_from("<16I", p)] += 1
        elif cid == "ALOC":
            aloc_rows.append((int(name), struct.unpack_from("<5I", p), struct.unpack_from("<5i", p), struct.unpack_from("<4fI", p), len(data)))
        elif cid == "ADST":
            adst_vals[(struct.unpack_from("<3I", p), struct.unpack_from("<3f", p))] += 1
        elif cid == "AOCH":
            if any(p):
                aoch_nonzero += 1
        elif cid == "ATEX":
            tex_names[cstr(p)] += 1
        elif cid == "ADOO":
            doo_names[cstr(p)] += 1
        elif cid == "ACNK" and first_acnk and len(p) >= 16 and i < 400:
            if len(p) > 64:
                acnk_hdr_idx.append((int(name), struct.unpack_from("<4i", p)))
            first_acnk = False

print("AHDR distinct:", len(ahdr_vals))
for v, n in ahdr_vals.most_common(3):
    print("  ", n, v)
print("\nALOC first 12 (fdid, as uint32, as int32, as 4f+I, filesize):")
for r in aloc_rows[:12]:
    print("  ", r)
print("ALOC distinct uint rows:", len({r[1] for r in aloc_rows}))
u = [r[1] for r in aloc_rows]
for k in range(5):
    col = [x[k] for x in u]
    print(f"  ALOC field{k}: min={min(col)} max={max(col)} distinct={len(set(col))}")
print("\nADST top:", adst_vals.most_common(5))
print("AOCH files with nonzero payload:", aoch_nonzero)
print("\nATEX names (top 10 of", len(tex_names), "):", tex_names.most_common(10))
print("ADOO names (top 10 of", len(doo_names), "):", doo_names.most_common(10))
print("\nfirst non-empty ACNK header ints (fdid, indexX, indexY, flags/reserved, areaId):", acnk_hdr_idx[:10])
