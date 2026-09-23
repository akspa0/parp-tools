"""ACNK index fields vs file order; ADST file count and placement; ACDO position range."""
import os, struct, collections

d = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
files = [f for f in os.listdir(d) if "." not in f]

def walk(buf, start=0):
    pos = start
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        yield cid, buf[pos + 8:pos + 8 + size]
        pos += 8 + size

orders = collections.Counter()
adst_files = 0
adst_per_file = collections.Counter()
order_sig = collections.Counter()
pos_min = [float("inf")] * 3
pos_max = [float("-inf")] * 3

for f in files:
    data = open(os.path.join(d, f), "rb").read()
    i = 0
    n_adst = 0
    sig = []
    for cid, p in walk(data):
        if not sig or sig[-1] != cid:
            sig.append(cid)
        if cid == "ADST":
            n_adst += 1
        if cid != "ACNK":
            continue
        x, y = struct.unpack_from("<II", p, 0)
        if (x, y) == (i % 16, i // 16):
            orders["indexX=i%16, indexY=i//16"] += 1
        elif (x, y) == (i // 16, i % 16):
            orders["indexX=i//16, indexY=i%16"] += 1
        else:
            orders["other"] += 1
        i += 1
        if len(p) > 64:
            for sid, sp in walk(p, 64):
                if sid == "ACDO":
                    for k, v in enumerate(struct.unpack_from("<3f", sp, 4)):
                        pos_min[k] = min(pos_min[k], v)
                        pos_max[k] = max(pos_max[k], v)
    if n_adst:
        adst_files += 1
        adst_per_file[n_adst] += 1
    order_sig[" ".join(sig)] += 1

print("ACNK index vs file order:", dict(orders))
print("files with ADST:", adst_files, "ADST chunks per file:", dict(adst_per_file))
print("top-level orders:", order_sig.most_common(4))
print("ACDO position ranges (fields 0..2):", list(zip(pos_min, pos_max)))
