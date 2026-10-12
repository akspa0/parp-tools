import os, struct, collections, sys
d = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))

def fourcc(b):
    return b[::-1].decode("ascii", "replace")

versions = collections.Counter()
sizes = collections.Counter()
sequences = collections.Counter()
chunk_occ = collections.Counter()
chunk_sizes = collections.defaultdict(collections.Counter)
unaccounted = collections.Counter()
examples = {}

for name in files:
    data = open(os.path.join(d, name), "rb").read()
    sizes[len(data)] += 1
    pos = 0
    seq = []
    while pos + 8 <= len(data):
        cid = fourcc(data[pos:pos + 4])
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > len(data):
            seq.append(f"{cid}!overrun")
            break
        if cid == "MVER":
            versions[struct.unpack_from("<I", data, pos + 8)[0]] += 1
        seq.append(cid)
        chunk_occ[cid] += 1
        chunk_sizes[cid][size] += 1
        pos += 8 + size
    unaccounted[len(data) - pos] += 1
    # collapse repeats for sequence signature
    sig = []
    for c in seq:
        if sig and sig[-1][0] == c:
            sig[-1][1] += 1
        else:
            sig.append([c, 1])
    key = " ".join(f"{c}x{n}" if n > 1 else c for c, n in sig)
    sequences[key] += 1
    examples.setdefault(key, name)

print("files", len(files))
print("MVER versions", dict(versions))
print("file sizes", sizes.most_common(10))
print("trailing unaccounted bytes", dict(unaccounted))
print("\nchunk occurrence (id: count, size histogram top3)")
for cid, n in chunk_occ.most_common():
    print(f"  {cid}: {n}  sizes={chunk_sizes[cid].most_common(3)}")
print("\ntop-level sequences")
for key, n in sequences.most_common(8):
    print(f"  [{n}] e.g. {examples[key]}: {key[:400]}")
