"""ACNK sub-chunk inventory for ADT v26 (wiki page draft support).
Reports: top-level position of ADST, ACNK header field distributions, sub-chunk ids/sizes,
ALYR fixed-part size and nested AMAP, ACDO record-size candidates."""
import os, struct, collections

d = r"I:\parp\parp-tools\wow-viewer\test_data\v22_adts\unknown"
files = [f for f in os.listdir(d) if "." not in f]

def walk(buf, start=0, end=None):
    end = len(buf) if end is None else end
    pos = start
    while pos + 8 <= end:
        cid = buf[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        if pos + 8 + size > end:
            yield cid, None, pos
            return
        yield cid, buf[pos + 8:pos + 8 + size], pos
        pos += 8 + size
    if pos != end:
        yield "<gap>", buf[pos:end], pos

adst_prev = collections.Counter()
acnk_sizes = collections.Counter()
hdr_fields = [collections.Counter() for _ in range(16)]
sub_occ = collections.Counter()
sub_sizes = collections.defaultdict(collections.Counter)
sub_seq = collections.Counter()
alyr_tail = collections.Counter()
alyr_fields = collections.defaultdict(collections.Counter)
acdo_sizes = collections.Counter()
acdo_samples = []
gaps = 0
nonempty_acnk = 0

for f in files:
    data = open(os.path.join(d, f), "rb").read()
    prev = None
    for cid, p, _ in walk(data):
        if cid == "ADST":
            adst_prev[prev] += 1
        prev = cid
        if cid != "ACNK" or p is None:
            continue
        acnk_sizes[len(p)] += 1
        if len(p) >= 64:
            for i, v in enumerate(struct.unpack_from("<16I", p, 0)):
                hdr_fields[i][v] += 1
        if len(p) <= 64:
            continue
        nonempty_acnk += 1
        seq = []
        for sid, sp, _ in walk(p, 64):
            if sid == "<gap>":
                gaps += 1
                continue
            seq.append(sid)
            sub_occ[sid] += 1
            sub_sizes[sid][len(sp) if sp is not None else -1] += 1
            if sid == "ALYR" and sp is not None:
                tex, flags = struct.unpack_from("<iI", sp, 0)
                alyr_fields["flags"][hex(flags)] += 1
                alyr_fields["reserved_nonzero"][any(sp[8:32])] += 1
                inner = [(iid, len(ip) if ip is not None else -1) for iid, ip, _ in walk(sp, 32)] if len(sp) > 32 else []
                alyr_tail[tuple(inner)] += 1
            if sid == "ACDO" and sp is not None:
                acdo_sizes[len(sp)] += 1
                if len(acdo_samples) < 3 and len(sp) >= 0x30:
                    acdo_samples.append((struct.unpack_from("<i3f3f3ffI", sp, 0), sp[0x30:].hex()))
        sub_seq[" ".join(seq)[:120]] += 1

print("ADST preceded by:", dict(adst_prev))
print("ACNK payload sizes (top 8):", acnk_sizes.most_common(8))
print("non-empty ACNK:", nonempty_acnk, "gaps inside ACNK walks:", gaps)
print("\nACNK header uint32 fields (distinct count, top 4):")
for i, c in enumerate(hdr_fields):
    print(f"  +0x{i*4:02X}: distinct={len(c)} top={c.most_common(4)}")
print("\nsub-chunk occurrence:")
for sid, n in sub_occ.most_common():
    print(f"  {sid}: {n} sizes={sub_sizes[sid].most_common(5)}")
print("\nsub-chunk sequences (top 6):", sub_seq.most_common(6))
print("\nALYR flags:", alyr_fields["flags"].most_common(8))
print("ALYR reserved bytes non-zero:", dict(alyr_fields["reserved_nonzero"]))
print("ALYR nested content (top 6):", alyr_tail.most_common(6))
print("\nACDO sizes:", acdo_sizes.most_common(6))
print("ACDO samples (int modelid, pos3, rot3, scale3, float, uint uniqueId; trailing hex):")
for s in acdo_samples:
    print("  ", s)
