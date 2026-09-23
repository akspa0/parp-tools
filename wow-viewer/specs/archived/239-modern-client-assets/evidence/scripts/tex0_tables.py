"""Dump MDID/MHID tables and per-chunk MCLY records of a tex0 ADT, plus which texture ids are 0.
Usage: python tex0_tables.py <wdt> <tileX> <tileY> <tex0 file>   (wdt used only to print the MAID row)"""
import struct, sys, collections

def chunks(d, s=0, e=None):
    e = len(d) if e is None else e
    p = s
    while p + 8 <= e:
        cid = d[p:p + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", d, p + 4)[0]
        yield cid, p + 8, size
        p += 8 + size

wdt, tx, ty, tex0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
w = open(wdt, "rb").read()
for cid, off, size in chunks(w):
    if cid == "MAID":
        slot = ty * 64 + tx
        print(f"MAID slot {slot} (tile {tx}_{ty}):", struct.unpack_from("<8I", w, off + slot * 32))

d = open(tex0, "rb").read()
tables = {}
for cid, off, size in chunks(d):
    if cid in ("MDID", "MHID"):
        tables[cid] = list(struct.unpack_from(f"<{size // 4}I", d, off))
        print(f"{cid} [{len(tables[cid])}]: {tables[cid]}")
    elif cid == "MAMP":
        print(f"MAMP size={size} value={d[off:off + size].hex()}")

used = collections.Counter()
chunk_index = 0
examples = []
for cid, off, size in chunks(d):
    if cid != "MCNK":
        continue
    for scid, soff, ssize in chunks(d, off, off + size):
        if scid == "MCLY":
            layers = [struct.unpack_from("<IIII", d, soff + i * 16) for i in range(ssize // 16)]
            for tex_id, flags, ofs_mcal, effect in layers:
                used[tex_id] += 1
            if chunk_index == 3 * 16 + 12 or (len(examples) < 2 and any(tables.get("MDID", [1] * 999)[l[0]] == 0 for l in layers if l[0] < len(tables.get("MDID", [])))):
                examples.append((chunk_index, layers))
    chunk_index += 1

print("MCLY texture index usage:", dict(sorted(used.items())))
mdid = tables.get("MDID", [])
print("indices whose MDID entry is 0:", [i for i in used if i < len(mdid) and mdid[i] == 0])
for idx, layers in examples:
    print(f"chunk {idx} (x={idx % 16}, y={idx // 16}) MCLY (textureId, flags, mcalOffset, effectId): {layers}")
