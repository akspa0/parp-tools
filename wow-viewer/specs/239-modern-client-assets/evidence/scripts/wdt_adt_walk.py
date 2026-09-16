"""Survey a FileDataID-era WDT (MPHD flags, MAIN, MAID) and one tile's split ADT files.
Usage: python wdt_adt_walk.py wdt <file.wdt>            -> prints MPHD, MAID tiles, first tile FDIDs
       python wdt_adt_walk.py adt <root> <tex0> <obj0>   -> chunk inventory, MCLY per MCNK in tex0"""
import struct, sys, collections

def chunks(data, start=0, end=None):
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        cid = data[pos:pos + 4][::-1].decode("ascii", "replace")
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > end:
            yield cid + "!overrun", pos + 8, end - pos - 8
            return
        yield cid, pos + 8, size
        pos += 8 + size

def wdt(path):
    d = open(path, "rb").read()
    for cid, off, size in chunks(d):
        if cid == "MPHD":
            flags = struct.unpack_from("<I", d, off)[0]
            print(f"MPHD flags=0x{flags:X} size={size} ids={struct.unpack_from('<7I', d, off + 4) if size >= 32 else ''}")
        elif cid == "MAID":
            n = size // 32
            tiles = []
            for i in range(n):
                rec = struct.unpack_from("<8I", d, off + i * 32)
                if any(rec):
                    tiles.append((i // 64, i % 64, rec))
            print(f"MAID entries={n} non-empty={len(tiles)} (fields: root, obj0, obj1, tex0, lod, mapTexture, mapTextureN, minimap)")
            for y, x, rec in tiles[:3]:
                print(f"  row={y} col={x} {rec}")
            mid = tiles[len(tiles) // 2]
            print(f"  middle tile row={mid[0]} col={mid[1]} {mid[2]}")
        else:
            print(f"{cid} size={size}")

def adt(root, tex0, obj0):
    for label, path in (("root", root), ("tex0", tex0), ("obj0", obj0)):
        d = open(path, "rb").read()
        top = collections.Counter(c for c, _, _ in chunks(d))
        print(f"== {label}: {len(d)} bytes, top-level {dict(top)}")
        if label == "tex0":
            layer_hist, sub = collections.Counter(), collections.Counter()
            for cid, off, size in chunks(d):
                if cid != "MCNK":
                    continue
                n = 0
                for scid, soff, ssize in chunks(d, off, off + size):
                    sub[scid] += 1
                    if scid == "MCLY":
                        n = ssize // 16
                layer_hist[n] += 1
            print(f"   MCNK sub-chunks {dict(sub)}")
            print(f"   MCLY layers per MCNK {sorted(layer_hist.items())}")
        if label == "root":
            sub = collections.Counter()
            for cid, off, size in chunks(d):
                if cid == "MCNK":
                    for scid, _, _ in chunks(d, off + 0x80, off + size):
                        sub[scid] += 1
            print(f"   MCNK sub-chunks (after 0x80 header) {dict(sub)}")

if __name__ == "__main__":
    if sys.argv[1] == "wdt":
        wdt(sys.argv[2])
    else:
        adt(*sys.argv[2:5])
