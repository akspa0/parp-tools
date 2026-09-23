"""Walk top-level chunks of M2 (MD21) / WMO files and summarise FileDataID-bearing chunks.
Usage: python chunk_walk.py <file> [<file> ...]"""
import struct, sys

FDID_CHUNKS = {b"SFID", b"TXID", b"SKID", b"BFID", b"AFID", b"PFID", b"LDV1", b"GFID", b"MODI", b"MOSI", b"MOTX", b"MOMT", b"MOHD"}

def walk(data):
    pos = 0
    while pos + 8 <= len(data):
        raw = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        yield raw, pos + 8, size
        pos += 8 + size

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    print(f"== {path} ({len(data)} bytes)")
    for raw, off, size in walk(data):
        # M2 chunk ids are stored in reading order ("MD21"); WMO ids are reversed ("REVM").
        cid = raw if raw in FDID_CHUNKS or raw == b"MD21" else raw[::-1]
        payload = data[off:off + size]
        line = f"  {cid.decode('ascii', 'replace')} size={size}"
        if cid in (b"SFID", b"TXID", b"GFID", b"MODI", b"BFID", b"AFID", b"SKID", b"PFID"):
            ids = struct.unpack_from(f"<{size // 4}I", payload)
            line += f" ids[{len(ids)}]={list(ids[:8])}"
        elif cid == b"MVER":
            line += f" version={struct.unpack_from('<I', payload)[0]}"
        elif cid == b"MD21":
            ver = struct.unpack_from("<I", payload, 4)[0]
            line += f" md20-version={ver}"
        elif cid == b"MOHD":
            vals = struct.unpack_from("<IIIIIII", payload)
            line += f" textures={vals[0]} groups={vals[1]} portals={vals[2]} lights={vals[3]} doodadNames={vals[4]} doodadDefs={vals[5]} doodadSets={vals[6]}"
            flags = struct.unpack_from("<H", payload, 0x3C)[0] if size >= 0x3E else None
            line += f" flags=0x{flags:X}" if flags is not None else ""
        elif cid == b"MOMT":
            n = size // 64
            recs = [struct.unpack_from("<IIIIIIIIIIIIIIII", payload, i * 64) for i in range(min(n, 3))]
            line += f" materials={n} first(flags,shader,blend,tex1,color,flags2,tex2,color2,group,tex3)={[r[:10] for r in recs]}"
        print(line)
