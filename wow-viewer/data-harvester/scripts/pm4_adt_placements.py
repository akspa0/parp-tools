"""Read _obj0.adt and dump WMO placements (MWMO+MODF) for correlation with PM4."""
from __future__ import annotations
import struct, json

ADT_PATH = r"I:\parp\parp-tools\wow-viewer\test_data\development\World\Maps\development\development_24_35_obj0.adt"
PM4_SEG_PATH = r"I:\parp\parp-tools\output\tmp\pm4_segments_v2.json"

# Read ADT chunks
chunks = {}
with open(ADT_PATH, 'rb') as f:
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        raw = hdr[:4]
        sz = struct.unpack('<I', hdr[4:8])[0]
        # Check for byte-swapped FourCC
        if raw in (b'REVM', b'XDMM', b'DIMM', b'OMWM', b'DIWM', b'FDDM', b'FDOM', b'KNCM'):
            fourcc = raw[::-1].decode('ascii')
        else:
            fourcc = raw.decode('ascii', errors='replace')
        chunks[fourcc] = f.read(sz)

# Parse MWMO (WMO name string block)
mwmo_data = chunks.get('MWMO', b'')
# Null-terminated strings
wmo_names = []
pos = 0
while pos < len(mwmo_data):
    end = mwmo_data.find(b'\x00', pos)
    if end < 0:
        break
    name = mwmo_data[pos:end].decode('utf-8', errors='replace')
    if name:
        wmo_names.append(name)
    pos = end + 1

# Parse MWID (32-bit offsets into MWMO)
mwid_data = chunks.get('MWID', b'')
mwid_offsets = []
for i in range(0, len(mwid_data), 4):
    mwid_offsets.append(struct.unpack_from('<I', mwid_data, i)[0])

# Map MWID offsets to WMO names
name_by_id = {}
for i, offset in enumerate(mwid_offsets):
    # Find the null-terminated string at this offset in MWMO
    end = mwmo_data.find(b'\x00', offset)
    if end >= 0:
        name_by_id[i] = mwmo_data[offset:end].decode('utf-8', errors='replace')
    else:
        name_by_id[i] = f"<unknown_{i}>"

# Parse MODF (WMO placements - 64 bytes each)
modf_data = chunks.get('MODF', b'')
MODF_SIZE = 64
print(f"=== WMO Placements from {ADT_PATH} ===\n")

for i in range(0, len(modf_data), MODF_SIZE):
    entry = modf_data[i:i+MODF_SIZE]
    if len(entry) < MODF_SIZE:
        break
    
    entry_index = i // MODF_SIZE
    
    # MODF layout: nameId, uniqueId, position(X,?,?), rotation, scale
    name_id, unique_id = struct.unpack_from('<II', entry, 0)
    px_raw, py_raw, pz_raw = struct.unpack_from('<fff', entry, 8)
    rx, ry, rz = struct.unpack_from('<fff', entry, 20)
    scale_raw = struct.unpack_from('<f', entry, 32)[0]
    flags = struct.unpack_from('<I', entry, 36)[0]
    doodad_set, name_set = struct.unpack_from('<HH', entry, 40)
    
    # ADT position: (X, Y, Z) but Y and Z appear swapped in this format
    # PM4 world-space uses X and Y as horizontal, Z as vertical
    # For tile 24_35: X ≈ 12800-13000, Y ≈ 18666-19200, Z ≈ 36-74
    pos_x = px_raw
    pos_y = pz_raw  # third component is actually Y
    pos_z = py_raw  # second component is actually Z (height)
    
    wmo_path = name_by_id.get(name_id, f"<id={name_id}>")
    wmo_name = wmo_path.split('\\')[-1].split('.')[0]
    
    print(f"Placement[{entry_index}]:")
    print(f"  NameID={name_id} UniqueID={unique_id}")
    print(f"  WMO: {wmo_name}")
    print(f"  Position: ({pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f})")
    print(f"  Scale: {scale_raw:.4f}")
    print()

# Now load the PM4 segments for this tile
with open(PM4_SEG_PATH) as f:
    seg_data = json.load(f)

print("\n=== PM4 Segments on this tile ===")
tile_segs = [s for s in seg_data['Segments'] if '24_35' in s.get('TileCoordinates', [''])]
for s in tile_segs:
    b = s.get('Bounds')
    if b:
        span_x = b['Max']['X'] - b['Min']['X']
        span_y = b['Max']['Y'] - b['Min']['Y']
        cx = (b['Min']['X'] + b['Max']['X']) / 2
        cy = (b['Min']['Y'] + b['Max']['Y']) / 2
    else:
        span_x = span_y = cx = cy = 0
    
    # Get type info from Ck24
    ck24_raw = s.get('Ck24', '0x000000')
    ck24_val = int(ck24_raw, 16) if ck24_raw.startswith('0x') else 0
    ck24_type = s.get('Ck24Type', 0)
    ck24_oid = s.get('Ck24ObjectId', 0)
    
    print(f"Segment: {s['SegmentId']}")
    print(f"  Ck24={ck24_raw} type=0x{ck24_type:02X} OID={ck24_oid}")
    print(f"  center=({cx:.2f},{cy:.2f}) span=({span_x:.2f},{span_y:.2f})")
    print(f"  surfaces={s['SurfaceCount']} indices={s['TotalIndexCount']}")
    print()

print("\n=== CORRELATION ===")
print("PM4 ObjectIDs on this tile:")
oids = set()
for s in tile_segs:
    oid = s.get('Ck24ObjectId', 0)
    if oid != 0:
        oids.add(oid)
        print(f"  OID={oid} (0x{s.get('Ck24','')[2:]}...{oid:04X})")

print("\n=== MODF UniqueIDs on this tile ===")
for i in range(0, len(modf_data), MODF_SIZE):
    entry = modf_data[i:i+MODF_SIZE]
    if len(entry) < MODF_SIZE:
        break
    _, unique_id = struct.unpack_from('<II', entry, 0)
    print(f"  UniqueID={unique_id}")

print("\n=== POSITION CORRELATION (closest PM4 segment to each placement) ===")
import math
for i in range(0, len(modf_data), MODF_SIZE):
    entry = modf_data[i:i+MODF_SIZE]
    if len(entry) < MODF_SIZE:
        break
    name_id, _ = struct.unpack_from('<II', entry, 0)
    px_raw, py_raw, pz_raw = struct.unpack_from('<fff', entry, 8)
    pos_x = px_raw
    pos_y = pz_raw
    
    wmo_path = name_by_id.get(name_id, "")
    wmo_name = wmo_path.split('\\')[-1].split('.')[0]
    
    # Find closest PM4 segment by position
    best_dist = float('inf')
    best_seg = None
    for s in tile_segs:
        b = s.get('Bounds')
        if b:
            cx = (b['Min']['X'] + b['Max']['X']) / 2
            cy = (b['Min']['Y'] + b['Max']['Y']) / 2
            dist = math.sqrt((cx - pos_x)**2 + (cy - pos_y)**2)
            if dist < best_dist:
                best_dist = dist
                best_seg = (s, cx, cy)
    
    if best_seg and best_dist < 50:
        s, cx, cy = best_seg
        print(f"  {wmo_name:40s} at ({pos_x:.1f},{pos_y:.1f})")
        print(f"    -> closest PM4: OID={s['Ck24ObjectId']} center=({cx:.1f},{cy:.1f}) dist={best_dist:.1f}")
    else:
        print(f"  {wmo_name:40s} at ({pos_x:.1f},{pos_y:.1f}) -> no close PM4 segment")
