"""Run the C# collision dumper against all OIDs on tiles 24_35, 24_36, 25_33, 25_34."""
from __future__ import annotations
import json, subprocess, sys, re
from pathlib import Path

TOOL = r"I:\parp\parp-tools\wow-viewer\tools\inspect\WowViewer.Tool.Inspect\bin\Debug\net10.0\WowViewer.Tool.Inspect.exe"
CLIENT = r"I:\parp\parp-tools\output\tmp\wowarchive-clients\3_3_5_12340\World of Warcraft"
SEGMENTS = r"I:\parp\parp-tools\output\tmp\pm4_segments_v2.json"
DEV = r"I:\parp\parp-tools\wow-viewer\test_data\development\World\Maps\development"

# Load segment export
with open(SEGMENTS) as f:
    data = json.load(f)

# Get per-tile OIDs with WMO placements (non-zero ObjectId, type 0x42/0x43)
from collections import defaultdict
tile_oids = defaultdict(list)  # tile -> [(oid, count)]

for seg in data['Segments']:
    oid = seg.get('Ck24ObjectId', 0)
    ck24_type = seg.get('Ck24Type', 0)
    if oid == 0:
        continue
    if ck24_type not in (0x42, 0x43):
        continue
    tiles = seg.get('TileCoordinates', [])
    for t in tiles:
        tile_oids[t].append(oid)

# Filter to tiles we care about
target_tiles = ['24_35', '24_36', '25_33', '25_34']
results = []

for tile in target_tiles:
    oids = list(set(tile_oids.get(tile, [])))
    oids.sort()
    print(f"\n{'='*60}")
    print(f"TILE {tile}: {len(oids)} unique OIDs: {oids}")
    print('='*60)

    for oid in oids:
        pm4_path = Path(DEV) / f"development_{tile}.pm4"
        if not pm4_path.exists():
            print(f"  [SKIP] OID={oid}: no PM4 for tile {tile}")
            continue

        print(f"\n  --- OID={oid} ---")
        result = subprocess.run(
            [TOOL, "pm4", "dump-collision",
             "--tile", tile,
             "--oid", str(oid),
             "--archive-root", CLIENT],
            capture_output=True, text=True, timeout=300
        )

        output = result.stdout
        # Parse key metrics
        surfaces = 0
        triangles = 0
        ratio = 0.0
        groups = 0
        wmo_name = ""
        dist = 0.0

        for line in output.split('\n'):
            m = re.search(r'Surfaces:\s*(\d+)', line)
            if m: surfaces = int(m.group(1))
            m = re.search(r'WMO triangles:\s*(\d+)', line)
            if m: triangles = int(m.group(1))
            m = re.search(r'Ratio:\s*([\d.]+)x', line)
            if m: ratio = float(m.group(1))
            m = re.search(r'Groups:\s*(\d+)', line)
            if m: groups = int(m.group(1))
            m = re.search(r'WMO:\s+(.+)', line)
            if m: wmo_name = m.group(1).strip().split('\\')[-1].split('.')[0]
            m = re.search(r'Closest:\s+dist=([\d.]+)', line)
            if m: dist = float(m.group(1))

        status = "OK" if (triangles > 0 and surfaces > 0) else "NO_DATA"
        results.append((tile, oid, surfaces, triangles, ratio, groups, wmo_name, dist, status))
        print(f"    WMO={wmo_name} surfaces={surfaces} triangles={triangles} ratio={ratio:.1f}x groups={groups} dist={dist:.1f} [{status}]")

        # Print errors if any
        if result.stderr:
            print(f"    stderr: {result.stderr.strip()[:200]}")

# Summary table
print(f"\n\n{'='*80}")
print(f"{'TILE':8s} {'OID':6s} {'Surf':6s} {'Tri':6s} {'Ratio':6s} {'Grp':4s} {'Dist':6s} {'WMO':40s} {'Status':10s}")
print('-'*80)
for tile, oid, surf, tri, ratio, grp, wmo, dist, status in results:
    wmo_short = wmo[:38] if len(wmo) > 38 else wmo
    print(f"{tile:8s} {oid:6d} {surf:6d} {tri:6d} {ratio:5.1f}x {grp:4d} {dist:5.1f} {wmo_short:40s} {status:10s}")
print('-'*80)
