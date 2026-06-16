"""Aggregate top PM4 match results across all tiles."""
import json
import os

base = r'I:\parp\parp-tools\output\tmp'
tiles = ['48_37', '35_37', '32_37', '45_41', '29_18']
all_results = []

for t in tiles:
    path = os.path.join(base, f'pm4_match_{t}.json')
    with open(path) as f:
        data = json.load(f)
    for seg in data['Segments']:
        if seg.get('Candidates') and seg['Candidates']:
            c = seg['Candidates'][0]
            all_results.append({
                'tile': t,
                'seg_id': seg['SegmentId'],
                'status': seg['Status'],
                'kind': seg.get('ExpectedAssetKind'),
                'score': c['OverallScore'],
                'asset': c['AssetPath'].split('\\')[-1],
                'breakdown': c.get('ScoreBreakdown', {}),
            })

all_results.sort(key=lambda r: -r['score'])

print(f'Total matchable segments: {len(all_results)}')
print()
print('=== TOP 10 MATCHES ===')
for r in all_results[:10]:
    b = r['breakdown']
    print(f'Score {r["score"]:.4f} | tile {r["tile"]} | status {r["status"]} | kind {r["kind"]}')
    print(f'  Asset: {r["asset"]}')
    print(f'  span={b.get("sortedSpanScore",0):.3f} foot={b.get("footprintAreaScore",0):.3f} '
          f'vol={b.get("volumeScore",0):.3f} diag={b.get("diagonalScore",0):.3f} '
          f'h={b.get("heightScore",0):.3f}')
    print(f'  shape={b.get("shapeScore",0):.3f} overlap={b.get("typedOverlapScore",0):.3f} '
          f'profile={b.get("typeProfileScore",0):.3f}')
    print()

# Per-tile summary
print('=== PER TILE SUMMARY ===')
for t in tiles:
    tile_results = [r for r in all_results if r['tile'] == t]
    if tile_results:
        max_score = max(r['score'] for r in tile_results)
        avg_score = sum(r['score'] for r in tile_results) / len(tile_results)
        print(f'Tile {t}: {len(tile_results)} segments, max={max_score:.4f}, avg={avg_score:.4f}')

# Show a full segment JSON example
print()
print('=== FULL MATCH RESULT EXAMPLE (top segment) ===')
top = all_results[0]
path = os.path.join(base, f'pm4_match_{top["tile"]}.json')
with open(path) as f:
    data = json.load(f)
for seg in data['Segments']:
    if seg['SegmentId'] == top['seg_id']:
        print(json.dumps(seg, indent=2)[:3000])
        break
