"""Spec 237: measure ACDO (object definitions) and ADST across the DAT v26 corpus.

Usage: python objects_v26.py <corpus dir> <community listfile csv>
Reports ACDO record sizes, field ranges, model index validity against ADOO, uniqueId overlap with ADST,
and whether ADST fields are FileDataIDs in the listfile.
"""
import collections
import os
import struct
import sys

corpus, listfile_path = sys.argv[1], sys.argv[2]


def walk(b, start, end):
    p = start
    while p + 8 <= end:
        cid = b[p:p + 4].decode('latin1')[::-1]
        sz = struct.unpack_from('<I', b, p + 4)[0]
        if p + 8 + sz > end:
            break
        yield cid, p + 8, sz
        p += 8 + sz


acdo_sizes = collections.Counter()
acdo_per_file = []
records = []  # (file, tileX, tileY, chunkIndex, bytes, adooCount)
adst_rows = []  # (file, tileX, tileY, values)
adoo_counts = collections.Counter()

for name in sorted(os.listdir(corpus)):
    path = os.path.join(corpus, name)
    if not os.path.isfile(path):
        continue
    b = open(path, 'rb').read()
    tx = ty = None
    adoo = 0
    count = 0
    chunk_index = 0
    for cid, off, sz in walk(b, 0, len(b)):
        if cid == 'ALOC':
            _, tx, ty = struct.unpack_from('<III', b, off)
        elif cid == 'ADOO':
            adoo += 1
        elif cid == 'ACNK':
            for sid, soff, ssz in walk(b, off + 0x40, off + sz):
                if sid == 'ACDO':
                    acdo_sizes[ssz] += 1
                    records.append((name, tx, ty, chunk_index, b[soff:soff + ssz]))
                    count += 1
            chunk_index += 1
        elif cid == 'ADST':
            adst_rows.append((name, tx, ty, struct.unpack_from('<%dI' % (sz // 4), b, off)))
    adoo_counts[adoo] += 1
    acdo_per_file.append((name, tx, ty, count, adoo))

files_with_objects = [f for f in acdo_per_file if f[3] > 0]
print('files', len(acdo_per_file), 'with ACDO', len(files_with_objects), 'total ACDO', len(records))
print('ACDO sizes', dict(acdo_sizes))
print('ADOO counts per file', dict(adoo_counts))
print('ADST rows', len(adst_rows), 'in files', len({r[0] for r in adst_rows}))

# Field dump by offset as both uint32 and float for the first records of each size.
for size in sorted(acdo_sizes):
    sample = [r for r in records if len(r[4]) == size][:6]
    print(f'\n== ACDO size {size} samples ==')
    for fname, tx, ty, ci, rec in sample:
        ints = struct.unpack_from('<%dI' % (size // 4), rec)
        floats = struct.unpack_from('<%df' % (size // 4), rec)
        cells = []
        for k in range(size // 4):
            f = floats[k]
            if abs(f) < 1e7 and (abs(f) > 1e-3 or ints[k] == 0) and ints[k] > 0x00FFFFFF or ints[k] == 0:
                cells.append(f'{f:.3f}' if ints[k] != 0 else '0')
            else:
                cells.append(str(ints[k]))
        print(f'  {fname} tile {tx},{ty} chunk {ci}: ' + ' | '.join(cells))

# Per-offset statistics across all records (as float and uint32).
print('\n== per-offset ranges (all records) ==')
max_size = max(acdo_sizes)
for k in range(max_size // 4):
    ints = [struct.unpack_from('<I', r[4], k * 4)[0] for r in records if len(r[4]) > k * 4]
    floats = [struct.unpack_from('<f', r[4], k * 4)[0] for r in records if len(r[4]) > k * 4]
    distinct = len(set(ints))
    print(f'  +0x{k * 4:02X}: n={len(ints)} distinct={distinct} uint[min={min(ints)} max={max(ints)}] float[min={min(floats):.3f} max={max(floats):.3f}]')

# Model index validity.
idx = [struct.unpack_from('<I', r[4], 0)[0] for r in records]
print('\nmodel index range', min(idx), max(idx), 'distinct', len(set(idx)))

# uniqueId candidates vs ADST first field.
adst_first = {r[3][0] for r in adst_rows if r[3]}
for k in range(max_size // 4):
    vals = {struct.unpack_from('<I', r[4], k * 4)[0] for r in records if len(r[4]) > k * 4}
    overlap = len(vals & adst_first)
    if overlap:
        print(f'ACDO +0x{k * 4:02X} values overlapping ADST[0]: {overlap} of {len(adst_first)} ADST ids')

# Listfile check for ADST fields.
wanted = set()
for r in adst_rows:
    wanted.update(r[3])
names = {}
for line in open(listfile_path, encoding='utf-8', errors='replace'):
    i = line.find(';')
    if i <= 0:
        continue
    try:
        fid = int(line[:i])
    except ValueError:
        continue
    if fid in wanted:
        names[fid] = line[i + 1:].strip()
print('\n== ADST ==')
for field in range(3):
    vals = [r[3][field] for r in adst_rows if len(r[3]) > field]
    hits = sum(1 for v in vals if v in names)
    print(f'  field {field}: distinct {len(set(vals))}, min {min(vals)}, max {max(vals)}, listfile hits {hits}/{len(vals)}')
for r in adst_rows[:12]:
    print('  ', r[0], r[1], r[2], r[3], [names.get(v) for v in r[3]])
