# WMO Group + MLIQ Contract — 0.9.1.3810 (Ghidra)

## Scope
Binary-derived group parser contract for `MOGP` subtree and `MLIQ` handling in build `0.9.1.3810`.

---

## 1) Function Map

- `0x002a1790` — `CMapObjGroup::CreateDataPointers(unsigned char*)`
  - Parses required group chunks in strict sequence.
- `0x002a1c4c` — `CMapObjGroup::CreateOptionalDataPointers(unsigned char*)`
  - Parses optional chunks gated by group flags, including `MLIQ`.

Confidence: **High**

---

## 2) Required Group Chunk Contract (W2)

From `CMapObjGroup::CreateDataPointers` (`0x002a1790`):

1. `MOPY`
2. `MOVI`
3. `MOVT`
4. `MONR`
5. `MOTV`
6. `MOBA`
7. then optional block parser (`CreateOptionalDataPointers`)

### Evidence snippets (pseudocode)
```text
assert token == 'MOPY'; decode poly flags (size >> 1)
assert token == 'MOVI'; decode indices (size >> 1)
assert token == 'MOVT'; decode vertices (size / 0x0C)
assert token == 'MONR'; decode normals (size / 0x0C)
assert token == 'MOTV'; decode UVs (size >> 3)
assert token == 'MOBA'; decode batches (size / 0x18)
CreateOptionalDataPointers(nextChunk)
```

### Constants/record sizes
- `MOPY`: element size `0x02`
- `MOVI`: element size `0x02`
- `MOVT`: element size `0x0C`
- `MONR`: element size `0x0C`
- `MOTV`: element size `0x08`
- `MOBA`: element size `0x18`

Confidence: **High**
Contradictions: none observed.

---

## 3) Optional Group Chunk Gates (W2)

From `CMapObjGroup::CreateOptionalDataPointers` (`0x002a1c4c`), gated by `groupFlags = *(this+0x0C)`:

- `0x00000200` -> `MOLR` (light refs)
- `0x00000800` -> `MODR` (doodad refs)
- `0x00000001` -> BSP block: `MOBN` then `MOBR`
- `0x00000400` -> Portal-related block: `MPBV`, `MPBP`, `MPBI`, `MPBG`
- `0x00000004` -> `MOCV` (vertex colors)
- `0x00001000` -> `MLIQ` (group liquid)
- `0x00020000` -> strip batch override: `MORI` then `MORB`

### Evidence snippets
```text
if flags & 0x200: assert 'MOLR'
if flags & 0x800: assert 'MODR'
if flags & 0x1:   assert 'MOBN'; assert 'MOBR'; build BSP
if flags & 0x400: assert 'MPBV'; 'MPBP'; 'MPBI'; 'MPBG'
if flags & 0x4:   assert 'MOCV'
if flags & 0x1000: assert 'MLIQ'
if flags & 0x20000: assert 'MORI'; assert 'MORB'
```

Confidence: **High**
Contradictions: none observed.

---

## 4) `MLIQ` Parsing Behavior (W3)

`MLIQ` is parsed only when `MOGP.flags & 0x1000`.

From `0x002a1c4c`:

- Validates token `MLIQ`.
- Reads and endian-converts:
  - two `C2iVector`-like dimension vectors at group offsets `+0xF0..+0xFF`
  - one `C3Vector` origin-like field at `+0x100..+0x10B`
  - one `uint16` type/flags-like field at `+0x10C`
- Sets pointers:
  - `this+0x110` -> liquid sample block start (`param+0x26`)
  - `this+0x114` -> mask/extra block start (`sampleStart + (dimA.x * dimA.y * 8)`)
- Uses first byte at `this+0x114` as mode switch (`& 0x04`) for per-sample decode path.
- Decoding modes:
  - mode A: swaps 32-bit lane(s) in each 8-byte sample
  - mode B: swaps two 16-bit fields + one 32-bit field per sample

### Structural inference
- Per-sample stride: `0x08` bytes.
- Primary sample domain size: `dimA.x * dimA.y`.
- Secondary block size contribution: `dimB.x * dimB.y` bytes (in pointer advance path).

Confidence: **Medium-High**
Contradictions:
- Exact semantic naming for each `MLIQ` header field (e.g., material/type vs flags) is not fully proven from this function alone.

---

## 5) Implementation-Ready `IWmoProfile` Deltas

```text
RequiredGroupChunks:
  [MOPY, MOVI, MOVT, MONR, MOTV, MOBA]

OptionalGroupChunkGates:
  MOLR: flags & 0x00000200
  MODR: flags & 0x00000800
  MOBN/MOBR: flags & 0x00000001
  MPBV/MPBP/MPBI/MPBG: flags & 0x00000400
  MOCV: flags & 0x00000004
  MLIQ: flags & 0x00001000
  MORI/MORB: flags & 0x00020000

LiquidPolicy (proposed):
  Chunk: MLIQ
  SampleStride: 0x08
  ModeBitMask: firstMaskByte & 0x04
  DecodeModes: [UInt32-lane mode, UInt16+UInt16+UInt32 mode]
```

---

## 6) Remaining Open Questions

1. Exact semantic map of `MLIQ` header fields at `+0xF0..+0x10C` (type/material/depth interpretation).
2. Whether `MPB*` and `MORI/MORB` are build-stable into `0.9.0.x` without drift.
3. Full cross-link between group `MLIQ` and scene liquid render/material selection path.

Impact severity:
- (1) visual artifact risk
- (2) parser break risk for nearby builds
- (3) visual/perf risk
