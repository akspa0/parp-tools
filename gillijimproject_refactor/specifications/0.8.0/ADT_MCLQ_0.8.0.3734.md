# ADT MCLQ Analysis — WoW 0.8.0.3734

## Summary
In build `0.8.0.3734`, `MCLQ` is still present in `MCNK` and is parsed via fixed-offset references from the MCNK header structure. The parser supports up to 4 liquid entries per chunk controlled by MCNK flags.

## Build
- **Build**: `0.8.0.3734`
- **Source confidence**: High for offsets/control flow, Medium for exact field semantics

---

## MCNK Relationship to MCLQ

### MCNK parser
- **Function**: `FUN_006b8f90`
- Validates MCNK subchunks by offsets in the MCNK header struct.
- Uses MCNK header field at offset `+0x60` as `MCLQ` relative offset.
- Assertion: token at target must be `MCLQ` (`0x4d434c51`).
- Stores resolved pointer in object field `param + 0xf20`.

### Related offsets in same parser
- `MCVT`: header offset field `+0x1c`
- `MCNR`: `+0x18`
- `MCLY`: `+0x1c` (from base pointer path shown in decompile)
- `MCAL`: `+0x24`
- `MCLQ`: `+0x60`
- `MCSE`: `+0x58`

(Exact header struct member naming pending full MCNK struct reconstruction.)

---

## MCLQ Parse Usage

### Liquid object fill routine
- **Function**: `FUN_006b8be0`
- Input source: pointer from `param + 0xf20` (resolved MCLQ payload).
- Checks MCNK flags and conditionally creates up to **4** liquid records.

### Flag gating
- Loop uses masks: `0x04`, `0x08`, `0x10`, `0x20`.
- If flag absent: corresponding liquid slot is freed/zeroed.
- If present: slot is allocated/updated from current MCLQ record.

### Per-record stride and inferred layout
From pointer math in `FUN_006b8be0`:
- Start of record: `puVar4`
- Assigned scalar header values:
  - `record->field0 = puVar4[0]`
  - `record->field1 = puVar4[1]`
- Pointers assigned:
  - `record->pHeights = puVar4 + 2`
  - `record->pTileFlagsOrAux = puVar4 + 0xA4`
  - `record->field2 = *(puVar4 + 0xB4)`
  - `record->pExtra = puVar4 + 0xB5`
- Next record increment:
  - `puVar4 += 0xB5` dwords (`0x2D4` bytes)

So each liquid record is parsed as a fixed-size block of approximately **724 bytes**.

### Refined in-record map (from pointer arithmetic)
`FUN_006b8be0` gives stronger structure hints for each `0x2D4`-byte layer block:

- `+0x000`: dword scalar (`record->field0`)
- `+0x004`: dword scalar (`record->field1`)
- `+0x008`: large sample block (`record->pHeights = base + 0x08`)
- `+0x290`: aux/tile block (`record->pTileFlagsOrAux = base + 0x290`)
- `+0x2D0`: dword scalar (`record->field2`)
- `+0x2D4`: next layer record

The `+0x008..+0x28F` block is `0x288` bytes total, i.e. `81 * 8` bytes. This strongly suggests a `9x9` sample grid with an 8-byte element stride.

---

## Runtime Path for Rotated/Bent Water

### Per-chunk liquid update
- **Function**: `FUN_006b9320`
- Uses chunk-local liquid sample data at `chunk + 0x830 + index*0x0C` and chunk world offsets at `+0x6C/+0x70/+0x74`.
- Computes a depth/sort metric into `chunk + 0x78` via camera basis dot-product:

```text
depth = dot(cameraBasis,
    liquidSample[index] + chunkWorldOffset)
  + cameraBias
```

- Selects render variant index at `chunk + 0x84` (normally `DAT_008c7674`, switches to `DAT_008c7678` under distance/flag condition).

This indicates liquid is not treated as one immutable flat plane; runtime chooses sample points and render mode dynamically per chunk/view.

### Horizon/occlusion sample projection over liquid grid
- **Entry**: `FUN_0068f720` (liquid sort pass) calls `FUN_006c4c80` for eligible entries.
- **Core**: `FUN_006c4c80(param_1)` + `FUN_0068d6a0(vertices, indices, count, worldOffset)`.

Observed behavior:
- Builds 9 indices per pass, then runs **two passes**.
- Uses `0x11` (17) row stride in index construction (`iVar4 += iVar1 * 0x11`), which matches sampling along one edge direction of a 17-wide logical lattice.
- Chooses direction/start based on camera-vs-world comparisons.
- Projects sampled vertices to screen space and updates per-column maxima buffer `DAT_00e96f68`.

Interpretation: this path is a visibility/horizon helper that consumes liquid geometry already carrying local deformation. The screen-space update explains why bent/tilted water can appear correctly clipped/occluded even when not axis-aligned.

### Practical implication for renderer implementations
To reproduce legacy visual behavior, implement liquid as a sampled surface (heightfield-style) rather than a single constant-height quad:

1. Parse up to 4 MCLQ layers per chunk using MCNK bits `0x04/0x08/0x10/0x20`.
2. For each active layer, treat `+0x08` sample region as a fixed 81-sample grid payload (8-byte/sample).
3. Build world-space liquid vertices from chunk origin + per-sample offsets/heights.
4. Derive triangle normals from neighboring samples (or equivalent) so shading/"bend" follows local slope.
5. Apply view-dependent sort/cull steps (depth metric + projected edge/horizon tests) before draw submission.

Steps 2-3 are high-confidence structurally; exact meaning of the second dword in each 8-byte sample remains medium-confidence.

### Confirmed MCLQ runtime query semantics
From direct runtime consumers `FUN_006a8aa0` and `FUN_006a8c80` (world liquid query paths):

- Per-layer tile map comes from `layer + 0x10` (set from `record + 0x290` in parser).
- Tile addressing is `tileByte = tileMap[(y & 7) * 8 + (x & 7)]`.
- `tileByte & 0x0F == 0x0F` means no liquid at that tile.

Bit meaning observed in this build:
- `tileByte & 0x03` = liquid behavior class selector used by query logic:
  - `0`: heightfield surface (interpolated from vertex samples)
  - `1`: special constant-level handling path
  - `2`: alternate math path (`FUN_006a91b0`)
- `(tileByte & 0x40) != 0`: exported by query API as an extra flag (`FUN_006a8aa0` output)
- `tileByte >> 7`: separate high-bit flag returned by the richer query (`FUN_006a8c80`)

### Confirmed 8-byte sample use
`FUN_006a8c80` performs bilinear interpolation for class `0` tiles using the `9x9` sample lattice:

- Lattice index: `v = tileX + tileY * 9`
- Runtime samples four corners via the same 8-byte stride and uses the value at `sample + 4` as height.
- Effective interpolation is:

```text
h00 = sample(v).h
h10 = sample(v + 1).h
h01 = sample(v + 9).h
h11 = sample(v + 10).h
hx0 = lerp(h00, h10, fracX)
hx1 = lerp(h01, h11, fracX)
h   = lerp(hx0, hx1, fracY)
```

This confirms the second dword/`+4` lane in each 8-byte sample is the runtime height term.

### New evidence on 8-byte sample semantics (cross-system)
Tracing WMO `MLIQ` runtime in the same client build (`FUN_006c0810`, `FUN_00695ea0`) shows:
- 8-byte vertex/sample stride is used there too.
- Runtime reads sample `+4` as the effective Z/height term.

By analogy, MCLQ’s `81 * 8` sample block likely follows the same pattern class (sample pair where one element is height, likely at `+4`).

Confidence is now **High** for MCLQ `+4` as height due to direct MCLQ query-path evidence (`FUN_006a8c80`).

---

## Interpretation vs Later Expansions
- This is **pre-MH2O** style parsing (classic `MCLQ` path is active).
- Parser behavior suggests packed, fixed-stride liquid records rather than later flexible MH2O tile schemas.
- Coexistence with `MCSE` (sound emitter chunk) is explicit in same MCNK parser.

---

## What This Answers
- `MCLQ` exists and is required when MCNK header points to it.
- Up to 4 per-chunk liquid layers are supported in this build.
- Layer activation is controlled by MCNK liquid-related flag bits (`0x04..0x20`).
- Record layout is fixed-stride (`0x2D4` bytes/record) in this implementation.
- Runtime uses per-sample liquid data for view/sort/cull decisions, consistent with rotated/bent surfaces rather than strictly flat planes.
- Runtime query code confirms MCLQ is an 8x8 tile map over a 9x9 height lattice with bilinear interpolation for class `0` tiles.

## Unknowns / Follow-up
- Exact semantic name of the first dword in each 8-byte sample (`sample + 0`) is still unresolved.
- Full reconstruction of final draw-triangle assembly function that consumes the `+0x008` sample block directly.
- Precise gameplay meaning of tile bits `0x40` and `0x80` beyond their query-API exposure.

---

## Converter Normalization Rules (`MCLQ`)

Use this profile when targeting WoW `0.8.0.3734` behavior.

### 1) Structural invariants
- Preserve MCNK liquid slot gating bits (`0x04/0x08/0x10/0x20`) and slot data presence consistently.
- Per active slot, keep fixed record stride `0x2D4` bytes.
- Keep sample lattice at exactly `81` entries (`9x9`) with `8` bytes per entry.
- Keep tile block at exactly `64` bytes (`8x8`) at record offset `+0x290`.

### 2) Height/value canonicalization
- Treat sample `+4` lane as authoritative height value.
- Do not swap lane order on import/export.
- For numeric sanitation, clamp NaN/Inf to finite fallback (recommended: neighbor average, else 0).

### 3) Tile byte canonicalization
For each tile byte `b` in the `8x8` block:
- `type = b & 0x03` should remain stable unless intentionally remapped.
- `empty` condition remains nibble-based (`(b & 0x0F) == 0x0F`) for compatibility with query/render logic.
- Preserve flags `0x40` and `0x80` in lossless mode; in compat mode, preserve but log any nonzero usage.

### 4) Slot-level consistency
- If a slot is disabled by MCNK flag, zero or omit its runtime-facing content deterministically.
- If enabled, ensure all of: header fields, 9x9 samples, 8x8 tiles, and trailing dword are present.

### 5) Writer-side validation checklist
Before final write, assert per slot:
- record size is `0x2D4`
- sample block size is `0x288`
- tile block size is `0x40`
- no out-of-range tile addressing assumptions (`8x8` only)
- no NaN/Inf in height lane (`+4`)

### 6) Recommended normalization modes
- `lossless-roundtrip`: preserve unknown fields/flags exactly; repair only hard corruption.
- `compat-0.8.0` (recommended default): preserve type bits and flags, sanitize heights, enforce exact fixed sizes.
- `strict-legacy`: reject records with invalid stride/size or impossible tile-map patterns.

## Confidence
- **High**: MCNK→MCLQ pointer path, 4-slot flag gating, per-layer stride (`0x2D4`), runtime depth/sort/cull call chain.
- **High**: per-tile indexing (`8x8`), per-vertex lattice indexing (`9x9`), and `sample+4` as height.
- **Medium**: exact meaning of `sample+0` and some non-height tile flags.