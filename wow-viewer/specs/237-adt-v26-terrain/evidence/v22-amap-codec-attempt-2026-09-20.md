# Spec 237 Evidence — v22 `AMAP` Codec: Attempt 1 (NOT CRACKED)

Date: 2026-09-20

**Status: OPEN, but now scoreable.** The v22 `AMAP` encoding is not decoded — however a
ground-truth oracle for scoring candidate decodes **is confirmed to exist** (see "Recommended next
attempt"). This records what was eliminated, what the
positive signals are, and the oracle the next attempt should use, so the next session does not repeat
this ground.

## Corpus

1373 cleanly-extracted `AMAP` payloads from the 4 Expansion01 v22 files (12 of the 1385 counted
earlier were rejected by a stricter extractor that requires the `AMAP` to lie wholly inside its parent
`ALYR` — worth investigating separately). Payload sizes 128–3474 bytes, never 4096.

Every `AMAP`-bearing `ALYR` has flags exactly `0x100`. **No flag bit varies**, so there is no
compressed/uncompressed selector the way LK `MCLY` `0x200` works — the encoding is uniform.

## Positive signals (what is almost certainly true)

1. **It is a fill/copy RLE with `0x80` as the fill bit.** The smallest payload is 128 bytes of
   `c0 00` repeated 64 times. Read as MCAL RLE that is `fill 64 × 0x00` × 64 = **exactly 4096 zero
   bytes**. This is the canonical empty-alpha case and it decodes perfectly.
2. **The target is 4096 bytes (8-bit, 64×64), not 2048.** Decoded lengths cluster tightly on 4096
   (4096, 4097, 4098, 4100, 4101, 4102, 4105, 4070 …) rather than scattering. A 2048 target produces
   0 exact hits under every variant tried.
3. **Literal values are small**: the dominant bytes across all payloads are `0x82, 0x01, 0x00, 0x83,
   0x06, 0x02, 0x0f, 0x04, 0x03, 0x05, 0x07, 0x0b` — nearly all ≤ `0x0F`. Alpha in this data is
   low-magnitude, which is consistent with 8-bit values that happen to be small, and is *not* by
   itself evidence of 4-bit packing (the 2048 tests failed).

## Eliminated (measured, not assumed)

| Hypothesis | Test | Result |
|---|---|---|
| v18 `MCAL` fill/copy RLE, target 4096 | full decode + full payload consume | **137 / 1373** — chance level |
| Same, target 2048 | " | **0 / 1373** |
| Fill bit inverted (`0x80` = copy) | grid over target × invert × count+1 × skip 0/1/2/4 | best 0 exact |
| Count = `(c & 0x7f) + 1` | " | best 0 exact |
| Leading header of 1/2/4 bytes | " | no improvement |
| Pure `(count, value)` pair stream | sum of counts == 4096 | **147 / 784** even-length payloads, and **589 payloads are odd-length**, which the hypothesis cannot produce at all |
| zlib/deflate | `0x78` magic scan | 0 / 1373 |

The full grid (2 targets × 2 invert × 2 count × 4 skip = 32 variants) never beat the plain MCAL
reading's 137.

### Detector power

These nulls are meaningful, not vacuous: the same decoder **succeeds** on 137 payloads including the
canonical all-zero case, and decoded lengths land within a few bytes of 4096 on most of the rest. The
test can find the thing; the codec simply differs in a detail.

## Where it diverges — worked example

Payload (283 bytes), first bytes:

```
83 0d 02 ec 85 0f 83 0c 01 07 83 05 01 07 85 05 85 03 01 02 a3 00 82 0d ...
```

MCAL reading: `83 0d` = fill 3 × `0d` (plausible), then `02` = copy 2 bytes → consumes `ec 85`. But
`85` reads naturally as the *next control byte* (fill 5) with `0f` its value. The copy-run length is
absorbing what should be control bytes. So the **copy-run count semantics are wrong**, while the fill
path looks right. Total decode for this payload: 3857 of 4096 from a fully-consumed payload.

Run-length histogram for this payload is dominated by `n = 64` (55 occurrences), then small values —
`64` being half of 128 and a natural row width for a 64×64 map is suggestive and unexplained.

## Recommended next attempt: use the oracle, stop guessing variants

Do **not** brute-force more variants. There is a ground-truth signal already proven on v26:

The `ACNK` header carries a **2-bit 8×8 predominant-layer map at +0x12** (documented in
[adt-v26-format.md](../../../docs/architecture/adt-v26-format.md); on v26 it matched the dominant
`AMAP` layer in 99.93% of 239,808 cells).

**CONFIRMED present in v22 (measured 2026-09-20).** Across all 997 v22 `ACNK` that carry at least one
`ALYR`, **no cell ever names a layer index >= that chunk's layer count — 0 violations**:

| Layers in chunk | Chunks | Max legal cell value |
|---|---|---|
| 1 | 187 | 0 |
| 2 | 360 | 1 |
| 3 | 324 | 2 |
| 4 | 126 | 3 |

798 of the 997 maps have at least one non-zero cell; 199 are all-zero. The 187 single-layer chunks are
the decisive part: every one of their 64 cells must be 0, and every one is. Random bytes could not do
that.

So v22 has a **per-cell answer key**. A candidate `AMAP` decode is correct when the highest-alpha
layer per 8×8 cell matches this map, and wrong when it does not. That turns the codec problem from
guessing variants into a **scored search** — the same method that proved v26's `AMAP` semantics.

Second avenue, if the oracle is inconclusive: Ghidra on a client build that reads v22 DAT, attributing
via string xrefs rather than decompiling callers.

## Impact while this stays open

[`AhdrTerrainAdapter.cs:178`](../../../src/viewer/WoWViewer/Terrain/AhdrTerrainAdapter.cs) requires a
4096-byte map on **every** layer, so v22 produces no alpha at all and renders **layer 0 only**.
Independently of the codec, that guard can be relaxed to blend the layers that do have maps — a small
change that does not depend on cracking anything. It is **not** made here.

## Verification

| Action | Result |
|---|---|
| Strict `AMAP` extraction (must lie inside parent `ALYR`) | 1373 payloads, 128–3474 bytes |
| `ALYR` flag distribution for `AMAP`-bearing layers | all exactly `0x100` |
| 32-variant RLE grid search | best 137/1373 exact (plain MCAL reading) |
| Pure pair-stream test | refuted by 589 odd-length payloads |
| zlib magic scan | 0 hits |
| Canonical empty case (`c0 00` × 64) | decodes to exactly 4096 zero bytes |

**Not claimed:** the codec is not identified, and no decoder is shipped. 12 payloads were rejected by
the strict extractor and have not been explained.
