# Spec 221 Phase 0 — Converter Inventory & Baseline (2026-09-04)

Phase 0 rule honored: **nothing was fixed**. This file is the regression bar.

## Phase 1 addendum — alpha round-trip defects found & fixed (2026-09-04)

Three genuine defects fixed while chasing the Alpha→LK→Alpha alpha failures. Full suite:
**1441 passed / 10 failed** — the 9 known baseline failures plus `AlphaToLk_FlagContract_AllowsAlphaRoundTripThroughLkBytes`,
a new PINNED regression test that is deliberately red while the remaining defect is open.

1. **MCLY flag contract** ([`AlphaToLkConverter`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs)):
   uncompressed 8-bit MCAL layers were flagged `0x200` (RLE-compressed); the comment even said
   "big alpha". Fixed to `0x100`. `0x200` makes consumers RLE-decode raw bytes.
2. **MCAL/MCSH subchunk header stripping** ([`LkAdtReader`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/LkAdtReader.cs)):
   MCNK subchunk offsets land on the FourCC and sizes INCLUDE the 8-byte header, but the reader
   copied `size` bytes from the FourCC — `AlphaMapData` began with the literal bytes `MCAL`+size
   and every alpha byte was shifted 8. Fixed to strip the header (the MCCV/MCLV scan already did).
   Writer offsets verified correct by file probe (reverted an attempted writer-side change that
   broke MCVT — the reader/writer offset conventions agree; only the payload copy was wrong).
3. **Validator resolution-faithful compare** ([`ValidateRoundTripCommand`](file:///I:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/ValidateRoundTripCommand.cs)):
   `AlphaTileData.McalAlphaPack` is the reader's 4× box-downsampled 256 signal while the return
   leg decodes 1024; the compare compared `orig256[y,x]` against `upsample(orig256)[y,x]`,
   reporting hard alpha edges as 1.000 flips. Fixed to box-downsample the round-trip pack to the
   original's resolution first. Verbose per-chunk drift table + LK MCNK byte probe added (`--verbose`).

**Still open**: the synthetic pinned test still drifts 0.9333 (> 2/255) through
pack256 → LK → pack256, so at least one more defect remains in that chain — suspect
[`SliceChunkAlphaBytes`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaToLkConverter.cs)
nearest-upsample (`y * 16 / 64` integer mapping) vs the 1024→256 downsample, or layer-span
inference in [`LkToAlphaConverter`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/LkToAlphaConverter.cs).
Also known and unmeasured: the Alpha→LK leg consumes the LOSSY 256 pack (4× alpha resolution
loss by design of `AlphaTileData`); a proper fix needs full-resolution alpha in the tile model
or MCAL re-decode from `rawChunks`.

**New regression bar**: full suite 1441 passed / 10 failed (9 known + the pinned alpha test);
alpha-mode validator drift counts now change measurably per fix (61876→41127→55753 bad pixels on
tile (0,0)) — use the verbose per-chunk table for the next pass.

## T001 — Inventory (conversion-relevant commands)

| Command | Core owner | Round-trip validator | Unit tests | Test data |
|---|---|---|---|---|
| `convert-alpha-to-lk` | `AlphaToLkConverter` | `validate-roundtrip --mode alpha` | `LkToAlphaRoundTripTests` | synthetic |
| `convert-lk-to-alpha` | `LkToAlphaConverter` | `validate-roundtrip --mode lk` | `LkToAlphaRoundTripTests` | synthetic |
| `convert-split-adt-to-lk` | `LkAdtReader`/`LkAdtWriter` | none | `AdtTerrainWriterTests` | synthetic |
| `convert-wmo-v14-to-v17` | `WmoV14ToV17Converter` | **none** | `WmoV14ToV17ConverterTests` | synthetic |
| `convert-wmo-v17-to-v14` | `WmoV17ToV14Converter` | **none** | `WmoV17ToV14ConverterTests` | synthetic |
| `convert-m2-to-mdx` | `M2ToMdxConverter` | **none** | `M2ToMdxConverterTests` | synthetic |
| `convert-mdx-to-m2` | `MdxToM2Converter` | **none** | `MdxToM2ConverterTests` | synthetic |
| `validate-roundtrip` | terrain only (LK + Alpha modes) | — | — | **real MPQ data** |

Remaining 17 commands are dataset/ML/patch tooling, out of 221's conversion scope for Phase 0.

**Finding**: every object-converter unit test builds bytes in memory (`synthetic_v14_root.wmo`,
`SyntheticCrate`, …). No test in the five suites reads a real client file. The only real-data
instrument in the whole estate is the terrain validator.

## T002 — Terrain round-trip baseline (real MPQ data, committed epsilons hε=0.5 / aε=0.05)

### Alpha mode — Alpha WDT → LK → Alpha, 0.5.3.3368 Azeroth, 16 tiles
- **Pass 2 / Fail 14.** Heights round-trip perfectly: global max Δh = 0.000008 yd.
- **Texture alpha is broken**: global max Δa = 1.000 (full flip). Sample failures:
  `(0,0) orig=0.000 rt=1.000 at 192,0 l=1`; `(2,0) orig=0.571→0.384 / orig=0.000 rt=0.318`;
  `(63,1) orig=0.000 rt=0.898 at 0,0 l=3`. Drift appears at tile-local x=0/192 and chunk edges
  across layers 1–3 — the 4-bit MCAL span/nibble and edge-fix signature documented in Spec 199
  (four independent MCAL decoders).
- **Directional asymmetry**: LK→Alpha→LK is clean (below), so the defect is confined to the
  Alpha→LK→Alpha return leg — `AlphaToLkConverter` re-encode or `LkToAlphaConverter` 4-bit decode.

### LK mode — LK ADT → Alpha → LK, 3.3.0.10958 Azeroth, 8 tiles
- **Pass 8 / Fail 0.** Max Δh = 0.000000, max Δa = 0.000000.

### Tooling note
Client roots nest under `World of Warcraft/` (both 0.5.3 and 3.3.0); the validator accepts the
outer directory silently but finds 0 tiles there. Future `validate-corpus` must normalize the root.

## T003 — Object unit suites
- Filter `WmoV14ToV17|WmoV17ToV14|M2ToMdx|MdxToM2|LkToAlphaRoundTrip`: **30/30 passed**.
- All synthetic (see T001). Green synthetic tests do not exercise the Alpha 4-bit/8-bit MCAL
  mixes, packed MCIN layouts, or MODN offset tables the real corpus carries.

## T004 — Regression bar
1. Terrain LK mode: 100% pass at Δh=0, Δa=0 (8-tile Azeroth slice).
2. Terrain Alpha mode: **known-defective at 2/16** — Phase 1's first duty is to fix the alpha leg
   and raise this bar; until then, any change must not drop below 2/16 or worsen max Δh.
3. Object suites: 30/30 green (synthetic). Phase 1 (221-T101+) replaces this with a real-corpus
   object round-trip validator; the bar becomes corpus-based, not unit-based.
