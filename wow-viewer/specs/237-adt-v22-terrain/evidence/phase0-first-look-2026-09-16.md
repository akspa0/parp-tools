# Phase 0 first look: the real corpus (2026-09-16)

**Status**: preliminary measurement ahead of the Phase 0 inventory tooling (T011–T017). The Python
probes in [scripts/](scripts/) will be replaced by the C# inventory; the numbers below are what that
tooling must reproduce.

## Corpus

| Item | Value |
|---|---|
| Location | `wow-viewer/test_data/v22_adts/unknown/` (git-ignored) |
| Files | 700 = 699 extensionless + `6893600.adt` (a byte-identical copy of `6893600`; SHA-256 `AC99569C…F199`) |
| Filenames | FileDataIDs 6893600–6894299. **They encode no tile position** (operator-confirmed) and are absent from the vendored listfile (`parts/world-maps.csv` max id 7936286) |
| File-list hash | SHA-256 of sorted `name:size` lines = `54AF5D58…E128E` |
| Provenance | **Public**: shipped in the 2026-09-16 `wow_classic_beta` build (the first WoW: Forever build on Battle.net). Passed on by Marlamin, who identified them at a glance as "v22". They are a **previously undocumented revision** (see the MVER finding below). **No WDT, map table entry or listfile names exist for them** (operator-confirmed) |

Commands (PowerShell, from repo root):

```powershell
python wow-viewer\specs\237-adt-v22-terrain\evidence\scripts\walk_v22.py
python wow-viewer\specs\237-adt-v22-terrain\evidence\scripts\dump_v22.py
python wow-viewer\specs\237-adt-v22-terrain\evidence\scripts\seam_probe.py
```

## Finding 1: the files are MVER 26 + AHDR, not AHDR-first

- Every file (700/700): top-level order `MVER AHDR ALOC AOCH AVTX ANRM [ATEX×n] ADOO×n ACNK×256 ACVT`.
- `MVER` = **26**. `AHDR.version` = **26** (not 22 or 23 as the wiki documents).
- Every file walks with **0 unaccounted bytes** and no overruns, using an unpadded top-level walk.
- **Consequence**: `WowFileDetector` only checks for `AHDR` as the *first* chunk, so it never recognizes these files. They take the `MVER` branch, and the kind they currently receive is unverified. Spec FR-001 must detect by `AHDR` as the second chunk after `MVER`.

## Finding 2: AHDR

The layout matches the wiki's first five fields; one reserved slot is non-zero.

`(26, 129, 129, 16, 16, 8396383, 0 × 10)`, identical in all files.
- `8396383` at +0x14: unexplained. It is in the FileDataID range just above the vendored listfile's coverage. Do not name it until the tile data explains it.

## Finding 3: ALOC is the tile location (MEASURED)

- 20 bytes = 5×uint32: `(2869, a, b, a, b)`. Field 0 is constant 2869. Fields 1/3 equal each other (18–45), and fields 2/4 equal each other (16–40).
- 699 distinct `(a, b)` pairs, one per unique file.
- **Seam proof** (`seam_probe.py`, AVTX outer 129×129 float32, row-major hypothesis):

| Neighbour | Best edge pairing | median \|Δh\| | Runner-up | Pairs |
|---|---|---|---|---|
| field1 + 1 | this.col128 ↔ next.col0 | **0.0000** | 796.9 | 35 |
| field2 + 1 | this.row128 ↔ next.row0 | **0.0000** | 651.5 | 36 |

- **Conclusion**: `ALOC[1]` = tile X (the grid's column axis), `ALOC[2]` = tile Y (row axis). AVTX outer is row-major. Heights are continuous across tile edges with no per-tile offset.
- **Detector power**: all 16 edge pairings × 2 axes were scored; the true seam is exactly 0 and every alternative is ≥650. A wrong axis or orientation cannot pass.
- **Open**: `ALOC[0]` = 2869 (constant, unexplained) and the duplicated fields 3/4. **No WDT, map table entry or listfile names exist for these files (operator-confirmed)**, so any explanation must come from the tile files.

## Finding 4: content shape

- **669/699 tiles are perfectly flat** (height range < 0.001). Only 30 carry terrain. Overall height range is −18798.58..9965.19; the extremes need a check for sentinel values on flat tiles.
- File sizes: 640 files are 400,141 bytes, and those files' `ACNK`s are all header-only (64 bytes). 25 are 533,261 bytes, 5 are 537,881 bytes, and the rest are unique sizes (larger, non-flat tiles).
- Chunk sizes all match the header dimensions: `AVTX` 132100 = (129²+128²)×4, `ANRM` 99075 = (129²+128²)×3, `ACVT` 132100 = (129²+128²)×4 RGBA. The wiki's v23-only `ACVT` is present in every file.
- `AOCH` (new, 2048 bytes): **all zero in all 700 files**.
- `ADST` (new, 12 bytes): present in 321 files, with values like `(63420377, 190719, 1)`. Unexplained.
- `ATEX`: **one chunk per name**. 33 distinct names; 484 files have none. Tilesets are TirisFall, SilverPine and Wetlands (Lordaeron/Eastern Kingdoms content).
- `ADOO`: **one chunk per name**, 225 or 285 per file. The same model names repeat in (almost) every file, so this looks like a map-global model table copied per tile. Examples: `WORLD\LORDAERON\SILVERPINE\...\SILVERPINETREE03.M2`, `...\TIRISFALLGLADECANOPYTREE07.M2`.
- The first non-empty `ACNK` header ints `(0, 0, 53248, 0)`: **the index fields are 0**, so ACNK does not carry tile/chunk position here. Tile position comes from `ALOC` only.

## Spec consequences (applied in the same pass)

1. Detection: `MVER` + `AHDR` second, with the version read from `AHDR` (26 observed). The "v22/v23 by AHDR.version" split is generalized to "AHDR-family revision".
2. Tile coordinates come from `ALOC` (measured), not filenames or ACNK indices.
3. New chunks `ALOC`, `AOCH`, `ADST` join the known set; `ACVT` is not v23-only.
4. Name tables are chunk-per-name.
5. Research R4 (vertex order) is **answered for the outer grid** (row-major, X = column). Research R5 (height frame) is **answered at tile edges** (absolute, continuous). The inner grid, normals, placements and alpha remain open.
