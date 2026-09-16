# Research: ADT v26 — a Brand-New Terrain Format

Primary source: **the 699 v26 tile files themselves**, first seen publicly on 2026-09-16 and first analysed here.
Reference relatives only: <https://wowdev.wiki/ADT/v22>, <https://wowdev.wiki/ADT/v23> (both self-described as incomplete; neither
documents v26). Repo survey on 2026-09-16.

**Standing rule for this spec:** a wiki field name is a hypothesis. Each decision below records what
must be *measured* to confirm it, and how that measurement is shown to have the power to reject a
wrong answer (see memory: "verify detector power before null results", "a name stops the looking").

## Repo baseline (measured 2026-09-16)

- `WowFileDetector` (AHDR branch, `WowFileDetector.cs:119-125`) returns `AdtV23`/`AdtV23Error` for **every** AHDR file; the version is read but never used to pick the kind.
- `AdtV23SummaryReader` decodes 20 bytes of `AHDR` and counts `ACNK`/`ATEX`/`ADOO` chunks. No payload decoding exists anywhere.
- `MapChunkIds` defines `AHDR AVTX ANRM ATEX ADOO ACNK AFBO ACVT`. `ALYR AMAP ASHD ACDO` are absent.
- All AHDR tests (`WowFileDetectorTests`, `MapFileSummaryReaderTests`, `AdtV23SummaryReaderTests`) use synthetic buffers of zeros. No real AHDR file has been parsed.
- Every "V22" hit elsewhere in the repo (`V22Enrich`, `V22ModelPayload`, specs 086–088) is an unrelated **dataset** version, not this file format.

---

### R0: First look at the real corpus (2026-09-16)

**Provenance**: public files from the `wow_classic_beta` build of 2026-09-16 (first WoW: Forever build on Battle.net). **No WDT, map entry or listfile names exist for these files (operator-confirmed)**. The unexplained values (`ALOC[0]` = 2869, `AHDR`+0x14 = 8396383, `AOCH`, `ADST`) can only be studied from the tile files themselves.

See [evidence/phase0-first-look-2026-09-16.md](evidence/phase0-first-look-2026-09-16.md). Summary: revision **26** (`MVER` 26 + `AHDR` 26), new chunks `ALOC`/`AOCH`/`ADST`, `ACVT` in every file,
chunk-per-name `ATEX`/`ADOO`, 669/699 tiles flat, 0 unaccounted bytes. The decisions below are updated where the corpus answered them.

### R1: Version identity

- **Decision (updated after first look)**: Split kinds by `AHDR.version`: `AdtV22`, `AdtV23`, `AdtV26`, `AdtAhdrUnknownVersion`, with `.error` variants where they already exist. Detection accepts `AHDR` first **or** second after `MVER`, and never relies on extension or filename (the corpus is extensionless, named by FileDataID).
- **Rationale**: FR-001. The current "every AHDR file is v23" behavior mislabels exactly the files that just arrived.
- **Alternatives**: a single `AdtAhdr` kind plus a version property. Rejected because every consumer already switches on kind, and a hidden version would be silently ignored the way it is now.

### R2: One reader for the AHDR family (v26 target; v22/v23 relatives)

- **Decision**: A single `AdtAhdrReader`, with version-gated `ACNK` header layout and v23-only `AFBO`/`ACVT`.
- **Rationale**: The wiki layouts match apart from the ACNK header and the two extra chunks.
- **v26 measured**: no `AFBO`; `ACVT` in every file. The v22/v23 claims ("v22 has no `AFBO`/`ACVT`") stay unverified because no real v22/v23 files exist.
- **Measure**: the inventory confirms per revision and that the ACNK header size matches per version.

### R3: Nested chunk walk and padding

- **Decision**: Walk `ACNK` as `[0x40 header if size > 0x40] + sub-chunks`, and `ALYR` as `[0x20 fixed] + optional AMAP`. Pick padding by **byte accounting**: the walk variant that leaves 0 unaccounted bytes across the corpus wins.
- **Why measure**: The repo already carries both padded and unpadded chunk walks (`padOddChunkSizes`), and the wiki says nothing on padding. The wiki also says `AMAP` presence is signalled by `flags & 0x100` in v23 only.
- **Detector power**: A synthetic file with an odd-sized sub-chunk must make the wrong variant report a gap.

### R4: Vertex order (AVTX outer/inner → 9-8-9 interleave)

- **Outer grid ANSWERED (revision 26)**: row-major. `ALOC[1]` runs along the column axis and `ALOC[2]` along the row axis. Seam median |Δh| = 0.0000 on both axes vs ≥650 for all 15 alternatives each (35/36 pairs). **Inner grid still open.**

- **Hypothesis (wiki)**: 129x129 outer block, then 128x128 inner block, row-major.
- **Decision**: Accept the order only after the seam probe. For each candidate order (transpose × flipI × flipJ), measure the median |Δh| along shared edges of adjacent real tiles. The winner must beat the runner-up by a clear margin.
- **Detector power**: Apply a known flip to a tile's grid and confirm its seam score degrades. If the corpus has no adjacent tile pairs, fall back to within-tile outer/inner consistency: an inner vertex should lie near the mean of its 4 outer neighbors.
- **Prior art**: MCVT order was solved against PM4 data (memory: terrain recovery from PM4, "direct flipI flipJ"). Treat that as a candidate, not the answer.

### R5: Height frame

- **Question**: Absolute world heights, or relative to something? v18 MCVT is relative to MCNK `position.z`, but the wiki v22 ACNK header documents no position, and in v26 the ACNK index fields are 0.
- **Partially ANSWERED (revision 26)**: heights are continuous across tile edges with no per-tile offset, so they are absolute at least at tile level. Within-tile chunk-boundary behaviour is still to measure.
- **Decision**: Measure. Seam continuity in absolute terms settles it, since relative heights would produce steps at chunk boundaries inside a tile. The chunk-boundary step statistic inside a tile is the detector.

### R6: Normal encoding (ANRM)

- **Hypothesis (wiki)**: signed-byte triples, 127 = 1.0.
- **Decision**: Pick the component permutation × sign by cosine agreement with normals computed from the decoded heights, and require magnitude ≈ 1 (SC-005).
- **Why**: MCNR component order turned out to be **era-split** (0.5.3 is x,z,y; Cata+ is x,y,z), so the order can't be inherited from either era. Note the wiki size formula for ANRM is mis-parenthesized. The real size should be `(129² + 128²) × 3` bytes, and the inventory must confirm it.

### R7: Alpha map encoding

- **Hypothesis (wiki)**: 8-bit uncompressed or "4-bit RLE", selected by WDT settings. **No WDT exists for this corpus**, so the encoding must be inferred from the data.
- **Decision**: Infer from payload size per map. 4096 means 8-bit 64x64; 2048 means 4-bit 64x64; anything else is treated as compressed (MCAL-style RLE is the first candidate) and flagged. Decode through the existing `AdtMcalDecoder` **without modifying it** (Terrain Alpha Risk Area). Record the chosen encoding and its reason per map (FR-007).
- **Measure**: the size histogram from the inventory. Also check whether 4-bit maps need the legacy edge fix (compare the last row/column against the neighboring chunk).

### R8: Shadow map (ASHD)

- **Decision**: 0x200 bytes = 64x64 bits. Expand the way MCSH is expanded (bit order confirmed by the occurrence of shadows on the terrain's steep, sun-facing-away slopes, not assumed).
- **Note**: MCSH is known not to be visible in minimaps (memory). Validate against terrain slope, not against any minimap.

### R9: Object placements (ACDO)

- **Hypothesis (wiki)**: 0x38 bytes. `modelid` (4) + position (12) + rotation (12) + scale (12) + float (4) + uniqueId (4) = 0x30, so 8 bytes remain for the "name/doodadsets" DWORDs.
- **Decision**: Measure the record size via ACDO payload sizes modulo candidate sizes. Classify each model id's target by `ADOO` name extension (`.m2`/`.mdx` → doodad, `.wmo` → world object).
- **Frame**: Settle position axis order and origin convention by **placement-Z vs decoded terrain height** at that XY. Doodads must sit near the ground in the correct frame; wrong frames scatter. MSVT/MDDF showed opposite field conventions (memory: PM4 frames), so no convention is inherited.
- **Scale**: 3 floats here vs uint16/1024 in MDDF. Check the per-axis distribution. If the axes are always equal, map to uniform scale; otherwise flag it, since the renderer supports uniform scale only.
- **uniqueId**: record it; check for collisions across tiles (the same object straddling tiles).

### R10: Asset resolution

- **Decision (2026-09-16)**: none. This is a never-before-seen engine version with no companions (no WDT, map table, listfile or CDN lookup), so the spec renders from the tile files alone: height/wireframe, layer indices as colours, placements as labelled markers. The inventory still lists the `ATEX`/`ADOO` names as data. Whether any real asset source should ever be attached is a separate future decision, not an assumption here.

### R11: Tile coordinates

- **ANSWERED (revision 26)**: `ALOC` = 5×uint32 `(2869, X, Y, X, Y)`. X = `ALOC[1]` (18–45), Y = `ALOC[2]` (16–40), and 699 distinct tiles. Proven by seam agreement (see R4). Filenames are FileDataIDs with no positional meaning. ACNK index fields are 0 in the corpus, so they are not usable.
- **Open**: `ALOC[0]` = 2869 (constant; unexplained, do not name it); why fields 3/4 duplicate 1/2.

### R12: Real-data tests

- **Decision**: The `WOWVIEWER_AHDR_CORPUS` environment variable points at the corpus root. Real-data tests skip when it is unset. No path is hardcoded (Constitution VI).

### R14: Chunks new in revision 26

| Chunk | Size | Observed | Status |
|---|---|---|---|
| `ALOC` | 20 | every file | tile location, measured (R11) |
| `AOCH` | 2048 | every file, **all bytes zero** | unexplained. 2048 = 64×32, so possibly a per-chunk occlusion/horizon table unused here; don't name it until non-zero data appears |
| `ADST` | 12 | 321/699 files | e.g. `(63420377, 190719, 1)`; unexplained |
| `AHDR`+0x14 | 4 | `8396383` in every file | unexplained; do not name it |

### R13: Liquid

- **Decision**: Out of scope unless Phase 0 finds a liquid-shaped unknown chunk, in which case it becomes a follow-up spec.
