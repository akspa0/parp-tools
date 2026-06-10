# Deep Dive — 3.3.5.12340 ADT/M2/Chunk Contracts + Freeze Analysis

## Scope
- Target build: `3.3.5.12340` (`WoW.exe` in Ghidra).
- Goal: full parser-contract understanding for ADT + M2/MD20 paths and likely freeze causes in viewer.
- Focus: chunk contracts, record-size invariants, MPQ patch-chain behavior, and freeze-risk mapping.

---

## Evidence baseline used
- Native (Ghidra):
  - ADT/WDT chain: `0x007bfce0`, `0x007bf8b0`, `0x007d9a20`, `0x007d6ef0`, `0x007c64b0`, `0x007c3a10`
  - WMO chain: `0x007d80c0`, `0x007d7eb0`, `0x007d7470`
  - M2 chain: `0x0081c390`, `0x0083d410`, `0x0083cf00`, `0x00838490`, `0x0083cc80`
- Current implementation paths:
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - `src/MdxViewer/Terrain/WorldAssetManager.cs`
  - `src/MdxViewer/DataSources/MpqDataSource.cs`
  - `src/WoWMapConverter/WoWMapConverter.Core/Services/NativeMpqService.cs`
  - `src/WoWMapConverter/WoWMapConverter.Core/Converters/M2ToMdxConverter.cs`
  - `src/MDX-L_Tool/Formats/Mdx/MdxFile.cs`

---

## ADT/WDT contract (native)

## 1) Tile root load chain
- `0x007bfce0` builds map root `%s\\%s.wdt` and initializes terrain systems.
- `0x007bf8b0` loads WDT and reads fixed blocks in order (header + map flags + tile bitset + optional map object block).
- `0x007d9a20` builds per-tile `%s\\%s_%d_%d.adt` and dispatches `0x007d7150` tile read.

## 2) ADT root pointer model
- `0x007d6ef0` is the key ADT data-pointer initializer.
- It derives all major pointers from root header offset table (`+8` data-start adjustment), including model name/id blocks and placements.
- It sets:
  - model-string/id tables,
  - placement arrays,
  - optional block at root-offset slot `+0x24` gated by flags,
  - optional trailing `+0x2c` block,
  - count derivations from chunk sizes.

### ADT implication
- This confirms **offset-table driven parsing** for root-level contract.
- Any malformed offset or patch-mismatched table can mis-point many downstream arrays in one step.
- Using wowdev ADT naming (`SMMapHeader`), neighboring offsets in this region align with `MFBO`, `MH2O`, and `MTXF` slots in WotLK-era root ADTs; this improves field naming even where native branch semantics still need per-path proof.

## 3) MCNK subchunk discovery
- `0x007c3a10` scans MCNK payload from `+0x88` onward by FourCC and captures pointers.
- Observed recognized subchunks in this scanner:
  - `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCCV`, `MCSE` (conditional),
- Special handling:
  - `MCNR` forced consumed size behavior (client-side normalization pattern exists),
  - `MCLQ` may use header-derived size path when relevant.

## 4) MCNK placement/registration path
- `0x007c64b0` binds parsed MCNK into tile/global structures and computes world-space bounds/positioning.
- `0x007b5950` governs active area chunk load/unload and can trigger tile chunk creation on demand.

---

## WMO contract snapshot (native)

## Root/group table derivation
- `0x007d7470` parses WMO root data pointers and computes record counts via fixed divisors:
  - `/0x0C`, `/0x14`, `/0x28`, `/0x30`, `>>5`, `>>3`, `>>2` patterns.
- Optional trailing `MCVP` recognized via `0x4D435650` check.
- Group flag normalization: when a specific root name/string block is absent, bit `0x00040000` is cleared over group records.

### Implication
- WMO reader is heavily schema-coupled to exact entry sizes; patch-mismatched roots/groups can produce misaligned arrays and expensive fallback paths.

---

## M2 (MD20) contract (native)

## 1) Entry/extension gate
- `0x0081c390` enforces extension policy:
  - valid model extensions accepted,
  - `.mdx` / `.mdl` normalized to `.m2`,
  - others rejected (`Model2: Invalid file extension`).
- File-not-found and invalid-extension are explicit hard-fail branches.

## 2) Root magic/version gate
- `0x0083cf00` validates model root and table contracts.
- Requires magic `0x3032444D` (`MD20`).
- Version window in this observed path is tightly constrained (`>0x107` and `<0x109`).

## 3) Section reader contract style
- `0x0083cf00` dispatches many helpers that validate and fix up `(count,offset)` tables.
- Helper evidence (record stride expectations):
  - `0x00835df0` -> `count*2`
  - `0x00835c20` -> `count*4`
  - `0x00835bd0` -> `count*0x0C`
  - `0x00835ae0` -> `count*0x30`
  - `0x00835c70` -> `count*0x40`
  - `0x00836b60` -> `count*0x10` (+ nested tables)
  - `0x008382a0` -> `count*0x14` (+ nested)
  - `0x00838b10` -> `count*0x3C` (+ multiple nested animation blocks)
  - `0x00839080` -> `count*0x28` (+ nested animation blocks)

## 4) Skin profile path
- `0x0083cc80` chooses skin profile and allocates texture array.
- `0x00838490` validates skin profile structure and can hard-fail on corrupt skin data.
- Sidecar naming behavior is consistent with `%02d.skin` generation (`0x00835a80` and related).

### Implication
- M2 parse and render is strongly bound to validated `(count,offset,stride)` tables plus skin/profile constraints.
- Any conversion pipeline that relaxes these rules risks producing large invalid loops/allocations downstream.

---

## Current implementation vs native contract (important mismatches)

## ADT pipeline
- `StandardTerrainAdapter` already has robust MCNK bounds guards and MHDR-based root placement path.
- However, `Mcnk` parser (`WoWMapConverter.Core`) mixes cross-build logic and relies on best-effort scanning/fallback behavior.
- `Mcal` implementation is marked with TODO/porting uncertainty and includes ambiguous flag semantics (`Sign` vs `CompressedAlpha` overlap).

## M2 pipeline
- `WorldAssetManager` converts MD20 to MDX via `M2ToMdxConverter` before loading with `MdxFile.Load`.
- Converter trusts many count/offset values with minimal global caps.
- Result: malformed/patched M2 or skin can inflate loops/allocations and appear as freeze.

## MPQ patch path
- `NativeMpqService` reads archives patch-last/search-reverse (correct precedence intent).
- But when patch entry has `FileSize==0`, it continues to older archives (base fallback).
  - This can resurrect a base file intentionally deleted/invalidated by patch chain.
  - That can create **cross-archive mismatches** (e.g., root from patch-era + sidecar from base-era).

---

## High-probability freeze vectors

## Vector A — MPQ sector table trust without sanity gates
File: `NativeMpqService.ReadFileData`
- Sector offset table values are used directly to compute `compressedSize` and seek/read boundaries.
- Missing explicit validations for monotonic offsets, offset bounds inside block, and sane compressed-size caps.
- A malformed patch block can trigger very large/degenerate sector reads and long decompression attempts.

## Vector B — Patch deletion fallback to base archive
File: `NativeMpqService.ReadFile`
- On `FileSize==0` in patch archive, code continues searching older archives.
- This may violate intended patch semantics and load stale base assets.
- M2/skin/texture mismatch can then cascade into expensive conversion or repeated fallback probing.

## Vector C — M2 converter trusts unbounded counts/offsets
File: `M2ToMdxConverter.ParseM2` / `ParseSkin`
- Reads multiple arrays from file-driven counts/offsets with limited global cap checks.
- Corrupted counts can allocate huge arrays or force long iteration.
- If this happens on the streaming path, UI appears frozen.

## Vector D — Excessive fallback probes + verbose logging under load
Files: `WorldAssetManager.ReadFileData`, `NativeMpqService.ReadFile`
- Multi-variant path probing (`original/normalized/lower/upper/prefix/extension swap`) multiplied by many placements.
- Console logging on each MPQ miss/hit amplifies stall when thousands of reads occur.

## Vector E — Mixed chunk profile behavior in MCAL/MCLQ handling
Files: `Mcnk.cs`, `Mcal.cs`, `StandardTerrainAdapter.cs`
- Cross-build heuristics can force expensive decode attempts on invalid data.
- MCAL decode uncertainty can drive repeated texture-layer decode fallbacks.

---

## “All chunks” contract snapshot for this deep dive

## ADT root-level (observed path)
- Offset-table referenced blocks include:
  - terrain index/data tables,
  - texture/model name + id tables,
  - placement blocks,
  - optional liquid/aux blocks.

## MCNK subchunks recognized by scanner
- `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCCV`, `MCSE`.
- Pointer capture and some conditional gates confirmed by `0x007c3a10`.

## WMO root/group (snapshot)
- Fixed-divisor table parsing confirms rigid record contracts.
- Optional `MCVP` detected.
- Group-flag normalization branch exists.

## M2/MD20 sections (snapshot)
- MD20 root + strict version gate.
- Many typed count/offset tables with fixed stride readers.
- Skin profile selection/validation mandatory for full render path.

## Second-pass proof updates

## A) `0x0083cf00` section-reader map (symbolic)
The validator walks `(count,offset)` pairs in header order and dispatches typed readers.

High-confidence core map:
- `piVar1+2`  (`idx 2/3`)  -> `nameCount/nameOfs`            via `0x00835b80` (`count*1` bytes)
- `piVar1+5`  (`idx 5/6`)  -> `globalSeqCount/globalSeqOfs`  via `0x00835c20` (`count*4`)
- `piVar1+7`  (`idx 7/8`)  -> `animCount/animOfs`            via `0x00835c70` (`count*0x40`)
- `piVar1+9`  (`idx 9/10`) -> `animLookupCount/animLookupOfs` via `0x00835df0` (`count*2`)
- `piVar1+0xb` (`idx 11/12`) -> `boneCount/boneOfs`          via `0x008385a0` (`count*0x58`)
- `piVar1+0xd` (`idx 13/14`) -> `keyBoneLookupCount/keyBoneLookupOfs` via `0x00835df0` (`count*2`)
- `piVar1+0xf` (`idx 15/16`) -> `vertexCount/vertexOfs`      via `0x00835ae0` (`count*0x30`)
- `piVar1+0x12` (`idx 18/19`) -> likely color blocks          via `0x00837ee0` (`count*0x28`, nested anim refs)
- `piVar1+0x14` (`idx 20/21`) -> textures                     via `0x00836b60` (`count*0x10`)

Lookup/table phase (third-pass named, high confidence):
- `+0x16` -> `texture_weights` (`M2TextureWeight`, `count*0x14`) via `0x008382a0`
- `+0x18` -> `texture_transforms` (`M2TextureTransform`, `count*0x3c`) via `0x00838b10`
- `+0x1a` -> `textureIndicesById` (`replacable_texture_lookup`, `uint16`, `count*2`) via `0x00835df0`
- `+0x1c` -> `materials` (`M2Material`, `count*4`) via `0x00835c20`
- `+0x1e` -> `boneCombos` (`bone_lookup_table`, `uint16`, `count*2`) via `0x00835df0`
- `+0x20` -> `textureCombos` (`texture_lookup_table`, `uint16`, `count*2`) via `0x00835df0`
- `+0x22` -> `textureCoordCombos` (`texture_mapping_lookup_table`, `uint16`, `count*2`) via `0x00835df0`
- `+0x24` -> `textureWeightCombos` (`transparency_lookup_table`, `uint16`, `count*2`) via `0x00835df0`
- `+0x26` -> `textureTransformCombos` (`texture_transforms_lookup_table`, `uint16`, `count*2`) via `0x00835df0`

Late sections (now named via M2 header crosswalk, high confidence):
- `+0x36` -> `collisionIndices` (`uint16`, `count*2`) via `0x00835df0`
- `+0x38` -> `collisionPositions` (`C3Vector`, `count*0x0c`) via `0x00835bd0`
- `+0x3a` -> `collisionFaceNormals` (`C3Vector`, `count*0x0c`) via `0x00835bd0`
- `+0x3c` -> `attachments` (`M2Attachment`, `count*0x28`, nested refs) via `0x00839080`
- `+0x3e` -> `attachmentIndicesById` (`uint16`, `count*2`) via `0x00835df0`
- `+0x40` -> `events` (`M2Event`, `count*0x24`) via `0x00836e40`
- `+0x42` -> `lights` (`M2Light`, `count*0x9c`, heavy nested refs) via `0x00839270`
- `+0x44` -> `cameras` (`M2Camera`, `count*0x64`) via `0x00839ef0`
- `+0x46` -> `cameraIndicesById` (`uint16`, `count*2`) via `0x00835df0`
- `+0x48` -> `ribbon_emitters` (`M2Ribbon`, `count*0xb0`) via `0x0083a460`
- `+0x4a` -> `particle_emitters` (`M2Particle`, `count*0x1dc`) via `0x0083af90`
- optional `+0x4c` -> `textureCombinerCombos` (`uint16`, `count*2`) via `0x00835df0`, gated by global flag `flag_use_texture_combiner_combos` (`0x08`)

Named-offset anchor (byte offsets in `MD20` header):
- `0x0D8` collision indices, `0x0E0` collision vertices, `0x0E8` collision normals
- `0x0F0` attachments, `0x0F8` attachment lookup
- `0x100` events, `0x108` lights, `0x110` cameras, `0x118` camera lookup
- `0x120` ribbons, `0x128` particles, optional `0x130` second texture combiner combos

Third-pass anchor (mid-table offsets, now named):
- `0x058` texture weights, `0x060` texture transforms
- `0x068` replacable texture lookup, `0x070` materials
- `0x078` bone lookup, `0x080` texture lookup
- `0x088` texture mapping lookup, `0x090` transparency lookup, `0x098` texture transform lookup

Interpretation: this is a strict, typed `MD20` header contract with dozens of fixed-size sections and nested animation/keyframe tables. Any count/offset corruption can explode work nonlinearly.

## B) ADT root `+0x24` block proof result
Result: in this path it is **not** behaving like MH2O data consumption.

Evidence:
- `0x007d6ef0` checks `*(pbVar1 + 0x28)` and if referenced block size is non-zero, allocates an object tagged from `".\\MapArea.cpp"` and calls `0x007d4f10` with that block pointer.
- Downstream consumer `0x007d5e70` builds/updates renderable grids using short height-style samples and mask-conditioned quad emission via `0x007d5150` / `0x007d5240`.
- This behavior matches map-area/overlay mesh generation semantics, not MH2O tile/layer water contracts.

So for this build/path, the ADT optional root slot around `+0x24/+0x28` should be documented as **MapArea auxiliary block path (non-MH2O)** until contradictory evidence appears in another loader branch.

Crosswalk note for naming: under wowdev `SMMapHeader`, nearby slots are `mfbo` (`0x24`), `mh2o` (`0x28`), and `mtxf` (`0x2C`) when interpreted from the full MHDR struct layout. This supports naming the observed auxiliary path as MFBO/overlay-adjacent rather than liquid-layer ingestion.

---

## Immediate documentation-backed guardrail plan (for freeze triage)

1. MPQ sector sanity checks (must-have)
- Validate sector offsets are:
  - monotonic non-decreasing,
  - within block bounds,
  - `sectorEnd >= sectorStart`,
  - `compressedSize` capped by remaining block bytes.
- Abort file read on violation; increment diagnostics.

2. Patch deletion semantics (must-have)
- Treat `FileSize==0` in higher-priority patch as authoritative delete (do not fallback to base automatically), behind a feature toggle if needed for compatibility tests.

3. M2 conversion hard caps (must-have)
- Add max caps for key counts before allocations (vertices, bones, textures, anims, skin indices/triangles/submeshes/texture units).
- Validate each `(count,offset,stride)` range against stream length before reading.

4. Parser budget instrumentation (must-have)
- Per file timing + counters:
  - parse time,
  - fallback attempts,
  - decompression failures,
  - oversized-table rejects,
  - profile fallback count.

5. Logging throttling (should-have)
- Promote per-file summary logs, demote per-probe line spam in hot paths.

---

## Diagnostics fields to add/verify for this build
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`
- `MpqSectorTableInvalidCount`
- `MpqPatchedDeleteHitCount`
- `M2TableValidationRejectCount`
- `M2ConversionFailureCount`

Log context:
- build id
- profile id
- virtual path
- archive source (patch/base/loose)
- chunk family (`ADT|WMO|MDX|M2|MPQ`)

---

## Open proof tasks (next pass)
1. Validate patch delete semantics against retail behavior with controlled archive fixtures.
2. Build corpus of failing files (freeze repro set) and correlate with new diagnostics.

Status update:
- MCNK signature-validation path resolved (`0x007c64b0 -> 0x007c3a10`) and no longer an open unknown.
- WMO root divisor-chain labeling around `0x007d7470` resolved and documented.
- Diagnostics counters listed in this report are now implemented in code (`Build335Diagnostics` sink + ADT/MPQ/M2 instrumentation).

---

## Appendix — MD20 header contract crosswalk (`0x0083cf00`)

`(count,offset)` pair index map used by the validator (resolved in this report):
- `+0x14` textures (`M2Texture`, `0x10`)
- `+0x16` texture weights (`M2TextureWeight`, `0x14`)
- `+0x18` texture transforms (`M2TextureTransform`, `0x3c`)
- `+0x1a` replacable texture lookup (`uint16`)
- `+0x1c` materials (`M2Material`, `0x04`)
- `+0x1e` bone lookup (`uint16`)
- `+0x20` texture lookup (`uint16`)
- `+0x22` texture mapping lookup (`uint16`)
- `+0x24` transparency lookup (`uint16`)
- `+0x26` texture transform lookup (`uint16`)
- `+0x36` collision triangles (`uint16`)
- `+0x38` collision vertices (`C3Vector`, `0x0c`)
- `+0x3a` collision normals (`C3Vector`, `0x0c`)
- `+0x3c` attachments (`M2Attachment`, `0x28`)
- `+0x3e` attachment lookup (`uint16`)
- `+0x40` events (`M2Event`, `0x24`)
- `+0x42` lights (`M2Light`, `0x9c`)
- `+0x44` cameras (`M2Camera`, `0x64`)
- `+0x46` camera lookup (`uint16`)
- `+0x48` ribbons (`M2Ribbon`, `0xb0`)
- `+0x4a` particles (`M2Particle`, `0x1dc`)
- optional `+0x4c` second texture material override combo (`uint16`, `global_flags & 0x08`)

---

## Conclusion
- Native contracts show strict, stride-driven parsing for both ADT subchunks and M2 tables.
- Viewer freeze is highly plausible from **patch-chain + parser-trust interaction**:
  - malformed or intentionally deleted-patch entries,
  - fallback to stale base assets,
  - large unbounded conversion/decode work.
- Highest-value fix path is MPQ read validation + patch-delete semantics + M2 table caps, then re-run with diagnostics to pinpoint offending files.