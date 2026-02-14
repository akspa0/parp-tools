# ADT Chunk Deep Dive — 0.8.0.3734 vs 0.9.0/0.9.1

## Purpose
Define **chunk-exact** binary structure differences that can break ADT loading between 0.8.0 and 0.9.x, and convert those findings into explicit parser branch points.

This document is intentionally format-first (not performance-first).

---

## Build Coverage
- `0.8.0.3734` (confirmed via assertions string in prior notes)
- `0.9.1.3810` (native function map already captured)
- `0.9.0.x` (target: fill via same workflow; use 0.9.1 as provisional baseline only where validated)

---

## Evidence Anchors (already available)
- `specifications/0.8.0/ADT_Unknown_Field_Resolution_0.8.0.3734.md`
- `specifications/0.8.0/ADT_Semantic_Diff_0.8.0.3734_to_0.9.1.3810.md`
- `specifications/0.9.1/Map_Load_Freeze_Analysis_0.9.1.3810.md`

---

## 1) Root ADT Structure (MVER/MHDR + offset family)

## 1.1 Confirmed stable fields
MHDR practical offset chain remains valid in both analyzed builds:
- `MHDR + 0x04` -> `MCIN`
- `MHDR + 0x08` -> `MTEX`
- `MHDR + 0x0C` -> `MMDX`
- `MHDR + 0x10` -> `MMID`
- `MHDR + 0x14` -> `MWMO`
- `MHDR + 0x18` -> `MWID`
- `MHDR + 0x1C` -> `MDDF`
- `MHDR + 0x20` -> `MODF`

`MHDR + 0x00` remains opaque/unconsumed in referenced create paths.

## 1.2 0.9.x parser behavior to mirror
0.9.1 native path resolves chunk pointers from MHDR offsets and then asserts token identity at each target (`MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF`).

### Parser implication
`StandardTerrainAdapter` should support a strict profile mode where root chunks are offset-driven + token-validated, with no permissive fallback scans for required families.

---

## 2) MCIN and per-chunk addressing

## 2.1 Confirmed
- MCIN entries interpreted as fixed-size records (`size >> 4`, 16-byte chunk info in 0.9.1 notes).
- Chunk preparation path uses MCIN entry location as authoritative pointer into MCNK payload.

## 2.2 Open verification tasks (0.9.0 specifically)
- Verify record size is still 16 bytes in 0.9.0.
- Verify whether any extra per-entry flags/fields are consumed in 0.9.0 compared to 0.9.1.

### Parser implication
Treat MCIN entry shape as profile data:
- `McinEntrySize`
- `McinOffsetFieldOffset`
- optional consumed fields list

---

## 3) MCNK header fields and subchunk contracts

## 3.1 Stable semantic fields (confirmed in 0.8.0 and 0.9.1 notes)
- `+0x10` doodad ref count
- `+0x38` mapobject ref count
- `+0x5C` sound emitter count (`MCSE` loop bound)
- liquid slot gating bits: `0x04`, `0x08`, `0x10`, `0x20`

## 3.2 Inferred/partially confirmed (0.8.0)
- `+0x34` likely `areaId`
- `+0x3C` low 16-bit likely hole/aux mask

## 3.3 0.9.1 strictness contract
Native `CreatePtrs` validates subchunks by expected offsets/tokens:
- `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`

### Parser implication
For 0.9.x profile, use strict per-subchunk token checks at computed offsets before decode, and skip chunk on violation.

---

## 4) MCLQ Deep Diff (critical)

## 4.1 Per-layer stride difference (confirmed)
- `0.8.0`: `0x2D4`
- `0.9.1`: `0x324`
- Delta: `+0x50`

## 4.2 Shared semantics (confirmed)
- Up to 4 liquid layers gated by `MCNK.flags & {0x04,0x08,0x10,0x20}`
- 9x9 sample lattice present; height lane read at sample offset `+4`
- 8x8 tile map semantics; empty tile marker `(tile & 0x0F) == 0x0F`

## 4.3 0.8.0 map (confirmed/incomplete)
Per layer:
- `+0x000`: scalar A (unknown semantic)
- `+0x004`: scalar B (unknown semantic)
- `+0x008`: 9x9 samples (`81 * 8`)
- `+0x290`: tile flags block (64 bytes)
- `+0x2D0`: mode/count scalar
- `+0x2D4`: next layer

## 4.4 0.9.1 map (partially resolved)
- Layer expands to `0x324`.
- Flow payload handling is more explicit (`SWFlowv[2]` conversion evidence in prior notes).
- Early scalars have stronger min/max Z interpretation confidence in 0.9.1 than in 0.8.0.

## 4.5 Required next step for 0.9.0/0.9.1
Produce exact offset table for the additional `+0x50` region:
- field boundaries
- scalar/vector types
- all runtime consumers
- whether region is optional or mandatory per liquid mode

### Parser implication
Define `MclqProfile` by build range:
- `LayerStride`
- `TileFlagsOffset`
- `SampleStride`
- `HeightLaneOffset`
- `HasFlowBlock`
- `FlowBlockOffset`
- `FlowDecodeMode`

---

## 5) Placement chunk chain (MMDX/MMID/MDDF and MWMO/MWID/MODF)

## 5.1 Confirmed chain semantics
- `nameId` in `MDDF/MODF` is an index into `MMID/MWID` tables.
- `MMID/MWID` entries are byte offsets into `MMDX/MWMO` string blocks.

## 5.2 Record sizes (0.9.1 confirmed)
- `MDDF`: `0x24`
- `MODF`: `0x40`

## 5.3 0.9.0 verification task
Confirm record sizes and any field reinterpretations in 0.9.0 before assuming 0.9.1 parity.

### Parser implication
Keep this chain strict and profile record sizes:
- `MddfRecordSize`
- `ModfRecordSize`
- optional field decoders by era

---

## 6) Chunk Matrix (Known vs Unknown)

| Chunk | 0.8.0.3734 | 0.9.0.x | 0.9.1.3810 | Status |
|---|---|---|---|---|
| MVER | required | TODO verify | required | partial |
| MHDR | offset table stable | TODO verify | offset table stable | partial |
| MCIN | 16-byte style implied | TODO verify exact | `size>>4` records | partial |
| MTEX | present/offset-driven | TODO verify strictness | strict token check | partial |
| MMDX/MMID | present | TODO verify | present + index->offset semantics | partial |
| MWMO/MWID | present | TODO verify | present + index->offset semantics | partial |
| MDDF | present | TODO verify size | size `0x24` | partial |
| MODF | present | TODO verify size | size `0x40` | partial |
| MCVT/MCNR/MCLY/MCRF/MCAL/MCSH/MCSE | consumed | TODO verify order/strictness | strict token checks | partial |
| MCLQ | stride `0x2D4` | TODO verify stride | stride `0x324` | partial |
| MH2O | unresolved for era | TODO verify usage | not central in observed 0.9.1 path | partial |

---

## 7) Binary Deep-Dive Task List (for LLM + Ghidra)

For each build (`0.8.0`, `0.9.0`, `0.9.1`), complete all tasks below and cite exact function addresses.

1. **Root ADT parser**
   - locate function equivalent to `CMapArea::Create`
   - capture required token sequence and all offset-origin math

2. **MCIN decoder**
   - identify entry struct size and accessed fields
   - verify chunk pointer derivation formula

3. **MCNK create path**
   - enumerate strict required subchunks and optional branches
   - map all `MCNK` header fields actually consumed

4. **MCLQ decode path**
   - derive full per-layer layout including flow block
   - list runtime query functions that consume tile flags and sample lanes

5. **Placement chain**
   - prove index-vs-offset semantics for name resolution
   - verify record sizes and any endian/byteswap differences

6. **Era-specific chunk gating**
   - identify explicit feature flags/guards for `MH2O`/liquid system variants

---

## 8) Parser Branch Map for MdxViewer (implementation-ready)

Target file: `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`

Required branch points:

1. `ParseAdtRoot(profile, ...)`
   - strict root token+offset validation policy per profile

2. `ParseMcin(profile, ...)`
   - profile-driven MCIN record shape

3. `ParseMcnk(profile, ...)`
   - profile-driven subchunk requirement set

4. `ParseMclq(profile, ...)`
   - profile-driven layer stride and field map

5. `ParsePlacements(profile, ...)`
   - profile-driven MDDF/MODF record size checks

6. `ParseMh2o(profile, ...)`
   - enabled only when profile explicitly supports/needs it

---

## 9) Immediate deliverables to unblock 0.9.x

1. Complete exact `MCLQ 0.9.x` layout table (including the extra `+0x50` region).
2. Complete `0.9.0` verification pass (do not assume complete parity with 0.9.1).
3. Produce `AdtProfile_080` and `AdtProfile_090_091` from validated findings.

---

## 10) Acceptance Criteria

We consider this deep dive complete only when:
- every required chunk for each target build has a validated struct/table,
- every parser offset/count assumption has native evidence,
- unresolved fields are explicitly marked with confidence,
- MdxViewer parser branch points are mapped 1:1 to validated chunk differences.
