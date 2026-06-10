# Parser Profile — 0.9.1.3810 (Binary-Derived)

## Purpose
Build a concrete build-level profile from live Ghidra evidence (`WoW.exe`), aligned with `Parser_Profile_Architecture.md` contracts.

---

## Profile ID
- `AdtProfile_091_3810`
- `WmoProfile_091_3810`
- `MdxProfile_091_3810_Provisional`

Build range:
- Exact build: `0.9.1.3810`

Fallback policy:
- Any non-exact 0.9.x build should route to `AdtProfile_090x_Unknown` / `WmoProfile_090x_Unknown` / `MdxProfile_090x_Unknown` until directly validated.

---

## A) ADT Profile Contract (`IAdtProfile`)

## BuildRange
- `0.9.1.3810` only.

## RootChunkPolicy
- `RequireStrictTokenOrder = true`
  - Evidence: `CMapArea::Create` (`0x00295c40`) validates `MVER`, then `MHDR`, then resolves and validates expected chunks via offsets.
- `UseMhdrOffsetsOnly = true`
  - Evidence: all ADT root chunks (`MCIN/MTEX/MMDX/MMID/MWMO/MWID/MDDF/MODF`) are accessed via `MHDR` offsets + `+8` chunk-header skip.

## McinPolicy
- `EntrySize = 0x10`
  - Evidence: `ConvertArrayToBinary<SMChunkInfo>(..., chunkSize >> 4)` in `CMapArea::Create`.
- `OffsetFieldOffset = 0x00` (within each MCIN entry)
  - Evidence: `CMap::PrepareChunk` (`0x0029673c`) computes `entryBase = area->mcinTable + index*0x10`, then uses first dword as raw chunk offset.

## McnkPolicy
- `RequiredSubchunks = [MCVT, MCNR, MCLY, MCRF, MCSH, MCAL, MCLQ, MCSE]`
  - Evidence: `CMapChunk::CreatePtrs` (`0x002973c4`) hard-asserts each token.
- `HeaderFieldMap` (validated usages)
  - `+0x0C` layerCount (used by `ConvertArrayToBinary<SMLayer>`)
  - `+0x10` doodadRefCount (used in `doodadRefCount + mapObjRefCount` decode loop)
  - `+0x14` ofsMCVT
  - `+0x18` ofsMCNR
  - `+0x1C` ofsMCLY
  - `+0x20` ofsMCRF
  - `+0x24` ofsMCAL
  - `+0x2C` ofsMCSH
  - `+0x38` mapObjRefCount
  - `+0x58` ofsMCSE
  - `+0x60` ofsMCLQ

## MclqPolicy
- `LayerStride = 0x324` (per liquid slot)
  - Evidence: `CMapChunk::CreateLiquids` (`0x00297ecc`) advances by `0xC9 dwords` per slot.
- `SampleStride = 0x08`
  - Evidence: vertex source consumed in 8-byte records.
- `HeightLaneOffset = 0x04`
  - Evidence: runtime vertex path reads height from record+4 (from existing deep-dive and chunk liquid creation chain).
- `TileFlagsOffset = 0x290`
  - Evidence: pointer wiring to tile flag region in `CreateLiquids` and liquid query use.
- `FlowBlockPolicy = DualFlowSupported`
  - Evidence: field at block `+0x2D0` controls flow mode (`0/1/2`) and `ConvertArrayToBinary<SWFlowv>(...,2)` is executed.

## PlacementPolicy
- `MddfRecordSize = 0x24`
  - Evidence: `CMapArea::Create` computes count as `size / 0x24`.
- `ModfRecordSize = 0x40`
  - Evidence: `CMapArea::Create` computes count as `size >> 6`.
- `NameIdIndirectionMode = NameIdToXidToStringBlock`
  - Chain: `nameId -> MMID/MWID[nameId] -> byte offset into MMDX/MWMO`.
  - Evidence: `CMapChunk::CreateRefs` and root pointer wiring in `CMapArea::Create`.

## Mh2oPolicy
- `Enabled = false` (for this exact build profile)
- `DetectionMode = None`
  - Evidence: ADT runtime parse path uses strict `MCLQ`; no confirmed MH2O consumer in this binary path.

---

## B) WMO Profile Contract (`IWmoProfile`)

## BuildRange
- `0.9.1.3810` only.

## RootChunkPolicy
- `RequiredRootChunks` (high confidence order):
  - `MVER(0x11), MOHD, MOTX, MOMT, MOGN, MOGI, MOSB, MOPV, MOPT, MOPR, MOVV, MOVB, MOLT, MODS, MODN, MODD, MFOG`
  - Optional trailing: `MCVP`
- Evidence:
  - `CMapObj::CreateDataPointers` at `0x0029ace8` validates ordered tokens and byte-swaps/array-converts.

## GroupChunkPolicy
- `RequiredGroupChunks` = provisional (not fully re-derived in this pass)
- `OptionalGroupChunkGates` = provisional
- Confirmed tokens present in parser strings:
  - `MOGP`, `MOTV`, `MLIQ`
- Evidence:
  - String anchors in binary and prior `0.9.1` WMO docs; root parser strongly strict.

## PlacementPolicy
- `MODD/MODF interpretation profile` = inherits current 0.9.1 docs; no contradictory evidence found in this pass.

## LiquidPolicy
- `MLIQ interpretation profile` = present, but full field map remains provisional in this pass.

---

## C) MDX Profile Contract (`IMdxProfile`)

## BuildRange
- `0.9.1.3810` only (provisional profile).

## Container identity policy (critical for `0.9+`)
- `.mdx` extension is not a sufficient parser selector in this era.
- Parser family must be selected by binary container identity first:
  - `MDLX` => classic chunk-seek MDX contract
  - `MD20`/M2-family => typed offset/count table contract
- If extension and root magic disagree, log diagnostics and continue with magic-selected contract.

## GeometryPolicy
- `GEOS` section is consumed in binary load path.
- Evidence:
  - `FUN_001aa138` seeks FourCC `GEOS` (`0x534F4547`) and allocates geoset buffers.
  - `MdxReadGeosets -> MdxReadGeosetsA<CModelSimple>` chain (`0x001aa0d0`, `0x005c9474`).

## MaterialPolicy
- Material arrays built and validated in model load path.
- Evidence:
  - `MdlReadLoadModel` (`0x001e0b04`) allocates/links `HMATERIAL` and `HMATERIALSHARED` with bounds checks.

## AnimationPolicy
- Sequence/global sequence handling confirmed present.
- Evidence:
  - String/logic evidence for sequence/global sequence sections in model load routines.
- Compression/rotation specifics: not fully extracted in this pass.

## TexturePolicy
- Texture/geoset flow exists, but detailed replaceable/UV/wrap policy remains provisional without deeper per-chunk decode pass.

---

## D) Parser Execution Constraints for Implementation

Use these hard constraints for `AdtProfile_091_3810`:

1. Resolve all root ADT chunks via MHDR offsets only; reject linear best-effort fallback.
2. Require MCIN entry size `0x10`; clamp to 256 chunk entries.
3. Require MCNK subchunk token contract exactly as `CreatePtrs` expects.
4. Treat MCLQ as primary liquid path with stride `0x324`; do not auto-assume 3.3.5/MH2O fallback for this build.
5. Enforce placement indirection (`nameId -> MMID/MWID -> MMDX/MWMO`) and reject out-of-range indices cheaply.
6. Bound per-frame heavy work (native `UpdateChunks` shows ~5ms budget behavior).

---

## E) Diagnostics Fields (minimum)

For this build profile, emit at least:
- `InvalidChunkSignatureCount`
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

Add context:
- `build=0.9.1.3810`
- `profileId=<...>`
- `file=<path>`
- `chunkFamily=<ADT|WMO|MDX>`

---

## F) Confidence by Domain

- ADT profile: **High**
- WMO root profile: **High**
- WMO group/liquid semantics: **Medium**
- MDX profile: **Medium-Low (provisional)**

---

## G) Evidence Index (Ghidra)

- `CMapArea::Create` `0x00295c40`
- `CMap::PrepareChunk` `0x0029673c`
- `CMapChunk::CreatePtrs` `0x002973c4`
- `CMapChunk::CreateLiquids` `0x00297ecc`
- `CMap::UpdateChunks` `0x00293bec`
- `CMapObj::CreateDataPointers` `0x0029ace8`
- `MdxReadGeosets` `0x001aa0d0`, `MdxReadGeosetsA<CModelSimple>` `0x005c9474`
- `FUN_001aa138` (GEOS binary section parse)
- `MdlReadLoadModel` `0x001e0b04`

---

## H) Suggested Registry Entry

```text
FormatProfileRegistry
  ResolveAdtProfile(0.9.1.3810) -> AdtProfile_091_3810
  ResolveWmoProfile(0.9.1.3810) -> WmoProfile_091_3810
  ResolveMdxProfile(0.9.1.3810) -> MdxProfile_091_3810_Provisional
```

Any other `0.9.0.x` / unknown `0.9.x`:
```text
-> *_090x_Unknown (strict parse, warning-heavy, no silent coercion)
```