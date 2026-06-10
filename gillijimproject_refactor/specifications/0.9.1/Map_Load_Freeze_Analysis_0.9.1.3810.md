# WoW 0.9.1.3810 Map-Load Freeze Analysis (Ghidra)

## Scope
Direct Ghidra analysis against the currently opened `WoW.exe` to explain map-load freeze risk and derive parser/streaming constraints for:
- `StandardTerrainAdapter.LoadTileWithPlacements -> ParseAdt -> ParseMh2o/CollectPlacementsViaMhdr`
- `TerrainManager.UpdateAOI -> SubmitPendingTiles`

---

## 1) Named Function Map (address → inferred role)

### Top-level world/map load chain
- `0049bfb4` `CWorld::LoadMap(char const*, NTempest::C3Vector&, int, int)`
  - Calls `CMap::Load(mapName, mapId)`.
- `0049c0a0` `CMap::Load(char const*, int)`
  - Initializes map state, calls `LoadWdl`, `LoadWdt`, then `Update(0)`.
- `0028b7c8` `CMap::Update(int)`
  - Calls `UpdateChunks`, `UpdateDoodadDefs`, `UpdateAreas`, `UpdateMapObjDefs`.

### WDT / area setup
- `0049cc98` `CMap::LoadWdt()`
  - Strict token sequence checks (`MVER`, `MPHD`, `MAIN`, optional `MWMO`/`MODF`).
- `002959f4` `CMapArea::Load(char const*)`
  - Reads full ADT (async or sync), then calls `CMapArea::Create`.
- `00295c40` `CMapArea::Create()`
  - Core ADT root parser via `MHDR` offsets; validates token identities.

### Chunk parse and setup
- `0029673c` `CMap::PrepareChunk(CMapArea*, int, int)`
  - Uses MCIN entry to locate chunk data and calls `CMapChunk::Create`.
- `00296cec` `CMapChunk::Create(unsigned char*, int)`
  - Calls `CreatePtrs`, `CreateVertices`, `CreateLiquids`, `CreateRefs`, `CreateLayer` loop.
- `002973c4` `CMapChunk::CreatePtrs(int)`
  - Strict per-subchunk token checks (`MCNK`, `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`).
- `00298e3c` `CMapChunk::CreateRefs(CMapArea*, unsigned long*, unsigned long, unsigned long)`
  - Resolves doodad/WMO refs from per-chunk ref arrays.

### Placement/object pipeline
- `00299144` `CMap::CreateDoodadDef(char const*, SMDoodadDef&, NTempest::C3Vector&)`
- `00299c54` `CMap::CreateMapObjDef(char const*, SMMapObjDef&, NTempest::C3Vector&)`
- `002a0924` `CMap::CreateMapObjDefGroups(CMapObj*, CMapObjDef*)`
- `002a311c` `CMap::CreateMapObjDefGroupDoodads(...)`

### Scheduling / throttling
- `00293bec` `CMap::UpdateChunks()`
  - Has explicit time-budget gating (≈5ms window) for non-priority chunk preparation.
- `0029ec2c` `CMap::UpdateAreas(int)`
- `0029ee84` `CMapArea::Update(int)`
- `0029fd44` `CMap::UpdateMapObjDefs(int)`

---

## 2) Native-Validated Pseudocode and Guards

## 2.1 ADT root parse (`CMapArea::Create` @ `00295c40`)
```text
require CMap::bActive
require data != null

read first chunk header -> must be MVER
advance by MVER.size + 8
read next header -> must be MHDR

byte-swap MHDR offset fields
base = mhdrDataStart (mverData + mverSize + mhdrDataOffset math in code)

resolve pointers using MHDR-relative offsets (+8 to skip each chunk header):
  MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF

assert token at each resolved pointer:
  MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF

convert arrays:
  MCIN entries: size >> 4 (16-byte `SMChunkInfo`)
  MMID/MWID: uint32 tables
  MDDF count: size / 0x24
  MODF count: size / 0x40
```

Observed constants:
- `MDDF` record size = `0x24`.
- `MODF` record size = `0x40`.
- `MMID/MWID` entries are `uint32` offsets.

## 2.2 MCNK and subchunks (`CMapChunk::CreatePtrs` @ `002973c4`)
```text
if byteswapMode: swap headers and arrays
assert root token == MCNK

assert MCVT at mcnk.ofsHeight
assert MCNR at mcnk.ofsNormal
assert MCLY at mcnk.ofsLayer
assert MCRF at mcnk.ofsRefs
assert MCSH at mcnk.ofsShadow
assert MCAL at mcnk.ofsAlpha
assert MCLQ at mcnk.ofsLiquid
assert MCSE at mcnk.ofsSound

decode MCVT(145), MCLY(layerCount), refs(doodadRefCount + mapObjRefCount)
```

Important behavioral property:
- This client path is **assert-heavy and strict-order/token strictness oriented**, not permissive scanning.

## 2.3 Placement indirection chain (`CMapArea::Create` + `CMapChunk::CreateRefs`)
```text
For each chunk doodad ref r:
  doodad = MDDF[r]
  nameId = doodad.nameId
  stringOffset = MMID[nameId]
  modelName = MMDX + stringOffset

For each chunk WMO ref r:
  obj = MODF[r]
  nameId = obj.nameId
  stringOffset = MWID[nameId]
  modelName = MWMO + stringOffset
```

Semantics confirmed:
- `nameId` in MDDF/MODF is **index into MMID/MWID**, not direct byte offset into MMDX/MWMO.
- MMID/MWID values are byte offsets into the corresponding string block.

## 2.4 Work throttling (`CMap::UpdateChunks` @ `00293bec`)
```text
startMs = OsGetAsyncTimeMsPrecise()
for chunks in AOI window:
  if chunk is in high-priority rect:
    prepare chunk immediately
  else:
    now = OsGetAsyncTimeMsPrecise()
    if (now - startMs) < 5ms:
      prepare chunk
    else:
      defer
```

This is explicit anti-stall behavior in native code.

---

## 3) Mismatch Table (Native vs current C#)

| Area | Native 0.9.1 behavior | Current C# behavior | Risk |
|---|---|---|---|
| ADT root chunk discovery | Strict `MHDR`-offset driven addressing + token asserts | `ParseAdt` uses MHDR offsets for MCIN/placements, but still does permissive top-level scan for MTEX and tolerant chunk extraction | Medium |
| MCNK subchunk expectations | `CreatePtrs` requires `MCLQ` and validates many subchunks | `ParseAdt` continues on partial/truncated chunks (`Math.Min(...)`) and catches per-chunk exceptions | High on malformed tiles |
| Placement index semantics | `MDDF/MODF.nameId -> MMID/MWID[nameId] -> MMDX/MWMO+offset` | `CollectPlacementsViaMhdr` matches this and bounds-checks | Low (good) |
| MH2O handling in this era | 0.9.1 path is MCLQ-centric in observed parse chain | `ParseMh2o` still runs when no MCLQ found | Medium-High if MCLQ parse misses and MH2O offsets are garbage-like |
| Per-frame work bounds | `UpdateChunks` has ~5ms soft budget for non-priority prepare | `TerrainManager` uploads up to 8 tiles/frame and can queue broad AOI loads | High (render-thread stall risk) |

---

## 4) Top 3 Freeze Candidates

1. **Render-thread saturation during upload/placement fanout** — **0.82 confidence**
   - Native explicitly budgets chunk prep time; C# can enqueue many tile parses and upload up to 8 tiles/frame with full mesh + liquid + placement notifications.

2. **Permissive/truncated MCNK parse path on malformed 0.9.x data** — **0.77 confidence**
   - Native is strict and fails early via assert expectations.
   - C# proceeds with partial data and per-chunk exception churn; bad offsets can cascade into expensive fallback behavior.

3. **Era-mismatch MH2O fallback when MCLQ path under-detects** — **0.69 confidence**
   - 0.9.1 chain observed in Ghidra is MCLQ-heavy.
   - If `mclqCount == 0` due parser mismatch, `ParseMh2o` may do unnecessary heavy work against incompatible structures.

---

## 5) Minimal Patch Guidance (file+line targets)

## A) Harden MCIN/MCNK guardrails (cheap skip, no partial parse)
- Target: `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, around current MCIN loop and MCNK extraction).
- Changes:
  - Require `mcnkOffsets.Count == 256` (or clamp to 256 with explicit warning once per tile).
  - For each `off`: require `off >= 0`, `off + 8 <= adtBytes.Length`, signature is `KNCM`, `mcnkSize > 0`, and `off + 8 + mcnkSize <= adtBytes.Length`.
  - If any check fails: **skip chunk immediately**; do not `Math.Min`-truncate into parser.

## B) Gate MH2O by era/feature, not by absence of parsed MCLQ
- Target: `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseAdt`, `ParseMh2o` call site).
- Changes:
  - Add explicit 0.9.x mode gate to disable `ParseMh2o` for known MCLQ-era tiles unless a verified MH2O marker is present.
  - Keep current MH2O parser for later eras only.

## C) Reduce per-frame upload pressure to native-like pacing
- Target: `src/MdxViewer/Terrain/TerrainManager.cs` (`SubmitPendingTiles`, `UpdateAOI`).
- Changes:
  - Lower `MaxGpuUploadsPerFrame` for heavy maps (e.g., dynamic budget or default 2-4).
  - Add frame-time aware cap (similar spirit to native 5ms budget) around upload/mesh build loop.
  - Keep queue ordering but avoid large bursts when camera crosses tile boundaries.

## D) Placement deref fail-fast
- Target: `src/MdxViewer/Terrain/StandardTerrainAdapter.cs` (`ParseMddfViaMmid`, `ParseModfViaMwid`, `ResolveNameViaXid`).
- Changes:
  - Preserve current bounds checks, but add counters + one summary warning per tile for invalid `nameId`/offsets to identify pathological assets quickly.

---

## 6) Checklist vs Current C#

- [x] Chunk offsets are bounds-checked in many places, but partial/truncated parse is still allowed.
- [x] MCNK loop is capped to 256.
- [~] Chunk sizes are sanity-checked, but truncation fallback still permits expensive malformed parse paths.
- [~] Optional chunk gating is incomplete for 0.9.x (`MH2O` fallback still active).
- [x] Placement indices are validated before string dereference.
- [~] Failure paths are mostly skip/catch, but repeated exception churn can still be costly.
- [ ] Per-frame work is not bounded as tightly as native (~5ms style budget).

---

## 7) Practical Next Step
Implement A + C first (strict MCNK guardrails + frame budget upload cap), then test freeze tiles; these two changes are most likely to convert hard stalls into bounded degradation.