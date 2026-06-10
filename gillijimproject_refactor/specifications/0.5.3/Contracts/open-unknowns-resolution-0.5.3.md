# Open Unknowns Resolution — 0.5.3 (Ghidra)

## Scope
Resolves open items from `baseline-diff-0.5.3.md` using direct function-level evidence.

## 1) WDT -> runtime terrain materialization (RESOLVED)

### Proven call chain
1. `0x0067f986` `CMap::Load`
   - opens `%s\%s.wdt`
   - calls `LoadWdt()`
   - calls `PrepareUpdate()`
2. `0x0067fe19` `CMap::LoadWdt`
   - strict chunk order: `MVER` -> `MPHD` -> `MAIN`
   - reads `MAIN` payload into `areaInfo` (`0x10000` bytes)
3. `0x00684071` `CMap::PrepareUpdate`
   - calls `PrepareAreas()` (`0x006840b0`)
   - calls `PrepareChunks()` (`0x006847f0`)
4. `0x006847f0` `CMap::PrepareChunks`
   - uses `areaInfo`/`areaTable` and AOI rects
   - area load call target: `0x00684a30` (`PrepareArea`)
   - chunk load call target: `0x00684be0` (`PrepareChunk`)
5. `0x00684a30` `CMap::PrepareArea`
   - calls `CMapArea::Load(this, &areaInfo[index])` via `0x006aaab0`
6. `0x00684be0` `CMap::PrepareChunk`
   - calls `CMapChunk::Load(this, &chunkInfo[index])` via `0x00698940`

### Conclusion
The runtime terrain materialization handoff is proven and WDT-driven.

---

## 2) ADT compatibility reachability under WDT-primary flow (RESOLVED)

### Evidence
- `0x006aaab0` `CMapArea::Load`
  - file source is `CMap::wdtFile`
  - reads by `SMAreaInfo.offset/size`
  - invokes `Create(this, buffer)` (`0x006aad30`)
- `0x006aad30` `CMapArea::Create`
  - asserts `MHDR` then `MCIN`
  - parses MTEX/MDDF/MODF and chunk table
- `0x00698940` `CMapChunk::Load`
  - file source is `CMap::wdtFile`
  - reads by `SMChunkInfo.offset/size`
  - invokes `Create(this, buffer)` (`0x00698e10`)
- `0x00698e10` `CMapChunk::Create`
  - asserts `MCNK`
  - enforces `MCLY`/`MCRF` and decodes legacy liquid block with stride `0x324`

### Conclusion
There is no requirement for standalone `.adt` file opens in the proven active path.
ADT-era chunk contracts remain active, but payloads are sourced from large WDT-backed blobs.

---

## 3) MDX top-level dispatcher unknown (RESOLVED)

### New evidence
- Canonical root dispatcher identified: `0x00421fc4` `BuildModelFromMdxData`
   - Call order (complex model path):
      1. `MdxLoadGlobalProperties`
      2. `MdxReadTextures`
      3. `MdxReadMaterials`
      4. `MdxReadGeosets`
      5. `MdxReadAttachments`
      6. `MdxReadAnimation` (when flags allow)
      7. `MdxReadRibbonEmitters` (when flags allow)
      8. `MdxReadEmitters2`
      9. `MdxReadNumMatrices`
      10. `MdxReadHitTestData` (flag-gated)
      11. `MdxReadLights` (flag-gated)
      12. `CollisionDataCreate`
      13. `MdxReadExtents`
      14. `MdxReadPositions`
      15. `MdxReadCameras`
- Simple-model canonical dispatcher identified: `0x00422d78` `BuildSimpleModelFromMdxData`
   - Call order (simple path): `Textures -> Materials -> Geosets -> Animation(flag-gated) -> NumMatrices -> CollisionDataCreate -> Extents -> Positions`
- Runtime entrypoints to canonical dispatcher:
   - Blocking load path: `0x00421cec` `IModelCreateBlocking` -> `BuildModelFromMdxData`
   - Async path: `0x00422ffa` `AsnycModelPostLoadCallback` -> `BuildModelFromMdxData`
- `0x004221c6` `MdxReadAnimation` -> `AnimCreate(rawMdxBytes, size, flags)`
- `0x00423789` `MdlReadLoadAnim` -> `AnimCreate(MDLDATA, flags, status)`
- `0x0073a94e` / `0x0073bae3` `AnimCreate` variants -> `AnimBuild`
- `0x0073ab12` / `0x0073bc1c` `AnimBuild` -> `AnimAddSequences`, cameras, geosets, texture anims, material layers
- `MDLFileBinarySeek` xrefs include `MdxReadTextures`, `MdxLoadGlobalProperties`, `MdxReadGeosets`, `MdxReadLights`, `MdxReadCameras`, confirming section-scoped parser family

### Remaining unknown
None for dispatcher identification/order in this pass.

---

## Impact on implementation
- Treat `0.5.3` as `WDT-primary terrain source`.
- Keep ADT chunk field contracts (`MHDR/MCIN/MCNK` etc.) but apply them to WDT-streamed payload buffers.
- Keep MDX profile strictness for known validated contracts (`MODL size=0x175`, sequence/global consistency checks) and use the canonical dispatcher order above as the 0.5.3 section-processing contract.
