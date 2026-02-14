# Deep Dive Evidence — 0.7.0.3694

## Scope
Focused resolution of prior unknowns from `baseline-diff-0.7.0.3694.md`:
1. ADT `MCLQ` stride/layout
2. MDX top-level binary section handler map
3. ADT `MH2O` participation in native load path

## 1) ADT MCLQ stride/layout (resolved)

### Anchor chain
- `0x00284df8` — `CMapChunk::Create(unsigned char*, int)`
  - calls `CreatePtrs(this, param_2)` then `CreateLiquids(this, param_2)`.
- `0x0028540c` — `CMapChunk::CreatePtrs(int)`
  - hard-asserts `MCLQ` at MCNK offset `+0x60` and stores pointer at `this + 0xF20`.
- `0x0028601c` — `CMapChunk::CreateLiquids(int)`
  - consumes the `MCLQ` data pointed to by `this + 0xF20`.

### Proving constants (from `0x0028601c`)
- Per-layer pointer advance: `puVar18 = puVar18 + 0xB5` dwords
  - stride = `0xB5 * 4 = 0x2D4` bytes.
- Layer field pointers:
  - sample base pointer: `puVar18 + 2` -> byte offset `+0x08`
  - tile/aux base pointer: `puVar18 + 0xA4` -> byte offset `+0x290`
  - tail scalar: `puVar18 + 0xB4` -> byte offset `+0x2D0`
- Layer count policy:
  - iterates up to 4 slots controlled by MCNK flags (`local_60` starts at `0x4`, shifts left each iteration).

### Result
- `MCLQ` for `0.7.0.3694` is confirmed legacy `0x2D4` stride layout.
- Confidence: **High**.

## 2) MDX handler map/order (resolved)

### Loader
- `0x0018eb2c` — top-level binary loader
  - requires `MDLX` magic (`0x584C444D`)
  - loops over sections as encountered in file
  - dispatches each section through `MDL::CallBinReadHandler`.

### Dispatcher
- `0x0018eec8` — `MDL::CallBinReadHandler(unsigned long, CMsgBuffer&, unsigned int, MDLDATA&, CMDLStatus*)`
- Explicit token-to-handler mapping present:
  - `VERS` -> `ReadBinVersion`
  - `MODL` -> `ReadBinModelGlobals`
  - `SEQS` -> `ReadBinSequences`
  - `GEOS` -> `ReadBinGeosets`
  - `MTLS` -> `ReadBinMaterials`
  - `TEXS` -> `ReadBinTextures`
  - `GLBS` -> `ReadBinGlobalSequences`
  - `LITE` -> `ReadBinLights`
  - `BONE` -> `ReadBinBone`
  - `HELP` -> `ReadBinHelpers`
  - `ATCH` -> `ReadBinAttachments`
  - `PIVT` -> `ReadBinPivotPoints`
  - `PREM` -> `ReadBinParticleEmitters`
  - `PRE2` -> `ReadBinParticleEmitters2`
  - `RIBB` -> `ReadBinRibbonEmitters`
  - `EVTS` -> `ReadBinEventObjects`
  - `CAMS` -> `ReadBinCameras`
  - `GEOA` -> `ReadBinGeosetAnim`
  - `TEXA` -> `ReadBinTextureAnims`
  - `CLID` -> `ReadBinCollision`
  - `HTST` -> `ReadBinHitTests`
- Unknown section tags: warning emitted, then section bytes are skipped (non-fatal).

### Additional strictness proofs
- `0x00190d04` — `ReadBinTextures`: requires `sectionBytes % 0x10C == 0`.
- `0x0018f404` — `ReadBinModelGlobals`: requires MODL size `0x175`.
- `0x0016e24c` (geoset parser helper): `VRTX`/`NRMS`/`UVAS` checks including `numVertices == numNormals`.

### Result
- Top-level MDX dispatch map is now explicit and implementation-ready.
- Parsing is file-order driven with token dispatch, not hardcoded global sequence order.
- Confidence: **High**.

## 3) MH2O path status (partially resolved)

### Evidence collected
- String scan in loaded binary: no `MH2O` token string present.
- Function search: no `MH2O`-named parser routines in indexed functions.
- Active ADT chain captured in this pass uses:
  - `CMapArea::Create` (root + placements)
  - `CMapChunk::CreatePtrs` (MCNK subchunks incl. `MCLQ`)
  - `CMapChunk::CreateLiquids` (`MCLQ`-driven liquid decode)

### Result
- No positive proof of `MH2O` usage in the observed native terrain path for `0.7.0.3694`.
- Remaining uncertainty: absence-of-evidence risk (unseen alternate paths may exist).
- Confidence: **Medium-Low** for "not used in active path", **Low** for "not used anywhere".

## Implementation impact summary
- ADT profile should explicitly lock `MCLQ` stride/layout to legacy values:
  - `MclqLayerStride = 0x2D4`
  - `MclqTileFlagsOffset = 0x290`
- MDX profile should encode known dispatcher token map and preserve unknown-section warning+skip behavior.
- Keep `MH2O` as open unknown until a full-function graph pass proves global absence.
