# MDX Binary Contract — 0.9.1.3810 (Ghidra)

## Scope
Binary model loading contract evidence for geometry/material/animation-adjacent sections in build `0.9.1.3810`.

---

## 1) Function Map

- `0x001a7960` — `MDLFileBinaryLoad(path, sizeOut, status)`
- `0x001a8e34` — `MDLFileBinarySeek(data, size, fourcc)`
- `0x001aa138` — GEOS section parse helper (called by `MdxReadGeosetsA`)
- `0x001a94e0` — `MdxReadTextures`
- `0x001a9398` — `MdxLoadGlobalProperties`
- `0x001c3f38` — `MdxReadLights`
- `0x001c4a58` — `MdxReadCameras`
- `0x001c9a94` — `MDL::ReadBinGeosets`
- `0x001cbb68` — `MDL::ReadBinLights`
- `0x001cdfb4` — `MDL::ReadBinCameras`
- `0x001cecac` — `MDL::ReadBinRibbonEmitters`

Confidence: **High** for function roles listed.

---

## 2) Loader Entry and Fail-Fast (M1)

### `MDLFileBinaryLoad` (`0x001a7960`)
- Reads file bytes, requires file magic `MDLX` (`0x584C444D` after endian conversion).
- If magic mismatch: frees buffer and emits fatal status (`File is not a binary model file`).

### `MDLFileBinarySeek` (`0x001a8e34`)
- Generic chunk scanner:
  - iterate `[fourcc][size][payload]`
  - compare endian-converted fourcc
  - move by `size + 8`
- Returns pointer to section-size dword for matching chunk.

Confidence: **High**
Contradictions: none observed.

---

## 3) Geometry / Material / Texture Contracts

## 3.1 GEOS (geometry)
### Evidence
- `FUN_001aa138` seeks FourCC `GEOS` (`0x534F4547`), allocates geoset data, then post-processes buffers.

### Enforced behavior
- If `GEOS` absent, function returns `0` (upstream path may proceed with limited rendering).
- Geometry memory and buffer creation are explicitly size-checked and allocation-checked.

Confidence: **High**

## 3.2 TEXS (textures)
### Evidence
- `MdxReadTextures` (`0x001a94e0`) seeks `TEXS` (`0x53584554`).
- Enforces `sectionBytes == numTextures * 0x10C`.

### Constants
- Texture section record size: `0x10C`.

Confidence: **High**

## 3.3 MODL global flags
### Evidence
- `MdxLoadGlobalProperties` (`0x001a9398`) seeks `MODL` (`0x4C444F4D`) and reads model flags byte at offset `+0x174`.
- Uses bits:
  - low 2 bits -> model shared property (`param_4 + 0x78`)
  - bit `0x04` toggles load flag mask (`loadFlags &= 0xFFFFFEFF`)

Confidence: **High**

---

## 4) Animation-Adjacent and Effect Sections (M4/M5)

## 4.1 LITE (lights)
- `MdxReadLights` seeks `LITE` (`0x4554494C`).
- Per-emitter variable-size records; validates `sectionBytes >= bytesThisEmitter` and ends with `sectionBytes == 0`.

## 4.2 CAMS (cameras)
- `MdxReadCameras` seeks `CAMS` (`0x534D4143`).
- Per-camera variable-size records; validates overrun and section exhaustion.

## 4.3 Binary reader counterparts
- `MDL::ReadBinLights` (`0x001cbb68`)
- `MDL::ReadBinCameras` (`0x001cdfb4`)
- `MDL::ReadBinRibbonEmitters` (`0x001cecac`)
- All enforce overrun checks and emit structured fatal status on parse failure.

Confidence: **High** (presence + structural behavior), **Medium** (exact field semantics without deeper per-record subdecode pass).

---

## 5) Required/Optional Behavior Summary (`IMdxProfile` seed)

```text
GeometryPolicy:
  RequiresMdlxMagic: true
  GeosetSectionSeek: GEOS
  GeosetHardFailIfMissing: false (degrade path possible)

MaterialPolicy:
  TextureSectionSeek: TEXS
  TextureRecordSize: 0x10C
  TextureSectionSizeStrict: true

AnimationPolicy:
  ModelGlobalsSeek: MODL
  UsesModelFlagByteAt: +0x174
  SequenceCompressionPolicy: unresolved in this pass

TexturePolicy:
  Section: TEXS
  Record-level field semantics: partial

EffectPolicy:
  LightSectionSeek: LITE
  CameraSectionSeek: CAMS
  RibbonSectionReaderPresent: true
```

---

## 6) Contradiction Notes

- No contradiction with existing 0.9.1 docs.
- One open ambiguity: whether absence of `GEOS` is always non-fatal in full call chain (function-local behavior suggests degradable, but top-level caller policy needs full dispatcher proof).

---

## 7) Open Questions (for next pass)

1. Top-level binary section dispatcher order (full required-vs-optional matrix across `SEQS`, `BONE`, `TEXS`, `GEOS`, etc.).
2. Keyframe compression/rotation format policy and tangent requirements.
3. Replaceable texture / UV-wrap exact per-build semantics.

Impact severity:
- (1) parser break
- (2) animation correctness
- (3) visual artifact
