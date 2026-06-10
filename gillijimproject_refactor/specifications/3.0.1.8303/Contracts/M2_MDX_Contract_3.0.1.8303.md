# M2/MDX Contract — 3.0.1.8303

## Scope
Define the executable-backed model parsing contract for `3.0.1.8303` and the delta vs current MDX baseline assumptions (`MdxProfile_091_3810`).

## Baseline reference (current implementation)
- Profile baseline: `MdxProfile_091_3810`
- Key assumptions in baseline profile code:
  - `RequiresMdlxMagic = true`
  - `TextureRecordSize = 0x10C`
  - `TextureSectionSizeStrict = true`
  - Chunk-oriented MDX section flow (`GEOS`/material/animation/texture chains)

## 3.0.1.8303 observed contract (Ghidra)

### 1) Root format identity is `MD20`-family, not classic `MDLX`-style chunk scanning
- Evidence: `0x0079A8C0`
  - Guard: `*param_3 == 0x3032444D` (`'MD20'`)
  - Version gate: `param_3[1] > 0x103 && param_3[1] < 0x109`
- Consequence:
  - Parser starts from a fixed header/table contract, not a token-by-token chunk walk.

### 2) Model data is validated through a large offset/count table with typed span checks
- Evidence: `0x0079A8C0` repeatedly validates header slots via typed validators:
  - `FUN_00797540` => size = `count * 1`
  - `FUN_00797950` => size = `count * 2`
  - `FUN_00797710` => size = `count * 4`
  - `FUN_007975D0` => size = `count * 0x0C`
- Validator internals:
  - `0x00797540`, `0x00797950`, `0x00797710`, `0x007975D0` all enforce:
    - `offset <= fileSize`
    - `offset + count*stride <= fileSize`
    - if `count != 0`, convert relative offset to absolute pointer (`offset += base`)

### 3) Nested records use explicit fixed record sizes (not inferred chunk payloads)
- Evidence (record/block validators):
  - `0x00798DA0` validates a table with stride `0x70` and nested typed arrays
  - `0x007985F0` validates a table with stride `0x2C` and nested typed arrays
  - `0x00799340` validates a table with stride `0xD4` and nested typed arrays
  - `0x0079A720` validates a table with stride `0x7C` and nested typed arrays
- Consequence:
  - Format contract is strongly struct-based and version-gated, not generic chunk-order-based.

### 4) Version split behavior exists inside this parser family
- Evidence: `0x0079A8C0`
  - Branch on `< 0x108` vs `>= 0x108` paths:
    - `<0x108`: uses `FUN_00799EE0(...)`
    - `>=0x108`: uses alternate path `FUN_00799640(...)` + `FUN_00799920(...)`
- Consequence:
  - `3.0.1.8303` must not share one undifferentiated “legacy MDX” parse mode.

### 5) Runtime pipeline identity is `M2Shared/M2Model`
- Evidence:
  - `0x0077D3C0` (`M2Cache.cpp`) normalizes/loads model files (including `.m2` path handling)
  - `0x0079BB30` (`M2Shared.cpp`) calls `0x0079A8C0` then initializes model runtime state
  - `0x00792F80` (`M2Model.cpp`) consumes validated shared model data for runtime structures

---

## Delta vs `MdxProfile_091_3810`

| Contract area | Baseline (`MdxProfile_091_3810`) | 3.0.1.8303 evidence-backed behavior | Impact |
|---|---|---|---|
| Root magic / entry model | MDX-style assumptions (`RequiresMdlxMagic`) | `MD20` header required (`0x3032444D`) | hard parse fail if MDLX-only path |
| Parse strategy | Chunk-centric (`GEOS`/`TEXS` etc profile assumptions) | Header table + typed offset/count spans | hard parse fail or misread buffers |
| Record sizing model | Profile-level chunk sizes (e.g., texture record) | Many explicit struct strides (`0x70`, `0x2C`, `0xD4`, `0x7C`, etc.) | wrong geometry/material/animation data |
| Version handling | Single baseline/provisional profile branch | Explicit internal split around `0x108` | subtle corruption across nearby builds |

---

## Required contract decisions for repository

1. Introduce a separate model profile family for this client line:
   - Suggested IDs:
     - `M2Profile_301_8303`
     - `M2Profile_30x_Unknown` (strict fallback)
2. Do **not** route `3.0.1.8303` through `MdxProfile_091_3810` semantics.
3. Contract fields for the new profile should include:
   - `RequiredRootMagic = MD20`
   - `AcceptedVersionRange = [0x104..0x108]` (inclusive from observed checks)
   - `UseTypedOffsetCountTable = true`
   - `TypedSpanValidation = strict`
   - `NestedRecordStrides = {0x70, 0x2C, 0xD4, 0x7C, ...}`
   - `VersionSplitAt = 0x108`

---

## Code touchpoints (implementation)
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
  - Add model-profile dispatch for `3.0.1.8303` that does not reuse `MdxProfile_091_3810` semantics.
- `src/MdxViewer/Formats/Mdx/*`
  - Guard existing MDX chunk parser behind profile capability checks.
- `src/MdxViewer/Rendering/MdxRenderer.cs`
- `src/MdxViewer/Rendering/MdxAnimator.cs`
  - Ensure only compatible decoded data path is used.

---

## Diagnostics requirements for this contract
Emit and tag at minimum:
- `InvalidChunkSignatureCount` (for legacy path attempts)
- `InvalidChunkSizeCount`
- `MissingRequiredChunkCount`
- `UnknownFieldUsageCount`
- `UnsupportedProfileFallbackCount`

Context tags:
- `build=3.0.1.8303`
- `profileId`
- `filePath`
- `chunkFamily=MDX|M2`

---

## Confidence
- Root magic / parser family (`MD20` + M2Shared path): **High**
- Typed span validation model and stride evidence: **High**
- Exact semantic naming of every header slot in `0x0079A8C0`: **Medium** (structural contract is clear; label names are still provisional)

---

## Evidence index
- `0x0079A8C0` — core model header/table validator (`MD20`, version bounds, typed validators)
- `0x00797540` — generic span check (stride `1`)
- `0x00797950` — generic span check (stride `2`)
- `0x00797710` — generic span check (stride `4`)
- `0x007975D0` — generic span check (stride `0x0C`)
- `0x00798DA0` — nested record table validator (stride `0x70`)
- `0x007985F0` — nested record table validator (stride `0x2C`)
- `0x00799340` — nested record table validator (stride `0xD4`)
- `0x0079A720` — nested record table validator (stride `0x7C`)
- `0x0079BB30` — shared model initialization path (`M2Shared.cpp`)
- `0x0077D3C0` — model cache/load entry (`M2Cache.cpp`)
- `0x00792F80` — runtime model consume path (`M2Model.cpp`)
