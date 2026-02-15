# M2/MDX Contract — 0.11.3925

## Scope
Executable-backed model parsing contract for build `0.11.3925`, with explicit delta against `0.9.1.3810` MDX assumptions.

## Baseline reference (what broke)
- `0.9.1.3810` contract is classic MDX chunk flow:
  - root magic `MDLX`
  - `MDLFileBinarySeek` over `[fourcc][size][payload]`
  - `TEXS` strict record size `0x10C`
- In `0.11.3925`, model loading is a different parser family and rejects those assumptions.

## 0.11.3925 observed model contract (Ghidra)

### 1) Root identity and version gate are strict `MD20 + v0x100`
- Evidence: `FUN_0070d500` (`0x0070d500`)
  - Requires `*param_3 == 0x3032444D` (`MD20`)
  - Requires `param_3[1] == 0x100` exactly
  - Fails hard (`return 0`) on mismatch
- Additional structural gate:
  - Requires header span through at least `param_3 + 0x51` (`0x144` bytes) before table validation.

### 2) Parse strategy is typed offset/count table validation, not chunk-seek
- Evidence: `FUN_0070d500` calls a chain of typed validators against header offset/count pairs.
- Core typed span validators:
  - `FUN_0070e680` / `FUN_0070eea0` / `FUN_0070ff40`: `count * 4`
  - `FUN_0070e350` / `FUN_0070fe40`: `count * 2`
  - `FUN_0070e450`: `count * 0x0C`
  - `FUN_0070fdc0`: `count * 0x10`
  - `FUN_0070f460`: `count * 8`
  - `FUN_0070ffc0`: `count * 0x24`
  - `FUN_0070fec0`: `count * 1`
- Enforced in all validators:
  - `offset <= fileSize`
  - `offset + count*stride <= fileSize`
  - if `count == 0`, pointer slot set to `0`; otherwise rebased to `fileBase + offset`

### 3) Nested tables are fixed-stride structs with recursive validation
- Evidence (called by `FUN_0070d500`):
  - `FUN_0070e4d0` => table stride `0x6C` + nested typed arrays
  - `FUN_0070e780` => table stride `0x2C` + nested validator `FUN_0070d420`
  - `FUN_0070e830` => table stride `0x38` + nested typed arrays
  - `FUN_0070e9c0` => table stride `0x10`
  - `FUN_0070ea90` => table stride `0x1C`
  - `FUN_0070ebb0` => table stride `0x1C`
  - `FUN_0070ecd0` => table stride `0x54`
  - `FUN_0070ef20` => table stride `0x30`
  - `FUN_0070f050` => table stride `0x2C`
  - `FUN_0070f140` => table stride `0xD4`
  - `FUN_0070f4e0` => table stride `0x7C`
  - `FUN_0070f690` => table stride `0xDC`
  - `FUN_0070f970` => table stride `0x1F8`

### 4) Extension handling coerces legacy names into this M2 path
- Evidence: `FUN_00710890` (`0x00710890`)
  - `.mdx` / `.mdl` suffixes are rewritten before dispatch to shared loader path.
- Loader path evidence:
  - `FUN_00706a40` -> `CM2Model` construction
  - `FUN_0070db30` + `FUN_0070dcc0` -> async read and `FUN_0070d500` validation

## Why `0.9.x` MDX viewer behavior breaks
1. `MDLX` chunk-seek assumptions are incompatible with `MD20` fixed header/table contract.
2. Texture/material/animation data is not discovered via `TEXS`/`GEOS` chunk scanning; it is read from typed spans and nested structs.
3. Version gate is exact (`0x100`) in this build; profile ranges tuned for later M2 (`0x104..0x108`) will reject or mis-handle this binary.

## Required profile decisions for repository
1. Add dedicated model profile for this build line:
   - Suggested profile ID: `M2Profile_011_3925`
2. Required fields:
   - `RequiredRootMagic = MD20`
   - `MinSupportedVersion = 0x100`
   - `MaxSupportedVersion = 0x100`
   - `UseTypedOffsetCountTable = true`
   - `StrictSpanValidation = true`
3. Nested stride contract snapshot for enforcement:
   - include at least `0x2C`, `0x30`, `0x38`, `0x54`, `0x6C`, `0x7C`, `0xD4`, `0xDC`, `0x1F8`
4. Do not route `0.11.3925` through `MdxProfile_091_3810` chunk semantics.

## Code touchpoints
- `src/MdxViewer/Terrain/FormatProfileRegistry.cs`
  - `ResolveModelProfile` (add exact build mapping)
  - model profile definitions (new `M2Profile_011_3925`)
- `src/MdxViewer/Terrain/WorldAssetManager.cs`
  - diagnostics tags for extension coercion and model-profile fallback
- Legacy MDX parsing path should remain profile-gated away from this build.

## Confidence
- Root magic/version gate (`MD20`, `0x100`): High
- Typed offset/count validator model: High
- Nested fixed stride table contract: High
- Exact semantic names for every header slot: Medium (structural contract is clear)

## Evidence index
- `0x0070d500` — root header + version + typed table dispatch
- `0x00710890` — extension rewrite path (`.mdx`/`.mdl` normalization)
- `0x00706a40`, `0x0070db30`, `0x0070dcc0` — shared loader path to validator
- `0x0070e350`, `0x0070e450`, `0x0070e680`, `0x0070eea0`, `0x0070f460`, `0x0070fdc0`, `0x0070ff40`, `0x0070ffc0`, `0x0070fe40`, `0x0070fec0` — typed span validators
- `0x0070e4d0`, `0x0070e780`, `0x0070e830`, `0x0070e9c0`, `0x0070ea90`, `0x0070ebb0`, `0x0070ecd0`, `0x0070ef20`, `0x0070f050`, `0x0070f140`, `0x0070f4e0`, `0x0070f690`, `0x0070f970` — nested fixed-stride validators
