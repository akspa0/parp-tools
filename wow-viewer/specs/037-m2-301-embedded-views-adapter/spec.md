# Feature Specification: 2.x / 3.0.1 Embedded-Views MDX/M2 Build-Profile Adapter

**Feature Branch**: `037-m2-301-embedded-views-adapter`
**Created**: 2026-06-02
**Updated**: 2026-06-02 — scope expanded to cover full 2.x / 3.0.1 embedded-views M2 family
**Status**: Draft
**Input**: User report: the `output/tmp/wowarchive-clients/3_0_1_8303` pre-release
Wrath client and the older `2.0.0` / `2.x` TBC-era clients carry an M2/MDX
layout that is **structurally different from the 3.3.5 path** our viewer was
built around. Some assets in those builds cannot be decoded by either the
3.3.5 reader or the `M2ToMdxConverter` fallback, producing
`geosets=0, validGeosets=0, vertices=0, triangles=0` and leaving whole object
classes invisible. Three prior attempts to "fix" the general M2 path for
those builds have already broken the 3.3.5 M2/MDX renderer; this slice must
land a complete, non-invasive build-profile adapter for the full embedded-views
M2 family, not a partial patch.

## Context — Why This Slice Exists

The existing research note
(`wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`) already
calls this out as a deferred target:

> do not assume every `3.0.1.8303` `.mdx` is a normal MDX render path
> do not assume later numbered `.skin` semantics are automatically valid for this build
> future action: Ghidra on staged `3.0.1.8303` `wow.exe` plus repo archaeology for older `MD20` handling

The native build matrix
(`wow-viewer/docs/architecture/m2/native-build-matrix.md`) and implementation
contract
(`wow-viewer/docs/architecture/m2/implementation-contract.md`) confirm the
embedded-views family boundary spans both `2.0.0.5610` and `3.0.1.8303`:

- `2.0.0.5610`: `MD20` v`0x100`, root-contained active profile, no `%02d.skin` proof on the traced path, BLS shader-bound
- `3.0.1.8303`: strict `MD20` v`0x104..0x108`, version split at `0x108`, embedded view/profile path, no external `.skin` proof

The 3.3.5 reader, the 3.3.5 skin reader, the `M2ToMdxConverter`, and the
runtime builders are all built against the 3.3.5 external-`.skin` contract.
They reject the embedded-views family. This slice must add a complete sibling
adapter that reads the embedded-views family directly and emits the existing
`M2ModelDocument` contract so the downstream renderer, runtime builder, and
tensor packer consume the result with zero changes.

**Hard non-goal** (per the user, repeated across three prior attempts): do
**not** modify any existing file under
`wow-viewer/src/core/WowViewer.Core/M2/`,
`wow-viewer/src/core/WowViewer.Core.IO/M2/`, or
`wow-viewer/src/core/WowViewer.Core.Runtime/M2/` outside a brand-new namespace
created for this slice. The existing 3.3.5 path must remain bit-for-bit
unchanged.

## Ghidra Evidence Summary (3.0.1.8303 `WoW.exe`)

The native binary has the model bootstrap rooted in `M2Model.cpp` (sources
header string `f:\buildserver\bs1\work\wow-code\trunk\engine\source\model2\M2Model.h`
at `0x00936d68`). All findings below are confirmed by direct decompilation
through the Ghidra bridge.

### Loader Chain

| Function | Role |
| --- | --- |
| `FUN_0077e2c0` | `CM2Scene::AddModel(name, replacement, flags)` — high-level model request. Falls back to `"Spells\\ErrorCube.mdx"` on failure. |
| `FUN_0077d3c0` | `Model2_LoadModel(name, flags)` — cache entrypoint. Validates extension against three 5/4-byte constants at `0x98f7b8`, `0x93775c`, `0x98f7b4`. On match, rewrites extension to `.m2\0` via `*(uint*)EAX = 0x326d2e`. On miss, emits `"Model2: Invalid file extension: %s\n"`. |
| `FUN_0079bc70` | Parse bootstrap / async-load setup. Installs `FUN_0079bc50`. |
| `FUN_0079bc50` | Worker callback. Calls `FUN_00797450()` then `FUN_0079bb30()`. |
| `FUN_0079bb30` | Shared model init. Calls root validator `FUN_0079a8c0`. Marks versions `< 0x108`. Calls `FUN_007988c0` for the active-profile selection. |
| `FUN_0079a8c0` | **The root validator / on-disk MD20 layout walker.** Detailed below. |
| `FUN_007988c0` | **Active-profile selector.** Detailed below. |
| `FUN_00797d20` | Builds `CM2Shared_vtx` from the selected root profile. |
| `FUN_00797ad0` | Builds `CM2Shared_idx` from the selected root profile. |
| `FUN_00792f80` | `CM2Model::Initialize(...)` — full in-memory model bootstrap, dispatches `case 3: FUN_0078dea0()` for view init. |

### 3.0.1 On-Disk MD20 Layout (Authoritative, from `FUN_0079a8c0`)

The root validator enforces `magic == 0x3032444D` ("MD20") and
`0x103 < version < 0x109` (i.e. `0x104..0x108`). It then walks
`param_3 + N` (interpreted as `int*` so offset is `N * 4` bytes) through a
chain of typed-span validators. The recovered header table for 3.0.1 is:

| Header Offset | Field Type / Validator | Stride | Notes |
| --- | --- | --- | --- |
| `+0x00` | `uint32` magic | n/a | Must be `0x3032444D` |
| `+0x04` | `uint32` version | n/a | Must be `0x104..0x108` (boundary at `0x108`) |
| `+0x08` | typed span, name block | 0x01 | `FUN_00797540` |
| `+0x0C` | `uint32` | n/a | global flags / typeFlags |
| `+0x10` | `uint32` (typed span) | 0x44 | `FUN_00797680` — global loops / sequences |
| `+0x14` | `uint32` (typed span) | 0x02 | `FUN_00797950` — sequenceLookup |
| `+0x18` | `uint32` (typed span) | 0x04 | `FUN_00797710` — bones count/offset |
| `+0x1C` | `uint32` (typed span) | 0x04 | `FUN_00797710` — boneLookupTable |
| `+0x20` | `uint32` (typed span) | 0x70 | `FUN_00798da0` — bone table (M2Bone 0x70-stride, expanded shape) |
| `+0x24` | `uint32` (typed span) | 0x02 | `FUN_00797950` — vertexLookup / vertexBoneLookup |
| `+0x28` | `uint32` (typed span) | 0x30 | `FUN_007977a0` — vertices? view hints? (stride 0x30) |
| `+0x2C` | `uint32` (typed span) | 0x2C | `FUN_00798320` — root-contained view/profile table (LOD profiles) |
| `+0x30` | `uint32` (typed span) | 0x38 | `FUN_00798f40` — secondary nested table (0x38 stride) |
| `+0x34` | `uint32` (typed span) | custom | `FUN_007983e0` — colors (or text) |
| `+0x38` | `uint32` (typed span) | custom | `FUN_007984a0` — text unit / texture weight region |
| `+0x3C` | `uint32` (typed span) | custom | `FUN_007984a0` — paired table |
| `+0x40` | `uint32` (typed span) | custom | `FUN_00799090` — texture transforms |
| `+0x44..+0x58` | typed spans | 0x02 / 0x04 | misc typed spans (5+ consecutive) |
| `+0x98..+0xA8` | typed spans | 0x02 | misc short tables (post-`FUN_007985b0` if version < 0x108) |
| `+0xAC` | typed span | 0x0C | `FUN_007975d0` — colors/light related |
| `+0xB0` | typed span | 0x0C | `FUN_007975d0` — paired |
| `+0xB4` | typed span | custom | `FUN_00799230` — geometry/state block |
| `+0xC0` | typed span | 0x02 | `FUN_00797950` |
| `+0xC4` | typed span | 0x2C | `FUN_007985f0` — second 0x2C-stride family (later init stage) |
| `+0xC8` | typed span | 0xD4 | `FUN_00799340` — particle/emitter family (0xD4 stride) |
| `+0xCC` | typed span | 0x7C | `FUN_0079a720` — texture unit / material family (0x7C stride) |
| `+0xD0` | typed span | 0x02 | `FUN_00797950` — final typed span |
| `+0x134` | `uint32` count | n/a | **version < 0x108**: count for 0xDC family; **version >= 0x108**: count for 0xE0 family |
| `+0x138` | `uint32` offset | n/a | **version < 0x108**: offset for 0xDC family; **version >= 0x108**: offset for 0xE0 family |
| `+0x13C` | `uint32` count | n/a | **version < 0x108**: count for 0x1F8 family; **version >= 0x108**: count for 0x234 family |
| `+0x140` | `uint32` offset | n/a | **version < 0x108**: offset for 0x1F8 family; **version >= 0x108**: offset for 0x234 family |
| `+0x144` | `uint32` (conditional) | 0x02 | gated by flag `0x8` in `param_3[4]` |

The validator calls a `FUN_007985b0(param_3 + 0x34, param_3 + 0x37)` ONLY
when `version < 0x108`. This is a 12-byte (3 × `int*`) caller/callee pairing
that mutates the header offsets before the rest of the validation continues.
This is a structural on-disk delta between the 0x104-0x107 and 0x108+ sub-builds
**of 3.0.1 itself**, not a generic 3.x/2.x split.

### Version Split Inside 3.0.1 — The `0x108` Gate

The 0x108 boundary is **not** cosmetic. The header offsets for the two
late-stage record families (texture units + view/skin region) are the same,
but the **per-record stride** is different:

- **`< 0x108` (legacy side of 3.0.1)**:
  - `FUN_00799ee0`: family stride `0xDC`
  - `FUN_0079a1c0`: family stride `0x1F8` (param_3 + 0x4D/0x4E for count/offset; param_3 + 0x4F/0x50 for second family)
  - The 0x1F8 family is copied into an in-memory `0xE0` block that contains expanded track-bearing layout
  - The 0x1F8-stride family mutates `0xDC`-stride records into the expanded `0xE0` runtime form
- **`>= 0x108` (later side of 3.0.1)**:
  - `FUN_00799640`: family stride `0xE0` (no remap needed)
  - `FUN_00799920(param_1, param_2, param_3[1], param_3 + 0x4F)`: family stride `0x234`

Both sides carry `param_3 + 0x144` validation gated by `flags & 0x8`.

### 0x2C-Stride Embedded View Record (from `FUN_00798320` + `FUN_007988c0`)

The `FUN_00798320` validator enforces a per-record stride of `0x2C` and then
dispatches per-record validation through `FUN_00797a40`. The 0x2C-stride
record has these sub-fields (offsets are bytes inside the 0x2C record):

| Sub-Offset | Validator | Stride | Semantic Role |
| --- | --- | --- | --- |
| `+0x00` | typed span | 0x02 | Triangle indices (per view) |
| `+0x08` | typed span | 0x02 | TriangleIndicesLow? vertex-related? |
| `+0x10` | typed span | 0x04 | Bone indices table per view |
| `+0x18` | typed span | 0x30 | Per-view submesh / 0x30-stride sub-record |
| `+0x20` | typed span | 0x18 | Per-view batch / 0x18-stride sub-record |
| `+0x28` | scalar (uint) | n/a | **LOD threshold selector** consumed by `FUN_007988c0` |

`FUN_007988c0` (the active-profile selector) reads `*(int *)(param_1 + 0x134) + 0x4C`
(count) and `+ 0x50` (offset to 0x2C records), then picks the active record:

1. Compute the quality threshold: `local_10 = 0x100` initially, but if `flags & 8` is set, it becomes `(maxQuality - 0x1F) / 3` (rounded down, capped at 0x100).
2. Iterate all `0x2C` records at stride `0x2C`.
3. For each, read `+0x28` (the LOD threshold).
4. Keep the record with the **largest `+0x28` that is still `<= local_10`**.
5. Store the chosen record at `param_1 + 0x13C`.
6. If the chosen record has `+0x08 == 0`, store `0x8000 / 1 = 0x8000` at `param_1 + 0x160` as a max-LOD divisor, else `0x8000 / (+0x08)`.

After selection, `FUN_007988c0` also builds the in-memory runtime layout at
`param_1 + 0x154` and `+ 0x158`:

- `+0x154`: per-batch runtime data (length = `+0x20`-field count of the chosen record) — each batch is `0x18` stride
- `+0x158`: per-section runtime data (length = `+0x18`-field count of the chosen record) — each section is `0x30` stride

This is the **runtime vertex/index layout** that downstream code consumes
through `FUN_00792f80` and `FUN_0078dea0`.

### Bone Table Stride (`FUN_00798da0`, 0x70)

The 3.0.1 bone record is 0x70 bytes wide. The 0x70 record holds:

| Sub-Offset | Stride | Notes |
| --- | --- | --- |
| `+0x14` | 0x08 | parent + flags? (uint64) |
| `+0x1C` | 0x04 | typed span |
| `+0x24` | 0x0C | typed span |
| `+0x30` | 0x08 | typed span |
| `+0x38` | 0x04 | typed span |
| `+0x40` | 0x08 | typed span |
| `+0x4C` | 0x08 | typed span |
| `+0x54` | 0x04 | typed span |
| `+0x5C` | 0x0C | typed span |

Compared to 3.3.5's 0x58-stride bone, 3.0.1's 0x70-stride bone is the
**expanded, track-bearing** form. This is one of the most important
differences for the renderer: do not try to map 3.0.1 bones into 3.3.5
0x58-byte bone records.

### Nested Record Family Strides (3.0.1)

| Function | Stride | Role | Sub-validators |
| --- | --- | --- | --- |
| `FUN_00798da0` | 0x70 | bone (expanded) | `0x08`, `0x04`, `0x0C`, `0x08`, `0x04`, `0x08`, `0x08`, `0x04`, `0x0C` |
| `FUN_00798320` | 0x2C | embedded view (root profile) | delegates to `FUN_00797a40` |
| `FUN_00798f40` | 0x38 | secondary nested family | (needs deeper probe) |
| `FUN_007985f0` | 0x2C | late-stage second 0x2C family | (needs deeper probe) |
| `FUN_00799340` | 0xD4 | particle / emitter | 18 typed spans |
| `FUN_0079a720` | 0x7C | texture unit / material | 8 typed spans |
| `FUN_00799ee0` (v<0x108) | 0xDC | legacy texture unit + view region | (remapped to 0xE0 in-memory) |
| `FUN_0079a1c0` (v<0x108) | 0x1F8 | legacy view/region + animation/emitter | (remapped to 0x234 in-memory) |
| `FUN_00799640` (v>=0x108) | 0xE0 | later texture unit + view region | (no remap) |
| `FUN_00799920` (v>=0x108) | 0x234 | later view/region + animation/emitter | (no remap) |

### In-Memory Runtime Layout (from `FUN_00792f80`)

`FUN_00792f80` reads from offsets `+0x14`..`+0x1E8` of the chosen embedded
profile record. The visible in-memory fields include:

- `+0x14`: bone count
- `+0x34`: bone count
- `+0x38`: bone data
- `+0x50`: boneLookup pointer
- `+0x54`: vertex count
- `+0x5C`: per-vertex count
- `+0x60`: (vertex data field)
- `+0x64`: color count
- `+0x6C`: texture count
- `+0x74`: transparency count
- `+0x104`: embedded skin section count
- `+0x108`: embedded skin section data
- `+0x11C`: particle count
- `+0x120`: particle data
- `+0x124`: ribbon count
- `+0x128`: ribbon data
- `+0x134`: texture unit count
- `+0x138`: texture unit data
- `+0x13C`: (related)
- `+0x140`: (related)
- `+0x160`: transparencyLookup count

This is the **same in-memory shape** the existing `M2ModelDocument` already
documents — but fed by embedded data, not by a companion `.skin` file.

### 2.0.0.5610 Evidence (from `wow-200-beta-m2-light-particle-terrain-guide.md` and the build matrix)

The 2.0.0 binary is not currently loaded in the Ghidra bridge; the
2.0.0.5610 evidence below is from the prior static Ghidra pass recorded in
the legacy `wow-200-beta-m2-light-particle-terrain-guide.md` and the
`m2-native-client-research-2026-03-31.md` notes.

| Concern | 2.0.0.5610 evidence |
| --- | --- |
| Magic | `MD20` (canonical) |
| Version | `0x100` |
| Skin strategy | **Root-contained**, no `%02d.skin` proof on traced path |
| Active profile storage | Root table at `+0x4C` / `+0x50`; chosen profile pointer at shared offset `+0x138` |
| Vertex buffer | Built by `FUN_0072f3f0` directly from selected embedded profile (`CM2Shared_vtx`) |
| Index buffer | Built by `FUN_0072f220` directly from selected embedded profile (`CM2Shared_idx`) |
| Active profile selector | `FUN_0072ee30` in `M2Shared.cpp` |
| Shader path | `shaders\vertex\Model2.bls`, `shaders\pixel\Model2.bls` — dedicated BLS programs |
| Map object shaders | `MapObjOverbright`, `MapObjSpecular`, `MapObjMetal`, `MapObjEnv`, `MapObjEnvMetal`, `MapObjExtWater0`, `MapObjTransDiffuse`, `MapObjTransSpecular` |
| Light runtime | `M2Light` is runtime-managed with spatial bucketing (`FUN_0072d1a0`, `FUN_0072cc60`, `FUN_0072cc90`, `FUN_0072cdc0`); `CMapLight`, `CLightList`, `CGxuLight`, `CGxuLightLink`, `LightRef` |
| Particle runtime | `CParticleEmitter2_idx`, `CParticle2`, `CParticle2_Model`; `ParticleSystem2.h` header string |
| OpenGL state | `glLightf`, `glLightfv`, `glLightModeli`, `glLightModelfv` — fixed-function light path still in use |

**Practical 2.x read**: 2.0.0.5610 is the same embedded-views M2 family as
3.0.1, with the same root-contained profile model. The 2.x `+0x4C`/`+0x50`
profile offsets correspond to 3.0.1's `+0x2C` 0x2C-stride embedded view
table, just with smaller/older per-record layouts. The 2.x binary is
required to confirm the exact sub-record strides for 0x100, but the
**family contract** is the same.

### Cross-Era Differences (Confirmed)

| Field | 2.0.0 (0x100) | 3.0.1 <0x108 (0x104..0x107) | 3.0.1 >=0x108 | 3.3.5 (0x108..0x10A) |
| --- | --- | --- | --- | --- |
| Magic | `MD20` | `MD20` | `MD20` | `MD20` |
| Skin strategy | embedded | embedded | embedded | external `.skin` |
| Bone stride | (likely 0x44) | 0x70 | 0x70 | 0x58 |
| View/profile record | embedded, root-contained | embedded, 0x2C stride | embedded, 0x2C stride | external, 0x30 stride |
| Texture unit record | (unknown) | 0xDC | 0xE0 | 0xE0 |
| View/region + emitter | (unknown) | 0x1F8 (remap to 0x234 in-mem) | 0x234 | 0x234 |
| Material record | (unknown) | 0x7C | 0x7C | 0x7C |
| Particle record | (unknown) | 0xD4 | 0xD4 | 0xD4 |
| Camera stride | (unknown) | 0x44 (textured via `FUN_00797680`) | 0x44 | 0x64 classic / 0x74 modern |
| Ribbons | (unknown) | (likely 0xAC) | (likely 0xAC) | 0xAC classic / 0xB0 modern |
| Particles | (unknown) | 0x1DC classic (off-stride 0xD4) | 0x1EC modern | 0x1DC classic / 0x1EC modern |
| Lights | runtime-managed w/ spatial bucket | 0x9C | 0x9C | 0x9C |
| Shader path | `Model2.bls` (BLS) | (post-2.x era) | (post-2.x era) | `Model2.wfx` (effect files) |
| Active profile selector | `FUN_0072ee30` | `FUN_007988c0` | `FUN_007988c0` | `M2_ChooseAndLoadSkinProfile` |

**Critical reading for the slice**: The 2.x+3.0.1 family shares the
**embedded-views** contract (no external `.skin`), the **0x2C-stride view
profile** layout, the **0x70-stride expanded bone** shape (3.0.1) and
**runtime active-profile selection** algorithm. It differs from 3.3.5
on: bone stride, view/profile storage location, secondary record strides,
and shader/effect vocabulary. The adapter must handle all of this in
shared-library code that emits a normal `M2ModelDocument`.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 2.x and 3.0.1 `.mdx` Round-Trip (Priority: P1)

As a dataset / model-building operator, I can point the existing M2 ingest
pipeline at a 2.x or 3.0.1 `.mdx` file and produce a renderable model
(visible geometry, populated vertex/triangle buffers, classified batches)
without touching the 3.3.5 code path.

**Why this priority**: This is the user-reported blocking symptom. Without
this, every 2.x and 3.0.1 placement fails with `vertices=0, triangles=0`
and the downstream tensor pack is empty.

**Independent Test**: Run a probe on at least one known 2.x `.mdx` (e.g.
the embedded `Model2.bls` references) and at least one known 3.0.1 `.mdx`
(e.g. `Spells\ErrorCube.mdx`) and produce non-zero `vertices` and
`triangles` plus a populated batch list.

**Acceptance Scenarios**:
1. **Given** a staged 3.0.1.8303 client and a `.mdx` path, **When** the
   adapter reads it, **Then** the resulting `M2ModelDocument` reports
   `viewCount >= 1`, `vertices > 0`, and `triangles > 0` from the
   embedded views.
2. **Given** a staged 2.x client (when one becomes available) and a
   `.mdx` path, **When** the adapter reads it, **Then** the resulting
   `M2ModelDocument` reports `viewCount >= 1`, `vertices > 0`, and
   `triangles > 0` from the embedded views.
3. **Given** a 3.0.1 asset with `version < 0x108`, **When** the adapter
   reads it, **Then** the legacy 0xDC/0x1F8 stride families are decoded
   and remapped to the in-memory 0xE0/0x234 layout without data loss.
4. **Given** a 3.0.1 asset with `version >= 0x108`, **When** the adapter
   reads it, **Then** the 0xE0/0x234 stride families are decoded directly
   (no remap).
5. **Given** the same input, **When** the existing 3.3.5 ingest path is
   rerun on the same file, **Then** its result is byte-identical to the
   pre-change baseline (no silent contract drift).

### User Story 2 — Non-Invasive Build-Profile Routing (Priority: P1)

As a renderer maintainer, I can see explicit, bounded code that detects
2.x and 3.0.1 assets and routes them through the adapter, while all
3.3.5 assets continue to flow through the existing `M2ModelReader` /
`M2SkinReader` / `M2ToMdxConverter` chain with zero behavioral change.

**Why this priority**: This is the explicit user constraint. The previous
three attempts "fixed" the problem by mutating shared code, which broke
3.3.5 rendering. This slice is structured so the adapter is the **only**
new code path and the existing path is bit-for-bit unchanged.

**Independent Test**: Run a 3.3.5 parity suite before and after the change
and assert identical results. Run a 2.x and a 3.0.1 parity suite and
assert they now populate the model where they previously did not.

**Acceptance Scenarios**:
1. **Given** a 3.3.5 staged client, **When** the M2 ingest pipeline runs,
   **Then** every call to `M2ModelReader.Read(...)`, `M2SkinReader.Read(...)`,
   and `M2ToMdxConverter.Convert(...)` produces output identical to the
   pre-change baseline.
2. **Given** a 3.0.1 staged client and any `.mdx` path, **When** the new
   adapter is invoked, **Then** the existing 3.3.5 code is **not** called
   for that file (verified by call-stack trace or stub-based test).
3. **Given** a 2.x staged client (when one becomes available) and any
   `.mdx` path, **When** the new adapter is invoked, **Then** the
   existing 3.3.5 code is **not** called for that file.
4. **Given** the new code, **When** I look at the diff, **Then** no file
   under `src/core/WowViewer.Core/M2/`, `src/core/WowViewer.Core.IO/M2/`,
   or `src/core/WowViewer.Core.Runtime/M2/` outside a new
   `M2BuildLegacy/` namespace has had its public contract changed.

### User Story 3 — Adapter Self-Reporting and Safe Fallback (Priority: P2)

As a dataset / model-building operator, when the adapter encounters a
2.x or 3.0.1 asset that it cannot decode (corrupt, missing required
section, unknown version), it reports a clear, actionable error and does
**not** silently swallow the failure or fall through to the 3.3.5 path
(which would silently produce `vertices=0`).

**Why this priority**: The current `M2ToMdxConverter` fallback silently
emits `geosets=0, validGeosets=0, vertices=0, triangles=0`, which is the
exact symptom we are trying to fix. The adapter must not repeat that
pattern.

**Independent Test**: Feed the adapter a deliberately truncated MDX and
verify the error reports which section failed (e.g. `viewCount mismatch`,
`embedded section table invalid`, `bone table stride mismatch`).

**Acceptance Scenarios**:
1. **Given** a 3.0.1 header with `viewCount > 0` but a truncated embedded
   section table, **When** the adapter reads it, **Then** it throws
   `InvalidDataException` with a message naming the failing section
   and the file path.
2. **Given** a 3.0.1 asset with `version = 0x109` (out of range),
   **When** the adapter is invoked, **Then** it throws
   `NotSupportedException` with the detected version and a pointer to the
   correct reader.
3. **Given** a 3.0.1 asset with `version = 0x107` and an invalid
   `FUN_007985b0` mutation in the header, **When** the adapter reads it,
   **Then** the mutation is applied and the rest of the validation
   continues, OR the file is rejected with a clear "header mutation
   failed" error — never silently.
4. **Given** a 2.x asset with `version = 0x100` and a malformed
   root-contained profile table, **When** the adapter reads it, **Then**
   it throws `InvalidDataException` with the file path and the bad
   profile index.

### Edge Cases

- 2.x and 3.0.1 `.mdx` with `viewCount = 0` (no embedded views —
  degenerate but legal in some animation-only models).
- 2.x and 3.0.1 `.mdx` with the classic TBC particle / ribbon stride
  (must not be misinterpreted as the modern Wrath stride).
- 2.x and 3.0.1 `.mdx` in an MPQ archive (read path must accept
  `Stream` not only `string`, mirroring the existing reader contracts).
- Model file referenced by a WDT/WMO doodad definition where the
  `.skin` companion is genuinely expected (e.g. 3.3.5) — must continue
  to route to the 3.3.5 path.
- Path with mixed casing (e.g. `.MDX`, `.M2`) — adapter detection must
  be case-insensitive, matching `M2ModelIdentity` convention.
- 3.0.1 asset with `flags & 8` set (the alternate LOD threshold formula
  `(quality - 0x1F) / 3` applies in `FUN_007988c0`).
- 3.0.1 asset with `version < 0x108` AND an `0x108`-style secondary
  header — the `FUN_007985b0` mutation must run before continuing.
- 3.0.1 asset with `viewCount > 1` where no record's `+0x28` threshold
  is `<= local_10` (active-profile selection must default to the
  first record rather than producing `viewCount = 0`).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide new public entrypoints
  `M2BuildLegacyAdapter.Read(path)` /
  `M2BuildLegacyAdapter.Read(stream, sourcePath)` that produce a fully
  populated `M2ModelDocument` for 2.x and 3.0.1 MDX/M2 assets, using only
  the embedded data in the file.
- **FR-002**: The adapter MUST NOT read or write to any external `.skin`
  companion file. The 2.x and 3.0.1 binaries do not use that path and the
  adapter must mirror native behavior.
- **FR-003**: The adapter MUST validate the input as 2.x or 3.0.1 before
  parsing (header magic `MD20`, version range, structural fingerprint)
  and throw `NotSupportedException` for non-matching inputs.
- **FR-004**: The adapter MUST live under a new `M2BuildLegacy` namespace
  with all new types grouped there. No new types may be added to the
  existing `WowViewer.Core.M2` / `WowViewer.Core.IO.M2` /
  `WowViewer.Core.Runtime.M2` namespaces.
- **FR-005**: The adapter MUST NOT modify the public contract of any
  existing type in `M2ModelDocument`, `M2ModelReader`, `M2SkinReader`,
  `M2SkinDocument`, `M2ToMdxConverter`, or the runtime M2 builders.
- **FR-006**: The adapter output MUST be a regular `M2ModelDocument` so
  that the downstream renderer, runtime builders, and
  `M2StaticRenderModelBuilder` consume it without any change.
- **FR-007**: System MUST provide a build-profile detection helper
  (`M2BuildProfileDetector.DetectFromPath(path)` /
  `M2BuildProfileDetector.DetectFromHeader(stream)`) that classifies an
  asset as `Classic` / `TBC` / `Build20` / `Build301` / `Build301Legacy` /
  `Build301Later` / `Wrath` / `Cata` / `Unknown` based on a small set of
  stable, version-gated fingerprints. The detector MUST be additive —
  existing code that does not call it MUST continue to work unchanged.
- **FR-008**: System MUST provide a typed-span validator helper
  (`M2LegacySpanValidator.Validate(count, offset, stride, fileSize)`)
  that exactly mirrors the seven 3.0.1 span validators
  (`FUN_00797540` stride 0x01, `FUN_00797950` stride 0x02, `FUN_00797710`
  stride 0x04, `FUN_00797830` stride 0x08, `FUN_007975D0` stride 0x0C,
  `FUN_007977A0` stride 0x30, `FUN_00797680` stride 0x44) and rejects
  out-of-bounds spans with the same `FUN_0068cf10` 0x85100000
  diagnostic-shape errors.
- **FR-009**: System MUST provide an active-profile selector
  (`M2LegacyActiveProfileSelector.Select(profileTable, flags, quality)`)
  that exactly mirrors `FUN_007988c0`:
  - default `local_10 = 0x100`
  - if `flags & 8`, recompute `local_10 = (maxQuality - 0x1F) / 3`, capped at `0x100`
  - pick the record with the largest `+0x28` threshold `<= local_10`
  - on tie or no match, default to the first record
  - emit `param_1 + 0x160 = 0x8000 / max(record.+0x08, 1)` as max-LOD divisor
- **FR-010**: System MUST provide a 3.0.1 version-split gate
  (`M2LegacyVersionGate.RouteFor301(version, legacyRecordTable, laterRecordTable)`)
  that selects the legacy (0xDC/0x1F8) or later (0xE0/0x234) record-family
  walk based on `version < 0x108` and applies the `FUN_007985b0`
  12-byte header mutation ONLY in the legacy path.
- **FR-011**: System MUST provide a nested-record walker
  (`M2LegacyNestedRecordReader.Read(buffer, offset, familyStride)`) that
  validates and reads all 9 nested record families with exact strides:
  0x70 (bone), 0x2C (embedded view), 0x38 (secondary), 0xD4 (particle),
  0x7C (material), 0xDC (legacy 301 secondary), 0xE0 (later 301 secondary),
  0x1F8 (legacy 301 view/emitter), 0x234 (later 301 view/emitter).
- **FR-012**: System MUST remap the 3.0.1 legacy 0xDC/0x1F8 stride
  families to the in-memory 0xE0/0x234 layout exactly as the native
  `FUN_0079a1c0` body does (with `iVar11`/`iVar7`/`iVar8` triplet
  computation, `0xE0`-stride copy, and `0x234`-stride copy with
  `local_18` accumulator).
- **FR-013**: System MUST provide unit tests
  (`WowViewer.Core.IO.M2BuildLegacy.Tests` project) that exercise the
  adapter on at least:
  - one 3.0.1 `version < 0x108` sample (e.g. extracted from
    `output/tmp/wowarchive-clients/3_0_1_8303` via existing MPQ tooling)
  - one 3.0.1 `version >= 0x108` sample (same MPQ)
  - one 2.x sample (when a staged 2.x client becomes available)
  Each test asserts `vertices > 0`, `triangles > 0`, and a populated
  batch list.
- **FR-014**: System MUST provide a 3.3.5 regression test that re-runs
  the existing M2 / SKIN / converter / runtime builder against a known
  3.3.5 sample and asserts output is unchanged from the pre-change
  baseline. This test MUST pass before and after the adapter is added.
- **FR-015**: The adapter MUST accept both `string path` and
  `Stream stream, string sourcePath` overloads, mirroring the
  signatures of `M2ModelReader.Read`.
- **FR-016**: The adapter MUST log a one-line summary per file (path,
  detected build profile, version, view count, vertex count, triangle
  count, bone count) to the existing logging surface so the existing
  WowViewer probe / inspect tools can report the 2.x and 3.0.1 surface
  without code changes.
- **FR-017**: The adapter MUST reuse the existing typed-span validator
  helper (FR-008) for ALL count/offset checks; it MUST NOT use
  ad-hoc `if (offset + count*stride > length)` checks scattered
  through the code. This is the only way the boundary checks can
  match the native diagnostic shape.
- **FR-018**: The active-profile selector (FR-009) MUST be tested with
  three cases: a 3.0.1 sample with `flags & 8 == 0` (default quality
  cap), a 3.0.1 sample with `flags & 8 != 0` (alternate quality
  formula), and a 3.0.1 sample with `viewCount > 1` and a clear
  threshold ladder.

### Key Entities

- **`M2BuildLegacyAdapter`** — public static entrypoint. Two `Read(...)`
  overloads matching `M2ModelReader`.
- **`M2BuildProfileDetector`** — static helper that classifies an asset
  by build profile (Classic, TBC, Build20, Build301, Build301Legacy,
  Build301Later, Wrath, Cata, Unknown). Additive.
- **`M2LegacySpanValidator`** — typed-span boundary helper mirroring
  the seven native 3.0.1 validators. Used by every record-family
  reader.
- **`M2LegacyActiveProfileSelector`** — implements the `FUN_007988c0`
  active-profile selection algorithm exactly.
- **`M2LegacyVersionGate`** — routes between the 0x108 split within
  3.0.1 and applies the `FUN_007985b0` header mutation.
- **`M2LegacyNestedRecordReader`** — reads all 9 nested record families
  with exact strides.
- **`M2Legacy301ViewRecord`** — internal record of one 0x2C-stride
  embedded view/profile record (typed spans + LOD threshold).
- **`M2Legacy301BoneRecord`** — internal record of one 0x70-stride
  bone record (3.0.1 expanded bone).
- **`M2Legacy301ParticleRecord`** — internal record of one 0xD4-stride
  particle/emitter record.
- **`M2Legacy301MaterialRecord`** — internal record of one 0x7C-stride
  material/texture-unit record.
- **`M2Legacy301SecondaryRecord`** — internal record of one 0x38-stride
  secondary nested record.
- **`M2Legacy301LateSecondaryRecord`** — internal record of one
  0x2C-stride late-stage second family record.
- **`M2Legacy301LegacySecondaryRecord`** — internal record of one
  0xDC-stride legacy 3.0.1 secondary record (version < 0x108).
- **`M2Legacy301LaterSecondaryRecord`** — internal record of one
  0xE0-stride later 3.0.1 secondary record (version >= 0x108).
- **`M2Legacy301LegacyViewEmitterRecord`** — internal record of one
  0x1F8-stride legacy 3.0.1 view/region + animation/emitter record
  (version < 0x108). Includes the in-memory 0x234 remap.
- **`M2Legacy301LaterViewEmitterRecord`** — internal record of one
  0x234-stride later 3.0.1 view/region + animation/emitter record
  (version >= 0x108). No remap.
- **`M2Legacy2xRecord`** — internal record families for 2.x (to be
  confirmed against an actual 2.0.0 binary in a future slice; the
  present slice provides the family contract and leaves the per-record
  reads as `NotSupportedException` until the 2.x binary is loaded).
- **`M2BuildLegacyDiagnosticReport`** — the per-file summary used by
  FR-016.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a known 3.0.1 `< 0x108` sample extracted from
  `output/tmp/wowarchive-clients/3_0_1_8303`, the adapter produces an
  `M2ModelDocument` with `vertices > 0`, `triangles > 0`,
  `viewCount >= 1`, and a non-empty batch list.
- **SC-002**: On a known 3.0.1 `>= 0x108` sample (same MPQ), the
  adapter produces the same shape (proving the version-split path
  is correctly routed).
- **SC-003**: The 3.3.5 regression test (FR-014) passes before and
  after the adapter lands. Specifically: `M2ModelReader.Read` output
  on the chosen 3.3.5 sample is byte-identical to the
  pre-change baseline.
- **SC-004**: `git diff --stat` of the change set shows zero lines
  added or removed in `M2ModelReader.cs`, `M2SkinReader.cs`,
  `M2ToMdxConverter.cs`, `M2StaticRenderModelBuilder.cs`,
  `M2SkinnedRenderModelBuilder.cs`, `M2TrackSampler.cs`. The new code
  is strictly additive under a new namespace.
- **SC-005**: The 3.0.1 probe (FR-013) produces a report naming the
  detected build profile, version, view count, vertex count,
  triangle count, and batch count, and that report is reproducible
  across runs (deterministic).
- **SC-006**: The full `dotnet build` and `dotnet test` of
  `wow-viewer/WowViewer.slnx` passes with the adapter included.
- **SC-007**: Every native 3.0.1 validator (the seven `FUN_00797xxx`
  span helpers and the nine nested-record walkers) has at least one
  unit test that exercises its boundary and stride behavior with
  known-good and known-bad inputs.

## Assumptions

- The staged `output/tmp/wowarchive-clients/3_0_1_8303` MPQ bundle
  contains at least one `.mdx` file (e.g. `Spells\ErrorCube.mdx`,
  `Spells\RainOfFire_Impact_Base.mdx`, `World\Generic\PassiveDoodads\DeathSkeletons\*DeathSkeleton.mdx`)
  that can be extracted for sample validation via existing MPQ
  tooling.
- A staged 2.x client is **not** currently available; the slice
  provides the 2.x family contract but leaves the per-record reads
  as `NotSupportedException` for 2.x-specific records until a 2.x
  binary is loaded into Ghidra. This is acknowledged in FR-001 and
  the Open Questions.
- The 3.0.1 build is functionally the embedded-views M2 form, and
  the Ghidra findings from `FUN_0079a8c0` / `FUN_007988c0` /
  `FUN_00792f80` are sufficient to derive the on-disk layout for
  the 0x104..0x108 version range. If the on-disk layout diverges
  from the in-memory layout observed in Ghidra, the slice is
  blocked and must be re-scoped (not silently papered over).
- The existing 3.3.5 / Wrath M2 reader, skin reader, runtime builder,
  and `M2ToMdxConverter` continue to be the canonical owners for
  3.3.5 assets. The adapter is a sibling, not a replacement.
- 2.x and 3.0.1 `.anim` (external animation) files are out of scope.
  This slice covers model geometry only.
- The 2.x `Model2.bls` shader path, the 2.x `M2Light` spatial-bucket
  runtime, and the 2.x `ParticleSystem2` runtime are all out of
  scope for the parser layer. The slice acknowledges them in the
  success criteria as "deferred to the renderer/runtime layer" —
  a future slice will own them.

## Out of Scope

- Changing any existing M2 / MDX / skin / runtime / converter code in
  `wow-viewer/`.
- Re-introducing cross-repo M2 ownership in `gillijimproject_refactor/`.
- Animated models, particle systems, ribbon emitters from 2.x or 3.0.1
  — geometry only.
- The 2.x `Model2.bls` shader path, the 2.x `M2Light` spatial-bucket
  runtime, and the 2.x `ParticleSystem2` runtime — these are
  acknowledged as future renderer-layer work, not parser work.
- Cross-version dataset-builder integration — that is a downstream
  consumer concern once the adapter is stable.

## Open Questions

- **OQ-1**: Is `Spells\ErrorCube.mdx` (or one of the other 3.0.1
  strings the client itself uses) extractable from the staged
  `3_0_1_8303` MPQ with existing tools, or do we need a dedicated MPQ
  extraction step in the slice?
- **OQ-2**: The on-disk version is `0x104..0x108`. Is the full
  0x104..0x107 range present in the staged 3.0.1.8303 client, or is
  it a single fixed version? If the staged client only carries one
  sub-version, the version-split gate (FR-010) will need synthetic
  fixtures for the other side until a wider MPQ corpus is staged.
- **OQ-3**: For 2.x, do we have access to a staged 2.0.0.5610 client
  (or any 2.x client)? If not, the 2.x slice of this adapter is
  blocked behind staging a 2.x client and loading its `WoW.exe` into
  the Ghidra bridge.
- **OQ-4**: The 3.0.1 0x104..0x107 `FUN_007985b0` mutation is a
  12-byte header rewrite. Are there any in-the-wild 3.0.1 assets in
  the MPQ that already carry the post-mutation header layout? If so,
  the adapter must detect "already mutated" and skip the rewrite.
  Otherwise, the rewrite must run unconditionally before the rest
  of the validation.
- **OQ-5**: Does the active-profile selector (`FUN_007988c0`) need
  to handle the `viewCount > 1` with no qualifying record case, or
  does the native binary always guarantee at least one record will
  match? The slice defaults to "first record wins" on no match; the
  next-session decision is whether to log a warning or fail hard.

## Notes

- This slice is the **first** concrete attempt at the
  "build-aware parser" requirement from
  `wow-viewer/docs/architecture/m2/implementation-contract.md` Section 2
  for the 2.x and 3.0.1 eras.
- The legacy contracts referenced in the Inputs section of
  `gillijimproject_refactor/specifications/3.0.1.8303/Contracts/M2_MDX_Implementation_Contract_3.0.1.8303.md`
  (the `M2Profile` schema, the registry entries `M2Profile_301_8303`
  and `M2Profile_30x_Unknown`, the nested record stride table) are
  **advisory for the wow-viewer implementation**. The wow-viewer
  implementation does not have to mirror the MdxViewer profile
  registry shape, but it does have to honor the same nested record
  strides and the same `0x108` version split.
- The legacy 2.x findings in
  `gillijimproject_refactor/documentation/wow-200-beta-m2-light-particle-terrain-guide.md`
  are the only 2.x source we have until a 2.x binary is loaded. The
  2.x family contract in this spec is consistent with those findings
  but is not itself a substitute for a 2.x native pass.
