# Phase 3 Research: WoW.exe 1.0.0 M2/MDX — Ghidra Static Trace (2026-07-15)

Static reverse-engineering of the **WoW 1.0.0** (Vanilla release) Windows client's
`Model2` (M2/MDX) subsystem, performed via the GhidraMCP HTTP API against a Ghidra
project with `WoW.exe` 1.0.0 loaded. This document is the implementation-ready evidence
for Spec 104 Phase 3 (the 1.0.0 / 0.12 / 0.11 alpha boundary). It is intended to be
consumed directly by a fresh implementation chat.

**Companion files**: raw decompilations of every function cited here are saved under
`output/ghidra_1.0.0/*.c` (and `sub_*.c` for the header sub-parsers). Re-query live via
the GhidraMCP bridge (see §1) if the project is still loaded.

---

## 1. Tooling / how this was recovered

- **Ghidra**: `H:\ghidra_11.3.2_PUBLIC` (Ghidra 11.3.2). The **GhidraMCP** plugin
  (Lakr233, release 1-4) is installed and serves a REST API at `http://127.0.0.1:8080/`.
- **MCP bridge**: `H:\ghidra_11.3.2_PUBLIC\GhidraMCP-release-1-4\bridge_mcp_ghidra.py`
  (a stdio MCP server wrapping the HTTP API; deps `requests` + `mcp`, run via
  `uv run --script`). It is now wired into `.mcp.json` as the `ghidra` server.
- All findings below come from decompiling functions reached by string-xref from the
  preserved `C:\build\buildWoW\Engine\Source\Model2\*.cpp` source-path strings and the
  `data-><field>.count` assert strings. No dynamic tracing was required — the static
  decompilation is unambiguous for the format-reading surface.

> **Correction to Spec 104 `research.md` Decision 5**: Ghidra IS now installed and the
> MCP bridge is live. The "do not depend on Ghidra" caveat is obsolete for 1.0.0.

---

## 2. The 1.x format boundary (and what the 1.0.0 client tells us)

**The 1.0.0 M2 parser hard-requires `MD20` magic AND version `0x100` (256).** From the
parser `FUN_0071e190` (the load-completion relocator, see §4):

```c
if (*param_3 != 0x3032444d) return 0;   // dword at 0x00 must be 'MD20'
if (param_3[1] != 0x100) return 0;       // dword at 0x04 must be EXACTLY 0x100
```

A failure of either check returns 0 → the completion callback `FUN_0071e940` logs
**`Corrupt model data: %s`** (`0x0083fa34`) and the model never initializes.

**What this tells us about the format (the viewer-relevant part):**
- **1.x uses version `0x100`** — the SAME version field as 1.12.1 (the 1.12.1 trace doc
  records `0x100`/`0x101`). This **corrects `research.md` line 28**, which grouped 1.0.0
  with 0.11/0.12 as "pre-256". Only **0.11 and 0.12 are pre-256**; **1.0.0 is 256**.
- The 1.0.0 *game client* rejects anything `!= 0x100` as `Corrupt model data` — this is a
  fact about the native client, **not** the viewer's bug. It confirms the 1.x on-disk
  format is `MD20` + version `0x100` with the layout recovered in §4.
- The extension gate (§3) is **not** a factor: `.mdx`/`.mdl` are accepted and normalized
  to `.m2` on 1.0.0, exactly as on later builds.

**The actual viewer gap (per user, 2026-07-15):** the wow-viewer M2 reader already
handles **0.11 / 0.12** (pre-`0x100`) fine; **1.x+ does not render correctly**. The
format was expanded incrementally from 1.0 (`0x100`) through 3.0.1, so no single layout
covers the whole range.

> **Design direction (user decision):** the M2 reader should **accept any version** and
> **dispatch to a per-version codepath** (one each for the 1.0 / 1.x / … / 3.0.1 steps
> where the layout changed), rather than hard-rejecting unknown versions. This Ghidra
> trace fully specifies the **`0x100` (1.0.0) codepath** — header map (§4), embedded
> divisions (§5), bones/animation (§6), lighting (§7), shaders (§8), and the remaining
> blocks (§9). Pre-`0x100` (0.11/0.12) already works and needs no new codepath here;
> later versions (TBC `~257–263`, WotLK `0x108`, 3.0.1, Cata `0x109+`) get their own
> codepaths as they are traced/validated.

---

## 3. Model cache + extension gate — `FUN_00721800` (`M2Cache.cpp`)

Decompiled: `output/ghidra_1.0.0/` (cache gate). Behavior:

- Copies the requested path, `strrchr` for the extension.
- Accepts exactly three extensions (data at `0x0083fba0/fba4/fbac`): the canonical one is
  kept as-is; the other two are rewritten to it. Mapping (matches 3.3.5 `FUN_0081c390`):
  - `.m2` → kept (canonical)
  - `.mdx`, `.mdl` → rewritten to `.m2`
  - anything else → **`Model2: Invalid file extension: %s`** (`0x0083fb48`)
- Opens the normalized `.m2`; missing → **`Model2: File not found: %s`** (`0x0083fb84`).
- Hashes the lowercased basename: `hash = hash*0x13 + tolower(c)`, bucketed into
  `hash % 0x3fd` (1021 buckets). Cache lookup is by hash then `strncmp(name, +0x20, 0x104)`.
- On miss, allocates a `CM2Shared` (RTTI `.?AVCM2Shared@@`), ctor `FUN_0071e540`,
  kicks the async load via `FUN_0071e7b0` (which installs completion callback
  `FUN_0071e940`), stores name at `+0x20`, hash at `+0x12C`, links into the bucket.

So the cache identity is the `.m2` basename; `.mdx`/`.mdl` are compatibility aliases only.

---

## 4. MD20 parser / relocator — `FUN_0071e190` (`M2DataInit.inl`)

This is the function that validates the header and relocates every `M2Array<T>`
(`{count, offset}` → `{count, base+offset}`) with bounds checks against the file size.
It is the 1.0.0 equivalent of the 3.3.5 `FUN_00987830` raw parser.

**Header (version 0x100) — complete field map.** `param_3` is the in-memory header
(indexed in dwords; byte offset = index*4). The first four arrays are relocated inline;
the rest delegate to per-block sub-parsers (§5).

| Byte ofs | Field | Element size | Relocator | Notes |
| --- | --- | --- | --- | --- |
| 0x00 | magic | 4 | inline | must be `0x3032444d` (`'MD20'`) |
| 0x04 | version | 4 | inline | must be `0x100` |
| 0x08 | nameLength | 4 | inline | |
| 0x0C | nameOfs | 4 | inline | → base+ofs |
| 0x10 | globalFlags | 4 | — | standalone dword |
| 0x14 | globalSequences.count | 4 | inline | |
| 0x18 | globalSequences.ofs | 4 | inline | array of `uint32` (count*4) |
| 0x1C | sequences.count | 4 | inline | |
| 0x20 | sequences.ofs | 4 | inline | **M2Sequence = 0x44 (68 B)** |
| 0x24 | sequenceLookup.count | 4 | inline | `sequenceFallbackById` |
| 0x28 | sequenceLookup.ofs | 4 | inline | array of `int16` (count*2) |
| 0x2C | (lookup).count | 4 | inline | array of `uint32` (count*4) — likely keyBoneLookup |
| 0x30 | (lookup).ofs | 4 | inline | |
| 0x34 | **bones** | **0x6c (108)** | `FUN_0071f440` | M2Bone, with nested tracks (§6) |
| 0x3C | keyBoneLookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0x44 | **vertices** | **0x30 (48)** | `FUN_0071f670` | M2Vertex |
| 0x4C | **divisions** | **0x2c (44)** | `FUN_0071f6f0` | **embedded skin profiles** (§5) |
| 0x54 | colors | 0x38 (56) | `FUN_0071f7a0` | M2Color |
| 0x5C | **textures** | **0x10 (16)** | `FUN_0071f930` | `{type, flags, nameLen, nameOfs}` |
| 0x64 | textureWeights | 0x1c (28) | `FUN_0071fa00` | M2TextureWeight |
| 0x6C | textureTransforms | 0x1c (28) | `FUN_0071fb20` | M2TextureTransform |
| 0x74 | replaceableTexLookup | 0x54 (84) | `FUN_0071fc40` | |
| 0x7C | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0x84 | lookup | 4 | `FUN_0071fe10` | `uint32[]` |
| 0x8C | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0x94 | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0x9C | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0xA4 | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0xAC | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0xEC | lookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0xF4 | (block) | 0xc (12) | `FUN_0071f3c0` | |
| 0xFC | (block) | 0xc (12) | `FUN_0071f3c0` | |
| 0x104 | **attachments** | **0x30 (48)** | `FUN_0071fe90` | M2Attachment |
| 0x10C | attachmentLookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0x114 | **events** | **0x2c (44)** | `FUN_0071ffc0` | M2Event |
| 0x11C | **lights** | **0xd4 (212)** | `FUN_007200b0` | M2Light (§7) |
| 0x124 | **cameras** | **0x7c (124)** | `FUN_00720450` | M2Camera |
| 0x12C | cameraLookup | 2 | `FUN_0071f2c0` | `int16[]` |
| 0x134 | **ribbons** | **0xdc (220)** | `FUN_00720600` | M2Ribbon |
| 0x13C | **particles** | **0x1f8 (504)** | `FUN_007208e0` | M2Particle (§8) |

Header length ≥ 0x140 bytes. The block order matches the classic M2 layout; the
**sizes differ from later eras** (e.g. 3.3.5 M2Light is 0x9c; here it is 0xd4), so the
1.0.0 reader must use these sizes, not the WotLK ones.

**Generic relocators**:
- `FUN_0071f2c0` — `M2Array<int16>`: validates `ofs + count*2 <= size`, sets `ofs += base`.
- `FUN_0071fe10` — `M2Array<uint32>` (count*4).
- `FUN_0071f3c0` — `M2Array<T>` with 0xc stride.
- Every relocator returns 0 (→ `Corrupt model data`) on any out-of-bounds offset, and
  treats `count==0` by nulling the pointer (sentinel `-1`/`0` handling lives in the
  per-block parsers).

**Init after parse** — `FUN_0071eab0` (`output/ghidra_1.0.0/model_init.c`, 178 lines):
runs after `FUN_0071e190` succeeds; builds the runtime side of the model (selects the
active division, builds render geometry, wires textures/materials). A failure here logs
**`Failed to initialize model: %s`** (`0x0083fa14`). The completion callback then sets
the loaded flag and rebuilds all live `CM2Model` instances via `FUN_007098b0`.

---

## 5. Embedded skin profiles — "divisions" (`FUN_0071f6f0` + `FUN_0071e0b0` + `FUN_006b7720`)

**There are no external `.skin` or `.anim` files on 1.0.0** — confirmed by string sweep:
zero hits for `.skin`, `%02d.skin`, and `.anim`. Skin profiles are **embedded** in the M2
as **`divisions`** (the 1.0.0 name for what 3.3.5 calls "views" and later externalizes to
`ModelNN.skin`). Assert string: `data->divisions.count > 0` (`0x00837bd0`).

**M2Division = 0x2c (44 bytes)**, relocated by `FUN_0071f6f0`, internals by `FUN_0071e0b0`:

| Div ofs | Field | Element size | Meaning |
| --- | --- | --- | --- |
| 0x00 | vertexLookup.count | — | |
| 0x04 | vertexLookup.ofs | 2 (`int16[]`) | remap table: division-local vtx → global M2Vertex |
| 0x08 | indices.count | — | |
| 0x0C | indices.ofs | 2 (`int16[]`) | triangle index buffer |
| 0x10 | (uint32 array).count | — | |
| 0x14 | (uint32 array).ofs | 4 (`uint32[]`) | bone-palette / property table |
| 0x18 | sections.count | — | |
| 0x1C | sections.ofs | 0x20 (32 B) | render-section / geometry-group records |
| 0x20 | **batches.count** | — | assert `division->batches.count == 1` (`0x00837bb0`) |
| 0x24 | batches.ofs | 0x18 (24 B) | M2Batch — material/texture binding (`FUN_0071f340`) |

**Materialization** — `FUN_006b7720` (`output/ghidra_1.0.0/FUN_006b7720_parser.c`) is the
1.0.0 analog of the 3.3.5 skin-profile section init (`FUN_00837a40`). It:
- validates `data->textures.count == 1`, `data->divisions.count > 0`,
  `division->batches.count == 1`, `data->bones.count == 1`;
- builds the **render vertex buffer** (0x20 / 32-byte render vertices) by walking the
  division's `vertexLookup` (ushort) into the global vertex table at `data+0x48`
  (M2Vertex = 0x30), copying 8 fields (position/normal/uv/etc.);
- builds the **index buffer** from `division.indices`, using a hash-bucket dedupe
  (`FUN_0059a180`) keyed by vertex index.

> **Reader contract for 1.0.0**: pick LOD/division 0; build submeshes from the division's
> 0x20-byte section records (vertexStart/Count + triangleStart/Count over the global
> vertex array); bind material per 0x18-byte batch. This is exactly the
> `contracts/m2-format-profile.md` "reader output contract" shape — just sourced from
> the embedded division instead of an external `.skin`.

---

## 6. Bones + animation system

**M2Bone = 0x6c (108 bytes)** (`FUN_0071f440`, `output/ghidra_1.0.0/bones_parser.c`).
Each bone carries three animation tracks (translation/rotation/scale), each an
`M2Track<T>` = `{count, ofs, …}` over 8-byte spline keys, plus lookups:

| Bone ofs | Field | Element size | Relocator |
| --- | --- | --- | --- |
| 0x10 | translation.count | — | inline (count*8) |
| 0x14 | translation.ofs | 8 | → base; then `FUN_0071f5f0` (count*4 timestamps) |
| 0x18 | (track meta) | — | `FUN_0071f5f0` |
| 0x20 | (lookup) | — | `FUN_0071f3c0` (0xc) |
| 0x2C | rotation.count | — | inline (count*8) |
| 0x30 | rotation.ofs | 8 | → base; `FUN_0071f5f0` |
| 0x34 | (track meta) | — | `FUN_0071f5f0` |
| 0x3C | (lookup) | — | `FUN_00720d30` (count*0x10) |
| 0x48 | scale.count | — | inline (count*8) |
| 0x4C | scale.ofs | 8 | → base |
| 0x50 | (lookup).count | — | inline (count*4) |
| 0x54 | (lookup).ofs | 4 | → base |
| 0x58 | (final lookup) | — | `FUN_0071f3c0` (0xc) |

Bone parent is `parentIndex` (`0xFFFF` = root), asserted:
`((0xffff) != data->bones[boneIndex].parentIndex) || (boneIndex > 0)` (`0x0083f3a8`).

**Animation update** — `FUN_0070f960` (`output/ghidra_1.0.0/` animation, ~1000 lines) is
`CM2Model::Update`: per-frame pose evaluation. It:
- samples each track against the active sequence's time range
  (`param_1+0x7c` = current anim time vs track ranges at `iVar11+0x94/0xb0/…`),
  using helpers `FUN_0070f6d0` (scalar track) and `FUN_00716330` (vector track);
- builds **bone matrices at 0x118 (280-byte) stride** in the bone-matrix buffer
  (`param_1+0x80`), indexed by `boneCount * 0x118`;
- walks **child/attached models** via the linked list at `param_1+0x1c8` (next at
  `+0x1d0`), recursively calling `FUN_0070f960` with inherited transform;
- asserts `data->bones.count > 0` (`0x0083f6e4`).

**Sequences**: `M2Sequence = 0x44 (68 B)` at header 0x1C; `sequenceFallbackById` (alias
table, `int16[]`) at 0x24, asserted `sequenceId < data->sequenceFallbackById.count`
(`0x0083f284`). **Animations are embedded** — no external `%04d-%02d.anim` loading exists
on 1.0.0 (unlike 3.3.5/Cata). `AnimationData.dbc` (`0x0081d5bc`) supplies sequence metadata.

---

## 7. Lighting

**M2Light = 0xd4 (212 bytes)** at header 0x11C (`FUN_007200b0`). Model-local lights are
evaluated per-frame in `M2Light.cpp`:

- `FUN_0071c9d0` (`output/ghidra_1.0.0/lighting_update.c`) — light **attenuation/falloff**
  over a `[start, end]` range (asserts `start < end`, `0x0083f8c0`). Computes a quadratic
  falloff: `fVar1 = k/(start*start)`, `fVar3 = (start+end)*k2`, then stores attenuation
  coefficients at light `+0x58` and `+0x5c`. This is the near/far light-falloff model.
- `FUN_0071ce30` — the second `M2Light.cpp` function (companion evaluator; deferred —
  same source file, handles the alternate light-type/flag path).
- Asserts: `lightIndex < m_shared->m_data->lights.count` (`0x0083f534`).

**Scene/world lighting** is DBC-driven (separate from model lights): `Light.dbc`,
`LightParams.dbc`, `LightIntBand.dbc`, `LightFloatBand.dbc`, `LightSkybox.dbc` are the
later-era set; on 1.0.0 the model-light path is the `M2Light.cpp` evaluator above.
`CreatureModelData.dbc` (`0x0081dcf8`) and `CreatureSoundData.dbc` map model paths to
metadata. `HelmetGeosetVisData.dbc` (`0x0081e26c`) drives helmet geoset visibility
(replaceable-texture / attachment visibility by geoset).

---

## 8. Shader / effect system — `.bls` + CGx + register combiners

**1.0.0 does NOT use the later `Combiners_*` / `Diffuse_T1` named-effect system.** String
sweep: zero `Combiners_*`/`Diffuse_*` hits; the only "combiner" strings are the OpenGL
extension probes `GL_NV_register_combiners`, `GL_NV_register_combiners2`,
`glCombinerStageParameterfvNV` (`0x00821b14…`). Shaders are the older **`.bls`** format
(Blizzard shader bytecode), not the later `.wfx`.

`FUN_007213b0` (`output/ghidra_1.0.0/shader_effect_load.c`, `M2Initialize`) sets up the M2
shader path:
- **Capability-gated**: `FUN_0042b8c0()` returns a GPU capability level; if `< 2`,
  shaders are disabled (`+0x400 = 0`), else `+0x400 = mode`. So M2 rendering has a
  fixed-function fallback.
- Registers the **`"Model2"`** effect (`0x0083fac4`) via `FUN_00651690`.
- Allocates a **`CGxVertexShader`** (RTTI `.?AVCGxVertexShader@@`, 0xe10 / 3600 bytes,
  900 dwords) when the renderer type (`FUN_0058d450() +0x5c` == 1 or 2) supports it.
  **CGx** is Blizzard's cross-API graphics abstraction (D3D + OpenGL); 900 vertex-shader
  constants.
- Sets up **`CGxTexFlags`** texture-combiner state (`FUN_0058df10`) with a combine mode
  selected from caps (3/4/5) — this is the **register-combiners / texture-combine** path
  on the OpenGL renderer.
- Fills a 0x200-float random table (`DAT_00aef770`) — a noise source for
  particle/vertex-shader randomness.
- `FUN_00780ff0()` finalizes (loads the `.bls` shader programs).

**`.bls` shader files referenced** (string sweep):
- `shaders\vertex\Model2.bls` (`0x0083fa88`) — the M2 vertex shader.
- Pixel shaders: `MapObjTransDiffuse.bls`, `MapObjTransSpecular.bls`, `MapObjSpecular.bls`,
  `MapObjMetal.bls`, `MapObjOverbright.bls`, `MapObjExtWater0.bls` (`0x00836a24…`) —
  map-object (WMO/doodad) pixel shaders reused by the M2 material path.
- Terrain/liquid: `terrain1..4.bls`, `terrainp*.bls`, `ocean0_s.bls`; post: `Desaturate.bls`,
  `FFXBlur_2.bls`, `FFXGlow_2.bls`, `FFXMidtoneMap.bls`.

**M2 runtime options** (cvar strings, minimal vs 3.3.5): `M2UseShaders` (`0x007f55bc`),
`M2UseThreads` (`0x007f55f0`). The later `M2UseZFill`/`M2UseClipPlanes`/`M2BatchDoodads`/
`M2BatchParticles`/`M2ForceAdditiveParticleSort`/`M2Faster` set does **not** exist yet on
1.0.0. (An `M2Batch` RTTI class exists — `.?AUM2Batch@@` — the 0x18-byte batch record in §5.)

> **wow-viewer implication**: a 1.0.0 M2 renderer does not need the `Combiners_*` effect
> vocabulary. It needs: (a) the `.bls`-class vertex transform + register-combiner texture
> combine, or (b) a modern GLSL equivalent. For parity, model the material as
> `{texture, blendMode, flags}` → a small fixed combine table, not the named-effect cache.

---

## 8a. Textures + materials

**M2Texture = 0x10 (16 bytes)** (`FUN_0071f930`): `{type, flags, nameLength, nameOfs}`
where `nameOfs` is relocated to the texture filename string (validated
`ofs + len <= size`). `data->textures.count == 1` is asserted at parse (`0x00837bec`) —
note this is a *parser invariant* (exactly one texture block expected), not a render-time
limit. Texture weights (`0x64`, 0x1c) and texture transforms (`0x6C`, 0x1c) animate
texture coords/alpha; replaceable textures route through `replaceableTexLookup` (`0x74`).

---

## 9. Particles, ribbons, cameras, attachments, events

- **Particles** — `M2Particle = 0x1f8 (504 B)` at header 0x13C (`FUN_007208e0`), the
  largest block. Accessed at runtime via `FUN_0070ef60` (`GetEmitter`): asserts
  `particleIndex < m_shared->m_data->particles.count` (`0x0083f614`), returns the emitter
  from the instance emitter array at `param_1+0x3a8 + index*4`. RTTI `M2ModelParticle`.
- **Ribbons** — `M2Ribbon = 0xdc (220 B)` at 0x134 (`FUN_00720600`). RTTI `M2ModelRibbon`.
- **Cameras** — `M2Camera = 0x7c (124 B)` at 0x124 (`FUN_00720450`), with
  `cameraLookup` (`int16[]`) at 0x12C. Asserts `cameraIndex < data->cameras.count`
  (`0x0083f580`) and `… < m_shared->m_data->cameras.count` (`0x0083f5c0`).
- **Attachments** — `M2Attachment = 0x30 (48 B)` at 0x104 (`FUN_0071fe90`),
  `attachmentLookup` at 0x10C. Assert `attachmentIndex < data->attachments.count`
  (`0x0083f408`).
- **Events** — `M2Event = 0x2c (44 B)` at 0x114 (`FUN_0071ffc0`). Assert
  `eventIndex < data->events.count` (`0x0083f4c8`).
- **Colors** — `M2Color = 0x38 (56 B)` at 0x54 (`FUN_0071f7a0`) — animated color blocks.

---

## 10. Render / scene submission — `M2Scene.cpp`

`FUN_00717da0` (`output/ghidra_1.0.0/scene_render.c`, 565 lines) is the M2 scene
submission path (one of several `M2Scene.cpp` functions: `FUN_0071a540`, `FUN_00719050`,
`FUN_00719220`, `FUN_00717d40`, `FUN_00717da0`, `FUN_00716800`). It walks the built render
sections from the active division (§5) and submits them through the CGx shader/combiner
state (§8). The submission is **division-driven**: the section records (0x20 B) and
batches (0x18 B) from §5 are the unit of draw dispatch. (Static analysis is sufficient to
identify the families; a full dynamic trace of the draw loop is optional follow-up.)

---

## 11. Native function anchors (1.0.0)

| Address | Function | Role |
| --- | --- | --- |
| `0x00721800` | `FUN_00721800` | `M2Cache` open: extension gate, normalize `.mdx`/`.mdl`→`.m2`, hash, cache lookup |
| `0x0071e540` | `FUN_0071e540` | `CM2Shared` constructor (zero fields) |
| `0x0071e7b0` | `FUN_0071e7b0` | async load job setup; installs completion cb `FUN_0071e940` |
| `0x0071e940` | `FUN_0071e940` | load completion: parse → init → rebuild instances; logs `Corrupt model data` / `Failed to initialize model` |
| `0x0071e190` | `FUN_0071e190` | **MD20 parser/relocator** — magic+version check, full header field map |
| `0x0071eab0` | `FUN_0071eab0` | post-parse model init (select division, build render geo) |
| `0x007098b0` | `FUN_007098b0` | live-instance rebuild after load |
| `0x0071f2c0` | `FUN_0071f2c0` | generic `M2Array<int16>` relocator |
| `0x0071fe10` | `FUN_0071fe10` | generic `M2Array<uint32>` relocator |
| `0x0071f3c0` | `FUN_0071f3c0` | generic relocator, 0xc stride |
| `0x0071f440` | `FUN_0071f440` | **bones** parser (M2Bone 0x6c + tracks) |
| `0x0071f670` | `FUN_0071f670` | **vertices** parser (M2Vertex 0x30) |
| `0x0071f6f0` | `FUN_0071f6f0` | **divisions** parser (M2Division 0x2c) |
| `0x0071e0b0` | `FUN_0071e0b0` | division internals (vertexLookup/indices/sections/batches) |
| `0x0071f340` | `FUN_0071f340` | division **batches** relocator (M2Batch 0x18) |
| `0x0071f7a0` | `FUN_0071f7a0` | colors parser (0x38) |
| `0x0071f930` | `FUN_0071f930` | **textures** parser (0x10) |
| `0x0071fa00` | `FUN_0071fa00` | textureWeights parser (0x1c) |
| `0x0071fb20` | `FUN_0071fb20` | textureTransforms parser (0x1c) |
| `0x0071fc40` | `FUN_0071fc40` | replaceableTexLookup parser (0x54) |
| `0x0071fe90` | `FUN_0071fe90` | attachments parser (0x30) |
| `0x0071ffc0` | `FUN_0071ffc0` | events parser (0x2c) |
| `0x007200b0` | `FUN_007200b0` | **lights** parser (0xd4) |
| `0x00720450` | `FUN_00720450` | cameras parser (0x7c) |
| `0x00720600` | `FUN_00720600` | ribbons parser (0xdc) |
| `0x007208e0` | `FUN_007208e0` | **particles** parser (0x1f8) |
| `0x006b7720` | `FUN_006b7720` | division materializer → render vertex (0x20) + index buffer |
| `0x0070f960` | `FUN_0070f960` | `CM2Model::Update` — animation pose eval, bone matrices (0x118 stride), child models |
| `0x0070f6d0` | `FUN_0070f6d0` | scalar track sampler |
| `0x00716330` | `FUN_00716330` | vector track sampler |
| `0x0070ef60` | `FUN_0070ef60` | particle `GetEmitter` |
| `0x0071c9d0` | `FUN_0071c9d0` | M2Light attenuation/falloff update |
| `0x0071ce30` | `FUN_0071ce30` | M2Light alternate evaluator |
| `0x007213b0` | `FUN_007213b0` | M2 shader init (`M2Initialize`): CGxVertexShader + CGxTexFlags + `Model2.bls` |
| `0x00780ff0` | `FUN_00780ff0` | `.bls` shader program load/finalize |
| `0x00717da0` | `FUN_00717da0` | M2 scene/render submission |

---

## 12. What this means for wow-viewer (implementation seeds for the fresh chat)

1. **Version dispatch (user design direction)**: the M2 reader should **accept any
   `MD20` version** and dispatch to a **per-version codepath**. The format expanded
   incrementally from 1.0 (`0x100`) through 3.0.1, so each layout-change step gets its
   own codepath. This trace specifies the **`0x100` (1.0.0 / 1.x) codepath**. Pre-`0x100`
   (0.11/0.12) already works in the viewer — leave it. Unknown/future versions should
   fall back to the nearest known codepath with a logged warning, not hard-abort (the
   game client hard-rejects; the viewer should be more permissive for inspection).
2. **Header reader**: implement the §4 field map with the 1.0.0 element sizes (NOT the
   WotLK sizes). Use the generic `M2Array<T>` relocator with strict bounds checks
   (return corrupt on any `ofs + count*stride > fileSize`).
3. **Embedded skin = divisions**: read `data->divisions` (0x4C), pick division 0, build
   submeshes from its 0x20-byte section records + 0x18-byte batches. Emit the same
   `{submesh index range + material binding}` contract the existing renderer consumes
   (per `contracts/m2-format-profile.md`), so `M2Renderer`/exporters need no version
   knowledge. This closes the Spec 104 "empty bounding box" bug for 1.0.0.
4. **No external `.skin`/`.anim`**: do not look for sidecar files for `0x100`. Animations
   are embedded (sequences 0x1C + bone tracks §6).
5. **Materials/shaders**: model as `{texture, blendMode, flags}` → a small fixed
   combine table. The `.bls`/register-combiner path is the native source of truth, but a
   modern GLSL equivalent is acceptable for the viewer; do NOT import the `Combiners_*`
   named-effect system for 1.0.0.
6. **Out of scope for the empty-box fix** (but now documented): animation playback,
   particles/ribbons, lights, cameras, attachments, events. The struct sizes above are
   enough to parse them later without re-RE.

---

## 13. Open follow-ups

- Pre-`0x100` (0.11 / 0.12) header layout — still unrecovered; needs the 0.12 client in
  Ghidra (or x64dbg). This is the *other* half of the user's "0.12 vs 1.x" question.
- 1.0.0 vs 1.12.1 header diff: both are version `0x100`; confirm whether field offsets
  match the existing `M2Era1121ModelReader` or differ (the 1.0.0 sizes here are the
  ground truth for 1.0.0 specifically).
- Full dynamic trace of the `M2Scene.cpp` draw loop (optional; static families identified).
- `.bls` bytecode format (if a faithful shader reimplementation is ever wanted).