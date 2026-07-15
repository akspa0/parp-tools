# WoW 1.0.0 Renderer Features — Ghidra Static Trace (2026-07-15)

Static reverse-engineering of the **WoW 1.0.0** client's world/M2 renderer features
beyond the M2 format itself: **liquids, particles, ribbons, attachments (armor/equipment),
helmet/hair geosets, skybox/sky, and camera/POV**. Recovered via the GhidraMCP HTTP API
(see [`specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md`](../../specs/104-legacy-m2-rendering/research-1.0.0-ghidra-trace.md) §1 for tooling).
Raw decompilations: `output/ghidra_1.0.0/liq_*.c` and `feat_*.c`.

This is implementation seed material for the wow-viewer renderer — the goal the user
stated: eventually reproduce all the rendering the 1.0.0 client does, in the viewer's
own renderer.

---

## 1. Liquids (water / ocean / lava / slime) — `MapWater.cpp`

**The viewer currently renders only "magma or water"; 1.0.0 has a 12-type liquid system.**

### Liquid types
- **`LIQUID_COUNT = 0xC` (12 types, indices 0–11)**, with a `LIQUID_NONE` sentinel
  (`liquid != LIQUID_NONE`, `liquid < LIQUID_COUNT`). Asserted in `FUN_00686d40`:
  `if (0xb < param_1) assert(liquid < LIQUID_COUNT)`.
- A **`liquidTexBaseName[liquid]` table at `0x00834d4c`** (12 string pointers) maps each
  liquid type to a `printf`-style animated-frame base name. Confirmed base names:
  - `XTextures\river\lake_a.%d.blp`  (river/lake water — table index 0, per Ghidra label)
  - `XTextures\river\fast_a.%d.blp`  (fast-flowing river)
  - `XTextures\ocean\ocean_h.%d.blp` (ocean)
  - `XTextures\slime\slime.%d.blp`   (slime)
  - `XTextures\lava\lava.%d.blp`     (lava = magma)
  - `XTextures\splash\wake.blp`, `XTextures\splash\splash.blp` (splash effects)
  - `Textures\WaterPoop02.blp`       (water fallback/variant)
  - (remaining table entries not yet individually index-mapped — read the 12 pointers at
    `0x00834d4c` to confirm each type's name; quick follow-up.)

### Texture animation
- `FUN_00686d40` (liquid texture resolver): for a given liquid type, **loads 30 animated
  frames** (`0x1e` = 30) from `liquidTexBaseName[type]` formatted as `base_<n>.blp`
  (n = 1..30), stored in a per-type array at `0x00a7e608 + type*0x1e` (30 texture handles
  per type). Returns the **time-animated current frame** (frame index derived from a time
  counter mod 30). So every liquid surface is a 30-frame looping animation.

### Storage / chunk format
- **MCLQ** (pre-3.0 liquid chunk), **NOT MH2O**. Confirmed: `iffChunk->token=='MCLQ'`
  (`0x00837930`). `FUN_006b4920` is the MCNK sub-chunk locator — it validates the MCNK
  sub-chunks in order and stores their pointers: `MCVT, MCNR, MCLY, MCRF, MCSH, MCAL,
  MCLQ (at chunk-header offset +0x60), MCSE`. So MCLQ is one sub-chunk of each terrain
  MCNK, exactly as in the existing `gillijimproject_refactor` ADT reader.
- Liquid grid: `MD_LIQUID_NPOLY` (8×8 tiles per chunk), `group->liquidVerts.x *
  liquidVerts.y` vertex grid. Asserted in `FUN_00687460` (liquid proximity/visibility
  query, used for water sounds + `WQF_liquid` queries).
- Runtime classes: `CChunkLiquid` (RTTI `.?AV?$TObjectAlloc@VCChunkLiquid@@`), `WCHUNKLIQUID`
  (`0x00835420`), created in `FUN_0068e9c0`.

### Shaders + effects
- `.bls` pixel shaders: `Shaders\Pixel\ocean0_s.bls` (ocean), `Shaders\Pixel\MapObjExtWater0.bls`
  (WMO exterior water). (No `vsLiquid*`/`psLiquid*` family yet — that naming is later; 1.0.0
  uses the generic `.bls` + CGx path.)
- **Ripple / wave system**: classes `Water0Ripple` (`.?AVWater0Ripple@@`), `WaterRadWave`
  (`.?AUWaterRadWave@@`). Ripple emission is gated by the `waterRipples` cvar.

### Cvars (liquid config surface)
`waterParticulates`, `waterRipples`, `waterSpecular`, `waterWaves`, `waterMaxLOD`,
`waterLOD` ("Water geometry LOD"), `SetWaterDetail`/`GetWaterDetail`, `Water enabled`/
`Water disabled`. Sound: `SoundWaterType.dbc` (`SoundWaterTypeRec`), `MapWaterSounds`.

### Native anchors
| Address | Function | Role |
| --- | --- | --- |
| `0x00686d40` | `FUN_00686d40` | liquid texture resolver: 30-frame anim per type, `liquidTexBaseName[type]` |
| `0x00686ee0` | `FUN_00686ee0` | liquid-type validation (`liquid < LIQUID_COUNT`) |
| `0x006b4920` | `FUN_006b4920` | MCNK sub-chunk locator (MCLQ at +0x60) |
| `0x0068e9c0` | `FUN_0068e9c0` | `CChunkLiquid`/`WCHUNKLIQUID` creation |
| `0x00687460` | `FUN_00687460` | liquid proximity/visibility query (8×8 grid, water sounds) |
| `0x0068afa0`…`0x006897b0` | MapWater.cpp family | MapWater render/management (multiple) |
| `0x006a4c20` | `FUN_006a4c20` | liquid-type query (`liquid != LIQUID_NONE`) |

> **wow-viewer implication**: extend the liquid renderer past magma/water to all 12 types
> via the `liquidTexBaseName` table (river/lake, river/fast, ocean, slime, lava, …), each
> with 30-frame animation. Source the type from the MCLQ chunk's liquidType field (already
> parsed by the existing ADT reader). Route ocean vs water vs lava to the right `.bls`-class
> shader. Add the ripple/wave system (`Water0Ripple`) for interactive water.

---

## 2. Particles — M2 + engine emitters

- **M2 particle block**: `M2Particle = 0x1f8 (504 B)` at M2 header `0x13C` (see M2 trace §9).
  Runtime access: `FUN_0070ef60` (`GetEmitter`) — asserts
  `particleIndex < m_shared->m_data->particles.count`, returns the emitter from the
  instance emitter array at `CM2Model+0x3a8 + index*4`. `GetNumEmitters` (`0x0083f604`).
- **Emitter class hierarchy** (RTTI): `CParticleEmitter2` (the M2 v2 emitter, `.PAVCParticleEmitter2@@`),
  `CParticleEmitter`, `CParticle2`, plus shape subclasses `CPlaneParticleEmitter`,
  `CSphereParticleEmitter`, `CSplineParticleEmitter`. Manager: `ParticleSystemManager`.
  Sorting: `CSortableParticleRecord`. Service interface: `IParticleMisc.h`.
- **Child emitters**: `childIndex != MAX_CHILD_EMITTERS` (`0x008461f8`) — emitters can spawn
  child emitters (chained particle effects).
- **Footprint particles**: `s_showFootPrintParticles` / `showfootprintparticles` cvar
  ("toggles rendering of footprint particles") — separate world-particle system for footprints.
- **Density cvar**: `particleDensity` ("Video option: Particle density") — LOD control.
- **Effect/shader**: the `"Particle"` effect name (`0x007f9eac`) is the particle shader/effect
  (loaded through the CGx `.bls` path, same as M2 — no `Particle_Unlit`/`Projected_*` named
  family yet on 1.0.0).

> **wow-viewer implication**: to render M2 particles, walk `data->particles` (0x1f8 records),
  build `CParticleEmitter2`-style emitters (plane/sphere/spline shapes), simulate + sort
  (`CSortableParticleRecord`), support child emitters, and gate by `particleDensity`. The M2
  trace §9 gives the struct size; the per-field layout of the 0x1f8 record is the next RE step.

---

## 3. Ribbons — `RibbonEmitter.cpp`

- **M2 ribbon block**: `M2Ribbon = 0xdc (220 B)` at M2 header `0x134` (M2 trace §9).
- Classes (RTTI): `CRibbonEmitter`, `RibbonManager`, `CRibbonMat` (ribbon material),
  `CRibbonVertex`. Source: `C:\build\buildWoW\ENGINE\Source\Services\RibbonEmitter.cpp`.
- Ribbons use their own material/effect path (`CRibbonMat`), separate from the doodad batch
  path — same architectural split as later eras.

> **wow-viewer implication**: ribbons are a distinct render family (trail effects behind
  weapons/spells). Walk `data->ribbons` (0xdc records), build `CRibbonEmitter`-style trails
  with per-vertex material. Separate submission path from particles.

---

## 4. Attachments — armor / equipment / mount anchor points

- **M2 attachment block**: `M2Attachment = 0x30 (48 B)` at M2 header `0x104`, with
  `attachmentLookup` (`int16[]`) at `0x10C` (M2 trace §9).
- **M2Attachment struct** (from `FUN_0070e500` GetAttachmentWorldTransform):
  - `+0x04`: `boneIndex` (ushort) — which bone the attachment is parented to.
  - `+0x08, +0x0C, +0x10`: position offset (3 floats) in bone-local space.
  - (remaining fields: rotation/flags — confirm in a focused pass.)
- **World transform formula** (`FUN_0070e500`):
  `attachmentWorld = modelWorldMatrix * boneMatrix[boneIndex] * attachmentOffset`
  where bone matrices live at `CM2Model+0x84` (0x40 / 4×4 stride) and the model world
  matrix at `CM2Shared+0xf8`. This is exactly how shoulder/helm/weapon equipment attaches
  to a character's bones.
- **Attachment API** (CM2Model): `HasAttachment` (`FUN_0070e350`), `GetAttachmentPivot`,
  `GetAttachmentPosition` (`FUN_0070e450`), `GetAttachmentWorldTransform` (`FUN_0070e500`).
- **Character equipment slots**: `CharacterAttachment` / `PetAttachment` (`0x007fd470/480`),
  `GetInventorySlotInfo` (`0x00812810`), `UNIT_FIELD_MOUNTDISPLAYID`. Mount attachment:
  `MOUNTDISPLAYIDNOMOUNTATTACHMENT|%d` (`0x008291cc`). Interact icon attachment:
  `Model %s is missing Interact Icon attachment`.

> **wow-viewer implication**: to attach armor/weapons to a character, resolve the
> attachment's `boneIndex` + offset from `data->attachments`, multiply by the evaluated
> bone matrix (from the animation system, M2 trace §6) and the model world matrix. This is
> the foundation for a character equipment renderer.

---

## 5. Helmet / hair geoset visibility

- **`HelmetGeosetVisData.dbc`** (`0x0081e26c`, `HelmetGeosetVisDataRec`) — controls which
  geosets (hair/ears) are hidden when a helmet is worn. Resolved in `FUN_0057ef40`.
- **`CharHairGeosets.dbc`** (`0x0081d8c4`, `CharHairGeosetsRec`) — character hair geoset
  definitions.
- Geoset visibility = toggling M2 divisions/sections (M2 trace §5) on/off per equipment.

> **wow-viewer implication**: when rendering a character with a helmet, consult
> `HelmetGeosetVisData.dbc` to hide the hair/ear geosets (skip those divisions/sections).
> This pairs with the attachment system (§4) for full equipment rendering.

---

## 6. Skybox / sky

- **Sky models** (MDX, loaded via the Model2 cache): `Environments\Stars\stars.mdl`
  (`0x00836058`), `Environments\Stars\DeathClouds.mdx` (`0x00827468`),
  `Environments\Stars\StratholmeSkybox.mdx` (`0x0083acfc`). So skyboxes are M2/MDX models
  rendered as a backdrop.
- **Sky data model** (RTTI): `LightDataSky` (`0x0083adb8`), `DNOverrideSky`
  (`0x0083b0c8`, dynamic night/day override). Sky system init in `FUN_006ce6c0`
  (references `SkyShow`).
- **Sky cvars**: `SkyShow`, `SkySunGlare`, `SkyCloudLOD`, `SkyCloudDensity`,
  `SkyCloudLayers`; `SkyCloudDensity set from data` / `set from override to %f`;
  `Sky enabled`; `The sky is falling.` (assert).
- **Lighting tie-in**: sky is part of the world-lighting system (the `Light*.dbc` family
  from the M2 trace §7); `LightDataSky` is the sky-specific light data record.

> **wow-viewer implication**: a skybox pass renders the `Environments\Stars\*` M2/MDX models
> as a backdrop + a cloud/sun-glare layer driven by `LightDataSky` and the day/night
> override. Reuses the M2 renderer (§M2 trace) for the sky models themselves.

---

## 7. Camera / POV (character-model-as-camera)

- **Engine camera**: `CGCamera` (`.?AVCGCamera@@`, `0x00804510`), `m_camera`
  (`0x008044fc`), `s_currentWorldFrame->m_camera` (`0x00804328`). Source:
  `CSimpleCamera.cpp` (`0x00815af4`), `Camera.cpp` (`0x008161d8`).
- **M2 model-defined cameras**: `M2ModelCamera` (`.?AUM2ModelCamera@@`), `M2Camera =
  0x7c (124 B)` at M2 header `0x124` with `cameraLookup` at `0x12C` (M2 trace §9). These
  are the cameras authored into model files (used for character preview / dressing room /
  cinematic model shots). `rec->m_Sex < UNITSEX_LAST || rec->m_Camera < CAMERA_LAST`
  (`0x007fe5e0`) — character camera variant selected per race/sex from a DBC
  (`CharSections`/creature camera field).
- **Camera cvars**: `cameraSmoothTimeMax/Min`, `cameraYawSmoothMax/Min`,
  `cameraPitchSmoothMax/Min`, `cameraSurfacePitch`, `cameraDivePitchMax`, `FlipCameraYaw`,
  `CameraZoomIn`, `CameraZoomOut`. So the camera is a smoothed yaw/pitch/zoom controller
  orbiting the player.
- **No true first-person string** on 1.0.0 (the `FirstPerson` hits are combat-log only) —
  the "single point of view" the user wants is the `CGCamera` orbiting camera + the
  `M2ModelCamera` model cameras, not a dedicated first-person mode.

> **wow-viewer implication**: to render "a character model as the camera" / a single POV,
  implement (a) the `CGCamera` smoothed orbit camera (yaw/pitch/zoom, surface pitch) for the
  world view, and (b) the `M2ModelCamera` path for model-authored camera shots (preview
  windows, character portrait). The M2 trace §9 gives the M2Camera struct size; the
  per-field camera layout (origin, target, fov, near/far) is the next RE step.

---

## 8. Cross-cutting: the CGx renderer abstraction

All of the above render through **CGx** — Blizzard's cross-API graphics layer
(`CGxVertexShader`, `CGxTexFlags`, `CGxDeviceD3d`) — with `.bls` shader programs and
`GL_NV_register_combiners` on the OpenGL path (see M2 trace §8). There is **no**
`Combiners_*`/`Diffuse_*` named-effect system on 1.0.0; materials are `{texture,
blendMode, flags}` → a small fixed combine table. The wow-viewer renderer should mirror
this: one CGx-like material/effect seam shared by terrain, WMO, M2, liquid, particle, and
ribbon paths.

---

## 9. Open follow-ups (per subsystem)

- **Liquids**: read the 12 pointers at `0x00834d4c` to fully index-map liquid type → base
  name; decompile the MapWater render function that routes type → shader (one of
  `FUN_0068afa0/89260/89340/89620/897b0`); MCLQ payload field layout (the existing ADT
  reader already parses this — cross-check).
- **Particles**: per-field layout of the 0x1f8 `M2Particle` record; the emitter
  simulate/submit function.
- **Ribbons**: per-field layout of the 0xdc `M2Ribbon` record.
- **Attachments**: remaining M2Attachment fields (rotation/flags beyond boneIndex+position).
- **Camera**: per-field layout of the 0x7c `M2Camera` record (origin/target/fov/near/far).
- **Skybox**: the sky render pass order + `LightDataSky` field layout.

All recoverable with the same GhidraMCP HTTP approach already wired up (`.mcp.json` `ghidra`
server).