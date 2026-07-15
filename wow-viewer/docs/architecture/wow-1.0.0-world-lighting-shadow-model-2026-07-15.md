# WoW 1.0.0 World Lighting / Shadow / Fog Model — Renderer Implementation Guide (2026-07-15)

Ghidra static trace of **WoW.exe 1.0.0.3980 (beta-3)**, image base `0x00400000`.
Goal: give the wow-viewer renderer a target that matches how the real 1.0.0 client
lit, shadowed, and fogged the world — so we stop approximating and start reproducing.

Companion docs (same client):
- Visibility / render order / portals / BSP: [`wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md`](wow-viewer/docs/architecture/wow-1.0.0-wmo-rendering-ghidra-trace-2026-07-15.md) §20
- Raw decompilations: [`evidence/1.0.0-ghidra/`](wow-viewer/specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/)

Confidence tags: **[V]** = verified in decomp/strings, **[I]** = inferred (era-standard,
consistent with the evidence) and flagged so we don't over-claim.

---

## 0. TL;DR — the reality of 1.0.0 world rendering

The 1.0.0 client is a **fixed-function-pipeline (FFP) renderer**. Lighting is *not* done
in shaders — the "shaders" (`.bls` + NV register combiners / ARB fragment programs) do
**texture combination only**. Lighting, material, and fog are classic OpenGL/D3D FFP state:

- **[V]** GL entry points imported and used: `glLightfv`, `glLightf`, `glLightModelfv`,
  `glLightModeli`, `glMaterialfv`, `glMaterialf`, `glColorMaterial`, `glFogfv`, `glFogf`, `glFogi`.
- **[V]** The world light is **one directional light (the sun/moon) + a global ambient**,
  optionally one omni. Exact model from the `SetLight` console command signature:
  ```
  SetLight(enabled[, omni, dirX,dirY,dirZ, ambIntensity[,ambR,ambG,ambB],
                             dirIntensity[,dirR,dirG,dirB]])
  ```
- **[V]** Colors/direction/fog come from the **day/night cycle** (`DayNight.cpp`,
  `LightData` / `LightDataFog` / `LightDataSky`) — they change continuously with time of day.
- **[V]** Terrain has **no vertex-color chunk** (no MCCV in 1.0.0). Terrain is lit
  dynamically from its **MCNR normals** against the outdoor light, then modulated by a
  **baked per-chunk shadow map (MCSH, 1 bit/texel)**.
- **[V]** Shadows are two separate systems: **baked terrain shadow maps (MCSH)** for the
  static world, and **projected/blob shadows (`ShadowBlob.blp`)** for units/creatures.
- **[V]** Fog is **fixed-function distance fog**, color + range from the day/night cycle,
  with **separate interior (in-WMO) and exterior fog** selected by the scene renderer.

**The one-line lighting equation the whole world obeys:**

```
litColor = ambientColor·ambientIntensity
         + dirColor·dirIntensity · max(0, N·L) · shadowFactor
finalColor = combineTextures(litColor, layers...)          // register combiners / FFP texenv
fogged     = lerp(finalColor, fogColor, fogFactor(dist))    // FFP fog
```

Everything below is the detail needed to reproduce each term.

---

## 1. The outdoor light (sun + ambient)

**[V]** State block, exposed as console vars registered in `FUN_0053ae20`
(`CVar name → handler`):

| CVar | Meaning |
|------|---------|
| `PLightEnable` | master enable |
| `PLightOmni` | treat the directional light as an omni/point light instead |
| `PLightDirPos` | light direction (for directional) / position (for omni) — `dirX,dirY,dirZ` |
| `PLightDirColor` | directional (sun) RGB |
| `PLightDirIntens` | directional intensity scalar |
| `PLightAmbColor` | ambient RGB |
| `PLightAmbIntens` | ambient intensity scalar |
| `PLightInfo` | debug dump of all of the above |

- **[V]** Applied through FFP: the client sets a single GL light (`GL_LIGHT0` equivalent)
  = the sun, plus `glLightModelfv(GL_LIGHT_MODEL_AMBIENT, …)` for the global ambient, and
  `glColorMaterial` so per-vertex color (where present, e.g. WMO MOCV) feeds the material.
- **[V]** Debug string `"Ambient intensity: %.2f, RGB: (%.2f, %.2f, %.2f)"` confirms
  ambient is stored as `intensity + RGB` (intensity is a separate multiplier, not premultiplied).
- **[I]** The lighting is **Lambert diffuse only** — no specular term in the outdoor light
  path (specular exists only in specific WMO materials via the `MapObjSpecular`/`…Metal`
  register-combiner shaders, see the WMO doc). Renderer: outdoor world = `ambient + N·L·sun`.

### Renderer action
- Maintain one directional light (sun) `{dir, color, intensity}` and one ambient
  `{color, intensity}`, both updated every frame from the day/night cycle (§2).
- Light everything (terrain, WMO exterior, M2) with `ambient + sun·max(0,N·L)`; do it
  **per-pixel** for quality (the client did per-vertex FFP — per-pixel is a free upgrade
  that stays faithful to the model).

---

## 2. Day/Night cycle → light, fog, sky (`DayNight.cpp`)

**[V]** Classes: `LightData` (the combined light state), `LightDataFog` (fog params),
`LightDataSky` (sky/skybox colors). The cycle drives sun direction + sun color + ambient
color + fog color/range + sky gradient as a function of time of day. `FUN_006d2460`
takes an `alpha ∈ [0,1)` blend factor (time-of-day interpolation weight) — the client
**interpolates between keyframed light sets**.

- **[V]** Skybox (from the renderer-features trace): `LightDataSky` + `DNOverrideSky`, sky
  models under `Environments\Stars\*`, cvars `SkyShow/SunGlare/CloudDensity/…`.
- **[I]** Source of the keyframes: a light table keyed by world position (light zones) +
  time. 1.0.0 has **no `Light.dbc` string** — the outdoor light set is driven from
  `DayNight.cpp`/`LightData` (either hardcoded gradients or an early data table), not the
  later `Light.dbc`/`LightParams` DBC chain. Treat the *values* as data to author, not
  something to read from a modern DBC.

### Renderer action
- Implement a time-of-day → `{sunDir, sunColor, ambientColor, fogColor, fogStart, fogEnd,
  skyColors}` evaluator with smooth interpolation between keyframes. Start with a single
  global set (dawn/noon/dusk/night); add per-zone light sets later.
- Sun direction sweeps across the sky with time; do **not** hardcode a fixed sun (except
  the *shadow* projection, which the client does fix — see §5).

---

## 3. Terrain rendering & lighting (`MapChunk.cpp` / `MapChunkRender.cpp`)

### 3.1 Chunk structure — `FUN_006b4920` (`CMapChunk::Read`) **[V]**
MCNK header offset table → sub-chunk pointers:

| Sub-chunk | Header ofs | Content |
|-----------|-----------|---------|
| MCVT | +0x14 | 145 height values (9×9 + 8×8 grid) |
| MCNR | +0x18 | 145 per-vertex **normals** (the lighting input) |
| MCLY | +0x1c | texture layer defs (up to 4 layers) |
| MCRF | +0x20 | doodad/object refs |
| MCAL | +0x24 | alpha maps (blend weights for layers 2-4) |
| MCSH | +0x2c | **baked shadow map** (1 bit/texel) |
| MCSE | +0x58 | sound emitters |
| MCLQ | +0x60 | liquid (MCLQ, pre-MH2O) |

No MCCV / MCLV → **terrain color is computed from MCNR normals at runtime**, never stored.

### 3.2 Texture blending **[V]**
- Up to **4 layers** (MCLY). Layer 0 is opaque base; layers 1-3 are alpha-blended using
  the **MCAL alpha maps**. Composited by the FFP multitexture / register combiners.
- Chunk render `FUN_006c0bc0` binds the layer textures + alpha and issues the draw; it
  branches into a plain path (`FUN_006c0db0/…e80`) vs. a **shadow+fog path**
  (`FUN_006c65c0` + `FUN_006c0eb0/…f20`) when the chunk has a shadow map or fog.

### 3.3 Terrain lighting = normals · sun + ambient **[V model / I mechanism]**
- The lighting **model** is `ambient + sunColor·sunIntensity·max(0, N·L)` using the MCNR
  normal per vertex, identical to the outdoor light of §1.
- **[I]** Mechanism: the client either (a) uploads MCNR normals and lets FFP light them
  with the one global GL light, or (b) pre-bakes per-vertex color from normals at
  light-change time. Either way the renderer should compute the same term. Per-pixel with
  interpolated normals is the faithful-but-better choice.

### 3.4 Terrain shadow map (MCSH) **[V]**
- MCSH is a **64×64 1-bit-per-texel** shadow mask per chunk (read as `byte & (1<<(x&7))`,
  confirmed in the detail-doodad renderer `FUN_006c1c50` and the parser). Baked offline
  (static world/doodad shadows).
- Uploaded to `chunk->shadowGxTexture` and **multiplied over the lit terrain** (darkens
  shadowed texels). Toggle: `mapShadows` cvar / "Terrain shadow enabled/disabled",
  mip level 0-1 ("Terrain shadow map mip level").

### 3.5 Detail doodads / grass — `FUN_006c1c50` **[V]**
- Procedural grass/detail scatter per chunk per texture layer, placed with a deterministic
  PRNG (`FUN_00457e20`), density from the layer, and **shadowed by sampling the same MCSH
  bit** so grass under shadow is darkened consistently with the ground.
- Toggles: `showDetailDoodads` / `showSimpleDoodads`.

### 3.6 Terrain LOD — `FUN_006c65c0` **[V]**
- Quadtree triangle-strip index generator with **neighbour-aware edge stitching**: the
  three packed bytes of `param_3` are the neighbour LOD levels, and the generator inserts
  T-junction-free seam triangles. Renderer: match the LOD seam handling or accept cracks.

### Renderer action (terrain)
1. Light per-vertex/pixel from MCNR normals with the §1 sun+ambient — **replace any flat/
   unlit terrain shading**.
2. Multiply by the **MCSH** shadow mask (sample the 64×64 bitfield, bilinear for softness).
3. Blend up to 4 MCLY layers via MCAL alpha.
4. Apply distance fog (§6). Add grass with MCSH-consistent shadowing later.

---

## 4. WMO lighting (interior/exterior) — see companion doc §6, plus:

- **[V]** Three light contributions (from the WMO trace): **MOCV** pre-baked vertex colors,
  **MOLR** dynamic light refs → `CMapLight` (`MapLight.cpp`), and a `CMapCacheLight` cache.
  `mapObjLightLOD` (0-2) controls dynamic-light LOD; `mapObjOverbright` boosts interiors.
- **[V]** Interior groups use their own ambient (set during the portal walk, `FUN_006b9d30`
  sets the in-WMO fog/ambient flag `DAT_00aadec8`); exterior groups use the outdoor light.
- **[V]** Register-combiner material shaders select the look: `MapObjSpecular`,
  `MapObjTransSpecular`, `MapObjTransDiffuse`, `MapObjOverbright`, `MapObjMetal`,
  `MapObjExtWater0` — this is where WMO **specular/metal** comes from (the only specular in
  the pipeline). Material flags (MOMT, 0x40 B) pick the pass; MOPY flags are collision-only.

### Renderer action (WMO)
- Interior: use MOCV (baked) as the base, add MOLR point lights, use the WMO's ambient (not
  the sky ambient). Exterior: sky light. Implement the specular/metal/overbright material
  variants (even as approximations) — they define the "indoor" look.

---

## 5. Shadows

### 5.1 Terrain (static) — **MCSH baked map [V]** (§3.4). Cheap, always-on, per-chunk.

### 5.2 Units / creatures — **projected / blob [V]**
- `FUN_006d5610` sets up the unit-shadow render: a **`Textures\ShadowBlob.blp` decal**
  projected onto the ground under each unit (blend state via `FUN_0058df10`, callbacks
  `FUN_006d5730/…5880/…5ab0/…5b90`).
- Cvars: `shadowLevel`, `shadowBias` ("Unit shadow depth bias", range 0-1),
  `shadowLOD` ("Unit shadow LOD", 0-1) → the client supports a **depth-biased projected
  shadow** (higher LOD) and falls back to the **blob decal** (lower LOD).
- **[V]** Shadow projection uses a **fixed sun angle** (`FUN_006cbd50` uses `cos(π/4)`),
  i.e. unit shadows are cast with a constant ~45° light, *not* the moving day/night sun.

### Renderer action (shadows)
- Static world: sample MCSH (and, for interiors, MOCV darkening) — do **not** try to
  real-time-shadow the static world; the client didn't.
- Units/dynamic models: blob decal by default (fast, faithful), optional projected shadow
  with depth bias for higher settings. Project along a fixed ~45° light.

---

## 6. Fog (`glFog*`) **[V]**

- Fixed-function distance fog: `glFogfv(GL_FOG_COLOR,…)`, `glFogf(GL_FOG_START/END/DENSITY)`,
  `glFogi(GL_FOG_MODE,…)`. ARB modes exp2/exp/linear exist (from the WMO trace).
- Color + range come from the day/night `LightDataFog`. Console: `SetFogColor`/`FogColor`,
  `SetFogNear`/`SetFogFar`/`ClearFog`.
- **[V]** The scene renderer (`FUN_0067c460`, companion doc §20.2) **selects interior vs
  exterior fog** per frame: `DAT_00aadec8` (set when the camera/visible group is inside a
  WMO) chooses fog params at object `+0xe4` (interior) vs `+0xf8` (exterior).

### Renderer action
- Implement per-pixel distance fog `lerp(color, fogColor, f(dist))`; drive `fogColor`,
  `fogStart`/`fogEnd` from the day/night cycle; switch to interior fog when the camera is
  inside a WMO. Exterior fog end also functions as the **far clip / draw distance** feel.

---

## 7. Render order (recap — companion doc §20.2)

`CWorldScene::Render` (`FUN_0067c460`): begin → **camera-in-WMO test** → interior portal
walk **or** exterior pass → opaque world (terrain + WMO opaque) → **fog select** →
transparent/effect passes (liquids, particles) → portal debug. Frustum is a 32-deep stack;
visibility is **screen-rect portal culling** for interiors and frustum culling for the
open world.

---

## 8. Gap analysis — current CPU renderer vs. the 1.0.0 reality

Prioritized for maximum visual truth per unit of work:

| # | Gap (likely current state) | 1.0.0 reality | Priority |
|---|----------------------------|---------------|----------|
| 1 | Terrain flat/unlit or minimap-textured | `ambient + sun·N·L` from **MCNR normals** | **P0** — biggest look delta |
| 2 | No day/night light | sun dir+color & ambient **interpolated by time of day** | **P0** |
| 3 | No terrain shadows | **MCSH** 1-bit/texel baked map, multiplied over terrain | **P1** |
| 4 | No distance fog / hard far clip | FFP fog, day/night color+range, interior/exterior select | **P1** |
| 5 | WMO interiors lit like exteriors | MOCV base + MOLR point lights + WMO ambient; specular/metal/overbright materials | **P1** |
| 6 | No unit shadows | ShadowBlob decal (default) / projected depth-biased shadow (fixed ~45°) | **P2** |
| 7 | Terrain single-texture | up to 4 MCLY layers, MCAL alpha blend | **P2** |
| 8 | No grass/detail | procedural detail doodads, MCSH-shadowed | **P3** |
| 9 | LOD cracks | quadtree strips with neighbour seam stitching | **P3** |

**Suggested build order:** (P0) sun+ambient day/night light applied to normal-lit terrain
and M2 → (P1) MCSH terrain shadows + distance fog + WMO interior lighting → (P2) 4-layer
terrain blend + unit blob shadows → (P3) grass + LOD seams.

None of P0-P1 needs a GPU shader graph — it's a directional light + ambient + a shadow
texture multiply + a fog lerp. That is the "improve the renderer yesterday" path: it is
small, faithful, and it is where ~all of the missing realism lives.

---

## 9. Evidence / function map (this trace)

| Function | Role |
|----------|------|
| `FUN_0053ae20` | registers the `PLight*` light cvars (light-state layout) |
| `FUN_006d2460` / `FUN_006d2190` | DayNight blend/fadeout setters (`alpha`, `amount`) |
| `FUN_006cbd50` | DayNight shadow-projection transform (fixed `cos(π/4)` sun) |
| `FUN_006b4920` | `CMapChunk::Read` — MCNK sub-chunk table (MCVT/MCNR/MCLY/MCAL/MCSH/MCLQ/MCSE) |
| `FUN_006c0bc0` | terrain chunk render (layers, plain vs shadow+fog path) |
| `FUN_006c65c0` | terrain LOD triangle-strip generator (neighbour seam stitching) |
| `FUN_006c1c50` | detail-doodad/grass scatter (samples MCSH shadow bit) |
| `FUN_006d5610` | unit shadow init (`ShadowBlob.blp`, blob + projected) |
| `FUN_0067c460` | `CWorldScene::Render` — render order + interior/exterior fog select |

GL FFP entry points (imported, `0x007ed8ec`-`0x007ed9d0`): `glLightf/fv`,
`glLightModeli/fv`, `glMaterialf/fv`, `glColorMaterial`, `glFogf/fv/i`.

Source files: `DayNight.cpp`, `MapChunk.cpp`, `MapChunkRender.cpp`, `MapLight.cpp`,
`WorldScene.cpp`, `MapObjRender.cpp`.

---

## 10. Still open (for a later pass)

- Exact day/night keyframe **values** (sun/ambient/fog per time slot) + whether zones
  override them — needs dynamic capture (x64dbg) or the light data table located.
- ~~Whether terrain lighting is FFP-GPU or a CPU vertex-color bake~~ — **RESOLVED**: terrain VBO
  is `position + normal` (24 B, no color), so it's **hardware FFP lighting** (normals uploaded, GPU
  computes N·L). See [world-render-systems](wow-viewer/docs/architecture/wow-1.0.0-world-render-systems-2026-07-15.md) §3.1.
- Per-light MOLR record layout; MOCV exact blend into the FFP color; `mapObjOverbright` math.
- The FFP fog mode/density constants (linear vs exp2) actually used outdoors vs indoors.
