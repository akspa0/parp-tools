**Here’s a side‑by‑side “Rosetta Stone” of WMOv14 (alpha) vs. Quake 3 BSP, based on the current wowdev.wiki documentation**. This should give you a concrete mapping for a converter experiment.  

---

# WMOv14 ↔ Quake 3 BSP Comparison

| **Concept** | **WMOv14 (Alpha, v14)** | **Quake 3 BSP** | **Notes for Conversion** |
|-------------|--------------------------|-----------------|---------------------------|
| **File structure** | Chunked IFF‑style (`MVER`, `MOMO`, `MOGP`, etc.) | Lump‑based BSP header with offsets/sizes | Both are containerized binary formats; WMO uses named chunks, BSP uses lump indices. |
| **Root vs. Groups** | Root WMO holds textures, materials, doodad refs; groups (`MOGP`) hold geometry | BSP has a single file with multiple lumps; submodels/leafs hold geometry | WMO groups ≈ BSP submodels/leafs. |
| **Geometry vertices** | `MOVT` (vertex positions), `MOVI` (indices), `MOPY` (face materials) | `vertices` lump, `faces` lump | Direct mapping possible: MOVT → vertices, MOVI+MOPY → faces. |
| **Textures** | `MOTX` (texture name strings), `MOMT` (materials referencing MOTX) | `textures` lump (shader names) | Convert BLP paths to placeholder Q3 shader names; optionally auto‑generate `.shader` scripts. |
| **Materials/shaders** | `MOMT` has flags (unlit, unfogged, two‑sided, emissive, etc.) | Q3 shader scripts define blend, fog, emissive | Many flags map conceptually; emissive → Q3 “surfaceparm” or light emission. |
| **Groups** | `MOGP` defines group geometry, flags, bounding box | BSP leafs/submodels | Treat each MOGP as a BSP leaf. |
| **Portals/visibility** | `MOPV` (portal vertices), `MOPT` (portal info), `MOPR` (portal refs) | `visdata` lump (PVS), portals implicit in BSP tree | WMO portals ≈ Q3 portals; can approximate by converting to vis portals. |
| **Lighting** | `MOLT` (light definitions: omni, spot, direct, ambient) | Lightmaps baked per surface | WMO lights can be ignored (to mimic “dark mines”) or baked into Q3 lightmaps. |
| **Doodads** | `MODN` (model names), `MODD` (instances with position/rotation/scale) | BSP doesn’t embed models; external MD3s placed in map | Convert doodads to entity placements referencing MD3 placeholders. |
| **Fog** | `MFOG` chunk defines fog zones | Q3 supports fog volumes via shaders | Map fog zones to Q3 fog brushes or shader fog volumes. |
| **Skybox** | `MOSB` (skybox filename) | Q3 skybox via shader/sky parameters | Direct mapping possible with shader substitution. |

---

## Conversion Roadmap

1. **Parse WMOv14 root (`MOMO`) and groups (`MOGP`).**  
   Extract vertices (`MOVT`), indices (`MOVI`), and face materials (`MOPY`).

2. **Build BSP geometry lumps.**  
   - Write vertices into BSP `vertices` lump.  
   - Write faces into BSP `faces` lump, referencing textures.  

3. **Translate textures/materials.**  
   - Convert `MOTX` BLP paths → Q3 shader names.  
   - Map `MOMT` flags to shader keywords (e.g., `surfaceparm nolightmap`, `cull disable`).  

4. **Handle portals/vis.**  
   - Convert `MOPV`/`MOPT` into Q3 portal polygons.  
   - Bake into `visdata` lump or approximate with leaf connectivity.  

5. **Lighting.**  
   - Option A: Ignore `MOLT` → dark maps (historically accurate).  
   - Option B: Bake `MOLT` lights into Q3 light entities or lightmaps.  

6. **Doodads.**  
   - Convert `MODN`/`MODD` into Q3 entity placements referencing MD3 placeholders.  

7. **Fog/Skybox.**  
   - Map `MFOG` → fog brushes.  
   - Map `MOSB` → Q3 skybox shader.  

---

## Why this matters
- **Historical validation:** If you can walk through a WMOv14 in a Q3 engine, it strongly supports the theory that Blizzard forked Q3’s map tech.  
- **Reproducible experiment:** You can literally recreate the “dark goldmine” Staats described by omitting `MOLT` lights.  
- **Tooling payoff:** A converter would serve as a Rosetta Stone for early WMO formats, bridging WoW archaeology with classic idTech workflows.  

---

**In short:** WMOv14 and Quake 3 BSP line up remarkably well — vertices, faces, textures, portals, and even lighting have direct analogues. The main work is writing a translator that maps WMO chunks (`MOVT`, `MOPY`, `MOTX`, `MOMT`, `MOPV`, `MOLT`) into BSP lumps (`vertices`, `faces`, `textures`, `visdata`, `lightmaps`).  

---

Source: wowdev.wiki’s detailed WMO documentation.  
use context7