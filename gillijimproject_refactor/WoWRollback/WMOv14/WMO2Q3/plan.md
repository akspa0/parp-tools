Here’s a **minimal proof‑of‑concept exporter outline** you can use as a starting point for a WMOv14 → Quake 3 BSP converter. It’s written in Markdown so you can drop it into your notes or repo as a design doc. I’ve based the mapping on the wowdev.wiki WMO documentation and the Quake 3 BSP lump structure.

---

# WMOv14 → Quake 3 BSP Exporter (Proof of Concept)

## 1. Parse WMOv14

- **Root chunks:**
  - `MVER` – version (v14 for alpha).
  - `MOTX` – texture name strings.
  - `MOMT` – material definitions (flags, texture indices).
  - `MODN` / `MODD` – doodad names and placements (optional for BSP).
  - `MFOG` / `MOSB` – fog and skybox (optional).

- **Group chunks (`MOGP`):**
  - `MOVT` – vertex positions.
  - `MOVI` – indices.
  - `MOPY` – face material assignments.
  - `MOTV` – texture coordinates.
  - `MOCV` – vertex colors (optional).
  - `MOPV` / `MOPT` / `MOPR` – portals and visibility.
  - `MOLT` – lights.

---

## 2. Build BSP lumps

- **Header:** Write BSP header with lump directory.
- **Vertices lump:** Fill with `MOVT` positions.
- **Faces lump:** Build from `MOVI` + `MOPY`. Each face references a texture index.
- **Textures lump:** Convert `MOTX` entries to Q3 shader names.  
  - Example: `textures/wmo/<basename>` → placeholder shader.
- **Lightmaps lump:** Optionally bake `MOLT` lights into lightmap data. For “dark mine” test, leave empty.
- **Visdata lump:** Convert `MOPV`/`MOPT` portal data into Q3 vis portals. If too complex, stub with trivial vis.
- **Entities lump:** Optionally place doodads (`MODD`) as Q3 entities referencing MD3 placeholders.

---

## 3. Mapping table

| **WMOv14 Chunk** | **Q3 BSP Lump** | **Notes** |
|------------------|-----------------|-----------|
| `MOVT` (vertices) | `vertices` | Direct copy. |
| `MOVI` + `MOPY` (indices + materials) | `faces` | Build face definitions. |
| `MOTX` (texture names) | `textures` | Convert BLP paths → shader names. |
| `MOMT` (materials) | `textures`/shader scripts | Map flags to shader keywords. |
| `MOGP` (group) | BSP submodel/leaf | Treat each group as a submodel. |
| `MOPV`/`MOPT`/`MOPR` (portals) | `visdata` | Approximate portal connectivity. |
| `MOLT` (lights) | `lightmaps` or `entities` | Bake or ignore. |
| `MODN`/`MODD` (doodads) | `entities` | Place as MD3 references. |
| `MFOG` | fog brushes/shaders | Optional. |
| `MOSB` | skybox shader | Optional. |

---

## 4. Export pipeline (pseudo‑code)

```csharp
// Load WMO root
var wmo = ParseWMO("input.wmo");

// Initialize BSP structure
var bsp = new BSPFile();

// 1. Textures
foreach (var tex in wmo.MOTX)
    bsp.Textures.Add(ConvertToShaderName(tex));

// 2. Geometry
foreach (var group in wmo.Groups)
{
    foreach (var v in group.MOVT)
        bsp.Vertices.Add(v);

    foreach (var face in BuildFaces(group.MOVI, group.MOPY))
        bsp.Faces.Add(face);
}

// 3. Portals/Vis
bsp.VisData = ConvertPortalsToVis(wmo.MOPV, wmo.MOPT, wmo.MOPR);

// 4. Lights (optional)
if (IncludeLights)
    bsp.Lightmaps = BakeLights(wmo.MOLT);

// 5. Entities (optional doodads)
bsp.Entities = ConvertDoodads(wmo.MODN, wmo.MODD);

// Write BSP
bsp.Save("output.bsp");
```

---

## 5. Testing strategy

1. Start with a simple WMO (e.g., a one‑room building).  
2. Export geometry only (vertices + faces).  
3. Load in GtkRadiant/ioquake3 → verify geometry appears.  
4. Add textures (MOTX → shaders).  
5. Add portals/vis.  
6. Add lights (optional).  
7. Add doodads/entities (optional).

---

## 6. Historical experiment

- **Dark mines test:** Export without `MOLT` lights → geometry loads dark in Q3 engine, matching *WoW Diary* anecdote.  
- **Enhanced test:** Add lights → compare to WoW client rendering.  

---

### TL;DR

- WMOv14 groups map neatly to Q3 BSP submodels.  
- MOVT/MOVI/MOPY → vertices/faces.  
- MOTX/MOMT → textures/shaders.  
- MOPV/MOPT/MOPR → visdata.  
- MOLT → lightmaps/entities.  
- MODN/MODD → entities.  

Start with geometry‑only export, then layer in textures, vis, and lights.  
