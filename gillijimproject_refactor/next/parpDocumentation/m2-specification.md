# M2 File Format Specification

This document provides a comprehensive specification of the M2 (Model) file format, reverse-engineered from the parpToolbox codebase and Warcraft.NET library. M2 files represent 3D models (e.g., trees, props, characters) in World of Warcraft, containing vertices, skins (render batches), animations, bones, and physics. M2 is a binary format with a header followed by named sections (e.g., MD21 for modern data, MD20 for legacy). Versions via MD20/MD21; supports skins for LOD/multi-texture rendering.

Spec derived from M2ModelHelper.cs (loads via Warcraft.NET.Files.M2), ADT MDDF/MMDX (placements/names), and WMO doodads. Why: Efficient model rendering (skinned verts, anim tracks); how: Section scan, vertex/skin parsing. Tools like Pm4BatchObjExporter reference M2 via ADT, but no direct M2 exporter; Warcraft.NET handles loading (MD21.Vertices, BoundingTriangles).

## Overall Structure
M2 starts with:
- MD20/MD21 (header/model data, ~100 bytes + arrays)
- Vertices (float XYZ, variable)
- Skins (render batches: indices, textures, matrices)
- Animations (tracks: pos/rot/scale keys)
- Bones (hierarchy, binds)
- ... (optional: physics, attachments, physics)

Size: 10KB-1MB. Validation: MD21 present; vertex/skin counts match. Legacy MD20 for pre-WotLK.

### Section Header (Named Chunks)
```c
struct SectionHeader {
    char name[4];       // e.g., "MD21"
    uint32_t size;      // Payload
    uint32_t version;   // Section version
};
```

## Key Sections

### MD21: Model Data (variable, core header)
**Purpose**: M2 header: bounds, anim info, counts (verts, skins, anims, bones).

**Plain English**: Defines model scope (nVertices, nSkins); bounding box for culling; anim lookup table. Code: `M2File.Load` reads counts, then arrays (Vertices as Vector3[]).

**C Struct**:
```c
struct MD21 {
    uint32_t nVertices;         // Vertex count
    uint32_t nSkins;            // Skin count
    uint32_t nAnimations;       // Animation count
    uint32_t nBones;            // Bone count
    float bbox_min_xyz[3];      // Bounding box min
    float bbox_max_xyz[3];      // Bounding box max
    uint32_t anim_lookup[10];   // Animation indices (padded)
    // ... (more: global sequences, colors, textures, transparency, etc.)
};  // ~100 bytes + offsets
```

**Usage**: `Warcraft.NET.Files.M2.M2File.Vertices`; `M2ModelHelper.LoadMeshFromFile` extracts verts/tris.

### Vertices: Positions (12 bytes/vertex, nVertices)
**Purpose**: Array of float XYZ positions (world/local space).

**Plain English**: Raw vertex coords; transformed by bones/skins. Code: `br.ReadVector3()` loop.

**C Struct**:
```c
struct Vertex {
    float x, y, z;  // Position
};  // 12 bytes/vertex
```

**Usage**: `MD21.Vertices`; used in M2Mesh for OBJ export (apply pos/rot/scale).

### Skins: Render Batches (variable per skin)
**Purpose**: Per-skin data: indices (tris), bone weights, textures, matrices.

**Plain English**: Groups verts into renderable parts (e.g., tree trunk); indices into Vertices. Code: Parse skin sections (indices, verts, bones).

**C Struct** (simplified):
```c
struct Skin {
    uint32_t nIndices;      // Triangle indices count
    uint32_t nVerts;        // Verts in this skin
    uint16_t indices[];     // Tris (3 per face)
    uint16_t bone_weights[]; // Per-vert bone influences
    // ... (matrices, textures)
};  // Variable
```

**Usage**: `M2File.Skins`; `BoundingTriangles` for collision/export.

### Animations/Bones: Skeleton (variable)
**Purpose**: Bone hierarchy and keyframe tracks (pos/rot/scale).

**Plain English**: Defines model pose/anim; bones as tree with quaternions. Code: `AnimationBlock` parses keys.

**C Struct** (Bone):
```c
struct Bone {
    uint32_t parent_id;     // Parent bone index
    float pos_xyz[3];       // Local position
    // ... (rot quaternion, scale)
};  // 28+ bytes/bone
```

**Usage**: `M2File.Bones`; applied in exports for posed meshes.

## Data Arrangement and Usage in Priority Tools
M2: MD21 header→vertices→skins (render tris)→bones/anims. Placed in ADT via MDDF (name_id→MMDX, pos/rot/scale); in WMO doodads.

- **M2ModelHelper**: `LoadMeshFromFile(path, pos, rot, scale)` → extracts M2Mesh (verts/tris via MD21/BoundingTriangles); transforms for OBJ.
- **Pm4BatchProcessor**: References M2 via ADT MDDF; matches to PM4 MSUR (M2 bucket GroupKey=0).
- **ADTPreFabTool**: Loads M2 names from MMDX for prefab scans.
- **parpDataHarvester**: Collects M2 refs in batch (CSV: names, placements).

Interrelation: ADT MDDF→M2 (doodads); WMO MODD→M2. Validate with Warcraft.NET `M2File.Load` (counts match MD21).

For ADT/WMO/PM4 integration, see respective specs.