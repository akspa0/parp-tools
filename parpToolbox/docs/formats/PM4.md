# PM4 Format Documentation

## 2025-08-19 Rewrite Preface
This page is now a LEGACY STUB. Authoritative PM4 documentation has moved to the canonical specification:

- PM4 Specification (Canonical): [PM4-Spec.md](PM4-Spec.md)

The content below is retained for historical reference only and may contain deprecated or overconfident claims. Prefer the canonical spec above for all current guidance.

> Note: All sections below marked as [Deprecated] are preserved for historical context and are not authoritative. See `PM4-Spec.md` for current guidance.

### Errata & history

See Deprecated Claims in the canonical spec: [PM4-Spec.md#deprecated-or-disproven-claims](PM4-Spec.md#deprecated-or-disproven-claims)

Previous claims about “global multi-tile vertex pools,” “MPRR defining building boundaries,” and “fully decoded fields” are deprecated. The legacy sections below are kept for historical context; prefer the canonical spec above.

---

## [Deprecated] Complete Field Analysis (2025-08-05)

> Deprecated legacy content retained for history. Some claims contradict the canonical spec and should not be used. See `PM4-Spec.md`.

**BREAKTHROUGH**: Successfully analyzed **502 PM4 files** with **100% success rate**, decoding ALL unknown fields through comprehensive cross-file pattern analysis.

### Data Recovery Results
- **1,134,074 vertices** (MSVT) across 309 files
- **1,930,146 triangle indices** (MSVI) 
- **1,273,335 surface links** (MSLK) with ALL fields decoded
- **518,092 surfaces** (MSUR) with material patterns identified
- **178,588 object placements** (MPRL) with type classification

### Key Field Discoveries

#### MSLK Fields - FULLY DECODED:
- **`HasGeometry`**: 🔑 Critical flag identifying entries with renderable geometry
- **`Unknown4` → `Object Type ID`**: 573, 577, etc. = building type classification
- **`Unknown6` → `Universal Flag`**: Always 32768 (0x8000) = geometry enabled
- **`LinkIdTileX/Y`**: Tile coordinate encoding within LinkId
- **`ReferenceIndexHigh/Low`**: Confirmed packed 32-bit index encoding
- **`ParentIndex ↔ MPRL.Unknown4`**: Primary object placement linkage (458 confirmed matches)

#### MSUR Fields - MATERIAL SYSTEM IDENTIFIED:
- **Repeating triplets** like `"1116401852","17034","61628"` = Global surface/material library IDs
- **IndexCount 1-7**: Confirmed N-gon geometry system (not just triangles)
- **Global consistency**: Same material IDs across all tiles

#### MPRL Fields - PLACEMENT SYSTEM DECODED:
- **`Unknown4`**: Object type classification (573=type A, 577=type B, etc.)
- **`Unknown6`**: Always 32768 = placement enabled flag
- **Y-coordinates ~40.2**: Standard ground level placement
- **Spatial clustering**: Objects group by type in regions

### Updated Object Assembly Algorithm

Based on complete field analysis:

1. **Filter by `HasGeometry = True`** in MSLK (ignore container nodes)
2. **Group by `ParentIndex`** to collect related surfaces
3. **Link to placement** via `MSLK.ParentIndex ↔ MPRL.Unknown4`
4. **Extract geometry** using `MSUR.StartIndex` and `IndexCount` from MSVI
5. **Apply world positioning** from MPRL coordinates

---

## [Deprecated] Scene Graph Architecture (legacy)

> Deprecated legacy analysis retained for history. Prefer the canonical spec for current architecture and assembly guidance.

**CRITICAL DISCOVERY**: PM4 files are fundamentally structured as **scene graphs** with hierarchical spatial organization and nested coordinate systems.

### Scene Graph Structure

```
PM4 Scene Graph Hierarchy:
MPRL (Building Root Nodes) - 458 buildings
  ├─ Coordinate Transform Matrix
  ├─ MSLK (Child Sub-objects) - ~13 per building
  │   ├─ Local Transform
  │   ├─ MSUR (Fine Geometry) → n-gon faces at full ADT resolution
  │   └─ MSCN (Spatial Bounds) → 1/4096 scale spatial anchors
  └─ Export as Single Unified Object
```

### Nested Coordinate Systems

**All chunks exist within a single quadrant** of an XYZ coordinate system, but use different scaling and ground planes:

- **MSCN Coordinates**: 1/4096 scale within quadrant (coarse spatial index)
- **MSUR/MSVT/MSPI**: Full resolution (same scale as ADT terrain)
- **Different Ground Planes**: Each coordinate space uses different orientations

### Transform Requirements

1. **Scale Transforms**: Convert between 1/4096 MSCN scale and full-resolution geometry
2. **Coordinate System Transforms**: Convert between different ground plane orientations
3. **Spatial Anchoring**: Use MSCN data to anchor/contain geometry from other chunks
4. **Unified Export Space**: Transform all geometry into single coordinate system

### Why Scene Graph Approach Works

- **Leverages Built-in Optimizations**: PM4 is optimized for fast spatial queries
- **Follows Hierarchical Structure**: MPRL → MSLK → geometry traversal
- **Proper Transform Order**: Parent→child coordinate transforms
- **Spatial Partitioning**: SurfaceKey as spatial index for fast queries
- **LoD Integration**: MSCN coarse bounds → MSUR fine detail

## [Deprecated] Verified Object Extraction Method

> Historical algorithm retained for context. Not authoritative.

### Working Algorithm (Source: poc_exporter.cs)
**Spatial Clustering Approach** - Combines hierarchical grouping with spatial proximity

1. **Root Node Identification**
   - Find MSLK entries where `Unknown_0x04 == entry_index` (self-referencing)
   - These are verified building root nodes

2. **Structural Grouping** 
   - Group MSLK entries by `Unknown_0x04` value matching root nodes
   - `Unknown_0x04` = Building group identifier

3. **Bounds Calculation**
   - Calculate bounding box from MSPV vertices
   - Data flow: MSLK → MSPI → MSPV

4. **Spatial Clustering**
   - Find MSUR surfaces within expanded bounds (50.0f tolerance)
   - Compensates for incomplete hierarchical relationships

5. **Hybrid Assembly**
   - Combine MSPV structural elements + nearby MSUR render surfaces
   - Creates complete building objects

### Verified Data Relationships
```
MPRL.Unknown4 = MSLK.ParentIndex (458 confirmed matches)
MSLK.Unknown_0x04 = Building group identifier
MSLK.MspiFirstIndex = -1 → Container/grouping nodes (no geometry)
MSLK.MspiFirstIndex ≥ 0 → Geometry nodes
```

### Coordinate Transformations (Verified)
```csharp
// MSPV structural vertices (direct)
Vector3 FromMspvVertex(vertex) => (vertex.X, vertex.Y, vertex.Z)

// MSVT render vertices (Y-X swap)
Vector3 FromMsvtVertex(vertex) => (vertex.Position.Y, vertex.Position.X, vertex.Position.Z)
```

## Overview

PM4 files are complex, phased model descriptor files that serve as server-side supplementary data for ADT (terrain) files. They represent the original approach to World of Warcraft's navigation mesh system before the 2016 split that created the simpler PD4 format.

**Key Characteristics:**
- **One file per root ADT**: Each PM4 corresponds to a terrain tile
- **Server-side only**: Not shipped to clients, used for server navigation/collision
- **Phased model descriptors**: Complex multi-object approach with intricate relationships
- **Pre-2016 format**: Legacy system that was later split into individual PD4 files
- **Multi-object complexity**: Single PM4 can contain data for multiple objects/structures
- **Interior MODF System**: PM4 acts as an "interior MODF" placing building components inside structures

## [Deprecated] Object Assembly Methodology (2025-07-27)

**Deprecated legacy claim:** Prior notes asserted a global-mesh, multi-tile requirement for geometry assembly. Current guidance: process strictly per tile and treat cross-tile references as non-rendering metadata unless proven otherwise (see `PM4-Spec.md`).

### **Global Mesh Architecture Confirmed:**

**Mathematical Validation:**
- **58.4% of triangles** reference vertices from adjacent tiles (30,677 out of 52,506)
- **63,297 cross-tile vertex indices** in perfect sequential range: 63,298-126,594
- **Zero gap** between local (0-63,297) and cross-tile vertex ranges
- **Complete architectural assembly requires processing entire PM4 directory**

**Surface Encoding System:**
- **GroupKey determines data interpretation**: spatial vs encoded vs mixed
- **GroupKey 3** (1,968 surfaces): **Spatial** - normal coordinates, local tile geometry
- **GroupKey 18** (8,988 surfaces): **Mixed** - boundary objects spanning tile edges
- **GroupKey 19** (30,468 surfaces): **Encoded** - cross-tile/inter-object references (74% of surfaces)

**Cross-Tile Linkage Data:**
- **BoundsMaxZ field** in encoded groups contains hex-encoded tile/object references
- **Example**: `1127049344` = `0x432D6880` = linkage data, not spatial coordinates
- **16-bit pair encoding**: High/low pairs encode tile+object IDs
- **95.5% consistency** in GroupKey 19 encoding - highly systematic linkage system

**PREVIOUS FINDINGS STILL VALID:**
- **MSUR.SurfaceGroupKey** hierarchy: 19 ➜ object-level, 20-23 ➜ sub-objects, 24 ➜ surface-level
- **IndexCount** grouping insufficient for complete objects (missing cross-tile data)
- **Composite key approach** still needed: `(ParentIndex, SurfaceGroupKey, IndexCount)` + **cross-tile resolution**

### Confirmed Chunk Relationships:
- **MPRL.Unknown4 = MSLK.ParentIndex** (458 confirmed matches) - links placements to geometry
- **MSLK entries with MspiFirstIndex = -1** are container/grouping nodes (no geometry)
- **MPRR.Value1 = 65535** are property separators (15,427 sentinel values out of 81,936 total)
- **MPRL.Unknown6 = 32768** consistently (likely type flag)
- **MSCN spatial alignment**: Collision vertices often align within 0.1 units of MSVT polygon centers
- **MSUR.SurfaceGroupKey patterns**: Value 18 = geometry surfaces, Value 3 = collision/special surfaces
- **Cross-chunk indexing**: Multiple MSLK entries can reference the same MPRL placement (hierarchical assembly)

### Object Assembly Flow:
1. **MPRL** defines object placements (positions + type IDs in Unknown4)
2. **MSLK** links placements to geometry via ParentIndex → MPRL.Unknown4
3. **MPRR** provides segmented properties between sentinel markers
4. **MSUR** defines surface geometry. `IndexCount` is diagnostic only; not an object identifier.

### Implementation Notes:
- Group geometry by **MSUR.IndexCount** to get complete building objects
- ParentIndex/ReferenceIndex grouping produces fragments, not complete objects
- Apply X-axis inversion (`-vertex.X`) for correct coordinate system orientation

## glTF 2.0 Export Strategy

Now that full per-object assembly is possible, the preferred interchange container is **glTF 2.0**.  
Using a glTF scene enables:

* Single file (or binary `.glb`) containing **all objects, groups, sub-groups, and materials**
* Explicit node hierarchy preserving `SurfaceGroupKey` → `IndexCount` relationships
* Efficient transmission (binary, Draco compression, etc.)
* Broad tooling support in game engines and viewers compared to Wavefront OBJ

### Proposed Mapping
| PM4 concept | glTF element |
|-------------|-------------|
| Assembled object (`IndexCount`) | `Node` + `Mesh` |
| Sub-objects / sub-surfaces (`SurfaceGroupKey` ≥ 20) | Child `Node`s |
| Vertex positions | `POSITION` accessor (float32) |
| Triangles | `indices` accessor (uint32) |
| Normals (Nx,Ny,Nz) | `NORMAL` accessor |
| MPRL placement | Node `translation` (world-space offset) |
| Collision flags / special bits | `extras` dictionary |

In the toolchain this will be exposed via **`--gltf`** and **`--glb`** CLI flags.  
Implementation will leverage the open-source `SharpGLTF` library for quick authoring, with a custom visitor that walks the assembled objects and emits node hierarchies.

Implementation work is tracked in the project plan (see *Evaluate/integrate glTF 2.0 exporter*).

---

## Format Structure

PM4 files use the standard WoW chunk-based format with FourCC identifiers. All chunk headers are stored in little-endian byte order.

## Chunk Specifications

### MVER - Version
```c
struct MVER {
    uint32_t version;  // Version identifier
};
```

### MSHD - Header
```c
struct MSHD {
    uint32_t field_0x00;
    uint32_t field_0x04;
    uint32_t field_0x08;
    uint32_t reserved[5];  // Placeholder fields
};
```

### MSPV - Primary Vertices
```c
struct MSPV {
    Vector3 vertices[];  // Navigation vertices (world-space coordinates)
};
```
**Implementation Notes:**
- Stride: 12 bytes (XYZ) or 24 bytes (XYZ + 3 unknown floats)
- Auto-detects stride based on chunk size
- Skips unknown floats when stride is 24 bytes

### MSPI - Primary Indices
```c
struct MSPI {
    uint32_t indices[];  // Triangle indices into MSPV array
};
```
**Implementation Notes:**
- Groups of 3 indices form triangles
- May be 16-bit or 32-bit indices (auto-detected)

### MSCN - Collision/Exterior Vertices
```c
struct MSCN {
    Vector3 collision_vertices[];  // Collision/exterior vertex positions
};
```
**Implementation Notes:**
- **Collision mesh vertices**: Simplified geometry for physics/pathfinding
- **Spatial alignment**: Many vertices align closely with MSVT geometry centers
- **Independent indexing**: Not directly indexed by other chunks, likely referenced by implicit position
- **Pathfinding data**: Used for navigation mesh and collision detection
- **Stride**: 12 bytes per vertex (XYZ float coordinates)
- **Relationship to MSVT**: Average distance to closest MSVT vertex typically < 0.1 units for aligned vertices

### MSLK - Link Data
```c
struct MSLK {
    uint8_t unknown_0x00;        // Flags
    uint8_t unknown_0x01;
    uint16_t unknown_0x02;       // Padding
    uint32_t parent_index;       // High-word parent/container identifier (unknown_0x04)
    int32_t mspi_first_index;    // First index in MSPI (24-bit signed, -1 if none)
    uint8_t mspi_index_count;    // Number of indices in MSPI
    uint16_t link_id_padding;    // Always 0xFFFF
    uint8_t link_id_YY;          // PM4 Tile YY position
    uint8_t link_id_XX;          // PM4 Tile XX position
    // uint32_t link_id_raw;        // Composite key (unknown_0x0c)
    uint16_t reference_index;    // Authoritative group key (unknown_0x10)
    uint16_t unknown_0x12;       // Always 0x8000
};
```
**Implementation Aliases:**
- `parent_index` = `unknown_0x04` (high-word parent identifier)
- `reference_index` = `unknown_0x10` (authoritative group key)
- `link_sub_key` = `link_id_raw & 0xFFFF` (low 16 bits)
- `has_geometry` = `mspi_first_index >= 0 && mspi_index_count > 0`

### MSVT - Secondary Vertices
```c
struct MSVT {
    Vector3 vertices[];  // Render vertices (stored as Y,X,Z order)
};
```
**Implementation Notes:**
- File format stores as (Y, X, Z) order
- Implementation reorders to (X, Y, Z)
- Stride: 12 bytes (XYZ) or 24 bytes (XYZ + 3 unknown floats)
- Auto-detects stride based on chunk size

### MSVI - Secondary Indices
```c
struct MSVI {
    uint32_t indices[];  // Indices into MSVT array
};
```
**Implementation Notes:**
- May represent quads or n-gons rather than triangles
- Structure determined by MSUR entries

### MSUR - Surface Definitions
```c
struct MSUR {
    uint8_t surface_group_key;   // Group key / flags (flags_or_unknown_0x00)
    uint8_t index_count;         // Diagnostic only; not an object identifier (see PM4-Spec.md)
    uint8_t surface_attr_mask;   // Attribute mask (unknown_0x02)
    uint8_t padding;             // Always 0
    float nx, ny, nz;            // Surface normal
    float height;                // Plane D or surface height
    uint32_t msvi_first_index;   // First index in MSVI for this surface
    uint32_t mdos_index;         // MDOS reference
    uint32_t surface_key;        // 32-bit composite key (packed_params) - this is wrong, it is two 16-bit keys.
};
```

**Deprecated legacy claim:** The `index_count` field (offset 0x01) was previously treated as a primary object identifier. Current guidance: treat `index_count` as diagnostic only; see `PM4-Spec.md`.

**Implementation Aliases:**
- `is_m2_bucket` = `surface_group_key == 0x00`
- `is_liquid_candidate` = `(surface_attr_mask & 0x80) != 0`
- `surface_key_high16` = `surface_key >> 16`
- `surface_key_low16` = `surface_key & 0xFFFF`

**[Deprecated] Object Assembly Logic:** Legacy grouping by `MSUR.IndexCount` is not authoritative. See `PM4_Assembly_Relationships.md` for current assembly guidance.

### MPRL - Property List
```c
struct MPRL {
    uint16_t unknown_0;      // Always 0
    int16_t unknown_2;       // Always -1
    uint16_t unknown_4;
    uint16_t unknown_6;
    Vector3 position;        // 3D position
    int16_t unknown_14;
    uint16_t unknown_16;
};
```
**Implementation Notes:**
- 24 bytes per entry
- Contains placement/prop positions
- Field meanings largely unknown

### MPRR - Property References
```c
struct MPRR {
    uint16_t unknown_0;
    uint16_t unknown_2;
};
```
**Implementation Notes:**
- References or links to MPRL data
- Structure and usage unclear

### MPRL - Placement List
```c
struct MPRL {
    uint16_t unknown_0;      // Always 0 – padding / flags
    int16_t  sentinel_0x02;  // Always -1 – separates logical groups
    uint16_t parent_index;   // **Authoritative placement index** (links to MSLK.ParentIndex)
    uint16_t type_flag;      // Always 32768 (0x8000) – object type/category flag
    Vector3  position;       // World-space position (Z -> -X, X -> Z, Y stays Y)
    int16_t  unknown_14;     // Observed 0 in all samples
    uint16_t unknown_16;     // Observed 0x8000 in all samples
};
```
**Confirmed Behaviour (2025-07-19):**
* `parent_index` (offset 0x04) **directly equals** `MSLK.ParentIndex` – this forms the core link between placement nodes and geometry.
* `type_flag` (0x06) is **always 0x8000** for real placements. Container MSLK nodes with `MspiFirstIndex = -1` map to MPRL entries where this flag is also 0x8000.
* `position` is used to offset assembled geometry; convert to engine-right-handed coordinates via `(-Z, Y, X)`.
* The pair `(unknown_0, sentinel_0x02)` is constant (0,-1) and acts as a record sentinel.

---

### MDBH - Destructible Building Header
```c
struct MDBH {
    uint32_t count;
    struct {
        uint32_t index;
        char filename[3][]; // Variable length filenames
    } buildings[count];
};
```
**Sub-chunks:**
- **MDBF**: Destructible building filename (null-terminated string)
- **MDBI**: Destructible building index (uint32_t)

### MDOS - Destructible Object System
```c
struct MDOS {
    uint32_t field_0x00;
    uint32_t field_0x04;
};
```
**Implementation Notes:**
- System data for destructible objects
- Structure and usage unclear

### MDSF - Destructible System Flags
```c
struct MDSF {
    // Structure details to be documented
};
```
**Implementation Notes:**
- Flags and configuration for destructible systems
- Structure not yet implemented

## Relationship to Other Formats

### vs PD4
- **PM4**: Complex, multi-object, phased descriptors
- **PD4**: Simple, single-object, full precision
- **Evolution**: PM4 → PD4 split simplified object handling
- **Compatibility**: Share many chunk types but different usage patterns

### vs WMO
- **PM4**: Server-side navigation data with full precision
- **WMO**: Client-side compressed geometry for rendering
- **Correlation**: No direct mathematical correlation found
- **Abstraction**: Different levels - PM4 for navigation, WMO for graphics

### vs ADT
- **PM4**: Supplementary object data for ADT terrain tiles
- **ADT**: Primary terrain geometry and texturing
- **Relationship**: One PM4 per ADT root file
- **Purpose**: PM4 adds navigation detail to ADT terrain


