# PM4 Format Documentation

## Overview

PM4 files are complex, phased model descriptor files that serve as server-side supplementary data for ADT (terrain) files. They represent the original approach to World of Warcraft's navigation mesh system before the 2016 split that created the simpler PD4 format.

**Key Characteristics:**
- **One file per root ADT**: Each PM4 corresponds to a terrain tile
- **Server-side only**: Not shipped to clients, used for server navigation/collision
- **Phased model descriptors**: Complex multi-object approach with intricate relationships
- **Pre-2016 format**: Legacy system that was later split into individual PD4 files
- **Multi-object complexity**: Single PM4 can contain data for multiple objects/structures
- **Interior MODF System**: PM4 acts as an "interior MODF" placing building components inside structures

## Object Assembly Methodology (CRITICAL)

**Individual objects are identified by the MSUR IndexCount (0x01 field)**, not by ParentIndex or ReferenceIndex grouping.

### Confirmed Chunk Relationships:
- **MPRL.Unknown4 = MSLK.ParentIndex** (458 confirmed matches) - links placements to geometry
- **MSLK entries with MspiFirstIndex = -1** are container/grouping nodes (no geometry)
- **MPRR.Value1 = 65535** are property separators (15,427 sentinel values)
- **MPRL.Unknown6 = 32768** consistently (likely type flag)

### Object Assembly Flow:
1. **MPRL** defines object placements (positions + type IDs in Unknown4)
2. **MSLK** links placements to geometry via ParentIndex → MPRL.Unknown4
3. **MPRR** provides segmented properties between sentinel markers
4. **MSUR** defines surface geometry with **IndexCount as the object identifier**

### Implementation Notes:
- Group geometry by **MSUR.IndexCount** to get complete building objects
- ParentIndex/ReferenceIndex grouping produces fragments, not complete objects
- Apply X-axis inversion (`-vertex.X`) for correct coordinate system orientation

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

### MSCN - Normals
```c
struct MSCN {
    Vector3 normals[];  // Surface normals (independent of MSPV)
};
```
**Implementation Notes:**
- Not directly related to MSPV geometry
- May have different count than vertices

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
    uint8_t index_count;         // **OBJECT IDENTIFIER** - Critical for PM4 object assembly
    uint8_t surface_attr_mask;   // Attribute mask (unknown_0x02)
    uint8_t padding;             // Always 0
    float nx, ny, nz;            // Surface normal
    float height;                // Plane D or surface height
    uint32_t msvi_first_index;   // First index in MSVI for this surface
    uint32_t mdos_index;         // MDOS reference
    uint32_t surface_key;        // 32-bit composite key (packed_params) - this is wrong, it is two 16-bit keys.
};
```

**CRITICAL DISCOVERY**: The `index_count` field (offset 0x01) is the **primary object identifier** for PM4 object assembly. All surfaces with the same `index_count` value belong to the same building object.

**Implementation Aliases:**
- `is_m2_bucket` = `surface_group_key == 0x00`
- `is_liquid_candidate` = `(surface_attr_mask & 0x80) != 0`
- `surface_key_high16` = `surface_key >> 16`
- `surface_key_low16` = `surface_key & 0xFFFF`

**Object Assembly Logic:**
1. Group all MSUR entries by their `index_count` value
2. For each group, collect all associated MSLK links via surface index matching
3. Extract geometry from MSLK → MSPI triangles
4. Result: Complete building objects instead of fragments

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


