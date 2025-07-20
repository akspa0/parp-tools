# PD4 Format Documentation

## Overview

PD4 files are individual object data descriptors introduced as part of World of Warcraft's server-side navigation mesh system. They represent a evolution from the earlier PM4 format, providing full mathematical precision for 3D geometry of individual objects on the map.

**Key Characteristics:**
- **One file per root WMO**: Each PD4 corresponds to a specific WMO object
- **Server-side only**: Not shipped to clients, used for server navigation/collision
- **Full precision geometry**: Contains complete mathematical representation of object surfaces
- **Individual object focus**: Unlike PM4's complex multi-object approach, PD4 isolates single objects
- **Post-2016 format**: Part of the PM4→PD4 split that simplified object handling

## Format Structure

PD4 files use the standard WoW chunk-based format with FourCC identifiers. All chunk headers are stored in little-endian byte order.

### Version Information
- **Current Version**: 48 (seen in 6.0.1.18297, 6.0.1.18443)
- **Format**: Standard WoW chunk format with 4-byte FourCC + 4-byte size

## Chunk Specifications

### MVER - Version
```c
struct MVER {
    uint32_t version;  // 48 for current format
};
```

### MCRC - Checksum
```c
struct MCRC {
    uint32_t checksum;  // Always 0 in version 48
};
```

### MSHD - Header
```c
struct MSHD {
    uint32_t field_0x00;
    uint32_t field_0x04;
    uint32_t field_0x08;
    uint32_t reserved[5];  // Always 0 in version 48
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
- **Spatial alignment**: Many vertices align closely with MSPV geometry centers
- **Independent indexing**: Not directly indexed by other chunks, likely referenced by implicit position
- **Pathfinding data**: Used for navigation mesh and collision detection
- **Stride**: 12 bytes per vertex (XYZ float coordinates)
- **Inheritance from PM4**: Same structure and purpose as PM4's MSCN chunk

### MSLK - Link Data
```c
struct MSLK {
    uint8_t unknown_0x00;        // Flags
    uint8_t unknown_0x01;
    uint16_t unknown_0x02;       // Padding (always 0 in version 48)
    uint32_t parent_index;       // High-word parent/container identifier (unknown_0x04)
    int32_t mspi_first_index;    // First index in MSPI (24-bit signed, -1 if none)
    uint8_t mspi_index_count;    // Number of indices in MSPI
    uint16_t link_id_padding;    // Always 0xFFFF
    uint8_t link_id_YY;          // PD4 Tile YY position
    uint8_t link_id_XX;          // PD4 Tile XX position
    uint16_t reference_index;    // Authoritative group key (unknown_0x10)
    uint16_t unknown_0x12;       // Always 0x8000 in version 48
};
```
**Implementation Aliases:**
- `parent_index` = `unknown_0x04` (high-word parent identifier)
- `reference_index` = `unknown_0x10` (authoritative group key)
- `tile_coordinate` = `(link_id_YY << 8) | link_id_XX` (composite tile key)
- `link_sub_key` = `tile_coordinate` (legacy compatibility)
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

**Coordinate Transformation:**
```c
worldPos.y = 17066.666f - position.y;
worldPos.x = 17066.666f - position.x;
worldPos.z = position.z / 36.0f;  // Convert internal inches to yards
```

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
    uint8_t index_count;         // Number of indices in MSVI for this surface
    uint8_t surface_attr_mask;   // Attribute mask (unknown_0x02)
    uint8_t padding;             // Always 0 in version 48
    float nx, ny, nz;            // Surface normal
    float height;                // Plane D or surface height
    uint32_t msvi_first_index;   // First index in MSVI for this surface
    uint32_t mdos_index;         // MDOS reference (field_0x18)
    uint32_t surface_key;        // 32-bit composite key (packed_params)
};
```
**Implementation Aliases:**
- `is_m2_bucket` = `surface_group_key == 0x00`
- `is_liquid_candidate` = `(surface_attr_mask & 0x80) != 0`
- `surface_key_high16` = `surface_key >> 16`
- `surface_key_low16` = `surface_key & 0xFFFF`

## Relationship to Other Formats

### vs PM4

**Format Evolution**: PD4 represents a simplification of PM4's complex multi-object system:

- **PM4**: Complex hierarchical assembly with MPRL placements, MSLK links, and MPRR properties
- **PD4**: Simplified single-object focus with direct geometry representation
- **Shared chunks**: Both formats use MSPV, MSPI, MSCN, and MSLK with similar structures
- **MSCN consistency**: Collision vertex analysis shows same spatial alignment patterns in both formats
- **Object assembly**: PD4 eliminates PM4's complex cross-chunk relationships for simpler processing
- **PD4**: Simple, single-object, full precision
- **PM4**: Complex, multi-object, phased descriptors
- **Evolution**: PM4 → PD4 split simplified object handling
- **Compatibility**: Share many chunk types but different usage patterns

### vs WMO
- **PD4**: Server-side navigation data with full precision
- **WMO**: Client-side compressed geometry for rendering
- **Correlation**: No direct mathematical correlation found
- **Abstraction**: Different levels - PD4 for navigation, WMO for graphics

### vs ADT
- **PD4**: Individual object data for server-side processing
- **ADT**: Primary terrain geometry and texturing
- **Relationship**: PD4 provides object-level detail for ADT terrain


