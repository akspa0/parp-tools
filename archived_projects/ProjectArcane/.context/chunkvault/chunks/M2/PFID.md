# PFID: Physics File ID

## Identification
- **Chunk ID**: PFID
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The PFID chunk contains a single FileDataID that references an external physics file associated with the M2 model. Prior to Legion, physics files were referenced by filename with the pattern `${basename}.phys`. This chunk was introduced in Legion (expansion level 7) and provides a direct FileDataID reference to the physics file.

## Structure
```cpp
struct PFIDChunk {
    uint32_t phys_file_id;  // FileDataID for the physics file
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| phys_file_id | uint32_t | FileDataID for the physics file associated with the model |

## Dependencies
- Depends on the MD21 chunk and specifically on bones with the `kinematic_bone` flag (0x400) set.
- The physics file referenced contains collision and physics simulation data.

## Implementation Notes
1. Physics files are only loaded if the model's header has the `flag_load_phys_data` (0x20) flag set.
2. The physics file contains data for physics simulation, including collision shapes and physical properties.
3. In Shadowlands and later, the physics data can also be embedded directly in the M2 file via the PFDC chunk.
4. The physics system primarily affects cloth, capes, chains, and other items that need realistic physical movement.

## Usage Context
Physics files (.phys) are used to define the physical properties of a model for:
- Cloth simulation on character capes, tabards, and robes
- Chain or rope-like objects that need to swing naturally
- Hair and other appendages that require physical movement
- Collision detection for movable parts of the model

The PFID chunk provides a direct FileDataID reference to the physics file, eliminating the need to construct filenames based on patterns. This allows the client to efficiently load the necessary physics data only when needed for models that have physical simulation. 