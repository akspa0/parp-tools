# TXID Chunk (Texture IDs)

## Overview
The TXID chunk contains FileDataIDs for textures used by the M2 model. It was introduced in Battle for Azeroth (8.0.1.26629) and replaces the texture filenames that were previously stored within the Textures section of the MD21 chunk.

## Structure
```cpp
struct TXID_Entry {
  uint32_t fileDataID;  // FileDataID referring to a texture
};

struct TXID_Chunk {
  TXID_Entry textureID[];  // Array of texture FileDataIDs
};
```

## Fields
- **textureID**: An array of FileDataIDs that reference textures used by the model. Each entry corresponds to a texture defined in the Textures section of the MD21 chunk.

## Dependencies
- Requires the MD21 chunk, specifically the Textures section
- Order of the texture IDs must match the order of textures defined in the MD21 chunk

## Usage
The TXID chunk is used to:
- Look up texture files by their FileDataID instead of by filename
- Provide a direct reference to textures in the game's file system
- Support the client's file lookup system for textures

## Legacy Support
- Not present in pre-BfA M2 files
- In older versions, texture filenames were stored directly in the Textures section of the model

## Implementation Notes
- The number of entries should match the number of textures defined in the MD21 chunk
- Each FileDataID points to a BLP texture file in the game archives
- This chunk simplified the texture lookup process and eliminated the need for path manipulation
- The change to FileDataIDs supports Blizzard's internal content delivery system

## Version History
- Introduced in Battle for Azeroth (8.0.1.26629)
- Part of Blizzard's ongoing efforts to transition to FileDataID-based resource references
- Replaced the string-based texture filename system 