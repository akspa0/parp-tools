# MDX Format Documentation

## Related Documentation
- [M2 Format](M2_index.md) - Related WoW model format
- [Common Types](common/types.md) - Shared data structures
- [Format Relationships](relationships.md) - Dependencies and connections

## Implementation Status
✅ **Complete** - Core parsing system and all blocks implemented

### Core Components
- ✅ `MdxFile` - Main MDX file parser
- ✅ `MdxBlock` - Base block implementation
- ✅ `MdxTrack` - Animation track system
- ✅ `MdxTrackFactory` - Track creation system
- ✅ `MdxKeyTrack` - Keyframe animation system

### Features
- ✅ Block-based parsing
- ✅ Animation system
- ✅ Asset reference tracking
- ✅ Validation reporting
- ⏳ M2 conversion (Planned)
- ⏳ Editing tools (Planned)

## Implemented Blocks

### Core Blocks
| Block | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| VERS | ✅ | Version information | [chunks/MDX/M001_VERS.md](chunks/MDX/M001_VERS.md) |
| ModelInfo | ✅ | Model information | [chunks/MDX/ModelInfo.md](chunks/MDX/ModelInfo.md) |
| Sequence | ✅ | Animation sequence | [chunks/MDX/Sequence.md](chunks/MDX/Sequence.md) |
| GlobalSequence | ✅ | Global timing | [chunks/MDX/GlobalSequence.md](chunks/MDX/GlobalSequence.md) |
| Texture | ✅ | Texture definitions | [chunks/MDX/Texture.md](chunks/MDX/Texture.md) |
| Material | ✅ | Material properties | [chunks/MDX/Material.md](chunks/MDX/Material.md) |
| TextureAnimation | ✅ | Texture animation | [chunks/MDX/TextureAnimation.md](chunks/MDX/TextureAnimation.md) |

### Geometry Blocks
| Block | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| Geoset | ✅ | Geometry set | [chunks/MDX/Geoset.md](chunks/MDX/Geoset.md) |
| GeosetAnimation | ✅ | Geometry animation | [chunks/MDX/GeosetAnimation.md](chunks/MDX/GeosetAnimation.md) |
| Bone | ✅ | Skeletal bone | [chunks/MDX/Bone.md](chunks/MDX/Bone.md) |
| Light | ✅ | Light source | [chunks/MDX/Light.md](chunks/MDX/Light.md) |
| Helper | ✅ | Helper object | [chunks/MDX/Helper.md](chunks/MDX/Helper.md) |
| Attachment | ✅ | Attachment point | [chunks/MDX/Attachment.md](chunks/MDX/Attachment.md) |
| Pivot | ✅ | Pivot point | [chunks/MDX/Pivot.md](chunks/MDX/Pivot.md) |

### Effect Blocks
| Block | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| ParticleEmitter | ✅ | Particle system | [chunks/MDX/ParticleEmitter.md](chunks/MDX/ParticleEmitter.md) |
| ParticleEmitter2 | ✅ | Enhanced particles | [chunks/MDX/ParticleEmitter2.md](chunks/MDX/ParticleEmitter2.md) |
| RibbonEmitter | ✅ | Ribbon system | [chunks/MDX/RibbonEmitter.md](chunks/MDX/RibbonEmitter.md) |
| EventObject | ✅ | Event timing | [chunks/MDX/EventObject.md](chunks/MDX/EventObject.md) |
| Camera | ✅ | Camera definition | [chunks/MDX/Camera.md](chunks/MDX/Camera.md) |

### Collision Blocks
| Block | Status | Description | Documentation |
|-------|--------|-------------|---------------|
| CollisionShape | ✅ | Collision geometry | [chunks/MDX/CollisionShape.md](chunks/MDX/CollisionShape.md) |
| Collision | ✅ | Collision data | [chunks/MDX/Collision.md](chunks/MDX/Collision.md) |

Total Progress: 23/24 blocks implemented (96%)

## Implementation Notes
- Block-based format similar to M2
- Animation system uses keyframe tracks
- Particle systems compatible with M2
- Conversion to M2 possible but complex

## File Structure
```
MDLX                  - File magic
<blocks>              - Series of blocks
  <block>             - Individual block
    <header>          - Block header
    <data>           - Block data
```

## Next Steps
1. Add M2 conversion support
2. Create editing tools
3. Add format validation

## References
- [MDX Format Specification](../docs/MDX.md)
- [MDX Block Reference](../docs/MDX_Blocks.md)

## Overview
The MDX format is a binary 3D model format used by Blizzard Entertainment for Warcraft III and early World of Warcraft. It is a proprietary format derived from the earlier MDL format.

## Chunks
The MDX format consists of several chunks, each identified by a four-character code (FourCC). Each chunk contains specific data related to the model, such as vertices, textures, animations, etc.

| Chunk ID | Description | Status | Priority |
|----------|-------------|--------|----------|
| MDLX | File header | Documented | High |
| VERS | Version information | Documented | High |
| MODL | Model information | Documented | High |
| SEQS | Animation sequences | Documented | High |
| GLBS | Global sequences | Documented | High |
| MTLS | Materials | Documented | High |
| TEXS | Textures | Documented | High |
| GEOS | Geosets (geometry) | Documented | High |
| GEOA | Geoset animations | Documented | High |
| BONE | Bones | Documented | High |
| LITE | Lights | Documented | Medium |
| HELP | Helper objects | Documented | Medium |
| ATCH | Attachments | Documented | Medium |
| PIVT | Pivots | Documented | Low |
| PREM | Particle emitters | Documented | Medium |
| PRE2 | Particle emitters v2 | Documented | Medium |
| RIBB | Ribbon emitters | Documented | Medium |
| EVTS | Events | Documented | Low |
| CLID | Collision shapes | Documented | Low |
| CORN | Corn effects (tentacle effects) | Documented | Low |
| CAMS | Cameras | Documented | Low |
| BPOS | Bind poses | Documented | Low |
| SKIN | Skinning (vertex weights) | Documented | Low |
| TXAN | Texture animations | Documented | Low |
| FAFX | Facial effects | Documented | Low |
| KGTR, KGRT, KGSC | Animation track types | Documented | High |

## Common Structures
Several structures are used throughout the format and are common to multiple chunks.

| Structure | Description | Status |
|-----------|-------------|--------|
| MDLGENOBJECT | Base object structure | Documented |

## Implementation Plan
1. ✅ Complete documentation of all chunks
   - ✅ Document MDLX, VERS, MODL, SEQS chunks (core model structure)
   - ✅ Document BONE chunk (skeletal system)
   - ✅ Document GEOS chunk (geometry)
   - ✅ Document GLBS, MTLS, TEXS chunks (materials & textures)
   - ✅ Document GEOA chunk (geoset animations)
   - ✅ Document LITE, HELP, PREM chunks (lights, helpers, basic particles)
   - ✅ Document PRE2, RIBB, ATCH chunks (advanced particles and attachments)
   - ✅ Document EVTS, CLID, CAMS, CORN chunks (utility components)
   - ✅ Document remaining chunks (BPOS, SKIN, TXAN, FAFX)
2. Implement parsers for each chunk type
   - Create basic model loader
   - Add support for core chunks
   - Add support for animation chunks
   - Add support for effect chunks
3. Implement model writer
   - Create model serialization
   - Add support for writing all chunks
4. Create validation and testing tools
   - Add model validators
   - Create test suite with sample models

## References
- Warcraft III: The Frozen Throne (official game)
- World of Warcraft Alpha/Beta clients
- Magos Model Editor documentation
- MDX format specification by community researchers
- Legacy MDL format documentation
- Blizzard WC3 community documentation 