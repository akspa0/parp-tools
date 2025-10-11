# MDX Format Documentation

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

## Implementation Status
- Documentation: 26/26 chunks+structures (100%)
- Parsing Implementation: 0/26 chunks+structures (0%)
- Writing Implementation: 0/26 chunks+structures (0%)

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