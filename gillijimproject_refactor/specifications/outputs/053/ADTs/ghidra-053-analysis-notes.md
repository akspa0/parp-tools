# Ghidra 0.5.3 Analysis Notes (ADT)

## Scope and goals
- Document 0.5.3 WDT->ADT->MCNK loading flow with Ghidra evidence.
- Capture MCNK subchunk usage, textures/alpha pipeline, and MDDF/MODF placements.

## Function map
- WDT load: `CMap::LoadWdt` @ `0x0067fde0`
- ADT load/parse: `Create` @ `0x006aad30`
- MCNK load path: `CMapChunk::SyncLoad` @ `0x00698d20`
- MCAL decode: `CMapChunk::UnpackAlphaBits` @ `0x0069a5f0`
- MCSH decode: `CMapChunk::UnpackShadowBits` @ `0x0069a6b0`
- Alpha+shadow combine: `CMapChunk::UnpackAlphaShadowBits` @ `0x0069a430`
- MDDF/MODF creation: `CMap::CreateDoodadDef` @ `0x006a6cf0`, `CMap::CreateMapObjDef` @ `0x00681250`

## Constants and offsets
- MDDF entry size: 0x24 (magic divide in 0x006aad30)
- MODF entry size: 0x40 (size >> 6 in 0x006aad30)
- Shadow texture sizes: 0x20 or 0x40 based on `CWorld::shadowMipLevel`
- Alpha texture sizes: 0x20 or 0x40 based on `CWorld::alphaMipLevel`

## Open questions
- Confirm exact MCNK header offsets for MCVT/MCNR/MCAL/MCSH pointers (currently inferred via call paths).
- Tie legacy liquid subchunk pointer (MCLQ) to the exact header field and flag bit.

## Next steps
- Extract MCNK header field offsets from the `SMChunk` structure and annotate in the deep dive.
- Trace liquid subchunk offsets and integrate into the liquid pipeline section.
