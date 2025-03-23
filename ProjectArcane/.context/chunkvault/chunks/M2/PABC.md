# PABC: Particle Animation Blacklist

## Identification
- **Chunk ID**: PABC
- **Parent Format**: M2
- **Source**: M2 file format documentation

## Description
The PABC chunk (Particle Animation Blacklist/Data) was introduced in patch 7.3 and appears to replace the animation lookup functionality for particle systems. It contains a list of animation IDs that are used to determine which parent model animations affect particle emissions.

## Structure
```cpp
struct PABCChunk {
    M2Array<uint16_t> m_replacementParentSequenceLookups;  // List of animation IDs
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| m_replacementParentSequenceLookups | M2Array<uint16_t> | Array of animation IDs, used for determining which animations are present in the parent model |

## Dependencies
- Depends on the MD21 chunk which contains animation sequences.
- Related to the particle emitter system defined in the main M2 data.
- May affect how parent model animations influence particle systems.

## Implementation Notes
1. This chunk is referred to as "BlacklistAnimData" in the client.
2. Unlike the header's sequence_lookups from the parent model, this is a straight array and not a map.
3. If an index with the target animation is not found in this array, the parent's sequence lookups are used instead.
4. The client seems to use this array only to check if the target animation is present, not for the actual index value.
5. Only seen in specific models like "quillboarbrute*.m2" according to the documentation.

## Usage Context
The PABC chunk is used for:
- Overriding the default parent sequence lookups for particle animations
- Enabling or disabling particle effects based on specific animations
- Controlling which animations of a parent model trigger or affect particle emissions
- Potentially blacklisting animations from affecting particle systems

This chunk represents part of the advanced animation control system introduced in patch 7.3, giving artists more precise control over how particles interact with model animations. 