# PEDC Chunk (Parent Event Data)

## Overview
The PEDC chunk contains parent event data for animations in M2 models. It was introduced in Legion (7.x) and stores animation events that are associated with parent sequences.

## Structure
```cpp
struct M2TrackBase {
  uint16_t trackType;
  uint16_t loopIndex;
  M2Array<M2SequenceTimes> sequenceTimes;
};

struct PEDC_Chunk {
  M2Array<M2TrackBase> parentEventData;  // Event tracks for parent animations
};
```

## Fields
- **parentEventData**: An array of M2TrackBase structures that define events triggered during parent animation sequences.

## Dependencies
- Requires the MD21 chunk for basic model data
- Related to animation sequences and events defined in the MD21 chunk
- May reference global sequences from the MD21 chunk

## Usage
The PEDC chunk is used for:
- Defining events that should occur during parent animation playback
- Synchronizing sounds, effects, or other triggers with parent animations
- Providing timing information for parent animation event processing

## Legacy Support
- Not present in pre-Legion M2 files
- In older versions, parent event data might have been stored differently or calculated at runtime

## Implementation Notes
- This chunk should be processed after the MD21 chunk is loaded
- Event tracks are typically used to trigger sounds, visual effects, or game logic at specific animation points
- The trackType field indicates what kind of event is represented
- The loopIndex may reference a global sequence for timing
- sequenceTimes provides specific timing data for when events should occur

## Version History
- Introduced in Legion (7.x) build
- The exact build number is currently unknown but was part of the Legion chunked format evolution 