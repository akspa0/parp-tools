# M2AnimTrack Structure

## Overview
The M2AnimTrack structure defines a keyframe animation track in an M2 model. Animation tracks contain keyframe data for various animated properties such as bone positions, rotations, colors, texture coordinates, and transparency values. Each track is associated with a specific sequence and animates a specific property.

## Structure

```cpp
struct M2TrackHead {
  uint16_t interpolation_type;   // Type of interpolation between keyframes
  uint16_t global_sequence;      // Global sequence ID or -1 if not global
  M2Array<M2Array<timestamp_t>> timestamps;  // Timestamps for each keyframe by sequence
  M2Array<M2Array<T>> values;    // Values for each keyframe by sequence
};
```

Note: The actual data type `T` of the values depends on what is being animated:
- For bone translations: C3Vector (three 32-bit floats)
- For bone rotations: C4Quaternion (four 16-bit fixed-point values or four 32-bit floats)
- For colors: C3Vector (three 8-bit or 32-bit color components)
- For transparency: float (32-bit float)
- For texture coordinates: C2Vector (two 32-bit floats)

## Fields

- **interpolation_type**: Determines how values are interpolated between keyframes
- **global_sequence**: ID of the global sequence this track belongs to, or -1 if it uses normal animation sequences
- **timestamps**: Array of timestamp arrays for each animation sequence
- **values**: Array of value arrays for each animation sequence, containing the actual keyframe data

## Interpolation Types

The interpolation_type field can have the following values:

- **0**: None - No interpolation, value snaps to the keyframe value
- **1**: Linear - Linear interpolation between keyframes
- **2**: Hermite - Hermite spline interpolation (C2 continuity)
- **3**: Bezier - Bezier curve interpolation (with control points)

Additional flags can be combined with the base interpolation type:
- **0x10**: Global - Track uses global sequences rather than animation sequences
- **0x20**: Compressed - Used for packed quaternion rotation tracks (later versions only)

## Timestamps and Values

Each animation track contains multiple arrays of timestamps and values:
- One array for each animation sequence in the model
- Each array contains the keyframes for that specific sequence
- Timestamps are in milliseconds, relative to the start of the sequence
- Values are the actual data for the animated property at each keyframe
- The number of timestamps must match the number of values for each sequence

For global sequence animations (interpolation_type & 0x10):
- Only one array of timestamps and values is used
- Timing is based on the global sequence timer rather than animation sequence
- Global sequences are used for continuous animations like blinking or breathing

## Implementation Notes

- Animation tracks are stored separately for different animated properties
- The same track structure is used regardless of the data type being animated
- The array structure allows for efficient animation playback
- Not all sequences have keyframes for all properties
- Compression may be used for quaternion tracks to save space
- Animation types have different storage formats:
  - **KGTR** tracks: Translations (C3Vector)
  - **KGRT** tracks: Rotations (C4Quaternion)
  - **KGSC** tracks: Scaling (C3Vector)
  - **KATV** tracks: Alpha/transparency (float)
  - **KRTX** tracks: Texture coordinates (C2Vector)
  - **KRVD** tracks: Vertex colors (C3Vector)

## Animation Track Processing

To animate a property using an animation track:
1. Determine the current animation sequence and timestamp
2. If the track uses a global sequence, calculate the timestamp from the global timer
3. Find the two keyframes that surround the current timestamp
4. Use the interpolation_type to determine how to interpolate between them
5. For compressed quaternions, expand to full quaternions before interpolation
6. Apply the interpolated value to the animated property

## Usage Example

For a bone rotation animation:
- Each bone has a rotation track with quaternion values
- During animation, the system looks up the current keyframes based on the timestamp
- The quaternions from the two nearest keyframes are interpolated based on the interpolation type
- The resulting quaternion is applied to the bone's transformation matrix
- This process repeats for each animated bone in each frame

For color or transparency animations:
- The process is similar, but uses different data types (colors or float values)
- These animations typically apply to materials rather than bones 