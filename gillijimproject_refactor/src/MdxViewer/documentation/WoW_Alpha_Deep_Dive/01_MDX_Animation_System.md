# MDX Animation System

This document provides detailed analysis of the MDX animation system in WoW Alpha 0.5.3, based on reverse engineering the WoWClient.exe binary.

## Overview

The animation system in WoW Alpha uses a sophisticated keyframe-based animation track system that supports multiple interpolation types and hierarchical object animation. The system is built around the `CAnim` class hierarchy and handles animations for bones, geosets, materials, lights, attachments, and more.

## Key Classes and Structures

### HANIM__ (Animation Handle)
```
Address: 0x0073a850+
Purpose: Handle object for animation instances
Fields:
  - Handle base fields (reference counting, type identification)
  - Pointer to CAnim (animation data)
  - Animation state and status flags
```

### CAnim (Animation Data)
```
Purpose: Main animation data container
Created by: AnimCreate() @ 0x0073a850
Key fields:
  - CAnimData* hdata: Pointer to animation data header
  - CAnimObjBlendStatus blendStatus: Blending state for smooth transitions
  - uint flags: Animation configuration flags
  - uint status: Current animation status
```

### CAnimData (Animation Header)
```
Purpose: Contains all animation tracks and sequences
Loaded from: "MTLS" chunk (Model chunk)
Sections:
  - Sequences (0x53455153 / "SEQS")
  - Bones (0x454e4f42 / "BONE")
  - Lights (0x4554494c / "LITE")
  - Attachments (0x48435441 / "ATCH")
  - Particle Emitters (0x54534550 / "PETx")
  - Ribbons (0x42424952 / "RIBB")
  - Cameras (0x534d4143 / "CAMS")
  - Geosets (0x414f4547 / "GAEO")
  - Events (0x53545645 / "EVTS")
```

## Animation Chunk Structure

### ATSQ - Animation Sequence
```
Magic: 0x54535141 ("AT SQ")
Structure:
  - uint32 numSequences
  - uint32 bytesPerSequence
  - SequenceData[numSequences]:
    - uint32 animationId
    - uint32 duration (milliseconds)
    - float moveSpeed
    - uint32 flags
    - uint32 frequency
    - uint16 blendTime
    - char[16] name
    - uint32 variationIndex
```

### Keyframe Track Chunks

#### Linear Keyframes (KTLinear / 0x52454c4b = "KELn")
```
Used for: Simple interpolation between values
Structure:
  - uint32 numKeys
  - uint32 globalSequenceId
  - KeyframeData[numKeys]:
    - uint32 time
    - T value (depends on track type)
```

#### Spline Keyframes (KTSpline / 0x534c5453 = "SLTS")
```
Used for: Smooth interpolation with tangents
Structure:
  - uint32 numKeys
  - uint32 globalSequenceId
  - SplineKeyData[numKeys]:
    - uint32 time
    - T value
    - T inTangent
    - T outTangent
```

#### Quaternion Spline Keyframes (KTSplineQ / 0x534c5431 = "SLT1")
```
Used for: Rotations with Squad interpolation
Structure:
  - uint32 numKeys
  - uint32 globalSequenceId
  - QSplineKeyData[numKeys]:
    - uint32 time
    - C4QuaternionCompressed value (12 bytes)
    - uint32 tangentPacked (packed quaternion tangent)
```

## Track Types (MDLTRACKTYPE)

| Value | Type | Data Type | Interpolation |
|-------|------|-----------|--------------|
| 0 | Position | C3Vector | Linear, Hermite, Bezier |
| 1 | Rotation | C4Quaternion | Linear, Hermite (Squad) |
| 2 | Scaling | C3Vector | Linear, Hermite, Bezier |
| 3 | Visibility | float | Linear, Constant |
| 4 | Alpha | float | Linear, Constant |
| 5 | Color (RGB) | C3Color | Linear, Spline |
| 6 | TextureUV | C2Vector | Linear, Hermite |
| 7 | EmissionRate | float | Linear, Constant |
| 8 | Gravity | float | Linear, Constant |
| 9 | LifeSpan | float | Linear, Constant |
| 10 | Speed | float | Linear, Constant |
| 11 | Latitude | float | Linear, Constant |
| 12 | Longitude | float | Linear, Constant |
| 13 | Velocity | float | Linear, Constant |
| 14 | Scale | float | Linear, Constant |
| 15 | TailLength | float | Linear, Constant |
| 16 | Width | float | Linear, Constant |

## Interpolation Methods

### Linear Interpolation (InterpolateLinear @ 0x0075de20)
```
Formula: result = (1 - t) * value0 + t * value1
Used for: Simple, fast interpolation
Supported types: floats, vectors, colors, quaternions
```

### Hermite Interpolation (InterpolateHermite @ 0x0075dc00)
```
Formula: h(2t³ - 3t² + 1) + h₁(-2t³ + 3t²) + m₀(t³ - 2t² + t) + m₁(t³ - t²)
Where: h = v₁ - v₀, m = tangents
Used for: Smooth, natural-looking animations
Supported types: floats, vectors, quaternions (Squad)
```

### Bezier Interpolation (InterpolateBezier @ 0x0075de00)
```
Formula: Cubic bezier with control points
Used for: Eased animations, special effects
Supported types: floats, vectors
```

### Spherical Quaternion Interpolation (Squad)
```
Function: NTempest::C4Quaternion::Squad()
Purpose: Smooth rotation interpolation on sphere
Quaternion Compression:
  - Stored as 3 components (X, Y, Z) in 10 bits each
  - W component computed from: W = sqrt(1 - X² - Y² - Z²)
  - Scale factors: 0x35000000 (0.03125), 0x35800000 (0.0625)
```

## Animation Playback Functions

### MdxReadAnimation @ 0x004221b0
```c
undefined4 MdxReadAnimation(
  uchar* data,      // MDX file data
  uint offset,      // Offset to animation data
  int animHandle,    // Output animation handle
  uint flags        // Creation flags
)
{
  // Convert flags from MDX to internal format
  byte blendFlag = ConvertAnimCreateFlags(flags);
  
  // Create animation instance
  HANIM__* handle = AnimCreate(data, offset, blendFlag);
  
  // Store handle in output parameter
  *(HANIM__**)(animHandle + 0x40) = handle;
  
  // Check for immediate-close flag
  if (AnimGetFlags(handle) & 4) {
    HandleClose(handle);
    *(uint*)(animHandle + 0x40) = 0;
  }
  
  return 1;
}
```

### AnimBuild @ 0x0073cb10
```c
CAnim* AnimBuild(
  uchar* data,
  uint dataSize,
  CAnim* anim,
  uint flags
)
{
  // Initialize blending system if needed
  if (flags & 8) {
    // Setup blend status buffer
    // Allocate memory for CAnimObjBlendStatus array
    // Initialize blend timers and scales
  }
  
  // Get animation data header
  CAnimData* hdata = anim->hdata;
  
  // Build all animation components
  AnimAddSequences(data, dataSize, anim, hdata);
  AnimAddCameras(data, dataSize, hdata, trackType);
  AnimAddGeosets(data, dataSize, hdata, trackType);
  AnimAddTextureAnims(data, dataSize, anim, hdata, trackType);
  AnimAddMaterialLayers(data, dataSize, anim, hdata, trackType);
  
  // Create animated objects
  IAnimCreateObjects(data, dataSize, hdata, flags, ...);
  
  // Build bone hierarchy
  BuildHierarchy(hdata, parentIds, ...);
  
  // Initialize animation system
  AnimInit(anim, hdata);
  
  return &DAT_00000001;
}
```

## Geoset Animation (GAEO Chunk)

### AnimAddGeosets @ 0x00754c00
```
Purpose: Parse and build geoset animations
Chunk Magic: 0x414f4547 ("GAEO")
Structure:
  - uint32 numGeosetAnims
  - GeosetAnimData[numGeosetAnims]:
    - uint32 geosetIndex
    - uint32 animationType (0=visibility, 1=alpha, 2=color)
    - uint32 trackId
    - uint32 globalSeqId
    - KeyframeData
  
Color Animation (KGAC / 0x4341474b):
  - uint32 numKeys (> 0 required)
  - SplineKeyData[numKeys]:
    - uint32 time
    - C3Color value (RGB float, 0-1 range)
    - uint32 tangentPacked (for smooth color transitions)
  
Alpha Animation (KGAL / 0x4c41474b):
  - uint32 numKeys
  - KeyframeData[numKeys]:
    - uint32 time
    - float alpha (0-1 range)
```

### CAnimGeoset Structure
```c
struct CAnimGeoset {
  uint32 sgGeosetId;      // Geoset identifier
  CKeyFrameTrack<float> visibility;  // Visibility track
  CKeyFrameTrack<float> alpha;      // Alpha track
  CKeyFrameTrack<C3Color> color;     // Color track
};
```

## Animation Sequence System

### AnimAddSequences @ 0x00754000
```
Chunk: "SEQS" (0x53514553)
Purpose: Load animation sequence definitions
Structure:
  - uint32 numSequences
  - SequenceDefinition[numSequences]:
    - uint32 id
    - uint32 duration (ms)
    - float moveSpeed
    - uint32 flags
    - uint32 frequency
    - uint16 blendTime
    - char[16] name
    - uint32 variationIndex
```

## Blend Modes and Animation Transitions

### AnimEnableBlending @ 0x00741590
```
Purpose: Enable animation blending for smooth transitions
Parameters:
  - animHandle: Animation to blend
  - blendDuration: Time to blend (milliseconds)
  - blendType: Type of blend operation
```

### AnimUsesBlending @ 0x00741560
```
Purpose: Check if animation uses blending
Returns: true if animation has blend data
```

## Implementation Recommendations for MdxViewer

### 1. Animation Data Parsing
```csharp
// Implement ATSQ chunk parsing
public class AnimationSequence {
  public uint Id { get; set; }
  public uint Duration { get; set; }  // milliseconds
  public float MoveSpeed { get; set; }
  public uint Flags { get; set; }
  public uint Frequency { get; set; }
  public ushort BlendTime { get; set; }
  public string Name { get; set; }
  public uint VariationIndex { get; set; }
}

// Implement track parsing for each type
public interface IKeyframeTrack<T> {
  T Interpolate(float time);
  int NumKeys { get; }
  uint GlobalSequenceId { get; }
}
```

### 2. Interpolation System
```csharp
public enum InterpolationType {
  Linear = 0,
  Hermite = 1,
  Bezier = 2,
  Constant = 3
}

public class Interpolator {
  public static T Linear<T>(T v0, T v1, float t) where T : struct { ... }
  public static Vector3 Hermite(Vector3 v0, Vector3 v1, Vector3 m0, Vector3 m1, float t) { ... }
  public static Quaternion Squad(Quaternion q0, Quaternion q1, Quaternion a, Quaternion b, float t) { ... }
}
```

### 3. Animation Playback
```csharp
public class AnimationPlayer {
  public float CurrentTime { get; set; }
  public uint CurrentSequence { get; set; }
  public float PlaybackSpeed { get; set; }
  public bool IsPlaying { get; set; }
  
  public void Update(float deltaTime) {
    CurrentTime += deltaTime * PlaybackSpeed;
    if (CurrentTime > GetCurrentSequenceDuration()) {
      CurrentTime = 0;  // Loop
    }
    UpdateAllTracks();
  }
  
  private void UpdateAllTracks() {
    foreach (var track in Tracks) {
      var value = track.Interpolate(CurrentTime);
      ApplyTrackValue(track, value);
    }
  }
}
```

### 4. Quaternion Handling
```csharp
public class QuaternionCompression {
  // Decompress from 30-bit packed format
  public static Quaternion Decompress(uint packed) {
    // Extract X, Y, Z from 10-bit fields
    float x = (float)(packed >> 20) / 1023.0f * 2.0f - 1.0f;
    float y = (float)((packed >> 10) & 0x3FF) / 1023.0f * 2.0f - 1.0f;
    float z = (float)(packed & 0x3FF) / 1023.0f * 2.0f - 1.0f;
    
    // Compute W from X, Y, Z to ensure normalized quaternion
    float sum = x*x + y*y + z*z;
    float w = (sum >= 1.0f) ? 0.0f : (float)Math.Sqrt(1.0 - sum);
    
    return new Quaternion(x, y, z, w);
  }
}
```

### 5. Animation UI Display
```csharp
// For debugging/animation list display
public class AnimationInfoDisplay {
  public void ShowAnimationInfo(AnimationPlayer player) {
    var seq = player.GetCurrentSequence();
    ImGui.Text($"Animation: {seq.Name}");
    ImGui.Text($"Duration: {seq.Duration}ms");
    ImGui.Text($"Time: {player.CurrentTime:F1}ms / {seq.Duration}ms");
    
    // Progress bar
    ImGui.ProgressBar(player.CurrentTime / seq.Duration);
    
    // Sequence selector
    for (int i = 0; i < TotalSequences; i++) {
      if (ImGui.Selectable(GetSequenceName(i), i == player.CurrentSequence)) {
        player.PlaySequence(i);
      }
    }
  }
}
```

## Key Functions Reference

| Function | Address | Purpose |
|----------|---------|---------|
| AnimCreate | 0x0073a850 | Create animation instance |
| AnimBuild | 0x0073cb10 | Build animation data from MDX |
| AnimInit | 0x0073cxxx | Initialize animation system |
| AnimAddSequences | 0x00754000 | Load sequences |
| AnimAddGeosets | 0x00754c00 | Load geoset animations |
| AnimAddMaterialLayers | 0x00755800 | Load material animations |
| InterpolateLinear | 0x0075de20 | Linear interpolation |
| InterpolateHermite | 0x0075dc00 | Hermite spline interpolation |
| InterpolateBezier | 0x0075de00 | Bezier curve interpolation |
| AnimEnableBlending | 0x00741590 | Enable animation blending |
| ChooseAnimation | 0x005fbd10 | Select animation |
| PlayBaseAnimation | 0x005f53c0 | Play base animation |
| UpdateBaseAnimation | 0x005f56a0 | Update animation frame |

## Debugging Tips

1. **Animation not playing?** Check:
   - ATSQ chunk exists and has valid sequences
   - Animation ID is valid for the model
   - Track data has valid timestamps

2. **Jittery animation?** Check:
   - Keyframe timestamps are in correct order
   - Using appropriate interpolation (Hermite for smooth)
   - Global sequence IDs are consistent

3. **Wrong colors?** Check:
   - C3Color values are in 0-1 range (not 0-255)
   - Alpha values are properly clamped
   - Color interpolation is using correct gamma

4. **Rotations look wrong?** Check:
   - Quaternion decompression is correct
   - Using Squad interpolation for rotations
   - Rotation order (XYZ vs ZYX) is correct

## References
- WoW MDX file format specifications
- Animation system implementation in retail WoW
- Ghidra decompilation of WoWClient.exe v0.5.3
