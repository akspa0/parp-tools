# Model Rendering Pipeline

This document provides a comprehensive overview of the model rendering pipeline in WoW Alpha 0.5.3, based on reverse engineering the WoWClient.exe binary.

## Overview

The model rendering pipeline in WoW Alpha is a sophisticated system that handles loading, animating, and rendering 3D models (MDX format). The pipeline integrates with the animation system, handles transparency sorting, supports various blend modes, and optimizes rendering through frustum culling and level-of-detail (LOD) management.

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL RENDERING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. LOADING                  2. ANIMATION            3. RENDER PREPARATION │
│  ┌────────────────┐         ┌────────────────┐        ┌────────────────────┐  │
│  │ Load MDX File  │────────▶│ Build CAnim   │──────▶│ Update Bone        │  │
│  │ Parse Chunks   │         │ Parse Tracks  │       │ Transforms         │  │
│  └────────────────┘         └────────────────┘       └────────────────────┘  │
│         │                                                  │                 │
│         ▼                                                  ▼                 │
│  ┌────────────────┐                         ┌────────────────────┐        │
│  │ Load Textures   │                         │ Calculate World    │        │
│  │ Load Geometries │                         │ Matrix             │        │
│  └────────────────┘                         └────────────────────┘        │
│                                                                   │         │
│                                                                   ▼         │
│  4. CULLING                 5. SORTING              6. RENDERING           │
│  ┌────────────────┐         ┌────────────────┐     ┌────────────────────┐  │
│  │ Frustum Test   │────────▶│ Opaque First   │────▶│ Bind Shaders       │  │
│  │ LOD Selection  │         │ Transparent   │     │ Bind Textures      │  │
│  └────────────────┘         │ Back-to-Front │     │ Draw Geometry      │  │
│                             └────────────────┘     └────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Functions

### Model Loading Pipeline

#### MdxReadAnimation @ 0x004221b0
```
Purpose: Load and parse MDX animation data
Parameters:
  - data: MDX file data
  - offset: Offset to animation data in file
  - animHandle: Output handle for animation
  - flags: Load flags

Flow:
  1. Read animation chunk header
  2. Parse sequence definitions (SEQS)
  3. Parse bone animations (BONE)
  4. Parse material animations (MTLA)
  5. Parse texture animations (TXAN)
  6. Parse geoset animations (GAEO)
  7. Parse attachment animations (ATCH)
  8. Parse particle animations (PREM)
  9. Build animation hierarchy
  10. Initialize animation state
```

#### AnimCreate @ 0x0073a850
```c
HANIM__* AnimCreate(uchar* data, uint dataSize, uint flags)
{
  // Count objects in each section
  uint numBones = CountSectionEntries(data, dataSize, "BONE");
  uint numLights = CountSectionEntries(data, dataSize, "LITE");
  uint numAttachments = CountSectionEntries(data, dataSize, "ATCH");
  uint numParticleEmitters = CountSectionEntries(data, dataSize, "PETx");
  uint numRibbons = CountSectionEntries(data, dataSize, "RIBB");
  uint numCameras = CountSectionEntries(data, dataSize, "CAMS");
  uint numGeosets = CountSectionEntries(data, dataSize, "GAEO");
  uint numEvents = CountSectionEntries(data, dataSize, "EVTS");
  
  // Calculate animated layers from MTLS chunk
  uint animatedLayers = 0;
  uchar* mtlChunk = MDLFileBinarySeek(data, dataSize, "MTLS");
  if (mtlChunk != nullptr) {
    animatedLayers = *(uint*)(mtlChunk + 8);
  }
  
  // Create animation data structures
  CAnim* anim = AnimCreate(objectCounts, animatedLayers, flags);
  
  // Build animation tracks
  AnimBuild(data, dataSize, anim, flags);
  
  // Create handle
  HANIM__* handle = HandleCreate(anim, "HANIM");
  
  return handle;
}
```

### Animation Update Pipeline

#### UpdateBaseAnimation @ 0x005f56a0
```c
void UpdateBaseAnimation(HANIM__* animHandle, float deltaTime)
{
  CAnim* anim = HandleGetObject(animHandle);
  
  // Get current sequence
  uint currentSeq = anim->currentSequence;
  CAnimSequence* seq = &anim->sequences[currentSeq];
  
  // Update animation time
  anim->currentTime += deltaTime * seq->playbackSpeed;
  
  // Handle looping
  if (anim->currentTime >= seq->duration) {
    anim->currentTime = anim->currentTime % seq->duration;
    anim->loopCount++;
  }
  
  // Update all animation tracks
  UpdateAnimationTracks(anim, anim->currentTime);
  
  // Update bone transforms
  UpdateBoneTransforms(anim);
  
  // Update geoset visibility
  UpdateGeosetAnimations(anim, anim->currentTime);
  
  // Update material properties
  UpdateMaterialAnimations(anim, anim->currentTime);
}
```

### Animation Track Interpolation

#### InterpolateLinear @ 0x0075de20
```c
// Generic linear interpolation for any type
T InterpolateLinear(T* keyframes, int numKeys, float time)
{
  // Find surrounding keyframes
  int i = 0;
  while (i < numKeys - 1 && keyframes[i + 1].time < time) {
    i++;
  }
  
  if (i >= numKeys - 1) {
    return keyframes[numKeys - 1].value;
  }
  
  // Get keyframe values
  float t0 = keyframes[i].time;
  float t1 = keyframes[i + 1].time;
  T v0 = keyframes[i].value;
  T v1 = keyframes[i + 1].value;
  
  // Calculate interpolation factor
  float t = (time - t0) / (t1 - t0);
  t = clamp(t, 0.0f, 1.0f);
  
  // Interpolate
  return Lerp(v0, v1, t);
}
```

#### InterpolateHermite @ 0x0075dc00
```c
// Quaternion interpolation using Squad (Spherical Quadrangle)
void InterpolateHermite(
  CKeyFrameTrack<C4QuaternionCompressed, C4Quaternion>* track,
  CSplineKeyFrame<C4QuaternionCompressed>* key0,
  CSplineKeyFrame<C4QuaternionCompressed>* key1,
  float t,
  C4Quaternion* outResult
)
{
  // Decompress quaternions
  C4Quaternion q0 = DecompressQuaternion(key0->value);
  C4Quaternion q1 = DecompressQuaternion(key1->value);
  
  // Decompress tangents
  C4Quaternion tan0 = DecompressQuaternion(key0->tangent);
  C4Quaternion tan1 = DecompressQuaternion(key1->tangent);
  
  // Calculate Squad control points
  C4Quaternion a = CalculateSquadControlPoint(q0, q0, q1, tan0);
  C4Quaternion b = CalculateSquadControlPoint(q0, q1, q1, tan1);
  
  // Interpolate using Squad
  Squad(outResult, q0, q1, a, b, t);
}
```

### Bone Transform Update

#### UpdateBoneTransforms @ 0x005f5xxx
```c
void UpdateBoneTransforms(CAnim* anim)
{
  // Get bone count
  int numBones = anim->boneCount;
  
  // Update each bone
  for (int i = 0; i < numBones; i++) {
    CBone* bone = &anim->bones[i];
    
    // Skip if bone is not animated
    if (!bone->isAnimated) {
      continue;
    }
    
    // Get animated values from tracks
    C3Vector position = bone->positionTrack.Evaluate(anim->currentTime);
    C4Quaternion rotation = bone->rotationTrack.Evaluate(anim->currentTime);
    C3Vector scale = bone->scaleTrack.Evaluate(anim->currentTime);
    
    // Apply parent transform if parent exists
    if (bone->parentIndex >= 0) {
      CBone* parent = &anim->bones[bone->parentIndex];
      
      // Calculate local matrix
      C4x4 localMatrix = ComposeMatrix(position, rotation, scale);
      
      // Combine with parent matrix
      bone->worldMatrix = parent->worldMatrix * localMatrix;
    } else {
      // Root bone - use as-is
      bone->worldMatrix = ComposeMatrix(position, rotation, scale);
    }
    
    // Calculate inverse bind pose for skinning
    bone->inverseBindPose = bone->worldMatrix.Invert();
  }
}
```

### Rendering Pipeline

#### ModelSetBlendMode @ 0x00440490
```c
void ModelSetBlendMode(HMODEL__* modelHandle, EGxBlend blendMode)
{
  CModel* model = HandleGetObject(modelHandle);
  
  // Get model geosets
  int numGeosets = ModelGetGeosetCount(model);
  
  for (int i = 0; i < numGeosets; i++) {
    CGeoset* geoset = ModelGetGeoset(model, i);
    
    // Get geoset material
    CMaterial* material = GeosetGetMaterial(geoset);
    
    // Update blend mode
    MaterialSetBlendMode(material, blendMode);
    
    // Update render flags
    if (blendMode == GxBlend_Opaque) {
      GeosetSetRenderFlag(geoset, GEOSET_RENDER_OPAQUE);
      GeosetClearRenderFlag(geoset, GEOSET_RENDER_TRANSPARENT);
    } else {
      GeosetSetRenderFlag(geoset, GEOSET_RENDER_TRANSPARENT);
      GeosetClearRenderFlag(geoset, GEOSET_RENDER_OPAQUE);
      
      // Calculate sort depth
      float sortDepth = CalculateGeosetDepth(model, geoset);
      GeosetSetSortDepth(geoset, sortDepth);
    }
  }
}
```

### Frustum Culling

#### FrustumCullModel @ 0x0044xxxx
```c
CullingResult FrustumCullModel(CModel* model, CFrustum* frustum)
{
  // Get model bounding sphere
  CSphere modelSphere = model->boundingSphere;
  
  // Transform sphere to world space
  C3Vector center = model->worldMatrix.TransformPoint(modelSphere.center);
  float radius = modelSphere.radius * model->worldMatrix.GetScale();
  
  // Test against frustum
  if (!FrustumContainsSphere(frustum, center, radius)) {
    return CULL_OUTSIDE;
  }
  
  // Check if fully contained
  if (FrustumContainsSphereFully(frustum, center, radius)) {
    return CULL_INSIDE;
  }
  
  // Test individual geosets
  int visibleGeosets = 0;
  int numGeosets = ModelGetGeosetCount(model);
  
  for (int i = 0; i < numGeosets; i++) {
    CGeoset* geoset = ModelGetGeoset(model, i);
    
    CSphere geoSphere = geoset->boundingSphere;
    C3Vector geoCenter = model->worldMatrix.TransformPoint(geoSphere.center);
    float geoRadius = geoSphere.radius * model->worldMatrix.GetScale();
    
    if (FrustumContainsSphere(frustum, geoCenter, geoRadius)) {
      visibleGeosets++;
      GeosetSetVisible(geoset, true);
    } else {
      GeosetSetVisible(geoset, false);
    }
  }
  
  return visibleGeosets > 0 ? CULL_PARTIAL : CULL_OUTSIDE;
}
```

### Transparency Sorting

#### SortRenderQueue @ 0x0044xxxx
```c
void SortRenderQueue(CRenderQueue* queue)
{
  // Separate opaque and transparent items
  List<RenderItem> opaqueItems;
  List<RenderItem> transparentItems;
  
  for (int i = 0; i < queue->itemCount; i++) {
    RenderItem* item = &queue->items[i];
    
    if (item->blendMode == GxBlend_Opaque) {
      opaqueItems.Add(item);
    } else {
      transparentItems.Add(item);
    }
  }
  
  // Sort opaque items (front-to-back for optimization)
  opaqueItems.Sort((a, b) => a.depth.CompareTo(b.depth));
  
  // Sort transparent items (back-to-front for correctness)
  transparentItems.Sort((a, b) => b.depth.CompareTo(a.depth));
  
  // Rebuild queue
  queue->itemCount = 0;
  
  // Add opaque items first
  for (int i = 0; i < opaqueItems.Count; i++) {
    queue->items[queue->itemCount++] = opaqueItems[i];
  }
  
  // Add transparent items second
  for (int i = 0; i < transparentItems.Count; i++) {
    queue->items[queue->itemCount++] = transparentItems[i];
  }
}
```

### Final Rendering

#### RenderModel @ 0x0044xxxx
```c
void RenderModel(HMODEL__* modelHandle)
{
  CModel* model = HandleGetObject(modelHandle);
  
  // Apply model transform
  GxSetWorldMatrix(model->worldMatrix);
  
  // Get visible geosets
  int numGeosets = ModelGetGeosetCount(model);
  
  for (int i = 0; i < numGeosets; i++) {
    CGeoset* geoset = ModelGetGeoset(model, i);
    
    // Skip if not visible
    if (!GeosetIsVisible(geoset)) {
      continue;
    }
    
    // Get geoset material
    CMaterial* material = GeosetGetMaterial(geoset);
    
    // Apply material properties
    SetMaterialBlendMode(material->blendMode);
    
    // Bind textures
    for (int stage = 0; stage < material->numTextures; stage++) {
      HTEXTURE__* texture = material->textures[stage];
      GxSetTexture(stage, texture->GetTextureId());
    }
    
    // Set shader
    HSHADER__* shader = GetTextureShader(material->textures[0]);
    GxBindShader(shader);
    
    // Draw geoset geometry
    DrawGeoset(geoset);
  }
}
```

## Render Queue Management

### RenderQueue Structure
```c
struct CRenderQueue {
  RenderItem* items;              // Array of render items
  int itemCapacity;              // Maximum items
  int itemCount;                 // Current items
  
  // State tracking
  uint currentShader;
  uint currentTexture;
  EGxBlend currentBlendMode;
  C4x4 currentWorldMatrix;
};
```

### RenderItem Structure
```c
struct RenderItem {
  HMODEL__* model;              // Model to render
  C4x4 worldMatrix;             // World transform
  CGeoset* geoset;               // Specific geoset (or null for all)
  EGxBlend blendMode;           // Blend mode
  float sortDepth;              // Depth for sorting
  uint renderFlags;             // Additional flags
};
```

## Implementation Recommendations for MdxViewer

### 1. Animation System
```csharp
public class AnimationSystem
{
    private Dictionary<string, AnimationData> _animations = new();
    private AnimationData? _currentAnimation;
    private float _currentTime;
    private float _playbackSpeed = 1.0f;
    private bool _isPlaying;
    
    public void LoadAnimation(string name, byte[] data)
    {
        var anim = new AnimationData();
        anim.Load(data);
        _animations[name] = anim;
    }
    
    public void Play(string name)
    {
        if (_animations.TryGetValue(name, out var anim)) {
            _currentAnimation = anim;
            _currentTime = 0;
            _isPlaying = true;
        }
    }
    
    public void Update(float deltaTime)
    {
        if (!_isPlaying || _currentAnimation == null) return;
        
        _currentTime += deltaTime * _playbackSpeed;
        
        if (_currentTime >= _currentAnimation.Duration) {
            if (_currentAnimation.IsLooping) {
                _currentTime %= _currentAnimation.Duration;
            } else {
                _currentTime = _currentAnimation.Duration;
                _isPlaying = false;
            }
        }
        
        // Update all tracks
        foreach (var track in _currentAnimation.Tracks) {
            track.Update(_currentTime);
        }
    }
    
    public IReadOnlyList<AnimationData> GetAnimations()
    {
        return _animations.Values.ToList();
    }
}
```

### 2. Bone Transform System
```csharp
public class BoneTransformSystem
{
    private List<Bone> _bones = new();
    
    public void UpdateBoneTransforms(AnimationSystem animSystem)
    {
        for (int i = 0; i < _bones.Count; i++) {
            var bone = _bones[i];
            
            // Get animated values from animation tracks
            if (bone.PositionTrack != null) {
                bone.LocalPosition = bone.PositionTrack.Evaluate(animSystem.CurrentTime);
            }
            
            if (bone.RotationTrack != null) {
                bone.LocalRotation = bone.RotationTrack.Evaluate(animSystem.CurrentTime);
            }
            
            if (bone.ScaleTrack != null) {
                bone.LocalScale = bone.ScaleTrack.Evaluate(animSystem.CurrentTime);
            }
            
            // Calculate local matrix
            bone.LocalMatrix = Matrix4x4.Compose(
                bone.LocalScale,
                bone.LocalRotation,
                bone.LocalPosition
            );
            
            // Apply parent transform
            if (bone.ParentIndex >= 0) {
                var parent = _bones[bone.ParentIndex];
                bone.WorldMatrix = parent.WorldMatrix * bone.LocalMatrix;
            } else {
                bone.WorldMatrix = bone.LocalMatrix;
            }
            
            // Calculate inverse bind pose
            bone.InverseBindPose = bone.WorldMatrix.Invert();
        }
    }
}
```

### 3. Render Queue System
```csharp
public class RenderQueue
{
    private List<RenderItem> _opaqueItems = new();
    private List<RenderItem> _transparentItems = new();
    private BlendStateManager _blendStates = new();
    
    public void AddItem(RenderItem item)
    {
        if (item.BlendMode == BlendMode.Opaque) {
            _opaqueItems.Add(item);
        } else {
            _transparentItems.Add(item);
        }
    }
    
    public void Render(Camera camera)
    {
        // Sort opaque items front-to-back
        _opaqueItems.Sort((a, b) => 
            a.SortDepth.CompareTo(b.SortDepth));
        
        // Render opaque items
        foreach (var item in _opaqueItems) {
            RenderItem(item, camera);
        }
        
        // Sort transparent items back-to-front
        _transparentItems.Sort((a, b) => 
            b.SortDepth.CompareTo(a.SortDepth));
        
        // Render transparent items
        foreach (var item in _transparentItems) {
            RenderItem(item, camera);
        }
        
        // Clear queue
        _opaqueItems.Clear();
        _transparentItems.Clear();
    }
    
    private void RenderItem(RenderItem item, Camera camera)
    {
        // Set world matrix
        GL.MatrixMode(MatrixMode.Modelview);
        var viewProj = camera.ViewMatrix * camera.ProjectionMatrix;
        var worldViewProj = item.WorldMatrix * viewProj;
        GL.LoadMatrix(ref worldViewProj);
        
        // Apply blend mode
        _blendStates.ApplyBlendMode(item.BlendMode);
        
        // Render mesh
        item.Mesh.Render(item.WorldMatrix);
    }
}
```

### 4. Frustum Culling System
```csharp
public class FrustumCuller
{
    private Plane[] _planes = new Plane[6];
    
    public void UpdateFrustum(Matrix4x4 viewProj)
    {
        // Extract frustum planes from view-projection matrix
        // Left plane
        _planes[0] = new Plane(
            viewProj.M14 + viewProj.M11,
            viewProj.M24 + viewProj.M21,
            viewProj.M34 + viewProj.M31,
            viewProj.M44 + viewProj.M41
        ).Normalized();
        
        // Right plane
        _planes[1] = new Plane(
            viewProj.M14 - viewProj.M11,
            viewProj.M24 - viewProj.M21,
            viewProj.M34 - viewProj.M31,
            viewProj.M44 - viewProj.M41
        ).Normalized();
        
        // ... extract remaining planes (top, bottom, near, far)
    }
    
    public bool TestSphere(Vector3 center, float radius)
    {
        foreach (var plane in _planes) {
            float distance = plane.DotCoordinate(center);
            if (distance < -radius) {
                return false; // Outside frustum
            }
        }
        return true;
    }
    
    public bool TestBox(BoundingBox box)
    {
        foreach (var plane in _planes) {
            Vector3 positive = box.GetPositiveVertex(plane.Normal);
            Vector3 negative = box.GetNegativeVertex(plane.Normal);
            
            if (plane.DotCoordinate(negative) > 0) {
                return false; // Outside frustum
            }
        }
        return true;
    }
}
```

### 5. Animation UI Display
```csharp
public class AnimationDebugUI
{
    public void Draw(AnimationSystem animSystem)
    {
        ImGui.Begin("Animation Debug");
        
        // Animation selector
        var animations = animSystem.GetAnimations();
        var animNames = animations.Select(a => a.Name).ToArray();
        
        int selectedIndex = Array.IndexOf(animNames, animSystem.CurrentAnimation?.Name);
        if (ImGui.Combo("Animation", ref selectedIndex, animNames)) {
            animSystem.Play(animNames[selectedIndex]);
        }
        
        // Playback controls
        ImGui.Checkbox("Playing", ref animSystem.IsPlaying);
        ImGui.SliderFloat("Speed", ref animSystem.PlaybackSpeed, 0.0f, 5.0f);
        
        // Timeline
        if (animSystem.CurrentAnimation != null) {
            ImGui.Text($"Duration: {animSystem.CurrentAnimation.Duration:F1}ms");
            ImGui.Text($"Time: {animSystem.CurrentTime:F1}ms");
            ImGui.ProgressBar(
                animSystem.CurrentTime / animSystem.CurrentAnimation.Duration,
                new Vector2(-1, 0)
            );
            
            // Sequence list
            ImGui.Separator();
            ImGui.Text("Sequences:");
            foreach (var seq in animSystem.CurrentAnimation.Sequences) {
                ImGui.BulletText($"{seq.Id}: {seq.Name} ({seq.Duration}ms)");
            }
            
            // Track list
            ImGui.Separator();
            ImGui.Text("Tracks:");
            foreach (var track in animSystem.CurrentAnimation.Tracks) {
                ImGui.BulletText($"{track.Name}: {track.Type} ({track.KeyCount} keys)");
            }
        }
        
        ImGui.End();
    }
}
```

## Key Functions Reference

| Function | Address | Purpose |
|----------|---------|---------|
| MdxReadAnimation | 0x004221b0 | Load MDX animation data |
| AnimCreate | 0x0073a850 | Create animation instance |
| AnimBuild | 0x0073cb10 | Build animation tracks |
| AnimInit | 0x0073cxxx | Initialize animation |
| UpdateBaseAnimation | 0x005f56a0 | Update animation frame |
| UpdateBaseAnimation | 0x005f5810 | Alternative update path |
| InterpolateLinear | 0x0075de20 | Linear interpolation |
| InterpolateHermite | 0x0075dc00 | Hermite spline interpolation |
| InterpolateBezier | 0x0075de00 | Bezier curve interpolation |
| ModelSetBlendMode | 0x00440490 | Set model blend mode |
| ChooseAnimation | 0x005fbd10 | Select animation to play |
| PlayBaseAnimation | 0x005f53c0 | Play base animation |
| SetTorsoAnimation | 0x005f5ec0 | Set torso animation |

## Performance Considerations

1. **Frustum Culling**: Always cull models before adding to render queue
2. **LOD Selection**: Use appropriate LOD based on distance
3. **Batch Rendering**: Group items with same material/shader
4. **Transparency Sorting**: Sort transparent objects back-to-front
5. **Bone Count**: Limit bone count for GPU skinning performance
6. **Texture Streaming**: Stream textures on background thread

## Debugging Tips

1. **Animation not updating?** Check:
   - Delta time is being passed correctly
   - Animation is set to playing
   - Current time is incrementing
   - Track interpolation functions are working

2. **Wrong bone positions?** Check:
   - Parent indices are correct
   - Matrix multiplication order is correct
   - Local vs world space is handled properly

3. **Sorting issues?** Check:
   - Depth calculation is correct
   - Opaque items come before transparent
   - Transparent items are sorted back-to-front

4. **Culling too much?** Check:
   - Frustum planes are extracted correctly
   - Bounding spheres are calculated correctly
   - World transform is applied before testing
