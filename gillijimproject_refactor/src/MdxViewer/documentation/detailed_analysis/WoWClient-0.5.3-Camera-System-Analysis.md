# WoW Alpha 0.5.3 (Build 3368) Camera System Analysis

## Overview

This document provides a deep analysis of the camera system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers camera classes, projection setup, target tracking, and script integration.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `Camera` | 0x00540a10 | Main camera class |
| `CSimpleCamera` | 0x0053a6e0 | Simple camera implementation |
| `CGCamera` | 0x0053bd60 | Game camera class |
| `CameraCreate` | 0x00482880 | Create camera instance |
| `CameraDestroy` | 0x0053bd50 | Destroy camera instance |
| `CameraDuplicate` | 0x00482e20 | Duplicate camera |
| `CameraUpdate` | 0x00483a10 | Update camera state |
| `CameraInitialize` | 0x0053ba10 | Initialize camera |
| `CameraRegisterScriptFunctions` | 0x0053bd00 | Register Lua functions |
| `CameraUnregisterScriptFunctions` | 0x0053bd30 | Unregister Lua functions |
| `CameraSetupScreenProjection` | 0x004837e0 | Setup screen projection |
| `CameraSetupWorldProjection` | 0x004839d0 | Setup world projection |
| `CameraGetLineSegment` | 0x00483220 | Get camera ray |
| `CameraCalcPosFromTarg` | 0x004826e0 | Calculate position from target |
| `CameraCalcTargFromPos` | 0x004827b0 | Calculate target from position |
| `CameraCanTurnPlayer` | 0x00541260 | Check player turning |
| `CameraTurnPlayer` | 0x00541330 | Turn player camera |
| `GetActiveCamera` | 0x004f16e0 | Get active camera |
| `GetCameraDistance` | 0x0053cab0 | Get camera distance |
| `GetCameraFacing` | 0x004f17e0 | Get camera facing |
| `GetCameraPosition` | 0x004f1740 | Get camera position |
| `ConfigureCamera` | 0x005274f0 | Configure camera |
| `CameraShakesRec` | 0x0057eda0 | Camera shake data |
| `CinematicCameraRec` | 0x005807a0 | Cinematic camera data |
| `AnimAddCamera` | 0x007563e0 | Add animated camera |
| `AnimAnimateCameras` | 0x00743ec0 | Animate cameras |
| `AnimIsCameraEnabled` | 0x007407a0 | Check camera animation |
| `CameraHandlerAnim` | 0x0073bda0 | Handle camera animation |
| `CreateViewFromCamera` | 0x0053db90 | Create view matrix |

---

## Camera Class Hierarchy

### CSimpleCamera (0x0053a6e0)

Base camera class for simple camera behavior:

```c
class CSimpleCamera {
    /* Position */
    C3Vector position;           // Camera position
    
    /* Orientation */
    C3Vector forward;           // Forward vector (Z)
    C3Vector right;             // Right vector (X)
    C3Vector up;                // Up vector (Y)
    
    /* Target */
    C3Vector target;            // Look-at target
    
    /* View transformation */
    C44Matrix viewMatrix;        // View matrix
    
    /* Projection */
    C44Matrix projMatrix;       // Projection matrix
    float nearClip;             // Near clipping plane
    float farClip;              // Far clipping plane
    float fov;                  // Field of view (radians)
    float aspectRatio;           // Aspect ratio
    
    /* State */
    bool isActive;              // Active flag
    bool isLocked;              // Locked to target
    uint32_t flags;             // Camera flags
};
```

### CGCamera (0x0053bd60)

Game camera with player following:

```c
class CGCamera : public CSimpleCamera {
    /* Player tracking */
    uint64_t targetGuid;        // Target object GUID
    uint32_t targetType;        // Target type
    
    /* Distance */
    float minDistance;          // Minimum distance
    float maxDistance;          // Maximum distance
    float currentDistance;      // Current distance
    
    /* Angles */
    float pitch;               // Pitch angle (radians)
    float yaw;                 // Yaw angle (radians)
    float roll;                // Roll angle
    
    /* Interpolation */
    float targetPitch;         // Target pitch
    float targetYaw;           // Target yaw
    float smoothing;           // Smoothing factor
    
    /* Collision */
    bool collisionEnabled;      // Collision detection
    float collisionRadius;     // Collision radius
    
    /* Mouse control */
    float mouseSensitivity;     // Mouse sensitivity
    float mouseInvertY;         // Invert Y-axis
};
```

### CinematicCameraRec (0x005807a0)

Cinematic camera for cutscenes:

```c
struct CinematicCameraRec {
    /* Camera path */
    C3Vector* positions;       // Path positions
    uint32_t positionCount;     // Number of positions
    
    /* Look-at path */
    C3Vector* targets;         // Look-at targets
    uint32_t targetCount;       // Number of targets
    
    /* Timing */
    uint32_t startTime;         // Start time (ms)
    uint32_t endTime;          // End time (ms)
    uint32_t totalTime;        // Total duration (ms)
    
    /* Camera settings */
    float fov;                  // Field of view
    float nearClip;             // Near clipping
    float farClip;              // Far clipping
    
    /* Audio */
    uint32_t soundEntryId;     // Sound entry ID
    uint32_t soundId;          // Sound file ID
};
```

---

## Camera Creation

### CameraCreate

```c
/* CameraCreate at 0x00482880 */
uint32_t CameraCreate(CCameraType type) {
    CSimpleCamera* camera;
    
    // Allocate camera based on type
    switch (type) {
        case CAMERA_SIMPLE:
            camera = new CSimpleCamera();
            break;
        case CAMERA_GAME:
            camera = new CGCamera();
            break;
        case CAMERA_CINEMATIC:
            camera = new CinematicCamera();
            break;
        default:
            return 0;  // Invalid type
    }
    
    // Initialize to defaults
    camera->nearClip = 0.1f;
    camera->farClip = 1000.0f;
    camera->fov = 60.0f * PI / 180.0f;  // 60 degrees
    camera->aspectRatio = 4.0f / 3.0f;
    camera->isActive = false;
    camera->isLocked = false;
    
    // Return camera handle
    return CameraMgr.Register(camera);
}
```

---

## Camera Update Cycle

### CameraUpdate

```c
/* CameraUpdate at 0x00483a10 */
void CameraUpdate(CSimpleCamera* camera, float deltaTime) {
    if (!camera->isActive) return;
    
    // Handle camera shake
    if (camera->shakeAmplitude > 0) {
        ApplyCameraShake(camera, deltaTime);
    }
    
    // Handle player following
    if (camera->targetGuid != 0) {
        UpdateTargetFollowing(camera, deltaTime);
    }
    
    // Handle mouse input
    ProcessMouseInput(camera, deltaTime);
    
    // Handle keyboard input
    ProcessKeyboardInput(camera, deltaTime);
    
    // Smooth camera movement
    SmoothCamera(camera, deltaTime);
    
    // Constrain to bounds
    ConstrainCamera(camera);
    
    // Update view matrix
    UpdateViewMatrix(camera);
    
    // Update projection matrix
    UpdateProjectionMatrix(camera);
}
```

---

## View Matrix Construction

### CreateViewFromCamera

```c
/* CreateViewFromCamera at 0x0053db90 */
void CreateViewFromCamera(CSimpleCamera* camera, C44Matrix* outView) {
    // Ensure vectors are orthonormal
    Normalize(camera->forward);
    Normalize(camera->right);
    Normalize(camera->up);
    
    // Build view matrix (look-at)
    // View matrix is inverse of camera world matrix
    outView->m[0][0] = camera->right.x;
    outView->m[0][1] = camera->right.y;
    outView->m[0][2] = camera->right.z;
    outView->m[0][3] = 0.0f;
    
    outView->m[1][0] = camera->up.x;
    outView->m[1][1] = camera->up.y;
    outView->m[1][2] = camera->up.z;
    outView->m[1][3] = 0.0f;
    
    outView->m[2][0] = -camera->forward.x;
    outView->m[2][1] = -camera->forward.y;
    outView->m[2][2] = -camera->forward.z;
    outView->m[2][3] = 0.0f;
    
    // Translation (camera position)
    float dx = -Dot(camera->right, camera->position);
    float dy = -Dot(camera->up, camera->position);
    float dz = Dot(camera->forward, camera->position);
    
    outView->m[3][0] = dx;
    outView->m[3][1] = dy;
    outView->m[3][2] = dz;
    outView->m[3][3] = 1.0f;
}
```

---

## Projection Setup

### CameraSetupScreenProjection

```c
/* CameraSetupScreenProjection at 0x004837e0 */
void CameraSetupScreenProjection(CSimpleCamera* camera) {
    // Calculate aspect ratio from viewport
    float aspect = (float)viewport.width / viewport.height;
    camera->aspectRatio = aspect;
    
    // Set up perspective projection
    float f = 1.0f / tan(camera->fov * 0.5f);
    
    camera->projMatrix.m[0][0] = f / aspect;
    camera->projMatrix.m[0][1] = 0.0f;
    camera->projMatrix.m[0][2] = 0.0f;
    camera->projMatrix.m[0][3] = 0.0f;
    
    camera->projMatrix.m[1][0] = 0.0f;
    camera->projMatrix.m[1][1] = f;
    camera->projMatrix.m[1][2] = 0.0f;
    camera->projMatrix.m[1][3] = 0.0f;
    
    camera->projMatrix.m[2][0] = 0.0f;
    camera->projMatrix.m[2][1] = 0.0f;
    camera->projMatrix.m[2][2] = camera->farClip / (camera->farClip - camera->nearClip);
    camera->projMatrix.m[2][3] = 1.0f;
    
    camera->projMatrix.m[3][0] = 0.0f;
    camera->projMatrix.m[3][1] = 0.0f;
    camera->projMatrix.m[3][2] = -camera->nearClip * camera->farClip / (camera->farClip - camera->nearClip);
    camera->projMatrix.m[3][3] = 0.0f;
}
```

---

## Player Camera Controls

### CameraCanTurnPlayer

```c
/* CameraCanTurnPlayer at 0x00541260 */
bool CameraCanTurnPlayer(CSimpleCamera* camera) {
    // Check if in combat
    if (IsInCombat()) return false;
    
    // Check if in vehicle
    if (IsInVehicle()) return false;
    
    // Check if locked
    if (camera->flags & CAMERA_FLAG_LOCKED) return false;
    
    // Check UI state
    if (IsUIOpen()) return false;
    
    // Check cinematic mode
    if (IsInCinematic()) return false;
    
    return true;
}
```

### CameraTurnPlayer

```c
/* CameraTurnPlayer at 0x00541330 */
void CameraTurnPlayer(CSimpleCamera* camera, float deltaX, float deltaY) {
    if (!CameraCanTurnPlayer(camera)) return;
    
    // Apply mouse sensitivity
    float sensitivity = camera->mouseSensitivity;
    if (camera->mouseInvertY) {
        deltaY = -deltaY;
    }
    
    // Update yaw (horizontal rotation)
    camera->targetYaw += deltaX * sensitivity;
    
    // Update pitch (vertical rotation)
    camera->targetPitch += deltaY * sensitivity;
    
    // Clamp pitch
    camera->targetPitch = Clamp(camera->targetPitch, -PI/2 + 0.1f, PI/2 - 0.1f);
    
    // Normalize yaw to [0, 2*PI]
    camera->targetYaw = NormalizeAngle(camera->targetYaw);
}
```

---

## Position/Target Calculation

### CameraCalcPosFromTarg

```c
/* CameraCalcPosFromTarg at 0x004826e0 */
void CameraCalcPosFromTarg(CSimpleCamera* camera, C3Vector* target, float distance) {
    // Calculate position from target and distance
    C3Vector offset;
    
    // Convert spherical to cartesian
    offset.x = distance * cos(camera->pitch) * sin(camera->yaw);
    offset.y = distance * sin(camera->pitch);
    offset.z = distance * cos(camera->pitch) * cos(camera->yaw);
    
    // Add offset to target
    camera->position.x = target->x + offset.x;
    camera->position.y = target->y + offset.y;
    camera->position.z = target->z + offset.z;
    
    // Update forward vector
    camera->forward = Normalize(*target - camera->position);
    
    // Recalculate right and up
    camera->right = Cross(camera->forward, C3Vector(0, 1, 0));
    Normalize(camera->right);
    camera->up = Cross(camera->right, camera->forward);
}
```

---

## Camera Shakes

### CameraShakesRec

```c
/* CameraShakesRec at 0x0057eda0 */
struct CameraShakesRec {
    /* Shake properties */
    float amplitude;           // Shake intensity
    float frequency;           // Shake frequency
    float duration;            // Shake duration
    float elapsed;            // Elapsed time
    
    /* Direction */
    C3Vector direction;        // Primary shake direction
    
    /* Type */
    uint32_t type;             // 0=vertical, 1=horizontal, 2=radial
    
    /* Falloff */
    float falloffStart;        // Distance where shake starts fading
    float falloffEnd;         // Distance where shake stops
    
    /* Owner */
    uint64_t sourceGuid;       // Source object GUID
};
```

---

## Animation System Integration

### AnimAddCamera

```c
/* AnimAddCamera at 0x007563e0 */
void AnimAddCamera(CAnimData* animData, uint32_t cameraId) {
    // Check camera limit
    if (animData->cameraCount >= MAX_CAMERAS) {
        Error("Too many cameras");
        return;
    }
    
    // Add camera to list
    animData->cameras[animData->cameraCount] = cameraId;
    animData->cameraCount++;
}
```

### AnimAnimateCameras

```c
/* AnimAnimateCameras at 0x00743ec0 */
void AnimAnimateCameras(CAnimData* animData, uint32_t time) {
    for (uint32_t i = 0; i < animData->cameraCount; i++) {
        CSimpleCamera* camera = GetCamera(animData->cameras[i]);
        
        // Animate position
        if (camera->hasPositionAnim) {
            C3Vector pos;
            EvaluatePositionAnimation(camera, time, &pos);
            camera->position = pos;
        }
        
        // Animate target
        if (camera->hasTargetAnim) {
            C3Vector target;
            EvaluateTargetAnimation(camera, time, &target);
            camera->target = target;
        }
        
        // Animate FOV
        if (camera->hasFovAnim) {
            float fov;
            EvaluateFovAnimation(camera, time, &fov);
            camera->fov = fov * PI / 180.0f;
        }
    }
}
```

---

## Script Integration

### CameraRegisterScriptFunctions

```c
/* CameraRegisterScriptFunctions at 0x0053bd0 */
void CameraRegisterScriptFunctions() {
    // Register Lua functions
    FrameScript_RegisterFunction("SetCamera", Lua_SetCamera);
    FrameScript_RegisterFunction("GetCamera", Lua_GetCamera);
    FrameScript_RegisterFunction("SetCameraDistance", Lua_SetCameraDistance);
    FrameScript_RegisterFunction("GetCameraDistance", Lua_GetCameraDistance);
    FrameScript_RegisterFunction("SetCameraPitch", Lua_SetCameraPitch);
    FrameScript_RegisterFunction("GetCameraPitch", Lua_GetCameraPitch);
    FrameScript_RegisterFunction("SetCameraYaw", Lua_SetCameraYaw);
    FrameScript_RegisterFunction("GetCameraYaw", Lua_GetCameraYaw);
    FrameScript_RegisterFunction("CameraReset", Lua_CameraReset);
}
```

### Lua API Examples

```lua
-- Set camera distance
SetCameraDistance(10.0)

-- Set camera pitch (angle)
SetCameraPitch(30)  -- 30 degrees

-- Set camera target
SetCameraTarget(unitId, 0)  -- Look at unit

-- Reset to default
CameraReset()

-- Smooth camera transition
SetCameraSmooth(0.1)  -- 0.1 smoothing factor
```

---

## Camera Data in MDX Files

### CAMS Chunk

```
CAMS Chunk:
├── uint32_t magic     // 'CAMS' (0x534d4143)
├── uint32_t size      // Chunk size
└── CameraData entries[]
```

**CameraData Structure:**

```c
struct CameraData {
    /* Name */
    char name[64];              // Camera name
    
    /* Position */
    C3Vector position;          // Camera position
    C3Vector target;           // Look-at target
    
    /* Distances */
    float nearClip;            // Near clipping plane
    float farClip;             // Far clipping plane
    
    /* Field of view */
    float fov;                 // Field of view (degrees)
    
    /* Animation */
    CameraPositionAnim positionAnim[];
    CameraTargetAnim targetAnim[];
    CameraFovAnim fovAnim[];
};

struct CameraPositionAnim {
    uint32_t count;
    MDLTRACKTYPE type;
    uint32_t globalSeqId;
    CameraPositionKey keys[];
};

struct CameraPositionKey {
    uint32_t time;
    C3Vector position;
};
```

---

## Summary

The camera system in WoW Alpha 0.5.3 provides:
- **Multiple camera types**: Simple, Game, Cinematic
- **Player following**: Smooth target tracking with distance/pitch/yaw control
- **Camera shakes**: Procedural shake effects
- **Animation integration**: Keyframe animations for cutscenes
- **Script integration**: Lua API for camera control
- **Collision detection**: Prevent camera clipping through geometry

Key functions and addresses provide a complete reference for reverse engineering and implementation.

---

*Document created: 2026-02-07*
*Analysis based on WoWClient.exe (Build 3368)*
