# WFX (Warcraft Effects) Format

## Overview
WFX files define shader configurations for various surface types in World of Warcraft. These files are hardcoded into the client and provide both fixed-function and programmable shader pipelines for different rendering scenarios.

## File Structure

### Effect Definition
```cpp
struct Effect {
    char name[256];     // Effect name (referenced by client)
    uint32_t flags;     // Effect flags
    FixedFunc* fixed;   // Fixed function pipeline
    Shader* shader;     // Programmable pipeline
};
```

### Fixed Function Pipeline
```cpp
struct FixedFunc {
    Pass passes[MAX_PASSES];  // Array of rendering passes
};

struct Pass {
    PassType type;           // Pass type enumeration
    uint32_t passCount;      // Must be ≤ 1
    ColorOp colorOp[2];      // Color operations
    AlphaOp alphaOp[2];      // Alpha operations
    RenderState* states;     // Render state array
};
```

### Shader Pipeline
```cpp
struct Shader {
    Pass passes[MAX_PASSES];  // Array of rendering passes
    char vertexShader[256];   // Vertex shader name
    char pixelShader[256];    // Pixel shader name
    bool specular;            // Specular lighting enable
};
```

## Enumerations

### Pass Types
```cpp
enum PassType {
    Default,
    Opaque,
    AlphaKey,
    Alpha,
    Add,
    Mod,
    Mod2x,
    ModAdd,
    InvSrcAlphaAdd,
    InvSrcAlphaOpaque,
    SrcAlphaOpaque,
    NoAlphaAdd,
    ConstantAlpha
};
```

### Operation Modes
```cpp
enum OpMode {
    Mod,
    Mod2x,
    Add,
    PassThru,
    Decal,
    Fade
};
```

### Render States
```cpp
enum RenderStateType {
    MatDiffuse,
    MatEmissive,
    MatSpecular,
    MatSpecularExp,
    NormalizeNormals,
    SceneAmbient,
    DepthTest,
    DepthFunc,
    DepthWrite,
    ColorWrite,
    Culling,
    ClipPlaneMask,
    Lighting,
    TexLodBias0,
    TexLodBias1,
    TexLodBias2,
    TexLodBias3,
    TexLodBias4,
    TexLodBias5,
    TexLodBias6,
    TexLodBias7,
    TexGen0,
    TexGen1,
    TexGen2,
    TexGen3,
    TexGen4,
    TexGen5,
    TexGen6,
    TexGen7,
    TextureShader0,
    TextureShader1,
    TextureShader2,
    TextureShader3,
    TextureShader4,
    TextureShader5,
    TextureShader6,
    TextureShader7,
    PointScale,
    PointScaleAttenuation,
    PointScaleMin,
    PointScaleMax,
    PointSprite,
    ConstBlendAlpha,
    Unknown
};
```

## Standard Effects

### MapObj.wfx
1. **Particle Effects**
   - Basic particle shaders
   - Particle blending modes
   - Particle lighting

2. **Map Object Shaders**
   - MapObjDiffuse
   - MapObjOpaque
   - MapObjSpecular
   - MapObjMetal
   - MapObjEnv
   - MapObjEnvMetal

### MapObjU.wfx
1. **Unlit Map Objects**
   - MapObjUDiffuse
   - MapObjUOpaque
   - MapObjUSpecular
   - MapObjUMetal
   - MapObjUEnv
   - MapObjUEnvMetal

### Model2.wfx
1. **Projection Effects**
   - Projected_ModMod
   - Projected_ModAdd

### ShadowMap.wfx
1. **Shadow Rendering**
   - ShadowMapRender
   - ShadowMapRenderSL

## Implementation Notes

### Effect Loading
1. **File Parsing**
   - Parse effect definitions
   - Load fixed function pipeline
   - Load shader pipeline
   - Validate pass counts

2. **State Management**
   - Track render states
   - Handle state transitions
   - Manage shader bindings
   - Control pipeline switches

### Render State Types
1. **Material States**
   - Diffuse color
   - Emissive color
   - Specular color
   - Specular power

2. **Texture States**
   - LOD bias
   - Texture coordinates
   - Texture shaders
   - Texture blending

3. **Rasterizer States**
   - Depth testing
   - Culling
   - Color writing
   - Alpha blending

### Best Practices
1. **Loading Strategy**
   - Validate effect names
   - Check pass limitations
   - Handle missing shaders
   - Support both pipelines

2. **Error Handling**
   - Validate state values
   - Check shader existence
   - Handle invalid passes
   - Provide fallbacks

3. **Performance**
   - Cache compiled shaders
   - Minimize state changes
   - Batch similar effects
   - Optimize transitions

### Usage Context
1. **Surface Types**
   - Map objects
   - Character models
   - Particle effects
   - Shadow rendering

2. **Rendering Scenarios**
   - Normal mapping
   - Environment mapping
   - Specular lighting
   - Alpha blending

### Integration Notes
1. **BLS Integration**
   - Reference BLS shaders
   - Match shader versions
   - Handle dependencies
   - Validate compatibility

2. **Client Integration**
   - Hardcoded references
   - Effect management
   - Resource handling
   - State tracking 