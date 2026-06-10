# Shader and Rendering System

This document provides detailed analysis of the shader and rendering system in WoW Alpha 0.5.3, based on reverse engineering the WoWClient.exe binary.

## Overview

The rendering system in WoW Alpha uses a hybrid approach combining fixed-function pipeline with programmable shaders. The system supports multiple blend modes for transparency, texture-based effects, and hardware-accelerated rendering via DirectX 9.

## Key Classes and Structures

### CGxCaps (Graphics Capabilities)
```
Purpose: Queries GPU capabilities and limits
Fields:
  - uint m_texFmtDxt: DXT texture format support (bitmask)
  - uint m_maxTextureWidth: Maximum texture width
  - uint m_maxTextureHeight: Maximum texture height
  - uint m_numTextureStages: Number of texture sampling stages
  - bool m_supportsMipmaps: Mipmap support
  - bool m_supportsNonPowerOf2: Non-power-of-2 texture support
```

### CGxPixelShader (Pixel Shader)
```
Address: 0x00597830+
Purpose: Handles pixel-level shading operations
Created by: PixelShaderCreate() @ 0x00594e90
Fields:
  - uint m_shaderId: Shader identifier
  - TSGrowableArray<uint> m_shaderData: Compiled shader bytecode
  - uint m_flags: Shader configuration flags
```

### CGxVertexShader (Vertex Shader)
```
Address: 0x005978c0+
Purpose: Handles vertex transformation and lighting
Created by: VertexShaderCreate() @ 0x00594bf0
Fields:
  - uint m_shaderId: Shader identifier
  - TSGrowableArray<uint> m_shaderData: Compiled shader bytecode
  - uint m_flags: Shader configuration flags
```

### CGxShaderParam (Shader Parameter)
```
Purpose: Holds shader uniform/constant values
Fields:
  - uint m_paramId: Parameter identifier
  - uint m_paramType: Parameter type (float, int, sampler, etc.)
  - void* m_paramData: Parameter value data
  - uint m_paramSize: Parameter data size in bytes
```

## Blend Mode Types (EGxBlend)

| Value | Name | Description | Blend Equation |
|-------|------|-------------|----------------|
| 0 | GxBlend_Opaque | No blending | None |
| 1 | GxBlend_Blend | Standard alpha blending | SrcAlpha, InvSrcAlpha |
| 2 | GxBlend_Add | Additive blending | SrcAlpha, One |
| 3 | GxBlend_AlphaKey | Alpha testing | Discard if alpha < threshold |

### StringToBlendMode @ 0x006d73b0
```c
int StringToBlendMode(char* blendString, EGxBlend* outBlend)
{
  // Mapping table:
  // "DISABLE" = 0 (GxBlend_Opaque)
  // "BLEND" = 1 (GxBlend_Blend)
  // "ALPHAKEY" = 3 (GxBlend_AlphaKey)
  // "ADD" = 2 (GxBlend_Add)
  
  if (SStrCmpI(blendString, "DISABLE", -1) == 0) {
    *outBlend = GxBlend_Opaque;
    return 1;
  }
  if (SStrCmpI(blendString, "BLEND", -1) == 0) {
    *outBlend = GxBlend_Blend;
    return 1;
  }
  if (SStrCmpI(blendString, "ALPHAKEY", -1) == 0) {
    *outBlend = GxBlend_AlphaKey;
    return 1;
  }
  if (SStrCmpI(blendString, "ADD", -1) == 0) {
    *outBlend = GxBlend_Add;
    return 1;
  }
  return 0; // Unknown blend mode
}
```

## Blend Mode Application

### SetMaterialBlendMode @ 0x00448cb0
```c
void SetMaterialBlendMode(EGxBlend blendMode)
{
  switch (blendMode) {
    case GxBlend_Opaque:
      // Disable blending
      GxDisable(GX_BLEND);
      // Enable depth write
      GxEnable(GX_DEPTH_WRITE);
      break;
    
    case GxBlend_Blend:
      // Enable alpha blending
      GxEnable(GX_BLEND);
      GxBlendFunc(GX_BLEND_SRC_ALPHA, GX_BLEND_INV_SRC_ALPHA);
      // Disable depth write for proper transparency sorting
      GxDisable(GX_DEPTH_WRITE);
      break;
    
    case GxBlend_Add:
      // Enable additive blending
      GxEnable(GX_BLEND);
      GxBlendFunc(GX_BLEND_SRC_ALPHA, GX_BLEND_ONE);
      // Disable depth write
      GxDisable(GX_DEPTH_WRITE);
      break;
    
    case GxBlend_AlphaKey:
      // Enable blending but with alpha testing
      GxEnable(GX_BLEND);
      GxBlendFunc(GX_BLEND_SRC_ALPHA, GX_BLEND_INV_SRC_ALPHA);
      // Enable alpha test for binary alpha
      GxEnable(GX_ALPHATEST);
      GxAlphaFunc(GX_GREATER, 0.5f);
      // Enable depth write
      GxEnable(GX_DEPTH_WRITE);
      break;
  }
}
```

## Transparency Sorting

### ComputeFogBlend @ 0x00689b40
```c
void ComputeFogBlend(float depth, float* outFogFactor)
{
  // Calculate fog factor based on distance
  // Linear fog: (end - depth) / (end - start)
  // Exponential fog: exp(-density * depth)
  // Exponential squared fog: exp(-density * depth)^2
  
  float fogStart = GetFogStart();
  float fogEnd = GetFogEnd();
  float fogDensity = GetFogDensity();
  float fogFactor;
  
  if (fogDensity == 0) {
    // Linear fog
    fogFactor = (fogEnd - depth) / (fogEnd - fogStart);
  } else {
    // Exponential fog
    fogFactor = exp(-fogDensity * depth);
  }
  
  fogFactor = clamp(fogFactor, 0.0f, 1.0f);
  
  *outFogFactor = fogFactor;
}
```

## Model Blend Mode Functions

### ModelSetBlendMode @ 0x00440490
```c
void ModelSetBlendMode(HMODEL__* model, EGxBlend blendMode)
{
  // Iterate through all geosets
  CGeoset** geosets = ModelGetGeosets(model);
  int numGeosets = ModelGetGeosetCount(model);
  
  for (int i = 0; i < numGeosets; i++) {
    CGeoset* geoset = geosets[i];
    
    // Get material for geoset
    CMaterial* material = GeosetGetMaterial(geoset);
    
    // Apply blend mode to material
    MaterialSetBlendMode(material, blendMode);
    
    // Mark geoset as needing sort if transparent
    if (blendMode == GxBlend_Blend || blendMode == GxBlend_Add) {
      GeosetSetSortOffset(geoset, CalculateSortDepth(model, geoset));
    }
  }
}
```

### ComplexModelSetBlendMode @ 0x00440630
```c
void ComplexModelSetBlendMode(HCOMPLEXMODEL__* complexModel, EGxBlend blendMode)
{
  // Handle multiple models (e.g., creatures with armor sets)
  int numModels = ComplexModelGetModelCount(complexModel);
  
  for (int i = 0; i < numModels; i++) {
    HMODEL__* model = ComplexModelGetModel(complexModel, i);
    
    // Apply blend mode
    ModelSetBlendMode(model, blendMode);
    
    // Update render order
    ComplexModelUpdateRenderOrder(complexModel);
  }
}
```

## Shader Creation

### VertexShaderCreate @ 0x00594bf0
```c
HVERTEXSHADER__* VertexShaderCreate(
  char* shaderName,
  uint* shaderCode,
  uint codeSize
)
{
  // Create vertex shader object
  HVERTEXSHADER__* shader = (HVERTEXSHADER__*)malloc(sizeof(HVERTEXSHADER__));
  
  // Initialize shader structure
  shader->m_shaderId = GenerateShaderId();
  shader->m_shaderData = TSGrowableArray<uint>::Create(codeSize / 4);
  
  // Copy shader bytecode
  for (uint i = 0; i < codeSize / 4; i++) {
    TSGrowableArray<uint>::PushBack(&shader->m_shaderData, shaderCode[i]);
  }
  
  // Compile shader (if needed)
  if (!ShaderPrecompile(shader)) {
    free(shader);
    return nullptr;
  }
  
  // Register shader in global list
  AddShaderToGlobalList(shader);
  
  return shader;
}
```

### PixelShaderCreate @ 0x00594e90
```c
HPIXELSHADER__* PixelShaderCreate(
  char* shaderName,
  uint* shaderCode,
  uint codeSize
)
{
  // Similar to vertex shader creation
  // ...
}
```

### _D3DXCompileShaderFromFileA @ 0x006f1267
```c
ID3DXBuffer* _D3DXCompileShaderFromFileA(
  LPCSTR pSrcFile,
  CONST D3DXMACRO* pDefines,
  LPD3DXINCLUDE pInclude,
  LPCSTR pFunctionName,
  LPCSTR pTarget,
  DWORD Flags,
  LPD3DXBUFFER* ppShader,
  LPD3DXBUFFER* ppErrorMsgs,
  LPD3DXCONSTANTTABLE* ppConstantTable
)
{
  // Load shader source from file
  FILE* fp = fopen(pSrcFile, "rb");
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  
  char* source = (char*)malloc(size + 1);
  fread(source, 1, size, fp);
  source[size] = '\0';
  fclose(fp);
  
  // Compile shader
  return D3DXCompileShader(
    source,
    size,
    pDefines,
    pInclude,
    pFunctionName,
    pTarget,
    Flags,
    ppShader,
    ppErrorMsgs,
    ppConstantTable
  );
}
```

## Texture Shader Binding

### UpdateShaderGxTexture @ 0x0069a970
```c
void UpdateShaderGxTexture(
  HSHADER__* shader,
  int textureStage,
  HTEXTURE__* texture
)
{
  // Bind texture to shader stage
  GxSetTexture(textureStage, texture->GetTextureId());
  
  // Set texture addressing modes
  GxSetTextureAddressU(textureStage, GX_TEXADDRESS_WRAP);
  GxSetTextureAddressV(textureStage, GX_TEXADDRESS_WRAP);
  
  // Set texture filtering
  GxSetTextureFilter(textureStage, GX_TEXF_LINEAR);
  GxSetTextureFilterMin(textureStage, GX_TEXF_LINEAR_MIPMAP_LINEAR);
  GxSetTextureFilterMag(textureStage, GX_TEXF_LINEAR);
}
```

### GetTextureShader @ 0x0044c8b0
```c
HSHADER__* GetTextureShader(HTEXTURE__* texture)
{
  // Determine appropriate shader based on texture properties
  uint texFlags = TextureGetFlags(texture);
  EGxBlend blendMode = TextureGetBlendMode(texture);
  
  // Select shader based on blend mode
  switch (blendMode) {
    case GxBlend_Opaque:
      return GetOpaqueTextureShader();
    
    case GxBlend_Blend:
      return GetAlphaBlendedTextureShader();
    
    case GxBlend_Add:
      return GetAdditiveTextureShader();
    
    case GxBlend_AlphaKey:
      return GetAlphaTestedTextureShader();
    
    default:
      return GetDefaultTextureShader();
  }
}
```

## Material System

### CMaterial (Material Properties)
```
Purpose: Holds rendering properties for geometry
Fields:
  - EGxBlend blendMode: Transparency mode
  - HTEXTURE__* texture: Primary texture
  - HTEXTURE__* texture2: Secondary texture (detail, etc.)
  - C4Vector emissiveColor: Emissive color
  - C4Vector diffuseColor: Diffuse color
  - C4Vector specularColor: Specular color
  - float specularPower: Shininess exponent
  - float opacity: Overall opacity (0-1)
  - uint renderFlags: Additional rendering flags
```

## Implementation Recommendations for MdxViewer

### 1. Blend Mode Enumeration
```csharp
public enum BlendMode
{
    /// <summary>No blending - opaque rendering with depth write</summary>
    Opaque = 0,
    
    /// <summary>Standard alpha blending - requires sorting</summary>
    Blend = 1,
    
    /// <summary>Additive blending - adds color to framebuffer</summary>
    Add = 2,
    
    /// <summary>Alpha testing - discards pixels below alpha threshold</summary>
    AlphaKey = 3
}
```

### 2. Blend State Manager
```csharp
public class BlendStateManager
{
    private Dictionary<BlendMode, BlendState> _states;
    
    public void ApplyBlendMode(BlendMode mode)
    {
        switch (mode) {
            case BlendMode.Opaque:
                GL.Disable(Enable.Blend);
                GL.Enable(Enable.DepthTest);
                GL.DepthMask(true);
                break;
            
            case BlendMode.Blend:
                GL.Enable(Enable.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                GL.Enable(Enable.DepthTest);
                GL.DepthMask(false); // Don't write to depth buffer
                break;
            
            case BlendMode.Add:
                GL.Enable(Enable.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                GL.Enable(Enable.DepthTest);
                GL.DepthMask(false);
                break;
            
            case BlendMode.AlphaKey:
                GL.Enable(Enable.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                GL.Enable(Enable.AlphaTest);
                GL.AlphaFunc(AlphaFunction.Greater, 0.5f);
                GL.Enable(Enable.DepthTest);
                GL.DepthMask(true);
                break;
        }
    }
}
```

### 3. Transparency Sorting System
```csharp
public class TransparencySorter
{
    private List<RenderItem> _transparentItems = new();
    
    public void AddTransparentItem(
        Mesh mesh,
        Matrix4x4 transform,
        BlendMode blendMode,
        float depth
    )
    {
        _transparentItems.Add(new RenderItem {
            Mesh = mesh,
            Transform = transform,
            BlendMode = blendMode,
            Depth = depth
        });
    }
    
    public void RenderSorted(Camera camera)
    {
        // Sort back-to-front
        var sorted = _transparentItems
            .OrderByDescending(item => Vector3.Transform(
                item.Mesh.BoundingBox.Center,
                item.Transform
            ).Z) // Or use camera distance
            .ToList();
        
        foreach (var item in sorted) {
            // Apply blend mode
            _blendStates.ApplyBlendMode(item.BlendMode);
            
            // Render mesh
            RenderMesh(item.Mesh, item.Transform);
        }
        
        _transparentItems.Clear();
    }
}
```

### 4. Material Class
```csharp
public class Material
{
    public BlendMode BlendMode { get; set; } = BlendMode.Opaque;
    public Texture? DiffuseTexture { get; set; }
    public Texture? EmissiveTexture { get; set; }
    public Vector4 DiffuseColor { get; set; } = Vector4.One;
    public Vector4 EmissiveColor { get; set; } = Vector4.Zero;
    public float Opacity { get; set; } = 1.0f;
    public bool TwoSided { get; set; } = false;
    public bool DepthWrite { get; set; } = true;
    
    public void Apply()
    {
        // Apply opacity to blend mode
        var effectiveBlendMode = BlendMode;
        if (Opacity < 1.0f && BlendMode == BlendMode.Opaque) {
            effectiveBlendMode = BlendMode.Blend;
        }
        
        _blendStates.ApplyBlendMode(effectiveBlendMode);
        
        // Set textures
        if (DiffuseTexture != null) {
            GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2D, DiffuseTexture.Handle);
        }
        
        // Set colors
        GL.Material(MaterialFace.FrontAndBack, MaterialParameter.Diffuse, DiffuseColor);
        GL.Material(MaterialFace.FrontAndBack, MaterialParameter.Emission, EmissiveColor);
        
        // Set culling
        GL.FrontFace(CullFaceMode.CCW);
        if (TwoSided) {
            GL.Disable(Enable.CullFace);
        } else {
            GL.Enable(Enable.CullFace);
        }
    }
}
```

### 5. Fog Calculation
```csharp
public class FogSystem
{
    public FogMode Mode { get; set; } = FogMode.Linear;
    public float Start { get; set; } = 0.0f;
    public float End { get; set; } = 100.0f;
    public float Density { get; set; } = 0.01f;
    public Vector4 Color { get; set; } = Vector4.Zero;
    
    public float CalculateFogFactor(float distance)
    {
        float factor;
        
        switch (Mode) {
            case FogMode.Linear:
                factor = (End - distance) / (End - Start);
                break;
            
            case FogMode.Exp:
                factor = (float)Math.Exp(-Density * distance);
                break;
            
            case FogMode.Exp2:
                factor = (float)Math.Exp(-Density * distance * Density * distance);
                break;
            
            default:
                factor = 1.0f;
                break;
        }
        
        return Math.Clamp(factor, 0.0f, 1.0f);
    }
    
    public void ApplyFog(float distance)
    {
        if (Mode == FogMode.None) return;
        
        float fogFactor = CalculateFogFactor(distance);
        
        // Apply fog color
        GL.Fog(FogParameter.FogColor, Color);
        GL.Fog(FogParameter.FogMode, (int)Mode);
        GL.Fog(FogParameter.FogStart, Start);
        GL.Fog(FogParameter.FogEnd, End);
        GL.Fog(FogParameter.FogDensity, Density);
        
        GL.Enable(Enable.Fog);
    }
}
```

### 6. Shader Program
```csharp
public class ShaderProgram
{
    private uint _vertexShader;
    private uint _fragmentShader;
    private uint _program;
    
    private Dictionary<string, int> _uniforms;
    
    public bool CompileFromSource(
        string vertexSource,
        string fragmentSource
    )
    {
        // Compile vertex shader
        _vertexShader = GL.CreateShader(ShaderType.VertexShader);
        GL.ShaderSource(_vertexShader, vertexSource);
        GL.CompileShader(_vertexShader);
        
        if (!GetShaderCompileStatus(_vertexShader)) {
            return false;
        }
        
        // Compile fragment shader
        _fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
        GL.ShaderSource(_fragmentShader, fragmentSource);
        GL.CompileShader(_fragmentShader);
        
        if (!GetShaderCompileStatus(_fragmentShader)) {
            return false;
        }
        
        // Link program
        _program = GL.CreateProgram();
        GL.AttachShader(_program, _vertexShader);
        GL.AttachShader(_program, _fragmentShader);
        GL.LinkProgram(_program);
        
        if (!GetProgramLinkStatus(_program)) {
            return false;
        }
        
        // Get uniform locations
        _uniforms = new Dictionary<string, int>();
        int uniformCount;
        GL.GetProgram(_program, ProgramParameterName.ActiveUniforms, out uniformCount);
        
        for (int i = 0; i < uniformCount; i++) {
            string name = GL.GetActiveUniform(_program, i, out _, out _);
            int location = GL.GetUniformLocation(_program, name);
            _uniforms[name] = location;
        }
        
        return true;
    }
    
    public void Use()
    {
        GL.UseProgram(_program);
    }
    
    public void SetInt(string name, int value)
    {
        if (_uniforms.TryGetValue(name, out int location)) {
            GL.Uniform1(location, value);
        }
    }
    
    public void SetFloat(string name, float value)
    {
        if (_uniforms.TryGetValue(name, out int location)) {
            GL.Uniform1(location, value);
        }
    }
    
    public void SetVector4(string name, Vector4 value)
    {
        if (_uniforms.TryGetValue(name, out int location)) {
            GL.Uniform4(location, value.X, value.Y, value.Z, value.W);
        }
    }
}
```

## Key Functions Reference

| Function | Address | Purpose |
|----------|---------|---------|
| StringToBlendMode | 0x006d73b0 | Convert blend mode string to enum |
| SetMaterialBlendMode | 0x00448cb0 | Set material blend mode |
| ModelSetBlendMode | 0x00440490 | Set model blend mode |
| ComplexModelSetBlendMode | 0x00440630 | Set complex model blend mode |
| ComputeFogBlend | 0x00689b40 | Calculate fog blend factor |
| VertexShaderCreate | 0x00594bf0 | Create vertex shader |
| PixelShaderCreate | 0x00594e90 | Create pixel shader |
| _D3DXCompileShaderFromFileA | 0x006f1267 | Compile shader from file |
| UpdateShaderGxTexture | 0x0069a970 | Update shader texture binding |
| GetTextureShader | 0x0044c8b0 | Get appropriate shader for texture |

## Debugging Tips

1. **Transparency not working?** Check:
   - Blend mode is set correctly
   - Depth write is disabled for transparent objects
   - Objects are sorted back-to-front

2. **Alpha test not working?** Check:
   - Alpha test is enabled
   - Alpha test threshold is correct (usually 0.5f)
   - Alpha values are properly calculated

3. **Fog not appearing?** Check:
   - Fog is enabled
   - Fog color matches background
   - Fog start/end or density is appropriate for scene scale

4. **Shaders not compiling?** Check:
   - Shader source is valid
   - Target profile is correct (vs_1_1, ps_2_0, etc.)
   - Uniform types match shader expectations
