# MLDB (Map Low Detail Blend)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MLDB chunk was introduced in Battle for Azeroth (BfA) and contains configuration data for blending between different level-of-detail (LOD) representations of terrain. It controls how and when terrain transitions between high-detail and low-detail models based on distance from the camera, ensuring smooth visual transitions while maximizing performance.

## Structure

```csharp
public struct MLDB
{
    public float[] BlendDistances;    // Array of blend distances for LOD transitions
    public float[] BlendRanges;       // Array of blend ranges over which transitions occur
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| BlendDistances | float[] | Array of distances at which LOD transitions begin (in game units) |
| BlendRanges | float[] | Array of distance ranges over which blending between LOD levels occurs |

## Dependencies

- **MHDR (C001)** - Contains flags that may affect LOD behavior
- **MCNK (C009)** - Contains the high-detail terrain chunks that blend with lower detail representations

## Implementation Notes

- The MLDB chunk was introduced to improve LOD transitions in Battle for Azeroth
- The arrays contain one entry per LOD level transition (typically 3-4 levels are used)
- Distances are specified in game world units (yards)
- A higher BlendDistance value means the transition happens further from the camera
- A higher BlendRange value creates a more gradual, less noticeable transition
- The chunk may not be present in all ADT files, particularly in older content
- Implementation should fall back to reasonable default values if the chunk is missing

## Implementation Example

```csharp
public class TerrainLODManager
{
    private float[] lodBlendDistances;
    private float[] lodBlendRanges;
    private readonly float[] defaultDistances = new float[] { 150.0f, 300.0f, 600.0f };
    private readonly float[] defaultRanges = new float[] { 50.0f, 100.0f, 200.0f };
    
    // Load LOD blend settings from MLDB chunk
    public void LoadLODBlendSettings(MLDB mldbChunk = null)
    {
        if (mldbChunk != null && mldbChunk.BlendDistances.Length > 0)
        {
            lodBlendDistances = mldbChunk.BlendDistances;
            lodBlendRanges = mldbChunk.BlendRanges;
        }
        else
        {
            // Use default values if MLDB chunk is missing
            lodBlendDistances = defaultDistances;
            lodBlendRanges = defaultRanges;
        }
    }
    
    // Calculate LOD and blend factor based on distance from camera
    public (int lodLevel, float blendFactor) CalculateLODLevel(float distanceFromCamera)
    {
        for (int i = 0; i < lodBlendDistances.Length; i++)
        {
            float blendStart = lodBlendDistances[i];
            float blendRange = lodBlendRanges[i];
            float blendEnd = blendStart + blendRange;
            
            if (distanceFromCamera < blendStart)
            {
                // Before this LOD transition, use previous LOD level
                return (i, 0.0f);
            }
            else if (distanceFromCamera < blendEnd)
            {
                // In the middle of this LOD transition, calculate blend factor
                float blendFactor = (distanceFromCamera - blendStart) / blendRange;
                return (i, blendFactor);
            }
        }
        
        // Beyond all transition distances, use the highest LOD level
        return (lodBlendDistances.Length, 1.0f);
    }
    
    // Apply LOD blending to terrain material
    public void ApplyLODBlending(Material terrainMaterial, float distanceFromCamera)
    {
        var (lodLevel, blendFactor) = CalculateLODLevel(distanceFromCamera);
        
        // Set LOD level and blend factor in shader
        terrainMaterial.SetInt("_LODLevel", lodLevel);
        terrainMaterial.SetFloat("_LODBlendFactor", blendFactor);
    }
}

// Example shader function for LOD blending
public static class ShaderSnippet
{
    public static string GetLODBlendingFunction()
    {
        return @"
        // LOD transition blending
        float4 ApplyLODBlending(float4 highDetailColor, float4 lowDetailColor, float blendFactor)
        {
            return lerp(highDetailColor, lowDetailColor, blendFactor);
        }
        
        // Usage in fragment shader:
        // float4 highDetail = SampleHighDetailTexture(uv);
        // float4 lowDetail = SampleLowDetailTexture(uv);
        // float4 finalColor = ApplyLODBlending(highDetail, lowDetail, _LODBlendFactor);
        ";
    }
}
```

## Usage Context

The MLDB chunk plays a crucial role in World of Warcraft's level-of-detail (LOD) system, which is essential for maintaining good performance while rendering large outdoor environments. As players move through the world, distant terrain needs to be represented with simplified geometry and textures to conserve resources, but these transitions need to be visually seamless.

Prior to Battle for Azeroth, LOD transitions in World of Warcraft could sometimes be visually jarring, with noticeable "popping" as terrain switched between detail levels. The MLDB chunk addresses this issue by providing configuration data for smooth, gradual blending between LOD levels.

In practice, the MLDB chunk's data is used to:

1. **Determine Transition Points**: Define at what distances from the camera LOD transitions should begin
2. **Control Transition Smoothness**: Specify how gradually the transition between LOD levels should occur
3. **Optimize Performance**: Allow for area-specific LOD settings based on visual importance and complexity
4. **Balance Visual Quality**: Ensure that visual fidelity is preserved where it matters most

The implementation of this chunk reflects Blizzard's ongoing efforts to improve both performance and visual quality simultaneously. In densely detailed zones introduced in Battle for Azeroth and beyond, these improved LOD transitions help maintain the illusion of a seamless, continuous world while ensuring the game performs well across a wide range of hardware configurations.

For developers implementing terrain rendering systems based on World of Warcraft's format, the MLDB chunk provides valuable configuration data to guide LOD implementation, particularly for view-dependent level of detail systems with distance-based blending. 