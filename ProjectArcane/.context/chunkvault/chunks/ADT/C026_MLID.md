# MLID (Map Height ID)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MLID chunk was introduced in Legion and contains FileDataIDs for height/displacement map textures used in terrain rendering. This chunk works alongside MDID, MNID, and MSID, replacing string-based texture references with direct numeric FileDataIDs. Height maps provide additional vertical detail to terrain, allowing for micro-displacement without increasing base geometry complexity.

## Structure

```csharp
public struct MLID
{
    public uint[] HeightMapFileDataIDs;  // Array of FileDataIDs for height/displacement map textures
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| HeightMapFileDataIDs | uint[] | Array of FileDataIDs for height map texture assets used in this ADT |

## Dependencies

- **MDID (C023)** - Contains FileDataIDs for corresponding diffuse textures
- **MNID (C024)** - Works with normal maps for coherent surface detail
- **MSID (C025)** - Often correlates with specular maps for consistent material properties
- **MCNK (C009)** - Uses these height maps for terrain chunk rendering
- **MCLY (S002)** - Contains texture layer definitions that reference these height maps

## Implementation Notes

- The MLID chunk was introduced alongside other FileDataID chunks in Legion
- Each entry corresponds positionally to an entry in the MDID chunk
- A value of 0 indicates no height map for that texture
- Height maps typically use grayscale values where:
  - White areas represent raised portions of the surface
  - Black areas represent indentations or flat portions
- The maps are used for displacement mapping or parallax occlusion mapping in shaders
- Height maps add micro-detail that would be too expensive to model with actual geometry
- The array size matches the number of diffuse textures in the MDID chunk
- Not all terrain textures have associated height maps

## Implementation Example

```csharp
public class HeightMapManager
{
    private Dictionary<uint, Texture2D> heightMapCache = new Dictionary<uint, Texture2D>();
    private FileDataService fileDataService;
    
    public HeightMapManager(FileDataService fileDataService)
    {
        this.fileDataService = fileDataService;
    }
    
    public void LoadHeightMaps(uint[] heightMapFileDataIDs)
    {
        foreach (uint fileDataID in heightMapFileDataIDs)
        {
            if (fileDataID == 0 || heightMapCache.ContainsKey(fileDataID))
                continue;
                
            try
            {
                byte[] fileData = fileDataService.GetFileData(fileDataID);
                
                // Create height map texture from file data
                Texture2D heightMap = CreateTextureFromFileData(fileData);
                if (heightMap != null)
                {
                    // Configure texture for height map usage
                    heightMap.SetTextureSettingsForHeightMap();
                    heightMapCache[fileDataID] = heightMap;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load height map {fileDataID}: {ex.Message}");
            }
        }
    }
    
    public Texture2D GetHeightMap(uint fileDataID)
    {
        if (fileDataID == 0 || !heightMapCache.ContainsKey(fileDataID))
            return null;
            
        return heightMapCache[fileDataID];
    }
    
    private Texture2D CreateTextureFromFileData(byte[] fileData)
    {
        // In a real implementation, this would decode the BLP format
        Texture2D texture = new Texture2D(2, 2);
        
        // This is a simplified placeholder - actual implementation would decode BLP format
        if (texture.LoadImage(fileData))
        {
            return texture;
        }
        
        return null;
    }
    
    // Apply height maps to terrain materials
    public void ApplyHeightMapsToMaterials(Material[] materials, uint[] diffuseIDs, uint[] heightIDs)
    {
        for (int i = 0; i < materials.Length; i++)
        {
            if (materials[i] == null)
                continue;
                
            // Apply height map if available
            if (i < heightIDs.Length && heightIDs[i] != 0)
            {
                Texture2D heightMap = GetHeightMap(heightIDs[i]);
                if (heightMap != null)
                {
                    materials[i].SetTexture("_ParallaxMap", heightMap);
                    materials[i].EnableKeyword("_PARALLAXMAP");
                    
                    // Set parallax/displacement strength
                    materials[i].SetFloat("_Parallax", 0.02f);  // Adjust based on desired effect
                }
            }
            else
            {
                // Disable parallax mapping if no height map is available
                materials[i].DisableKeyword("_PARALLAXMAP");
                materials[i].SetFloat("_Parallax", 0.0f);
            }
        }
    }
}

// Extension method for configuring textures as height maps
public static class HeightTextureExtensions
{
    public static void SetTextureSettingsForHeightMap(this Texture2D texture)
    {
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Bilinear;
        texture.anisoLevel = 4;  // Better anisotropic filtering for detailed surface displacement
    }
}

// Example shader implementation for parallax mapping using height maps
public static class ShaderSnippet
{
    public static string GetParallaxMappingFunction()
    {
        return @"
        // Parallax mapping function using height map
        float2 ParallaxMapping(float2 texCoords, float3 viewDir, sampler2D heightMap, float heightScale)
        {
            float height = tex2D(heightMap, texCoords).r;    
            float2 p = viewDir.xy / viewDir.z * (height * heightScale);
            return texCoords - p;  
        }
        
        // Usage in vertex/fragment shader:
        // float2 texCoords = ParallaxMapping(IN.uv_MainTex, normalize(IN.viewDir), _ParallaxMap, _Parallax);
        // fixed4 albedo = tex2D(_MainTex, texCoords);
        ";
    }
}
```

## Usage Context

The MLID chunk significantly enhances the visual richness of World of Warcraft's terrain by enabling height-based detail rendering. These height maps work in conjunction with the base terrain mesh to create the appearance of much more detailed surface geometry without the performance cost of additional vertices.

Introduced in Legion, the MLID chunk supports several techniques:

1. **Parallax Occlusion Mapping (POM)**: Creates the illusion of depth by offsetting texture coordinates based on view angle
2. **Displacement Mapping**: Actually modifies vertex positions in hardware-supported rendering
3. **Tessellation Control**: Can be used to guide adaptive tessellation for terrain detail

In the context of World of Warcraft's terrain rendering, height maps are particularly valuable for:

- **Rocky Surfaces**: Adding the appearance of cracks, ledges, and protrusions
- **Cobblestone Roads**: Creating convincing uneven stone pathways
- **Organic Terrain**: Adding small bumps, divots, and terrain irregularities
- **Architectural Details**: Adding relief to flat surfaces like walls and stone floors
- **Environmental Storytelling**: Showing worn paths, wheel ruts, or creature tracks

The combination of height maps with normal and specular maps creates a comprehensive physically-based rendering approach. While normal maps affect lighting and specular maps control reflection properties, height maps add actual perceived depth to surfaces. This multi-texture approach allows World of Warcraft to achieve impressive visual fidelity with efficient performance characteristics.

The transition to FileDataIDs in Legion through the MLID chunk made these advanced rendering techniques more accessible and efficient within the game's engine, contributing to the significant visual improvements introduced in that expansion. 