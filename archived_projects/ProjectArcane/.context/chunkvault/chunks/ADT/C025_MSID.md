# MSID (Map Specular ID)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MSID chunk was introduced in Legion and contains FileDataIDs for specular map textures used in terrain rendering. This chunk works alongside MDID and MNID, replacing string-based texture references with direct numeric FileDataIDs. Specular maps control how light reflects off surfaces, defining shininess and reflective properties of the terrain.

## Structure

```csharp
public struct MSID
{
    public uint[] SpecularMapFileDataIDs;  // Array of FileDataIDs for specular map textures
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| SpecularMapFileDataIDs | uint[] | Array of FileDataIDs for specular map texture assets used in this ADT |

## Dependencies

- **MDID (C023)** - Contains FileDataIDs for corresponding diffuse textures
- **MNID (C024)** - Contains FileDataIDs for corresponding normal maps
- **MCNK (C009)** - Uses these specular maps for terrain chunk rendering
- **MCLY (S002)** - Contains texture layer definitions that reference these specular maps

## Implementation Notes

- The MSID chunk was introduced alongside MDID and MNID in Legion as part of the transition to FileDataIDs
- Each entry corresponds positionally to an entry in the MDID chunk
- A value of 0 indicates no specular map for that texture
- Specular maps typically use grayscale values where:
  - White areas represent highly reflective/shiny surfaces (metals, water, ice)
  - Dark/black areas represent matte surfaces (dirt, cloth, grass)
- Some specular maps may use RGB channels to control different specular properties
- The array size matches the number of diffuse textures in the MDID chunk
- Not all terrain textures have associated specular maps

## Implementation Example

```csharp
public class SpecularMapManager
{
    private Dictionary<uint, Texture2D> specularMapCache = new Dictionary<uint, Texture2D>();
    private FileDataService fileDataService;
    
    public SpecularMapManager(FileDataService fileDataService)
    {
        this.fileDataService = fileDataService;
    }
    
    public void LoadSpecularMaps(uint[] specularMapFileDataIDs)
    {
        foreach (uint fileDataID in specularMapFileDataIDs)
        {
            if (fileDataID == 0 || specularMapCache.ContainsKey(fileDataID))
                continue;
                
            try
            {
                byte[] fileData = fileDataService.GetFileData(fileDataID);
                
                // Create specular map texture from file data
                Texture2D specularMap = CreateTextureFromFileData(fileData);
                if (specularMap != null)
                {
                    // Configure texture for specular map usage
                    specularMap.SetTextureSettingsForSpecularMap();
                    specularMapCache[fileDataID] = specularMap;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load specular map {fileDataID}: {ex.Message}");
            }
        }
    }
    
    public Texture2D GetSpecularMap(uint fileDataID)
    {
        if (fileDataID == 0 || !specularMapCache.ContainsKey(fileDataID))
            return null;
            
        return specularMapCache[fileDataID];
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
    
    // Apply specular maps to terrain materials
    public void ApplySpecularMapsToMaterials(Material[] materials, uint[] diffuseIDs, uint[] specularIDs)
    {
        for (int i = 0; i < materials.Length; i++)
        {
            if (materials[i] == null)
                continue;
            
            // Set default specular properties
            materials[i].SetFloat("_Glossiness", 0.1f);  // Default low glossiness
            materials[i].SetColor("_SpecColor", new Color(0.2f, 0.2f, 0.2f, 1.0f));  // Default low specular color
                
            // Apply specular map if available
            if (i < specularIDs.Length && specularIDs[i] != 0)
            {
                Texture2D specularMap = GetSpecularMap(specularIDs[i]);
                if (specularMap != null)
                {
                    materials[i].SetTexture("_SpecGlossMap", specularMap);
                    materials[i].EnableKeyword("_SPECGLOSSMAP");
                    
                    // Increase base glossiness when specular map is present
                    materials[i].SetFloat("_Glossiness", 0.5f);
                }
            }
        }
    }
}

// Extension method for configuring textures as specular maps
public static class SpecularTextureExtensions
{
    public static void SetTextureSettingsForSpecularMap(this Texture2D texture)
    {
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Bilinear;
        texture.anisoLevel = 2;  // Moderate anisotropic filtering (specular details are less critical than diffuse/normal)
    }
}
```

## Usage Context

The MSID chunk plays a vital role in World of Warcraft's physically-based rendering (PBR) system, particularly for creating realistic lighting interactions with terrain. Specular maps define how light bounces off surfaces, controlling the intensity and spread of reflections.

Introduced in Legion alongside the other FileDataID chunks, the MSID chunk helps create more realistic terrain materials by specifying:

1. **Shininess/Glossiness**: How focused or diffuse light reflections are (sharp reflections for ice, soft reflections for damp grass)
2. **Reflectivity**: How much light is reflected vs. absorbed by a surface (high for metals, low for soil)
3. **Surface Type Indication**: Indirectly helps suggest the physical properties of materials

In the context of World of Warcraft's environments, specular maps are particularly important for:

- **Water Edges**: Creating wet shorelines with appropriate reflectivity
- **Ice and Snow**: Giving appropriate shine to frozen areas
- **Metallic Surfaces**: Highlighting ore veins or metal objects embedded in terrain
- **Rock Variations**: Distinguishing between rough stone and polished/wet stone
- **Environmental Storytelling**: Showing worn paths or areas affected by magical effects

In game rendering, the specular maps are combined with normal maps and diffuse textures in a shader pipeline to create a final composite appearance. The use of FileDataIDs in the MSID chunk allows the game to efficiently load and manage these texture resources, ensuring optimal performance while maintaining high visual quality. 