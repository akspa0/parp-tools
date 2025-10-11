# MNID (Map Normal ID)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MNID chunk was introduced in Legion and contains FileDataIDs for normal map textures used in terrain rendering. This chunk complements the MDID chunk, replacing string-based texture references in favor of direct numeric FileDataIDs. Normal maps provide surface detail information that enhances lighting and shading without requiring additional geometry.

## Structure

```csharp
public struct MNID
{
    public uint[] NormalMapFileDataIDs;  // Array of FileDataIDs for normal map textures
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| NormalMapFileDataIDs | uint[] | Array of FileDataIDs for normal map texture assets used in this ADT |

## Dependencies

- **MDID (C023)** - Contains FileDataIDs for corresponding diffuse textures
- **MCNK (C009)** - Uses these normal maps for terrain chunk rendering
- **MCLY (S002)** - Contains texture layer definitions that reference these normal maps

## Implementation Notes

- The MNID chunk was introduced alongside MDID in Legion as part of the transition to FileDataIDs
- Each entry corresponds positionally to an entry in the MDID chunk
- A value of 0 indicates no normal map for that texture
- Normal maps use a tangent-space format with RGB channels encoding XYZ normal vectors
- The array size matches the number of diffuse textures in the MDID chunk
- Normal maps significantly enhance the perceived detail of terrain without additional geometry

## Implementation Example

```csharp
public class NormalMapManager
{
    private Dictionary<uint, Texture2D> normalMapCache = new Dictionary<uint, Texture2D>();
    private FileDataService fileDataService;
    
    public NormalMapManager(FileDataService fileDataService)
    {
        this.fileDataService = fileDataService;
    }
    
    public void LoadNormalMaps(uint[] normalMapFileDataIDs)
    {
        foreach (uint fileDataID in normalMapFileDataIDs)
        {
            if (fileDataID == 0 || normalMapCache.ContainsKey(fileDataID))
                continue;
                
            try
            {
                byte[] fileData = fileDataService.GetFileData(fileDataID);
                
                // Create normal map texture from file data (implementation depends on format)
                Texture2D normalMap = CreateTextureFromFileData(fileData);
                if (normalMap != null)
                {
                    // Configure texture for normal map usage
                    normalMap.SetTextureSettingsForNormalMap();
                    normalMapCache[fileDataID] = normalMap;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load normal map {fileDataID}: {ex.Message}");
            }
        }
    }
    
    public Texture2D GetNormalMap(uint fileDataID)
    {
        if (fileDataID == 0 || !normalMapCache.ContainsKey(fileDataID))
            return null;
            
        return normalMapCache[fileDataID];
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
    
    // Apply materials with normal maps to terrain
    public void ApplyNormalMapsToMaterials(Material[] materials, uint[] diffuseIDs, uint[] normalIDs)
    {
        for (int i = 0; i < materials.Length; i++)
        {
            if (materials[i] == null)
                continue;
                
            if (i < normalIDs.Length && normalIDs[i] != 0)
            {
                Texture2D normalMap = GetNormalMap(normalIDs[i]);
                if (normalMap != null)
                {
                    materials[i].SetTexture("_BumpMap", normalMap);
                    materials[i].EnableKeyword("_NORMALMAP");
                    
                    // Set normal map intensity
                    materials[i].SetFloat("_BumpScale", 1.0f);
                }
            }
        }
    }
}

// Extension method for configuring textures as normal maps
public static class TextureExtensions
{
    public static void SetTextureSettingsForNormalMap(this Texture2D texture)
    {
        // Normal maps should not use compression that would distort normal data
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.filterMode = FilterMode.Bilinear;
        texture.anisoLevel = 4;  // Better anisotropic filtering for terrain viewing angles
    }
}
```

## Usage Context

The MNID chunk is a key component in World of Warcraft's terrain rendering system, particularly for adding visual detail without increasing geometry complexity. Normal maps work by encoding surface detail as RGB color data, where each pixel's color represents the direction a surface is facing at that point.

Introduced in Legion, the MNID chunk replaced the previous approach of embedding normal map references in texture filenames. This change provides several benefits:

1. **Performance**: Direct ID lookups are faster than string manipulation and filesystem searches
2. **Flexibility**: Normal maps can be changed independently of diffuse textures
3. **Space Efficiency**: Integer IDs require less storage than string paths
4. **Art Pipeline**: Allows for more efficient art pipeline with independent asset updates

When rendering terrain, the normal maps referenced in this chunk are combined with the base mesh normals to create the appearance of much more detailed geometry. This technique is especially effective for terrain features like:

- Small rocks and pebbles
- Cracks and crevices
- Subtle elevation changes
- Surface roughness
- Fabric wrinkles on tents or structures

The combination of base geometry, diffuse textures, and normal maps creates visually rich environments without the performance cost of high-polygon models. This technique is fundamental to modern game rendering and is optimized in World of Warcraft through the MNID chunk's efficient FileDataID-based approach. 