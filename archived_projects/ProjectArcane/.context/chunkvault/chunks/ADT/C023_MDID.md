# MDID (Map Diffuse ID)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MDID chunk was introduced in Legion and contains FileDataIDs for diffuse texture maps. This chunk replaced the string-based texture references in the MTEX chunk with direct FileDataIDs, improving load times and resource management. It provides direct references to the diffuse textures used for terrain rendering.

## Structure

```csharp
public struct MDID
{
    public uint[] DiffuseTextureFileDataIDs;  // Array of FileDataIDs for diffuse textures
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| DiffuseTextureFileDataIDs | uint[] | Array of FileDataIDs for diffuse texture assets used in this ADT |

## Dependencies

- **MTEX (C004)** - May still be present in the file for backward compatibility
- **MCNK (C009)** - Uses texture indices that reference entries in this chunk
- **MCLY (S002)** - Contains layer definitions that reference texture indices

## Implementation Notes

- The MDID chunk complements or replaces the MTEX chunk starting in Legion
- Each FileDataID directly references a texture file in the game's database
- A value of 0 indicates an invalid or unused texture slot
- The array size matches the number of texture entries needed by the ADT
- In newer files, the MTEX chunk might be empty or missing entirely
- FileDataIDs provide a more efficient way to access resources than filename strings

## Implementation Example

```csharp
public class DiffuseTextureManager
{
    private Dictionary<uint, Texture2D> textureCache = new Dictionary<uint, Texture2D>();
    private FileDataService fileDataService;
    
    public DiffuseTextureManager(FileDataService fileDataService)
    {
        this.fileDataService = fileDataService;
    }
    
    public void LoadTextures(uint[] diffuseTextureFileDataIDs)
    {
        foreach (uint fileDataID in diffuseTextureFileDataIDs)
        {
            if (fileDataID == 0 || textureCache.ContainsKey(fileDataID))
                continue;
                
            try
            {
                byte[] fileData = fileDataService.GetFileData(fileDataID);
                
                // Create texture from file data (implementation depends on format)
                Texture2D texture = CreateTextureFromFileData(fileData);
                if (texture != null)
                {
                    textureCache[fileDataID] = texture;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load texture {fileDataID}: {ex.Message}");
            }
        }
    }
    
    public Texture2D GetTexture(uint fileDataID)
    {
        if (fileDataID == 0 || !textureCache.ContainsKey(fileDataID))
            return null;
            
        return textureCache[fileDataID];
    }
    
    private Texture2D CreateTextureFromFileData(byte[] fileData)
    {
        // In a real implementation, this would detect the format (BLP, etc.)
        // and properly decode the texture data
        Texture2D texture = new Texture2D(2, 2);
        
        // This is a simplified placeholder - actual implementation would decode BLP format
        if (texture.LoadImage(fileData))
        {
            return texture;
        }
        
        return null;
    }
    
    // Get materials for a terrain chunk using MDID texture references
    public Material[] GetLayerMaterials(int[] textureIndices, uint[] diffuseTextureFileDataIDs)
    {
        Material[] materials = new Material[textureIndices.Length];
        
        for (int i = 0; i < textureIndices.Length; i++)
        {
            int index = textureIndices[i];
            if (index < 0 || index >= diffuseTextureFileDataIDs.Length)
            {
                materials[i] = null;
                continue;
            }
            
            uint fileDataID = diffuseTextureFileDataIDs[index];
            Texture2D texture = GetTexture(fileDataID);
            
            if (texture != null)
            {
                Material mat = new Material(Shader.Find("World/Terrain"));
                mat.mainTexture = texture;
                materials[i] = mat;
            }
            else
            {
                materials[i] = null;
            }
        }
        
        return materials;
    }
}
```

## Usage Context

The MDID chunk represents a significant evolution in World of Warcraft's resource management system. Prior to Legion, terrain textures were referenced by filename strings stored in the MTEX chunk. This approach had several limitations:

1. Strings consume more space than integer IDs
2. String lookups are slower than direct ID lookups
3. Filenames could change, breaking references
4. Localized content required different filenames

With the introduction of the MDID chunk, the game transitioned to a more efficient FileDataID system. Each texture is assigned a unique integer ID that remains constant regardless of filename changes or localization. This system provides several benefits:

1. **Improved Performance**: Direct ID lookups are faster than string-based file searches
2. **Reduced File Size**: Integer IDs consume less space than strings
3. **Better Localization Support**: The same ID can point to different files based on client locale
4. **Enhanced Patching**: Files can be renamed or moved without breaking references

In practice, client applications now primarily use the MDID chunk for texture references when available, falling back to the MTEX chunk only for compatibility with older content or development tools. This transition is part of Blizzard's broader move toward the FileDataID system across all of their game assets. 