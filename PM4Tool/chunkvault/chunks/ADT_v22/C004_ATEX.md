# C004: ATEX

## Type
ADT v22 Chunk

## Source
ADT_v22.md

## Description
Contains texture filenames used for texturing the terrain in ADT v22 format.

## Structure
```csharp
struct ATEX
{
    // An array of chunks, each containing a texture filename
    // There are as many chunks as there are textures used in this ADT tile
    // Limited to 128 textures per ADT
    char filenames[]; // Zero-terminated strings
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Array of zero-terminated strings with texture filenames |

## Dependencies
- AHDR (C001) - Provides overall file structure
- Referenced by: ALYR (in ACNK) - Texture references use indices into this array

## Implementation Notes
- Size: Variable
- The textures are referenced by index in ALYR chunks
- This is an array of chunks, with each chunk containing a single texture filename
- ADT tiles are limited to 128 unique textures
- Similar to MTEX in ADT v18 format, but part of the new chunk system
- String format likely includes full path to the texture

## Implementation Example
```csharp
public class ATEX
{
    public List<string> TextureFilenames { get; set; } = new List<string>();
    
    // Helper method to get a texture by index
    public string GetTexture(int index)
    {
        if (index < 0 || index >= TextureFilenames.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        return TextureFilenames[index];
    }
    
    // Helper method to find index of a texture by name
    public int FindTextureIndex(string textureName)
    {
        return TextureFilenames.FindIndex(t => t.Equals(textureName, StringComparison.OrdinalIgnoreCase));
    }
    
    // Helper method to add a texture and return its index
    public int AddTexture(string textureName)
    {
        // Check if we already have this texture
        int existingIndex = FindTextureIndex(textureName);
        if (existingIndex >= 0)
            return existingIndex;
            
        // Check texture limit
        if (TextureFilenames.Count >= 128)
            throw new InvalidOperationException("ADT tile cannot have more than 128 textures");
            
        // Add the new texture
        TextureFilenames.Add(textureName);
        return TextureFilenames.Count - 1;
    }
    
    // Helper method to check if the texture index is valid
    public bool IsValidTextureIndex(int index)
    {
        return index >= 0 && index < TextureFilenames.Count;
    }
}
```

## Usage Context
The ATEX chunk stores the filenames of textures used for terrain rendering. These textures are applied to the terrain surface through texture layers defined in ALYR chunks. Each texture is referenced by its index in the ATEX array. The textures typically include diffuse maps for the terrain surface, such as grass, dirt, snow, etc. This system allows for blending multiple textures on the terrain through alpha maps. 