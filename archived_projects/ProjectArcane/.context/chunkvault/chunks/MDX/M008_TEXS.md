# TEXS - MDX Textures Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The TEXS (Textures) chunk defines all the textures used by the model. Each texture entry provides information about the texture's path, type, and various flags that control how the texture is loaded and used. Textures are referenced by materials in the MTLS chunk.

## Structure

```csharp
public struct TEXS
{
    /// <summary>
    /// Array of texture definitions
    /// </summary>
    // MDLTEXTURE textures[numTextures] follows
}

public struct MDLTEXTURE
{
    /// <summary>
    /// Replacement ID for team-colored textures
    /// </summary>
    public uint replaceableId;
    
    /// <summary>
    /// Path to the texture file (null-terminated string, max length 260)
    /// </summary>
    public fixed byte fileName[260];
    
    /// <summary>
    /// Flags controlling texture behavior
    /// </summary>
    public uint flags;
}
```

## Properties

### TEXS Chunk
The TEXS chunk consists of an array of MDLTEXTURE structures. The number of textures is determined by the chunk size divided by the size of the MDLTEXTURE structure.

### MDLTEXTURE Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | replaceableId | uint | ID for replaceable textures (e.g., team colors) |
| 0x04 | fileName | char[260] | Null-terminated path to the texture file |
| 0x108 | flags | uint | Texture flags (see Texture Flags) |

## Replaceable IDs

| Value | Name | Description |
|-------|------|-------------|
| 0 | None | Not a replaceable texture |
| 1 | TeamColor | Primary team color |
| 2 | TeamGlow | Secondary team color (glowing parts) |
| 3-10 | Reserved | Reserved for future use, typically not used |
| 11+ | Custom | Custom replaceable textures |

## Texture Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | WrapWidth | Wrap texture horizontally (repeat U) |
| 1 | WrapHeight | Wrap texture vertically (repeat V) |
| 2-31 | Reserved | Reserved for future use |

## Texture Types
Based on file extension and usage:
- BLP: Blizzard's proprietary texture format (most common)
- TGA: Targa image format (used in some early models)
- JPG/PNG: Standard image formats (rarely used directly)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described, textures typically embedded in model |
| 1300-1500 (WoW Alpha) | Same structure but with UseExternalTextures flag in MODL chunk that determines if textures are loaded from external files or embedded |

## Dependencies
- MTLS - Materials reference textures by ID from the TEXS chunk
- MODL - The UseExternalTextures flag in MODL affects how texture paths are resolved

## Implementation Notes
- For Warcraft 3 models (800-1000), textures are typically stored with relative paths within the MPQ archives
- For WoW Alpha models (1300-1500), texture paths may be stored as absolute paths or relative to a base directory
- Replaceable textures (team colors) use a special system where the texture path is ignored and a color is used instead
- The fileName field uses Windows-style path separators (backslashes)
- The max path length of 260 characters matches the Windows MAX_PATH constant
- In the WoW client, textures with relative paths are looked up in the game's file system using model-relative paths

## Texture Path Resolution
Texture paths are resolved differently based on the model version and flags:
1. If replaceableId is non-zero, the texture is a color-replaced texture and the path is ignored
2. For v800-1000 (WC3):
   - Textures are loaded from the MPQ archive using the relative path
3. For v1300-1500 (WoW Alpha):
   - If UseExternalTextures flag is set in MODL:
     - For absolute paths, load directly
     - For relative paths, resolve relative to the model's directory
   - Otherwise, textures may be embedded in the model file

## Usage Context
The TEXS chunk:
- Defines all texture resources used by the model
- Establishes the system for team colors and other replaceable textures
- Provides file paths for texture loading
- Sets wrapping behavior for texture coordinates

## Implementation Example

```csharp
public class TEXSChunk : IMdxChunk
{
    public string ChunkId => "TEXS";
    
    public List<MdxTexture> Textures { get; private set; } = new List<MdxTexture>();
    
    public void Parse(BinaryReader reader, long size)
    {
        // Each MDLTEXTURE is 268 bytes (4 + 260 + 4)
        int numTextures = (int)(size / 268);
        
        // Clear any existing textures
        Textures.Clear();
        
        // Read all textures
        for (int i = 0; i < numTextures; i++)
        {
            var texture = new MdxTexture();
            
            // Read replaceableId
            texture.ReplaceableId = reader.ReadUInt32();
            
            // Read fileName (null-terminated string, 260 bytes)
            byte[] fileNameBytes = reader.ReadBytes(260);
            int fileNameLength = 0;
            while (fileNameLength < fileNameBytes.Length && fileNameBytes[fileNameLength] != 0)
            {
                fileNameLength++;
            }
            texture.FileName = System.Text.Encoding.ASCII.GetString(fileNameBytes, 0, fileNameLength);
            
            // Read flags
            texture.Flags = reader.ReadUInt32();
            
            Textures.Add(texture);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var texture in Textures)
        {
            // Write replaceableId
            writer.Write(texture.ReplaceableId);
            
            // Write fileName (pad with nulls to 260 bytes)
            byte[] fileNameBytes = new byte[260];
            if (!string.IsNullOrEmpty(texture.FileName))
            {
                byte[] temp = System.Text.Encoding.ASCII.GetBytes(texture.FileName);
                int copyLength = Math.Min(temp.Length, 259); // Leave at least one byte for null terminator
                Array.Copy(temp, fileNameBytes, copyLength);
            }
            writer.Write(fileNameBytes);
            
            // Write flags
            writer.Write(texture.Flags);
        }
    }
    
    /// <summary>
    /// Resolves a texture path based on model version and flags
    /// </summary>
    /// <param name="textureIndex">Index of the texture</param>
    /// <param name="modelVersion">MDX model version</param>
    /// <param name="modelPath">Path to the model file</param>
    /// <param name="useExternalTextures">Value of MODL.UseExternalTextures flag</param>
    /// <returns>Resolved texture path or null for replaceable textures</returns>
    public string ResolveTexturePath(int textureIndex, uint modelVersion, string modelPath, bool useExternalTextures)
    {
        if (textureIndex < 0 || textureIndex >= Textures.Count)
        {
            return null;
        }
        
        var texture = Textures[textureIndex];
        
        // Check if this is a replaceable texture
        if (texture.ReplaceableId != 0)
        {
            return null; // Replaceable textures don't use file paths
        }
        
        string fileName = texture.FileName;
        
        // Handle different versions
        if (modelVersion >= 1300 && useExternalTextures)
        {
            // WoW Alpha model with external textures
            if (Path.IsPathRooted(fileName))
            {
                return fileName; // Absolute path
            }
            else
            {
                // Relative path - resolve against model directory
                string modelDir = Path.GetDirectoryName(modelPath);
                return Path.Combine(modelDir, fileName);
            }
        }
        else
        {
            // Warcraft 3 model or WoW model with embedded textures
            return fileName; // Just return the relative path for lookup in archives
        }
    }
}

public class MdxTexture
{
    public uint ReplaceableId { get; set; }
    public string FileName { get; set; }
    public uint Flags { get; set; }
    
    // Flag accessors
    public bool WrapWidth => (Flags & 0x1) != 0;
    public bool WrapHeight => (Flags & 0x2) != 0;
    
    // Helper properties
    public bool IsReplaceable => ReplaceableId != 0;
    public bool IsTeamColor => ReplaceableId == 1;
    public bool IsTeamGlow => ReplaceableId == 2;
}
``` 