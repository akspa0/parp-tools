# MODL - MDX Model Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The MODL (Model) chunk contains basic information about the model, including its name, animation bounds, and various flags that control rendering behavior. This chunk appears early in the file and provides essential context for interpreting other chunks.

## Structure

```csharp
public struct MODL
{
    /// <summary>
    /// Model name (null-terminated string, max length 80)
    /// </summary>
    public fixed byte name[80];
    
    /// <summary>
    /// Animation bounding box minimum corner
    /// </summary>
    public Vector3 boundingBoxMin;
    
    /// <summary>
    /// Animation bounding box maximum corner
    /// </summary>
    public Vector3 boundingBoxMax;
    
    /// <summary>
    /// Radius of bounding sphere for model animations
    /// </summary>
    public float boundingSphereRadius;
    
    /// <summary>
    /// Flags controlling model behavior and rendering
    /// </summary>
    public uint flags;
    
    /// <summary>
    /// Number of bones in the model
    /// </summary>
    public uint numBones;
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | name | char[80] | Null-terminated model name string |
| 0x50 | boundingBoxMin | Vector3 | Minimum corner of animation bounding box |
| 0x5C | boundingBoxMax | Vector3 | Maximum corner of animation bounding box |
| 0x68 | boundingSphereRadius | float | Radius of bounding sphere for animations |
| 0x6C | flags | uint | Model flags (see Flags section) |
| 0x70 | numBones | uint | Number of bones/nodes in the model |

## Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | TiltX | Model tilts around X axis |
| 1 | TiltY | Model tilts around Y axis |
| 2 | Unknown0x4 | Unknown purpose |
| 3 | Unknown0x8 | Unknown purpose |
| 4 | Unknown0x10 | Unknown purpose |
| 5 | Unknown0x20 | Unknown purpose |
| 6 | Unknown0x40 | Unknown purpose |
| 7 | Unknown0x80 | Unknown purpose |
| 8 | Unknown0x100 | Unknown purpose |
| 9 | Unknown0x200 | Unknown purpose |
| 10 | Unknown0x400 | Unknown purpose |
| 11 | UseExternalTextures | Textures are referenced by filename rather than embedded |
| 12-31 | Reserved | Reserved for future use, typically set to 0 |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Added support for UseExternalTextures flag used in WoW versions |

## Dependencies
- VERS - Version information affects how model flags are interpreted

## Implementation Notes
- The model name is critical for texture path resolution in WoW versions
- The bounding box and sphere are used for culling and collision detection
- The TiltX and TiltY flags affect how the model is positioned in-game
- The UseExternalTextures flag is primarily used in WoW versions (1300-1500) and indicates that textures are loaded from separate files rather than being embedded
- The numBones field should match the actual number of bone entries in the BONE chunk

## Usage Context
The MODL chunk provides:
- Identifying information for the model
- Size and scale information for proper placement and rendering
- Flags that affect rendering behavior
- Basic information about model structure

## Implementation Example

```csharp
public class MODLChunk : IMdxChunk
{
    public string ChunkId => "MODL";
    
    public string Name { get; private set; }
    public Vector3 BoundingBoxMin { get; private set; }
    public Vector3 BoundingBoxMax { get; private set; }
    public float BoundingSphereRadius { get; private set; }
    public uint Flags { get; private set; }
    public uint NumBones { get; private set; }
    
    // Flag accessors
    public bool TiltX => (Flags & 0x1) != 0;
    public bool TiltY => (Flags & 0x2) != 0;
    public bool UseExternalTextures => (Flags & 0x800) != 0;
    
    public void Parse(BinaryReader reader, long size)
    {
        if (size != 116) // 80 + 12 + 12 + 4 + 4 + 4
        {
            throw new InvalidDataException($"MODL chunk size must be 116 bytes, found {size} bytes");
        }
        
        // Read name (null-terminated string, 80 bytes)
        byte[] nameBytes = reader.ReadBytes(80);
        int nameLength = 0;
        while (nameLength < nameBytes.Length && nameBytes[nameLength] != 0)
        {
            nameLength++;
        }
        Name = System.Text.Encoding.ASCII.GetString(nameBytes, 0, nameLength);
        
        // Read bounding box
        BoundingBoxMin = new Vector3(
            reader.ReadSingle(), 
            reader.ReadSingle(), 
            reader.ReadSingle()
        );
        
        BoundingBoxMax = new Vector3(
            reader.ReadSingle(), 
            reader.ReadSingle(), 
            reader.ReadSingle()
        );
        
        // Read other properties
        BoundingSphereRadius = reader.ReadSingle();
        Flags = reader.ReadUInt32();
        NumBones = reader.ReadUInt32();
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write name (pad with nulls to 80 bytes)
        byte[] nameBytes = new byte[80];
        if (!string.IsNullOrEmpty(Name))
        {
            byte[] temp = System.Text.Encoding.ASCII.GetBytes(Name);
            int copyLength = Math.Min(temp.Length, 79); // Leave at least one byte for null terminator
            Array.Copy(temp, nameBytes, copyLength);
        }
        writer.Write(nameBytes);
        
        // Write bounding box
        writer.Write(BoundingBoxMin.X);
        writer.Write(BoundingBoxMin.Y);
        writer.Write(BoundingBoxMin.Z);
        
        writer.Write(BoundingBoxMax.X);
        writer.Write(BoundingBoxMax.Y);
        writer.Write(BoundingBoxMax.Z);
        
        // Write other properties
        writer.Write(BoundingSphereRadius);
        writer.Write(Flags);
        writer.Write(NumBones);
    }
} 