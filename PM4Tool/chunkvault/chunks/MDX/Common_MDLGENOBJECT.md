# MDLGENOBJECT - MDX Generic Object Structure

## Type
MDX Common Structure

## Source
MDX_index.md

## Description
The MDLGENOBJECT (Model Generic Object) structure is a common base structure used by many MDX chunks, particularly those representing scene objects like bones, lights, helpers, attachments, and emitters. It defines common properties like object name, ID, parent relationships, and flags, and provides a consistent framework for animation data.

## Structure

```csharp
public struct MDLGENOBJECT
{
    /// <summary>
    /// Object name (null-terminated string, max length 80)
    /// </summary>
    public fixed byte name[80];
    
    /// <summary>
    /// Unique object identifier
    /// </summary>
    public uint objectId;
    
    /// <summary>
    /// Parent object ID (0xFFFFFFFF = no parent)
    /// </summary>
    public uint parentId;
    
    /// <summary>
    /// Object flags (varies by object type)
    /// </summary>
    public uint flags;
    
    /// <summary>
    /// Animation data (variable size and structure based on object type)
    /// </summary>
    // Animation tracks follow - specific to each object type
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | name | char[80] | Null-terminated object name string |
| 0x50 | objectId | uint | Unique ID for the object (referenced by child objects) |
| 0x54 | parentId | uint | ID of parent object (0xFFFFFFFF = no parent) |
| 0x58 | flags | uint | Object-specific flags |
| 0x5C | ... | ... | Animation data (varies by object type) |

## Common Flag Values

| Bit | Name | Description |
|-----|------|-------------|
| 0 | DontInheritTranslation | Don't inherit translation from parent |
| 1 | DontInheritRotation | Don't inherit rotation from parent |
| 2 | DontInheritScaling | Don't inherit scaling from parent |
| 3-31 | ObjectSpecific | Object-specific flags - meaning depends on object type |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Same structure but expanded animation support |

## Usage Context
The MDLGENOBJECT structure:
- Creates a consistent object representation across many chunk types
- Establishes the scene hierarchy through parent-child relationships
- Defines object transformation inheritance behavior
- Provides a framework for animation tracks

## Object Hierarchy
Objects in MDX form a hierarchical tree where:
- Each object has a unique ID
- Objects can reference parent objects by ID
- Transformations can be inherited from parents
- The inheritance flags control which transformations are inherited
- Multiple independent hierarchies can exist in a single file

## Inheritance System
The inheritance system works as follows:
- If a flag bit is set (1), the object does NOT inherit that property from its parent
- If a flag bit is clear (0), the object inherits that property from its parent
- Inheritance is applied recursively through the object hierarchy
- The actual transformations are determined by animation tracks or static values

## Implementation Notes
- The name field is critical for identifying objects in tools and debugging
- Object IDs must be unique within each object type (bones, helpers, etc.)
- Parent IDs reference objects of the same type (bones can only parent to bones, etc.)
- The special parent ID 0xFFFFFFFF (-1 as signed int) indicates no parent (root object)
- Animation data that follows this structure varies significantly by object type
- Each chunk type using MDLGENOBJECT handles animation tracks differently

## Implementation Example

```csharp
public abstract class MdxGenericObject
{
    public string Name { get; set; }
    public uint ObjectId { get; set; }
    public uint ParentId { get; set; }
    public uint Flags { get; set; }
    
    // Flag accessors
    public bool DontInheritTranslation => (Flags & 0x1) != 0;
    public bool DontInheritRotation => (Flags & 0x2) != 0;
    public bool DontInheritScaling => (Flags & 0x4) != 0;
    
    public bool HasParent => ParentId != 0xFFFFFFFF;
    
    protected void ParseBaseObject(BinaryReader reader)
    {
        // Read name (null-terminated string, 80 bytes)
        byte[] nameBytes = reader.ReadBytes(80);
        int nameLength = 0;
        while (nameLength < nameBytes.Length && nameBytes[nameLength] != 0)
        {
            nameLength++;
        }
        Name = System.Text.Encoding.ASCII.GetString(nameBytes, 0, nameLength);
        
        // Read object ID and parent ID
        ObjectId = reader.ReadUInt32();
        ParentId = reader.ReadUInt32();
        Flags = reader.ReadUInt32();
    }
    
    protected void WriteBaseObject(BinaryWriter writer)
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
        
        // Write object ID and parent ID
        writer.Write(ObjectId);
        writer.Write(ParentId);
        writer.Write(Flags);
    }
    
    public abstract void ParseAnimationData(BinaryReader reader, uint version);
    public abstract void WriteAnimationData(BinaryWriter writer, uint version);
}
``` 