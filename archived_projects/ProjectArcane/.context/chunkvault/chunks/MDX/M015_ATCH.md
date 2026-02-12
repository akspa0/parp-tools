# ATCH - MDX Attachments Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The ATCH (Attachments) chunk defines attachment points for models and effects. Attachments connect models to other models or anchor special effects to specific locations. They are used for weapon mountings, spell effect origins, and connection points for armor pieces, shields, and other accessories. Attachments are crucial for proper model composition in games.

## Structure

```csharp
public struct ATCH
{
    /// <summary>
    /// Array of attachment definitions
    /// </summary>
    // MDLATTACHMENT attachments[numAttachments] follows
}

public struct MDLATTACHMENT : MDLGENOBJECT
{
    /// <summary>
    /// Path to attachment model file
    /// </summary>
    public string path;
    
    /// <summary>
    /// Reserved field (set to 0)
    /// </summary>
    public uint reserved;
    
    /// <summary>
    /// Attachment ID (index in the attachment array)
    /// </summary>
    public uint attachmentId;
    
    /// <summary>
    /// Animation data for the attachment
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLATTACHMENT Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00..0x58 | MDLGENOBJECT | struct | Base generic object (see MDLGENOBJECT structure) |
| 0x58 | path | string | Path to attachment model file, null-terminated |
| varies | reserved | uint | Reserved field (set to 0) |
| varies | attachmentId | uint | Unique identifier for the attachment (index) |
| varies | ... | ... | Animation tracks follow |

## Animation Tracks
After the base properties, several animation tracks may follow:

- Visibility track (int, 0 or 1) - Controls when the attachment is visible

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Additional attachment types and capabilities |

## Dependencies
- MDLGENOBJECT - All attachments inherit from the generic object structure
- MDLKEYTRACK - Used for animation tracks within the structure
- BONE - Attachments can be attached to bones via parentId
- HELP - Helper objects may be used to define attachment points

## Special Attachment Types
Several standard attachment points are recognized by game engines:

| Attachment Name | Purpose |
|-----------------|---------|
| "Weapon" | Primary weapon mounting point |
| "Shield" | Shield mounting point |
| "Head" | Head slot for helmets |
| "Chest" | Chest slot for armor |
| "Origin" | Base point for effects |
| "Overhead" | Point for overhead elements (nameplates) |
| "Left Hand" | Left hand weapon/item mount |
| "Right Hand" | Right hand weapon/item mount |
| "Backpack" | Back mounting point |

## Implementation Notes
- Attachments define the connection points where other models and effects can be attached
- They can be animated to show/hide at specific animation frames
- The path string may point to an external model file that should be loaded at this attachment point
- Attachments are used extensively in character customization systems
- They must properly follow the parent bone's transformations
- The visibility track determines when attachments are shown or hidden
- Attachment coordinates are relative to the parent bone or node
- The name of the attachment often indicates its purpose (e.g., "Attachment_Head")
- Attachments with empty paths serve as reference points for effects
- In games, attachments are typically queried by name to find relevant points
- The reserved field is currently unused and should be set to 0
- Some game engines use attachment IDs as lookup indices

## Usage Context
Attachments in MDX models are used for:
- Weapon mounting points
- Shield and off-hand item positions
- Armor and clothing attachment
- Mount connection points (saddles, etc.)
- Special effect origins (spell casting points)
- Projectile launch points
- Character customization slots
- Interface element anchors (nameplates, health bars)
- NPC interaction points
- Equipment visualization

## Implementation Example

```csharp
public class ATCHChunk : IMdxChunk
{
    public string ChunkId => "ATCH";
    
    public List<MdxAttachment> Attachments { get; private set; } = new List<MdxAttachment>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear existing attachments
        Attachments.Clear();
        
        // Read attachments until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var attachment = new MdxAttachment();
            
            // Read base object properties
            attachment.ParseBaseObject(reader);
            
            // Read attachment specific properties
            attachment.Path = reader.ReadCString();
            attachment.Reserved = reader.ReadUInt32();
            attachment.AttachmentId = reader.ReadUInt32();
            
            // Read animation tracks
            attachment.VisibilityTrack = new MdxKeyTrack<int>();
            attachment.VisibilityTrack.Parse(reader, r => r.ReadInt32());
            
            Attachments.Add(attachment);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var attachment in Attachments)
        {
            // Write base object properties
            attachment.WriteBaseObject(writer);
            
            // Write attachment specific properties
            writer.WriteCString(attachment.Path);
            writer.Write(attachment.Reserved);
            writer.Write(attachment.AttachmentId);
            
            // Write animation tracks
            attachment.VisibilityTrack.Write(writer, (w, i) => w.Write(i));
        }
    }
    
    /// <summary>
    /// Finds an attachment by name
    /// </summary>
    /// <param name="name">Name to search for</param>
    /// <returns>The attachment with the given name, or null if not found</returns>
    public MdxAttachment FindAttachmentByName(string name)
    {
        return Attachments.FirstOrDefault(a => string.Equals(a.Name, name, StringComparison.OrdinalIgnoreCase));
    }
    
    /// <summary>
    /// Gets an attachment by ID
    /// </summary>
    /// <param name="id">Attachment ID to find</param>
    /// <returns>The attachment with the given ID, or null if not found</returns>
    public MdxAttachment GetAttachmentById(uint id)
    {
        return Attachments.FirstOrDefault(a => a.AttachmentId == id);
    }
    
    /// <summary>
    /// Checks if an attachment is visible at the given time
    /// </summary>
    /// <param name="attachment">The attachment to check</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>True if the attachment is visible, false otherwise</returns>
    public bool IsAttachmentVisible(MdxAttachment attachment, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        // If no visibility track, attachment is always visible
        if (attachment.VisibilityTrack.NumKeys == 0)
        {
            return true;
        }
        
        // Check the visibility track value at the current time
        int visibility = attachment.VisibilityTrack.Evaluate(time, sequenceDuration, globalSequences);
        return visibility > 0;
    }
    
    /// <summary>
    /// Gets visible attachments at the given time
    /// </summary>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>List of visible attachments</returns>
    public List<MdxAttachment> GetVisibleAttachments(uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        return Attachments.Where(a => IsAttachmentVisible(a, time, sequenceDuration, globalSequences)).ToList();
    }
}

public class MdxAttachment : MdxGenericObject
{
    public string Path { get; set; }
    public uint Reserved { get; set; }
    public uint AttachmentId { get; set; }
    public MdxKeyTrack<int> VisibilityTrack { get; set; }
    
    /// <summary>
    /// Gets whether this attachment has an external model
    /// </summary>
    public bool HasExternalModel => !string.IsNullOrEmpty(Path);
    
    /// <summary>
    /// Gets whether this is a standard attachment type
    /// </summary>
    public bool IsStandardAttachment
    {
        get
        {
            if (string.IsNullOrEmpty(Name)) return false;
            
            string[] standardAttachments = new[]
            {
                "Weapon", "Shield", "Head", "Chest", "Origin",
                "Overhead", "Left Hand", "Right Hand", "Backpack"
            };
            
            return standardAttachments.Any(a => Name.Contains(a, StringComparison.OrdinalIgnoreCase));
        }
    }
    
    /// <summary>
    /// Gets the attachment type based on the name
    /// </summary>
    public string AttachmentType
    {
        get
        {
            if (string.IsNullOrEmpty(Name)) return "Unknown";
            
            // Extract from name format "Attachment_Type"
            if (Name.StartsWith("Attachment_", StringComparison.OrdinalIgnoreCase))
            {
                return Name.Substring("Attachment_".Length);
            }
            
            return "Generic";
        }
    }
} 