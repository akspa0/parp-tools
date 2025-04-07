# HELP - MDX Helper Objects Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The HELP (Helper Objects) chunk defines non-rendering reference objects used for attachment points, effects origins, hit testing, or other technical reference purposes. Helper objects can be attached to the bone hierarchy and provide named markers that game logic can use to locate specific points on the model, such as weapon attachment points, spell cast origins, or target locations.

## Structure

```csharp
public struct HELP
{
    /// <summary>
    /// Array of helper object definitions
    /// </summary>
    // MDLHELPER helpers[numHelpers] follows
}

public struct MDLHELPER : MDLGENOBJECT
{
    // MDLHELPER inherits all fields from MDLGENOBJECT and has no additional fields
    // However, animation tracks follow the base structure
}
```

## Properties

### MDLHELPER Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00..0x58 | MDLGENOBJECT | struct | Base generic object (see MDLGENOBJECT structure) |
| 0x58+ | ... | ... | Animation tracks follow |

## Helper Object Types
Helper objects are differentiated by their names, which follow specific naming conventions:

| Prefix | Type | Description |
|--------|------|-------------|
| "Attachment_" | Attachment Point | Used for attaching weapons, shields, effects, etc. |
| "Origin_" | Effect Origin | Used as the source for spell/ability effects |
| "Target_" | Target Point | Used as the target location for effects |
| "Hit_" | Hit Test Point | Used for calculating hit detection |
| "Mount_" | Mount Point | Used for character mounting locations |
| "Camera_" | Camera Reference | Used for camera positioning during animations |
| "Ref_" | Generic Reference | Used for general positioning needs |

## Animation Tracks
After the base properties, several animation tracks may follow:

- Translation track (Vector3 XYZ)
- Rotation track (Quaternion XYZW)
- Scaling track (Vector3 XYZ)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Same structure with expanded naming conventions for WoW-specific helper types |

## Dependencies
- MDLGENOBJECT - All helper objects inherit from the generic object structure
- MDLKEYTRACK - Used for animation tracks within the structure
- BONE - Helper objects can be attached to bones in the model hierarchy via the parentId

## Implementation Notes
- Helper objects use the MDLGENOBJECT structure for base properties (name, ID, parent, flags)
- Helpers are non-rendering objects; they are not visible in the final model
- The name field of a helper object is critical as it defines the purpose of the helper
- Helper objects are often used by game code to locate specific points on a model
- The translation, rotation, and scaling tracks define the position and orientation of the helper object over time
- For Warcraft 3 models, typical helper objects include attachment points for weapons and projectiles
- For WoW Alpha models, a broader range of helper types might be used for character customization, effects, and interactions

## Usage Context
Helper objects in MDX models serve several key purposes:
- Providing attachment points for weapons, shields, or armor
- Creating spell cast origins for visual effects
- Defining hit test locations for combat calculations
- Setting up mount points for character interactions
- Creating reference points for camera positioning
- Establishing effect target locations for animations
- Providing technical reference points for game logic

## Common Helper Names
- "Attachment_right_hand" - Right hand weapon attachment
- "Attachment_left_hand" - Left hand weapon/shield attachment
- "Attachment_head" - Head item attachment point
- "Attachment_chest" - Chest effect attachment
- "Origin_cast1" - Primary spell cast origin
- "Origin_cast2" - Secondary spell cast origin
- "Target_chest" - Target point for chest hits
- "Mount_0" - Primary mount attachment point

## Implementation Example

```csharp
public class HELPChunk : IMdxChunk
{
    public string ChunkId => "HELP";
    
    public List<MdxHelper> Helpers { get; private set; } = new List<MdxHelper>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear any existing helpers
        Helpers.Clear();
        
        // Read helpers until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var helper = new MdxHelper();
            
            // Read base object properties
            helper.ParseBaseObject(reader);
            
            // Read animation tracks
            // Translation
            helper.TranslationTrack = new MdxKeyTrack<Vector3>();
            helper.TranslationTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Rotation
            helper.RotationTrack = new MdxKeyTrack<Quaternion>();
            helper.RotationTrack.Parse(reader, r => new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Scaling
            helper.ScalingTrack = new MdxKeyTrack<Vector3>();
            helper.ScalingTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            Helpers.Add(helper);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var helper in Helpers)
        {
            // Write base object properties
            helper.WriteBaseObject(writer);
            
            // Write animation tracks
            helper.TranslationTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            helper.RotationTrack.Write(writer, (w, q) => { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); });
            helper.ScalingTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
        }
    }
    
    /// <summary>
    /// Finds a helper object by its name
    /// </summary>
    /// <param name="name">Name of the helper to find</param>
    /// <returns>The helper object, or null if not found</returns>
    public MdxHelper FindHelperByName(string name)
    {
        return Helpers.FirstOrDefault(h => string.Equals(h.Name, name, StringComparison.OrdinalIgnoreCase));
    }
    
    /// <summary>
    /// Gets all helper objects of a specific type based on prefix
    /// </summary>
    /// <param name="prefix">The prefix to search for (e.g., "Attachment_")</param>
    /// <returns>List of helpers matching the prefix</returns>
    public List<MdxHelper> GetHelpersByType(string prefix)
    {
        return Helpers.Where(h => h.Name != null && h.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)).ToList();
    }
    
    /// <summary>
    /// Gets the current transform for a helper object
    /// </summary>
    /// <param name="helperIndex">Index of the helper</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The current transform matrix for the helper</returns>
    public Matrix4x4 GetHelperTransform(int helperIndex, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        if (helperIndex < 0 || helperIndex >= Helpers.Count)
        {
            return Matrix4x4.Identity;
        }
        
        var helper = Helpers[helperIndex];
        
        // Get current position, rotation, and scale
        Vector3 position = helper.TranslationTrack.Evaluate(time, sequenceDuration, globalSequences);
        Quaternion rotation = helper.RotationTrack.Evaluate(time, sequenceDuration, globalSequences);
        Vector3 scale = helper.ScalingTrack.Evaluate(time, sequenceDuration, globalSequences);
        
        // Create transform matrix
        Matrix4x4 scaleMatrix = Matrix4x4.CreateScale(scale);
        Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(rotation);
        Matrix4x4 translationMatrix = Matrix4x4.CreateTranslation(position);
        
        // Combine transforms: scale -> rotate -> translate
        return scaleMatrix * rotationMatrix * translationMatrix;
    }
}

public class MdxHelper : MdxGenericObject
{
    /// <summary>
    /// Gets the type of helper based on its name prefix
    /// </summary>
    public HelperType Type
    {
        get
        {
            if (string.IsNullOrEmpty(Name))
                return HelperType.Generic;
                
            if (Name.StartsWith("Attachment_", StringComparison.OrdinalIgnoreCase))
                return HelperType.Attachment;
                
            if (Name.StartsWith("Origin_", StringComparison.OrdinalIgnoreCase))
                return HelperType.Origin;
                
            if (Name.StartsWith("Target_", StringComparison.OrdinalIgnoreCase))
                return HelperType.Target;
                
            if (Name.StartsWith("Hit_", StringComparison.OrdinalIgnoreCase))
                return HelperType.HitTest;
                
            if (Name.StartsWith("Mount_", StringComparison.OrdinalIgnoreCase))
                return HelperType.Mount;
                
            if (Name.StartsWith("Camera_", StringComparison.OrdinalIgnoreCase))
                return HelperType.Camera;
                
            if (Name.StartsWith("Ref_", StringComparison.OrdinalIgnoreCase))
                return HelperType.Reference;
                
            return HelperType.Generic;
        }
    }
    
    /// <summary>
    /// Gets whether this helper is an attachment point
    /// </summary>
    public bool IsAttachment => Type == HelperType.Attachment;
    
    /// <summary>
    /// Gets whether this helper is a spell cast origin
    /// </summary>
    public bool IsOrigin => Type == HelperType.Origin;
    
    /// <summary>
    /// Gets whether this helper is a mount point
    /// </summary>
    public bool IsMount => Type == HelperType.Mount;
}

public enum HelperType
{
    Generic,
    Attachment,
    Origin,
    Target,
    HitTest,
    Mount,
    Camera,
    Reference
} 