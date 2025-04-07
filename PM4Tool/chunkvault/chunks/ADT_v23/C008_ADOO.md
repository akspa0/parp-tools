# C008: ADOO

## Type
ADT v23 Chunk

## Source
ADT_v23.md

## Description
Doodad (M2 object) placement data for ADT v23 format. Contains information about M2 models placed in the terrain, enhanced with WoD-specific features like phasing and scaling improvements.

## Structure
```csharp
struct ADOO
{
    struct {
        float posX, posY, posZ;    // Object position in the world
        float rotX, rotY, rotZ;    // Rotation angles (radians)
        float scale;               // Uniform scale factor
        uint32 flags;              // Object flags (enhanced in WoD)
        uint32 nameId;             // Index of the model name in string block
        uint16 uniqueId;           // Unique object ID (added in WoD)
        uint16 phaseId;            // Phase ID for visibility (added in WoD)
        uint32 reserved[2];        // Reserved for future use (added in WoD)
    } entries[];                   // Array of doodad entries
    
    char modelNames[];             // Null-terminated model filenames
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| entries | struct[] | Array of doodad placement entries |
| modelNames | char[] | String block containing model filenames |

## Dependencies
- AHDR (C004) - References this chunk via offsets
- ACDO (S004) - References entries in this chunk by index

## Implementation Notes
- Size: Variable, depends on number of doodad entries and string length
- Enhanced in WoD (v23) with:
  - uniqueId field for object identification
  - phaseId field for phased content visibility
  - Additional flags for advanced object behavior
  - Reserved fields for future expansion
- Each entry defines position, rotation, scale, and model for an object
- Model filenames are stored in a separate string block
- Referenced by ACDO subchunks to determine which doodads belong to which terrain chunk
- Similar to ADOO in ADT v22, but with additional fields for WoD features

## Implementation Example
```csharp
[Flags]
public enum ADOOFlags
{
    None = 0,
    HideOnLowSettings = 0x1,    // Hide on low graphics settings
    AlphaBlended = 0x2,         // Uses alpha blending
    DontCastShadow = 0x4,       // Object doesn't cast shadows
    IsShadowOnly = 0x8,         // Object is shadow-only (not rendered)
    HasReflection = 0x10,       // Uses reflection maps
    HasLighting = 0x20,         // Uses dynamic lighting
    IsAreaTrigger = 0x40,       // Functions as an area trigger (WoD)
    UsesLodModel = 0x80,        // Uses level-of-detail models (WoD)
    IsDestructible = 0x100,     // Can be destroyed (WoD)
    HasPhysics = 0x200          // Has physics simulation (WoD)
}

public class ADOO
{
    public class DoodadEntry
    {
        // Position in world coordinates
        public System.Numerics.Vector3 Position { get; set; }
        
        // Rotation in radians
        public System.Numerics.Vector3 Rotation { get; set; }
        
        // Scale factor (uniform)
        public float Scale { get; set; }
        
        // Flags controlling rendering/behavior
        public ADOOFlags Flags { get; set; }
        
        // Index to model name in string block
        public int NameId { get; set; }
        
        // Actual model filename (filled in after parsing)
        public string ModelName { get; set; }
        
        // New fields in v23 (WoD)
        public ushort UniqueId { get; set; }   // Unique object identifier
        public ushort PhaseId { get; set; }    // Phase ID for visibility
        public uint[] Reserved { get; set; } = new uint[2]; // Reserved fields
        
        public DoodadEntry()
        {
            Position = new System.Numerics.Vector3();
            Rotation = new System.Numerics.Vector3();
            Scale = 1.0f;
        }
        
        // Convert rotation from Euler angles to quaternion
        public System.Numerics.Quaternion GetRotationQuaternion()
        {
            return System.Numerics.Quaternion.CreateFromYawPitchRoll(
                Rotation.Y, Rotation.X, Rotation.Z);
        }
        
        // Get model transformation matrix
        public System.Numerics.Matrix4x4 GetTransformMatrix()
        {
            var rotation = GetRotationQuaternion();
            
            return System.Numerics.Matrix4x4.CreateScale(Scale) *
                   System.Numerics.Matrix4x4.CreateFromQuaternion(rotation) *
                   System.Numerics.Matrix4x4.CreateTranslation(Position);
        }
        
        // Check if object is visible in a specific phase
        public bool IsVisibleInPhase(ushort phase)
        {
            // PhaseId of 0 means visible in all phases
            return PhaseId == 0 || PhaseId == phase;
        }
        
        // Helper method to get LOD model name based on distance
        public string GetLodModelName(float distance)
        {
            if (!UsesLodModel || distance <= 50.0f)
                return ModelName;
                
            // LOD models typically have _lod1, _lod2 suffixes
            int extIndex = ModelName.LastIndexOf('.');
            if (extIndex > 0)
            {
                string baseName = ModelName.Substring(0, extIndex);
                string extension = ModelName.Substring(extIndex);
                
                if (distance <= 100.0f)
                    return baseName + "_lod1" + extension;
                else
                    return baseName + "_lod2" + extension;
            }
            
            return ModelName;
        }
        
        // Helper properties for flags
        public bool HideOnLowSettings => (Flags & ADOOFlags.HideOnLowSettings) != 0;
        public bool UsesAlphaBlending => (Flags & ADOOFlags.AlphaBlended) != 0;
        public bool CastsShadow => (Flags & ADOOFlags.DontCastShadow) == 0;
        public bool IsShadowOnly => (Flags & ADOOFlags.IsShadowOnly) != 0;
        public bool HasReflection => (Flags & ADOOFlags.HasReflection) != 0;
        public bool UsesLighting => (Flags & ADOOFlags.HasLighting) != 0;
        public bool IsAreaTrigger => (Flags & ADOOFlags.IsAreaTrigger) != 0;
        public bool UsesLodModel => (Flags & ADOOFlags.UsesLodModel) != 0;
        public bool IsDestructible => (Flags & ADOOFlags.IsDestructible) != 0;
        public bool HasPhysics => (Flags & ADOOFlags.HasPhysics) != 0;
    }
    
    // List of doodad entries
    public List<DoodadEntry> Entries { get; private set; }
    
    // List of model filenames
    public List<string> ModelNames { get; private set; }
    
    public ADOO()
    {
        Entries = new List<DoodadEntry>();
        ModelNames = new List<string>();
    }
    
    public ADOO(List<DoodadEntry> entries, List<string> modelNames)
    {
        Entries = entries;
        ModelNames = modelNames;
        
        // Associate model names with entries
        foreach (var entry in Entries)
        {
            if (entry.NameId >= 0 && entry.NameId < ModelNames.Count)
            {
                entry.ModelName = ModelNames[entry.NameId];
            }
        }
    }
    
    // Add a new doodad entry
    public int AddDoodadEntry(DoodadEntry entry)
    {
        Entries.Add(entry);
        return Entries.Count - 1;
    }
    
    // Add a new model name
    public int AddModelName(string modelName)
    {
        // Check if the model name already exists
        int existingIndex = ModelNames.IndexOf(modelName);
        if (existingIndex >= 0)
            return existingIndex;
        
        ModelNames.Add(modelName);
        return ModelNames.Count - 1;
    }
    
    // Get all entries for a specific model
    public List<DoodadEntry> GetEntriesByModel(string modelName)
    {
        return Entries.Where(e => string.Equals(e.ModelName, modelName, 
                                             StringComparison.OrdinalIgnoreCase))
                     .ToList();
    }
    
    // Get entries visible in a specific phase
    public List<DoodadEntry> GetEntriesForPhase(ushort phase)
    {
        return Entries.Where(e => e.IsVisibleInPhase(phase)).ToList();
    }
    
    // Get entries within a specific area
    public List<DoodadEntry> GetEntriesInArea(float minX, float minZ, float maxX, float maxZ)
    {
        return Entries.Where(e => 
            e.Position.X >= minX && e.Position.X <= maxX &&
            e.Position.Z >= minZ && e.Position.Z <= maxZ)
                     .ToList();
    }
}
```

## Usage Context
The ADOO chunk contains placement data for M2 objects (doodads) in the terrain, such as trees, rocks, buildings, and other decorative elements. Each entry defines the position, rotation, scale, and model of a single object, with flags controlling its rendering behavior.

In the ADT v23 format used since Warlords of Draenor, the ADOO chunk has been enhanced with additional fields to support WoD's improved object management. The new uniqueId field provides a way to uniquely identify objects for gameplay interactions, while the phaseId field supports WoD's expanded phasing system, allowing objects to appear or disappear based on player progress or faction. The expanded flags include support for physics simulation, level-of-detail models, destructible objects, and area triggers.

During terrain rendering, the ACDO subchunks in each terrain chunk reference entries in the ADOO chunk to determine which objects should be placed in that chunk. This two-level reference system allows for efficient memory usage, as object data only needs to be stored once even if it's visible from multiple terrain chunks. 