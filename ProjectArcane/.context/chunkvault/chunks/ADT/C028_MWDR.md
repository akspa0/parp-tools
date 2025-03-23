# MWDR (Map WMO Doodad References)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MWDR chunk was introduced in Shadowlands and contains references to doodad instances that should be placed in the world. This chunk replaces the older MMID and MWID chunks with a more efficient storage format for model references. It uses FileDataIDs instead of filenames, directly referencing models in the game's database.

## Structure

```csharp
public struct MWDR
{
    public SMDoodadDef[] DoodadDefinitions;  // Array of doodad definitions
}

public struct SMDoodadDef
{
    public uint FileDataID;         // FileDataID of the model (M2 or WMO)
    public Vector3 Position;        // Position in world space (X, Y, Z)
    public Vector3 Rotation;        // Rotation as quaternion (X, Y, Z components)
    public float RotationW;         // W component of the quaternion rotation
    public float Scale;             // Uniform scale factor
    public uint Flags;              // Placement flags
    public uint SetID;              // Set ID for grouping doodads
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| FileDataID | uint | Direct reference to the model file in the game's database |
| Position | Vector3 | 3D world position (X, Y, Z) of the doodad |
| Rotation | Vector3 | X, Y, Z components of the quaternion rotation |
| RotationW | float | W component of the quaternion rotation |
| Scale | float | Uniform scale factor applied to the model |
| Flags | uint | Bit flags controlling various aspects of the doodad's appearance and behavior |
| SetID | uint | Identifier for the doodad set this instance belongs to |

## Flags Definition

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x1 | MDoodadFlag_HasTerrainBlend | Doodad has terrain blending |
| 0x2 | MDoodadFlag_SortByMapObjID | Sort by map object ID |
| 0x4 | MDoodadFlag_UseLightmapOnlyColor | Use only the lightmap color |
| 0x8 | MDoodadFlag_UseLargeObjectCull | Use large object culling |
| 0x10 | MDoodadFlag_IsLightSource | Doodad emits light |
| 0x20 | MDoodadFlag_FixForAllowedLiquids | Fix for allowed liquids |
| 0x40 | MDoodadFlag_DoodadLarge | Doodad is considered large |
| 0x80 | MDoodadFlag_DoodadUseElevation | Use the ground elevation for placement |
| 0x100 | MDoodadFlag_FixIsNotPrimaryColor | Fix is not primary color |
| 0x200 | MDoodadFlag_FixIsNotSecondaryColor | Fix is not secondary color |
| 0x400 | MDoodadFlag_IgnoreHeightmap | Ignore the terrain heightmap when placing |
| 0x800 | MDoodadFlag_HasTintColor | Doodad uses custom tint color |

## Dependencies

- **MWDS (C029)** - Contains doodad set definitions that group MWDR entries
- **MCNK (C009)** - May reference doodad sets to be rendered with specific terrain chunks

## Implementation Notes

- The MWDR chunk was introduced to replace the older MMID/MWID system in Shadowlands
- The chunk uses a more compact and efficient representation with FileDataIDs
- Quaternion rotation provides more precise orientation control than the older Euler angles
- Each doodad definition is self-contained with all placement information
- The SetID field allows for grouping doodads into sets for selective rendering
- Not all doodad instances may be visible at the same time (controlled by doodad sets)
- The FileDataID directly references either an M2 or WMO file in the game's database

## Implementation Example

```csharp
public class DoodadManager
{
    private Dictionary<uint, List<DoodadInstance>> doodadsBySet = new Dictionary<uint, List<DoodadInstance>>();
    private Dictionary<uint, GameObject> loadedModels = new Dictionary<uint, GameObject>();
    private FileDataService fileDataService;
    
    public DoodadManager(FileDataService fileDataService)
    {
        this.fileDataService = fileDataService;
    }
    
    // Load doodad definitions from MWDR chunk
    public void LoadDoodadDefinitions(MWDR mwdrChunk)
    {
        if (mwdrChunk == null || mwdrChunk.DoodadDefinitions == null)
            return;
            
        foreach (var def in mwdrChunk.DoodadDefinitions)
        {
            // Create doodad instance from definition
            var instance = new DoodadInstance
            {
                FileDataID = def.FileDataID,
                Position = def.Position,
                Rotation = new Quaternion(def.Rotation.X, def.Rotation.Y, def.Rotation.Z, def.RotationW),
                Scale = def.Scale,
                Flags = def.Flags,
                SetID = def.SetID
            };
            
            // Add to set collection
            if (!doodadsBySet.TryGetValue(def.SetID, out var setList))
            {
                setList = new List<DoodadInstance>();
                doodadsBySet[def.SetID] = setList;
            }
            
            setList.Add(instance);
        }
    }
    
    // Load model for a doodad
    private async Task<GameObject> LoadDoodadModel(uint fileDataID)
    {
        if (loadedModels.TryGetValue(fileDataID, out var existingModel))
            return existingModel;
            
        // Load model data from file service
        byte[] fileData = await fileDataService.GetFileDataAsync(fileDataID);
        
        // Determine if this is an M2 or WMO and load appropriately
        GameObject model;
        if (IsM2Model(fileData))
        {
            model = await LoadM2Model(fileData);
        }
        else
        {
            model = await LoadWMOModel(fileData);
        }
        
        loadedModels[fileDataID] = model;
        return model;
    }
    
    // Spawn doodads for a specific set
    public async Task SpawnDoodadSet(uint setID)
    {
        if (!doodadsBySet.TryGetValue(setID, out var setList))
            return;
            
        foreach (var instance in setList)
        {
            var model = await LoadDoodadModel(instance.FileDataID);
            
            // Clone model and apply transformation
            var doodadObject = GameObject.Instantiate(model);
            doodadObject.transform.position = instance.Position;
            doodadObject.transform.rotation = instance.Rotation;
            doodadObject.transform.localScale = Vector3.one * instance.Scale;
            
            // Apply flags-based settings
            ApplyDoodadFlags(doodadObject, instance.Flags);
        }
    }
    
    // Apply flag-based settings to a doodad
    private void ApplyDoodadFlags(GameObject doodadObject, uint flags)
    {
        // Set up terrain blending if needed
        if ((flags & 0x1) != 0) // MDoodadFlag_HasTerrainBlend
        {
            EnableTerrainBlending(doodadObject);
        }
        
        // Set up lighting properties
        if ((flags & 0x10) != 0) // MDoodadFlag_IsLightSource
        {
            AddLightComponent(doodadObject);
        }
        
        // Set up ground snapping
        if ((flags & 0x80) != 0) // MDoodadFlag_DoodadUseElevation
        {
            SnapToGround(doodadObject);
        }
        
        // Handle large object culling
        if ((flags & 0x8) != 0) // MDoodadFlag_UseLargeObjectCull
        {
            SetLargeObjectCulling(doodadObject);
        }
    }
    
    // Helper methods (implementation would depend on game engine)
    private bool IsM2Model(byte[] fileData) => 
        fileData.Length >= 4 && fileData[0] == 'M' && fileData[1] == 'D' && fileData[2] == '2' && fileData[3] == '0';
        
    private Task<GameObject> LoadM2Model(byte[] fileData) => 
        throw new NotImplementedException("Would load an M2 model from binary data");
        
    private Task<GameObject> LoadWMOModel(byte[] fileData) => 
        throw new NotImplementedException("Would load a WMO model from binary data");
        
    private void EnableTerrainBlending(GameObject obj) => 
        throw new NotImplementedException("Would set up terrain blending for this object");
        
    private void AddLightComponent(GameObject obj) => 
        throw new NotImplementedException("Would add a light component to this object");
        
    private void SnapToGround(GameObject obj) => 
        throw new NotImplementedException("Would snap object to ground height");
        
    private void SetLargeObjectCulling(GameObject obj) => 
        throw new NotImplementedException("Would set up special culling for large objects");
}

public class DoodadInstance
{
    public uint FileDataID { get; set; }
    public Vector3 Position { get; set; }
    public Quaternion Rotation { get; set; }
    public float Scale { get; set; }
    public uint Flags { get; set; }
    public uint SetID { get; set; }
}
```

## Usage Context

The MWDR chunk represents a significant evolution in World of Warcraft's approach to doodad (props and decorative elements) placement in the game world. Prior to Shadowlands, doodads were referenced through a combination of the MMID and MWID chunks, which used filename strings and separate placement information.

The introduction of the MWDR chunk in Shadowlands provides several key advantages:

1. **Performance Optimization**: Direct FileDataIDs are more efficient than string lookups
2. **Reduced File Size**: Compact binary representation requires less storage
3. **Precision Improvements**: Quaternion rotations provide more accurate orientation than Euler angles
4. **Integration with Doodad Sets**: The SetID field allows for more efficient grouping and selective rendering
5. **Unified Format**: Both M2 and WMO references use the same structure

In the game world, doodads placed via the MWDR chunk include a wide variety of objects:

- **Vegetation**: Trees, bushes, flowers, and other plant life
- **Rocks and Terrain Features**: Boulders, cave formations, and natural elements
- **Props**: Furniture, containers, and interactive objects
- **Debris**: Scattered items, wreckage, and environmental storytelling elements
- **Small Structures**: Fences, wells, and other minor constructions

The ability to group doodads into sets via the SetID field allows the game engine to selectively load and render specific groups of objects based on various criteria, such as distance from the player, graphics settings, or gameplay requirements. This contributes to both performance optimization and the ability to create varied visual experiences within the same map area.

The transition to the MWDR chunk format in Shadowlands reflects Blizzard's continued efforts to modernize World of Warcraft's engine while maintaining backward compatibility with older content. 