# MWDS (Map WMO Doodad Sets)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MWDS chunk was introduced in Shadowlands and contains definitions for doodad sets that group related doodad instances together. This chunk works in conjunction with the MWDR chunk to provide a more efficient and flexible system for managing decorative objects in the game world. Doodad sets allow the game to selectively show or hide groups of objects based on various criteria.

## Structure

```csharp
public struct MWDS
{
    public SMDoodadSet[] DoodadSets;  // Array of doodad set definitions
}

public struct SMDoodadSet
{
    public uint SetID;              // Unique identifier for this doodad set
    public string Name;             // Human-readable name of the set
    public uint Flags;              // Bit flags controlling the set's behavior
    public uint StartIndex;         // Starting index in the MWDR array
    public uint Count;              // Number of doodads in this set
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| SetID | uint | Unique identifier that matches the SetID field in MWDR entries |
| Name | string | Human-readable name for the set (e.g., "Default", "Winter", "Destroyed") |
| Flags | uint | Bit flags defining set behavior and visibility conditions |
| StartIndex | uint | Starting index in the MWDR doodad array for this set's doodads |
| Count | uint | Number of consecutive doodads from StartIndex that belong to this set |

## Flags Definition

| Flag Value | Name | Description |
|------------|------|-------------|
| 0x1 | MSetFlag_Default | This is the default doodad set shown at map load |
| 0x2 | MSetFlag_PhaseA | Set is shown in phase A |
| 0x4 | MSetFlag_PhaseB | Set is shown in phase B |
| 0x8 | MSetFlag_InvisibleUntilDynamic | Invisible until triggered by a dynamic event |
| 0x10 | MSetFlag_Seasonal | Set is only visible during specific seasons |
| 0x20 | MSetFlag_Skybox | Set is used for skybox/distant view decorations |
| 0x40 | MSetFlag_EnvironmentSpecific | Set is specific to a certain environment condition |
| 0x80 | MSetFlag_QuestState | Set visibility is controlled by quest state |
| 0x100 | MSetFlag_EventSpecific | Set is used during specific game events |
| 0x200 | MSetFlag_RareSpawnOnly | Set is only visible when rare spawns are active |
| 0x400 | MSetFlag_HighDetailOnly | Set is only shown on high graphics settings |

## Dependencies

- **MWDR (C028)** - Contains the actual doodad instances referenced by doodad sets
- **MCNK (C009)** - May reference specific doodad sets for terrain chunks

## Implementation Notes

- The MWDS chunk was introduced alongside MWDR in Shadowlands
- Doodad sets allow for more efficient organization and conditional rendering of objects
- Multiple doodad sets can reference overlapping sets of doodads
- The game engine can selectively activate or deactivate sets based on game state
- The StartIndex and Count fields define a range within the MWDR chunk's doodad array
- Doodad sets often represent alternative world states (seasonal, damaged, phased)
- The default set (with MSetFlag_Default) is loaded when no specific conditions apply

## Implementation Example

```csharp
public class DoodadSetManager
{
    private Dictionary<uint, DoodadSetInfo> doodadSets = new Dictionary<uint, DoodadSetInfo>();
    private DoodadManager doodadManager;
    private uint activePhaseFlags;
    private bool isHighDetailEnabled;
    private GameEventManager eventManager;
    
    public DoodadSetManager(DoodadManager doodadManager, GameEventManager eventManager)
    {
        this.doodadManager = doodadManager;
        this.eventManager = eventManager;
        this.isHighDetailEnabled = QualitySettings.GetQualityLevel() >= 3; // Determine if high detail is enabled
    }
    
    // Load doodad set definitions from MWDS chunk
    public void LoadDoodadSets(MWDS mwdsChunk)
    {
        if (mwdsChunk == null || mwdsChunk.DoodadSets == null)
            return;
            
        foreach (var set in mwdsChunk.DoodadSets)
        {
            doodadSets[set.SetID] = new DoodadSetInfo
            {
                SetID = set.SetID,
                Name = set.Name,
                Flags = set.Flags,
                StartIndex = set.StartIndex,
                Count = set.Count,
                IsActive = (set.Flags & 0x1) != 0 // Default sets start active
            };
        }
    }
    
    // Activate appropriate doodad sets based on current game state
    public async Task UpdateActiveSets()
    {
        // First hide all doodads
        await HideAllDoodads();
        
        // Then activate appropriate sets
        foreach (var setInfo in doodadSets.Values)
        {
            bool shouldActivate = ShouldActivateSet(setInfo);
            
            if (shouldActivate && !setInfo.IsActive)
            {
                await ActivateSet(setInfo.SetID);
                setInfo.IsActive = true;
            }
            else if (!shouldActivate && setInfo.IsActive)
            {
                await DeactivateSet(setInfo.SetID);
                setInfo.IsActive = false;
            }
        }
    }
    
    // Determine if a doodad set should be active based on its flags and current game state
    private bool ShouldActivateSet(DoodadSetInfo setInfo)
    {
        // Default set is active when no other conditions apply
        if ((setInfo.Flags & 0x1) != 0) // MSetFlag_Default
            return true;
            
        // Check phase flags
        if ((setInfo.Flags & 0x2) != 0 && (activePhaseFlags & 0x2) == 0) // MSetFlag_PhaseA
            return false;
            
        if ((setInfo.Flags & 0x4) != 0 && (activePhaseFlags & 0x4) == 0) // MSetFlag_PhaseB
            return false;
            
        // Check high detail flag
        if ((setInfo.Flags & 0x400) != 0 && !isHighDetailEnabled) // MSetFlag_HighDetailOnly
            return false;
            
        // Check event-specific flag
        if ((setInfo.Flags & 0x100) != 0) // MSetFlag_EventSpecific
        {
            // In a real implementation, we would check if the specific event is active
            if (!eventManager.IsEventActiveForSet(setInfo.SetID))
                return false;
        }
        
        // Check seasonal flag
        if ((setInfo.Flags & 0x10) != 0) // MSetFlag_Seasonal
        {
            if (!IsDoodadSetInSeason(setInfo.SetID))
                return false;
        }
        
        // If we passed all checks, the set should be active
        return true;
    }
    
    // Activate a specific doodad set
    private async Task ActivateSet(uint setID)
    {
        if (doodadSets.TryGetValue(setID, out var setInfo))
        {
            await doodadManager.SpawnDoodadSet(setID);
        }
    }
    
    // Deactivate a specific doodad set
    private Task DeactivateSet(uint setID)
    {
        if (doodadSets.TryGetValue(setID, out var setInfo))
        {
            doodadManager.HideDoodadSet(setID);
        }
        
        return Task.CompletedTask;
    }
    
    // Hide all doodads from all sets
    private Task HideAllDoodads()
    {
        doodadManager.HideAllDoodads();
        return Task.CompletedTask;
    }
    
    // Set the current phase flags
    public async Task SetPhaseFlags(uint phaseFlags)
    {
        if (activePhaseFlags != phaseFlags)
        {
            activePhaseFlags = phaseFlags;
            await UpdateActiveSets();
        }
    }
    
    // Check if a doodad set is appropriate for the current season
    private bool IsDoodadSetInSeason(uint setID)
    {
        // In a real implementation, this would check the current game season
        // and determine if this set should be shown
        return eventManager.IsDoodadSetInCurrentSeason(setID);
    }
}

public class DoodadSetInfo
{
    public uint SetID { get; set; }
    public string Name { get; set; }
    public uint Flags { get; set; }
    public uint StartIndex { get; set; }
    public uint Count { get; set; }
    public bool IsActive { get; set; }
}
```

## Usage Context

The MWDS chunk plays a pivotal role in World of Warcraft's dynamic world system, allowing developers to create multiple versions of an area's decorative elements that can be swapped in and out based on various game conditions. This capability is essential for creating a world that can change and evolve while maintaining performance.

In Shadowlands and beyond, the doodad set system enables several key gameplay and visual features:

1. **Phased Content**: Different players can see different versions of the world based on their quest progress or story choices
2. **Seasonal Changes**: The game world can reflect seasonal events like Winter Veil, Hallow's End, or Lunar Festival
3. **Dynamic Events**: Areas can change in response to world events, faction control, or server-wide progress
4. **Performance Scaling**: Lower-end hardware can show fewer decorative elements while maintaining the essential world structure
5. **Progressive World Changes**: Zones can evolve over time, showing damage, regrowth, construction, or other changes

Prior to the introduction of the MWDS/MWDR system, these features were implemented through more complex and less efficient methods. The new format provides a more structured and optimized approach, particularly when combined with the FileDataID references in MWDR.

Some practical examples of doodad sets in the game include:

- **Alternate Versions**: War-torn vs. peaceful versions of the same location
- **Weather Effects**: Additional objects that appear during rain, snow, or other weather conditions
- **Time of Day**: Different lighting and atmospheric elements for day vs. night
- **Faction Control**: Alliance vs. Horde decorations in contested territories
- **Progressive Construction**: Buildings that appear to be under construction, complete, or damaged

The MWDS chunk, working in conjunction with MWDR, enables these dynamic world elements while maintaining efficient loading and rendering, ensuring that World of Warcraft's environments can be both varied and performant across a wide range of hardware capabilities. 