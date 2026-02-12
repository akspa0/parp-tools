# O005: MOGN

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOGN (Map Object Group Name) chunk contains a list of null-terminated strings that provide names for each of the groups in the WMO. These names are referenced from the MOGI chunk and are primarily used for development and debugging purposes. They help identify specific sections of a WMO during design and can be used for gameplay functionality like detecting when a player enters a named area.

## Structure
```csharp
struct MOGN
{
    char[] groupNames;  // Concatenated list of null-terminated group name strings
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| groupNames | char[] | A continuous block of null-terminated strings containing group names |

## Dependencies
- MOHD: The nGroups field indicates how many groups (and therefore group names) should be present
- MOGI: References group names by offset from the beginning of the MOGN chunk

## Implementation Notes
- Group names are stored as null-terminated strings in a continuous block
- Names are referenced by offset (in bytes) from the beginning of this chunk
- Unlike file paths, these names don't follow a specific format and are primarily for identification
- Empty names are represented by a single null byte (0x00)
- The total size of this chunk can vary significantly based on the length of group names
- Multiple consecutive null terminators may indicate empty names or padding
- This chunk follows the same storage pattern as MOTX (textures) and MODN (doodad names)
- The number of group names should match the nGroups field in the MOHD chunk

## Implementation Example
```csharp
public class MOGN : IChunk
{
    public byte[] RawData { get; private set; }
    public List<string> GroupNames { get; private set; }
    private Dictionary<string, int> _offsetLookup;
    
    public MOGN()
    {
        GroupNames = new List<string>();
        _offsetLookup = new Dictionary<string, int>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Store the raw data for calculating offsets later
        RawData = reader.ReadBytes((int)size);
        
        // Extract group names from the raw data
        GroupNames.Clear();
        _offsetLookup.Clear();
        
        int offset = 0;
        while (offset < RawData.Length)
        {
            // Find the null terminator
            int stringStart = offset;
            int stringEnd = stringStart;
            
            while (stringEnd < RawData.Length && RawData[stringEnd] != 0)
            {
                stringEnd++;
            }
            
            // Extract the group name as a string
            if (stringEnd > stringStart)
            {
                string groupName = Encoding.ASCII.GetString(RawData, stringStart, stringEnd - stringStart);
                GroupNames.Add(groupName);
                _offsetLookup[groupName] = stringStart;
            }
            else
            {
                // Empty string (just a null terminator)
                GroupNames.Add(string.Empty);
                _offsetLookup[string.Empty] = stringStart;
            }
            
            // Move past the null terminator to the next string
            offset = stringEnd + 1;
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Calculate the total size needed
        int totalSize = 0;
        foreach (string groupName in GroupNames)
        {
            totalSize += groupName.Length + 1; // +1 for null terminator
        }
        
        // Create a buffer to hold all group names
        byte[] buffer = new byte[totalSize];
        int offset = 0;
        
        // Reset the offset lookup
        _offsetLookup.Clear();
        
        // Write each group name to the buffer
        foreach (string groupName in GroupNames)
        {
            _offsetLookup[groupName] = offset;
            
            if (!string.IsNullOrEmpty(groupName))
            {
                byte[] nameBytes = Encoding.ASCII.GetBytes(groupName);
                Buffer.BlockCopy(nameBytes, 0, buffer, offset, nameBytes.Length);
                offset += nameBytes.Length;
            }
            
            // Add null terminator
            buffer[offset++] = 0;
        }
        
        // Write the buffer to the output
        writer.Write(buffer);
        
        // Update the raw data
        RawData = buffer;
    }
    
    public int GetOffsetForName(string groupName)
    {
        if (_offsetLookup.TryGetValue(groupName, out int offset))
        {
            return offset;
        }
        
        return -1; // Not found
    }
    
    public string GetNameByOffset(int offset)
    {
        if (offset < 0 || offset >= RawData.Length)
        {
            return string.Empty;
        }
        
        // Find end of string (null terminator)
        int end = offset;
        while (end < RawData.Length && RawData[end] != 0)
        {
            end++;
        }
        
        // Extract the string
        int length = end - offset;
        if (length > 0)
        {
            return Encoding.ASCII.GetString(RawData, offset, length);
        }
        
        return string.Empty;
    }
    
    public void AddGroupName(string groupName)
    {
        GroupNames.Add(groupName);
        // Note: offsets will be calculated during Write
    }
}
```

## Validation Requirements
- All group names should be properly null-terminated
- The number of group names should match the nGroups field in the MOHD chunk
- Group name offsets referenced from MOGI should be valid within the bounds of this chunk
- Group names should be ASCII-encoded text

## Usage Context
The MOGN chunk serves several important purposes in the WMO format:

1. **Development Reference**: Group names help developers identify specific parts of a model during creation and editing
2. **Debugging Support**: When troubleshooting rendering issues, names provide context for which part of the model is problematic
3. **Area Identification**: The game uses named groups to identify areas, such as rooms or zones within a building
4. **Gameplay Functionality**: Names are used to trigger events or apply effects when a player enters a named area

Common naming patterns include:
- Descriptive location names (e.g., "Entrance", "MainHall", "Basement")
- Functional identifiers (e.g., "Collision", "NonCollide", "Portal")
- Organizational markers (e.g., "Interior01", "Exterior", "Roof")

The client can use these names to determine:
- Which map area a player is currently in (for minimap and location text)
- Environmental settings to apply (lighting, sound, weather effects)
- Collision properties of different sections
- Visibility relationships between connected areas

While not directly visible to players, these names form an important part of the WMO's logical structure and are essential for many gameplay systems. 