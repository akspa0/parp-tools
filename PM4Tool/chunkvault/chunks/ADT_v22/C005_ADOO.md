# C005: ADOO

## Type
ADT v22 Chunk

## Source
ADT_v22.md

## Description
Contains M2 model and WMO filenames for objects placed in the terrain.

## Structure
```csharp
struct ADOO
{
    // An array of chunks, each containing a model filename
    // Both M2 and WMO filenames are stored here
    char filenames[]; // Zero-terminated strings with complete paths
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| filenames | char[] | Array of zero-terminated strings with model filenames |

## Dependencies
- AHDR (C001) - Provides overall file structure
- Referenced by: ACDO (in ACNK) - Model references use indices into this array

## Implementation Notes
- Size: Variable
- Contains both M2 model filenames and WMO filenames
- Unlike ADT v18, which separates M2 and WMO references into different chunks (MMDX/MWMO)
- Each filename is a zero-terminated string with a complete path
- Referenced by ACDO chunks within the ACNK chunks
- This consolidation simplifies the object reference system compared to v18

## Implementation Example
```csharp
public class ADOO
{
    public List<string> ModelFilenames { get; set; } = new List<string>();
    
    // Helper method to determine if a filename is an M2 or WMO
    public bool IsWMO(int index)
    {
        if (index < 0 || index >= ModelFilenames.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        // Check if the filename ends with .wmo (case insensitive)
        return ModelFilenames[index].EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);
    }
    
    // Helper method to get a model filename by index
    public string GetModelFilename(int index)
    {
        if (index < 0 || index >= ModelFilenames.Count)
            throw new ArgumentOutOfRangeException(nameof(index));
            
        return ModelFilenames[index];
    }
    
    // Helper method to find index of a model by filename
    public int FindModelIndex(string filename)
    {
        return ModelFilenames.FindIndex(m => m.Equals(filename, StringComparison.OrdinalIgnoreCase));
    }
    
    // Helper method to add a model and return its index
    public int AddModel(string filename)
    {
        // Check if we already have this model
        int existingIndex = FindModelIndex(filename);
        if (existingIndex >= 0)
            return existingIndex;
            
        // Add the new model
        ModelFilenames.Add(filename);
        return ModelFilenames.Count - 1;
    }
    
    // Helper methods to get M2 and WMO models separately
    public IEnumerable<string> GetM2Models()
    {
        return ModelFilenames.Where(f => f.EndsWith(".m2", StringComparison.OrdinalIgnoreCase));
    }
    
    public IEnumerable<string> GetWMOModels()
    {
        return ModelFilenames.Where(f => f.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase));
    }
}
```

## Usage Context
The ADOO chunk stores filenames for both M2 models (doodads) and WMO models (world map objects) that are placed in the terrain. This is a significant change from ADT v18, which separated these into MMDX (for M2) and MWMO (for WMO) chunks. In v22, all model filenames are stored in a single array, and object placements in ACDO reference these by index. This unified approach simplifies the model reference system. 