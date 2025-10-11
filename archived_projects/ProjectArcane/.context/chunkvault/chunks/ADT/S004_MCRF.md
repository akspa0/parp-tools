# S004: MCRF

## Type
MCNK Subchunk

## Source
ADT_v18.md

## Description
The MCRF (Map Chunk Reference) subchunk contains indices to doodad/object entries that appear in this map chunk. These indices reference entries in the MMID chunk, which in turn reference model filenames in the MMDX chunk.

## Structure
```csharp
struct MCRF
{
    /*0x00*/ uint32_t indices[doodadRefs];  // doodadRefs from MCNK header
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| indices | uint32[] | Array of doodad indices, size defined by doodadRefs in MCNK header |

## Dependencies
- MCNK (C018) - Parent chunk that contains this subchunk
- MMID (C006) - Contains offsets to model filenames in MMDX
- MMDX (C005) - Contains model filenames
- MDDF (C007) - Contains doodad placement data

## Index Interpretation
Each index in the MCRF array points to an entry in the MDDF (Model Doodad Definition) chunk:
- The index is a zero-based index into the MDDF array
- The corresponding MDDF entry contains:
  - nameId: Index into MMID (which points to a filename in MMDX)
  - position, rotation, scale, flags
- This indirection allows many instances of the same model to be placed efficiently

## Implementation Notes
- The size of the array is determined by the doodadRefs field in the MCNK header
- Indices are always into the MDDF chunk, not directly into MMID or MMDX
- All indices in MCRF should be within the bounds of the MDDF array
- MCRF entries represent only the doodads that appear in this specific map chunk
- A doodad may appear in multiple chunks' MCRF arrays if it spans chunk boundaries

## Implementation Example
```csharp
public class MCRF : IChunk
{
    public List<uint> DoodadIndices { get; set; } = new List<uint>();
    
    public void Parse(BinaryReader reader, uint doodadRefsCount)
    {
        for (int i = 0; i < doodadRefsCount; i++)
        {
            DoodadIndices.Add(reader.ReadUInt32());
        }
    }
    
    // Helper method to resolve indices to actual MDDF entries
    public List<MDDFEntry> ResolveDoodadReferences(List<MDDFEntry> mddfEntries)
    {
        List<MDDFEntry> references = new List<MDDFEntry>();
        
        foreach (uint index in DoodadIndices)
        {
            if (index < mddfEntries.Count)
            {
                references.Add(mddfEntries[(int)index]);
            }
            else
            {
                // Handle invalid reference
                Console.WriteLine($"Warning: Invalid MDDF reference {index}");
            }
        }
        
        return references;
    }
}
```

## Reference Chain

The full reference chain to get from an MCRF index to an actual model file:

```
MCRF.indices[i] → MDDF[index] → MMID[MDDF.nameId] → MMDX[offset] → "Model.m2"
```

In newer versions (8.1.0+), if the MDDF entry has the `mddf_entry_is_filedata_id` flag set (0x10):

```
MCRF.indices[i] → MDDF[index] → fileId → "Model.m2"
```

## Doodad Placement

This system allows the ADT format to efficiently place many instances of models:
1. Model filenames are stored only once in MMDX
2. MMID provides offsets into the MMDX strings
3. MDDF entries reference these models and specify placement
4. MCRF groups doodads by the map chunk they appear in

## Usage Context
The MCRF subchunk is used to determine which doodads (small objects like rocks, trees, mushrooms, etc.) appear in a specific section of the map. By organizing doodad references by map chunk, the game engine can quickly load only the objects needed for the area being rendered. This system allows for efficient memory usage and rendering of detailed environments with many objects. When combined with terrain data and textures, these doodads help create the rich, detailed world environments seen in World of Warcraft. 