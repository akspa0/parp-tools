# WA02: MAOI

## Type
Alpha WDT Chunk

## Source
Alpha.md

## Description
The MAOI (Map Object Information) chunk contains the actual map object data for the Alpha WDT format. While the MAOT chunk provides an index to this data, the MAOI chunk holds the terrain data itself, effectively embedding what would later become ADT files directly within the WDT file. This chunk represents the self-contained approach of the Alpha WDT format.

## Structure
```csharp
struct MAOI
{
    /*0x00*/ byte data[];  // Raw binary data of map objects
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| data | byte[] | Raw binary data containing the map object information |

## Data Organization
The MAOI chunk is essentially a data container with the following structure:

1. The chunk contains concatenated blocks of map object data
2. Each block corresponds to an entry in the MAOT chunk
3. The MAOT entries provide:
   - Size of each block
   - Offset to the start of each block within the MAOI data array
4. Blocks may contain height maps, texture information, object placements, and other terrain data
5. The format of each block is complex and likely similar to what would later become ADT format

## Dependencies
- MAOT (WA01) - Contains the offsets and sizes for accessing data in this chunk
- MOTX (WA04) - May contain texture information referenced by data in this chunk
- MAOH (WA03) - Contains header information related to the map objects

## Implementation Notes
- The MAOI chunk lacks a formal structure beyond a container of raw binary data
- The interpretation of this data relies on offsets and sizes provided by the MAOT chunk
- Each block of data likely has its own internal structure for terrain information
- This chunk essentially stores embedded ADT-like data directly in the WDT file
- Direct access to specific map objects requires first consulting the MAOT chunk for offset and size information
- The data likely includes terrain height information, texture references, and object placements
- This monolithic approach was later replaced by the modular system of separate ADT files

## Implementation Example
```csharp
public class MAOI : IChunk
{
    public byte[] Data { get; private set; }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Read the entire chunk as a byte array
        Data = reader.ReadBytes((int)size);
    }
    
    public void Write(BinaryWriter writer)
    {
        writer.Write(Data);
    }
    
    // Helper method to extract a specific map object's data using info from MAOT
    public byte[] GetObjectData(int offset, int size)
    {
        if (offset < 0 || offset + size > Data.Length)
            throw new ArgumentOutOfRangeException($"Invalid offset/size: {offset}/{size} (data length: {Data.Length})");
            
        byte[] objectData = new byte[size];
        Array.Copy(Data, offset, objectData, 0, size);
        return objectData;
    }
    
    // Helper method to extract a specific object using its MAOT entry
    public byte[] GetObjectData(SMMapObjDef mapObj)
    {
        return GetObjectData((int)mapObj.Offset, (int)mapObj.Size);
    }
    
    // Helper method to parse a specific map object into a more structured format
    // This is a simplified example - the actual implementation would be more complex
    public MapObjectData ParseObjectData(byte[] objectData)
    {
        // Create a memory stream from the object data
        using (MemoryStream ms = new MemoryStream(objectData))
        using (BinaryReader reader = new BinaryReader(ms))
        {
            // A hypothetical representation of parsed map object data
            var data = new MapObjectData();
            
            // Parse the object data according to its internal structure
            // This structure would need to be determined through reverse engineering
            // and would likely contain terrain heights, texture information, etc.
            
            // Example parsing logic (simplified):
            // data.HeightMap = ParseHeightMap(reader);
            // data.TextureInfo = ParseTextureInfo(reader);
            // data.ObjectPlacements = ParseObjectPlacements(reader);
            
            return data;
        }
    }
}

// A hypothetical class representing structured map object data
public class MapObjectData
{
    // This would contain parsed and structured information from the raw object data
    // Similar to what would later become ADT structure
    public float[] HeightMap { get; set; }
    public TextureInfo[] Textures { get; set; }
    public ObjectPlacement[] Objects { get; set; }
    // Other terrain properties
}
```

## Data Block Contents
Based on the Alpha file format, each data block in the MAOI chunk likely contains:

1. **Terrain Height Data**: Similar to MCVT in later ADT format
2. **Texture Information**: References to textures, possibly indexed into MOTX
3. **Object Placements**: Positions of models and objects in the terrain
4. **Terrain Properties**: Ground types, flags, and other properties
5. **Rendering Information**: Data needed for rendering the terrain

The exact structure of these blocks would require detailed reverse engineering of the Alpha format.

## Version Information
- Present only in the Alpha version of the WDT format
- In later versions, this embedded data approach was replaced by separate ADT files
- Represents a historical snapshot of Blizzard's early approach to world data storage

## Architectural Significance
The MAOI chunk, along with MAOT, forms the core of the self-contained approach in Alpha WDT:

1. **Data Encapsulation**: All terrain data is contained within a single file
2. **Direct Embedding**: Instead of references to external files, data is directly embedded
3. **Single-File Architecture**: The entire world (or large sections of it) exists in one file
4. **Historical Significance**: Shows how the WoW engine evolved from a monolithic to a modular approach

This monolithic design was eventually replaced by the more modular approach with separate ADT files, likely to improve memory usage, loading times, and development workflow. 