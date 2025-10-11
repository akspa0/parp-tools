# MOSB - WMO Skybox

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOSB chunk defines the skybox model (M2) used for this WMO. A skybox provides the background visuals (sky, clouds, stars, etc.) that are visible when looking beyond the physical world model. The skybox appears as a distant background that moves relative to the camera to create the illusion of a vast environment.

## Structure

```csharp
public struct MOSB
{
    public char[] skyboxFilename; // Null-terminated string containing the M2 model filename for the skybox
}
```

## Properties

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | skyboxFilename | char[] | Null-terminated string containing the filename of the M2 model to use as the skybox for this WMO. The string length is variable and determined by the chunk size minus 8 bytes (for the chunk identifier and size fields). |

## Dependencies
- **MOHD**: The header chunk may contain skybox-related flags that affect how the skybox is rendered.

## Implementation Notes
- The size of this chunk is variable and depends on the length of the skybox filename string.
- The skybox filename is stored as a null-terminated string within the chunk.
- If the chunk is present but contains an empty string (just a null terminator), it indicates that no skybox should be displayed.
- The filename is typically relative to the World of Warcraft data path and points to an M2 model file.
- Common skybox models include various themed skies (e.g., "World\\Sky\\[skyname].m2").
- Some WMOs might not include a MOSB chunk if they don't have a custom skybox.

## Implementation Example

```csharp
public class MOSBChunk : IWmoChunk
{
    public string ChunkId => "MOSB";
    public string SkyboxFilename { get; set; }

    public void Read(BinaryReader reader, uint size)
    {
        // Read the skybox filename as a null-terminated string
        List<byte> bytes = new List<byte>();
        byte b;
        while ((b = reader.ReadByte()) != 0)
        {
            bytes.Add(b);
        }
        SkyboxFilename = Encoding.ASCII.GetString(bytes.ToArray());
        
        // Skip any remaining bytes in the chunk
        long bytesRead = bytes.Count + 1; // +1 for null terminator
        if (bytesRead < size)
        {
            reader.BaseStream.Position += (size - bytesRead);
        }
    }

    public void Write(BinaryWriter writer)
    {
        // Write the chunk header
        writer.Write(ChunkUtils.GetChunkIdBytes(ChunkId));
        
        // Calculate the size of the chunk data
        uint dataSize = (uint)(SkyboxFilename?.Length ?? 0) + 1; // +1 for null terminator
        writer.Write(dataSize);
        
        // Write the skybox filename
        if (!string.IsNullOrEmpty(SkyboxFilename))
        {
            writer.Write(Encoding.ASCII.GetBytes(SkyboxFilename));
        }
        writer.Write((byte)0); // Write null terminator
    }
}
```

## Validation Requirements
- The skybox filename, if present, must be a valid file path for an M2 model.
- The string must be properly null-terminated.
- The file path should use backslashes as directory separators (Windows style).
- The skybox model should exist and be a valid M2 file.

## Usage Context
- **Environmental Context:** The skybox provides the background environment for the WMO, establishing the time of day, weather conditions, and general ambiance.
- **Immersion:** A properly matched skybox enhances the player's immersion by providing a seamless transition between the world model and the surrounding environment.
- **Interior vs. Exterior:** Different skyboxes might be used depending on whether the WMO represents an interior or exterior space.
- **Visual Consistency:** The skybox helps maintain visual consistency across the game world by providing standardized sky appearances.
- **Customization:** Special WMOs (like dungeons or raid instances) might have unique skyboxes to create specific atmospheric effects. 