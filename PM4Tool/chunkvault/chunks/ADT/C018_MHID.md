# MHID (Map Height ID)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MHID chunk was introduced in Mists of Pandaria (MoP) and contains heightmap ID information. It maps terrain heightmap data to specific heightmap assets, allowing for more detailed and varied terrain rendering. This chunk allows the game to reference specific heightmap textures by their FileDataID.

## Structure

```csharp
public struct MHID
{
    public uint[] HeightmapFileDataIDs;  // Array of FileDataIDs for heightmaps
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| HeightmapFileDataIDs | uint[] | Array of FileDataIDs for heightmap assets used in this ADT |

## Dependencies

- **MCNK (C009)** - May reference heightmap IDs for specific terrain chunks
- **MCVT (S001)** - Contains the actual height values modified by the heightmap textures

## Implementation Notes

- The MHID chunk was added to separate heightmap data from texture references
- Each FileDataID points to a heightmap asset in the game's data files
- Heightmaps are used to add fine detail to terrain geometry beyond the base vertex heights
- A value of 0 indicates no heightmap for that slot
- The array size may vary based on the number of heightmaps used in the ADT

## Implementation Example

```csharp
public class HeightmapManager
{
    private Dictionary<uint, Texture2D> heightmapCache = new Dictionary<uint, Texture2D>();
    private FileDataService fileDataService;
    
    public HeightmapManager(FileDataService fileDataService)
    {
        this.fileDataService = fileDataService;
    }
    
    public void LoadHeightmaps(uint[] heightmapFileDataIDs)
    {
        foreach (uint fileDataID in heightmapFileDataIDs)
        {
            if (fileDataID == 0 || heightmapCache.ContainsKey(fileDataID))
                continue;
                
            try
            {
                byte[] fileData = fileDataService.GetFileData(fileDataID);
                Texture2D heightmap = new Texture2D(256, 256, TextureFormat.R16, false);
                heightmap.LoadRawTextureData(fileData);
                heightmap.Apply();
                
                heightmapCache[fileDataID] = heightmap;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load heightmap {fileDataID}: {ex.Message}");
            }
        }
    }
    
    public Texture2D GetHeightmap(uint fileDataID)
    {
        if (fileDataID == 0 || !heightmapCache.ContainsKey(fileDataID))
            return null;
            
        return heightmapCache[fileDataID];
    }
    
    // Apply heightmap details to mesh vertices
    public void ApplyHeightmapToMesh(Mesh mesh, uint heightmapID, float heightScale = 1.0f)
    {
        var heightmap = GetHeightmap(heightmapID);
        if (heightmap == null)
            return;
            
        Vector3[] vertices = mesh.vertices;
        
        for (int i = 0; i < vertices.Length; i++)
        {
            // Calculate UV coordinates for the vertex
            float u = vertices[i].x / 1600.0f; // ADT is 1600x1600 units
            float v = vertices[i].z / 1600.0f;
            
            // Sample the heightmap
            Color heightValue = heightmap.GetPixelBilinear(u, v);
            
            // Apply the height offset (using red channel as height)
            vertices[i].y += heightValue.r * heightScale;
        }
        
        mesh.vertices = vertices;
        mesh.RecalculateNormals();
    }
}
```

## Usage Context

The MHID chunk plays a crucial role in World of Warcraft's terrain rendering system, particularly for adding high-frequency detail to terrain without increasing the base mesh complexity.

Prior to Mists of Pandaria, terrain detail was limited by the resolution of the height map directly stored in the ADT files. With the introduction of the MHID chunk, the game can now reference external heightmap assets that provide additional detail through displacement mapping or other height-based rendering techniques.

These heightmaps can be used in several ways:
1. **Displacement mapping** - Modifying the actual mesh geometry during rendering for more detailed terrain
2. **Normal mapping** - Deriving surface normals from the heightmap to create the appearance of detailed surfaces
3. **Parallax mapping** - Creating the illusion of depth and detail without actually modifying the geometry

The separation of heightmap references into their own chunk allows for more efficient asset management and loading, as well as enabling the game to use different heightmap assets for different detail levels or rendering settings. This contributes to both the visual quality of the game world and the performance scalability across different hardware configurations. 