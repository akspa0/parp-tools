# L007: MAHE

## Type
WDL Chunk

## Source
WDL_v18.md

## Description
The MAHE (Map HEight) chunk contains the actual heightmap data for a specific map area in the WDL format. This chunk stores a low-resolution grid of height values that represent the terrain elevation for distant rendering. The MAHE chunk is referenced by a MARE chunk through its heightMapOffset field, creating the relationship between area metadata and actual height data.

## Structure
```csharp
struct MAHE
{
    /*0x00*/ uint8_t heightValues[17][17]; // Height values in a 17×17 grid
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| heightValues | uint8_t[17][17] | Grid of height values representing terrain elevation |

## Height Data Format
- Each height value is a single unsigned byte (uint8_t)
- The grid is typically 17×17 (289 points) or 33×33 (1089 points)
- Values range from 0 to 255
- Raw values must be converted to actual heights using the scale and mean values from the MARE chunk
- The grid is stored in row-major order (row by row, starting from the top-left)

## Dependencies
- MARE (L006) - Contains metadata including the scale and mean values needed to interpret the height data
- MAOF (L005) - Indirectly related, as it contains offsets to the MARE chunks

## Implementation Notes
- Each MAHE chunk corresponds to one map area (defined by a MARE chunk)
- The raw height values need to be transformed using the heightScale and heightMean from the MARE chunk
- The formula is: actualHeight = (rawHeight * heightScale) + heightMean
- The 17×17 grid of heights represents the same geographical area as an ADT file, but at a much lower resolution
- For version 18 files, the grid size is usually 17×17
- Later versions may use different grid sizes (e.g., 33×33)
- The height grid forms a mesh of 16×16 quads (for a 17×17 grid)
- The height data is used for rendering distant terrain where detailed ADT data is not needed

## Implementation Example
```csharp
public class MAHE : IChunk
{
    public const int DEFAULT_GRID_SIZE = 17;
    
    public byte[,] HeightValues { get; private set; }
    public int GridSize { get; private set; }
    
    public MAHE(int gridSize = DEFAULT_GRID_SIZE)
    {
        GridSize = gridSize;
        HeightValues = new byte[GridSize, GridSize];
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Determine grid size based on chunk size
        // Common sizes are 17×17 = 289 and 33×33 = 1089
        int dataPoints = (int)size;
        GridSize = (int)Math.Sqrt(dataPoints);
        
        if (GridSize * GridSize != dataPoints)
            throw new InvalidDataException($"MAHE chunk has invalid size: {size} (not a perfect square)");
        
        HeightValues = new byte[GridSize, GridSize];
        
        // Read all height values in row-major order
        for (int y = 0; y < GridSize; y++)
        {
            for (int x = 0; x < GridSize; x++)
            {
                HeightValues[y, x] = reader.ReadByte();
            }
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        // Write all height values in row-major order
        for (int y = 0; y < GridSize; y++)
        {
            for (int x = 0; x < GridSize; x++)
            {
                writer.Write(HeightValues[y, x]);
            }
        }
    }
    
    // Helper method to get a height value at specific coordinates
    public byte GetRawHeight(int x, int y)
    {
        if (x < 0 || x >= GridSize || y < 0 || y >= GridSize)
            return 0;
            
        return HeightValues[y, x];
    }
    
    // Helper method to set a height value at specific coordinates
    public void SetRawHeight(int x, int y, byte height)
    {
        if (x < 0 || x >= GridSize || y < 0 || y >= GridSize)
            return;
            
        HeightValues[y, x] = height;
    }
    
    // Helper method to convert raw height to actual height using MARE data
    public float GetActualHeight(int x, int y, MARE mare)
    {
        byte rawHeight = GetRawHeight(x, y);
        return mare.ConvertHeight(rawHeight);
    }
    
    // Helper method to get a bilinearly interpolated height at non-integer coordinates
    public float GetInterpolatedHeight(float x, float y, MARE mare)
    {
        if (x < 0 || x >= GridSize - 1 || y < 0 || y >= GridSize - 1)
            return 0;
            
        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        
        float dx = x - x0;
        float dy = y - y0;
        
        float h00 = mare.ConvertHeight(GetRawHeight(x0, y0));
        float h10 = mare.ConvertHeight(GetRawHeight(x1, y0));
        float h01 = mare.ConvertHeight(GetRawHeight(x0, y1));
        float h11 = mare.ConvertHeight(GetRawHeight(x1, y1));
        
        // Bilinear interpolation
        float h0 = h00 * (1 - dx) + h10 * dx;
        float h1 = h01 * (1 - dx) + h11 * dx;
        
        return h0 * (1 - dy) + h1 * dy;
    }
}
```

## Terrain Representation
The heightmap in the MAHE chunk represents the terrain elevation for the corresponding map area:

- Each point represents the terrain height at that location
- The space between four adjacent points forms a quad (terrain face)
- The entire grid forms a continuous mesh of quads
- The density of points is much lower than in the full ADT data
- Typically 17×17 points (16×16 quads) for the same area as an ADT file, which has 145×145 points

## Height Conversion
Raw heights from the MAHE chunk must be converted to actual terrain heights:

1. Start with the raw height value (0-255) from the MAHE chunk
2. Apply the formula: actualHeight = (rawHeight * heightScale) + heightMean
3. The heightScale and heightMean values come from the corresponding MARE chunk

## Relationship to ADT
The MAHE heightmap provides a simplified version of the terrain defined in the MCVT chunks of the corresponding ADT file:

- MAHE: Single grid of 17×17 points for the entire map area
- MCVT: 9×9 grid of terrain chunks, each with 17×17 height points (145×145 total)

This significant reduction in detail is appropriate for distant terrain rendering, where the high resolution of ADT files is not needed and would be too memory-intensive for large view distances.

## Version Differences
- Version 18: Typically uses a 17×17 grid
- Later versions: May use different grid sizes (e.g., 33×33)

The actual grid size can be determined from the chunk size, as each point is a single byte. 