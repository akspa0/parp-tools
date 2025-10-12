# GEOA - MDX Geometry Animation Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The GEOA (Geometry Animation) chunk defines animations that directly affect geoset properties. These animations can control the visibility, color, and alpha values of entire geosets, allowing for effects like fading in/out parts of the model, color pulsing, or conditional display of model components.

## Structure

```csharp
public struct GEOA
{
    /// <summary>
    /// Array of geoset animations
    /// </summary>
    // MDLGEOSETANIM geosetAnims[numGeosetAnims] follows
}

public struct MDLGEOSETANIM
{
    /// <summary>
    /// Alpha value for the geoset (static value, overridden by track if present)
    /// </summary>
    public float alpha;
    
    /// <summary>
    /// Flags controlling the animation behavior
    /// </summary>
    public uint flags;
    
    /// <summary>
    /// Color value for the geoset (RGB, static value, overridden by track if present)
    /// </summary>
    public Vector3 color;
    
    /// <summary>
    /// ID of the geoset to animate (index in the GEOS chunk)
    /// </summary>
    public uint geosetId;
    
    /// <summary>
    /// Animation data for color and alpha
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLGEOSETANIM Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | alpha | float | Static alpha value (0.0-1.0) |
| 0x04 | flags | uint | Animation flags (see Animation Flags) |
| 0x08 | color | Vector3 | Static color value (RGB, 0.0-1.0) |
| 0x14 | geosetId | uint | ID of the geoset to animate |
| 0x18 | ... | ... | Animation tracks follow |

## Animation Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | DropShadow | Geoset casts a shadow on the ground |
| 1 | Color | Use the color value/track for this geoset |
| 2-31 | Reserved | Reserved for future use |

## Animation Tracks
After the basic properties, animation tracks may follow:

- Alpha track (float, 0.0-1.0)
- Color track (Vector3 RGB, 0.0-1.0)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Same structure with additional usage patterns |

## Dependencies
- GEOS - References geosets by ID from the GEOS chunk
- MDLKEYTRACK - Used for animation tracks within the structure

## Implementation Notes
- Geoset animations are applied to entire geosets, not individual vertices
- The alpha value controls the transparency of the entire geoset
- The color value is multiplied with the underlying geoset's vertex colors
- If a track is present for alpha or color, it overrides the static value
- Geoset visibility can be achieved by animating alpha between 0 and 1
- Multiple geoset animations can affect different geosets simultaneously
- The DropShadow flag is only used in Warcraft 3 ground projections

## Usage Context
The GEOA chunk enables:
- Fading geosets in and out for visibility control
- Changing the color of geosets for effects like glowing
- Controlling which parts of a model cast shadows
- Creating complex visual effects by combining multiple animated geosets

## Visual Effects Examples
- Fading armor pieces in/out for equipment changes
- Glowing weapon effects with pulsing colors
- Conditional visibility for optional model parts
- Color shifts for magical effects or transformations

## Implementation Example

```csharp
public class GEOAChunk : IMdxChunk
{
    public string ChunkId => "GEOA";
    
    public List<MdxGeosetAnimation> GeosetAnimations { get; private set; } = new List<MdxGeosetAnimation>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear any existing animations
        GeosetAnimations.Clear();
        
        // Read animations until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var geoAnim = new MdxGeosetAnimation();
            
            // Read basic properties
            geoAnim.Alpha = reader.ReadSingle();
            geoAnim.Flags = reader.ReadUInt32();
            
            // Read color (RGB Vector3)
            geoAnim.Color = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
            
            geoAnim.GeosetId = reader.ReadUInt32();
            
            // Read animation tracks
            // Alpha track
            geoAnim.AlphaTrack = new MdxKeyTrack<float>();
            geoAnim.AlphaTrack.Parse(reader, r => r.ReadSingle());
            
            // Color track (only if Color flag is set)
            if (geoAnim.UseColor)
            {
                geoAnim.ColorTrack = new MdxKeyTrack<Vector3>();
                geoAnim.ColorTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            }
            
            GeosetAnimations.Add(geoAnim);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var geoAnim in GeosetAnimations)
        {
            // Write basic properties
            writer.Write(geoAnim.Alpha);
            writer.Write(geoAnim.Flags);
            
            // Write color
            writer.Write(geoAnim.Color.X);
            writer.Write(geoAnim.Color.Y);
            writer.Write(geoAnim.Color.Z);
            
            writer.Write(geoAnim.GeosetId);
            
            // Write animation tracks
            // Alpha track
            geoAnim.AlphaTrack.Write(writer, (w, f) => w.Write(f));
            
            // Color track (only if Color flag is set)
            if (geoAnim.UseColor && geoAnim.ColorTrack != null)
            {
                geoAnim.ColorTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            }
        }
    }
    
    /// <summary>
    /// Gets the current alpha value for a geoset
    /// </summary>
    /// <param name="geosetId">ID of the geoset</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The alpha value, or 1.0 if no animation exists</returns>
    public float GetGeosetAlpha(uint geosetId, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        foreach (var geoAnim in GeosetAnimations)
        {
            if (geoAnim.GeosetId == geosetId)
            {
                if (geoAnim.AlphaTrack.NumKeys > 0)
                {
                    return geoAnim.AlphaTrack.Evaluate(time, sequenceDuration, globalSequences);
                }
                else
                {
                    return geoAnim.Alpha;
                }
            }
        }
        
        return 1.0f; // Default alpha if no animation exists
    }
    
    /// <summary>
    /// Gets the current color value for a geoset
    /// </summary>
    /// <param name="geosetId">ID of the geoset</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The color value, or white if no animation exists</returns>
    public Vector3 GetGeosetColor(uint geosetId, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        foreach (var geoAnim in GeosetAnimations)
        {
            if (geoAnim.GeosetId == geosetId && geoAnim.UseColor)
            {
                if (geoAnim.ColorTrack != null && geoAnim.ColorTrack.NumKeys > 0)
                {
                    return geoAnim.ColorTrack.Evaluate(time, sequenceDuration, globalSequences);
                }
                else
                {
                    return geoAnim.Color;
                }
            }
        }
        
        return new Vector3(1, 1, 1); // Default color (white) if no animation exists
    }
}

public class MdxGeosetAnimation
{
    public float Alpha { get; set; } = 1.0f;
    public uint Flags { get; set; }
    public Vector3 Color { get; set; } = new Vector3(1, 1, 1);
    public uint GeosetId { get; set; }
    public MdxKeyTrack<float> AlphaTrack { get; set; }
    public MdxKeyTrack<Vector3> ColorTrack { get; set; }
    
    // Flag accessors
    public bool DropShadow => (Flags & 0x1) != 0;
    public bool UseColor => (Flags & 0x2) != 0;
    
    // Helper property - determines if geoset is visible at a given alpha threshold
    public bool IsVisibleAtAlpha(float alpha, float threshold = 0.01f)
    {
        return alpha >= threshold;
    }
}
``` 