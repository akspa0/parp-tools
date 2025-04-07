# RIBB - MDX Ribbon Emitters Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The RIBB (Ribbon Emitters) chunk defines ribbon-like visual effects such as weapon trails, spell casts, and flowing energy streams. Unlike particle emitters that produce individual particles, ribbon emitters generate continuous strips of connected segments that can trail behind moving objects, creating smooth curved paths in 3D space. Ribbons are useful for effects like sword swings, magic trails, and energy beams.

## Structure

```csharp
public struct RIBB
{
    /// <summary>
    /// Array of ribbon emitter definitions
    /// </summary>
    // MDLRIBBONEMITTER emitters[numEmitters] follows
}

public struct MDLRIBBONEMITTER : MDLGENOBJECT
{
    /// <summary>
    /// Number of slices (segments) in the ribbon
    /// </summary>
    public uint slices;
    
    /// <summary>
    /// Material ID from the MTLS chunk
    /// </summary>
    public uint materialId;
    
    /// <summary>
    /// Width of the ribbon at the base (attached to model)
    /// </summary>
    public float baseWidth;
    
    /// <summary>
    /// Width of the ribbon at the edges (tips)
    /// </summary>
    public float edgeWidth;
    
    /// <summary>
    /// Base color (RGBA) where the ribbon attaches to the model
    /// </summary>
    public uint baseColor;
    
    /// <summary>
    /// Edge color (RGBA) for the tips of the ribbon
    /// </summary>
    public uint edgeColor;
    
    /// <summary>
    /// Above/below parameter
    /// </summary>
    public float edgeParameter;
    
    /// <summary>
    /// Texture slot used by the ribbon
    /// </summary>
    public uint textureSlot;
    
    /// <summary>
    /// Visibility of the ribbon (0 = invisible, 1 = visible)
    /// </summary>
    public uint visibility;
    
    /// <summary>
    /// Animation data for the ribbon properties
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLRIBBONEMITTER Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00..0x58 | MDLGENOBJECT | struct | Base generic object (see MDLGENOBJECT structure) |
| 0x58 | slices | uint | Number of segments along the ribbon |
| 0x5C | materialId | uint | ID of material in MTLS chunk |
| 0x60 | baseWidth | float | Width at the base (start) of the ribbon |
| 0x64 | edgeWidth | float | Width at the edge (end) of the ribbon |
| 0x68 | baseColor | uint | RGBA color at the base (start) |
| 0x6C | edgeColor | uint | RGBA color at the edge (end) |
| 0x70 | edgeParameter | float | Position parameter for ribbon vertices |
| 0x74 | textureSlot | uint | Which texture coordinate set to use |
| 0x78 | visibility | uint | Visibility flag (0 = hidden, 1 = visible) |
| 0x7C | ... | ... | Animation tracks follow |

## Color Format
Both baseColor and edgeColor use 32-bit RGBA format:
- Bits 0-7: Blue
- Bits 8-15: Green
- Bits 16-23: Red
- Bits 24-31: Alpha

## Animation Tracks
After the base properties, several animation tracks may follow:

- Height above track (float)
- Height below track (float)
- Base width track (float)
- Edge width track (float)
- Base color track (Vector3 RGB)
- Base alpha track (float)
- Edge color track (Vector3 RGB)
- Edge alpha track (float)
- Texture slot track (int)
- Visibility track (int, 0 or 1)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Extended with additional animation capabilities |

## Dependencies
- MDLGENOBJECT - All ribbon emitters inherit from the generic object structure
- MDLKEYTRACK - Used for animation tracks within the structure
- BONE - Ribbon emitters can be attached to bones via parentId
- MTLS - References materials via the materialId field

## Implementation Notes
- Ribbons are rendered as a series of connected quads forming a continuous strip
- Each slice is a cross-section of the ribbon, creating a segment between itself and the next slice
- The ribbon forms behind the emitter, similar to a trail effect
- The baseWidth and edgeWidth control the tapering of the ribbon from its attachment point to its end
- The baseColor and edgeColor control the color gradient along the ribbon's length
- Ribbons typically use alpha blending and may be translucent
- The visibility field can be animated to show/hide the ribbon during specific animations
- For optimal performance, ribbons should use additive blending and relatively simple textures
- The textureSlot specifies which set of texture coordinates to use from the material
- Ribbon vertices must be dynamically calculated each frame based on the emitter's movement history
- The edgeParameter controls the position of vertices along the ribbon's cross-section
- The lifetime of a ribbon is controlled by the number of slices, with each slice representing a historical position

## Usage Context
Ribbon emitters in MDX models are used for:
- Weapon trails and slashing effects
- Magic spells and energy beams
- Flowing energies and auras
- Spell channeling visualizations
- Flying object trails
- Movement paths and traces
- Connecting beams between objects
- Lightning and electricity effects

## Implementation Example

```csharp
public class RIBBChunk : IMdxChunk
{
    public string ChunkId => "RIBB";
    
    public List<MdxRibbonEmitter> Emitters { get; private set; } = new List<MdxRibbonEmitter>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear any existing emitters
        Emitters.Clear();
        
        // Read emitters until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var emitter = new MdxRibbonEmitter();
            
            // Read base object properties
            emitter.ParseBaseObject(reader);
            
            // Read ribbon emitter specific properties
            emitter.Slices = reader.ReadUInt32();
            emitter.MaterialId = reader.ReadUInt32();
            emitter.BaseWidth = reader.ReadSingle();
            emitter.EdgeWidth = reader.ReadSingle();
            emitter.BaseColor = reader.ReadUInt32();
            emitter.EdgeColor = reader.ReadUInt32();
            emitter.EdgeParameter = reader.ReadSingle();
            emitter.TextureSlot = reader.ReadUInt32();
            emitter.Visibility = reader.ReadUInt32();
            
            // Read animation tracks
            // Height above
            emitter.HeightAboveTrack = new MdxKeyTrack<float>();
            emitter.HeightAboveTrack.Parse(reader, r => r.ReadSingle());
            
            // Height below
            emitter.HeightBelowTrack = new MdxKeyTrack<float>();
            emitter.HeightBelowTrack.Parse(reader, r => r.ReadSingle());
            
            // Base width
            emitter.BaseWidthTrack = new MdxKeyTrack<float>();
            emitter.BaseWidthTrack.Parse(reader, r => r.ReadSingle());
            
            // Edge width
            emitter.EdgeWidthTrack = new MdxKeyTrack<float>();
            emitter.EdgeWidthTrack.Parse(reader, r => r.ReadSingle());
            
            // Base color
            emitter.BaseColorTrack = new MdxKeyTrack<Vector3>();
            emitter.BaseColorTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Base alpha
            emitter.BaseAlphaTrack = new MdxKeyTrack<float>();
            emitter.BaseAlphaTrack.Parse(reader, r => r.ReadSingle());
            
            // Edge color
            emitter.EdgeColorTrack = new MdxKeyTrack<Vector3>();
            emitter.EdgeColorTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            // Edge alpha
            emitter.EdgeAlphaTrack = new MdxKeyTrack<float>();
            emitter.EdgeAlphaTrack.Parse(reader, r => r.ReadSingle());
            
            // Texture slot
            emitter.TextureSlotTrack = new MdxKeyTrack<int>();
            emitter.TextureSlotTrack.Parse(reader, r => r.ReadInt32());
            
            // Visibility
            emitter.VisibilityTrack = new MdxKeyTrack<int>();
            emitter.VisibilityTrack.Parse(reader, r => r.ReadInt32());
            
            Emitters.Add(emitter);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var emitter in Emitters)
        {
            // Write base object properties
            emitter.WriteBaseObject(writer);
            
            // Write ribbon emitter specific properties
            writer.Write(emitter.Slices);
            writer.Write(emitter.MaterialId);
            writer.Write(emitter.BaseWidth);
            writer.Write(emitter.EdgeWidth);
            writer.Write(emitter.BaseColor);
            writer.Write(emitter.EdgeColor);
            writer.Write(emitter.EdgeParameter);
            writer.Write(emitter.TextureSlot);
            writer.Write(emitter.Visibility);
            
            // Write animation tracks
            emitter.HeightAboveTrack.Write(writer, (w, f) => w.Write(f));
            emitter.HeightBelowTrack.Write(writer, (w, f) => w.Write(f));
            emitter.BaseWidthTrack.Write(writer, (w, f) => w.Write(f));
            emitter.EdgeWidthTrack.Write(writer, (w, f) => w.Write(f));
            emitter.BaseColorTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            emitter.BaseAlphaTrack.Write(writer, (w, f) => w.Write(f));
            emitter.EdgeColorTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            emitter.EdgeAlphaTrack.Write(writer, (w, f) => w.Write(f));
            emitter.TextureSlotTrack.Write(writer, (w, i) => w.Write(i));
            emitter.VisibilityTrack.Write(writer, (w, i) => w.Write(i));
        }
    }
    
    /// <summary>
    /// Gets the current emitter parameters for a ribbon
    /// </summary>
    /// <param name="emitterIndex">Index of the emitter</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The current ribbon parameters, or null if the ribbon is invisible</returns>
    public MdxRibbonParams GetRibbonParams(int emitterIndex, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        if (emitterIndex < 0 || emitterIndex >= Emitters.Count)
        {
            return null;
        }
        
        var emitter = Emitters[emitterIndex];
        var result = new MdxRibbonParams();
        
        // Check visibility
        bool isVisible = emitter.Visibility == 1;
        if (emitter.VisibilityTrack.NumKeys > 0)
        {
            isVisible = emitter.VisibilityTrack.Evaluate(time, sequenceDuration, globalSequences) > 0;
        }
        
        if (!isVisible)
        {
            return null;
        }
        
        // Get current values for all animated properties
        result.HeightAbove = 0.0f;
        if (emitter.HeightAboveTrack.NumKeys > 0)
        {
            result.HeightAbove = emitter.HeightAboveTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.HeightBelow = 0.0f;
        if (emitter.HeightBelowTrack.NumKeys > 0)
        {
            result.HeightBelow = emitter.HeightBelowTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.BaseWidth = emitter.BaseWidth;
        if (emitter.BaseWidthTrack.NumKeys > 0)
        {
            result.BaseWidth = emitter.BaseWidthTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.EdgeWidth = emitter.EdgeWidth;
        if (emitter.EdgeWidthTrack.NumKeys > 0)
        {
            result.EdgeWidth = emitter.EdgeWidthTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        // Extract base color and alpha
        byte baseR = (byte)((emitter.BaseColor >> 16) & 0xFF);
        byte baseG = (byte)((emitter.BaseColor >> 8) & 0xFF);
        byte baseB = (byte)(emitter.BaseColor & 0xFF);
        byte baseA = (byte)((emitter.BaseColor >> 24) & 0xFF);
        
        // Apply base color track if it exists
        if (emitter.BaseColorTrack.NumKeys > 0)
        {
            var baseColor = emitter.BaseColorTrack.Evaluate(time, sequenceDuration, globalSequences);
            baseR = (byte)(baseColor.X * 255);
            baseG = (byte)(baseColor.Y * 255);
            baseB = (byte)(baseColor.Z * 255);
        }
        
        // Apply base alpha track if it exists
        if (emitter.BaseAlphaTrack.NumKeys > 0)
        {
            baseA = (byte)(emitter.BaseAlphaTrack.Evaluate(time, sequenceDuration, globalSequences) * 255);
        }
        
        // Combine into base color
        result.BaseColor = (uint)((baseA << 24) | (baseR << 16) | (baseG << 8) | baseB);
        
        // Extract edge color and alpha
        byte edgeR = (byte)((emitter.EdgeColor >> 16) & 0xFF);
        byte edgeG = (byte)((emitter.EdgeColor >> 8) & 0xFF);
        byte edgeB = (byte)(emitter.EdgeColor & 0xFF);
        byte edgeA = (byte)((emitter.EdgeColor >> 24) & 0xFF);
        
        // Apply edge color track if it exists
        if (emitter.EdgeColorTrack.NumKeys > 0)
        {
            var edgeColor = emitter.EdgeColorTrack.Evaluate(time, sequenceDuration, globalSequences);
            edgeR = (byte)(edgeColor.X * 255);
            edgeG = (byte)(edgeColor.Y * 255);
            edgeB = (byte)(edgeColor.Z * 255);
        }
        
        // Apply edge alpha track if it exists
        if (emitter.EdgeAlphaTrack.NumKeys > 0)
        {
            edgeA = (byte)(emitter.EdgeAlphaTrack.Evaluate(time, sequenceDuration, globalSequences) * 255);
        }
        
        // Combine into edge color
        result.EdgeColor = (uint)((edgeA << 24) | (edgeR << 16) | (edgeG << 8) | edgeB);
        
        // Set texture slot
        result.TextureSlot = emitter.TextureSlot;
        if (emitter.TextureSlotTrack.NumKeys > 0)
        {
            result.TextureSlot = (uint)emitter.TextureSlotTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        // Copy other static properties
        result.Slices = emitter.Slices;
        result.MaterialId = emitter.MaterialId;
        result.EdgeParameter = emitter.EdgeParameter;
        
        return result;
    }
    
    /// <summary>
    /// Generates ribbon vertices based on emitter positions and parameters
    /// </summary>
    /// <param name="emitterHistory">History of emitter positions over time</param>
    /// <param name="parameters">Current ribbon parameters</param>
    /// <param name="cameraRight">Camera right vector for ribbon orientation</param>
    /// <returns>Array of vertices for the ribbon</returns>
    public static Vector3[] GenerateRibbonVertices(Vector3[] emitterHistory, MdxRibbonParams parameters, Vector3 cameraRight)
    {
        if (emitterHistory == null || emitterHistory.Length == 0 || parameters == null)
        {
            return Array.Empty<Vector3>();
        }
        
        // Calculate number of vertices based on slices
        int numVertices = (int)parameters.Slices * 2;
        Vector3[] vertices = new Vector3[numVertices];
        
        // Calculate widths along the ribbon length
        float widthStep = (parameters.EdgeWidth - parameters.BaseWidth) / (parameters.Slices - 1);
        
        // Generate ribbon vertices
        for (int i = 0; i < parameters.Slices; i++)
        {
            // Get position from history (newest to oldest)
            Vector3 position = emitterHistory[i];
            
            // Calculate width at this point along the ribbon
            float width = parameters.BaseWidth + (widthStep * i);
            
            // Create vertices on either side of the ribbon
            vertices[i * 2] = position + cameraRight * width * parameters.EdgeParameter;
            vertices[i * 2 + 1] = position - cameraRight * width * parameters.EdgeParameter;
        }
        
        return vertices;
    }
}

public class MdxRibbonEmitter : MdxGenericObject
{
    public uint Slices { get; set; }
    public uint MaterialId { get; set; }
    public float BaseWidth { get; set; }
    public float EdgeWidth { get; set; }
    public uint BaseColor { get; set; }
    public uint EdgeColor { get; set; }
    public float EdgeParameter { get; set; }
    public uint TextureSlot { get; set; }
    public uint Visibility { get; set; }
    
    public MdxKeyTrack<float> HeightAboveTrack { get; set; }
    public MdxKeyTrack<float> HeightBelowTrack { get; set; }
    public MdxKeyTrack<float> BaseWidthTrack { get; set; }
    public MdxKeyTrack<float> EdgeWidthTrack { get; set; }
    public MdxKeyTrack<Vector3> BaseColorTrack { get; set; }
    public MdxKeyTrack<float> BaseAlphaTrack { get; set; }
    public MdxKeyTrack<Vector3> EdgeColorTrack { get; set; }
    public MdxKeyTrack<float> EdgeAlphaTrack { get; set; }
    public MdxKeyTrack<int> TextureSlotTrack { get; set; }
    public MdxKeyTrack<int> VisibilityTrack { get; set; }
    
    /// <summary>
    /// Unpacks the RGBA components from a color value
    /// </summary>
    public static void UnpackRGBA(uint color, out byte r, out byte g, out byte b, out byte a)
    {
        b = (byte)(color & 0xFF);
        g = (byte)((color >> 8) & 0xFF);
        r = (byte)((color >> 16) & 0xFF);
        a = (byte)((color >> 24) & 0xFF);
    }
    
    /// <summary>
    /// Gets the base color as a Vector4 (RGBA normalized to 0-1)
    /// </summary>
    public Vector4 GetBaseColorVector()
    {
        byte r, g, b, a;
        UnpackRGBA(BaseColor, out r, out g, out b, out a);
        
        return new Vector4(
            r / 255.0f,
            g / 255.0f,
            b / 255.0f,
            a / 255.0f
        );
    }
    
    /// <summary>
    /// Gets the edge color as a Vector4 (RGBA normalized to 0-1)
    /// </summary>
    public Vector4 GetEdgeColorVector()
    {
        byte r, g, b, a;
        UnpackRGBA(EdgeColor, out r, out g, out b, out a);
        
        return new Vector4(
            r / 255.0f,
            g / 255.0f,
            b / 255.0f,
            a / 255.0f
        );
    }
}

public class MdxRibbonParams
{
    public uint Slices { get; set; }
    public uint MaterialId { get; set; }
    public float BaseWidth { get; set; }
    public float EdgeWidth { get; set; }
    public uint BaseColor { get; set; }
    public uint EdgeColor { get; set; }
    public float EdgeParameter { get; set; }
    public uint TextureSlot { get; set; }
    public float HeightAbove { get; set; }
    public float HeightBelow { get; set; }
    
    /// <summary>
    /// Gets the color at a specific point along the ribbon
    /// </summary>
    /// <param name="position">Normalized position along the ribbon (0 = base, 1 = edge)</param>
    /// <returns>Interpolated color value</returns>
    public uint GetColorAt(float position)
    {
        byte baseR, baseG, baseB, baseA;
        byte edgeR, edgeG, edgeB, edgeA;
        
        MdxRibbonEmitter.UnpackRGBA(BaseColor, out baseR, out baseG, out baseB, out baseA);
        MdxRibbonEmitter.UnpackRGBA(EdgeColor, out edgeR, out edgeG, out edgeB, out edgeA);
        
        byte r = (byte)(baseR + (edgeR - baseR) * position);
        byte g = (byte)(baseG + (edgeG - baseG) * position);
        byte b = (byte)(baseB + (edgeB - baseB) * position);
        byte a = (byte)(baseA + (edgeA - baseA) * position);
        
        return (uint)((a << 24) | (r << 16) | (g << 8) | b);
    }
    
    /// <summary>
    /// Gets the width at a specific point along the ribbon
    /// </summary>
    /// <param name="position">Normalized position along the ribbon (0 = base, 1 = edge)</param>
    /// <returns>Interpolated width value</returns>
    public float GetWidthAt(float position)
    {
        return BaseWidth + (EdgeWidth - BaseWidth) * position;
    }
} 