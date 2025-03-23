# PREM - MDX Particle Emitters Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The PREM (Particle Emitters) chunk defines basic particle emitters for visual effects such as smoke, fire, and magic. Each emitter creates particles with specific behaviors, appearances, and lifespans, providing dynamic visual elements to the model. These particle systems are rendered as billboarded sprites that always face the camera.

## Structure

```csharp
public struct PREM
{
    /// <summary>
    /// Array of particle emitter definitions
    /// </summary>
    // MDLPARTICLEEMITTER emitters[numEmitters] follows
}

public struct MDLPARTICLEEMITTER : MDLGENOBJECT
{
    /// <summary>
    /// Emission rate (particles per second)
    /// </summary>
    public float emissionRate;
    
    /// <summary>
    /// Gravity factor applied to particles
    /// </summary>
    public float gravity;
    
    /// <summary>
    /// Longitude of emission direction
    /// </summary>
    public float longitude;
    
    /// <summary>
    /// Latitude of emission direction
    /// </summary>
    public float latitude;
    
    /// <summary>
    /// Name of sprite used for particles (null-terminated string, max length 260)
    /// </summary>
    public fixed byte spriteName[260];
    
    /// <summary>
    /// Lifespan of particles in seconds
    /// </summary>
    public float lifespan;
    
    /// <summary>
    /// Initial speed of particles
    /// </summary>
    public float initialVelocity;
    
    /// <summary>
    /// Animation data for the emitter properties
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLPARTICLEEMITTER Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00..0x58 | MDLGENOBJECT | struct | Base generic object (see MDLGENOBJECT structure) |
| 0x58 | emissionRate | float | Number of particles emitted per second |
| 0x5C | gravity | float | Gravity factor affecting particle movement |
| 0x60 | longitude | float | Horizontal emission angle (radians) |
| 0x64 | latitude | float | Vertical emission angle (radians) |
| 0x68 | spriteName | char[260] | Null-terminated path to the sprite texture |
| 0x16C | lifespan | float | How long particles live (seconds) |
| 0x170 | initialVelocity | float | Starting velocity of particles |
| 0x174 | ... | ... | Animation tracks follow |

## Animation Tracks
After the base properties, several animation tracks may follow:

- Emission rate track (float, particles per second)
- Gravity track (float, factor)
- Longitude track (float, radians)
- Latitude track (float, radians)
- Speed track (float, units per second)
- Lifespan track (float, seconds)
- Visibility track (int, 0 or 1)

## Emission Geometry
Particles are emitted in a direction based on the longitude and latitude:
- Longitude: Rotation around the Y axis (0 = positive Z direction)
- Latitude: Angle from the XZ plane (0 = horizontal, π/2 = straight up)

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Similar structure but with preference for PRE2 emitters for more complex effects |

## Dependencies
- MDLGENOBJECT - All particle emitters inherit from the generic object structure
- MDLKEYTRACK - Used for animation tracks within the structure
- BONE - Particle emitters can be attached to bones via parentId
- TEXS - May reference textures used for particle sprites

## Implementation Notes
- Particles are emitted continuously based on the emission rate
- Each particle has position, velocity, age, and texture coordinates
- Particles are affected by gravity over their lifespan
- The spriteName field references the texture used for rendering particles
- Particles are typically rendered as camera-facing billboards
- For optimal performance, particles should use additive blending and small textures
- The visibility track can be used to enable/disable the emitter during specific animations
- In MDX models, this is the basic particle emitter type - for more complex effects, PRE2 emitters are used

## Usage Context
Particle emitters in MDX models are used for:
- Smoke and fire effects
- Magic and spell effects
- Environmental elements like mist and dust
- Character auras and ambient effects
- Weapon trails and impact effects
- Atmospheric effects like rain and snow

## Implementation Example

```csharp
public class PREMChunk : IMdxChunk
{
    public string ChunkId => "PREM";
    
    public List<MdxParticleEmitter> Emitters { get; private set; } = new List<MdxParticleEmitter>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear any existing emitters
        Emitters.Clear();
        
        // Read emitters until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var emitter = new MdxParticleEmitter();
            
            // Read base object properties
            emitter.ParseBaseObject(reader);
            
            // Read emitter specific properties
            emitter.EmissionRate = reader.ReadSingle();
            emitter.Gravity = reader.ReadSingle();
            emitter.Longitude = reader.ReadSingle();
            emitter.Latitude = reader.ReadSingle();
            
            // Read sprite name (null-terminated string, 260 bytes)
            byte[] spriteNameBytes = reader.ReadBytes(260);
            int spriteNameLength = 0;
            while (spriteNameLength < spriteNameBytes.Length && spriteNameBytes[spriteNameLength] != 0)
            {
                spriteNameLength++;
            }
            emitter.SpriteName = System.Text.Encoding.ASCII.GetString(spriteNameBytes, 0, spriteNameLength);
            
            emitter.Lifespan = reader.ReadSingle();
            emitter.InitialVelocity = reader.ReadSingle();
            
            // Read animation tracks
            // Emission rate
            emitter.EmissionRateTrack = new MdxKeyTrack<float>();
            emitter.EmissionRateTrack.Parse(reader, r => r.ReadSingle());
            
            // Gravity
            emitter.GravityTrack = new MdxKeyTrack<float>();
            emitter.GravityTrack.Parse(reader, r => r.ReadSingle());
            
            // Longitude
            emitter.LongitudeTrack = new MdxKeyTrack<float>();
            emitter.LongitudeTrack.Parse(reader, r => r.ReadSingle());
            
            // Latitude
            emitter.LatitudeTrack = new MdxKeyTrack<float>();
            emitter.LatitudeTrack.Parse(reader, r => r.ReadSingle());
            
            // Initial velocity
            emitter.InitialVelocityTrack = new MdxKeyTrack<float>();
            emitter.InitialVelocityTrack.Parse(reader, r => r.ReadSingle());
            
            // Lifespan
            emitter.LifespanTrack = new MdxKeyTrack<float>();
            emitter.LifespanTrack.Parse(reader, r => r.ReadSingle());
            
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
            
            // Write emitter specific properties
            writer.Write(emitter.EmissionRate);
            writer.Write(emitter.Gravity);
            writer.Write(emitter.Longitude);
            writer.Write(emitter.Latitude);
            
            // Write sprite name (pad with nulls to 260 bytes)
            byte[] spriteNameBytes = new byte[260];
            if (!string.IsNullOrEmpty(emitter.SpriteName))
            {
                byte[] temp = System.Text.Encoding.ASCII.GetBytes(emitter.SpriteName);
                int copyLength = Math.Min(temp.Length, 259); // Leave at least one byte for null terminator
                Array.Copy(temp, spriteNameBytes, copyLength);
            }
            writer.Write(spriteNameBytes);
            
            writer.Write(emitter.Lifespan);
            writer.Write(emitter.InitialVelocity);
            
            // Write animation tracks
            emitter.EmissionRateTrack.Write(writer, (w, f) => w.Write(f));
            emitter.GravityTrack.Write(writer, (w, f) => w.Write(f));
            emitter.LongitudeTrack.Write(writer, (w, f) => w.Write(f));
            emitter.LatitudeTrack.Write(writer, (w, f) => w.Write(f));
            emitter.InitialVelocityTrack.Write(writer, (w, f) => w.Write(f));
            emitter.LifespanTrack.Write(writer, (w, f) => w.Write(f));
            emitter.VisibilityTrack.Write(writer, (w, i) => w.Write(i));
        }
    }
    
    /// <summary>
    /// Gets the current emitter parameters for a particle system
    /// </summary>
    /// <param name="emitterIndex">Index of the emitter</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>The current emitter parameters, or null if the emitter is inactive</returns>
    public MdxParticleEmitterParams GetEmitterParams(int emitterIndex, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        if (emitterIndex < 0 || emitterIndex >= Emitters.Count)
        {
            return null;
        }
        
        var emitter = Emitters[emitterIndex];
        var result = new MdxParticleEmitterParams();
        
        // Check visibility
        bool isVisible = true;
        if (emitter.VisibilityTrack.NumKeys > 0)
        {
            isVisible = emitter.VisibilityTrack.Evaluate(time, sequenceDuration, globalSequences) > 0;
        }
        
        if (!isVisible)
        {
            return null;
        }
        
        // Get current values for all animated properties
        result.EmissionRate = emitter.EmissionRate;
        if (emitter.EmissionRateTrack.NumKeys > 0)
        {
            result.EmissionRate = emitter.EmissionRateTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.Gravity = emitter.Gravity;
        if (emitter.GravityTrack.NumKeys > 0)
        {
            result.Gravity = emitter.GravityTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.Longitude = emitter.Longitude;
        if (emitter.LongitudeTrack.NumKeys > 0)
        {
            result.Longitude = emitter.LongitudeTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.Latitude = emitter.Latitude;
        if (emitter.LatitudeTrack.NumKeys > 0)
        {
            result.Latitude = emitter.LatitudeTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.InitialVelocity = emitter.InitialVelocity;
        if (emitter.InitialVelocityTrack.NumKeys > 0)
        {
            result.InitialVelocity = emitter.InitialVelocityTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.Lifespan = emitter.Lifespan;
        if (emitter.LifespanTrack.NumKeys > 0)
        {
            result.Lifespan = emitter.LifespanTrack.Evaluate(time, sequenceDuration, globalSequences);
        }
        
        result.SpriteName = emitter.SpriteName;
        
        return result;
    }
}

public class MdxParticleEmitter : MdxGenericObject
{
    public float EmissionRate { get; set; }
    public float Gravity { get; set; }
    public float Longitude { get; set; }
    public float Latitude { get; set; }
    public string SpriteName { get; set; }
    public float Lifespan { get; set; }
    public float InitialVelocity { get; set; }
    
    public MdxKeyTrack<float> EmissionRateTrack { get; set; }
    public MdxKeyTrack<float> GravityTrack { get; set; }
    public MdxKeyTrack<float> LongitudeTrack { get; set; }
    public MdxKeyTrack<float> LatitudeTrack { get; set; }
    public MdxKeyTrack<float> InitialVelocityTrack { get; set; }
    public MdxKeyTrack<float> LifespanTrack { get; set; }
    public MdxKeyTrack<int> VisibilityTrack { get; set; }
    
    /// <summary>
    /// Calculates the emission direction based on longitude and latitude
    /// </summary>
    /// <returns>A normalized direction vector</returns>
    public Vector3 CalculateEmissionDirection()
    {
        // Calculate direction based on longitude and latitude
        float y = (float)Math.Sin(Latitude);
        float r = (float)Math.Cos(Latitude);
        float x = r * (float)Math.Sin(Longitude);
        float z = r * (float)Math.Cos(Longitude);
        
        return new Vector3(x, y, z);
    }
}

public class MdxParticleEmitterParams
{
    public float EmissionRate { get; set; }
    public float Gravity { get; set; }
    public float Longitude { get; set; }
    public float Latitude { get; set; }
    public string SpriteName { get; set; }
    public float Lifespan { get; set; }
    public float InitialVelocity { get; set; }
    
    /// <summary>
    /// Calculates the emission direction vector based on longitude and latitude
    /// </summary>
    public Vector3 EmissionDirection
    {
        get
        {
            // Calculate direction based on longitude and latitude
            float y = (float)Math.Sin(Latitude);
            float r = (float)Math.Cos(Latitude);
            float x = r * (float)Math.Sin(Longitude);
            float z = r * (float)Math.Cos(Longitude);
            
            return new Vector3(x, y, z);
        }
    }
    
    /// <summary>
    /// Calculates the cone spread for particle emission
    /// </summary>
    /// <returns>The half-angle of the emission cone in radians</returns>
    public float EmissionConeAngle
    {
        get
        {
            // For basic particle emitters, we typically use a small cone angle
            return (float)(Math.PI / 18.0); // 10 degrees
        }
    }
} 