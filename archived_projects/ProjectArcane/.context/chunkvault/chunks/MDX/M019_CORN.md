# CORN - MDX Corn/Tentacle Effects Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The CORN (Corn Effects, also known as Tentacle Effects) chunk defines flexible, segmented appendages that can simulate tentacles, vines, cables, or other similar structures. These effects create organic-looking tentacles or whip-like appendages that bend and sway based on physics simulations. They are often used for creatures like hydras, squid, or plant-like entities, creating dynamic appendages that respond to movement.

## Structure

```csharp
public struct CORN
{
    /// <summary>
    /// Array of corn/tentacle effect definitions
    /// </summary>
    // MDLCORNEFFECT effects[numEffects] follows
}

public struct MDLCORNEFFECT
{
    /// <summary>
    /// ID of the corn effect
    /// </summary>
    public uint id;
    
    /// <summary>
    /// Number of segments in the tentacle
    /// </summary>
    public uint numSegments;
    
    /// <summary>
    /// ID of the material used for rendering
    /// </summary>
    public uint materialId;
    
    /// <summary>
    /// Position of the corn effect base
    /// </summary>
    public Vector3 position;
    
    /// <summary>
    /// Width at base of the effect
    /// </summary>
    public float baseWidth;
    
    /// <summary>
    /// Width at tip of the effect
    /// </summary>
    public float tipWidth;
    
    /// <summary>
    /// Stiffness of the tentacle (resistance to bending)
    /// </summary>
    public float stiffness;
    
    /// <summary>
    /// Damping factor (reduces oscillation)
    /// </summary>
    public float damping;
    
    /// <summary>
    /// Displacement of tentacle segments
    /// </summary>
    public float displacement;
    
    /// <summary>
    /// Number of animation control points
    /// </summary>
    public uint numControlPoints;
    
    /// <summary>
    /// Fixed/static points for animation (0 = dynamic, 1 = fixed)
    /// </summary>
    // uint fixedPoints[numControlPoints] follows
    
    /// <summary>
    /// Animation data for the corn effect
    /// </summary>
    // MDLKEYTRACK animations follow
}
```

## Properties

### MDLCORNEFFECT Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | id | uint | Unique identifier for this corn effect |
| 0x04 | numSegments | uint | Number of segments that make up the tentacle |
| 0x08 | materialId | uint | Material ID from MTLS chunk |
| 0x0C | position | Vector3 | Base position of the tentacle |
| 0x18 | baseWidth | float | Width at base of the tentacle |
| 0x1C | tipWidth | float | Width at tip of the tentacle (tapers if < baseWidth) |
| 0x20 | stiffness | float | Resistance to bending (higher = stiffer) |
| 0x24 | damping | float | How quickly oscillations die down (higher = less movement) |
| 0x28 | displacement | float | Initial displacement of segments from rest position |
| 0x2C | numControlPoints | uint | Number of animation control points |
| 0x30 | fixedPoints | uint[] | Array of flags (0 = dynamic, 1 = fixed) for each control point |
| varies | ... | ... | Animation tracks follow |

## Animation Tracks
After the base properties and fixed points array, several animation tracks may follow:

- Position track (Vector3) - Base position of the tentacle
- Direction track (Vector3) - Direction of the tentacle
- Target position track (Vector3) - Target position for the tentacle tip
- Rotation track (Quaternion) - Base rotation of the tentacle

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Extended physics parameters |

## Dependencies
- MDLKEYTRACK - Used for animation tracks within the structure
- MTLS - References materials via the materialId field
- BONE - May be attached to bones via animation

## Implementation Notes
- Corn effects create tentacle-like appendages that respond to physics
- The tentacle is divided into segments with decreasing width from base to tip
- The stiffness parameter controls how easily the tentacle bends
- The damping parameter controls how quickly the tentacle stops oscillating
- Fixed points allow certain control points to be pinned in place
- Material ID determines the texture and rendering properties
- Each segment is typically rendered as a tapered cylinder or cone
- Higher segment counts create smoother, more flexible tentacles
- Physics simulation should include gravity, inertia, and collision
- The base position is typically animated to follow character movement
- Target position can be used to make the tentacle reach for objects
- The displacement parameter adds initial randomness to segment positions
- For optimal performance, segment count should be kept moderate (10-20)
- The physics simulation should run at a fixed time step for stability

## Usage Context
Corn/Tentacle effects in MDX models are used for:
- Creature tentacles (squid, hydra, etc.)
- Plant vines and tendrils
- Hair and fur simulation
- Cables and chains
- Tails and appendages
- Whips and similar weapons
- Flexible antennas or feelers
- Magic effects (energy tendrils)
- Cloth-like elements (banners, cloths)
- Organic decorative elements

## Implementation Example

```csharp
public class CORNChunk : IMdxChunk
{
    public string ChunkId => "CORN";
    
    public List<MdxCornEffect> Effects { get; private set; } = new List<MdxCornEffect>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear existing effects
        Effects.Clear();
        
        // Read effects until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var effect = new MdxCornEffect();
            
            // Read basic properties
            effect.Id = reader.ReadUInt32();
            effect.NumSegments = reader.ReadUInt32();
            effect.MaterialId = reader.ReadUInt32();
            
            // Read position
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            effect.Position = new Vector3(x, y, z);
            
            // Read width and physics parameters
            effect.BaseWidth = reader.ReadSingle();
            effect.TipWidth = reader.ReadSingle();
            effect.Stiffness = reader.ReadSingle();
            effect.Damping = reader.ReadSingle();
            effect.Displacement = reader.ReadSingle();
            
            // Read control points
            effect.NumControlPoints = reader.ReadUInt32();
            effect.FixedPoints = new uint[effect.NumControlPoints];
            
            for (int i = 0; i < effect.NumControlPoints; i++)
            {
                effect.FixedPoints[i] = reader.ReadUInt32();
            }
            
            // Read animation tracks
            effect.PositionTrack = new MdxKeyTrack<Vector3>();
            effect.PositionTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            effect.DirectionTrack = new MdxKeyTrack<Vector3>();
            effect.DirectionTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            effect.TargetPositionTrack = new MdxKeyTrack<Vector3>();
            effect.TargetPositionTrack.Parse(reader, r => new Vector3(r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            effect.RotationTrack = new MdxKeyTrack<Quaternion>();
            effect.RotationTrack.Parse(reader, r => new Quaternion(r.ReadSingle(), r.ReadSingle(), r.ReadSingle(), r.ReadSingle()));
            
            Effects.Add(effect);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var effect in Effects)
        {
            // Write basic properties
            writer.Write(effect.Id);
            writer.Write(effect.NumSegments);
            writer.Write(effect.MaterialId);
            
            // Write position
            writer.Write(effect.Position.X);
            writer.Write(effect.Position.Y);
            writer.Write(effect.Position.Z);
            
            // Write width and physics parameters
            writer.Write(effect.BaseWidth);
            writer.Write(effect.TipWidth);
            writer.Write(effect.Stiffness);
            writer.Write(effect.Damping);
            writer.Write(effect.Displacement);
            
            // Write control points
            writer.Write(effect.NumControlPoints);
            
            for (int i = 0; i < effect.NumControlPoints; i++)
            {
                writer.Write(effect.FixedPoints[i]);
            }
            
            // Write animation tracks
            effect.PositionTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            effect.DirectionTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            effect.TargetPositionTrack.Write(writer, (w, v) => { w.Write(v.X); w.Write(v.Y); w.Write(v.Z); });
            effect.RotationTrack.Write(writer, (w, q) => { w.Write(q.X); w.Write(q.Y); w.Write(q.Z); w.Write(q.W); });
        }
    }
    
    /// <summary>
    /// Simulates the corn effects for the given time and external forces
    /// </summary>
    /// <param name="deltaTime">Time step for simulation in seconds</param>
    /// <param name="gravity">Gravity vector</param>
    /// <param name="windForce">Wind force vector</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    public void SimulateEffects(float deltaTime, Vector3 gravity, Vector3 windForce, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        foreach (var effect in Effects)
        {
            // Get current animated values
            Vector3 position = effect.Position;
            if (effect.PositionTrack.NumKeys > 0)
            {
                position = effect.PositionTrack.Evaluate(time, sequenceDuration, globalSequences);
            }
            
            Vector3 direction = Vector3.UnitY; // Default up direction
            if (effect.DirectionTrack.NumKeys > 0)
            {
                direction = Vector3.Normalize(effect.DirectionTrack.Evaluate(time, sequenceDuration, globalSequences));
            }
            
            Vector3 targetPosition = position + direction * effect.NumSegments * 1.0f; // Default target based on direction
            if (effect.TargetPositionTrack.NumKeys > 0)
            {
                targetPosition = effect.TargetPositionTrack.Evaluate(time, sequenceDuration, globalSequences);
            }
            
            Quaternion rotation = Quaternion.Identity;
            if (effect.RotationTrack.NumKeys > 0)
            {
                rotation = effect.RotationTrack.Evaluate(time, sequenceDuration, globalSequences);
            }
            
            // Update the segment positions and velocities based on physics
            if (effect.Segments.Count == 0)
            {
                // Initialize segments if not already created
                InitializeSegments(effect, position, direction);
            }
            else
            {
                // Update the base segment with the current animated position
                effect.Segments[0].Position = position;
                
                // Apply fixed constraints from control points
                for (int i = 0; i < effect.NumControlPoints && i < effect.Segments.Count; i++)
                {
                    if (effect.FixedPoints[i] == 1) // Fixed point
                    {
                        int segmentIndex = (int)(i * (effect.Segments.Count - 1) / (effect.NumControlPoints - 1));
                        if (i == effect.NumControlPoints - 1) // Last control point targets the targetPosition
                        {
                            effect.Segments[segmentIndex].Position = targetPosition;
                        }
                        else
                        {
                            // Other fixed points stay in their current position
                            effect.Segments[segmentIndex].Velocity = Vector3.Zero;
                        }
                    }
                }
                
                // Apply physics to each segment
                for (int i = 1; i < effect.Segments.Count; i++)
                {
                    // Skip segments that are fixed control points
                    bool isFixed = false;
                    for (int j = 0; j < effect.NumControlPoints; j++)
                    {
                        int segmentIndex = (int)(j * (effect.Segments.Count - 1) / (effect.NumControlPoints - 1));
                        if (segmentIndex == i && effect.FixedPoints[j] == 1)
                        {
                            isFixed = true;
                            break;
                        }
                    }
                    
                    if (isFixed)
                    {
                        continue;
                    }
                    
                    // Get previous segment
                    var prevSegment = effect.Segments[i - 1];
                    var segment = effect.Segments[i];
                    
                    // Calculate the force from the constraints (connection to previous segment)
                    float segmentLength = 1.0f; // Length between segments
                    Vector3 toNext = segment.Position - prevSegment.Position;
                    float distance = toNext.Length();
                    Vector3 forceDirection = Vector3.Zero;
                    
                    if (distance > 0.0001f)
                    {
                        forceDirection = toNext / distance;
                    }
                    
                    // Spring force (keeps segments at the right distance)
                    float springFactor = effect.Stiffness * 10.0f;
                    Vector3 springForce = forceDirection * (distance - segmentLength) * springFactor;
                    
                    // Apply damping to reduce oscillation
                    Vector3 dampingForce = -segment.Velocity * effect.Damping;
                    
                    // Apply gravity
                    Vector3 gravityForce = gravity;
                    
                    // Apply wind
                    Vector3 windEffect = windForce * (1.0f - i / (float)effect.Segments.Count); // Less effect at the base
                    
                    // Combine forces
                    Vector3 totalForce = springForce + dampingForce + gravityForce + windEffect;
                    
                    // Update velocity (F = ma, assume mass = 1)
                    segment.Velocity += totalForce * deltaTime;
                    
                    // Update position
                    segment.Position += segment.Velocity * deltaTime;
                }
                
                // Apply constraints to maintain correct segment separation
                for (int iteration = 0; iteration < 3; iteration++) // Multiple iterations for stability
                {
                    for (int i = 1; i < effect.Segments.Count; i++)
                    {
                        // Skip segments that are fixed control points
                        bool isFixed = false;
                        for (int j = 0; j < effect.NumControlPoints; j++)
                        {
                            int segmentIndex = (int)(j * (effect.Segments.Count - 1) / (effect.NumControlPoints - 1));
                            if (segmentIndex == i && effect.FixedPoints[j] == 1)
                            {
                                isFixed = true;
                                break;
                            }
                        }
                        
                        if (isFixed)
                        {
                            continue;
                        }
                        
                        // Enforce distance constraint with previous segment
                        var prevSegment = effect.Segments[i - 1];
                        var segment = effect.Segments[i];
                        float segmentLength = 1.0f; // Length between segments
                        
                        Vector3 toNext = segment.Position - prevSegment.Position;
                        float distance = toNext.Length();
                        
                        if (distance > 0.0001f)
                        {
                            Vector3 correction = toNext * (1.0f - segmentLength / distance);
                            segment.Position -= correction;
                        }
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// Initializes the segments for a corn effect
    /// </summary>
    /// <param name="effect">The effect to initialize</param>
    /// <param name="position">The starting position</param>
    /// <param name="direction">The direction of the tentacle</param>
    private void InitializeSegments(MdxCornEffect effect, Vector3 position, Vector3 direction)
    {
        effect.Segments.Clear();
        
        // Create segments along the initial direction
        for (int i = 0; i < effect.NumSegments; i++)
        {
            var segment = new TentacleSegment();
            segment.Position = position + direction * i * 1.0f; // Space segments evenly
            
            // Add some initial displacement perpendicular to direction
            if (i > 0 && effect.Displacement > 0)
            {
                // Find a perpendicular vector
                Vector3 perp;
                if (Math.Abs(direction.Y) > 0.9f)
                {
                    perp = Vector3.Cross(direction, Vector3.UnitX);
                }
                else
                {
                    perp = Vector3.Cross(direction, Vector3.UnitY);
                }
                
                perp = Vector3.Normalize(perp);
                
                // Apply random displacement perpendicular to direction
                Random rand = new Random((int)(effect.Id + i));
                float displacement = (float)rand.NextDouble() * effect.Displacement;
                float angle = (float)rand.NextDouble() * MathF.PI * 2.0f;
                
                Vector3 offset = perp * displacement * MathF.Sin(angle);
                Vector3 offset2 = Vector3.Cross(direction, perp) * displacement * MathF.Cos(angle);
                
                segment.Position += offset + offset2;
            }
            
            segment.Velocity = Vector3.Zero;
            effect.Segments.Add(segment);
        }
    }
    
    /// <summary>
    /// Gets the current vertex positions for rendering a corn effect
    /// </summary>
    /// <param name="effect">The corn effect to render</param>
    /// <returns>Tuple of positions and widths for each segment</returns>
    public (Vector3[] Positions, float[] Widths) GetRenderData(MdxCornEffect effect)
    {
        if (effect.Segments.Count == 0)
        {
            return (Array.Empty<Vector3>(), Array.Empty<float>());
        }
        
        Vector3[] positions = new Vector3[effect.Segments.Count];
        float[] widths = new float[effect.Segments.Count];
        
        for (int i = 0; i < effect.Segments.Count; i++)
        {
            positions[i] = effect.Segments[i].Position;
            
            // Calculate width based on position along the tentacle (taper from base to tip)
            float t = i / (float)(effect.Segments.Count - 1);
            widths[i] = MathHelper.Lerp(effect.BaseWidth, effect.TipWidth, t);
        }
        
        return (positions, widths);
    }
}

public class MdxCornEffect
{
    public uint Id { get; set; }
    public uint NumSegments { get; set; }
    public uint MaterialId { get; set; }
    public Vector3 Position { get; set; }
    public float BaseWidth { get; set; }
    public float TipWidth { get; set; }
    public float Stiffness { get; set; }
    public float Damping { get; set; }
    public float Displacement { get; set; }
    public uint NumControlPoints { get; set; }
    public uint[] FixedPoints { get; set; }
    
    public MdxKeyTrack<Vector3> PositionTrack { get; set; }
    public MdxKeyTrack<Vector3> DirectionTrack { get; set; }
    public MdxKeyTrack<Vector3> TargetPositionTrack { get; set; }
    public MdxKeyTrack<Quaternion> RotationTrack { get; set; }
    
    // Runtime data for simulation
    public List<TentacleSegment> Segments { get; private set; } = new List<TentacleSegment>();
    
    /// <summary>
    /// Checks if this is a tapered tentacle
    /// </summary>
    public bool IsTapered => TipWidth < BaseWidth;
    
    /// <summary>
    /// Gets the average width of the tentacle
    /// </summary>
    public float AverageWidth => (BaseWidth + TipWidth) * 0.5f;
    
    /// <summary>
    /// Determines if the tentacle has any fixed control points
    /// </summary>
    public bool HasFixedPoints
    {
        get
        {
            if (FixedPoints == null) return false;
            
            for (int i = 0; i < FixedPoints.Length; i++)
            {
                if (FixedPoints[i] == 1) return true;
            }
            
            return false;
        }
    }
    
    /// <summary>
    /// Determines if the tentacle tip position is animated
    /// </summary>
    public bool HasAnimatedTip => TargetPositionTrack.NumKeys > 0;
}

public class TentacleSegment
{
    public Vector3 Position { get; set; }
    public Vector3 Velocity { get; set; }
}

public static class MathHelper
{
    public static float Lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }
}
``` 