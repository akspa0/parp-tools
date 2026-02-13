using System.Numerics;
using MdxLTool.Formats.Mdx;

namespace MdxViewer.Rendering;

/// <summary>
/// Manages a single particle emitter instance with active particles.
/// Handles particle spawning, physics, and lifecycle.
/// </summary>
public class ParticleEmitter
{
    private readonly MdlParticleEmitter2 _emitterDef;
    private readonly List<Particle> _particles = new();
    private readonly Random _random = new();
    private float _timeSinceLastEmit = 0f;
    private float _emitterAge = 0f;
    
    public Matrix4x4 Transform { get; set; } = Matrix4x4.Identity;
    public bool IsActive { get; set; } = true;
    public IReadOnlyList<Particle> Particles => _particles;
    public MdlParticleEmitter2 Definition => _emitterDef;
    
    public ParticleEmitter(MdlParticleEmitter2 emitterDef)
    {
        _emitterDef = emitterDef;
    }
    
    /// <summary>
    /// Update particle physics and spawn new particles.
    /// </summary>
    public void Update(float deltaTime)
    {
        if (!IsActive) return;
        
        _emitterAge += deltaTime;
        _timeSinceLastEmit += deltaTime;
        
        // Update existing particles
        for (int i = _particles.Count - 1; i >= 0; i--)
        {
            var particle = _particles[i];
            particle.Age += deltaTime;
            
            // Remove dead particles
            if (particle.Age >= particle.Lifespan)
            {
                _particles.RemoveAt(i);
                continue;
            }
            
            // Apply gravity
            var velocity = particle.Velocity;
            velocity.Z -= _emitterDef.Gravity * deltaTime;
            particle.Velocity = velocity;
            
            // Update position
            particle.Position += particle.Velocity * deltaTime;
            
            // Update lifecycle interpolation (0=birth, 0.5=mid, 1=death)
            particle.LifePhase = particle.Age / particle.Lifespan;
        }
        
        // Spawn new particles based on emission rate
        if (_emitterDef.EmissionRate > 0)
        {
            float emitInterval = 1.0f / _emitterDef.EmissionRate;
            while (_timeSinceLastEmit >= emitInterval)
            {
                SpawnParticle();
                _timeSinceLastEmit -= emitInterval;
            }
        }
    }
    
    private void SpawnParticle()
    {
        // Get emitter world position
        Vector3 emitterPos = Vector3.Transform(Vector3.Zero, Transform);
        
        // Random direction within latitude cone
        float theta = (float)(_random.NextDouble() * Math.PI * 2);
        float phi = (float)(_random.NextDouble() * _emitterDef.Latitude);
        
        // Speed with variation
        float speed = _emitterDef.Speed + (float)(_random.NextDouble() - 0.5) * _emitterDef.Variation;
        
        // Calculate velocity vector
        Vector3 velocity = new Vector3(
            MathF.Sin(phi) * MathF.Cos(theta),
            MathF.Sin(phi) * MathF.Sin(theta),
            MathF.Cos(phi)
        ) * speed;
        
        // Transform velocity by emitter orientation
        velocity = Vector3.TransformNormal(velocity, Transform);
        
        var particle = new Particle
        {
            Position = emitterPos,
            Velocity = velocity,
            Age = 0f,
            Lifespan = _emitterDef.Lifespan,
            LifePhase = 0f,
            Size = 1.0f,
            TextureIndex = 0
        };
        
        _particles.Add(particle);
    }
    
    /// <summary>
    /// Get interpolated color for a particle based on its lifecycle phase.
    /// </summary>
    public Vector4 GetParticleColor(Particle particle)
    {
        float t = particle.LifePhase;
        Vector3 color;
        float alpha;
        
        if (t < 0.5f)
        {
            // Birth to mid
            float localT = t * 2f;
            color = Vector3.Lerp(
                new Vector3(_emitterDef.SegmentColor[0].X, _emitterDef.SegmentColor[0].Y, _emitterDef.SegmentColor[0].Z),
                new Vector3(_emitterDef.SegmentColor[1].X, _emitterDef.SegmentColor[1].Y, _emitterDef.SegmentColor[1].Z),
                localT
            );
            alpha = MathHelper.Lerp(_emitterDef.SegmentAlpha[0] / 255f, _emitterDef.SegmentAlpha[1] / 255f, localT);
        }
        else
        {
            // Mid to death
            float localT = (t - 0.5f) * 2f;
            color = Vector3.Lerp(
                new Vector3(_emitterDef.SegmentColor[1].X, _emitterDef.SegmentColor[1].Y, _emitterDef.SegmentColor[1].Z),
                new Vector3(_emitterDef.SegmentColor[2].X, _emitterDef.SegmentColor[2].Y, _emitterDef.SegmentColor[2].Z),
                localT
            );
            alpha = MathHelper.Lerp(_emitterDef.SegmentAlpha[1] / 255f, _emitterDef.SegmentAlpha[2] / 255f, localT);
        }
        
        return new Vector4(color, alpha);
    }
    
    /// <summary>
    /// Get interpolated size for a particle based on its lifecycle phase.
    /// </summary>
    public float GetParticleSize(Particle particle)
    {
        float t = particle.LifePhase;
        
        if (t < 0.5f)
        {
            float localT = t * 2f;
            return MathHelper.Lerp(_emitterDef.SegmentScaling[0], _emitterDef.SegmentScaling[1], localT);
        }
        else
        {
            float localT = (t - 0.5f) * 2f;
            return MathHelper.Lerp(_emitterDef.SegmentScaling[1], _emitterDef.SegmentScaling[2], localT);
        }
    }
}

/// <summary>
/// Individual particle instance.
/// </summary>
public class Particle
{
    public Vector3 Position { get; set; }
    public Vector3 Velocity { get; set; }
    public float Age { get; set; }
    public float Lifespan { get; set; }
    public float LifePhase { get; set; } // 0.0 to 1.0
    public float Size { get; set; }
    public int TextureIndex { get; set; }
}

/// <summary>
/// Helper math functions.
/// </summary>
internal static class MathHelper
{
    public static float Lerp(float a, float b, float t)
    {
        return a + (b - a) * t;
    }
}
