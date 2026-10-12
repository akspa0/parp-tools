using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

/// <summary>
/// Individual live particle instance simulated in model space.
/// </summary>
public sealed class M2Particle
{
    public Vector3 Position;
    public Vector3 Velocity;
    public float Age;
    public float Lifespan;
    public float LifePhase;
}

/// <summary>
/// Simulates an M2 particle emitter according to authored tracks and parameters.
/// </summary>
public sealed class M2ParticleEmitter
{
    private readonly M2ParticleDefinition _def;
    private readonly List<M2Particle> _particles = new();
    private readonly Random _random = new();
    private float _timeSinceLastEmit;

    public M2ParticleDefinition Definition => _def;
    public IReadOnlyList<M2Particle> Particles => _particles;
    public uint TextureId { get; set; }
    public ushort BlendingType => _def.BlendingType;
    public bool IsAdditive { get; }
    public int Rows => Math.Max((int)_def.TextureRows, 1);
    public int Columns => Math.Max((int)_def.TextureColumns, 1);
    public bool IsActive { get; set; } = true;
    public float MidPoint { get; private set; } = 0.5f;
    public Vector4[] Colors { get; } = new Vector4[3];
    public float[] Scales { get; } = new float[3];

    public M2ParticleEmitter(M2ParticleDefinition def, byte[] rawBytes, M2ModelDocument model)
    {
        ArgumentNullException.ThrowIfNull(def);
        ArgumentNullException.ThrowIfNull(rawBytes);
        ArgumentNullException.ThrowIfNull(model);

        _def = def;
        IsAdditive = def.BlendingType is 1 or 4 or 5 or 6;
        ParseColorsAndScales(rawBytes, model);
    }

    private void ParseColorsAndScales(byte[] rawBytes, M2ModelDocument model)
    {
        MidPoint = 0.5f;
        Colors[0] = new Vector4(1f, 1f, 1f, 0.0f);
        Colors[1] = new Vector4(1f, 1f, 1f, 1.0f);
        Colors[2] = new Vector4(1f, 1f, 1f, 0.0f);
        Scales[0] = 0.5f;
        Scales[1] = 1.0f;
        Scales[2] = 0.5f;

        if (rawBytes.Length < 0x130)
            return;

        try
        {
            uint pOffset = BinaryPrimitives.ReadUInt32LittleEndian(rawBytes.AsSpan(0x12C, 4));
            int stride = ((model.Flags & 0x200u) != 0 || model.Version > 271u) ? 0x1EC : 0x1DC;
            int entryOffset = (int)pOffset + (_def.Index * stride);

            if (entryOffset + 0x120 <= rawBytes.Length)
            {
                float mid = BitConverter.ToSingle(rawBytes, entryOffset + 0x104);
                if (float.IsFinite(mid) && mid > 0.01f && mid < 0.99f)
                    MidPoint = mid;

                bool hasValidColor = false;
                bool hasAlpha = false;
                for (int c = 0; c < 3; c++)
                {
                    int cOfs = entryOffset + 0x108 + (c * 4);
                    byte b = rawBytes[cOfs + 0];
                    byte g = rawBytes[cOfs + 1];
                    byte r = rawBytes[cOfs + 2];
                    byte a = rawBytes[cOfs + 3];

                    if (r > 0 || g > 0 || b > 0)
                        hasValidColor = true;
                    if (a > 0)
                        hasAlpha = true;

                    Colors[c] = new Vector4(r / 255f, g / 255f, b / 255f, a / 255f);
                }

                float maxAlpha = Math.Max(Colors[0].W, Math.Max(Colors[1].W, Colors[2].W));
                if (!hasValidColor)
                {
                    Colors[0] = new Vector4(1f, 1f, 1f, 0.0f);
                    Colors[1] = new Vector4(1f, 1f, 1f, 1.0f);
                    Colors[2] = new Vector4(1f, 1f, 1f, 0.0f);
                }
                else if (!hasAlpha || maxAlpha < 0.15f)
                {
                    // When alpha is unauthored or near-zero across all keys (e.g. placeholder DBC index), provide standard lifecycle fade
                    float peakAlpha = IsAdditive ? 1.0f : 0.85f;
                    Colors[0] = new Vector4(Colors[0].X, Colors[0].Y, Colors[0].Z, 0.0f);
                    Colors[1] = new Vector4(Colors[1].X, Colors[1].Y, Colors[1].Z, peakAlpha);
                    Colors[2] = new Vector4(Colors[2].X, Colors[2].Y, Colors[2].Z, 0.0f);
                }

                bool hasValidScale = false;
                for (int s = 0; s < 3; s++)
                {
                    int sOfs = entryOffset + 0x114 + (s * 4);
                    float val = BitConverter.ToSingle(rawBytes, sOfs);
                    if (float.IsFinite(val) && val > 0.001f)
                    {
                        Scales[s] = val;
                        hasValidScale = true;
                    }
                }

                if (!hasValidScale)
                {
                    Scales[0] = 0.5f;
                    Scales[1] = 1.0f;
                    Scales[2] = 0.5f;
                }
            }
        }
        catch
        {
            // Retain safe defaults on format edge cases
        }
    }

    public void Update(float deltaTime, Matrix4x4[]? boneMatrices, int sequenceIndex, int timeMs, byte[] rawBytes, M2ModelDocument model)
    {
        if (!IsActive)
            return;

        byte enabledByte = M2TrackSampler.SampleByte(rawBytes, model, sequenceIndex, timeMs, _def.EnabledTrack, 255);
        bool enabled = enabledByte != 0;

        float emissionRate = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.EmissionRateTrack, 0.0f);
        float lifespan = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.LifespanTrack, 0.0f);
        if (lifespan <= 0.01f)
            lifespan = 1.0f;

        float speed = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.EmissionSpeedTrack, 0.0f);
        float speedVariation = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.SpeedVariationTrack, 0.0f);
        float verticalRange = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.VerticalRangeTrack, 0.0f);
        float horizontalRange = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.HorizontalRangeTrack, 0.0f);
        float gravity = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.GravityTrack, 0.0f);
        float areaLength = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.EmissionAreaLengthTrack, 0.0f);
        float areaWidth = M2TrackSampler.SampleSingle(rawBytes, model, sequenceIndex, timeMs, _def.EmissionAreaWidthTrack, 0.0f);

        // Update live particles
        for (int i = _particles.Count - 1; i >= 0; i--)
        {
            M2Particle p = _particles[i];
            p.Age += deltaTime;
            if (p.Age >= p.Lifespan)
            {
                _particles.RemoveAt(i);
                continue;
            }

            p.Velocity.Z -= gravity * deltaTime;
            p.Position += p.Velocity * deltaTime;
            p.LifePhase = Math.Clamp(p.Age / p.Lifespan, 0.0f, 1.0f);
        }

        // Spawn new particles
        if (enabled && emissionRate > 0.01f)
        {
            float emitInterval = 1.0f / emissionRate;
            _timeSinceLastEmit += deltaTime;
            while (_timeSinceLastEmit >= emitInterval && _particles.Count < 2000)
            {
                SpawnParticle(boneMatrices, speed, speedVariation, verticalRange, horizontalRange, areaLength, areaWidth, lifespan);
                _timeSinceLastEmit -= emitInterval;
            }
        }
    }

    private void SpawnParticle(
        Matrix4x4[]? boneMatrices,
        float speed, float speedVariation,
        float verticalRange, float horizontalRange,
        float areaLength, float areaWidth,
        float lifespan)
    {
        Matrix4x4 boneTransform = Matrix4x4.Identity;
        if (boneMatrices != null && _def.BoneIndex < boneMatrices.Length)
        {
            boneTransform = boneMatrices[_def.BoneIndex];
        }

        Vector3 spawnPos = _def.Position;
        if (_def.EmitterType == 1) // Plane generator
        {
            float rx = (float)(_random.NextDouble() * 2.0 - 1.0) * (areaLength * 0.5f);
            float ry = (float)(_random.NextDouble() * 2.0 - 1.0) * (areaWidth * 0.5f);
            spawnPos += new Vector3(rx, ry, 0);
        }
        else if (_def.EmitterType == 2) // Sphere generator
        {
            float radius = areaLength + (float)_random.NextDouble() * Math.Max(areaWidth - areaLength, 0f);
            float u = (float)_random.NextDouble() * 2.0f - 1.0f;
            float theta = (float)(_random.NextDouble() * Math.PI * 2.0);
            float r = MathF.Sqrt(Math.Max(0f, 1f - u * u));
            spawnPos += new Vector3(r * MathF.Cos(theta), r * MathF.Sin(theta), u) * radius;
        }

        spawnPos = Vector3.Transform(spawnPos, boneTransform);

        float polar = (float)(_random.NextDouble() * (verticalRange > 0.001f ? verticalRange : MathF.PI));
        float azimuth = (float)(_random.NextDouble() * (horizontalRange > 0.001f ? horizontalRange : MathF.PI * 2.0));
        float actualSpeed = Math.Max(0.01f, speed + (float)(_random.NextDouble() - 0.5) * speedVariation);

        Vector3 dir = new(
            MathF.Sin(polar) * MathF.Cos(azimuth),
            MathF.Sin(polar) * MathF.Sin(azimuth),
            MathF.Cos(polar)
        );

        Vector3 velocity = Vector3.TransformNormal(dir * actualSpeed, boneTransform);

        _particles.Add(new M2Particle
        {
            Position = spawnPos,
            Velocity = velocity,
            Age = 0f,
            Lifespan = lifespan,
            LifePhase = 0f
        });
    }

    public Vector4 GetParticleColor(M2Particle p)
    {
        float t = p.LifePhase;
        if (t <= MidPoint)
        {
            float localT = MidPoint > 0.001f ? t / MidPoint : 0f;
            return Vector4.Lerp(Colors[0], Colors[1], localT);
        }
        else
        {
            float denom = 1.0f - MidPoint;
            float localT = denom > 0.001f ? (t - MidPoint) / denom : 1f;
            return Vector4.Lerp(Colors[1], Colors[2], localT);
        }
    }

    public float GetParticleSize(M2Particle p)
    {
        float t = p.LifePhase;
        if (t <= MidPoint)
        {
            float localT = MidPoint > 0.001f ? t / MidPoint : 0f;
            return MathHelper.Lerp(Scales[0], Scales[1], localT);
        }
        else
        {
            float denom = 1.0f - MidPoint;
            float localT = denom > 0.001f ? (t - MidPoint) / denom : 1f;
            return MathHelper.Lerp(Scales[1], Scales[2], localT);
        }
    }
}
