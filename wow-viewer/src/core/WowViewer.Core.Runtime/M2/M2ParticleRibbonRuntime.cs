using System.Numerics;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2EffectRuntimeState
{
    public M2EffectRuntimeState(
        IReadOnlyList<M2ParticleRuntimeState> particles,
        IReadOnlyList<M2RibbonRuntimeState> ribbons)
    {
        ArgumentNullException.ThrowIfNull(particles);
        ArgumentNullException.ThrowIfNull(ribbons);

        Particles = particles;
        Ribbons = ribbons;
    }

    public IReadOnlyList<M2ParticleRuntimeState> Particles { get; }

    public IReadOnlyList<M2RibbonRuntimeState> Ribbons { get; }

    public int VisibleParticleEmitterCount => Particles.Count(static particle => particle.Enabled && particle.EstimatedParticleCount > 0);

    public int VisibleRibbonEmitterCount => Ribbons.Count(static ribbon => ribbon.Visible && ribbon.EstimatedEdgeCount > 0);
}

public sealed class M2ParticleRuntimeState
{
    public M2ParticleRuntimeState(
        int index,
        bool enabled,
        Vector3 position,
        ushort boneIndex,
        ushort textureIndex,
        ushort blendingType,
        ushort emitterType,
        byte particleType,
        byte headOrTail,
        float emissionSpeed,
        float speedVariation,
        float verticalRange,
        float horizontalRange,
        float gravity,
        float lifespan,
        float emissionRate,
        float emissionAreaLength,
        float emissionAreaWidth,
        float zSource,
        int estimatedParticleCount,
        string effectKey,
        int stateBucket,
        bool isAdditive,
        bool allowsBatching)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedParticleCount);

        Index = index;
        Enabled = enabled;
        Position = position;
        BoneIndex = boneIndex;
        TextureIndex = textureIndex;
        BlendingType = blendingType;
        EmitterType = emitterType;
        ParticleType = particleType;
        HeadOrTail = headOrTail;
        EmissionSpeed = emissionSpeed;
        SpeedVariation = speedVariation;
        VerticalRange = verticalRange;
        HorizontalRange = horizontalRange;
        Gravity = gravity;
        Lifespan = lifespan;
        EmissionRate = emissionRate;
        EmissionAreaLength = emissionAreaLength;
        EmissionAreaWidth = emissionAreaWidth;
        ZSource = zSource;
        EstimatedParticleCount = estimatedParticleCount;
        EffectKey = effectKey;
        StateBucket = stateBucket;
        IsAdditive = isAdditive;
        AllowsBatching = allowsBatching;
    }

    public int Index { get; }

    public bool Enabled { get; }

    public Vector3 Position { get; }

    public ushort BoneIndex { get; }

    public ushort TextureIndex { get; }

    public ushort BlendingType { get; }

    public ushort EmitterType { get; }

    public byte ParticleType { get; }

    public byte HeadOrTail { get; }

    public float EmissionSpeed { get; }

    public float SpeedVariation { get; }

    public float VerticalRange { get; }

    public float HorizontalRange { get; }

    public float Gravity { get; }

    public float Lifespan { get; }

    public float EmissionRate { get; }

    public float EmissionAreaLength { get; }

    public float EmissionAreaWidth { get; }

    public float ZSource { get; }

    public int EstimatedParticleCount { get; }

    public string EffectKey { get; }

    public int StateBucket { get; }

    public bool IsAdditive { get; }

    public bool AllowsBatching { get; }

    public int EstimatedVertexCount => EstimatedParticleCount * 4;

    public int EstimatedIndexCount => EstimatedParticleCount * 6;
}

public sealed class M2RibbonRuntimeState
{
    public M2RibbonRuntimeState(
        int index,
        bool visible,
        Vector3 position,
        uint boneIndex,
        Vector3 color,
        float alpha,
        float heightAbove,
        float heightBelow,
        float edgesPerSecond,
        float edgeLifetime,
        float gravity,
        ushort textureSlot,
        int textureSortKey,
        int materialSortKey,
        int estimatedEdgeCount,
        string effectKey,
        int stateBucket)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedEdgeCount);

        Index = index;
        Visible = visible;
        Position = position;
        BoneIndex = boneIndex;
        Color = color;
        Alpha = alpha;
        HeightAbove = heightAbove;
        HeightBelow = heightBelow;
        EdgesPerSecond = edgesPerSecond;
        EdgeLifetime = edgeLifetime;
        Gravity = gravity;
        TextureSlot = textureSlot;
        TextureSortKey = textureSortKey;
        MaterialSortKey = materialSortKey;
        EstimatedEdgeCount = estimatedEdgeCount;
        EffectKey = effectKey;
        StateBucket = stateBucket;
    }

    public int Index { get; }

    public bool Visible { get; }

    public Vector3 Position { get; }

    public uint BoneIndex { get; }

    public Vector3 Color { get; }

    public float Alpha { get; }

    public float HeightAbove { get; }

    public float HeightBelow { get; }

    public float EdgesPerSecond { get; }

    public float EdgeLifetime { get; }

    public float Gravity { get; }

    public ushort TextureSlot { get; }

    public int TextureSortKey { get; }

    public int MaterialSortKey { get; }

    public int EstimatedEdgeCount { get; }

    public string EffectKey { get; }

    public int StateBucket { get; }

    public int EstimatedVertexCount => EstimatedEdgeCount * 2;

    public int EstimatedIndexCount => Math.Max(0, EstimatedEdgeCount - 1) * 6;
}

public static class M2ParticleRibbonRuntimeEvaluator
{
    public static M2EffectRuntimeState Evaluate(M2ModelDocument model, int sequenceIndex, int timeMs)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (sequenceIndex < 0 || sequenceIndex >= model.Sequences.Count)
            throw new ArgumentOutOfRangeException(nameof(sequenceIndex), "Sequence index is outside the M2 sequence table.");

        List<M2ParticleRuntimeState> particles = new(model.Particles.Count);
        foreach (M2ParticleDefinition particle in model.Particles)
            particles.Add(EvaluateParticle(model, particle, sequenceIndex, timeMs));

        List<M2RibbonRuntimeState> ribbons = new(model.Ribbons.Count);
        foreach (M2RibbonDefinition ribbon in model.Ribbons)
            ribbons.Add(EvaluateRibbon(model, ribbon, sequenceIndex, timeMs));

        return new M2EffectRuntimeState(particles, ribbons);
    }

    public static IEnumerable<M2ParticleSubmissionDescriptor> BuildParticleSubmissionDescriptors(M2EffectRuntimeState state, string modelKey)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);

        foreach (M2ParticleRuntimeState particle in state.Particles)
        {
            if (!particle.Enabled || particle.EstimatedParticleCount == 0)
                continue;

            yield return new M2ParticleSubmissionDescriptor(
                $"particle:{particle.Index}",
                modelKey,
                particle.EffectKey,
                particle.TextureIndex,
                particle.StateBucket,
                particle.EstimatedVertexCount,
                particle.EstimatedIndexCount,
                particle.Position.Z,
                particle.IsAdditive,
                particle.AllowsBatching);
        }
    }

    public static IEnumerable<M2RibbonSubmissionDescriptor> BuildRibbonSubmissionDescriptors(M2EffectRuntimeState state, string modelKey)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);

        foreach (M2RibbonRuntimeState ribbon in state.Ribbons)
        {
            if (!ribbon.Visible || ribbon.EstimatedEdgeCount == 0)
                continue;

            yield return new M2RibbonSubmissionDescriptor(
                $"ribbon:{ribbon.Index}",
                modelKey,
                ribbon.EffectKey,
                ribbon.TextureSortKey,
                ribbon.StateBucket,
                ribbon.EstimatedVertexCount,
                ribbon.EstimatedIndexCount,
                ribbon.Position.Z);
        }
    }

    private static M2ParticleRuntimeState EvaluateParticle(M2ModelDocument model, M2ParticleDefinition particle, int sequenceIndex, int timeMs)
    {
        byte enabled = M2TrackSampler.SampleByte(model.RawBytes, model, sequenceIndex, timeMs, particle.EnabledTrack, byte.MaxValue);
        float emissionRate = Math.Max(0.0f, M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.EmissionRateTrack, 0.0f));
        float lifespan = Math.Max(0.0f, M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.LifespanTrack, 0.0f));
        bool isEnabled = enabled != 0;
        int estimatedParticleCount = isEnabled
            ? ClampDispatchCount(MathF.Ceiling(emissionRate * lifespan))
            : 0;

        return new M2ParticleRuntimeState(
            particle.Index,
            isEnabled,
            particle.Position,
            particle.BoneIndex,
            particle.TextureIndex,
            particle.BlendingType,
            particle.EmitterType,
            particle.ParticleType,
            particle.HeadOrTail,
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.EmissionSpeedTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.SpeedVariationTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.VerticalRangeTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.HorizontalRangeTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.GravityTrack, 0.0f),
            lifespan,
            emissionRate,
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.EmissionAreaLengthTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.EmissionAreaWidthTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, particle.ZSourceTrack, 0.0f),
            estimatedParticleCount,
            ResolveParticleEffectKey(particle.BlendingType),
            BuildParticleStateBucket(particle),
            IsAdditiveParticleBlend(particle.BlendingType),
            !particle.UsesModelParticle && !particle.UsesRecursiveParticleModel);
    }

    private static M2RibbonRuntimeState EvaluateRibbon(M2ModelDocument model, M2RibbonDefinition ribbon, int sequenceIndex, int timeMs)
    {
        byte visible = M2TrackSampler.SampleByte(model.RawBytes, model, sequenceIndex, timeMs, ribbon.VisibilityTrack, byte.MaxValue);
        float edgesPerSecond = Math.Max(0.0f, ribbon.EdgesPerSecond);
        float edgeLifetime = Math.Max(0.0f, ribbon.EdgeLifetime);
        bool isVisible = visible != 0;
        int estimatedEdgeCount = isVisible
            ? ClampDispatchCount(MathF.Ceiling(edgesPerSecond * edgeLifetime))
            : 0;
        int textureSortKey = ribbon.TextureIndices.Count > 0 ? ribbon.TextureIndices[0] : 0;
        int materialSortKey = ribbon.MaterialIndices.Count > 0 ? ribbon.MaterialIndices[0] : -1;

        return new M2RibbonRuntimeState(
            ribbon.Index,
            isVisible,
            ribbon.Position,
            ribbon.BoneIndex,
            Clamp01(M2TrackSampler.SampleVector3(model.RawBytes, model, sequenceIndex, timeMs, ribbon.ColorTrack, Vector3.One)),
            DecodeFixedAlpha(M2TrackSampler.SampleInt16(model.RawBytes, model, sequenceIndex, timeMs, ribbon.AlphaTrack, short.MaxValue)),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, ribbon.HeightAboveTrack, 0.0f),
            M2TrackSampler.SampleSingle(model.RawBytes, model, sequenceIndex, timeMs, ribbon.HeightBelowTrack, 0.0f),
            ribbon.EdgesPerSecond,
            ribbon.EdgeLifetime,
            ribbon.Gravity,
            M2TrackSampler.SampleUInt16(model.RawBytes, model, sequenceIndex, timeMs, ribbon.TextureSlotTrack, 0),
            textureSortKey,
            materialSortKey,
            estimatedEdgeCount,
            materialSortKey >= 0 ? $"Ribbon_Material_{materialSortKey}" : "Ribbon_Default",
            BuildRibbonStateBucket(ribbon, materialSortKey));
    }

    private static string ResolveParticleEffectKey(ushort blendingType)
    {
        return blendingType switch
        {
            0 => "Particle_Opaque",
            1 => "Particle_Mod",
            2 => "Particle_AlphaBlend",
            3 => "Particle_AlphaKey",
            4 => "Particle_Additive",
            5 => "Particle_Mod2x",
            6 => "Particle_BlendAdditive",
            _ => $"Particle_Blend_{blendingType}",
        };
    }

    private static bool IsAdditiveParticleBlend(ushort blendingType)
    {
        return blendingType is 1 or 4 or 5 or 6;
    }

    private static int BuildParticleStateBucket(M2ParticleDefinition particle)
    {
        return (particle.BlendingType & 0xFF)
            | ((particle.EmitterType & 0xFF) << 8)
            | ((particle.ParticleType & 0x0F) << 16)
            | ((particle.HeadOrTail & 0x0F) << 20);
    }

    private static int BuildRibbonStateBucket(M2RibbonDefinition ribbon, int materialSortKey)
    {
        return (materialSortKey & 0xFFFF)
            | ((ribbon.TextureRows & 0xFF) << 16)
            | ((ribbon.TextureColumns & 0xFF) << 24);
    }

    private static int ClampDispatchCount(float value)
    {
        if (!float.IsFinite(value) || value <= 0.0f)
            return 0;

        return (int)Math.Clamp(value, 0.0f, 65535.0f);
    }

    private static float DecodeFixedAlpha(short value)
    {
        return Math.Clamp(value / 32767.0f, 0.0f, 1.0f);
    }

    private static Vector3 Clamp01(Vector3 value)
    {
        return new Vector3(
            Math.Clamp(value.X, 0.0f, 1.0f),
            Math.Clamp(value.Y, 0.0f, 1.0f),
            Math.Clamp(value.Z, 0.0f, 1.0f));
    }
}
