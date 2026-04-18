using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Runtime.Mdx;

public sealed class MdxEffectRuntimeState
{
    public MdxEffectRuntimeState(
        IReadOnlyList<MdxEventRuntimeState> events,
        IReadOnlyList<MdxParticleEmitter2RuntimeState> particles,
        IReadOnlyList<MdxRibbonRuntimeState> ribbons)
    {
        ArgumentNullException.ThrowIfNull(events);
        ArgumentNullException.ThrowIfNull(particles);
        ArgumentNullException.ThrowIfNull(ribbons);

        Events = events;
        Particles = particles;
        Ribbons = ribbons;
    }

    public IReadOnlyList<MdxEventRuntimeState> Events { get; }

    public IReadOnlyList<MdxParticleEmitter2RuntimeState> Particles { get; }

    public IReadOnlyList<MdxRibbonRuntimeState> Ribbons { get; }

    public int TriggeredEventCount => Events.Count(static effectEvent => effectEvent.Triggered);

    public int VisibleParticleEmitterCount => Particles.Count(static particle => particle.Enabled && particle.EstimatedParticleCount > 0);

    public int VisibleRibbonEmitterCount => Ribbons.Count(static ribbon => ribbon.Visible && ribbon.EstimatedEdgeCount > 0);
}

public sealed class MdxEventRuntimeState
{
    public MdxEventRuntimeState(
        int index,
        string name,
        string tag,
        Vector3 position,
        int resolvedFrameTime,
        int keyCount,
        bool triggered,
        int? nextKeyTime)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(tag);
        ArgumentOutOfRangeException.ThrowIfNegative(keyCount);

        Index = index;
        Name = name;
        Tag = tag;
        Position = position;
        ResolvedFrameTime = resolvedFrameTime;
        KeyCount = keyCount;
        Triggered = triggered;
        NextKeyTime = nextKeyTime;
    }

    public int Index { get; }

    public string Name { get; }

    public string Tag { get; }

    public Vector3 Position { get; }

    public int ResolvedFrameTime { get; }

    public int KeyCount { get; }

    public bool Triggered { get; }

    public int? NextKeyTime { get; }
}

public sealed class MdxParticleEmitter2RuntimeState
{
    public MdxParticleEmitter2RuntimeState(
        int index,
        string name,
        bool enabled,
        Vector3 position,
        uint blendMode,
        int textureId,
        uint replaceableId,
        float visibility,
        float speed,
        float variation,
        float latitude,
        float longitude,
        float gravity,
        float life,
        float emissionRate,
        float width,
        float length,
        float zSource,
        int estimatedParticleCount,
        string effectKey,
        int stateBucket,
        bool usesModelParticles)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedParticleCount);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);

        Index = index;
        Name = name;
        Enabled = enabled;
        Position = position;
        BlendMode = blendMode;
        TextureId = textureId;
        ReplaceableId = replaceableId;
        Visibility = visibility;
        Speed = speed;
        Variation = variation;
        Latitude = latitude;
        Longitude = longitude;
        Gravity = gravity;
        Life = life;
        EmissionRate = emissionRate;
        Width = width;
        Length = length;
        ZSource = zSource;
        EstimatedParticleCount = estimatedParticleCount;
        EffectKey = effectKey;
        StateBucket = stateBucket;
        UsesModelParticles = usesModelParticles;
    }

    public int Index { get; }

    public string Name { get; }

    public bool Enabled { get; }

    public Vector3 Position { get; }

    public uint BlendMode { get; }

    public int TextureId { get; }

    public uint ReplaceableId { get; }

    public float Visibility { get; }

    public float Speed { get; }

    public float Variation { get; }

    public float Latitude { get; }

    public float Longitude { get; }

    public float Gravity { get; }

    public float Life { get; }

    public float EmissionRate { get; }

    public float Width { get; }

    public float Length { get; }

    public float ZSource { get; }

    public int EstimatedParticleCount { get; }

    public string EffectKey { get; }

    public int StateBucket { get; }

    public bool UsesModelParticles { get; }
}

public sealed class MdxRibbonRuntimeState
{
    public MdxRibbonRuntimeState(
        int index,
        string name,
        bool visible,
        Vector3 position,
        Vector3 color,
        float alpha,
        float heightAbove,
        float heightBelow,
        float edgesPerSecond,
        float edgeLifetime,
        float gravity,
        int textureSlot,
        uint materialId,
        int estimatedEdgeCount,
        string effectKey,
        int stateBucket)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedEdgeCount);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);

        Index = index;
        Name = name;
        Visible = visible;
        Position = position;
        Color = color;
        Alpha = alpha;
        HeightAbove = heightAbove;
        HeightBelow = heightBelow;
        EdgesPerSecond = edgesPerSecond;
        EdgeLifetime = edgeLifetime;
        Gravity = gravity;
        TextureSlot = textureSlot;
        MaterialId = materialId;
        EstimatedEdgeCount = estimatedEdgeCount;
        EffectKey = effectKey;
        StateBucket = stateBucket;
    }

    public int Index { get; }

    public string Name { get; }

    public bool Visible { get; }

    public Vector3 Position { get; }

    public Vector3 Color { get; }

    public float Alpha { get; }

    public float HeightAbove { get; }

    public float HeightBelow { get; }

    public float EdgesPerSecond { get; }

    public float EdgeLifetime { get; }

    public float Gravity { get; }

    public int TextureSlot { get; }

    public uint MaterialId { get; }

    public int EstimatedEdgeCount { get; }

    public string EffectKey { get; }

    public int StateBucket { get; }
}

public static class MdxEffectRuntimeEvaluator
{
    public static MdxEffectRuntimeState Evaluate(
        MdxSummary summary,
        MdxEventFile events,
        MdxParticleEmitter2File particles,
        MdxRibbonEmitterFile ribbons,
        int sequenceIndex,
        int timeMs)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentNullException.ThrowIfNull(events);
        ArgumentNullException.ThrowIfNull(particles);
        ArgumentNullException.ThrowIfNull(ribbons);

        List<MdxEventRuntimeState> eventStates = new(events.EventCount);
        foreach (MdxEvent effectEvent in events.Events)
            eventStates.Add(EvaluateEvent(summary, effectEvent, sequenceIndex, timeMs));

        List<MdxParticleEmitter2RuntimeState> particleStates = new(particles.ParticleEmitterCount);
        foreach (MdxParticleEmitter2 particle in particles.ParticleEmitters)
            particleStates.Add(EvaluateParticle(summary, particle, sequenceIndex, timeMs));

        List<MdxRibbonRuntimeState> ribbonStates = new(ribbons.RibbonCount);
        foreach (MdxRibbonEmitter ribbon in ribbons.Ribbons)
            ribbonStates.Add(EvaluateRibbon(summary, ribbon, sequenceIndex, timeMs));

        return new MdxEffectRuntimeState(eventStates, particleStates, ribbonStates);
    }

    private static MdxEventRuntimeState EvaluateEvent(MdxSummary summary, MdxEvent effectEvent, int sequenceIndex, int timeMs)
    {
        string tag = effectEvent.EventTrack?.Tag ?? "KEVT";
        int resolvedFrameTime = ResolveTrackFrame(summary, sequenceIndex, timeMs, effectEvent.EventTrack?.GlobalSequenceId ?? -1);
        bool triggered = effectEvent.EventTrack?.KeyTimes.Contains(resolvedFrameTime) == true;
        int? nextKeyTime = effectEvent.EventTrack?.KeyTimes.FirstOrDefault(static _ => true);
        if (effectEvent.EventTrack is not null)
        {
            nextKeyTime = effectEvent.EventTrack.KeyTimes
                .Where(keyTime => keyTime >= resolvedFrameTime)
                .Cast<int?>()
                .FirstOrDefault();
        }

        return new MdxEventRuntimeState(
            effectEvent.Index,
            effectEvent.Name,
            tag,
            ResolveNodePosition(summary, effectEvent.PivotPoint, effectEvent.TranslationTrack, sequenceIndex, timeMs),
            resolvedFrameTime,
            effectEvent.EventTrack?.KeyCount ?? 0,
            triggered,
            nextKeyTime);
    }

    private static MdxParticleEmitter2RuntimeState EvaluateParticle(MdxSummary summary, MdxParticleEmitter2 particle, int sequenceIndex, int timeMs)
    {
        float visibility = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.VisibilityTrack, summary, sequenceIndex, timeMs, 1.0f));
        float speed = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.SpeedTrack, summary, sequenceIndex, timeMs, particle.StaticSpeed));
        float variation = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.VariationTrack, summary, sequenceIndex, timeMs, particle.StaticVariation));
        float latitude = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.LatitudeTrack, summary, sequenceIndex, timeMs, particle.StaticLatitude));
        float longitude = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.LongitudeTrack, summary, sequenceIndex, timeMs, particle.StaticLongitude));
        float gravity = MdxAnimationSampler.SampleScalarTrack(particle.GravityTrack, summary, sequenceIndex, timeMs, particle.StaticGravity);
        float life = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.LifeTrack, summary, sequenceIndex, timeMs, particle.StaticLife));
        float emissionRate = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.EmissionRateTrack, summary, sequenceIndex, timeMs, particle.StaticEmissionRate));
        float width = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.WidthTrack, summary, sequenceIndex, timeMs, particle.StaticWidth));
        float length = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(particle.LengthTrack, summary, sequenceIndex, timeMs, particle.StaticLength));
        float zSource = MdxAnimationSampler.SampleScalarTrack(particle.ZSourceTrack, summary, sequenceIndex, timeMs, particle.StaticZSource);

        bool enabled = visibility > 0.001f;
        int estimatedParticleCount = enabled
            ? ClampDispatchCount(MathF.Ceiling(emissionRate * life))
            : 0;

        return new MdxParticleEmitter2RuntimeState(
            particle.Index,
            particle.Name,
            enabled,
            ResolveNodePosition(summary, particle.PivotPoint, particle.TranslationTrack, sequenceIndex, timeMs),
            particle.BlendMode,
            particle.TextureId,
            particle.ReplaceableId,
            visibility,
            speed,
            variation,
            latitude,
            longitude,
            gravity,
            life,
            emissionRate,
            width,
            length,
            zSource,
            estimatedParticleCount,
            ResolveParticleEffectKey(particle.BlendMode),
            BuildParticleStateBucket(particle),
            particle.HasGeometryModel || particle.HasRecursionModel);
    }

    private static MdxRibbonRuntimeState EvaluateRibbon(MdxSummary summary, MdxRibbonEmitter ribbon, int sequenceIndex, int timeMs)
    {
        float visibility = Math.Max(0.0f, MdxAnimationSampler.SampleScalarTrack(ribbon.VisibilityTrack, summary, sequenceIndex, timeMs, 1.0f));
        bool visible = visibility > 0.001f;
        float alpha = Math.Clamp(MdxAnimationSampler.SampleScalarTrack(ribbon.AlphaTrack, summary, sequenceIndex, timeMs, ribbon.StaticAlpha), 0.0f, 1.0f);
        float heightAbove = MdxAnimationSampler.SampleScalarTrack(ribbon.HeightAboveTrack, summary, sequenceIndex, timeMs, ribbon.StaticHeightAbove);
        float heightBelow = MdxAnimationSampler.SampleScalarTrack(ribbon.HeightBelowTrack, summary, sequenceIndex, timeMs, ribbon.StaticHeightBelow);
        Vector3 color = Clamp01(MdxAnimationSampler.SampleColorTrack(ribbon.ColorTrack, summary, sequenceIndex, timeMs, ribbon.StaticColor));
        int textureSlot = Math.Max(0, MdxAnimationSampler.SampleIntTrack(ribbon.TextureSlotTrack, summary, sequenceIndex, timeMs, (int)ribbon.StaticTextureSlot));
        int estimatedEdgeCount = visible
            ? ClampDispatchCount(MathF.Ceiling(ribbon.EdgesPerSecond * ribbon.EdgeLifetime))
            : 0;

        return new MdxRibbonRuntimeState(
            ribbon.Index,
            ribbon.Name,
            visible,
            ResolveNodePosition(summary, ribbon.PivotPoint, ribbon.TranslationTrack, sequenceIndex, timeMs),
            color,
            alpha,
            heightAbove,
            heightBelow,
            ribbon.EdgesPerSecond,
            ribbon.EdgeLifetime,
            ribbon.Gravity,
            textureSlot,
            ribbon.MaterialId,
            estimatedEdgeCount,
            ribbon.MaterialId != 0 ? $"Ribbon_Material_{ribbon.MaterialId}" : "Ribbon_Default",
            BuildRibbonStateBucket(ribbon));
    }

    private static Vector3 ResolveNodePosition(MdxSummary summary, Vector3 pivotPoint, MdxVector3NodeTrack? translationTrack, int sequenceIndex, int timeMs)
    {
        Vector3 translation = MdxAnimationSampler.SampleVector3Track(translationTrack, summary, sequenceIndex, timeMs, Vector3.Zero);
        return pivotPoint + translation;
    }

    private static int ResolveTrackFrame(MdxSummary summary, int sequenceIndex, int timeMs, int globalSequenceId)
    {
        if (globalSequenceId >= 0 && globalSequenceId < summary.GlobalSequenceCount)
        {
            uint duration = summary.GlobalSequences[globalSequenceId].Duration;
            if (duration == 0)
                return 0;

            return (int)(Math.Max(0, timeMs) % duration);
        }

        return MdxAnimationSampler.ResolveSequenceFrame(summary, sequenceIndex, timeMs);
    }

    private static string ResolveParticleEffectKey(uint blendMode)
    {
        return blendMode switch
        {
            0 => "Particle_Opaque",
            1 => "Particle_TransparentKey",
            2 => "Particle_AlphaBlend",
            3 => "Particle_Additive",
            4 => "Particle_AddAlpha",
            5 => "Particle_Modulate",
            6 => "Particle_Modulate2X",
            _ => $"Particle_Blend_{blendMode}",
        };
    }

    private static int BuildParticleStateBucket(MdxParticleEmitter2 particle)
    {
        return (int)(particle.BlendMode & 0xFF)
            | ((particle.EmitterType & 0xFF) << 8)
            | (((int)particle.ParticleType & 0xFF) << 16)
            | ((particle.TextureId & 0xFF) << 24);
    }

    private static int BuildRibbonStateBucket(MdxRibbonEmitter ribbon)
    {
        return ((int)ribbon.MaterialId & 0xFFFF)
            | (((int)ribbon.TextureRows & 0xFF) << 16)
            | (((int)ribbon.TextureColumns & 0xFF) << 24);
    }

    private static int ClampDispatchCount(float value)
    {
        if (!float.IsFinite(value) || value <= 0.0f)
            return 0;

        return (int)Math.Clamp(value, 0.0f, 65535.0f);
    }

    private static Vector3 Clamp01(Vector3 value)
    {
        return new Vector3(
            Math.Clamp(value.X, 0.0f, 1.0f),
            Math.Clamp(value.Y, 0.0f, 1.0f),
            Math.Clamp(value.Z, 0.0f, 1.0f));
    }
}