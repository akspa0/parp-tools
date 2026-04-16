using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2ParticleSubmissionDescriptor
{
    public M2ParticleSubmissionDescriptor(
        string emitterKey,
        string modelKey,
        string effectKey,
        int textureSortKey,
        int stateBucket,
        int estimatedVertices,
        int estimatedIndices,
        float depthSortValue,
        bool isAdditive,
        bool allowsBatching = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(emitterKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedVertices);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedIndices);

        EmitterKey = emitterKey;
        ModelKey = modelKey;
        EffectKey = effectKey;
        TextureSortKey = textureSortKey;
        StateBucket = stateBucket;
        EstimatedVertices = estimatedVertices;
        EstimatedIndices = estimatedIndices;
        DepthSortValue = depthSortValue;
        IsAdditive = isAdditive;
        AllowsBatching = allowsBatching;
    }

    public string EmitterKey { get; }

    public string ModelKey { get; }

    public string EffectKey { get; }

    public int TextureSortKey { get; }

    public int StateBucket { get; }

    public int EstimatedVertices { get; }

    public int EstimatedIndices { get; }

    public float DepthSortValue { get; }

    public bool IsAdditive { get; }

    public bool AllowsBatching { get; }
}

public sealed class M2RibbonSubmissionDescriptor
{
    public M2RibbonSubmissionDescriptor(
        string ribbonKey,
        string modelKey,
        string effectKey,
        int textureSortKey,
        int stateBucket,
        int estimatedVertices,
        int estimatedIndices,
        float depthSortValue)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ribbonKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectKey);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedVertices);
        ArgumentOutOfRangeException.ThrowIfNegative(estimatedIndices);

        RibbonKey = ribbonKey;
        ModelKey = modelKey;
        EffectKey = effectKey;
        TextureSortKey = textureSortKey;
        StateBucket = stateBucket;
        EstimatedVertices = estimatedVertices;
        EstimatedIndices = estimatedIndices;
        DepthSortValue = depthSortValue;
    }

    public string RibbonKey { get; }

    public string ModelKey { get; }

    public string EffectKey { get; }

    public int TextureSortKey { get; }

    public int StateBucket { get; }

    public int EstimatedVertices { get; }

    public int EstimatedIndices { get; }

    public float DepthSortValue { get; }
}

public static class M2SceneSubmissionEntryBuilder
{
    public static IEnumerable<M2SceneSubmissionEntry> BuildRenderEntries(
        M2ModelDocument model,
        M2StaticRenderModel renderModel,
        M2RenderConsumerFrameState consumerState)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(renderModel);
        ArgumentNullException.ThrowIfNull(consumerState);

        foreach (M2RenderConsumerPassState passState in consumerState.Passes)
        {
            M2StaticRenderMaterial material = passState.SourcePass.Material;
            M2StructuredRenderSection? section = renderModel.StructuredSections.FirstOrDefault(section => section.SectionIndex == passState.AnimatedPass.SectionIndex);
            int vertexCount = section?.Vertices.Count ?? 0;
            int indexCount = section?.Indices.Count ?? 0;
            M2ResolvedEffect effect = passState.ResolvedEffect;
            M2RenderEntryFamily family = effect.IsProjected
                ? M2RenderEntryFamily.Projected
                : M2RenderEntryFamily.Core;

            yield return new M2SceneSubmissionEntry(
                $"section{passState.AnimatedPass.SectionIndex}:pass{passState.AnimatedPass.PassIndex}:batch{passState.AnimatedPass.BatchIndex}",
                model.Identity.CanonicalModelPath,
                family,
                effect.EffectObjectKey,
                material.TextureComboIndex,
                effect.StateBucket,
                vertexCount,
                indexCount,
                depthSortValue: passState.AnimatedPass.SectionIndex,
                isTransparent: effect.IsTransparent || passState.Alpha < 0.999f,
                isAdditive: effect.IsAdditive,
                sectionIndex: passState.AnimatedPass.SectionIndex,
                passIndex: passState.AnimatedPass.PassIndex,
                batchIndex: passState.AnimatedPass.BatchIndex);
        }
    }

    public static IEnumerable<M2SceneSubmissionEntry> BuildParticleEntries(IEnumerable<M2ParticleSubmissionDescriptor> particles)
    {
        ArgumentNullException.ThrowIfNull(particles);

        foreach (M2ParticleSubmissionDescriptor particle in particles)
        {
            int stateBucket = particle.AllowsBatching
                ? particle.StateBucket
                : particle.StateBucket | (1 << 14);

            yield return new M2SceneSubmissionEntry(
                particle.EmitterKey,
                particle.ModelKey,
                M2RenderEntryFamily.Particle,
                particle.EffectKey,
                particle.TextureSortKey,
                stateBucket,
                particle.EstimatedVertices,
                particle.EstimatedIndices,
                particle.DepthSortValue,
                isTransparent: true,
                isAdditive: particle.IsAdditive,
                forceDirect: !particle.AllowsBatching);
        }
    }

    public static IEnumerable<M2SceneSubmissionEntry> BuildRibbonEntries(IEnumerable<M2RibbonSubmissionDescriptor> ribbons)
    {
        ArgumentNullException.ThrowIfNull(ribbons);

        foreach (M2RibbonSubmissionDescriptor ribbon in ribbons)
        {
            yield return new M2SceneSubmissionEntry(
                ribbon.RibbonKey,
                ribbon.ModelKey,
                M2RenderEntryFamily.Ribbon,
                ribbon.EffectKey,
                ribbon.TextureSortKey,
                ribbon.StateBucket,
                ribbon.EstimatedVertices,
                ribbon.EstimatedIndices,
                ribbon.DepthSortValue,
                isTransparent: true,
                isAdditive: false);
        }
    }
}
