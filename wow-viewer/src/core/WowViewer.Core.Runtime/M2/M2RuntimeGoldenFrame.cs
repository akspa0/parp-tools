using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed record M2RuntimeGoldenFrame(
    string CanonicalModelPath,
    uint ModelVersion,
    int RequestedSequenceIndex,
    int ResolvedSequenceIndex,
    int TimeMs,
    bool UsesExternalAnimationPayload,
    int BoneCount,
    int SkinnedVertexCount,
    int RenderPassCount,
    int VisiblePassCount,
    string ModelAmbient,
    string ModelDiffuse,
    IReadOnlyList<M2RuntimeGoldenEffect> Effects,
    IReadOnlyList<M2RuntimeGoldenBatch> Batches,
    string RuntimeHash);

public sealed record M2RuntimeGoldenEffect(
    int SectionIndex,
    int PassIndex,
    string RecipeKey,
    string EffectObjectKey,
    string Diffuse,
    string Emissive,
    float Alpha,
    bool Visible);

public sealed record M2RuntimeGoldenBatch(
    int BatchIndex,
    string Family,
    string Handler,
    bool Direct,
    int EntryCount,
    string EffectKey,
    int VertexCount,
    int IndexCount);

public static class M2RuntimeGoldenFrameBuilder
{
    public static M2RuntimeGoldenFrame Build(
        M2ModelDocument model,
        M2AnimatedRenderState animatedState,
        M2BonePoseState bonePoseState,
        M2SkinnedRenderModel? skinnedRenderModel,
        M2RenderConsumerFrameState consumerState,
        M2SceneSubmissionPlan submissionPlan)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(animatedState);
        ArgumentNullException.ThrowIfNull(bonePoseState);
        ArgumentNullException.ThrowIfNull(consumerState);
        ArgumentNullException.ThrowIfNull(submissionPlan);

        M2RuntimeGoldenEffect[] effects = consumerState.Passes
            .Select(static pass => new M2RuntimeGoldenEffect(
                pass.AnimatedPass.SectionIndex,
                pass.AnimatedPass.PassIndex,
                pass.EffectKey,
                pass.ResolvedEffect.EffectObjectKey,
                FormatVector(pass.DiffuseColor),
                FormatVector(pass.EmissiveColor),
                Round(pass.Alpha),
                pass.Visible))
            .ToArray();

        M2RuntimeGoldenBatch[] batches = submissionPlan.Batches
            .Select(static batch => new M2RuntimeGoldenBatch(
                batch.BatchIndex,
                batch.Family.ToString(),
                batch.HandlerName,
                batch.IsDirect,
                batch.Entries.Count,
                batch.EffectKey,
                batch.VertexCount,
                batch.IndexCount))
            .ToArray();

        string hash = ComputeHash(model, animatedState, bonePoseState, skinnedRenderModel, consumerState, batches, effects);
        return new M2RuntimeGoldenFrame(
            model.Identity.CanonicalModelPath,
            model.Version,
            animatedState.RequestedSequenceIndex,
            animatedState.ResolvedSequenceIndex,
            animatedState.TimeMs,
            animatedState.UsesExternalPayload,
            bonePoseState.BoneCount,
            skinnedRenderModel?.VertexCount ?? 0,
            consumerState.Passes.Count,
            consumerState.VisiblePassCount,
            FormatVector(consumerState.ModelAmbient),
            FormatVector(consumerState.ModelDiffuse),
            effects,
            batches,
            hash);
    }

    private static string ComputeHash(
        M2ModelDocument model,
        M2AnimatedRenderState animatedState,
        M2BonePoseState bonePoseState,
        M2SkinnedRenderModel? skinnedRenderModel,
        M2RenderConsumerFrameState consumerState,
        IReadOnlyList<M2RuntimeGoldenBatch> batches,
        IReadOnlyList<M2RuntimeGoldenEffect> effects)
    {
        StringBuilder builder = new();
        builder.Append(model.Identity.CanonicalModelPath).Append('|')
            .Append(model.Version.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(animatedState.RequestedSequenceIndex.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(animatedState.ResolvedSequenceIndex.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(animatedState.TimeMs.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(animatedState.UsesExternalPayload).Append('|')
            .Append(bonePoseState.BoneCount.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append((skinnedRenderModel?.VertexCount ?? 0).ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(consumerState.Passes.Count.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(consumerState.VisiblePassCount.ToString(CultureInfo.InvariantCulture)).Append('|')
            .Append(FormatVector(consumerState.ModelAmbient)).Append('|')
            .Append(FormatVector(consumerState.ModelDiffuse));

        foreach (M2RuntimeGoldenEffect effect in effects)
        {
            builder.Append("|effect:")
                .Append(effect.SectionIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(effect.PassIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(effect.RecipeKey).Append(',')
                .Append(effect.EffectObjectKey).Append(',')
                .Append(effect.Diffuse).Append(',')
                .Append(effect.Emissive).Append(',')
                .Append(effect.Alpha.ToString("F4", CultureInfo.InvariantCulture)).Append(',')
                .Append(effect.Visible);
        }

        foreach (M2RuntimeGoldenBatch batch in batches)
        {
            builder.Append("|batch:")
                .Append(batch.BatchIndex.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(batch.Family).Append(',')
                .Append(batch.Handler).Append(',')
                .Append(batch.Direct).Append(',')
                .Append(batch.EntryCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(batch.EffectKey).Append(',')
                .Append(batch.VertexCount.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(batch.IndexCount.ToString(CultureInfo.InvariantCulture));
        }

        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string FormatVector(Vector3 value)
    {
        return string.Create(CultureInfo.InvariantCulture, $"{Round(value.X):F4},{Round(value.Y):F4},{Round(value.Z):F4}");
    }

    private static float Round(float value)
    {
        return MathF.Round(value, 4, MidpointRounding.AwayFromZero);
    }
}
