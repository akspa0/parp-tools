using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2RuntimeFrameResult
{
    public M2RuntimeFrameResult(
        M2AnimatedRenderState animatedState,
        M2BonePoseState bonePoseState,
        M2SkinnedRenderModel skinnedRenderModel,
        M2RenderConsumerFrameState consumerState,
        M2EffectRuntimeState effectRuntimeState,
        M2SceneSubmissionPlan submissionPlan,
        M2RenderFrame renderFrame,
        M2SoftwareVisualSnapshot visualSnapshot,
        M2RuntimeGoldenFrame goldenFrame)
    {
        ArgumentNullException.ThrowIfNull(animatedState);
        ArgumentNullException.ThrowIfNull(bonePoseState);
        ArgumentNullException.ThrowIfNull(skinnedRenderModel);
        ArgumentNullException.ThrowIfNull(consumerState);
        ArgumentNullException.ThrowIfNull(effectRuntimeState);
        ArgumentNullException.ThrowIfNull(submissionPlan);
        ArgumentNullException.ThrowIfNull(renderFrame);
        ArgumentNullException.ThrowIfNull(visualSnapshot);
        ArgumentNullException.ThrowIfNull(goldenFrame);

        AnimatedState = animatedState;
        BonePoseState = bonePoseState;
        SkinnedRenderModel = skinnedRenderModel;
        ConsumerState = consumerState;
        EffectRuntimeState = effectRuntimeState;
        SubmissionPlan = submissionPlan;
        RenderFrame = renderFrame;
        VisualSnapshot = visualSnapshot;
        GoldenFrame = goldenFrame;
    }

    public M2AnimatedRenderState AnimatedState { get; }

    public M2BonePoseState BonePoseState { get; }

    public M2SkinnedRenderModel SkinnedRenderModel { get; }

    public M2RenderConsumerFrameState ConsumerState { get; }

    public M2EffectRuntimeState EffectRuntimeState { get; }

    public M2SceneSubmissionPlan SubmissionPlan { get; }

    public M2RenderFrame RenderFrame { get; }

    public M2SoftwareVisualSnapshot VisualSnapshot { get; }

    public M2RuntimeGoldenFrame GoldenFrame { get; }
}

public static class M2RuntimeFramePipeline
{
    public const M2RuntimeOptions DefaultSubmissionOptions =
        M2RuntimeOptions.BatchDoodads
        | M2RuntimeOptions.BatchParticles
        | M2RuntimeOptions.ForceAdditiveParticleSort;

    public static M2RuntimeFrameResult Build(
        M2ModelDocument model,
        M2StaticRenderModel renderModel,
        int sequenceIndex,
        int timeMs,
        M2ExternalAnimationRuntimeState? externalAnimationState = null,
        M2RuntimeOptions submissionOptions = DefaultSubmissionOptions,
        int visualWidth = 256,
        int visualHeight = 256)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(renderModel);

        M2AnimatedRenderState animatedState = M2AnimatedRenderStateEvaluator.Evaluate(model, renderModel, sequenceIndex, timeMs, externalAnimationState);
        M2BonePoseState bonePoseState = M2BonePoseEvaluator.Evaluate(model, sequenceIndex, timeMs, externalAnimationState);
        M2SkinnedRenderModel skinnedRenderModel = M2SkinnedRenderModelBuilder.ApplyPose(renderModel, bonePoseState);
        M2RenderConsumerFrameState consumerState = M2RenderConsumerFrameStateBuilder.Build(renderModel, animatedState);
        M2EffectRuntimeState effectRuntimeState = M2ParticleRibbonRuntimeEvaluator.Evaluate(model, sequenceIndex, timeMs);
        M2SceneSubmissionEntry[] entries = M2SceneSubmissionEntryBuilder.BuildRenderEntries(model, renderModel, consumerState)
            .Concat(M2SceneSubmissionEntryBuilder.BuildParticleEntries(M2ParticleRibbonRuntimeEvaluator.BuildParticleSubmissionDescriptors(effectRuntimeState, model.Identity.CanonicalModelPath)))
            .Concat(M2SceneSubmissionEntryBuilder.BuildRibbonEntries(M2ParticleRibbonRuntimeEvaluator.BuildRibbonSubmissionDescriptors(effectRuntimeState, model.Identity.CanonicalModelPath)))
            .ToArray();
        M2SceneSubmissionPlan submissionPlan = M2SceneSubmissionCoordinator.BuildPlan(entries, submissionOptions);
        M2RenderFrame renderFrame = M2RenderFrameBuilder.Build(renderModel, skinnedRenderModel, consumerState, submissionPlan, timeMs);
        M2SoftwareVisualSnapshot visualSnapshot = M2SoftwareVisualSnapshotBuilder.Build(renderFrame, visualWidth, visualHeight);
        M2RuntimeGoldenFrame goldenFrame = M2RuntimeGoldenFrameBuilder.Build(model, animatedState, bonePoseState, skinnedRenderModel, consumerState, submissionPlan);

        return new M2RuntimeFrameResult(
            animatedState,
            bonePoseState,
            skinnedRenderModel,
            consumerState,
            effectRuntimeState,
            submissionPlan,
            renderFrame,
            visualSnapshot,
            goldenFrame);
    }
}