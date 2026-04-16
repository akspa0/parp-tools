using System.Numerics;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2RenderConsumerFrameState
{
    public M2RenderConsumerFrameState(
        M2StaticRenderModel renderModel,
        M2AnimatedRenderState animatedState,
        IReadOnlyList<M2RenderConsumerPassState> passes,
        Vector3 modelAmbient,
        Vector3 modelDiffuse)
    {
        ArgumentNullException.ThrowIfNull(renderModel);
        ArgumentNullException.ThrowIfNull(animatedState);
        ArgumentNullException.ThrowIfNull(passes);

        RenderModel = renderModel;
        AnimatedState = animatedState;
        Passes = passes;
        ModelAmbient = modelAmbient;
        ModelDiffuse = modelDiffuse;
    }

    public M2StaticRenderModel RenderModel { get; }

    public M2AnimatedRenderState AnimatedState { get; }

    public IReadOnlyList<M2RenderConsumerPassState> Passes { get; }

    public Vector3 ModelAmbient { get; }

    public Vector3 ModelDiffuse { get; }

    public int VisiblePassCount => Passes.Count(static pass => pass.Visible);
}

public sealed class M2RenderConsumerPassState
{
    public M2RenderConsumerPassState(
        M2StructuredRenderPass sourcePass,
        M2AnimatedRenderPassState animatedPass,
        Vector3 diffuseColor,
        Vector3 emissiveColor,
        float alpha,
        bool receivesLighting,
        bool visible,
        IReadOnlyList<M2RenderConsumerTextureState> textures)
    {
        ArgumentNullException.ThrowIfNull(sourcePass);
        ArgumentNullException.ThrowIfNull(animatedPass);
        ArgumentNullException.ThrowIfNull(textures);

        SourcePass = sourcePass;
        AnimatedPass = animatedPass;
        DiffuseColor = diffuseColor;
        EmissiveColor = emissiveColor;
        Alpha = alpha;
        ReceivesLighting = receivesLighting;
        Visible = visible;
        Textures = textures;
        ResolvedEffect = M2EffectRegistry.Resolve(sourcePass.Material);
    }

    public M2StructuredRenderPass SourcePass { get; }

    public M2AnimatedRenderPassState AnimatedPass { get; }

    public Vector3 DiffuseColor { get; }

    public Vector3 EmissiveColor { get; }

    public float Alpha { get; }

    public bool ReceivesLighting { get; }

    public bool Visible { get; }

    public IReadOnlyList<M2RenderConsumerTextureState> Textures { get; }

    public string EffectKey => SourcePass.Material.EffectRecipe.RecipeKey;

    public M2ResolvedEffect ResolvedEffect { get; }
}

public sealed class M2RenderConsumerTextureState
{
    public M2RenderConsumerTextureState(
        int stageIndex,
        string? texturePath,
        uint replaceableId,
        float alpha,
        Vector3 translation,
        Quaternion rotation,
        Vector3 scaling)
    {
        StageIndex = stageIndex;
        TexturePath = texturePath;
        ReplaceableId = replaceableId;
        Alpha = alpha;
        Translation = translation;
        Rotation = rotation;
        Scaling = scaling;
    }

    public int StageIndex { get; }

    public string? TexturePath { get; }

    public uint ReplaceableId { get; }

    public float Alpha { get; }

    public Vector3 Translation { get; }

    public Quaternion Rotation { get; }

    public Vector3 Scaling { get; }
}

public static class M2RenderConsumerFrameStateBuilder
{
    public static M2RenderConsumerFrameState Build(M2StaticRenderModel renderModel, M2AnimatedRenderState animatedState)
    {
        ArgumentNullException.ThrowIfNull(renderModel);
        ArgumentNullException.ThrowIfNull(animatedState);

        Dictionary<(int SectionIndex, int PassIndex), M2StructuredRenderPass> passesByKey = new();
        foreach (M2StructuredRenderSection section in renderModel.StructuredSections)
        {
            foreach (M2StructuredRenderPass pass in section.Passes)
                passesByKey[(section.SectionIndex, pass.PassIndex)] = pass;
        }

        Vector3 modelAmbient = Vector3.Zero;
        Vector3 modelDiffuse = Vector3.Zero;
        int visibleLightCount = 0;
        foreach (M2AnimatedLightState light in animatedState.Lights)
        {
            if (!light.Visible)
                continue;

            modelAmbient += light.AmbientColor * light.AmbientIntensity;
            modelDiffuse += light.DiffuseColor * light.DiffuseIntensity;
            visibleLightCount++;
        }

        if (visibleLightCount > 0)
        {
            modelAmbient /= visibleLightCount;
            modelDiffuse /= visibleLightCount;
        }

        List<M2RenderConsumerPassState> passes = new(animatedState.Passes.Count);
        foreach (M2AnimatedRenderPassState animatedPass in animatedState.Passes)
        {
            if (!passesByKey.TryGetValue((animatedPass.SectionIndex, animatedPass.PassIndex), out M2StructuredRenderPass? sourcePass))
                continue;

            M2StaticRenderMaterial material = sourcePass.Material;
            Vector3 diffuse = Clamp01(animatedPass.Color);
            bool receivesLighting = !material.IsUnshaded;
            Vector3 emissive = material.IsUnshaded ? diffuse : Vector3.Zero;
            float alpha = Math.Clamp(animatedPass.CombinedAlpha, 0.0f, 1.0f);
            bool visible = alpha > 0.001f;

            List<M2RenderConsumerTextureState> textureStates = new(animatedPass.TextureBindings.Count);
            foreach (M2AnimatedTextureBindingState animatedTexture in animatedPass.TextureBindings)
            {
                M2StaticRenderTextureBinding? staticTexture = material.TextureBindings.FirstOrDefault(binding => binding.StageIndex == animatedTexture.StageIndex);
                textureStates.Add(new M2RenderConsumerTextureState(
                    animatedTexture.StageIndex,
                    staticTexture?.TexturePath,
                    staticTexture?.ReplaceableId ?? 0u,
                    Math.Clamp(animatedTexture.TransparencyAlpha, 0.0f, 1.0f),
                    animatedTexture.Translation,
                    animatedTexture.Rotation,
                    animatedTexture.Scaling));
            }

            passes.Add(new M2RenderConsumerPassState(sourcePass, animatedPass, diffuse, emissive, alpha, receivesLighting, visible, textureStates));
        }

        return new M2RenderConsumerFrameState(renderModel, animatedState, passes, Clamp01(modelAmbient), Clamp01(modelDiffuse));
    }

    private static Vector3 Clamp01(Vector3 value)
    {
        return new Vector3(
            Math.Clamp(value.X, 0.0f, 1.0f),
            Math.Clamp(value.Y, 0.0f, 1.0f),
            Math.Clamp(value.Z, 0.0f, 1.0f));
    }
}
