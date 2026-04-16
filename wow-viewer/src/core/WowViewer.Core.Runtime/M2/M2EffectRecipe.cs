using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public enum M2DiffuseEffectFamily
{
    None = 0,
    T1 = 1,
    T1T2 = 2,
    T1T2T3 = 3,
    T1T2T3T4 = 4,
    Projected = 5,
}

public enum M2CombinerEffectFamily
{
    Opaque = 0,
    AlphaKey = 1,
    Decal = 2,
    Add = 3,
    Mod = 4,
    Mod2X = 5,
    Fade = 6,
    Unknown = 7,
}

public sealed class M2EffectRecipe
{
    public M2EffectRecipe(
        M2DiffuseEffectFamily diffuseFamily,
        M2CombinerEffectFamily combinerFamily,
        bool isProjected,
        bool usesColorAnimation,
        bool usesTransparencyAnimation,
        bool usesTextureTransformAnimation,
        bool suppressCombinedTransparency,
        bool isHeuristic)
    {
        DiffuseFamily = diffuseFamily;
        CombinerFamily = combinerFamily;
        IsProjected = isProjected;
        UsesColorAnimation = usesColorAnimation;
        UsesTransparencyAnimation = usesTransparencyAnimation;
        UsesTextureTransformAnimation = usesTextureTransformAnimation;
        SuppressCombinedTransparency = suppressCombinedTransparency;
        IsHeuristic = isHeuristic;
    }

    public M2DiffuseEffectFamily DiffuseFamily { get; }

    public M2CombinerEffectFamily CombinerFamily { get; }

    public bool IsProjected { get; }

    public bool UsesColorAnimation { get; }

    public bool UsesTransparencyAnimation { get; }

    public bool UsesTextureTransformAnimation { get; }

    public bool SuppressCombinedTransparency { get; }

    public bool IsHeuristic { get; }

    public bool IsAnimated => UsesColorAnimation || UsesTransparencyAnimation || UsesTextureTransformAnimation;

    public string DiffuseFamilyName => DiffuseFamily switch
    {
        M2DiffuseEffectFamily.None => "Diffuse_None",
        M2DiffuseEffectFamily.T1 => "Diffuse_T1",
        M2DiffuseEffectFamily.T1T2 => "Diffuse_T1_T2",
        M2DiffuseEffectFamily.T1T2T3 => "Diffuse_T1_T2_T3",
        M2DiffuseEffectFamily.T1T2T3T4 => "Diffuse_T1_T2_T3_T4",
        M2DiffuseEffectFamily.Projected => "Diffuse_Projected",
        _ => "Diffuse_Unknown",
    };

    public string CombinerFamilyName => CombinerFamily switch
    {
        M2CombinerEffectFamily.Opaque => "Combiners_Opaque",
        M2CombinerEffectFamily.AlphaKey => "Combiners_AlphaKey",
        M2CombinerEffectFamily.Decal => "Combiners_Decal",
        M2CombinerEffectFamily.Add => "Combiners_Add",
        M2CombinerEffectFamily.Mod => "Combiners_Mod",
        M2CombinerEffectFamily.Mod2X => "Combiners_Mod2x",
        M2CombinerEffectFamily.Fade => "Combiners_Fade",
        _ => "Combiners_Unknown",
    };

    public string RecipeKey => $"{DiffuseFamilyName}:{CombinerFamilyName}";

    public string NativeEffectFamilyKey => $"{DiffuseFamilyName}{CombinerFamilyName}";
}

public sealed class M2ResolvedEffect
{
    public M2ResolvedEffect(
        string recipeKey,
        string nativeEffectFamilyKey,
        string effectObjectKey,
        M2BlendMode blendMode,
        bool depthWrite,
        bool alphaTest,
        bool isTransparent,
        bool isAdditive,
        bool receivesLighting,
        bool isTwoSided,
        bool isProjected,
        bool isHeuristic,
        int stateBucket)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(recipeKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(nativeEffectFamilyKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(effectObjectKey);

        RecipeKey = recipeKey;
        NativeEffectFamilyKey = nativeEffectFamilyKey;
        EffectObjectKey = effectObjectKey;
        BlendMode = blendMode;
        DepthWrite = depthWrite;
        AlphaTest = alphaTest;
        IsTransparent = isTransparent;
        IsAdditive = isAdditive;
        ReceivesLighting = receivesLighting;
        IsTwoSided = isTwoSided;
        IsProjected = isProjected;
        IsHeuristic = isHeuristic;
        StateBucket = stateBucket;
    }

    public string RecipeKey { get; }

    public string NativeEffectFamilyKey { get; }

    public string EffectObjectKey { get; }

    public M2BlendMode BlendMode { get; }

    public bool DepthWrite { get; }

    public bool AlphaTest { get; }

    public bool IsTransparent { get; }

    public bool IsAdditive { get; }

    public bool ReceivesLighting { get; }

    public bool IsTwoSided { get; }

    public bool IsProjected { get; }

    public bool IsHeuristic { get; }

    public int StateBucket { get; }
}

public static class M2EffectRegistry
{
    public static M2ResolvedEffect Resolve(M2StaticRenderMaterial material)
    {
        ArgumentNullException.ThrowIfNull(material);

        M2EffectRecipe recipe = material.EffectRecipe;
        bool isAdditive = material.BlendMode is M2BlendMode.NoAlphaAdd or M2BlendMode.Add or M2BlendMode.BlendAdd;
        bool alphaTest = material.BlendMode == M2BlendMode.AlphaKey;
        bool depthWrite = material.BlendMode is M2BlendMode.Opaque or M2BlendMode.AlphaKey;
        bool receivesLighting = !material.IsUnshaded;
        string effectPrefix = recipe.IsProjected ? "Model2Displ_" : "Model2_";
        string nativeEffectFamilyKey = recipe.NativeEffectFamilyKey;
        int stateBucket = BuildStateBucket(material, depthWrite, alphaTest, receivesLighting, isAdditive);

        return new M2ResolvedEffect(
            recipe.RecipeKey,
            nativeEffectFamilyKey,
            effectPrefix + nativeEffectFamilyKey,
            material.BlendMode,
            depthWrite,
            alphaTest,
            material.IsTransparent,
            isAdditive,
            receivesLighting,
            material.IsTwoSided,
            recipe.IsProjected,
            recipe.IsHeuristic,
            stateBucket);
    }

    private static int BuildStateBucket(
        M2StaticRenderMaterial material,
        bool depthWrite,
        bool alphaTest,
        bool receivesLighting,
        bool isAdditive)
    {
        int bucket = (int)material.BlendMode & 0xF;
        if (depthWrite)
            bucket |= 1 << 4;
        if (alphaTest)
            bucket |= 1 << 5;
        if (material.IsTwoSided)
            bucket |= 1 << 6;
        if (!receivesLighting)
            bucket |= 1 << 7;
        if (isAdditive)
            bucket |= 1 << 8;
        if (material.EffectRecipe.IsProjected)
            bucket |= 1 << 9;

        return bucket;
    }
}
