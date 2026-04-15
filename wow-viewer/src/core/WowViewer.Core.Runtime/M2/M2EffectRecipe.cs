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
}
