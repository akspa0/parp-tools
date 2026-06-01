namespace WoWViewer.Rendering;

/// <summary>
/// World M2 material pass classification.
/// Defined in data-model.md as M2MaterialPassProfile.PassClass.
/// </summary>
public enum M2PassClass
{
    Opaque,
    Cutout,
    Blended,
}

/// <summary>
/// Captures world-pass semantics at material/layer level for parity checks.
/// Defined in data-model.md as M2MaterialPassProfile.
/// </summary>
public sealed record M2MaterialPassProfile(
    string ModelPath,
    int SectionIndex,
    int MaterialIndex,
    int LayerIndex,
    string BlendDeclaration,
    M2PassClass PassClass,
    bool DepthWrite,
    bool BlendEnabled,
    float? AlphaThreshold)
{
    public static M2MaterialPassProfile Create(string modelPath, int sectionIndex, int materialIndex, int layerIndex,
        string blendDeclaration, M2PassClass passClass, bool depthWrite, bool blendEnabled, float? alphaThreshold = null)
        => new(modelPath, sectionIndex, materialIndex, layerIndex, blendDeclaration, passClass, depthWrite, blendEnabled, alphaThreshold);
}