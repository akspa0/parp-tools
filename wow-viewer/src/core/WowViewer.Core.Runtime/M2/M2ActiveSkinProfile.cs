using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2ActiveSkinProfile
{
    public M2ActiveSkinProfile(
        M2ModelDocument model,
        M2SkinProfileSelection selection,
        M2SkinDocument skin,
        bool usesCompatibilityFallback)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(selection);
        ArgumentNullException.ThrowIfNull(skin);

        Model = model;
        Selection = selection;
        Skin = skin;
        UsesCompatibilityFallback = usesCompatibilityFallback;
    }

    public M2ModelDocument Model { get; }

    public M2SkinProfileSelection Selection { get; }

    public M2SkinDocument Skin { get; }

    public bool UsesCompatibilityFallback { get; }

    public int ActiveSubmeshCount => Skin.SubmeshCount;

    public int ActiveBatchCount => Skin.BatchCount;

    public int ActiveVertexLookupCount => Skin.VertexLookupCount;

    public int ActiveTriangleIndexCount => Skin.TriangleIndexCount;
}