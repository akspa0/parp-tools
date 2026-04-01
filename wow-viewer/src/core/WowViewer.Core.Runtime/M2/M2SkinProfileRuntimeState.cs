using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2SkinProfileRuntimeState
{
    public M2SkinProfileRuntimeState(
        M2ModelDocument model,
        M2SkinProfileSelection selection,
        M2SkinProfileStage stage,
        M2SkinDocument? loadedSkin,
        M2ActiveSkinProfile? activeSkinProfile)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(selection);

        Model = model;
        Selection = selection;
        Stage = stage;
        LoadedSkin = loadedSkin;
        ActiveSkinProfile = activeSkinProfile;
    }

    public M2ModelDocument Model { get; }

    public M2SkinProfileSelection Selection { get; }

    public M2SkinProfileStage Stage { get; }

    public M2SkinDocument? LoadedSkin { get; }

    public M2ActiveSkinProfile? ActiveSkinProfile { get; }
}