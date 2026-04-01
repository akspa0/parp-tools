using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public static class M2SkinProfileRuntime
{
    public static M2SkinProfileRuntimeState Choose(M2ModelDocument model, int profileIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(model);

        M2SkinProfileSelection selection = new(profileIndex, model.Identity.BuildSkinPath(profileIndex));
        return new M2SkinProfileRuntimeState(model, selection, M2SkinProfileStage.Chosen, loadedSkin: null, activeSkinProfile: null);
    }

    public static M2SkinProfileRuntimeState Load(M2SkinProfileRuntimeState state, M2SkinDocument skin)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(skin);

        if (state.Stage != M2SkinProfileStage.Chosen)
            throw new InvalidOperationException($"Cannot load a skin profile from stage '{state.Stage}'. Expected '{M2SkinProfileStage.Chosen}'.");

        if (!M2ModelIdentity.PathsEqual(state.Selection.CompanionPath, skin.SourcePath))
        {
            throw new InvalidDataException(
                $"Loaded skin path '{skin.SourcePath}' does not match the exact selected companion '{state.Selection.CompanionPath}'.");
        }

        return new M2SkinProfileRuntimeState(state.Model, state.Selection, M2SkinProfileStage.Loaded, skin, activeSkinProfile: null);
    }

    public static M2SkinProfileRuntimeState Initialize(M2SkinProfileRuntimeState state)
    {
        ArgumentNullException.ThrowIfNull(state);

        if (state.Stage != M2SkinProfileStage.Loaded || state.LoadedSkin is null)
            throw new InvalidOperationException("Cannot initialize a skin profile before the exact numbered .skin companion is loaded.");

        M2ActiveSkinProfile activeSkinProfile = new(state.Model, state.Selection, state.LoadedSkin, usesCompatibilityFallback: false);
        return new M2SkinProfileRuntimeState(state.Model, state.Selection, M2SkinProfileStage.Initialized, state.LoadedSkin, activeSkinProfile);
    }
}