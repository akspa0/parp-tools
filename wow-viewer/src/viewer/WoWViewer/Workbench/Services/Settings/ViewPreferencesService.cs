using WoWViewer.Logging;

namespace WoWViewer;

/// <summary>
/// The single owner of the user's interface text size and of the view reset actions
/// (Spec 286 U1). The View menu, the Settings window and the settings loader all go through it,
/// so there is exactly one stored value (<see cref="IViewerAppHost.UiFontScale"/>) and one
/// normalisation rule. It never touches ImGui: <see cref="ViewerUiManager"/> composes the value
/// with the display scale each frame, so a change here reaches every open window on the next frame.
/// </summary>
internal sealed class ViewPreferencesService
{
    private readonly IViewerAppHost _host;

    internal ViewPreferencesService(IViewerAppHost host) => _host = host;

    /// <summary>The user's text-size adjustment, always inside the supported range.</summary>
    internal float UserScale => _host.UiFontScale;

    /// <summary>
    /// Applies a user adjustment immediately without writing settings. Returns whether the stored
    /// value changed. Call <see cref="Commit"/> when the adjustment is finished (for example when
    /// a slider drag ends) so the file is not rewritten on every frame of a drag.
    /// </summary>
    internal bool SetUserScale(float value)
    {
        float normalized = ViewPreferencePolicy.NormalizeUserScale(value);
        if (normalized == _host.UiFontScale) return false;
        _host.UiFontScale = normalized;
        return true;
    }

    /// <summary>Persists the current preference through the existing settings store.</summary>
    internal void Commit() => _host.Settings.SaveViewerSettings();

    /// <summary>
    /// Normalises a value read from the settings file. An out-of-range legacy value is corrected
    /// here, once; the corrected value is what every later save writes back.
    /// </summary>
    internal float LoadPersisted(float persisted)
    {
        float normalized = ViewPreferencePolicy.NormalizePersisted(persisted, out bool corrected);
        if (corrected)
            ViewerLog.Trace($"[ViewPreferences] Saved text scale {persisted} is outside {ViewPreferencePolicy.MinUserScale}-{ViewPreferencePolicy.MaxUserScale}; using {normalized}.");
        return normalized;
    }

    /// <summary>
    /// Runs a named reset. Layout and scale are independent; none of them unload the client or
    /// the scene, and scale reset restores the user factor (1.0) rather than the display scale.
    /// </summary>
    internal void Reset(ViewResetKind kind)
    {
        ViewResetEffect effect = ViewPreferencePolicy.Resolve(kind);
        if (effect.ResetLayout) _host.UiManager.ResetLayout();
        if (effect.ResetUserScale)
        {
            SetUserScale(ViewPreferencePolicy.DefaultUserScale);
            Commit();
        }
    }
}
