using ImGuiNET;

namespace WoWViewer;

/// <summary>
/// View menu (Spec 286 U1): the one home for camera, interface visibility, text size, the existing
/// visual-setting windows and the three named resets. Every entry delegates to the owner of the
/// setting; the menu keeps no copy of any value.
/// </summary>
internal sealed partial class MainMenuBarService
{
    private void DrawViewMenuContent()
    {
        ViewPreferencesService view = _shellLayout.ViewPreferences;

        if (ImGui.MenuItem("Reset Camera")) ResetCamera();
        if (ImGui.MenuItem("Hide UI Chrome", "Tab", _hideUiChrome)) _hideUiChrome = !_hideUiChrome;
        ImGui.MenuItem("Left Sidebar", "", ref _showLeftSidebar);
        ImGui.MenuItem("Right Sidebar", "I", ref _showRightSidebar);

        ImGui.Separator();

        // Both open the window that already owns the settings: Options holds Display (render
        // quality, lighting, theme, minimap), Overlays, Layers, Grids, Objects, Audio and
        // Simulation; Workspace holds camera, fog and the client source.
        if (ImGui.MenuItem("Visual Options...")) _shellLayout.OpenOptions();
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Open Options: render quality, lighting, theme, minimap, overlays, layers, grids and objects.");
        if (ImGui.MenuItem("Camera and Fog...")) _shellLayout.OpenWorkspaceWindow();
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Open Workspace: camera, fog and client source controls.");

        DrawViewTextSizeMenu(view);

        ImGui.Separator();

        // Three distinct recoveries. None of them unloads the client or the current map.
        if (ImGui.MenuItem("Reset Text Size Only")) view.Reset(ViewResetKind.TextScaleOnly);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Return the text size to 100%. Panel layout is unchanged; high-resolution displays keep their automatic sizing.");
        if (ImGui.MenuItem("Reset Layout Only")) view.Reset(ViewResetKind.LayoutOnly);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Restore the default panel layout and close tool windows. Text size is unchanged.");
        if (ImGui.MenuItem("Reset Layout and Text Size")) view.Reset(ViewResetKind.LayoutAndTextScale);
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Restore the default panel layout and return the text size to 100%.");
    }

    private void DrawViewTextSizeMenu(ViewPreferencesService view)
    {
        if (!ImGui.BeginMenu("Text Size")) return;

        float scale = view.UserScale;
        ImGui.SetNextItemWidth(160f * _shellLayout.Layout.Scale);
        if (ImGui.SliderFloat("Text Scale", ref scale, ViewPreferencePolicy.MinUserScale, ViewPreferencePolicy.MaxUserScale, "%.2fx"))
            view.SetUserScale(scale);
        // Apply on every drag step, write the settings file once when the drag ends.
        if (ImGui.IsItemDeactivatedAfterEdit()) view.Commit();

        // Preset labels are computed, so the current size is shown as the checked entry.
        foreach (float preset in ViewPreferencePolicy.Presets)
        {
            bool current = ViewPreferencePolicy.IsSameScale(view.UserScale, preset);
            if (ImGui.MenuItem(ViewPreferencePolicy.FormatPercent(preset), "", current))
            {
                view.SetUserScale(preset);
                view.Commit();
            }
        }

        ImGui.EndMenu();
    }
}
