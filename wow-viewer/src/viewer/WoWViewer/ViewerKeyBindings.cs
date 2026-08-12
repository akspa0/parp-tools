using ImGuiNET;

namespace WoWViewer;

internal enum ViewerKeyContext
{
    Global,
    Model,
    World,
    Tools,
    Utilities,
    Capture,
}

internal sealed record ViewerKeyBinding(
    string Id,
    string Category,
    string Gesture,
    string Description,
    ViewerKeyContext Context,
    bool IsGlobal = false);

internal static class ViewerKeyBindingCatalog
{
    public static IReadOnlyList<ViewerKeyBinding> All { get; } =
    [
        new("ui.toggle_chrome", "Global", "Tab", "Show or hide the viewer chrome.", ViewerKeyContext.Global, true),
        new("ui.toggle_inspector", "Global", "I", "Show or hide the right inspector/workbench.", ViewerKeyContext.Global, true),
        new("ui.focus_pm4", "Global", "P", "Focus PM4 tools when available.", ViewerKeyContext.Global, true),
        new("ui.toggle_minimap", "Global", "M", "Toggle the fullscreen minimap when terrain is loaded.", ViewerKeyContext.Global, true),
        new("camera.free_fly", "Global", "WASD / Q E", "Move the free-fly camera; hold Shift for a speed boost.", ViewerKeyContext.Global, true),
        new("model.animation", "Model", "Left / Right / Space", "Step or play the loaded model animation.", ViewerKeyContext.Model),
        new("world.navigation", "World", "WASD / Q E", "Navigate the loaded world with the camera.", ViewerKeyContext.World),
        new("capture.add_key", "Capture", "K", "Add a camera-path key at the current playhead.", ViewerKeyContext.Capture),
        new("capture.update_key", "Capture", "U", "Update the selected camera-path key.", ViewerKeyContext.Capture),
        new("capture.delete_key", "Capture", "Delete", "Delete the selected camera-path key.", ViewerKeyContext.Capture),
        new("capture.play_pause", "Capture", "Space", "Play or pause the camera path.", ViewerKeyContext.Capture),
        new("capture.roll", "Capture", "Z / X", "Roll the camera left or right; hold Shift for a larger step.", ViewerKeyContext.Capture),
        new("capture.select_key", "Capture", "Left / Right", "Select the previous or next camera-path key.", ViewerKeyContext.Capture),
        new("capture.retime_key", "Capture", "Ctrl+Left / Ctrl+Right", "Retime the selected camera-path key.", ViewerKeyContext.Capture),
        new("capture.save", "Capture", "Ctrl+S", "Save the authored camera path as JSON.", ViewerKeyContext.Capture),
        new("capture.export", "Capture", "Ctrl+E", "Export the authored camera path as native M2.", ViewerKeyContext.Capture),
        new("capture.playhead", "Capture", "Ctrl+Up / Ctrl+Down", "Nudge the camera-path playhead.", ViewerKeyContext.Capture),
        new("capture.endpoints", "Capture", "Home / End", "Jump to the beginning or end of the camera path.", ViewerKeyContext.Capture),
    ];

    public static IEnumerable<ViewerKeyBinding> ForContext(ViewerKeyContext context)
    {
        return All.Where(binding => binding.IsGlobal || binding.Context == context);
    }
}

public partial class ViewerApp
{
    private bool _showKeyboardShortcutsWindow;

    private ViewerKeyContext GetActiveKeyContext()
    {
        if (!_useTabUi)
            return (_showCaptureAutomationWindow || _showCameraPathWindow) ? ViewerKeyContext.Capture : ViewerKeyContext.Global;

        return _activeTopTab switch
        {
            Workbench.WorkbenchTab.Model => ViewerKeyContext.Model,
            Workbench.WorkbenchTab.World => ViewerKeyContext.World,
            Workbench.WorkbenchTab.Tools when (Workbench.ToolsBottomTab)_activeBottomTabIndex == Workbench.ToolsBottomTab.Utilities
                && (Workbench.UtilitiesBottomTab)_activeUtilitiesTabIndex == Workbench.UtilitiesBottomTab.Capture => ViewerKeyContext.Capture,
            Workbench.WorkbenchTab.Tools when (Workbench.ToolsBottomTab)_activeBottomTabIndex == Workbench.ToolsBottomTab.Utilities => ViewerKeyContext.Utilities,
            Workbench.WorkbenchTab.Tools => ViewerKeyContext.Tools,
            _ => ViewerKeyContext.Global,
        };
    }

    private bool IsCaptureKeyboardContextActive()
    {
        return GetActiveKeyContext() == ViewerKeyContext.Capture;
    }

    private void DrawKeyboardShortcutsWindow()
    {
        if (!_showKeyboardShortcutsWindow)
            return;

        ImGui.SetNextWindowSize(new System.Numerics.Vector2(560f, 420f), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("Keyboard Shortcuts", ref _showKeyboardShortcutsWindow, ImGuiWindowFlags.NoCollapse))
        {
            ViewerKeyContext activeContext = GetActiveKeyContext();
            ImGui.TextUnformatted("Keyboard Shortcuts");
            ImGui.TextDisabled($"Active page: {activeContext}");
            ImGui.Separator();

            if (ImGui.BeginChild("##KeyboardShortcutsBody", new System.Numerics.Vector2(0f, 0f), false,
                ImGuiWindowFlags.None))
            {
                DrawKeyboardBindingGroup("Global", ViewerKeyBindingCatalog.All.Where(binding => binding.IsGlobal));
                foreach (ViewerKeyContext context in new[]
                {
                    ViewerKeyContext.Model,
                    ViewerKeyContext.World,
                    ViewerKeyContext.Tools,
                    ViewerKeyContext.Utilities,
                    ViewerKeyContext.Capture,
                })
                {
                    ImGui.Spacing();
                    string title = context == activeContext ? $"{context} (active)" : context.ToString();
                    DrawKeyboardBindingGroup(title, ViewerKeyBindingCatalog.All.Where(binding => !binding.IsGlobal && binding.Context == context));
                }
            }
            ImGui.EndChild();
        }
        ImGui.End();
    }

    private static void DrawKeyboardBindingGroup(string title, IEnumerable<ViewerKeyBinding> bindings)
    {
        ImGui.TextColored(new System.Numerics.Vector4(0.95f, 0.78f, 0.28f, 1f), title);
        bool any = false;
        foreach (ViewerKeyBinding binding in bindings)
        {
            any = true;
            ImGui.TextDisabled(binding.Gesture);
            ImGui.SameLine(175f);
            ImGui.TextWrapped(binding.Description);
        }

        if (!any)
            ImGui.TextDisabled("No page-specific shortcuts are assigned here.");
    }
}
