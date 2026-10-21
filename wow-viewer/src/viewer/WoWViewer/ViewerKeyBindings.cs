using ImGuiNET;

namespace WoWViewer;

internal enum ViewerKeyContext
{
    Global,
    Quick,
    Inspect,
    Scene,
    Utilities,
    Audio,
    Experimental,
    Capture,
    Archaeology,
    Editor,
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
        new("ui.toggle_left_sidebar", "Global", "[", "Show or hide the left sidebar.", ViewerKeyContext.Global, true),
        new("ui.toggle_right_sidebar", "Global", "]", "Show or hide the right sidebar.", ViewerKeyContext.Global, true),
        new("ui.toggle_inspector", "Global", "I", "Show the right sidebar on its Inspector page.", ViewerKeyContext.Global, true),
        new("ui.focus_pm4", "Global", "P", "Show the right sidebar on its PM4 page.", ViewerKeyContext.Global, true),
        new("ui.toggle_minimap", "Global", "M", "Show or hide the minimap instrument.", ViewerKeyContext.Global, true),
        new("ui.fullscreen_minimap", "Global", "F", "Open or close the fullscreen minimap when terrain is loaded.", ViewerKeyContext.Global, true),
        new("ui.keyboard_shortcuts", "Global", "F1", "Open the keyboard shortcut list.", ViewerKeyContext.Global, true),
        new("ui.close_popup", "Global", "Esc", "Close the topmost popup.", ViewerKeyContext.Global, true),
        new("camera.free_fly", "Global", "WASD / Q E", "Move the free-fly camera; hold Shift for a speed boost.", ViewerKeyContext.Global, true),
        new("model.animation", "Inspect", "Left / Right / Space", "Step or play the loaded model animation.", ViewerKeyContext.Inspect),
        new("world.navigation", "Scene", "WASD / Q E", "Navigate the loaded world with the camera.", ViewerKeyContext.Scene),
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

    private ViewerKeyContext GetActiveKeyContext()
    {
        if (_uiManager.IsCaptureVisible) return ViewerKeyContext.Capture;
        if (_uiManager.IsAudioVisible) return ViewerKeyContext.Audio;
        return _activeTopTab switch
        {
            Workbench.WorkbenchTab.Editor => ViewerKeyContext.Editor,
            Workbench.WorkbenchTab.Archaeology => ViewerKeyContext.Archaeology,
            _ => ViewerKeyContext.Inspect,
        };
    }
}
