using ImGuiNET;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// MainMenuBarService: members moved from ViewerKeyBindings.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class MainMenuBarService
{
    private bool _showKeyboardShortcutsWindow;

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
                    ViewerKeyContext.Quick,
                    ViewerKeyContext.Inspect,
                    ViewerKeyContext.Scene,
                    ViewerKeyContext.Utilities,
                    ViewerKeyContext.Audio,
                    ViewerKeyContext.Experimental,
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
