using ImGuiNET;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// MainMenuBarService: members moved from ViewerKeyBindings.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class MainMenuBarService
{
    private bool _showKeyboardShortcutsWindow;

    public void DrawKeyboardShortcutsContent()
    {
            ViewerKeyContext activeContext = GetActiveKeyContext();
            SharedUiWidgets.SectionTitle("Keyboard Shortcuts");
            ImGui.TextDisabled($"Active page: {activeContext}");
            ImGui.Separator();

            if (ImGui.BeginChild("##KeyboardShortcutsBody", new System.Numerics.Vector2(0f,
                    Math.Max(80f, ImGui.GetContentRegionAvail().Y - ImGui.GetFrameHeightWithSpacing())), false,
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

    private static void DrawKeyboardBindingGroup(string title, IEnumerable<ViewerKeyBinding> bindings)
    {
        SharedUiWidgets.SectionTitle(title);
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
