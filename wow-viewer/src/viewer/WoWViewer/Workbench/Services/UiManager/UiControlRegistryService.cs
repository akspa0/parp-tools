using System.Reflection;
using ImGuiNET;
using WoWViewer.UI;

namespace WoWViewer;

/// <summary>
/// The control registry (Spec 280, ui-manager.md §7).
///
/// The registry is embedded at build time from <c>wow-viewer/eng/ui/ui-controls.json</c>.
/// The same file generates the committed <c>docs/WoWViewer/UI-REFERENCE.md</c>, and the
/// build check fails when a visible control has no entry or the document drifts.
/// </summary>
internal sealed class UiControlRegistry
{
    private const string RegistryResourceName = "WoWViewer.UI.ui-controls.json";

    private readonly List<UiControlDoc> _controls;

    internal UiControlRegistry()
    {
        _controls = Load();
    }

    internal IReadOnlyList<UiControlDoc> Controls => _controls;

    private static List<UiControlDoc> Load()
    {
        Assembly assembly = typeof(UiControlRegistry).Assembly;
        using Stream? stream = assembly.GetManifestResourceStream(RegistryResourceName);
        if (stream == null)
        {
            // The registry is documentation, not behaviour: a missing resource must not
            // take the viewer down, but the missing help is visible in the UI.
            return new List<UiControlDoc>
            {
                new("registry-missing", "UI reference unavailable", "Help", null,
                    "The UI control registry was not embedded in this build."),
            };
        }

        using StreamReader reader = new(stream);
        return UiReferenceDocument.ParseRegistry(reader.ReadToEnd()).ToList();
    }

    /// <summary>Searchable list of every documented control.</summary>
    internal void DrawReferenceContent()
    {
        ImGui.TextDisabled($"{_controls.Count} documented controls.");
        ImGui.Spacing();

        string filter = string.Empty;
        ImGui.InputTextWithHint("##uiRefFilter", "Filter controls", ref filter, 256);
        ImGui.Separator();

        if (!ImGui.BeginChild("##uiRefList", new System.Numerics.Vector2(0f, 0f), true))
        {
            ImGui.EndChild();
            return;
        }

        foreach (UiControlDoc control in _controls)
        {
            if (!control.Visible) continue;
            if (!string.IsNullOrWhiteSpace(filter)
                && !control.Label.Contains(filter, StringComparison.OrdinalIgnoreCase)
                && !control.Help.Contains(filter, StringComparison.OrdinalIgnoreCase)
                && !control.Region.Contains(filter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            ImGui.PushTextWrapPos(0f);
            ImGui.TextUnformatted($"{control.Label}  [{control.Region}]");
            ImGui.TextDisabled(control.Help);
            if (!string.IsNullOrWhiteSpace(control.Shortcut))
                ImGui.TextDisabled($"Shortcut: {control.Shortcut}");
            ImGui.PopTextWrapPos();
            ImGui.Separator();
        }

        ImGui.EndChild();
    }
}
