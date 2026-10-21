using System.Text.Json;

namespace WoWViewer.UI;

/// <summary>One documented viewer control (Spec 280, ui-manager.md §7).</summary>
/// <param name="Id">Stable identifier used by the generated reference.</param>
/// <param name="Label">The visible label of the control.</param>
/// <param name="Region">The UI region the control lives in.</param>
/// <param name="Shortcut">Keyboard shortcut, or <c>null</c> when the control has none.</param>
/// <param name="Help">One-line explanation of what the control does.</param>
/// <param name="Visible">False when the control is registered but hidden in the current build.</param>
public sealed record UiControlDoc(
    string Id,
    string Label,
    string Region,
    string? Shortcut,
    string Help,
    bool Visible = true);

/// <summary>
/// Reads the control registry and renders the generated UI reference.
///
/// This type is deliberately free of ImGui so the drift and coverage checks can
/// reuse the exact same rendering rules the committed document was produced with.
/// </summary>
public static class UiReferenceDocument
{
    /// <summary>Parses the registry JSON (the source of truth for the UI reference).</summary>
    public static IReadOnlyList<UiControlDoc> ParseRegistry(string json)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        if (!document.RootElement.TryGetProperty("controls", out JsonElement controls))
            throw new InvalidDataException("UI control registry has no 'controls' array.");

        List<UiControlDoc> entries = new();
        foreach (JsonElement control in controls.EnumerateArray())
        {
            string id = control.GetProperty("id").GetString() ?? string.Empty;
            string label = control.GetProperty("label").GetString() ?? string.Empty;
            string region = control.GetProperty("region").GetString() ?? string.Empty;
            string? shortcut = control.TryGetProperty("shortcut", out JsonElement shortcutElement)
                && shortcutElement.ValueKind == JsonValueKind.String
                    ? shortcutElement.GetString()
                    : null;
            string help = control.GetProperty("help").GetString() ?? string.Empty;
            bool visible = !control.TryGetProperty("visible", out JsonElement visibleElement)
                || visibleElement.GetBoolean();

            if (string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(label))
                throw new InvalidDataException("UI control registry entry is missing an id or label.");
            if (string.IsNullOrWhiteSpace(help))
                throw new InvalidDataException($"UI control '{id}' is missing help text.");

            entries.Add(new UiControlDoc(id, label, region, shortcut, help, visible));
        }

        return entries;
    }

}
