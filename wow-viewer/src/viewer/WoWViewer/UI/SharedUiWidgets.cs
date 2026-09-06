using System.Numerics;
using ImGuiNET;

namespace WoWViewer.UI;

/// <summary>
/// Small, presentation-only widgets shared by the workbench destinations.
///
/// These helpers deliberately do not own viewer state or actions. Callers
/// still invoke the same authoritative action methods used by their full
/// panels, which keeps Quick and legacy adapters as mirrors rather than new
/// implementations.
/// </summary>
public static class SharedUiWidgets
{
    private static readonly Vector4 ImportantColor = new(1f, 0.85f, 0.4f, 1f);
    private static readonly Vector4 StatusColor = new(0.62f, 0.82f, 0.92f, 1f);

    /// <summary>Draws a consistent collapsible section heading and optional help affordance.</summary>
    public static bool DrawSectionHeader(
        string title,
        string? help = null,
        bool defaultOpen = true,
        string? id = null)
    {
        string suffix = string.IsNullOrWhiteSpace(id) ? title : id;
        bool open = ImGui.CollapsingHeader(
            $"{title}##SharedSection_{MakeId(suffix)}",
            defaultOpen ? ImGuiTreeNodeFlags.DefaultOpen : ImGuiTreeNodeFlags.None);

        if (!string.IsNullOrWhiteSpace(help))
        {
            // CollapsingHeader consumes the full content width. Only place
            // the marker on that row when there is room; otherwise put it on
            // the following row so a compact sidebar never clips it.
            if (ImGui.GetContentRegionAvail().X >= ImGui.GetFrameHeight() + 4f)
                HelpMarker(help, suffix);
            else
            {
                ImGui.NewLine();
                HelpMarker(help, suffix);
            }
        }

        return open;
    }

    /// <summary>Compatibility-friendly short name for <see cref="DrawSectionHeader"/>.</summary>
    public static bool SectionHeader(
        string title,
        string? help = null,
        bool defaultOpen = true,
        string? id = null) => DrawSectionHeader(title, help, defaultOpen, id);

    /// <summary>
    /// Draws a compact [ ? ] button. Clicking it opens a small popup while
    /// hovering retains the fast tooltip path for dense sidebars.
    /// </summary>
    public static void HelpMarker(string help, string id = "help")
    {
        string popupId = $"SharedHelp_{MakeId(id)}";
        ImGui.SameLine(0f, 4f);
        if (ImGui.SmallButton($"[?]##{popupId}_Button"))
            ImGui.OpenPopup(popupId);

        if (ImGui.IsItemHovered())
            ImGui.SetTooltip(help);

        if (ImGui.BeginPopup(popupId))
        {
            ImGui.PushTextWrapPos(ImGui.GetFontSize() * 30f);
            ImGui.TextWrapped(help);
            ImGui.PopTextWrapPos();
            ImGui.EndPopup();
        }
    }

    /// <summary>Draws an enabled/disabled action button with a shared tooltip path.</summary>
    public static bool ActionButton(
        string label,
        bool enabled = true,
        string? help = null,
        bool small = false)
    {
        if (!enabled)
            ImGui.BeginDisabled();

        bool clicked = small ? ImGui.SmallButton(label) : ImGui.Button(label);

        if (!string.IsNullOrWhiteSpace(help) && ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
            ImGui.SetTooltip(help);

        if (!enabled)
            ImGui.EndDisabled();

        return clicked;
    }

    /// <summary>Shared action-group button contract used by compact workbench destinations.</summary>
    public static bool DrawActionGroup(
        string label,
        bool enabled = true,
        string? help = null,
        bool small = false) => ActionButton(label, enabled, help, small);

    /// <summary>Draws a titled action group and invokes its body while the section is open.</summary>
    public static void DrawActionGroup(
        string title,
        Action drawBody,
        string? help = null,
        bool defaultOpen = true,
        string? id = null)
    {
        if (DrawSectionHeader(title, help, defaultOpen, id))
            drawBody();
    }

    /// <summary>Draws a compact label/value readout used by Inspector and utility pages.</summary>
    public static void StatusReadout(
        string label,
        string value,
        bool important = false,
        string? help = null)
    {
        DrawLabeledValue(label, value, important, help);
    }

    /// <summary>Shared compact label/value contract; long values wrap inside the sidebar.</summary>
    public static void DrawLabeledValue(
        string label,
        string value,
        bool important = false,
        string? help = null)
    {
        string prefix = $"{label}: ";
        if (value.Length > 96)
        {
            if (important)
                ImGui.TextColored(ImportantColor, prefix);
            else
                ImGui.TextDisabled(prefix);
            ImGui.SameLine(0f, 0f);
            ImGui.TextWrapped(value);
        }
        else if (important)
            ImGui.TextColored(ImportantColor, prefix + value);
        else
            ImGui.TextDisabled(prefix + value);

        if (!string.IsNullOrWhiteSpace(help))
            HelpMarker(help, $"{label}_{value}");
    }

    /// <summary>Draws a compact state badge for utility and runtime status lines.</summary>
    public static void DrawStatusBadge(string label, string value, bool active = true)
    {
        Vector4 color = active
            ? new Vector4(0.24f, 0.58f, 0.32f, 1f)
            : new Vector4(0.30f, 0.30f, 0.34f, 1f);
        ImGui.PushStyleColor(ImGuiCol.Text, color);
        ImGui.TextUnformatted($"[{label}: {value}]");
        ImGui.PopStyleColor();
    }

    /// <summary>Shared float slider wrapper with optional help marker.</summary>
    public static bool DrawCompactSlider(
        string label,
        ref float value,
        float min,
        float max,
        string format = "%.2f",
        string? help = null)
    {
        bool changed = ImGui.SliderFloat(label, ref value, min, max, format);
        if (!string.IsNullOrWhiteSpace(help))
            HelpMarker(help, label);
        return changed;
    }

    /// <summary>Shared integer slider wrapper with optional help marker.</summary>
    public static bool DrawCompactSlider(
        string label,
        ref int value,
        int min,
        int max,
        string? help = null)
    {
        bool changed = ImGui.SliderInt(label, ref value, min, max);
        if (!string.IsNullOrWhiteSpace(help))
            HelpMarker(help, label);
        return changed;
    }

    /// <summary>Draws a one-line status message without taking a full section.</summary>
    public static void CompactStatus(string text, bool emphasized = false)
    {
        if (emphasized)
            ImGui.TextColored(StatusColor, text);
        else
            ImGui.TextDisabled(text);
    }

    /// <summary>Draws a consistent separator between dense workbench groups.</summary>
    public static void Divider() => ImGui.Separator();

    private static string MakeId(string value)
    {
        Span<char> buffer = stackalloc char[value.Length];
        int length = 0;
        foreach (char c in value)
        {
            if (char.IsLetterOrDigit(c) || c == '_')
                buffer[length++] = c;
            else
                buffer[length++] = '_';
        }

        return length == 0 ? "Widget" : new string(buffer[..length]);
    }
}
