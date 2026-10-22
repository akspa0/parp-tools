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
///
/// Spec 280 P8: every section, body line, button row and empty state in the
/// manager-owned surfaces is drawn through this class, and the colours and
/// spacing come from <see cref="UiTokens"/> instead of local literals.
/// </summary>
public static class SharedUiWidgets
{
    private static readonly Vector4 ImportantColor = new(1f, 0.85f, 0.4f, 1f);
    private static readonly Vector4 StatusColor = new(0.62f, 0.82f, 0.92f, 1f);

    /// <summary>One button of a <see cref="DrawButtonRow"/>.</summary>
    public readonly record struct ButtonSpec(
        string Label,
        Action? Action,
        bool Enabled = true,
        string? Help = null,
        bool Small = false);

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

    /// <summary>Framed icon hit target with a full-name tooltip and selected/hover feedback.</summary>
    public static bool IconButton(string label, string id, Vector2 size, string help, bool selected = false)
    {
        Vector2 position = ImGui.GetCursorScreenPos();
        bool clicked = ImGui.InvisibleButton($"{label}##{id}", size);
        bool hovered = ImGui.IsItemHovered();
        Vector4 background = selected || ImGui.IsItemActive() ? UiTokens.Surface2 : UiTokens.Surface1;
        if (hovered) background = UiTokens.Surface2;
        var draw = ImGui.GetWindowDrawList();
        float scale = MathF.Max(1f, size.X / 34f);
        draw.AddRectFilled(position, position + size, ImGui.ColorConvertFloat4ToU32(background), 3f * scale);
        draw.AddRect(position, position + size,
            ImGui.ColorConvertFloat4ToU32(selected || hovered ? UiTokens.Brass : UiTokens.Border),
            3f * scale, ImDrawFlags.None, MathF.Max(1f, scale));
        if (hovered) ImGui.SetTooltip($"{label}\n{help}");
        return clicked;
    }

    /// <summary>Shared action-group button contract used by compact workbench destinations.</summary>
    public static bool DrawActionGroup(
        string label,
        bool enabled = true,
        string? help = null,
        bool small = false) => ActionButton(label, enabled, help, small);

    /// <summary>
    /// Draws a section title that is not collapsible. Sections that hold data
    /// (lists, filters) keep their own tree nodes; this is for the headings
    /// that only group content.
    /// </summary>
    public static void SectionTitle(string title)
    {
        ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, new Vector2(UiTokens.Space8, UiTokens.Space4));
        ImGui.SeparatorText(title);
        ImGui.PopStyleVar();
    }

    /// <summary>Body text in the shared body colour (token text or token muted text).</summary>
    public static void BodyText(string text, bool muted = false)
    {
        ImGui.PushStyleColor(ImGuiCol.Text, muted ? UiTokens.TextMuted : UiTokens.Text);
        ImGui.TextUnformatted(text);
        ImGui.PopStyleColor();
    }

    /// <summary>Wrapped body text. The text is never treated as a format string.</summary>
    public static void BodyTextWrapped(string text, bool muted = false)
    {
        ImGui.PushStyleColor(ImGuiCol.Text, muted ? UiTokens.TextMuted : UiTokens.Text);
        ImGui.PushTextWrapPos(0f);
        ImGui.TextUnformatted(text);
        ImGui.PopTextWrapPos();
        ImGui.PopStyleColor();
    }

    /// <summary>Status text that reports something worked, in the shared success colour.</summary>
    public static void SuccessText(string text)
    {
        ImGui.PushStyleColor(ImGuiCol.Text, UiTokens.Success);
        ImGui.TextUnformatted(text);
        ImGui.PopStyleColor();
    }

    /// <summary>A selectable row drawn in the muted text colour.</summary>
    public static bool MutedSelectable(
        string label,
        bool selected = false,
        ImGuiSelectableFlags flags = ImGuiSelectableFlags.None,
        Vector2 size = default)
    {
        ImGui.PushStyleColor(ImGuiCol.Text, UiTokens.TextMuted);
        bool clicked = ImGui.Selectable(label, selected, flags, size);
        ImGui.PopStyleColor();
        return clicked;
    }

    /// <summary>Builds one button of a <see cref="DrawButtonRow"/>.</summary>
    public static ButtonSpec Btn(
        string label,
        Action action,
        bool enabled = true,
        string? help = null,
        bool small = false) => new(label, action, enabled, help, small);

    /// <summary>Draws a row of action buttons separated by the shared token spacing.</summary>
    public static void DrawButtonRow(params ButtonSpec[] buttons)
    {
        if (buttons == null || buttons.Length == 0)
            return;

        for (int i = 0; i < buttons.Length; i++)
        {
            if (i > 0)
                ImGui.SameLine(0f, UiTokens.Space8);

            ButtonSpec button = buttons[i];
            if (!button.Enabled)
                ImGui.BeginDisabled();

            bool clicked = button.Small ? ImGui.SmallButton(button.Label) : ImGui.Button(button.Label);

            if (!button.Enabled)
                ImGui.EndDisabled();

            if (!string.IsNullOrWhiteSpace(button.Help) && ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
                ImGui.SetTooltip(button.Help);

            if (clicked)
                button.Action?.Invoke();
        }
    }

    /// <summary>
    /// Explicit empty state: what is missing, and the action that fixes it when
    /// the panel owns one.
    /// </summary>
    public static void DrawEmptyState(
        string title,
        string message,
        string? actionLabel = null,
        Action? action = null,
        string? help = null)
    {
        BodyText(title);
        BodyTextWrapped(message, muted: true);

        if (string.IsNullOrWhiteSpace(actionLabel) || action == null)
            return;

        ImGui.Spacing();
        if (ActionButton(actionLabel, help: help))
            action();
    }

    /// <summary>
    /// Applies the shared window surface (background, border, rounding, padding)
    /// so every manager-owned region reads as the same surface.
    /// </summary>
    public static void PushSurfaceStyle(float scale = 1f)
    {
        float s = float.IsFinite(scale) ? Math.Clamp(scale, 1f, 2.5f) : 1f;
        ImGui.PushStyleVar(ImGuiStyleVar.WindowRounding, UiTokens.WindowRounding);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowBorderSize, UiTokens.WindowBorderSize);
        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, UiTokens.WindowPadding * s);
        ImGui.PushStyleColor(ImGuiCol.WindowBg, UiTokens.Surface1);
        ImGui.PushStyleColor(ImGuiCol.Border, UiTokens.Border);
    }

    /// <summary>Undoes <see cref="PushSurfaceStyle"/>.</summary>
    public static void PopSurfaceStyle()
    {
        ImGui.PopStyleColor(2);
        ImGui.PopStyleVar(3);
    }

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
