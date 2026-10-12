using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Spec 247: which client formats a map export writes. One export button, the formats chosen by
/// checkbox, so the same action serves LK and Alpha without a second button per target.
/// <para>
/// This is an owned settings service rather than fields on the viewer shell (AGENTS.md §10). It is static
/// because the selection is a single application-wide preference, and it is read from both the File menu
/// and the sidebar.
/// </para>
/// </summary>
public static class MapExportFormats
{
    /// <summary>Wrath / LK v18: one .adt per tile plus a .wdt.</summary>
    public static bool LkAdtV18 { get; set; } = true;

    /// <summary>Alpha 0.5.3: a single monolithic .wdt carrying every tile.</summary>
    public static bool AlphaWdt053 { get; set; }

    public static bool Any => LkAdtV18 || AlphaWdt053;

    /// <summary>The selection as the core conversion enum, in a stable order.</summary>
    public static IReadOnlyList<MapConversionTargetFormat> Selected
    {
        get
        {
            var targets = new List<MapConversionTargetFormat>(2);
            if (LkAdtV18)
                targets.Add(MapConversionTargetFormat.LkAdtV18);
            if (AlphaWdt053)
                targets.Add(MapConversionTargetFormat.AlphaWdt053);
            return targets;
        }
    }

    /// <summary>Short label for a button or status line, e.g. "LK v18 + Alpha 0.5.3".</summary>
    public static string Summary => !Any
        ? "no format selected"
        : string.Join(" + ", new[]
        {
            LkAdtV18 ? "LK v18" : null,
            AlphaWdt053 ? "Alpha 0.5.3" : null,
        }.Where(static s => s is not null));

    /// <summary>
    /// Draws the format checkboxes. Shared by the File menu and the sidebar so the two can never drift.
    /// </summary>
    public static void DrawCheckboxes(string idScope)
    {
        bool lk = LkAdtV18;
        if (ImGuiNET.ImGui.Checkbox($"LK v18 ADT (.adt + .wdt)##{idScope}", ref lk))
            LkAdtV18 = lk;

        bool alpha = AlphaWdt053;
        if (ImGuiNET.ImGui.Checkbox($"Alpha 0.5.3 WDT (monolithic)##{idScope}", ref alpha))
            AlphaWdt053 = alpha;

        if (!Any)
            ImGuiNET.ImGui.TextDisabled("Select at least one output format.");
    }
}
