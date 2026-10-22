using System.Numerics;
using ImGuiNET;

namespace WoWViewer;

/// <summary>
/// Spec 286 U0: bounded FPS-hover disclosure.
/// <para>
/// Draws the immutable <see cref="FooterDiagnosticsSnapshot"/> the status bar already produced.
/// It never collects telemetry, touches assets, or opens a second diagnostics pipeline. The
/// tooltip is size-constrained so a long area name or a large pending counter cannot push it
/// outside the client area.
/// </para>
/// </summary>
internal static class FooterDiagnosticsPresenter
{
    /// <summary>
    /// Show the runtime detail tooltip while the last drawn item (the FPS readout) is hovered.
    /// </summary>
    public static void DrawHoverDisclosure(FooterDiagnosticsSnapshot snapshot, Vector2 display, float scale)
    {
        if (!ImGui.IsItemHovered()) return;

        float s = MathF.Max(1f, scale);
        ImGui.SetNextWindowSizeConstraints(
            new Vector2(0f, 0f),
            new Vector2(ViewerFooterLayout.TooltipMaxWidth * s, ViewerFooterLayout.TooltipMaxHeight * s));

        ImGui.BeginTooltip();
        ImGui.PushTextWrapPos(ImGui.GetFontSize() * 28f);
        ImGui.TextUnformatted("Runtime detail");
        ImGui.Separator();
        foreach (string line in snapshot.DetailLines)
            ImGui.TextUnformatted(line);
        ImGui.PopTextWrapPos();
        ImGui.EndTooltip();
    }
}
