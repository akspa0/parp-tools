using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Runtime.Marketing;

namespace WoWViewer.Capture;

/// <summary>
/// Viewer-only presentation adapter for the active feature-tour beat. It draws one intentionally
/// small callout on top of a clean scene; it does not open, move, or depend on normal UI chrome.
/// </summary>
public static class MarketingTourOverlayRenderer
{
    public static void Draw(in FeatureTourPresentation presentation)
    {
        Vector2 displaySize = ImGui.GetIO().DisplaySize;
        if (displaySize.X <= 0 || displaySize.Y <= 0)
            return;

        const float margin = 36f;
        const float panelHeight = 104f;
        float panelWidth = Math.Clamp(displaySize.X * 0.42f, 360f, 680f);
        Vector2 panelMin = new(margin, displaySize.Y - margin - panelHeight);
        Vector2 panelMax = panelMin + new Vector2(panelWidth, panelHeight);
        ImDrawListPtr drawList = ImGui.GetForegroundDrawList();
        drawList.AddRectFilled(panelMin, panelMax, 0xE61B1522, 10f);
        drawList.AddRect(panelMin, panelMax, 0xFF85C8FF, 10f, ImDrawFlags.None, 1.5f);
        drawList.AddText(panelMin + new Vector2(18f, 16f), 0xFFFFFFFF, presentation.Title);
        if (!string.IsNullOrWhiteSpace(presentation.Body))
            drawList.AddText(panelMin + new Vector2(18f, 48f), 0xFFD8D2E5, presentation.Body);
        drawList.AddText(panelMax - new Vector2(18f, 22f), 0xFF85C8FF, presentation.FeatureId);
    }
}
