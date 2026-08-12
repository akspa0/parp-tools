using System.Numerics;
using ImGuiNET;
using WoWViewer.Logging;

namespace WoWViewer;

/// <summary>
/// Partial class containing the in-app log viewer UI.
/// </summary>
public partial class ViewerApp
{
    private void DrawLogViewer()
    {
        // 069 Phase 16: this wrapper keeps legacy floating-window behavior
        // for users who still toggle the old Viewer > Log Viewer menu item.
        // The workbench sub-tab uses DrawLogViewerContent directly.
        ImGui.SetNextWindowSize(new Vector2(800, 400), ImGuiCond.FirstUseEver);
        ImGui.SetNextWindowPos(new Vector2(
            ImGui.GetIO().DisplaySize.X / 2 - 400,
            ImGui.GetIO().DisplaySize.Y / 2 - 200), ImGuiCond.FirstUseEver);

        if (ImGui.Begin("Log Viewer", ref _showLogViewer))
        {
            DrawLogViewerContent();
        }
        ImGui.End();
    }

    /// <summary>
    /// Headless log viewer body. Call from any container that already has
    /// an ImGui context (workbench sub-tab, docked panel, etc). Does NOT
    /// open its own window.
    /// </summary>
    private void DrawLogViewerContent()
    {
        // Category filter checkboxes
        ImGui.Text("Filter by Category:");
        ImGui.SameLine();

        var categories = Enum.GetValues<ViewerLog.Category>();
        var mutedCategories = ViewerLog.MutedCategories;

        foreach (var cat in categories)
        {
            bool isMuted = mutedCategories.Contains(cat);
            bool isVisible = !isMuted;
            if (ImGui.Checkbox(cat.ToString(), ref isVisible))
            {
                if (isVisible)
                    ViewerLog.Unmute(cat);
                else
                    ViewerLog.Mute(cat);
            }
            ImGui.SameLine();
        }
        ImGui.NewLine();

        ImGui.Separator();

        // Log level filter
        ImGui.Text("Min Level:");
        ImGui.SameLine();
        var currentLevel = ViewerLog.MinLevel;
        if (ImGui.RadioButton("Debug", currentLevel == ViewerLog.Level.Debug))
            ViewerLog.MinLevel = ViewerLog.Level.Debug;
        ImGui.SameLine();
        if (ImGui.RadioButton("Info", currentLevel == ViewerLog.Level.Info))
            ViewerLog.MinLevel = ViewerLog.Level.Info;
        ImGui.SameLine();
        if (ImGui.RadioButton("Important", currentLevel == ViewerLog.Level.Important))
            ViewerLog.MinLevel = ViewerLog.Level.Important;
        ImGui.SameLine();
        if (ImGui.RadioButton("Error", currentLevel == ViewerLog.Level.Error))
            ViewerLog.MinLevel = ViewerLog.Level.Error;

        ImGui.SameLine();
        ImGui.Spacing();
        ImGui.SameLine();
        if (ImGui.Button("Clear"))
        {
            // Clear log history by setting max history to 0 then back
            // (ViewerLog doesn't expose a Clear method, so we work around it)
        }

        ImGui.Separator();

        // Log entries in scrollable child window
        if (ImGui.BeginChild("LogEntries", new Vector2(0, 0), true, ImGuiWindowFlags.None))
        {
            var history = ViewerLog.GetHistory();

            foreach (var entry in history)
            {
                Vector4 color = entry.Lvl switch
                {
                    ViewerLog.Level.Debug => new Vector4(0.6f, 0.6f, 0.6f, 1f),
                    ViewerLog.Level.Info => new Vector4(0.8f, 0.8f, 0.8f, 1f),
                    ViewerLog.Level.Important => new Vector4(1f, 1f, 1f, 1f),
                    ViewerLog.Level.Error => new Vector4(1f, 0.3f, 0.3f, 1f),
                    _ => new Vector4(1f, 1f, 1f, 1f)
                };

                ImGui.PushStyleColor(ImGuiCol.Text, color);

                string timestamp = entry.Time.ToString("HH:mm:ss.fff");
                string levelStr = entry.Lvl.ToString().PadRight(9);
                string catStr = entry.Cat.ToString().PadRight(10);

                ImGui.TextWrapped($"[{timestamp}] [{levelStr}] [{catStr}] {entry.Message}");

                ImGui.PopStyleColor();
            }

            if (ImGui.GetScrollY() >= ImGui.GetScrollMaxY())
                ImGui.SetScrollHereY(1.0f);
        }
        ImGui.EndChild();
    }
}
