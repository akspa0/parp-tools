using System.Numerics;
using ImGuiNET;
using WoWViewer.UI;
using WoWViewer.Workbench;

namespace WoWViewer;

/// <summary>
/// Small, anchored icon rails that open independent tool windows instead of permanently
/// displaying full-height data sidebars (Spec 283 operator amendment, 2026-10-21).
/// Tool content remains owned by its existing provider.
/// </summary>
internal sealed class ViewerActionBarsService
{
    private enum IconKind { Workspace, Open, Assets, Taxi, Capture, Inspector, World, Model, Gossip, Pm4, Maps, Diagnostics, Export, Audio, Spawns, Layers }

    private sealed class WindowState
    {
        public bool Open;
    }

    private readonly IViewerAppHost _host;
    private readonly Dictionary<string, WindowState> _windows = new(StringComparer.Ordinal)
    {
        ["workspace"] = new(), ["assets"] = new(), ["taxi"] = new(), ["capture"] = new(),
        ["inspector"] = new(), ["world"] = new(), ["model"] = new(), ["gossip"] = new(), ["pm4"] = new(),
        ["destination"] = new(), ["maps"] = new(), ["files"] = new(), ["diagnostics"] = new(), ["options"] = new(), ["layers"] = new(),
    };

    internal ViewerActionBarsService(IViewerAppHost host) => _host = host;

    internal bool IsCaptureOpen => _host.UiManager.LeftDockOpen && _host.UiManager.ActiveLeftPanel == LeftPanel.Capture;
    internal bool IsOptionsOpen => _host.UiManager.LeftDockOpen && _host.UiManager.ActiveLeftPanel == LeftPanel.Options;
    internal bool IsLayersOpen => _host.UiManager.LeftDockOpen && _host.UiManager.ActiveLeftPanel == LeftPanel.Layers;
    internal void OpenLayersWindow() => _host.UiManager.OpenLeftDock(LeftPanel.Layers);
    internal void ToggleLayersWindow() => _host.UiManager.ToggleLeftDock(LeftPanel.Layers);
    internal bool IsMapsOpen => _host.UiManager.LeftDockOpen && _host.UiManager.ActiveLeftPanel == LeftPanel.Maps;
    internal void OpenMapsWindow() => _host.UiManager.OpenLeftDock(LeftPanel.Maps);
    internal void CloseMapsWindow() { if (_host.UiManager.ActiveLeftPanel == LeftPanel.Maps) _host.UiManager.CloseLeftDock(); }
    internal void OpenOptionsWindow() => _host.UiManager.OpenLeftDock(LeftPanel.Options);
    internal void ToggleOptionsWindow() => _host.UiManager.ToggleLeftDock(LeftPanel.Options);
    internal void CloseAll()
    {
        _host.UiManager.CloseLeftDock();
        _host.UiManager.CloseRightDock();
    }

    internal void OpenViewerPanel(ViewerPanel panel) => _host.UiManager.OpenViewerDock(panel);

    internal void OpenSceneTool(int page) => _host.UiManager.OpenLeftDock(page == 1 ? LeftPanel.Capture : LeftPanel.Taxi);
    internal void ToggleSceneTool(int page) => _host.UiManager.ToggleLeftDock(page == 1 ? LeftPanel.Capture : LeftPanel.Taxi);

    internal void OpenDestinationWindow()
    {
        if (_host.ActiveTopTab == WorkbenchTab.Viewer)
            OpenViewerPanel(_host.UiManager.ViewerPanel);
        else
            _host.UiManager.Navigate(_host.ActiveTopTab);
    }
    internal void OpenWorkspaceWindow() => _host.UiManager.OpenLeftDock(LeftPanel.Workspace);

    internal void DrawLeftBar()
    {
        float scale = MathF.Max(1f, _host.UiManager.Layout.Scale);
        var size = new Vector2(34f * scale, 34f * scale);
        bool dockOpen = _host.UiManager.LeftDockOpen;
        LeftPanel active = _host.UiManager.ActiveLeftPanel;
        DrawActionBarIcon("Workspace", "ActionWorkspace", IconKind.Workspace, size, "Open camera, fog, and source controls", () => _host.UiManager.ToggleLeftDock(LeftPanel.Workspace), dockOpen && active == LeftPanel.Workspace);
        DrawActionBarIcon("File Browser", "ActionFiles", IconKind.Open, size, "Browse files in the active client", () => _host.UiManager.ToggleLeftDock(LeftPanel.Files), dockOpen && active == LeftPanel.Files);
        DrawActionBarIcon("Maps", "ActionMaps", IconKind.Maps, size, "Browse maps and choose a spawn point", () => _host.UiManager.ToggleLeftDock(LeftPanel.Maps), dockOpen && active == LeftPanel.Maps);
        DrawActionBarIcon("Assets", "ActionAssets", IconKind.Assets, size, "Browse and preview assets", () => _host.UiManager.ToggleLeftDock(LeftPanel.Assets), dockOpen && active == LeftPanel.Assets);
        DrawActionBarIcon("Layers", "ActionLayers", IconKind.Layers, size, "Open texture layers, alpha masks, and opacity blending workbench", () => _host.UiManager.ToggleLeftDock(LeftPanel.Layers), dockOpen && active == LeftPanel.Layers);
        ImGui.Separator();
        DrawActionBarIcon("Taxi", "ActionTaxi", IconKind.Taxi, size, "Open taxi and route tools", () => _host.UiManager.ToggleLeftDock(LeftPanel.Taxi), dockOpen && active == LeftPanel.Taxi);
        DrawActionBarIcon("Capture", "ActionCapture", IconKind.Capture, size, "Open camera and capture tools", () => _host.UiManager.ToggleLeftDock(LeftPanel.Capture), dockOpen && active == LeftPanel.Capture);
        DrawActionBarIcon("Diagnostics", "ActionDiagnostics", IconKind.Diagnostics, size, "Open runtime statistics, logs, and performance controls", () => _host.UiManager.ToggleLeftDock(LeftPanel.Diagnostics), dockOpen && active == LeftPanel.Diagnostics);
        DrawActionBarIcon("Audio", "ActionAudio", IconKind.Audio, size, "Open audio controls, volumes, and emitter diagnostics", () => _host.UiManager.ToggleLeftDock(LeftPanel.Audio), dockOpen && active == LeftPanel.Audio);
    }

    internal void DrawRightBar()
    {
        float scale = MathF.Max(1f, _host.UiManager.Layout.Scale);
        var size = new Vector2(34f * scale, 34f * scale);
        bool dockOpen = _host.UiManager.RightDockOpen;
        ViewerPanel active = _host.UiManager.ViewerPanel;
        // One line per call: the UI control scanner reads source line-by-line.
        DrawActionBarIcon("Inspector", "ActionInspector", IconKind.Inspector, size, "Inspect the selected scene asset", () => _host.UiManager.ToggleRightDock(ViewerPanel.Inspector), dockOpen && active == ViewerPanel.Inspector);
        DrawActionBarIcon("World", "ActionWorld", IconKind.World, size, "Inspect objects, placements, and world data", () => _host.UiManager.ToggleRightDock(ViewerPanel.World), dockOpen && active == ViewerPanel.World);
        DrawActionBarIcon("Gossip", "ActionGossip", IconKind.Gossip, size, "Inspect creature gossip, quests, and dialogue", () => _host.UiManager.ToggleRightDock(ViewerPanel.Gossip), dockOpen && active == ViewerPanel.Gossip);
        DrawActionBarIcon("PM4", "ActionPm4", IconKind.Pm4, size, "Inspect PM4 navigation and object data", () => _host.UiManager.ToggleRightDock(ViewerPanel.PM4), dockOpen && active == ViewerPanel.PM4);
        DrawActionBarIcon("Exports", "ActionExports", IconKind.Export, size, "Export PM4 OBJ/JSON models, placement ledgers, and reports", () => _host.UiManager.ToggleRightDock(ViewerPanel.Exports), dockOpen && active == ViewerPanel.Exports);
        DrawActionBarIcon("Spawns", "ActionSpawns", IconKind.Spawns, size, "Browse map spawns and population points", () => _host.UiManager.ToggleRightDock(ViewerPanel.Spawns), dockOpen && active == ViewerPanel.Spawns);
    }

    /// <summary>
    /// Draw the four bottom actions at the placements the footer fit policy computed. The caller
    /// owns the row origin so destination buttons, these actions and Options share one measured
    /// layout instead of each drawing left-to-right into a scrolling window.
    /// </summary>
    internal void DrawBottomActions(FooterFitResult fit, float originX, float y)
    {
        float height = ImGui.GetFrameHeight();

        // Literal labels stay inline so the UI control scanner keeps finding these four routes.
        FooterItemPlacement spawns = fit.Get("bottom-spawns");
        if (spawns.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + spawns.X, y));
            if (ImGui.Button("Spawns##BottomSpawns", new Vector2(spawns.Width, height)))
                _host.UiManager.Popups.Request(ViewerUiManager.MapSpawnsPopupId);
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Choose a map spawn or open population controls");
        }
        // No bar-toggle buttons: Tab hides all chrome and [ / ] toggle the rails from the keyboard.
    }

    internal void DrawToolWindows()
    {
        // Spec 291: all left tools, options, and destinations dock inside UiRegion.LeftDock and UiRegion.RightDock.
    }

    private void ToggleWindow(string key) => _windows[key].Open = !_windows[key].Open;

    private void DrawActionBarIcon(string label, string id, IconKind icon, Vector2 size,
        string help, Action action, bool? selected = null)
    {
        Vector2 position = ImGui.GetCursorScreenPos();
        string key = icon == IconKind.Open ? "files" : icon.ToString().ToLowerInvariant();
        bool isSelected = selected ?? (_windows.TryGetValue(key, out var window) && window.Open);
        if (SharedUiWidgets.IconButton(label, id, size, help, isSelected)) action();

        ImDrawListPtr draw = ImGui.GetWindowDrawList();
        Vector2 center = position + size * 0.5f;
        float unit = MathF.Min(size.X, size.Y);
        float half = unit * 0.28f;
        float stroke = MathF.Max(1.5f, unit * 0.055f);
        uint ink = ImGui.ColorConvertFloat4ToU32(UiTokens.Brass);
        uint muted = ImGui.ColorConvertFloat4ToU32(UiTokens.BrassDim);
        Vector2 a, b, c, d;

        switch (icon)
        {
            case IconKind.Workspace:
                for (int i = -1; i <= 1; i++)
                {
                    float y = center.Y + i * unit * 0.18f;
                    draw.AddLine(new(center.X - half, y), new(center.X + half, y), ink, stroke);
                    float knob = center.X + (i == 0 ? -unit * 0.08f : unit * 0.12f);
                    draw.AddCircleFilled(new(knob, y), unit * 0.07f, muted, 8);
                }
                break;
            case IconKind.Open:
                a = new(center.X - half, center.Y - unit * 0.12f);
                b = new(center.X - unit * 0.12f, center.Y - unit * 0.12f);
                c = new(center.X - unit * 0.04f, center.Y - unit * 0.22f);
                d = new(center.X + half, center.Y - unit * 0.22f);
                draw.AddRectFilled(new(center.X - half, center.Y - unit * 0.12f), new(center.X + half, center.Y + unit * 0.26f), muted, unit * 0.04f);
                draw.AddRect(new(center.X - half, center.Y - unit * 0.12f), new(center.X + half, center.Y + unit * 0.26f), ink, unit * 0.04f, ImDrawFlags.None, stroke);
                draw.AddLine(a, b, ink, stroke); draw.AddLine(b, c, ink, stroke); draw.AddLine(c, d, ink, stroke);
                break;
            case IconKind.Assets:
                draw.AddRect(new(center.X - half, center.Y - half), new(center.X + half, center.Y + half), ink, 0f, ImDrawFlags.None, stroke);
                draw.AddRect(new(center.X - unit * 0.16f, center.Y - unit * 0.16f), new(center.X + unit * 0.16f, center.Y + unit * 0.16f), muted, 0f, ImDrawFlags.None, stroke);
                break;
            case IconKind.Taxi:
                a = new(center.X - half, center.Y + unit * 0.2f);
                b = center;
                c = new(center.X + half, center.Y - unit * 0.2f);
                draw.AddLine(a, b, ink, stroke); draw.AddLine(b, c, ink, stroke);
                draw.AddCircleFilled(a, unit * 0.08f, muted, 10); draw.AddCircleFilled(c, unit * 0.08f, muted, 10);
                break;
            case IconKind.Capture:
                draw.AddRect(new(center.X - half, center.Y - unit * 0.2f), new(center.X + half, center.Y + unit * 0.24f), ink, unit * 0.06f, ImDrawFlags.None, stroke);
                draw.AddCircle(center, unit * 0.14f, muted, 16, stroke);
                draw.AddLine(new(center.X - unit * 0.16f, center.Y - unit * 0.2f), new(center.X + unit * 0.12f, center.Y - unit * 0.2f), ink, stroke);
                break;
            case IconKind.Inspector:
                draw.AddCircle(new(center.X - unit * 0.06f, center.Y - unit * 0.05f), unit * 0.21f, ink, 16, stroke);
                draw.AddLine(new(center.X + unit * 0.1f, center.Y + unit * 0.11f), new(center.X + half, center.Y + half), muted, stroke * 1.4f);
                break;
            case IconKind.World:
                draw.AddCircle(center, half, ink, 20, stroke);
                draw.AddLine(new(center.X - half, center.Y), new(center.X + half, center.Y), muted, stroke);
                draw.AddLine(new(center.X, center.Y - half), new(center.X, center.Y + half), muted, stroke);
                draw.AddCircle(center, unit * 0.16f, muted, 16, stroke);
                break;
            case IconKind.Model:
                a = new(center.X, center.Y - half); b = new(center.X + half, center.Y - unit * 0.14f);
                c = new(center.X + half, center.Y + unit * 0.24f); d = new(center.X, center.Y + half);
                Vector2 leftTop = new(center.X - half, center.Y - unit * 0.14f);
                Vector2 leftBottom = new(center.X - half, center.Y + unit * 0.24f);
                draw.AddLine(a, b, ink, stroke); draw.AddLine(b, c, ink, stroke); draw.AddLine(c, d, ink, stroke);
                draw.AddLine(d, leftBottom, ink, stroke); draw.AddLine(leftBottom, leftTop, ink, stroke); draw.AddLine(leftTop, a, ink, stroke);
                draw.AddLine(a, center, muted, stroke); draw.AddLine(b, center, muted, stroke); draw.AddLine(center, d, muted, stroke);
                break;
            case IconKind.Gossip:
                Vector2 bMin = new(center.X - half * 0.9f, center.Y - half * 0.8f);
                Vector2 bMax = new(center.X + half * 0.9f, center.Y + half * 0.35f);
                draw.AddRect(bMin, bMax, ink, unit * 0.12f, ImDrawFlags.None, stroke);
                Vector2 t1 = new(center.X - half * 0.35f, bMax.Y);
                Vector2 t2 = new(center.X - half * 0.7f, center.Y + half * 0.9f);
                Vector2 t3 = new(center.X - half * 0.05f, bMax.Y);
                draw.AddLine(t1, t2, ink, stroke);
                draw.AddLine(t2, t3, ink, stroke);
                draw.AddLine(new(bMin.X + unit * 0.14f, center.Y - unit * 0.10f), new(bMax.X - unit * 0.14f, center.Y - unit * 0.10f), muted, stroke);
                draw.AddLine(new(bMin.X + unit * 0.14f, center.Y + unit * 0.08f), new(bMax.X - unit * 0.28f, center.Y + unit * 0.08f), muted, stroke);
                break;
            case IconKind.Pm4:
                for (int i = -1; i <= 1; i++)
                {
                    float offset = i * unit * 0.18f;
                    draw.AddLine(new(center.X - half, center.Y + offset), new(center.X + half, center.Y + offset), i == 0 ? ink : muted, stroke);
                    draw.AddLine(new(center.X + offset, center.Y - half), new(center.X + offset, center.Y + half), i == 0 ? ink : muted, stroke);
                }
                break;
            case IconKind.Maps:
                a = new(center.X - half, center.Y - half);
                b = new(center.X - unit * 0.08f, center.Y - unit * 0.18f);
                c = new(center.X + unit * 0.08f, center.Y - half);
                d = new(center.X + half, center.Y - unit * 0.18f);
                draw.AddLine(a, b, ink, stroke); draw.AddLine(b, c, ink, stroke); draw.AddLine(c, d, ink, stroke);
                draw.AddLine(a, a + new Vector2(0f, half * 2f), ink, stroke);
                draw.AddLine(b, b + new Vector2(0f, half * 2f), muted, stroke);
                draw.AddLine(c, c + new Vector2(0f, half * 2f), muted, stroke);
                draw.AddLine(d, d + new Vector2(0f, half * 2f), ink, stroke);
                draw.AddLine(a + new Vector2(0f, half * 2f), b + new Vector2(0f, half * 2f), ink, stroke);
                draw.AddLine(b + new Vector2(0f, half * 2f), c + new Vector2(0f, half * 2f), ink, stroke);
                draw.AddLine(c + new Vector2(0f, half * 2f), d + new Vector2(0f, half * 2f), ink, stroke);
                break;
            case IconKind.Diagnostics:
                draw.AddLine(new(center.X - half, center.Y + half), new(center.X + half, center.Y + half), ink, stroke);
                for (int i = 0; i < 3; i++)
                {
                    float x = center.X - half + (i + 0.5f) * unit * 0.18f;
                    draw.AddLine(new(x, center.Y + half), new(x, center.Y + half - (i + 1) * unit * 0.16f), ink, stroke * 2f);
                }
                break;
            case IconKind.Audio:
                draw.AddRect(new(center.X - half, center.Y - half * 0.4f), new(center.X - half * 0.3f, center.Y + half * 0.4f), ink, 0f, ImDrawFlags.None, stroke);
                draw.AddLine(new(center.X - half * 0.3f, center.Y - half * 0.4f), new(center.X + half * 0.2f, center.Y - half * 0.8f), ink, stroke);
                draw.AddLine(new(center.X + half * 0.2f, center.Y - half * 0.8f), new(center.X + half * 0.2f, center.Y + half * 0.8f), ink, stroke);
                draw.AddLine(new(center.X + half * 0.2f, center.Y + half * 0.8f), new(center.X - half * 0.3f, center.Y + half * 0.4f), ink, stroke);
                draw.AddLine(new(center.X + half * 0.5f, center.Y - half * 0.4f), new(center.X + half * 0.5f, center.Y + half * 0.4f), muted, stroke);
                draw.AddLine(new(center.X + half * 0.8f, center.Y - half * 0.7f), new(center.X + half * 0.8f, center.Y + half * 0.7f), muted, stroke);
                break;
            case IconKind.Spawns:
                draw.AddCircle(new(center.X, center.Y - half * 0.3f), half * 0.5f, ink, 12, stroke);
                draw.AddLine(new(center.X, center.Y + half * 0.2f), new(center.X, center.Y + half), ink, stroke);
                draw.AddLine(new(center.X - half * 0.4f, center.Y - half * 0.3f), new(center.X + half * 0.4f, center.Y - half * 0.3f), muted, stroke);
                break;
            case IconKind.Export:
                draw.AddLine(new(center.X - half, center.Y), new(center.X - half, center.Y + half), ink, stroke);
                draw.AddLine(new(center.X - half, center.Y + half), new(center.X + half, center.Y + half), ink, stroke);
                draw.AddLine(new(center.X + half, center.Y + half), new(center.X + half, center.Y), ink, stroke);
                draw.AddLine(new(center.X, center.Y + half * 0.4f), new(center.X, center.Y - half), ink, stroke);
                draw.AddLine(new(center.X, center.Y - half), new(center.X - unit * 0.14f, center.Y - half * 0.3f), muted, stroke);
                draw.AddLine(new(center.X, center.Y - half), new(center.X + unit * 0.14f, center.Y - half * 0.3f), muted, stroke);
                break;
            case IconKind.Layers:
                draw.AddRect(new(center.X - half * 0.9f, center.Y - half * 0.8f), new(center.X + half * 0.5f, center.Y - half * 0.1f), muted, 0f, ImDrawFlags.None, stroke);
                draw.AddRect(new(center.X - half * 0.7f, center.Y - half * 0.4f), new(center.X + half * 0.7f, center.Y + half * 0.3f), muted, 0f, ImDrawFlags.None, stroke);
                draw.AddRect(new(center.X - half * 0.5f, center.Y), new(center.X + half * 0.9f, center.Y + half * 0.7f), ink, 0f, ImDrawFlags.None, stroke);
                break;
        }
    }

    private void DrawToolWindow(string key, string title, Vector2 position, Vector2 logicalSize, Action content)
    {
        WindowState state = _windows[key];
        if (!state.Open) return;

        float scale = MathF.Max(1f, _host.UiManager.Layout.Scale);
        Vector2 display = _host.UiManager.DisplaySize;
        UiRect rect = UiLayout.ToolWindowRect(display, position, logicalSize, scale);
        ImGui.SetNextWindowPos(rect.Position, ImGuiCond.Appearing);
        ImGui.SetNextWindowSize(rect.Size, ImGuiCond.Appearing);
        ImGui.SetNextWindowSizeConstraints(Vector2.Min(new Vector2(280f, 180f) * scale, rect.Size),
            Vector2.Max(rect.Size, display - new Vector2(16f * scale)));
        SharedUiWidgets.PushSurfaceStyle(scale);
        if (ImGui.Begin($"{title}###ActionWindow_{key}", ref state.Open,
                ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoCollapse))
            content();
        ImGui.End();
        SharedUiWidgets.PopSurfaceStyle();
    }

}
