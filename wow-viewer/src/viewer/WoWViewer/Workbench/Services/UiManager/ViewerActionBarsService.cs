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
    private enum IconKind { Workspace, Open, Assets, Taxi, Capture, Inspector, World, Model, Pm4, Maps, Diagnostics }

    private sealed class WindowState
    {
        public bool Open;
    }

    private readonly IViewerAppHost _host;
    private readonly Dictionary<string, WindowState> _windows = new(StringComparer.Ordinal)
    {
        ["workspace"] = new(), ["assets"] = new(), ["taxi"] = new(), ["capture"] = new(),
        ["inspector"] = new(), ["world"] = new(), ["model"] = new(), ["pm4"] = new(),
        ["destination"] = new(), ["maps"] = new(), ["files"] = new(), ["diagnostics"] = new(), ["options"] = new(),
    };

    internal ViewerActionBarsService(IViewerAppHost host) => _host = host;

    internal bool IsCaptureOpen => _windows["capture"].Open;
    internal bool IsOptionsOpen => _windows["options"].Open;
    internal void OpenOptionsWindow() => _windows["options"].Open = true;
    internal void ToggleOptionsWindow() => ToggleWindow("options");
    internal void CloseAll()
    {
        foreach (var window in _windows.Values) window.Open = false;
    }

    internal void OpenViewerPanel(ViewerPanel panel) => _host.UiManager.OpenViewerDock(panel);

    internal void OpenSceneTool(int page)
    {
        _windows[page == 1 ? "capture" : "taxi"].Open = true;
    }

    internal void ToggleSceneTool(int page)
    {
        string key = page == 1 ? "capture" : "taxi";
        _windows[key].Open = !_windows[key].Open;
    }

    internal void OpenDestinationWindow()
    {
        if (_host.ActiveTopTab == WorkbenchTab.Viewer)
            OpenViewerPanel(_host.UiManager.ViewerPanel);
        else
            _windows["destination"].Open = true;
    }
    internal void OpenWorkspaceWindow() => _windows["workspace"].Open = true;

    internal void DrawLeftBar()
    {
        float scale = MathF.Max(1f, _host.UiManager.Layout.Scale);
        var size = new Vector2(34f * scale, 34f * scale);
        DrawActionBarIcon("Workspace", "ActionWorkspace", IconKind.Workspace, size, "Open camera, fog, and source controls", () => ToggleWindow("workspace"));
        DrawActionBarIcon("File Browser", "ActionFiles", IconKind.Open, size, "Browse files in the active client", () => ToggleWindow("files"));
        DrawActionBarIcon("Maps", "ActionMaps", IconKind.Maps, size, "Browse maps and choose a spawn point", () => ToggleWindow("maps"));
        DrawActionBarIcon("Assets", "ActionAssets", IconKind.Assets, size, "Browse and preview assets", () => ToggleWindow("assets"));
        ImGui.Separator();
        DrawActionBarIcon("Taxi", "ActionTaxi", IconKind.Taxi, size, "Open taxi and route tools", () => _host.UiManager.ToggleSceneToolsWindow(0));
        DrawActionBarIcon("Capture", "ActionCapture", IconKind.Capture, size, "Open camera and capture tools", () => _host.UiManager.ToggleSceneToolsWindow(1));
        DrawActionBarIcon("Diagnostics", "ActionDiagnostics", IconKind.Diagnostics, size, "Open runtime statistics, logs, and performance controls", () => ToggleWindow("diagnostics"));
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
        DrawActionBarIcon("Model", "ActionModel", IconKind.Model, size, "Inspect the loaded model and animations", () => _host.UiManager.ToggleRightDock(ViewerPanel.Model), dockOpen && active == ViewerPanel.Model);
        DrawActionBarIcon("PM4", "ActionPm4", IconKind.Pm4, size, "Inspect PM4 navigation and object data", () => _host.UiManager.ToggleRightDock(ViewerPanel.PM4), dockOpen && active == ViewerPanel.PM4);
    }

    /// <summary>
    /// Draw the four bottom actions at the placements the footer fit policy computed. The caller
    /// owns the row origin so destination buttons, these actions and Options share one measured
    /// layout instead of each drawing left-to-right into a scrolling window.
    /// </summary>
    internal void DrawBottomActions(FooterFitResult fit, float originX, float y, float scale)
    {
        float height = 36f * scale;

        // Literal labels stay inline so the UI control scanner keeps finding these four routes.
        FooterItemPlacement spawns = fit.Get("bottom-spawns");
        if (spawns.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + spawns.X, y));
            if (ImGui.Button("Spawns##BottomSpawns", new Vector2(spawns.Width, height)))
                _host.UiManager.Popups.Request(ViewerUiManager.MapSpawnsPopupId);
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Choose a map spawn or open population controls");
        }

        FooterItemPlacement pages = fit.Get("bottom-pages");
        if (pages.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + pages.X, y));
            if (ImGui.Button("Pages##BottomPages", new Vector2(pages.Width, height))) OpenDestinationWindow();
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Open pages for the selected preset");
        }

        FooterItemPlacement leftBar = fit.Get("toggle-left-action-bar");
        if (leftBar.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + leftBar.X, y));
            if (ImGui.Button("Left Bar##ToggleLeftBar", new Vector2(leftBar.Width, height))) _host.UiManager.ToggleLeftSidebar();
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Show or hide the left action bar");
        }

        FooterItemPlacement rightBar = fit.Get("toggle-right-action-bar");
        if (rightBar.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + rightBar.X, y));
            if (ImGui.Button("Right Bar##ToggleRightBar", new Vector2(rightBar.Width, height))) _host.UiManager.ToggleRightSidebar();
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Show or hide the right action bar");
        }
    }

    internal void DrawToolWindows()
    {
        float scale = MathF.Max(1f, _host.UiManager.Layout.Scale);
        Vector2 display = _host.UiManager.DisplaySize;
        float rail = 44f * scale;
        float mapHeight = _host.UiManager.MinimapVisible ? _host.UiManager.GetRegionRect(UiRegion.Minimap).Height : 0f;
        float top = 36f * scale;
        DrawToolWindow("workspace", "Workspace", new(rail + 12f * scale, top), new(390f, 500f), _host.NavigatorPanel.DrawWorkspaceContent);
        DrawToolWindow("files", "File Browser", new(rail + 12f * scale, top), new(520f, 560f), _host.NavigatorPanel.DrawFilesContent);
        DrawToolWindow("maps", "Maps", new(rail + 12f * scale, top), new(520f, 560f), _host.NavigatorPanel.DrawMapsContent);
        DrawToolWindow("diagnostics", "Diagnostics", new(rail + 12f * scale, top), new(520f, 560f), _host.NavigatorPanel.DrawDiagnosticsContent);
        DrawToolWindow("options", "Options", new(rail + 12f * scale, top), new(430f, 560f), _host.ViewerChrome.DrawOptionsContent);
        DrawToolWindow("assets", "Asset Browser", new(rail + 12f * scale, top + 40f * scale), new(500f, 650f),
            _host.WorkbenchPanels.DrawAssetsContent);
        DrawToolWindow("taxi", "Taxi & Routes", new(rail + 12f * scale, top + 70f * scale), new(500f, 600f),
            _host.WorkbenchPanels.DrawTaxiWindowContent);
        DrawToolWindow("capture", "Capture", new(rail + 12f * scale, top + 100f * scale), new(520f, 620f),
            _host.DrawCapturePanelContent);

        // Inspector/World/Model/PM4 are docked in UiRegion.RightDock, not floating windows.
        float rightX = Math.Max(rail, display.X - rail - 430f * scale - 12f * scale);
        float rightY = top + mapHeight + 12f * scale;
        if (_host.ActiveTopTab != WorkbenchTab.Viewer)
            DrawToolWindow("destination", _host.ActiveTopTab.ToString(), new(rightX, rightY), new(500f, 620f),
                DrawDestinationContent);
    }

    private void DrawDestinationContent()
    {
        string[] pages = WorkbenchNavigator.GetPageLabels(_host.ActiveTopTab);
        int page = WorkbenchPanelsService.DrawPageCombo("##ActionDestinationPage", pages, _host.ActiveBottomTabIndex);
        if (page != _host.ActiveBottomTabIndex)
            _host.UiManager.SelectDestinationPage(page);

        if (_host.ActiveTopTab == WorkbenchTab.Editor)
            _host.NavigatorPanel.DrawProjectOutputContent();
        _host.WorkbenchPanels.DrawDestinationContent(_host.ActiveTopTab);
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
