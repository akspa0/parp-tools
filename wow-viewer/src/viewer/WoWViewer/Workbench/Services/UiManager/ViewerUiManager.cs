using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.UI;
using WoWViewer.Workbench;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>Owns shell windows, destination navigation, popup lifetime and scene input authority.</summary>
internal sealed partial class ViewerUiManager
{
    internal const string MapSpawnsPopupId = "MapSpawns";
    private readonly IViewerAppHost _host;
    private readonly ViewerActionBarsService _actionBars;
    private readonly UiPopupRegistry _popups = new();
    private readonly Dictionary<UiRegion, UiRegionState> _hosted = new();
    private bool _registered;
    private bool _minimapVisible = true;
    private bool _minimapCollapsed;
    private bool _chatCollapsed = true;
    private bool _rightDockOpen;
    private float _rightDockWidth = 360f;
    private float _appliedFontScale = -1f;
    private float _appliedStyleScale = -1f;
    internal ViewerUiManager(IViewerAppHost host)
    {
        _host = host;
        _actionBars = new ViewerActionBarsService(host);
        ViewPreferences = new ViewPreferencesService(host);
        // Resolve the provider at draw time: the manager is constructed before the
        // minimap/status service, so capturing the delegate here would bind a null target.
        StatusContent = () => host.MinimapAndStatus.DrawStatusContent();
    }
    internal UiLayout Layout { get; private set; } = new(Vector2.Zero, false, true, true, 280, 360);
    internal UiPopupRegistry Popups => _popups;
    /// <summary>The one owner of the user's text size and the view reset actions.</summary>
    internal ViewPreferencesService ViewPreferences { get; }
    private readonly UiControlRegistry _controlRegistry = new();
    internal WorkbenchTab Destination => _host.ActiveTopTab;
    internal ViewerPanel ViewerPanel { get; private set; }
    internal int ModelPage { get; private set; }
    internal int WorldPage { get; private set; }
    internal int InspectorPage { get; private set; }
    internal int SceneToolsPage { get; private set; }
    internal bool SceneToolsVisible { get; set; }
    internal Action? StatusContent { get; set; }
    internal Action<UiRect>? MinimapHost { get; set; }
    internal Action<UiRect>? ChatHost { get; set; }

    /// <summary>
    /// The drawable client area in ImGui coordinates. The window size is authoritative:
    /// ImGui's DisplaySize can be the framebuffer size, which puts the bottom bars below
    /// the visible client area on high-DPI or maximized windows.
    /// </summary>
    internal Vector2 DisplaySize
    {
        get
        {
            var size = _host.Window?.Size ?? default;
            return size.X > 0 && size.Y > 0 ? new Vector2(size.X, size.Y) : ImGui.GetIO().DisplaySize;
        }
    }

    internal void UpdateLayout(Vector2 display)
    {
        if (!_layoutLoaded) LoadLayout(display.X);
        float scale = UiLayout.ScaleFor(display);
        // One font scale for the whole shell: the operator's setting times the display scale.
        float fontScale = ViewPreferencePolicy.EffectiveFontScale(_host.UiFontScale, scale);
        if (MathF.Abs(fontScale - _appliedFontScale) > 0.001f)
        {
            ImGui.GetIO().FontGlobalScale = fontScale;
            _appliedFontScale = fontScale;
        }

        // FontGlobalScale scales glyphs but not the style metrics, so at 4K the text grew while
        // line spacing, frame padding and scrollbars stayed at 1080p sizes and every line
        // overlapped the next. Scale the metrics with the same factor.
        if (MathF.Abs(scale - _appliedStyleScale) > 0.001f)
        {
            var style = ImGui.GetStyle();
            style.ItemSpacing = new Vector2(8f, 4f) * scale;
            style.ItemInnerSpacing = new Vector2(4f, 4f) * scale;
            style.FramePadding = new Vector2(6f, 3f) * scale;
            style.CellPadding = new Vector2(4f, 2f) * scale;
            style.ScrollbarSize = 14f * scale;
            style.GrabMinSize = 10f * scale;
            style.IndentSpacing = 21f * scale;
            style.SeparatorTextBorderSize = MathF.Max(1f, scale);
            _appliedStyleScale = scale;
        }

        Layout = new UiLayout(display, _host.HideUiChrome, _host.ShowLeftSidebar, _host.ShowRightSidebar,
            _host.LeftSidebarWidth, _host.RightSidebarWidth, false,
            _minimapVisible, _minimapCollapsed, _host.SqlSpawnStreaming.ChatVisible,
            _chatCollapsed, scale, _rightDockOpen, _rightDockWidth, MeasureFooterRow(fontScale));
    }

    /// <summary>
    /// Height of one button row at the current text size: base font pixels x effective font scale
    /// plus frame padding. Computed from the values this method just applied rather than read back
    /// from ImGui, whose frame font size lags a scale change by a frame.
    /// </summary>
    private static float MeasureFooterRow(float fontScale)
    {
        if (!HasImGuiContext()) return 0f;
        var fonts = ImGui.GetIO().Fonts.Fonts;
        if (fonts.Size == 0) return 0f;
        return fonts[0].FontSize * fontScale + ImGui.GetStyle().FramePadding.Y * 2f;
    }

    internal void Navigate(WorkbenchTab destination, int page = -1)
    {
        _pages[(int)Destination] = _host.ActiveBottomTabIndex;
        _host.ActiveTopTab = Enum.IsDefined(destination) ? destination : WorkbenchTab.Viewer;
        _host.WorkspaceMode = destination switch
        {
            WorkbenchTab.Editor => WorkspaceMode.Editor,
            WorkbenchTab.Archaeology => WorkspaceMode.Archaeology,
            _ => WorkspaceMode.Viewer,
        };
        int count = WorkbenchNavigator.GetPageLabels(destination).Length;
        _host.ActiveBottomTabIndex = Math.Clamp(page < 0 ? _pages[(int)Destination] : page, 0, Math.Max(0, count - 1));
        _host.ShowRightSidebar = true;
        // The destination window owns the right side on Editor/Archaeology; close the viewer
        // dock so the two surfaces cannot overlap.
        if (destination != WorkbenchTab.Viewer) _rightDockOpen = false;
        SaveLayout();
    }

    internal void OpenViewer(ViewerPanel panel, int page = 0)
    {
        ViewerPanel = panel;
        switch (panel)
        {
            case ViewerPanel.Model: ModelPage = Math.Clamp(page, 0, 2); break;
            case ViewerPanel.World: WorldPage = Math.Clamp(page, 0, 3); break;
            case ViewerPanel.Inspector: InspectorPage = Math.Clamp(page, 0, 3); break;
        }
        Navigate(WorkbenchTab.Viewer);
        OpenViewerDock(panel);
    }

    /// <summary>
    /// Open the right dock on <paramref name="panel"/>. The dock is the single home for the
    /// inspector, world, model and PM4 panels: below the minimap, above the bottom bars.
    /// </summary>
    internal void OpenViewerDock(ViewerPanel panel)
    {
        ViewerPanel = panel;
        _rightDockOpen = true;
        SaveLayout();
    }

    internal void ToggleRightDock(ViewerPanel panel)
    {
        if (_rightDockOpen && ViewerPanel == panel)
        {
            _rightDockOpen = false;
        }
        else
        {
            if (Destination != WorkbenchTab.Viewer) Navigate(WorkbenchTab.Viewer);
            ViewerPanel = panel;
            _rightDockOpen = true;
        }
        SaveLayout();
    }

    internal void CloseRightDock()
    {
        _rightDockOpen = false;
        SaveLayout();
    }

    internal bool RightDockOpen => _rightDockOpen;

    internal void OpenSceneTools(int page = 0)
    {
        SceneToolsPage = Math.Clamp(page, 0, 1);
        SceneToolsVisible = true;
        Navigate(WorkbenchTab.Viewer);
        _host.ShowLeftSidebar = true;
        _actionBars.OpenSceneTool(SceneToolsPage);
    }

    internal void ToggleSceneToolsWindow(int page)
    {
        SceneToolsPage = Math.Clamp(page, 0, 1);
        SceneToolsVisible = true;
        _host.ShowLeftSidebar = true;
        _actionBars.ToggleSceneTool(SceneToolsPage);
        SaveLayout();
    }

    internal bool IsMapsOpen => _actionBars.IsMapsOpen;
    internal void OpenMapsWindow() => _actionBars.OpenMapsWindow();
    internal void CloseMapsWindow() => _actionBars.CloseMapsWindow();
    internal void OpenOptions() => _actionBars.OpenOptionsWindow();
    internal void ToggleOptions() => _actionBars.ToggleOptionsWindow();
    internal void OpenWorkspaceWindow() => _actionBars.OpenWorkspaceWindow();
    internal void OpenDestinationPage(WorkbenchTab destination, int page = -1)
    {
        Navigate(destination, page);
        _actionBars.OpenDestinationWindow();
    }
    internal void SelectDestinationPage(int page)
    {
        int count = WorkbenchNavigator.GetPageLabels(Destination).Length;
        _host.ActiveBottomTabIndex = Math.Clamp(page, 0, Math.Max(0, count - 1));
        SaveLayout();
    }
    internal void ToggleMinimap()
    {
        _minimapVisible = !_minimapVisible;
        SaveLayout();
    }

    // The minimap panel carries no controls of its own; these live in Options > Display.
    internal bool MinimapVisible
    {
        get => _minimapVisible;
        set { _minimapVisible = value; SaveLayout(); }
    }

    internal bool MinimapCollapsed
    {
        get => _minimapCollapsed;
        set { _minimapCollapsed = value; SaveLayout(); }
    }

    internal void ToggleLeftSidebar()
    {
        _host.ShowLeftSidebar = !_host.ShowLeftSidebar;
        SaveLayout();
    }
    internal void ToggleRightSidebar()
    {
        _host.ShowRightSidebar = !_host.ShowRightSidebar;
        SaveLayout();
    }
    internal void ToggleChrome() => _host.HideUiChrome = !_host.HideUiChrome;
    internal void RequestKeyboardShortcuts() => _host.MainMenuBar.RequestKeyboardShortcuts();
    internal void CloseTopmostPopup() => _popups.CloseTopmost();
    internal bool IsCaptureVisible => !_host.HideUiChrome && _actionBars.IsCaptureOpen;
    internal bool IsAudioVisible => !_host.HideUiChrome && _actionBars.IsOptionsOpen;

    // Parent minimap/chat conversion reports the actual displayed rectangle here each frame.
    internal void ReportHostedRegion(UiRegion region, UiRect rect, bool visible) => _hosted[region] = new(rect, visible);
    internal UiRect GetRegionRect(UiRegion region) => Layout[region].Rect;

    internal void Draw()
    {
        UpdateLayout(DisplaySize);
        RegisterPopups();
        if (!_host.HideUiChrome)
        {
            DrawRegion(UiRegion.TopBar, "##UiTopBar", () =>
            {
                if (ImGui.BeginMenuBar()) { _host.MainMenuBar.DrawContent(); ImGui.EndMenuBar(); }
            }, ImGuiWindowFlags.MenuBar);
            DrawRegion(UiRegion.LeftSidebar, "##UiLeftSidebar", DrawLeft);
            DrawRegion(UiRegion.RightSidebar, "##UiRightSidebar", DrawRight);
            DrawRegion(UiRegion.RightDock, "##UiRightDock", DrawRightDock,
                ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse);
            DrawRegion(UiRegion.Minimap, "Minimap##UiMinimap", DrawMinimap,
                ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse);
            // Docked panel: no resize and no move. A floating chat could be dragged over the
            // sidebars and had no reliable way back.
            DrawRegion(UiRegion.Chat, "Chat##UiChat", DrawChat, ImGuiWindowFlags.NoScrollbar);
            // Spec 286 U0: footer regions fit their content. A scrollbar here hid essential
            // actions off-screen instead of laying them out inside the real region.
            DrawRegion(UiRegion.BottomBar, "##UiBottomBar", DrawBottom,
                ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse);
            DrawDiagnosticsWindows();
            _actionBars.DrawToolWindows();
        }
        DrawRegion(UiRegion.StatusBar, "##UiStatusBar", () =>
        {
            if (StatusContent != null) StatusContent();
            else ImGui.TextUnformatted(_host.StatusMessage);
        }, ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse);
        // The WDL map-spawn chooser is a standalone window opened from the map list's Spawn
        // button. The Spec 280 shell rewrite dropped its draw call, so the button set the flag and
        // nothing ever rendered the window. It is drawn here, inside the ImGui frame, regardless
        // of chrome visibility because it is a chooser, not chrome.
        _host.WdlPreview.DrawWdlPreviewDialog();
        // Requests can originate from any destination. Drain deferred file operations exactly once.
        _host.MainMenuBar.DrawDeferredActions();
        if (_host.SettingsWindow._showSettingsWindow)
        {
            _host.SettingsWindow._showSettingsWindow = false;
            _popups.Request("Settings");
        }
        while (_host.MainMenuBar.TryTakePopupRequest(out string popup)) _popups.Request(popup);
        _popups.Draw();
    }

    private void DrawChat()
    {
        if (_chatCollapsed)
        {
            if (ImGui.SmallButton("Expand chat##ExpandChat"))
            {
                _chatCollapsed = false;
                SaveLayout();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Expand the chat dock");
            return;
        }

        // Configuration lives in tool windows. The compact chat frame is a full-width log,
        // not a miniature two-column workbench with clipped controls.
        _host.SqlSpawnStreaming.DrawChatContent(() =>
        {
            ImGui.SameLine();
            if (ImGui.SmallButton("Collapse##ChatCollapse"))
            {
                _chatCollapsed = true;
                SaveLayout();
            }
        });
    }

    private void DrawMinimap()
    {
        // The minimap is the minimap. No control row, no zone label, no coordinate readout: the
        // real client's minimap is a map you click, drag and scroll, and every extra line here
        // stole height from the map itself. Collapse, dock and hide live in Options > Display.
        if (_minimapCollapsed)
        {
            ImGui.TextDisabled("Minimap collapsed");
            return;
        }

        _host.MinimapAndStatus.DrawMinimapPanelContent();
    }

    private void DrawLeft()
    {
        _actionBars.DrawLeftBar();
    }

    private void DrawRight()
    {
        _actionBars.DrawRightBar();
    }

    private void DrawRightDock()
    {
        float scale = MathF.Max(1f, Layout.Scale);
        float grip = 6f * scale;
        Vector2 origin = ImGui.GetCursorPos();
        Vector2 available = ImGui.GetContentRegionAvail();

        // The panel body is a scrolling child to the right of the grip. The body is drawn first
        // and the splitter is submitted last as an overlay on the left edge: a full-height item
        // on the same line as the page selector made ImGui's line height the whole window, so
        // every widget after the selector was laid out below the window and the panels looked dead.
        ImGui.SetCursorPos(origin + new Vector2(grip, 0f));
        ImGui.BeginChild("##RightDockBody", new Vector2(MathF.Max(1f, available.X - grip), MathF.Max(1f, available.Y)), false);
        _host.WorkbenchPanels.DrawViewerContent(ViewerPanel);
        ImGui.EndChild();

        // A thin splitter on the left edge. Dragging it left widens the dock; the width is
        // persisted so the operator's choice survives a restart.
        ImGui.SetCursorPos(origin);
        ImGui.InvisibleButton("##RightDockSplitter", new Vector2(grip, MathF.Max(1f, available.Y)));
        bool gripActive = ImGui.IsItemActive();
        if (ImGui.IsItemHovered() || gripActive)
        {
            ImGui.SetMouseCursor(ImGuiMouseCursor.ResizeEW);
            ImGui.GetWindowDrawList().AddRectFilled(ImGui.GetItemRectMin(), ImGui.GetItemRectMax(),
                ImGui.ColorConvertFloat4ToU32(UiTokens.BrassDim));
        }
        if (gripActive)
            _rightDockWidth = Math.Clamp(_rightDockWidth - ImGui.GetIO().MouseDelta.X / scale, 240f, 720f);
        if (ImGui.IsItemDeactivated()) SaveLayout();
    }

    private void DrawBottom()
    {
        float scale = MathF.Max(1f, Layout.Scale);
        // Standard button height: the region is sized from the same measurement (UiLayout footer
        // row), so the row always fits. Tab hides every bar, so there are no bar-toggle buttons here.
        float height = ImGui.GetFrameHeight();

        // Measure every item first, then place it. The old code drew left-to-right and let the
        // window scroll, so a long label or a narrow client pushed Options off-screen.
        var items = new List<FooterItemMeasure>();
        foreach (WorkbenchTab destination in Enum.GetValues<WorkbenchTab>())
            items.Add(MeasureFooterButton($"preset-{destination}", DestinationLabel(destination), scale));
        items.Add(MeasureFooterButton("bottom-spawns", "Spawns", scale));
        items.Add(MeasureFooterButton("bottom-pages", "Pages", scale));
        items.Add(MeasureFooterButton("bottom-options", "Options", scale));

        FooterFitResult fit = ViewerFooterLayout.Fit(GetRegionRect(UiRegion.BottomBar), scale, items);
        float originX = ImGui.GetCursorPosX();
        float y = ImGui.GetCursorPosY();

        foreach (WorkbenchTab destination in Enum.GetValues<WorkbenchTab>())
        {
            FooterItemPlacement placement = fit.Get($"preset-{destination}");
            if (!placement.Visible) continue;
            ImGui.SetCursorPos(new Vector2(originX + placement.X, y));
            if (ImGui.Button($"{DestinationLabel(destination)}##Preset_{destination}", new Vector2(placement.Width, height)))
            {
                Navigate(destination);
                if (destination != WorkbenchTab.Viewer) _actionBars.OpenDestinationWindow();
            }
        }

        _actionBars.DrawBottomActions(fit, originX, y);

        FooterItemPlacement options = fit.Get("bottom-options");
        if (options.Visible)
        {
            ImGui.SetCursorPos(new Vector2(originX + options.X, y));
            if (ImGui.Button("Options##BottomOptions", new Vector2(options.Width, height))) ToggleOptions();
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Open display, rendering, and viewer options");
        }
    }

    /// <summary>A footer button sized to its label plus a small click margin, compacting to 80%.</summary>
    private static FooterItemMeasure MeasureFooterButton(string id, string label, float scale)
    {
        float natural = ImGui.CalcTextSize(label).X + ImGui.GetStyle().FramePadding.X * 2f + 6f * scale;
        return new FooterItemMeasure(id, natural, natural * 0.8f, FooterItemPriority.Essential);
    }

    private static string DestinationLabel(WorkbenchTab destination) => destination switch
    {
        WorkbenchTab.Viewer => "Viewer",
        WorkbenchTab.Editor => "Editor",
        _ => "Archaeology",
    };

    private void DrawRegion(UiRegion region, string name, Action content, ImGuiWindowFlags extra = 0,
        bool resizable = false, bool movable = false)
    {
        var state = Layout[region];
        if (!state.Visible || state.Rect.Width <= 0 || state.Rect.Height <= 0) return;
        ImGui.SetNextWindowPos(state.Rect.Position);
        ImGui.SetNextWindowSize(state.Rect.Size);
        // Every region shares one token surface, so the shell reads as a single
        // surface treatment instead of a set of unrelated windows.
        SharedUiWidgets.PushSurfaceStyle(Layout.Scale);
        bool actionRail = region is UiRegion.LeftSidebar or UiRegion.RightSidebar;
        // The footer bars are one button row tall. The shared 8 px window padding alone consumed
        // most of the status bar, so its button and text were clipped at the window edge.
        bool footer = region is UiRegion.BottomBar or UiRegion.StatusBar;
        if (actionRail)
            ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(3f, 3f) * Layout.Scale);
        else if (footer)
            ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(ViewerFooterLayout.RegionPadding,
                region == UiRegion.BottomBar ? UiLayout.BottomBarPadY : UiLayout.StatusBarPadY) * Layout.Scale);
        ImGuiWindowFlags flags = ImGuiWindowFlags.NoTitleBar |
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoDocking | extra;
        if (!resizable) flags |= ImGuiWindowFlags.NoResize;
        if (!movable) flags |= ImGuiWindowFlags.NoMove;
        if (ImGui.Begin(name, flags)) content();
        ImGui.End();
        if (actionRail || footer) ImGui.PopStyleVar();
        SharedUiWidgets.PopSurfaceStyle();
    }

    private void RegisterPopups()
    {
        if (_registered) return;
        _registered = true;
        _popups.Register("Settings", () => { _host.SettingsWindow.DrawSettingsContent(); return false; });
        _popups.Register(MapSpawnsPopupId, () =>
        {
            _host.SqlSpawnStreaming.DrawMapSpawnsContent();
            return ImGui.Button("Close##mapSpawns");
        }, size: new(760f, 600f));
        foreach (string id in new[] { MainMenuBarService.AboutPopupId, MainMenuBarService.KeyboardShortcutsPopupId, MainMenuBarService.ForgetBasePopupId })
            _popups.Register(id, () => _host.MainMenuBar.DrawPopupContent(id), () => _host.MainMenuBar.OnPopupClosed(id));
        _popups.Register(MainMenuBarService.UiReferencePopupId, () =>
        {
            _controlRegistry.DrawReferenceContent();
            return ImGui.Button("Close##uiReference");
        }, size: new(760f, 560f));
    }

    internal void RequestOpacity() => OpenOptions();

    internal void DrawPopupButton(string label, string summary, string popupId, Action content)
    {
        _popups.Register(popupId, () => { content(); return false; });
        if (ImGui.Button(label)) _popups.Request(popupId);
        if (ImGui.IsItemHovered()) ImGui.SetTooltip(summary);
    }

    private void DrawDiagnosticsWindows()
    {
        if (_host.ShowLogViewer)
        {
            SharedUiWidgets.PushSurfaceStyle();
            if (ImGui.Begin("Log", ref _host.ShowLogViewer, ImGuiWindowFlags.NoSavedSettings)) _host.DrawLogViewerContent();
            ImGui.End();
            SharedUiWidgets.PopSurfaceStyle();
        }
        if (_host.ShowPerfWindow)
        {
            SharedUiWidgets.PushSurfaceStyle();
            if (ImGui.Begin("Performance", ref _host.ShowPerfWindow, ImGuiWindowFlags.NoSavedSettings)) _host.PerfPanel.DrawPerfContent();
            ImGui.End();
            SharedUiWidgets.PopSurfaceStyle();
        }
    }
}
