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
    private readonly UiPopupRegistry _popups = new();
    private readonly Dictionary<UiRegion, UiRegionState> _hosted = new();
    private bool _registered;
    private bool _options;
    private bool _focusRight;
    private bool _minimapVisible = true;
    private bool _minimapCollapsed;
    // Docked at the top of the right sidebar is the default home; Undock floats it.
    private bool _minimapDocked = true;
    private bool _chatCollapsed;
    private float _appliedFontScale = -1f;
    private float _appliedStyleScale = -1f;
    internal ViewerUiManager(IViewerAppHost host)
    {
        _host = host;
        // Resolve the provider at draw time: the manager is constructed before the
        // minimap/status service, so capturing the delegate here would bind a null target.
        StatusContent = () => host.MinimapAndStatus.DrawStatusContent();
    }
    internal UiLayout Layout { get; private set; } = new(Vector2.Zero, false, true, true, 280, 360);
    internal UiPopupRegistry Popups => _popups;
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
        float fontScale = Math.Clamp(_host.UiFontScale * scale, 0.75f, 4f);
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
            _host.LeftSidebarWidth, _host.RightSidebarWidth, _options,
            _hosted.TryGetValue(UiRegion.Chat, out var chat) && chat.Visible ? chat.Rect : null,
            _minimapVisible, _minimapCollapsed, _minimapDocked, _host.SqlSpawnStreaming.ChatVisible,
            _chatCollapsed, scale);
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
        _focusRight = true;
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
    }

    internal void OpenSceneTools(int page = 0)
    {
        SceneToolsPage = Math.Clamp(page, 0, 1);
        SceneToolsVisible = true;
        Navigate(WorkbenchTab.Viewer);
        _host.ShowLeftSidebar = true;
    }

    internal void OpenOptions() => _options = true;
    internal void ToggleOptions() => _options = !_options;
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

    internal bool MinimapDocked
    {
        get => _minimapDocked;
        set { _minimapDocked = value; SaveLayout(); }
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
    internal bool IsCaptureVisible => !_host.HideUiChrome && _host.ShowLeftSidebar && Destination == WorkbenchTab.Viewer && SceneToolsVisible && SceneToolsPage == 1;
    internal bool IsAudioVisible => !_host.HideUiChrome && _options;

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
            if (_focusRight) ImGui.SetNextWindowFocus();
            DrawRegion(UiRegion.RightSidebar, "##UiRightSidebar", DrawRight);
            _focusRight = false;
            DrawRegion(UiRegion.Minimap, "Minimap##UiMinimap", DrawMinimap,
                ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse);
            // Docked panel: no resize and no move. A floating chat could be dragged over the
            // sidebars and had no reliable way back.
            DrawRegion(UiRegion.Chat, "Chat##UiChat", DrawChat, ImGuiWindowFlags.NoScrollbar);
            DrawRegion(UiRegion.BottomBar, "##UiBottomBar", DrawBottom, ImGuiWindowFlags.HorizontalScrollbar);
            if (_options) DrawRegion(UiRegion.Options, "Options##UiOptions", () =>
            {
                if (ImGui.SmallButton("Close Options")) _options = false;
                _host.ViewerChrome.DrawOptionsContent();
            });
            DrawDiagnosticsWindows();
        }
        DrawRegion(UiRegion.StatusBar, "##UiStatusBar", () =>
        {
            if (StatusContent != null) StatusContent();
            else ImGui.TextUnformatted(_host.StatusMessage);
        });
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

        // Two columns: the chat log on the left, contextual tools on the right. The right side of
        // this panel was dead whitespace; it now carries the actions the operator reaches for
        // while reading the world, and the barker section appears only when the local area
        // actually has creatures with ambient barks.
        float scale = MathF.Max(1f, Layout.Scale);
        float available = ImGui.GetContentRegionAvail().X;
        float toolsWidth = MathF.Min(220f * scale, available * 0.35f);
        float chatWidth = MathF.Max(0f, available - toolsWidth - 8f * scale);

        if (ImGui.BeginChild("##ChatBody", new Vector2(chatWidth, 0f), false))
        {
            _host.SqlSpawnStreaming.DrawChatContent();
            ImGui.EndChild();
        }

        ImGui.SameLine();

        if (ImGui.BeginChild("##ChatTools", new Vector2(0f, 0f), false))
        {
            DrawChatTools();
            ImGui.EndChild();
        }
    }

    private void DrawChatTools()
    {
        SharedUiWidgets.SectionTitle("Tools");

        if (ImGui.SmallButton("Map Spawns##ChatMapSpawns"))
            _popups.Request(MapSpawnsPopupId);
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("Open the current map's spawn list and population controls");

        if (ImGui.SmallButton("Options##ChatToolsOptions"))
            _options = !_options;
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("Open the options sub-menu");

        if (ImGui.SmallButton("Minimap##ChatToolsMinimap"))
            ToggleMinimap();
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("Show or hide the minimap (M)");

        if (ImGui.SmallButton("Inspector##ChatToolsInspector"))
            OpenViewer(ViewerPanel.Inspector);
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("Open the Inspector in the right sidebar");

        if (ImGui.SmallButton("PM4##ChatToolsPm4"))
            OpenViewer(ViewerPanel.PM4);
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("Open the PM4 workbench in the right sidebar");

        if (ImGui.SmallButton("Collapse##ChatToolsCollapse"))
        {
            _chatCollapsed = true;
            SaveLayout();
        }
        if (ImGui.IsItemHovered()) ImGui.SetTooltip("Collapse the chat to a one-line bar");

        // Barkers only when the current map is loaded and the local spawn set has creatures with
        // ambient barks. An empty section on every map would be noise.
        if (_host.SqlSpawnStreaming.HasLocalBarkers)
        {
            ImGui.Separator();
            SharedUiWidgets.SectionTitle("Barkers");
            SharedUiWidgets.CompactStatus("Ambient NPC barks are active in this area.");
            if (ImGui.SmallButton("Clear chat##ChatToolsClearBarkers"))
                _host.SqlSpawnStreaming.ClearChat();
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("Clear the chat log");
        }
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
        if (_host.ShowLeftSidebar)
        {
            if (ImGui.SmallButton("Collapse left sidebar##CollapseLeft"))
            {
                _host.ShowLeftSidebar = false;
                SaveLayout();
            }
        }
        else
        {
            if (ImGui.SmallButton("Expand left sidebar##ExpandLeft"))
            {
                _host.ShowLeftSidebar = true;
                SaveLayout();
            }
        }
        if (_host.ShowLeftSidebar) _host.NavigatorPanel.DrawContent();
    }

    private void DrawRight()
    {
        if (_host.ShowRightSidebar)
        {
            if (ImGui.SmallButton("Collapse right sidebar##CollapseRight"))
            {
                _host.ShowRightSidebar = false;
                SaveLayout();
            }
        }
        else
        {
            if (ImGui.SmallButton("Expand right sidebar##ExpandRight"))
            {
                _host.ShowRightSidebar = true;
                SaveLayout();
            }
        }
        if (!_host.ShowRightSidebar) return;
        if (Destination == WorkbenchTab.Viewer)
        {
            if (ImGui.BeginTabBar("##ViewerPanels"))
            {
                foreach (ViewerPanel panel in Enum.GetValues<ViewerPanel>())
                {
                    var flags = _focusRight && panel == ViewerPanel ? ImGuiTabItemFlags.SetSelected : ImGuiTabItemFlags.None;
                    if (BeginPanelTab(panel.ToString(), flags))
                    {
                        ViewerPanel = panel;
                        _host.WorkbenchPanels.DrawViewerContent(panel);
                        ImGui.EndTabItem();
                    }
                }
                ImGui.EndTabBar();
            }
        }
        else
        {
            int page = WorkbenchPanelsService.DrawPageCombo("##DestinationPage", WorkbenchNavigator.GetPageLabels(Destination), _host.ActiveBottomTabIndex);
            if (page != _host.ActiveBottomTabIndex) { _host.ActiveBottomTabIndex = page; SaveLayout(); }
            _host.WorkbenchPanels.DrawDestinationContent(Destination);
        }
    }

    private void DrawBottom()
    {
        foreach (WorkbenchTab destination in Enum.GetValues<WorkbenchTab>())
        {
            if (ImGui.Button(destination.ToString())) Navigate(destination);
            ImGui.SameLine();
        }
        if (ImGui.Button("Options")) _options = !_options;
        ImGui.SameLine();
        _host.ViewerChrome.DrawStatsContent();
    }

    private static unsafe bool BeginPanelTab(string label, ImGuiTabItemFlags flags)
    {
        byte[] text = System.Text.Encoding.UTF8.GetBytes(label + "\0");
        fixed (byte* pointer = text)
            return ImGuiNative.igBeginTabItem(pointer, null, flags) != 0;
    }

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
        ImGuiWindowFlags flags = ImGuiWindowFlags.NoTitleBar |
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoDocking | extra;
        if (!resizable) flags |= ImGuiWindowFlags.NoResize;
        if (!movable) flags |= ImGuiWindowFlags.NoMove;
        if (ImGui.Begin(name, flags)) content();
        ImGui.End();
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
