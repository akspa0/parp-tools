using System.Text.Json;
using WoWViewer.Logging;
using WoWViewer.Workbench;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

internal sealed partial class ViewerUiManager
{
    // v7 adds the docked left panel and unifies destinations into sidebars.
    private const int LayoutVersion = 7;
    private static readonly string LayoutPath = Path.Combine(SettingsDir, "ui-layout-v7.json");
    private static readonly JsonSerializerOptions LayoutJsonOptions = new() { WriteIndented = true };
    private bool _layoutLoaded;
    private string? _lastSavedLayoutJson;
    private readonly int[] _pages = [0, 0, 4];

    internal sealed class SavedLayout
    {
        public int Version { get; set; }
        public bool Left { get; set; }
        public bool Right { get; set; }
        public float LeftWidth { get; set; }
        public float RightWidth { get; set; }
        public WorkbenchTab Destination { get; set; }
        public ViewerPanel ViewerPanel { get; set; }
        public int[]? Pages { get; set; }
        public bool MinimapVisible { get; set; } = true;
        public bool MinimapCollapsed { get; set; }
        public bool ChatCollapsed { get; set; } = true;
        public bool RightDockOpen { get; set; }
        public float RightDockWidth { get; set; } = 360f;
        public bool LeftDockOpen { get; set; }
        public float LeftDockWidth { get; set; } = 360f;
        public LeftPanel LeftPanel { get; set; } = LeftPanel.Workspace;
    }

    internal void LoadLayout(float width)
    {
        _layoutLoaded = true;
        ApplyDefaults(width);
        // Deliberately do not read viewer_settings navigation, imgui.ini docking or old shell rects.
        // Those indices belong to deleted pages; only this versioned document can restore new state.
        try
        {
            if (!File.Exists(LayoutPath)) return;
            var saved = JsonSerializer.Deserialize<SavedLayout>(File.ReadAllText(LayoutPath));
            if (saved?.Version != LayoutVersion) return;
            _host.ShowLeftSidebar = saved.Left;
            _host.ShowRightSidebar = saved.Right;
            if (float.IsFinite(saved.LeftWidth)) _host.LeftSidebarWidth = Math.Clamp(saved.LeftWidth, 180, 1080);
            if (float.IsFinite(saved.RightWidth)) _host.RightSidebarWidth = Math.Clamp(saved.RightWidth, 180, 1080);
            _minimapVisible = saved.MinimapVisible;
            _minimapCollapsed = saved.MinimapCollapsed;
            _chatCollapsed = saved.ChatCollapsed;
            _rightDockOpen = saved.RightDockOpen;
            if (float.IsFinite(saved.RightDockWidth)) _rightDockWidth = Math.Clamp(saved.RightDockWidth, 240f, 720f);
            _leftDockOpen = saved.LeftDockOpen;
            if (float.IsFinite(saved.LeftDockWidth)) _leftDockWidth = Math.Clamp(saved.LeftDockWidth, 240f, 720f);
            ActiveLeftPanel = Enum.IsDefined(saved.LeftPanel) ? saved.LeftPanel : LeftPanel.Workspace;
            _host.ActiveTopTab = Enum.IsDefined(saved.Destination) ? saved.Destination : WorkbenchTab.Viewer;
            ViewerPanel = Enum.IsDefined(saved.ViewerPanel) ? saved.ViewerPanel : ViewerPanel.Inspector;
            if (saved.Pages is { Length: 3 })
                for (int i = 0; i < 3; i++) _pages[i] = Math.Clamp(saved.Pages[i], 0, Math.Max(0, WorkbenchNavigator.GetPageLabels((WorkbenchTab)i).Length - 1));
            _host.ActiveBottomTabIndex = _pages[(int)Destination];
            _host.WorkspaceMode = Destination switch
            {
                WorkbenchTab.Editor => WorkspaceMode.Editor,
                WorkbenchTab.Archaeology => WorkspaceMode.Archaeology,
                _ => WorkspaceMode.Viewer,
            };
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            ViewerLog.Trace($"[UI Layout] Ignoring unreadable layout: {ex.Message}");
        }
    }

    private void ApplyDefaults(float width)
    {
        var defaults = UiLayout.Defaults(width);
        _host.ShowLeftSidebar = defaults.Left;
        _host.ShowRightSidebar = defaults.Right;
        _host.LeftSidebarWidth = defaults.LeftWidth;
        _host.RightSidebarWidth = defaults.RightWidth;
        _minimapVisible = true;
        _minimapCollapsed = false;
        _chatCollapsed = true;
        _rightDockOpen = false;
        _rightDockWidth = 360f;
        _leftDockOpen = false;
        _leftDockWidth = 360f;
        ActiveLeftPanel = LeftPanel.Workspace;
        _host.ActiveTopTab = WorkbenchTab.Viewer;
        _host.ActiveBottomTabIndex = 0;
        _host.ActivePm4TabIndex = 0;
        _host.WorkspaceMode = WorkspaceMode.Viewer;
        ViewerPanel = ViewerPanel.Inspector;
        _pages[0] = 0; _pages[1] = 0; _pages[2] = 4;
    }

    internal void ResetLayout()
    {
        ApplyDefaults(HasImGuiContext() ? ImGuiNET.ImGui.GetIO().DisplaySize.X : 1600);
        _actionBars.CloseAll();
        SaveLayout();
    }

    internal void SaveLayout()
    {
        if (!_layoutLoaded) return;
        try
        {
            Directory.CreateDirectory(SettingsDir);
            _pages[(int)Destination] = _host.ActiveBottomTabIndex;
            var saved = new SavedLayout
            {
                Version = LayoutVersion, Left = _host.ShowLeftSidebar, Right = _host.ShowRightSidebar,
                LeftWidth = _host.LeftSidebarWidth, RightWidth = _host.RightSidebarWidth,
                Destination = Destination, ViewerPanel = ViewerPanel, Pages = _pages,
                MinimapVisible = _minimapVisible, MinimapCollapsed = _minimapCollapsed,
                ChatCollapsed = _chatCollapsed,
                RightDockOpen = _rightDockOpen, RightDockWidth = _rightDockWidth,
                LeftDockOpen = _leftDockOpen, LeftDockWidth = _leftDockWidth,
                LeftPanel = ActiveLeftPanel,
            };
            string json = JsonSerializer.Serialize(saved, LayoutJsonOptions);
            // Per-frame callers (e.g. the diagnostics redirect) must not hit disk every frame:
            // skip the write when the document is byte-identical to the last one saved.
            if (json == _lastSavedLayoutJson) return;
            string pending = LayoutPath + ".pending";
            File.WriteAllText(pending, json);
            File.Move(pending, LayoutPath, overwrite: true);
            _lastSavedLayoutJson = json;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            ViewerLog.Trace($"[UI Layout] Save failed; previous layout retained: {ex.Message}");
        }
    }
}
