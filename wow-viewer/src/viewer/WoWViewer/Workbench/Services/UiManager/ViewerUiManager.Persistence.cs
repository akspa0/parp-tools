using System.Text.Json;
using WoWViewer.Logging;
using WoWViewer.Workbench;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

internal sealed partial class ViewerUiManager
{
    // v3: DPI-scaled metrics, docked-by-default minimap, chat is a docked panel (no floating
    // rect). Older files are ignored so a stale floating chat rect cannot be restored.
    private const int LayoutVersion = 3;
    private static readonly string LayoutPath = Path.Combine(SettingsDir, "ui-layout-v3.json");
    private bool _layoutLoaded;
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
        public bool MinimapDocked { get; set; }
        public bool ChatCollapsed { get; set; }
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
            _minimapDocked = saved.MinimapDocked;
            _chatCollapsed = saved.ChatCollapsed;
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
        _minimapDocked = true;
        _chatCollapsed = false;
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
        _options = false;
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
                MinimapDocked = _minimapDocked, ChatCollapsed = _chatCollapsed,
            };
            string pending = LayoutPath + ".pending";
            File.WriteAllText(pending, JsonSerializer.Serialize(saved, new JsonSerializerOptions { WriteIndented = true }));
            File.Move(pending, LayoutPath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            ViewerLog.Trace($"[UI Layout] Save failed; previous layout retained: {ex.Message}");
        }
    }
}
