using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Maps;
using WoWViewer.Terrain.Vlm;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Shell layout: dockable panel state, panel focus/requests, saved/default panel rects, sidebar/drawer insets, scene viewport rects and ImGui mouse routing.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class ShellLayoutService
{
    private readonly IViewerAppHost _host;

    internal ShellLayoutService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref FixedBottomDrawerTab _activeBottomDrawerTab => ref _host.ActiveBottomDrawerTab;
    private ref float _bottomDrawerHeight => ref _host.BottomDrawerHeight;
    private ref Vector2 _dockspaceHostPosition => ref _host.DockspaceHostPosition;
    private ref Vector2 _dockspaceHostSize => ref _host.DockspaceHostSize;
    private ref bool _forceApplyShellPanelLayout => ref _host.ForceApplyShellPanelLayout;
    private ref bool _fullscreenMinimap => ref _host.FullscreenMinimap;
    private ref bool _hideUiChrome => ref _host.HideUiChrome;
    private ref ImGuiController _imGui => ref _host.ImGui;
    private ref float _leftSidebarWidth => ref _host.LeftSidebarWidth;
    private ref string _modelInfo => ref _host.ModelInfo;
    private ref ShellPanelId? _pendingFocusedShellPanel => ref _host.PendingFocusedShellPanel;
    private ref FixedBottomDrawerTab? _pendingRightSidebarSection => ref _host.PendingRightSidebarSection;
    private HashSet<ShellPanelId> _pendingShellPanelLayoutRestore => _host.PendingShellPanelLayoutRestore;
    private ref float _rightSidebarWidth => ref _host.RightSidebarWidth;
    private Dictionary<ShellPanelId, SavedShellPanelLayout> _savedShellPanelLayouts => _host.SavedShellPanelLayouts;
    private ref bool _showLeftSidebar => ref _host.ShowLeftSidebar;
    private ref bool _showMinimapWindow => ref _host.ShowMinimapWindow;
    private ref bool _showModelInfo => ref _host.ShowModelInfo;
    private ref bool _showRightSidebar => ref _host.ShowRightSidebar;
    private ref bool _showWorkspaceBarsPanel => ref _host.ShowWorkspaceBarsPanel;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref bool _useDockspaceUi => ref _host.UseDockspaceUi;
    private ref bool _useTabUi => ref _host.UseTabUi;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref IWindow _window => ref _host.Window;
    private ref WorkspaceMode _workspaceMode => ref _host.WorkspaceMode;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private float ClampFixedSidebarWidth(float width, bool isLeftSidebar, float displayWidth) => _host.ClampFixedSidebarWidth(width, isLeftSidebar, displayWidth);
    private float GetTopChromeHeight() => _host.GetTopChromeHeight();
    private void SaveViewerSettings() => _host.SaveViewerSettings();
    private void SetEditorWorkspaceTask(EditorWorkspaceTask task) => _host.SetEditorWorkspaceTask(task);

    private static readonly MethodInfo? ImGuiControllerWindowResizedMethod =
        typeof(ImGuiController).GetMethod("WindowResized", BindingFlags.Instance | BindingFlags.NonPublic);
    private readonly Lock _pendingImGuiMouseEventLock = new();
    private readonly Queue<(int ButtonIndex, bool Down)> _pendingImGuiMouseButtonEvents = new();
    private Vector2D<int> _lastSyncedImGuiWindowSize;
    private Vector2D<int> _lastSyncedImGuiFramebufferSize;
    private bool _showTerrainControls = false;

    private struct DockPanelState
    {
        public bool Visible;
        public bool IsDocked;
        public Vector2 Position;
        public Vector2 Size;
    }

    private static readonly ShellPanelId[] TopLeftQuadrantPanels = { ShellPanelId.Navigator };
    private static readonly ShellPanelId[] TopRightQuadrantPanels = { ShellPanelId.Inspector, ShellPanelId.WorldObjects, ShellPanelId.ModelInfo, ShellPanelId.RuntimeStats };
    private static readonly ShellPanelId[] BottomRightQuadrantPanels = { ShellPanelId.Pm4Workbench, ShellPanelId.Pm4Info, ShellPanelId.TerrainControls, ShellPanelId.Pm4SceneGraph };
    private static readonly ShellPanelId[] BottomLeftQuadrantPanels = { ShellPanelId.Minimap };

    private DockPanelState _navigatorDockState;
    private DockPanelState _inspectorDockState;
    private DockPanelState _pm4WorkbenchDockState;
    private DockPanelState _terrainControlsDockState;
    private DockPanelState _runtimeStatsDockState;
    private DockPanelState _worldObjectsDockState;
    private DockPanelState _modelInfoDockState;
    private DockPanelState _minimapDockState;
    private DockPanelState _workspaceBarsDockState;
    private DockPanelState _pm4InfoDockState;
    private DockPanelState _pm4SceneGraphDockState;
    private const float BottomDrawerMinHeight = 220f;
    private const float BottomDrawerCompactMinHeight = 160f;
    private const float BottomDrawerMaxHeight = 520f;
    private const float SceneViewportPreferredMinHeight = 280f;
    private const float SceneViewportHardMinHeight = 160f;
    private bool _suppressLeftSidebarForLayout;
    private bool _suppressRightSidebarForLayout;
    private bool _suppressMinimapForLayout;
    private bool _showPm4SceneGraph = true;

    private bool IsPointInSceneViewport(float x, float y)
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            if (!IsShellPanelActive(panel.Id))
                continue;

            if (IsPointInVisibleShellPanel(GetDockPanelStateRef(panel.Id), x, y))
                return false;
        }

        if (!TryGetSceneViewportRect(out float vpX, out float vpY, out float vpW, out float vpH))
            return false;
        return x >= vpX && x <= vpX + vpW && y >= vpY && y <= vpY + vpH;
    }

    internal bool CanSceneConsumeMouse(float x, float y)
    {
        return IsPointInSceneViewport(x, y) && !IsSceneMouseCaptureBlocked(x, y);
    }

    internal bool IsSceneMouseCaptureBlocked(float x, float y)
    {
        if (!ImGui.GetIO().WantCaptureMouse)
            return false;

        return !ShouldBypassDockspaceMouseCapture(x, y);
    }

    private bool ShouldBypassDockspaceMouseCapture(float x, float y)
    {
        return _useDockspaceUi
            && _dockspaceHostSize.X > 10f
            && _dockspaceHostSize.Y > 10f
            && IsPointInSceneViewport(x, y);
    }

    private static bool IsPointInVisibleShellPanel(in DockPanelState state, float x, float y)
    {
        if (!state.Visible || state.Size.X <= 1f || state.Size.Y <= 1f)
            return false;

        return x >= state.Position.X
            && x <= state.Position.X + state.Size.X
            && y >= state.Position.Y
            && y <= state.Position.Y + state.Size.Y;
    }

    internal void QueueImGuiMouseButtonEvent(MouseButton button, bool down)
    {
        int? buttonIndex = button switch
        {
            MouseButton.Left => 0,
            MouseButton.Right => 1,
            MouseButton.Middle => 2,
            _ => null,
        };

        if (!buttonIndex.HasValue)
            return;

        lock (_pendingImGuiMouseEventLock)
        {
            _pendingImGuiMouseButtonEvents.Enqueue((buttonIndex.Value, down));
        }
    }

    internal void FlushPendingImGuiMouseButtonEvents()
    {
        lock (_pendingImGuiMouseEventLock)
        {
            if (_pendingImGuiMouseButtonEvents.Count == 0)
                return;

            var io = ImGui.GetIO();
            while (_pendingImGuiMouseButtonEvents.Count > 0)
            {
                var (buttonIndex, down) = _pendingImGuiMouseButtonEvents.Dequeue();
                io.AddMouseButtonEvent(buttonIndex, down);
            }
        }
    }

    internal static ShellPanelDefinition GetShellPanelDefinition(ShellPanelId panelId)
    {
        return ShellPanelDefinitions[(int)panelId];
    }

    private ref DockPanelState GetDockPanelStateRef(ShellPanelId panelId)
    {
        switch (panelId)
        {
            case ShellPanelId.Navigator:
                return ref _navigatorDockState;
            case ShellPanelId.Inspector:
                return ref _inspectorDockState;
            case ShellPanelId.Pm4Workbench:
                return ref _pm4WorkbenchDockState;
            case ShellPanelId.TerrainControls:
                return ref _terrainControlsDockState;
            case ShellPanelId.RuntimeStats:
                return ref _runtimeStatsDockState;
            case ShellPanelId.WorldObjects:
                return ref _worldObjectsDockState;
            case ShellPanelId.ModelInfo:
                return ref _modelInfoDockState;
            case ShellPanelId.Minimap:
                return ref _minimapDockState;
            case ShellPanelId.WorkspaceBars:
                return ref _workspaceBarsDockState;
            case ShellPanelId.Pm4Info:
                return ref _pm4InfoDockState;
            case ShellPanelId.Pm4SceneGraph:
                return ref _pm4SceneGraphDockState;
            default:
                throw new ArgumentOutOfRangeException(nameof(panelId), panelId, null);
        }
    }

    private bool IsShellPanelRequested(ShellPanelId panelId)
    {
        return panelId switch
        {
            ShellPanelId.Navigator => _showLeftSidebar,
            ShellPanelId.Inspector => _showRightSidebar,
            ShellPanelId.Pm4Workbench => _showRightSidebar && _worldScene != null,
            ShellPanelId.TerrainControls => _showRightSidebar && _showTerrainControls && (_terrainManager != null || _vlmTerrainManager != null),
            ShellPanelId.RuntimeStats => _showRightSidebar && (_terrainManager != null || _vlmTerrainManager != null || _worldScene != null),
            ShellPanelId.WorldObjects => _showRightSidebar && _worldScene != null,
            ShellPanelId.ModelInfo => _showRightSidebar && _showModelInfo && !string.IsNullOrWhiteSpace(_modelInfo),
            ShellPanelId.Minimap => _showMinimapWindow,
            ShellPanelId.WorkspaceBars => false,
            ShellPanelId.Pm4Info => _showRightSidebar && _worldScene != null,
            ShellPanelId.Pm4SceneGraph => _showPm4SceneGraph && _worldScene != null,
            _ => false,
        };
    }

    private bool IsShellPanelSuppressedForLayout(ShellPanelId panelId)
    {
        return panelId switch
        {
            ShellPanelId.Navigator => _suppressLeftSidebarForLayout,
            ShellPanelId.Inspector => _suppressRightSidebarForLayout,
            ShellPanelId.Minimap => _suppressMinimapForLayout,
            ShellPanelId.WorkspaceBars => _suppressLeftSidebarForLayout,
            ShellPanelId.Pm4Info => _suppressRightSidebarForLayout,
            ShellPanelId.Pm4SceneGraph => _suppressRightSidebarForLayout,
            _ => false,
        };
    }

    internal bool IsShellPanelActive(ShellPanelId panelId)
    {
        return IsShellPanelRequested(panelId) && !IsShellPanelSuppressedForLayout(panelId);
    }

    internal bool HasAnyShellPanelsInLane(ShellPanelLane lane)
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            if (panel.Lane == lane && IsShellPanelActive(panel.Id))
                return true;
        }

        return false;
    }

    internal void FocusShellPanel(ShellPanelId panelId)
    {
        if (!_useDockspaceUi)
        {
            switch (panelId)
            {
                case ShellPanelId.Navigator:
                    _showLeftSidebar = true;
                    return;
                case ShellPanelId.Inspector:
                    _showRightSidebar = true;
                    return;
                case ShellPanelId.WorkspaceBars:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Workspace;
                    return;
                case ShellPanelId.Pm4Workbench:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Pm4;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Pm4;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Pm4Evidence);
                    return;
                case ShellPanelId.TerrainControls:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Terrain;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Terrain;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Terrain);
                    return;
                case ShellPanelId.WorldObjects:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.World;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.World;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Objects);
                    return;
                case ShellPanelId.RuntimeStats:
                case ShellPanelId.ModelInfo:
                    _showRightSidebar = true;
                    _activeBottomDrawerTab = FixedBottomDrawerTab.Diagnostics;
                    _pendingRightSidebarSection = FixedBottomDrawerTab.Diagnostics;
                    if (_workspaceMode == WorkspaceMode.Editor)
                        SetEditorWorkspaceTask(EditorWorkspaceTask.Inspect);
                    return;
                case ShellPanelId.Minimap:
                    _showMinimapWindow = true;
                    return;
                case ShellPanelId.Pm4Info:
                    _showRightSidebar = true;
                    return;
                case ShellPanelId.Pm4SceneGraph:
                    _showPm4SceneGraph = true;
                    return;
            }
        }

        if (panelId == ShellPanelId.WorkspaceBars)
        {
            _showWorkspaceBarsPanel = true;
            _pendingFocusedShellPanel = panelId;
            return;
        }

        switch (GetShellPanelDefinition(panelId).Lane)
        {
            case ShellPanelLane.Left:
                _showLeftSidebar = true;
                break;
            case ShellPanelLane.Right:
                _showRightSidebar = true;
                break;
            case ShellPanelLane.Floating:
                if (panelId == ShellPanelId.Minimap)
                    _showMinimapWindow = true;
                break;
        }

        _pendingFocusedShellPanel = panelId;
    }

    internal void ResetDockPanelStates()
    {
        foreach (var panel in ShellPanelDefinitions)
        {
            ref DockPanelState state = ref GetDockPanelStateRef(panel.Id);
            state = default;
        }
    }

    internal void ResetShellLayoutToDefaults()
    {
        _savedShellPanelLayouts.Clear();
        _pendingShellPanelLayoutRestore.Clear();
        _showLeftSidebar = true;
        _showRightSidebar = true;
        _showTerrainControls = false;
        _leftSidebarWidth = DefaultSidebarWidth;
        _rightSidebarWidth = DefaultRightSidebarWidth;
        _bottomDrawerHeight = DefaultBottomDrawerHeight;
        _activeBottomDrawerTab = FixedBottomDrawerTab.Workspace;
        _useDockspaceUi = true;
        _showPm4SceneGraph = true;
        _forceApplyShellPanelLayout = true;
        SaveViewerSettings();
    }

    internal void CaptureDockPanelState(ShellPanelId panelId)
    {
        ref DockPanelState state = ref GetDockPanelStateRef(panelId);
        state.Visible = true;
        state.IsDocked = ImGui.IsWindowDocked();
        state.Position = ImGui.GetWindowPos();
        state.Size = ImGui.GetWindowSize();

        CaptureSavedShellPanelLayout(panelId, state);
    }

    private void CaptureSavedShellPanelLayout(ShellPanelId panelId, in DockPanelState state)
    {
        if (!_useDockspaceUi || !state.Visible || state.Size.X <= 1f || state.Size.Y <= 1f)
            return;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 size))
            return;

        float normalizedWidth = Math.Clamp(state.Size.X / Math.Max(size.X, 1f), 0.12f, 1f);
        float normalizedHeight = Math.Clamp(state.Size.Y / Math.Max(size.Y, 1f), 0.12f, 1f);
        float normalizedX = Math.Clamp((state.Position.X - origin.X) / Math.Max(size.X, 1f), 0f, 1f - normalizedWidth);
        float normalizedY = Math.Clamp((state.Position.Y - origin.Y) / Math.Max(size.Y, 1f), 0f, 1f - normalizedHeight);

        _savedShellPanelLayouts[panelId] = new SavedShellPanelLayout
        {
            PanelId = (int)panelId,
            NormalizedX = normalizedX,
            NormalizedY = normalizedY,
            NormalizedWidth = normalizedWidth,
            NormalizedHeight = normalizedHeight,
        };
    }

    internal void PrepareDockableShellPanelWindow(ShellPanelId panelId, Vector2 defaultSize, Vector2 minSize, Vector2 maxSize)
    {
        if (!_useDockspaceUi)
        {
            ImGui.SetNextWindowSize(defaultSize, ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowSizeConstraints(minSize, maxSize);
            return;
        }

        bool shouldForceLayout = _forceApplyShellPanelLayout || _pendingShellPanelLayoutRestore.Contains(panelId);
        if (TryResolveShellPanelRect(panelId, minSize, maxSize, out Vector2 position, out Vector2 size))
        {
            ImGuiCond cond = shouldForceLayout ? ImGuiCond.Always : ImGuiCond.Appearing;
            ImGui.SetNextWindowPos(position, cond);
            ImGui.SetNextWindowSize(size, cond);

            if (shouldForceLayout)
                _pendingShellPanelLayoutRestore.Remove(panelId);
        }
        else
        {
            ImGui.SetNextWindowSize(defaultSize, ImGuiCond.FirstUseEver);
        }

        ImGui.SetNextWindowSizeConstraints(minSize, maxSize);
    }

    private bool TryResolveShellPanelRect(ShellPanelId panelId, Vector2 minSize, Vector2 maxSize, out Vector2 position, out Vector2 size)
    {
        if (TryGetSavedShellPanelRect(panelId, minSize, maxSize, out position, out size))
            return true;

        return TryGetDefaultShellPanelRect(panelId, minSize, maxSize, out position, out size);
    }

    private bool TryGetSavedShellPanelRect(ShellPanelId panelId, Vector2 minSize, Vector2 maxSize, out Vector2 position, out Vector2 size)
    {
        position = Vector2.Zero;
        size = Vector2.Zero;

        if (!_savedShellPanelLayouts.TryGetValue(panelId, out SavedShellPanelLayout? savedLayout))
            return false;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 hostSize))
            return false;

        size = new Vector2(
            hostSize.X * savedLayout.NormalizedWidth,
            hostSize.Y * savedLayout.NormalizedHeight);
        position = new Vector2(
            origin.X + hostSize.X * savedLayout.NormalizedX,
            origin.Y + hostSize.Y * savedLayout.NormalizedY);

        ClampShellPanelRect(origin, hostSize, minSize, maxSize, ref position, ref size);
        return true;
    }

    private bool TryGetDefaultShellPanelRect(ShellPanelId panelId, Vector2 minSize, Vector2 maxSize, out Vector2 position, out Vector2 size)
    {
        position = Vector2.Zero;
        size = Vector2.Zero;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 hostSize))
            return false;

        ShellPanelId[] group = GetDefaultShellPanelGroup(panelId);
        int activeCount = 0;
        int panelIndex = -1;
        for (int i = 0; i < group.Length; i++)
        {
            if (!IsShellPanelActive(group[i]))
                continue;

            if (group[i] == panelId)
                panelIndex = activeCount;

            activeCount++;
        }

        if (activeCount == 0 || panelIndex < 0)
            return false;

        const float padding = 12f;
        const float gap = 10f;
        float columnWidth = Math.Clamp(hostSize.X * 0.26f, 280f, 420f);
        float quadrantHeight = Math.Max(220f, (hostSize.Y - padding * 2f - gap) * 0.5f);
        float leftX = origin.X + padding;
        float rightX = Math.Max(leftX + gap, origin.X + hostSize.X - columnWidth - padding);
        float topY = origin.Y + padding;
        float bottomY = origin.Y + hostSize.Y - quadrantHeight - padding;

        bool isLeftQuadrant = panelId == ShellPanelId.Navigator
            || panelId == ShellPanelId.Inspector
            || panelId == ShellPanelId.Pm4Workbench
            || panelId == ShellPanelId.Minimap;
        bool isTopQuadrant = panelId == ShellPanelId.Navigator
            || panelId == ShellPanelId.Inspector
            || panelId == ShellPanelId.RuntimeStats
            || panelId == ShellPanelId.ModelInfo;

        float groupX = isLeftQuadrant ? leftX : rightX;
        float groupY = isTopQuadrant ? topY : bottomY;
        float slotHeight = (quadrantHeight - gap * Math.Max(0, activeCount - 1)) / activeCount;
        position = new Vector2(groupX, groupY + panelIndex * (slotHeight + gap));
        size = new Vector2(columnWidth, slotHeight);

        if (panelId == ShellPanelId.Minimap)
        {
            float squareSize = MathF.Min(size.X, size.Y);
            size = new Vector2(squareSize, squareSize);
        }

        ClampShellPanelRect(origin, hostSize, minSize, maxSize, ref position, ref size);
        return true;
    }

    private static ShellPanelId[] GetDefaultShellPanelGroup(ShellPanelId panelId)
    {
        return panelId switch
        {
            ShellPanelId.Navigator => TopLeftQuadrantPanels,
            ShellPanelId.Inspector or ShellPanelId.WorldObjects or ShellPanelId.ModelInfo or ShellPanelId.RuntimeStats => TopRightQuadrantPanels,
            ShellPanelId.Pm4Workbench or ShellPanelId.Pm4Info or ShellPanelId.TerrainControls => BottomRightQuadrantPanels,
            ShellPanelId.Minimap => BottomLeftQuadrantPanels,
            _ => TopRightQuadrantPanels,
        };
    }

    private bool TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 size)
    {
        var io = ImGui.GetIO();
        float topOffset = GetTopChromeHeight();
        float height = io.DisplaySize.Y - topOffset - StatusBarHeight;

        if (_useDockspaceUi && _dockspaceHostSize.X > 10f && _dockspaceHostSize.Y > 10f)
        {
            origin = _dockspaceHostPosition;
            size = _dockspaceHostSize;
            return true;
        }

        origin = new Vector2(0f, topOffset);
        size = new Vector2(io.DisplaySize.X, MathF.Max(0f, height));
        return size.X > 10f && size.Y > 10f;
    }

    private static void ClampShellPanelRect(Vector2 origin, Vector2 hostSize, Vector2 minSize, Vector2 maxSize, ref Vector2 position, ref Vector2 size)
    {
        float clampedWidth = Math.Clamp(size.X, minSize.X, Math.Min(maxSize.X, hostSize.X));
        float clampedHeight = Math.Clamp(size.Y, minSize.Y, Math.Min(maxSize.Y, hostSize.Y));
        size = new Vector2(clampedWidth, clampedHeight);

        float maxX = Math.Max(origin.X, origin.X + hostSize.X - size.X);
        float maxY = Math.Max(origin.Y, origin.Y + hostSize.Y - size.Y);
        position = new Vector2(
            Math.Clamp(position.X, origin.X, maxX),
            Math.Clamp(position.Y, origin.Y, maxY));
    }

    private bool TryGetDockedShellPanelState(ShellPanelLane lane, out DockPanelState state)
    {
        bool found = false;
        state = default;

        foreach (var panel in ShellPanelDefinitions)
        {
            if (panel.Lane != lane || !IsShellPanelActive(panel.Id))
                continue;

            ref DockPanelState panelState = ref GetDockPanelStateRef(panel.Id);
            if (!panelState.Visible || !panelState.IsDocked)
                continue;

            if (!found)
            {
                state = panelState;
                found = true;
                continue;
            }

            float left = MathF.Min(state.Position.X, panelState.Position.X);
            float top = MathF.Min(state.Position.Y, panelState.Position.Y);
            float right = MathF.Max(state.Position.X + state.Size.X, panelState.Position.X + panelState.Size.X);
            float bottom = MathF.Max(state.Position.Y + state.Size.Y, panelState.Position.Y + panelState.Size.Y);

            state.Visible = true;
            state.IsDocked = true;
            state.Position = new Vector2(left, top);
            state.Size = new Vector2(right - left, bottom - top);
        }

        if (found)
            return true;

        return false;
    }

    private bool TryGetVisibleShellPanelInsetState(bool isLeftPanel, out DockPanelState state)
    {
        state = default;

        if (!TryGetDockableShellLayoutRect(out Vector2 origin, out Vector2 hostSize))
            return false;

        bool found = false;
        float hostLeft = origin.X;
        float hostRight = origin.X + hostSize.X;
        const float edgeTolerance = 24f;

        foreach (var panel in ShellPanelDefinitions)
        {
            if (!IsShellPanelActive(panel.Id))
                continue;

            ref DockPanelState panelState = ref GetDockPanelStateRef(panel.Id);
            if (!panelState.Visible || panelState.Size.X <= 1f || panelState.Size.Y <= 1f)
                continue;

            bool touchesEdge = isLeftPanel
                ? panelState.Position.X <= hostLeft + edgeTolerance
                : panelState.Position.X + panelState.Size.X >= hostRight - edgeTolerance;
            if (!touchesEdge)
                continue;

            if (!found)
            {
                state = panelState;
                found = true;
                continue;
            }

            float left = MathF.Min(state.Position.X, panelState.Position.X);
            float top = MathF.Min(state.Position.Y, panelState.Position.Y);
            float right = MathF.Max(state.Position.X + state.Size.X, panelState.Position.X + panelState.Size.X);
            float bottom = MathF.Max(state.Position.Y + state.Size.Y, panelState.Position.Y + panelState.Size.Y);

            state.Visible = true;
            state.IsDocked = state.IsDocked || panelState.IsDocked;
            state.Position = new Vector2(left, top);
            state.Size = new Vector2(right - left, bottom - top);
        }

        return found;
    }

    internal void UpdateShellLayout(Vector2 displaySize)
    {
        _suppressLeftSidebarForLayout = false;
        _suppressRightSidebarForLayout = false;
        _suppressMinimapForLayout = false;
        if (_hideUiChrome || displaySize.X <= 0f)
            return;

        float maxSidebarWidthBudget = MathF.Max(0f, displaySize.X - SceneViewportHardMinWidth);
        float requiredCompactWidth = (_showLeftSidebar ? SidebarCompactMinWidth : 0f)
            + (_showRightSidebar ? SidebarCompactMinWidth : 0f);

        if (requiredCompactWidth > maxSidebarWidthBudget && _showRightSidebar)
            _suppressRightSidebarForLayout = true;

        requiredCompactWidth = (_showLeftSidebar ? SidebarCompactMinWidth : 0f)
            + (IsShellPanelActive(ShellPanelId.Inspector) ? SidebarCompactMinWidth : 0f);

        if (requiredCompactWidth > maxSidebarWidthBudget && _showLeftSidebar)
            _suppressLeftSidebarForLayout = true;

        ClampFixedSidebarLayout(displaySize.X);

        if (_showMinimapWindow && !_fullscreenMinimap && _useDockspaceUi)
        {
            float requiredMinimapWidth = GetShellPanelDefinition(ShellPanelId.Minimap).CompactMinWidth;
            _suppressMinimapForLayout = displaySize.X < SceneViewportHardMinWidth + requiredMinimapWidth;
        }
    }

    private float ClampFixedBottomDrawerHeight(float height, float displayHeight)
    {
        GetFixedBottomDrawerHeightRange(displayHeight, out float minHeight, out float maxHeight);
        return Math.Clamp(height, minHeight, maxHeight);
    }

    private void GetFixedBottomDrawerHeightRange(float displayHeight, out float minHeight, out float maxHeight)
    {
        float availableHeight = MathF.Max(0f, displayHeight - GetTopChromeHeight() - StatusBarHeight);
        float preferredMaxHeight = availableHeight - SceneViewportPreferredMinHeight;
        float hardMaxHeight = availableHeight - SceneViewportHardMinHeight;
        maxHeight = MathF.Min(BottomDrawerMaxHeight, MathF.Max(BottomDrawerCompactMinHeight, MathF.Max(preferredMaxHeight, hardMaxHeight)));
        minHeight = MathF.Min(BottomDrawerMinHeight, maxHeight);
    }

    private void ClampFixedSidebarLayout(float displayWidth)
    {
        if (displayWidth <= 0f)
            return;

        if (IsShellPanelActive(ShellPanelId.Navigator))
            _leftSidebarWidth = Math.Clamp(_leftSidebarWidth, SidebarCompactMinWidth, SidebarMaxWidth);

        if (IsShellPanelActive(ShellPanelId.Inspector))
            _rightSidebarWidth = Math.Clamp(_rightSidebarWidth, SidebarCompactMinWidth, SidebarMaxWidth);

        if (IsShellPanelActive(ShellPanelId.Navigator))
            _leftSidebarWidth = ClampFixedSidebarWidth(_leftSidebarWidth, isLeftSidebar: true, displayWidth);

        if (IsShellPanelActive(ShellPanelId.Inspector))
            _rightSidebarWidth = ClampFixedSidebarWidth(_rightSidebarWidth, isLeftSidebar: false, displayWidth);
    }

    private static void ApplyDockedSidePanelInset(in DockPanelState state, bool isLeftPanel, float viewportY, float viewportHeight, ref float x, ref float width)
    {
        if (!state.Visible || !state.IsDocked || state.Size.X <= 1f || state.Size.Y <= 1f)
            return;

        float panelTop = state.Position.Y;
        float panelBottom = state.Position.Y + state.Size.Y;
        float viewportBottom = viewportY + viewportHeight;
        if (panelBottom <= viewportY || panelTop >= viewportBottom)
            return;

        const float edgeTolerance = 4f;
        if (isLeftPanel)
        {
            if (state.Position.X > x + edgeTolerance)
                return;

            x += state.Size.X;
            width -= state.Size.X;
            return;
        }

        float viewportRight = x + width;
        if (state.Position.X + state.Size.X < viewportRight - edgeTolerance)
            return;

        width -= state.Size.X;
    }

    internal bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height)
    {
        var io = ImGui.GetIO();

        if (_hideUiChrome)
        {
            x = 0f;
            y = 0f;
            width = io.DisplaySize.X;
            height = io.DisplaySize.Y;
            return width > 10f && height > 10f;
        }

        float topOffset = GetTopChromeHeight();
        x = 0f;
        y = topOffset;
        width = io.DisplaySize.X;
        height = io.DisplaySize.Y - topOffset - BottomBarHeight - StatusBarHeight;

        // 071: tab system uses fixed left/right sidebars; viewport is the
        // middle area between them. Sidebars auto-hide when the window is
        // too small (see UpdateShellLayout suppression logic).
        if (_useTabUi)
        {
            if (_showLeftSidebar)
            {
                x += _leftSidebarWidth;
                width -= _leftSidebarWidth;
            }

            if (_showRightSidebar)
                width -= _rightSidebarWidth;

            width = MathF.Max(width, 0f);
            height = MathF.Max(height, 0f);
            return width > 10f && height > 10f;
        }

        if (_useDockspaceUi && _dockspaceHostSize.X > 10f && _dockspaceHostSize.Y > 10f)
        {
            x = _dockspaceHostPosition.X;
            y = _dockspaceHostPosition.Y;
            width = _dockspaceHostSize.X;
            height = _dockspaceHostSize.Y;

            if (TryGetVisibleShellPanelInsetState(isLeftPanel: true, out DockPanelState leftDockPanel))
                ApplyDockedSidePanelInset(leftDockPanel, isLeftPanel: true, y, height, ref x, ref width);

            if (TryGetVisibleShellPanelInsetState(isLeftPanel: false, out DockPanelState rightDockPanel))
                ApplyDockedSidePanelInset(rightDockPanel, isLeftPanel: false, y, height, ref x, ref width);
        }
        else
        {
            if (IsShellPanelActive(ShellPanelId.Navigator))
            {
                x += _leftSidebarWidth;
                width -= _leftSidebarWidth;
            }

            if (IsShellPanelActive(ShellPanelId.Inspector))
                width -= _rightSidebarWidth;

        }

        width = MathF.Max(width, 0f);
        height = MathF.Max(height, 0f);
        return width > 10f && height > 10f;
    }

    internal bool TryGetSceneFramebufferViewport(out int x, out int y, out uint width, out uint height)
    {
        x = y = 0;
        width = height = 0;

        if (!TryGetSceneViewportRect(out float viewportX, out float viewportY, out float viewportWidth, out float viewportHeight))
            return false;

        Vector2D<int> windowSize = _window.Size;
        Vector2D<int> framebufferSize = _window.FramebufferSize;
        if (windowSize.X <= 0 || windowSize.Y <= 0 || framebufferSize.X <= 0 || framebufferSize.Y <= 0)
            return false;

        float scaleX = (float)framebufferSize.X / windowSize.X;
        float scaleY = (float)framebufferSize.Y / windowSize.Y;

        int viewportLeft = (int)MathF.Round(viewportX * scaleX);
        int viewportTop = (int)MathF.Round(viewportY * scaleY);
        int viewportRight = (int)MathF.Round((viewportX + viewportWidth) * scaleX);
        int viewportBottom = (int)MathF.Round((viewportY + viewportHeight) * scaleY);

        viewportLeft = Math.Clamp(viewportLeft, 0, framebufferSize.X);
        viewportRight = Math.Clamp(viewportRight, viewportLeft, framebufferSize.X);
        viewportTop = Math.Clamp(viewportTop, 0, framebufferSize.Y);
        viewportBottom = Math.Clamp(viewportBottom, viewportTop, framebufferSize.Y);

        x = viewportLeft;
        y = framebufferSize.Y - viewportBottom;
        width = (uint)Math.Max(1, viewportRight - viewportLeft);
        height = (uint)Math.Max(1, viewportBottom - viewportTop);
        return true;
    }

    internal void SyncImGuiWindowMetrics(Vector2D<int> windowSize, Vector2D<int> framebufferSize)
    {
        if (_imGui == null || !HasImGuiContext())
            return;

        if (windowSize.X <= 0 || windowSize.Y <= 0 || framebufferSize.X <= 0 || framebufferSize.Y <= 0)
            return;

        bool windowSizeChanged = !windowSize.Equals(_lastSyncedImGuiWindowSize);
        bool framebufferSizeChanged = !framebufferSize.Equals(_lastSyncedImGuiFramebufferSize);
        if (!windowSizeChanged && !framebufferSizeChanged)
            return;

        if (windowSizeChanged)
            ImGuiControllerWindowResizedMethod?.Invoke(_imGui, new object[] { windowSize });

        ImGuiIOPtr io = ImGui.GetIO();
        io.DisplaySize = new Vector2(windowSize.X, windowSize.Y);
        io.DisplayFramebufferScale = new Vector2(
            windowSize.X > 0 ? (float)framebufferSize.X / windowSize.X : 1f,
            windowSize.Y > 0 ? (float)framebufferSize.Y / windowSize.Y : 1f);

        _lastSyncedImGuiWindowSize = windowSize;
        _lastSyncedImGuiFramebufferSize = framebufferSize;
    }

    internal static bool HasImGuiContext()
        => ImGui.GetCurrentContext() != IntPtr.Zero;
}
