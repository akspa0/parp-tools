from pathlib import Path
import re

root = Path('wow-viewer/src/viewer/WoWViewer')
def read(p): return (root / p).read_text(encoding='utf-8-sig')
def write(p, s): (root / p).write_text(s, encoding='utf-8')
def method(s, name):
    m = re.search(r'^    (?:internal|private|public|protected)[^\n]*\b' + name + r'\(.*?^    \{.*?^    }\n', s, re.M|re.S)
    if not m: raise Exception('method not found '+name)
    return m.group()
def remove(s, name): return s.replace(method(s, name), '')

# Retain the proven controller metric sync, event queue and framebuffer conversion, not the old shell.
shell = read('Workbench/Services/ShellLayout/ShellLayoutService.cs')
names = ['QueueImGuiMouseButtonEvent','FlushPendingImGuiMouseButtonEvents','TryGetSceneFramebufferViewport','SyncImGuiWindowMetrics']
inputsrc = '''using System.Numerics;
using System.Reflection;
using ImGuiNET;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL.Extensions.ImGui;
namespace WoWViewer;
internal sealed partial class ViewerUiManager
{
    private static readonly MethodInfo? ImGuiControllerWindowResizedMethod = typeof(ImGuiController).GetMethod("WindowResized", BindingFlags.Instance | BindingFlags.NonPublic);
    private readonly Lock _pendingImGuiMouseEventLock = new();
    private readonly Queue<(int ButtonIndex, bool Down)> _pendingImGuiMouseButtonEvents = new();
    private Vector2D<int> _lastSyncedImGuiWindowSize;
    private Vector2D<int> _lastSyncedImGuiFramebufferSize;
    internal static bool HasImGuiContext() => ImGui.GetCurrentContext() != IntPtr.Zero;
    internal bool CanSceneConsumeMouse(float x, float y)
    {
        if (!HasImGuiContext()) return false;
        UpdateLayout(ImGui.GetIO().DisplaySize);
        if (!Layout[UiRegion.Scene].Rect.Contains(x, y) || _popups.IsOpen || ImGui.GetIO().WantCaptureMouse) return false;
        foreach (var state in Layout.Regions.Values)
            if (state.Visible && state.BlocksSceneInput && state.Rect.Contains(x, y)) return false;
        foreach (var state in _hosted.Values)
            if (state.Visible && state.BlocksSceneInput && state.Rect.Contains(x, y)) return false;
        return !_host.FullscreenMinimap;
    }
    internal bool IsSceneMouseCaptureBlocked(float x, float y) => !CanSceneConsumeMouse(x, y);
    internal bool IsSceneKeyboardCaptureBlocked() => !HasImGuiContext() || _popups.IsOpen || ImGui.GetIO().WantCaptureKeyboard || ImGui.GetIO().WantTextInput;
    internal bool TryGetSceneViewportRect(out float x, out float y, out float width, out float height)
    {
        UpdateLayout(HasImGuiContext() ? ImGui.GetIO().DisplaySize : new Vector2(_host.Window.Size.X, _host.Window.Size.Y));
        var rect = Layout[UiRegion.Scene].Rect;
        x = rect.X; y = rect.Y; width = rect.Width; height = rect.Height;
        return width > 0 && height > 0;
    }
    internal void HandleChromeShortcuts()
    {
        if (ImGui.GetIO().WantTextInput) return;
        if (ImGui.IsKeyPressed(ImGuiKey.F1)) _host.MainMenuBar.RequestKeyboardShortcuts();
        if (ImGui.IsKeyPressed(ImGuiKey.LeftBracket)) { _host.ShowLeftSidebar = !_host.ShowLeftSidebar; SaveLayout(); }
        if (ImGui.IsKeyPressed(ImGuiKey.RightBracket)) { _host.ShowRightSidebar = !_host.ShowRightSidebar; SaveLayout(); }
    }
'''
inputsrc += '\n'.join(method(shell,n).replace('_window','_host.Window').replace('_imGui','_host.ImGui') for n in names) + '}\n'
write('Workbench/Services/UiManager/ViewerUiManager.Input.cs',inputsrc)

write('Workbench/WorkbenchTab.cs', '''namespace WoWViewer.Workbench;
public enum WorkbenchTab { Viewer, Editor, Archaeology }
public enum ViewerPanel { Inspector, World, Model, PM4 }
''')
write('Workbench/WorkbenchNavigator.cs', '''namespace WoWViewer.Workbench;
public static class WorkbenchNavigator
{
    public static string[] GetPageLabels(WorkbenchTab destination) => destination switch
    {
        WorkbenchTab.Editor => ["Placement & Objects", "Terrain Tools", "Data I/O", "Converters"],
        WorkbenchTab.Archaeology => ["Weak Signal & Stratigraphy", "UniqueId Timeline", "Layers & Provenance", "Playback & Capture", "Cartography"],
        _ => [],
    };
}
''')

app = read('ViewerApp.cs').replace('ShellLayoutService','ViewerUiManager').replace('_shellLayout','_uiManager').replace('WorkbenchTab.Quick','WorkbenchTab.Viewer')
app = app.replace('        Correlation,\n','')
app = app.replace('        ImGui.GetIO().ConfigFlags |= ImGuiConfigFlags.DockingEnable;','        ImGui.GetIO().ConfigFlags &= ~ImGuiConfigFlags.DockingEnable;')
app = app.replace('        _settings.LoadViewerSettings();','        _settings.LoadViewerSettings();\n        _uiManager.LoadLayout(_window.Size.X);')
app = app.replace('        HandleKeyboardInput((float)dt);','        _uiManager.HandleChromeShortcuts();\n        HandleKeyboardInput((float)dt);')
app = app.replace('        return io.WantCaptureKeyboard || io.WantTextInput;', '        return _uiManager.IsSceneKeyboardCaptureBlocked();')
app = re.sub(r'            _activeBottomDrawerTab = FixedBottomDrawerTab.Pm4;\n            if \(_workspaceMode == WorkspaceMode.Editor\)\n                _workspaces.SetEditorWorkspaceTask\(EditorWorkspaceTask.Pm4Evidence\);', '            _uiManager.OpenViewer(ViewerPanel.PM4);', app)
app = app.replace('_uiManager.FocusShellPanel(ShellPanelId.Inspector)', '_uiManager.OpenViewer(ViewerPanel.Inspector)')
start = app.index('        _uiManager.UpdateShellLayout', app.index('    private void DrawUI()'))
end = app.index('        if (_recordingCoordinator.ActiveTourPresentation',start)
app = app[:start] + '        _uiManager.Draw();\n\n' + app[end:]
app = app.replace('        _forceApplyShellPanelLayout = false;\n','')
app = remove(app,'DrawDockspaceHost')
app = app.replace('_workbenchPanels.OpenWorkbenchTab(ModelBottomTab.Info)', '_uiManager.OpenViewer(ViewerPanel.Model)')
app = app.replace('        return MenuBarHeight + GetActiveToolbarHeight();','        return _hideUiChrome ? 0f : 28f;')
app = re.sub(r'^    private (?:Pm4ObjectMatchStore\?|Dictionary<string, Pm4ObjectMatchEntry>)[^\n]*\n','',app,flags=re.M)
app = app.replace('        _settings.SaveViewerSettings();\n\n        _loadingScreen', '        _settings.SaveViewerSettings();\n        _uiManager.SaveLayout();\n\n        _loadingScreen')
write('ViewerApp.cs',app)

# Content-only right-side provider, keeping the feature bodies rather than old route adapters.
p='Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs'
old=read(p)
keep=['DrawUnifiedInspectorContent','DrawInspectorWorldContextContent','DrawCameraControlsContent','DrawSelectedObjectSummaryContent','DrawSelectedObjectInspectorSection','DrawWorldLodSubTab','DrawPageCombo']
content='''using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;
using WoWViewer.Workbench;
using static WoWViewer.ViewerApp;
using static WoWViewer.ImGuiListLayout;
namespace WoWViewer;
internal sealed partial class WorkbenchPanelsService
{
    private readonly IViewerAppHost _host;
    internal WorkbenchPanelsService(IViewerAppHost host) => _host = host;
    internal void DrawDestinationContent(WorkbenchTab destination)
    {
        if (destination == WorkbenchTab.Editor) { EnsureEditorHost(); EnsureEditorPages().Draw(_activeBottomTabIndex); }
        else if (destination == WorkbenchTab.Archaeology) _archaeologyPanel.DrawArchaeologyWorkbenchSubTabContent();
    }
    internal void DrawViewerContent(ViewerPanel panel)
    {
        var ui = _host.UiManager;
        switch (panel)
        {
            case ViewerPanel.Inspector:
                int inspector = DrawPageCombo("##InspectorPage", ["Context", "Scene Investigation", "MCNK Flags", "World Context"], ui.InspectorPage);
                if (inspector != ui.InspectorPage) ui.OpenViewer(panel, inspector);
                switch (inspector)
                {
                    case 0: DrawUnifiedInspectorContent(); break;
                    case 1: _investigation.DrawVisualInvestigationToolbox(showWorldObjectRangeControls: _worldScene != null); break;
                    case 2: _investigation.DrawMcnkFlagOverlayControls(); break;
                    case 3: DrawInspectorWorldContextContent(); break;
                }
                break;
            case ViewerPanel.World:
                int world = DrawPageCombo("##WorldPage", ["Objects", "Placements", "LOD & Budget", "Exports"], ui.WorldPage);
                if (world != ui.WorldPage) ui.OpenViewer(panel, world);
                if (_worldScene == null) { SharedUiWidgets.CompactStatus("Load a world to inspect it."); break; }
                switch (world)
                {
                    case 0: _worldObjectsPanel.DrawWorldObjectsPanelContent(); break;
                    case 1: _worldObjectsPanel.DrawWorldPlacementsSubTab(); break;
                    case 2: DrawWorldLodSubTab(); break;
                    case 3: _mainMenuBar.DrawWorldExportContent(); break;
                }
                break;
            case ViewerPanel.Model:
                int model = DrawPageCombo("##ModelPage", ["Info", "Animations", "Actions & Exports"], ui.ModelPage);
                if (model != ui.ModelPage) ui.OpenViewer(panel, model);
                switch (model)
                {
                    case 0: _modelInspector.DrawModelInfoPanelContent(); break;
                    case 1: _modelInspector.DrawModelAnimationsSubTab(); break;
                    case 2: _modelInspector.DrawModelActionsSubTab(); _mainMenuBar.DrawModelExportContent(); break;
                }
                break;
            case ViewerPanel.PM4:
                if (_worldScene != null) _pm4Workbench.DrawPm4WorkbenchInspector();
                else SharedUiWidgets.CompactStatus("Load a world to inspect PM4.");
                break;
        }
    }
    internal void DrawConvertersSubTabContent() => _mainMenuBar.DrawConvertersContent();
    internal void DrawSceneToolsContent()
    {
        var ui = _host.UiManager;
        int page = DrawPageCombo("##SceneToolsPage", ["Taxi", "Capture"], ui.SceneToolsPage);
        if (page != ui.SceneToolsPage) ui.OpenSceneTools(page);
        if (page == 0) { if (_worldScene != null) _taxiPanel.DrawTaxiContent(); else SharedUiWidgets.CompactStatus("Load a world to use taxi tools."); }
        else DrawCapturePanelContent();
    }
    internal void DrawAssetsContent()
    {
        if (_catalogView == null)
        {
            _catalogView = new Catalog.AssetCatalogView(_gl);
            _catalogView.SetDataSource(_dataSource);
            _catalogView.OnLoadModelRequested = _modelLoader.OnCatalogLoadModel;
        }
        _catalogView.DrawContent();
    }
'''
content += '\n'.join(method(old,n) for n in keep) + '}\n'
write(p,content)

# Left provider: no window ownership, no second minimap, no utility page aliases.
p='Workbench/Services/Navigator/NavigatorPanelService.Sidebars.cs'
nav=read(p)
start=nav.index('    internal void DrawLeftSidebar()')
end=nav.index('    /// <summary>',nav.index('    private void DrawSharedWorldOverviewSection'))
nav=nav[:start]+'''    internal void DrawContent()
    {
        if (SharedUiWidgets.SectionHeader("Workspace", defaultOpen: true, id: "Workspace"))
        {
            if (_dataSource != null) SharedUiWidgets.CompactStatus($"Source: {_dataSource.Name}");
            _host.WorkbenchPanels.DrawCameraContent();
            float width = _host.LeftSidebarWidth;
            if (ImGui.SliderFloat("Sidebar width", ref width, 180, 600, "%.0f px")) _host.LeftSidebarWidth = width;
            if (ImGui.IsItemDeactivatedAfterEdit()) _host.UiManager.SaveLayout();
        }
        if (SharedUiWidgets.SectionHeader("World Overview", defaultOpen: true, id: "WorldOverview"))
        {
            DrawWorldOverviewContent();
            DrawMapExportControls();
            DrawMapDiscoveryContent();
        }
        if (SharedUiWidgets.SectionHeader("File Browser", defaultOpen: true, id: "FileBrowser")) DrawFileBrowserContent(260);
        switch (_host.ActiveTopTab)
        {
            case WorkbenchTab.Viewer:
                bool sceneTools = SharedUiWidgets.SectionHeader("Scene Tools", defaultOpen: _host.UiManager.SceneToolsVisible, id: "SceneTools");
                _host.UiManager.SceneToolsVisible = sceneTools;
                if (sceneTools) _host.WorkbenchPanels.DrawSceneToolsContent();
                if (SharedUiWidgets.SectionHeader("Assets", id: "Assets")) _host.WorkbenchPanels.DrawAssetsContent();
                break;
            case WorkbenchTab.Editor:
                if (SharedUiWidgets.SectionHeader("Editor Workspace", defaultOpen: true, id: "EditorWorkspace"))
                {
                    string[] pages = WorkbenchNavigator.GetPageLabels(WorkbenchTab.Editor);
                    for (int i = 0; i < pages.Length; i++)
                        if (ImGui.Button(pages[i])) _host.UiManager.Navigate(WorkbenchTab.Editor, i);
                }
                if (SharedUiWidgets.SectionHeader("Project Output", id: "ProjectOutput")) _host.ProjectOutput.DrawProjectOutputSettingsContent();
                break;
            case WorkbenchTab.Archaeology:
                if (SharedUiWidgets.SectionHeader("Archaeology Workspace", defaultOpen: true, id: "ArchaeologyWorkspace"))
                    SharedUiWidgets.CompactStatus("Analysis and reconstruction pages are in the right sidebar.");
                if (SharedUiWidgets.SectionHeader("Layer Projects", id: "LayerProjects"))
                {
                    SharedUiWidgets.CompactStatus("Map layers and their project actions have one home: Cartography.");
                    if (ImGui.Button("Open Cartography")) _host.UiManager.Navigate(WorkbenchTab.Archaeology, 4);
                }
                break;
        }
        if (SharedUiWidgets.SectionHeader("Diagnostics", id: "Diagnostics"))
        {
            if (ImGui.Button("Log")) _host.ShowLogViewer = !_host.ShowLogViewer;
            ImGui.SameLine();
            if (ImGui.Button("Performance")) _host.ShowPerfWindow = !_host.ShowPerfWindow;
            _host.ViewerChrome.DrawRuntimeStatsPanelContent();
        }
    }

'''+nav[end:]
nav=remove(nav,'DrawSharedWorldMapsSection')
overview=method(nav,'DrawWorldOverviewContent')
overview=re.sub(r'        if \(TryGetActiveMinimapState.*?\n        if \(_worldScene', '        if (_worldScene', overview,flags=re.S)
overview=overview[:overview.index('        if (ImGui.Button(_fullscreenMinimap')]+ '    }\n'
nav=nav.replace(method(nav,'DrawWorldOverviewContent'),overview)
write(p,nav)

# Editor delegates are injected at construction, not dangling optional parameters.
p='Workbench/Pages/EditorWorkbenchPages.cs'; s=read(p)
s=s.replace('EditorWorkbenchPages(ViewerAppContext context)', 'EditorWorkbenchPages(ViewerAppContext context, Action? terrainFileOperations = null, Action? converters = null)')
s=s.replace('new EditorTerrainToolsPage(context)', 'new EditorTerrainToolsPage(context, terrainFileOperations)').replace('new EditorConvertersPage(context)', 'new EditorConvertersPage(context, converters)')
s=re.sub(r'    /// <summary>\n    /// Maps a pre-231.*?\n    };\n','',s,flags=re.S)
write(p,s)
p='ViewerApp_Editor.cs';s=read(p).replace('new Workbench.Pages.ViewerAppContext(this));','new Workbench.Pages.ViewerAppContext(this), _mainMenuBar.DrawTerrainFileOperationsContent, _mainMenuBar.DrawConvertersContent);')
s=re.sub(r'^    void Workbench.Pages.IEditorPageHost.(?:DrawPm4Exports|OpenPm4Workbench).*\n','',s,flags=re.M)
write(p,s)

# Settings content loses its top-level window; the registry owns it.
p='Workbench/Services/Settings/SettingsWindowService.cs';s=read(p);s=remove(s,'DrawSettingsWindow').replace('private void DrawSettingsContent()', 'internal void DrawSettingsContent()');write(p,s)

# Remove the actual curation action only; retain the library selection/placement/export content.
p='Workbench/Services/EditorPanels/EditorPanelsService.cs';s=read(p)
s=re.sub(r'                ImGui.SameLine\(\);\n                if \(ImGui.Button\("Match to selected PM4 object"\)\).*?^                }\n','',s,flags=re.M|re.S)
write(p,s)

# Remove the old shell implementation itself.
for p in (root/'Workbench/Services/ShellLayout').glob('*.cs'): p.unlink()
print('Manager input, destination providers, app delegation, editor injection and shell deletion written.')
