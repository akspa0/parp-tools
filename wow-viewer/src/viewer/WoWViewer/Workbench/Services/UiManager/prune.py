from pathlib import Path
import re
root=Path('wow-viewer/src/viewer/WoWViewer')
def read(p):return (root/p).read_text(encoding='utf-8-sig')
def write(p,s):(root/p).write_text(s,encoding='utf-8')
def drop_method(s,name):
    s,n=re.subn(r'^    (?:internal|private|public)[^\n]*\b'+name+r'\(.*?^    \{.*?^    }\n','',s,flags=re.M|re.S)
    return s
p='Workbench/Services/Chrome/ViewerChromeService.Sidebars.cs';s=read(p)
for name in ['DrawFixedSidebarWidthControl','ClampFixedSidebarWidth','GetFixedSidebarWidthRange']:
    s,n=re.subn(r'^    (?:internal|private)[^\n]*\b'+name+r'\(.*?^    \{.*?^    }\n','',s,flags=re.M|re.S)
write(p,s)
p='Workbench/Services/Settings/ViewerSettingsService.cs';s=read(p)
s=re.sub(r'                        _savedPm4ObjectMatches.Clear\(\);.*?\n                        _savedObjectPathFiltersByMap.Clear\(\);','                        _savedObjectPathFiltersByMap.Clear();',s,flags=re.S)
s=re.sub(r'                Pm4ObjectMatchSelections = _savedPm4ObjectMatches.Values.*?                    .ToList\(\),\n','',s,flags=re.S)
s=re.sub(r'^        public List<(?:SavedPm4ObjectMatchSelection|SavedShellPanelLayout)>[^\n]*\n','',s,flags=re.M)
write(p,s)

# Private forwarding declarations no longer referenced by their owning service are obsolete.
for folder in ['WorkbenchPanels','Navigator','MainMenu','Chrome','Pm4Workbench','Investigation','Settings']:
    directory=root/f'Workbench/Services/{folder}'
    for f in directory.glob('*.cs'):
        if not (f.name.endswith('.Host.cs') or f.name in ['ViewerSettingsService.cs','SettingsWindowService.cs']):continue
        s=f.read_text(encoding='utf-8-sig')
        alltext='\n'.join(t.read_text(encoding='utf-8-sig') for t in directory.glob('*.cs'))
        lines=[]
        for line in s.splitlines(True):
            if re.match(r'    private .*=>',line):
                m=re.search(r'\b(\w+)\s*(?:\([^\n]*\))?\s*$',line.split('=>')[0])
                if m and len(re.findall(r'\b'+m[1]+r'\b',alltext))==1:continue
            lines.append(line)
        f.write_text(''.join(lines),encoding='utf-8')

# Audit the entire broad host, removing unreferenced declarations and explicit implementations.
iface=read('Workbench/Services/IViewerAppHost.cs');host=read('ViewerApp_Host.cs')
consumers='\n'.join(f.read_text(encoding='utf-8-sig') for f in root.rglob('*.cs') if f.name not in ['ViewerApp_Host.cs','IViewerAppHost.cs'])
removed=[]
for line in iface.splitlines():
    m=re.match(r'    [^\n]*?\b(\w+)\s*(?:\([^;\n]*\);|\{ get; })\s*$',line)
    if not m:continue
    name=m[1]
    if not re.search(r'\.'+name+r'\b',consumers):
        iface=iface.replace(line+'\n','')
        host=re.sub(r'^    [^\n]*IViewerAppHost\.'+name+r'\b[^\n]*\n','',host,flags=re.M)
        removed.append(name)
write('Workbench/Services/IViewerAppHost.cs',iface);write('ViewerApp_Host.cs',host)
print('Removed obsolete host members: '+', '.join(removed))

# Deleted shell and curated-selection types/state no longer exist in the app.
p='ViewerApp.cs';s=read(p)
s=re.sub(r'    \[Obsolete\("Shell panel system[^\n]*\n','',s)
for name in ['ShellPanelId','ShellPanelLane','SavedShellPanelLayout','SavedPm4ObjectMatchSelection']:
    s=re.sub(r'^    internal (?:enum|sealed class) '+name+r'\n    \{.*?^    }\n','',s,flags=re.M|re.S)
s=re.sub(r'^    internal readonly record struct ShellPanelDefinition\(.*?MaxWidth\);\n','',s,flags=re.M|re.S)
s=re.sub(r'^    internal static readonly ShellPanelDefinition\[\] ShellPanelDefinitions =\n    \{.*?^    };\n','',s,flags=re.M|re.S)
s=re.sub(r'^    private [^\n]*(?:_dockspaceHost|_savedShellPanelLayouts|_pendingShellPanelLayoutRestore|_forceApplyShellPanelLayout|_pendingFocusedShellPanel|_savedPm4ObjectMatches)[^\n]*\n','',s,flags=re.M)
write(p,s)

# No generator run is performed by the diagnostic harness; it must run the real SDK generators.
print('Old shell state and curated-selection persistence removed.')

# Remove the remaining legacy shell toggles and floating-window wrappers. Their content is
# already reachable from the manager-owned destination pages.
p='ViewerApp.cs';s=read(p)
s=s.replace('if (_useTabUi && _worldScene.Selection.SelectedObjectType', 'if (_worldScene.Selection.SelectedObjectType')
for name in ['_useDockspaceUi','_useTabUi','_activeUtilitiesTabIndex','_workbenchOpen','_showWorkspaceBarsPanel',
             '_showFileBrowser','_showUniqueIdArchaeologyWindow','_showCaptureAutomationWindow','_showCameraPathWindow',
             '_showWeakSignalWindow','_activeBottomDrawerTab','_pendingRightSidebarSection']:
    s=re.sub(r'^    private [^\n]*\b'+name+r'\b[^\n]*\n','',s,flags=re.M)
s=re.sub(r'^    internal enum FixedBottomDrawerTab\n    \{.*?^    }\n','',s,flags=re.M|re.S)
write(p,s)

p='ViewerApp_Host.cs';s=read(p)
for name in ['UseDockspaceUi','UseTabUi','ActiveUtilitiesTabIndex','WorkbenchOpen','ShowWorkspaceBarsPanel',
             'ShowFileBrowser','ShowUniqueIdArchaeologyWindow','ShowCaptureAutomationWindow','ShowCameraPathWindow',
             'ShowWeakSignalWindow','ActiveBottomDrawerTab','PendingRightSidebarSection']:
    s=re.sub(r'^    [^\n]*IViewerAppHost\.'+name+r'\b[^\n]*\n','',s,flags=re.M)
write(p,s)

p='Workbench/Services/IViewerAppHost.cs';s=read(p)
for name in ['UseDockspaceUi','UseTabUi','ActiveUtilitiesTabIndex','WorkbenchOpen','ShowWorkspaceBarsPanel',
             'ShowFileBrowser','ShowUniqueIdArchaeologyWindow','ShowCaptureAutomationWindow','ShowCameraPathWindow',
             'ShowWeakSignalWindow','ActiveBottomDrawerTab','PendingRightSidebarSection']:
    s=re.sub(r'^    [^\n]*\b'+name+r'\b[^\n]*\n','',s,flags=re.M)
write(p,s)

for p in ['Workbench/Services/CameraPaths/CameraPathsService.Host.cs','Workbench/Services/Investigation/InvestigationService.Host.cs',
          'Workbench/Services/Settings/SettingsWindowService.Host.cs','Workbench/Services/Settings/ViewerSettingsService.cs',
          'Workbench/Services/Workspaces/WorkspacesService.Host.cs','Workbench/Services/TerrainControls/TerrainControlsPanelService.Host.cs',
          'Workbench/Services/CaptureAutomation/CaptureAutomationService.Host.cs']:
    s=read(p)
    for name in ['_useDockspaceUi','_useTabUi','_activeUtilitiesTabIndex','_workbenchOpen','_showCaptureAutomationWindow',
                 '_showCameraPathWindow','_showWeakSignalWindow','_activeBottomDrawerTab','_pendingRightSidebarSection',
                 '_showLogViewer','_showPerfWindow']:
        s=re.sub(r'^    private [^\n]*\b'+name+r'\b[^\n]*\n','',s,flags=re.M)
    write(p,s)

p='Workbench/Services/Investigation/InvestigationService.Investigation.cs';s=read(p)
s=s.replace('''        if (_useTabUi)
        {
            ImGui.TextDisabled("Chunk details and flag overlays are in Inspector > Context.");
            if (ImGui.Button("Open Terrain Inspector##Investigation"))
                _host.UiManager.OpenViewer(WoWViewer.Workbench.ViewerPanel.Inspector);
        }
        else
        {
            DrawUnifiedInspectorContent();
        }''','''        ImGui.TextDisabled("Chunk details and flag overlays are in Inspector > Context.");
        if (ImGui.Button("Open Terrain Inspector##Investigation"))
            _host.UiManager.OpenViewer(WoWViewer.Workbench.ViewerPanel.Inspector);''')
write(p,s)

p='Workbench/Services/Workspaces/WorkspacesService.cs';s=read(p)
s=s.replace('''        _editorWorkspaceTask = task;
        _activeBottomDrawerTab = task switch
        {
            EditorWorkspaceTask.Terrain => FixedBottomDrawerTab.Terrain,
            EditorWorkspaceTask.Objects => FixedBottomDrawerTab.World,
            EditorWorkspaceTask.Pm4Evidence => FixedBottomDrawerTab.Pm4,
            EditorWorkspaceTask.Inspect => FixedBottomDrawerTab.Workspace,
            EditorWorkspaceTask.Publish => FixedBottomDrawerTab.Diagnostics,
            _ => _activeBottomDrawerTab,
        };
        _pendingRightSidebarSection = _activeBottomDrawerTab;''','''        _editorWorkspaceTask = task;
        _host.UiManager.Navigate(WorkbenchTab.Editor, task switch
        {
            EditorWorkspaceTask.Terrain => 1,
            EditorWorkspaceTask.Objects => 0,
            EditorWorkspaceTask.Pm4Evidence => 0,
            EditorWorkspaceTask.Publish => 2,
            _ => 0,
        });''')
s=s.replace('''        ImGui.SameLine();
        if (_useTabUi)
        {
            if (ImGui.Button("Log Viewer"))
                _host.ShowLogViewer = true;

            if (ImGui.Button("Perf"))
                _host.ShowPerfWindow = true;
        }
        else
        {
            if (ImGui.Button(_showLogViewer ? "Hide Log Viewer" : "Show Log Viewer"))
                _showLogViewer = !_showLogViewer;

            if (ImGui.Button(_showPerfWindow ? "Hide Perf" : "Show Perf"))
                _showPerfWindow = !_showPerfWindow;
        }''','''        ImGui.SameLine();
        if (ImGui.Button("Log Viewer"))
            _host.ShowLogViewer = true;
        if (ImGui.Button("Perf"))
            _host.ShowPerfWindow = true;''')
write(p,s)

p='Workbench/Services/CameraPaths/CameraPathsService.cs';s=read(p);s=drop_method(s,'DrawCameraPathWindow');write(p,s)
p='Workbench/Services/CaptureAutomation/CaptureAutomationService.cs';s=read(p);s=drop_method(s,'DrawCaptureAutomationWindow');write(p,s)
p='Workbench/Services/TerrainControls/TerrainControlsPanelService.Sidebars.cs';s=read(p);s=drop_method(s,'DrawWeakSignalWindow');write(p,s)
p='Workbench/Services/CameraPaths/CameraPathsService.ViewerKeyBindings.cs';s=read(p)
s=s.replace('&& (_showCameraPathWindow || _activeCapturePanelTabIndex == (int)CapturePanelTab.CameraPath);','&& _activeCapturePanelTabIndex == (int)CapturePanelTab.CameraPath;')
write(p,s)
print('Legacy shell toggles and floating-window wrappers removed.')

# Log/Perf are manager-owned windows; their providers expose content only.
p='Workbench/Services/LogViewer/LogViewerService.cs';s=read(p);s=drop_method(s,'DrawLogViewer');write(p,s)
p='Workbench/Services/Perf/PerfPanelService.cs';s=read(p);s=drop_method(s,'DrawPerfWindow');write(p,s)
print('Log/Perf window wrappers removed; manager owns their windows.')
