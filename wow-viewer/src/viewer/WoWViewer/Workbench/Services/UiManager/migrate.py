from pathlib import Path
import re
root=Path('wow-viewer/src/viewer/WoWViewer')
def read(p):return (root/p).read_text(encoding='utf-8-sig')
def write(p,s):(root/p).write_text(s,encoding='utf-8')
def replace_method(s,name,body):
    pattern=r'^    (?:internal|private|public)[^\n]*\b'+name+r'\(.*?^    \{.*?^    }\n'
    s,n=re.subn(pattern,lambda m:body,s,flags=re.M|re.S)
    assert n==1,(name,n)
    return s

# Only mechanical shell-type/delegation references outside the excluded minimap/chat providers.
for f in root.rglob('*.cs'):
    rel=f.relative_to(root).as_posix()
    if '/MinimapStatus/' in rel or '/Dialogue/' in rel: continue
    s=f.read_text(encoding='utf-8-sig')
    t=s.replace('ShellLayoutService','ViewerUiManager').replace('_host.ShellLayout','_host.UiManager')
    t=t.replace('IViewerAppHost.ShellLayout => _shellLayout','IViewerAppHost.UiManager => _uiManager').replace('ViewerUiManager ShellLayout {','ViewerUiManager UiManager {')
    if rel=='ViewerApp_Host.cs':t=t.replace('_shellLayout','_uiManager')
    if t!=s:f.write_text(t,encoding='utf-8')

# Explicit owned-service navigation; no enum translating overloads remain.
p='Workbench/Services/Pm4Workbench/Pm4WorkbenchService.cs';s=read(p)
s=replace_method(s,'OpenPm4Workbench','''    internal void OpenPm4Workbench(Pm4WorkbenchTab tab)
    {
        _host.UiManager.OpenViewer(ViewerPanel.PM4);
        _pendingPm4WorkbenchTab = tab;
    }
''');write(p,s)
p='Workbench/Services/Investigation/InvestigationService.Investigation.cs';s=read(p).replace('OpenWorkbenchTab(WoWViewer.Workbench.WorkbenchTab.Inspect)', '_host.UiManager.OpenViewer(WoWViewer.Workbench.ViewerPanel.Inspector)');write(p,s)
p='Workbench/Services/CameraPaths/CameraPathsService.cs';s=read(p)
s=replace_method(s,'OpenCapturePanelTab','''    internal void OpenCapturePanelTab(CapturePanelTab tab)
    {
        _pendingCapturePanelTab = tab;
        _activeCapturePanelTabIndex = (int)tab;
        _host.UiManager.OpenSceneTools(1);
    }
''');write(p,s)
p='Workbench/Services/Workspaces/WorkspacesService.cs';s=read(p)
s=replace_method(s,'SetWorkspaceMode','''    internal void SetWorkspaceMode(WorkspaceMode mode)
    {
        _host.UiManager.Navigate(mode switch
        {
            WorkspaceMode.Editor => WorkbenchTab.Editor,
            WorkspaceMode.Archaeology => WorkbenchTab.Archaeology,
            _ => WorkbenchTab.Viewer,
        });
    }
''')
s=s.replace('_workbenchPanels.OpenWorkbenchTab(UtilitiesBottomTab.Log)','_host.ShowLogViewer = true').replace('_workbenchPanels.OpenWorkbenchTab(UtilitiesBottomTab.Perf)','_host.ShowPerfWindow = true')
write(p,s)
p='ViewerKeyBindings.cs';s=read(p)
s=replace_method(s,'GetActiveKeyContext','''    private ViewerKeyContext GetActiveKeyContext()
    {
        if (_uiManager.IsCaptureVisible) return ViewerKeyContext.Capture;
        if (_uiManager.IsAudioVisible) return ViewerKeyContext.Audio;
        return _activeTopTab switch
        {
            Workbench.WorkbenchTab.Editor => ViewerKeyContext.Editor,
            Workbench.WorkbenchTab.Archaeology => ViewerKeyContext.Archaeology,
            _ => ViewerKeyContext.Inspect,
        };
    }
''');write(p,s)

# The final empty compatibility-state partial file has no survivor.
(root/'ViewerApp_Sidebars.cs').unlink()

# Settings no longer reads/writes legacy shell geometry/navigation. Manager loads after this service.
p='Workbench/Services/Settings/ViewerSettingsService.cs';s=read(p)
s=re.sub(r'^    private[^\n]*(?:Pm4ObjectMatchStore|Pm4ObjectMatchEntry|NormalizeWorkbenchStateAfterLoad)[^\n]*\n','',s,flags=re.M)
s=re.sub(r'            // Load the human-curated PM4 object.*?\n\n','',s,flags=re.S)
s=re.sub(r'            _useDockspaceUi = settings.ShellPanelLayoutVersion.*?settings.UseDockspaceUi;\n','',s,flags=re.S)
s=re.sub(r'            _useTabUi = settings.UseTabUi;.*?            _showWorkspaceBarsPanel = settings.ShowWorkspaceBarsPanel;\n','            // Legacy navigation/layout intentionally reset by ViewerUiManager.LoadLayout.\n',s,flags=re.S)
s=re.sub(r'            _leftSidebarWidth = float.IsFinite\(settings.LeftSidebarWidth\).*?                : DefaultBottomDrawerHeight;\n','',s,flags=re.S)
s=re.sub(r'                        _savedShellPanelLayouts.Clear\(\);.*?\n            ApplySavedPm4AlignmentToScene\(\);','            ApplySavedPm4AlignmentToScene();',s,flags=re.S)
s=re.sub(r'                ShellPanelLayouts = _savedShellPanelLayouts.Values.*?                    .ToList\(\),\n','',s,flags=re.S)
s=re.sub(r'^                (?:UseDockspaceUi|ShowLeftSidebar|ShowRightSidebar|ShowWorkspaceBarsPanel|LeftSidebarWidth|RightSidebarWidth|BottomDrawerHeight|UseTabUi|WorkbenchNavigationVersion|ActiveTopTab|ActiveBottomTab) = [^\n]*\n','',s,flags=re.M)
s=s.replace('ArcheologyApplyToVideoRecording = _archeologyApplyToVideoRecording,\n            };','ArcheologyApplyToVideoRecording = _archeologyApplyToVideoRecording\n            };')
write(p,s)

# Delete all translating host bridges, deleted providers, and curation bridges rather than stubbing.
bad=r'OpenWorkbenchTab|NormalizeWorkbenchStateAfterLoad|LegacyUtilityPage|Pm4ObjectMatchStore|Pm4ObjectMatchEntr|TrySavePm4ObjectMatch|DrawDockedShellPanelsForLane|DrawArcheologySubTabContent|DrawArchaeologyQuickSection'
for f in [root/'ViewerApp_Host.cs',root/'Workbench/Services/IViewerAppHost.cs',*root.glob('Workbench/Services/*/*.Host.cs')]:
    s=f.read_text(encoding='utf-8-sig')
    s=re.sub(r'^    [^\n]*(?:'+bad+r')[^\n]*\n(?:        => [^\n]*\n)?','',s,flags=re.M)
    f.write_text(s,encoding='utf-8')

# Narrow mechanical method replacement in menu. Content bodies remain untouched.
p='Workbench/Services/MainMenu/MainMenuBarService.cs';s=read(p).replace('_shellLayout.ResetShellLayoutToDefaults()', '_shellLayout.ResetLayout()');write(p,s)

# The already-removed Chrome popup opener is now a request to the registry, no new Chrome body.
for p in ['Workbench/Services/ModelInspector/ModelInspectorPanelService.Sidebars.cs','Workbench/Services/SqlSpawnStreaming/SqlSpawnStreamingService.cs','Workbench/Services/WorldObjects/WorldObjectsPanelService.cs']:
    s=read(p)
    s=s.replace('_viewerChrome.OpenTransparencyPopup()', '_host.UiManager.RequestOpacity()')
    s=s.replace('_host.ViewerChrome.OpenTransparencyPopup()', '_host.UiManager.RequestOpacity()')
    s=s.replace('_host.OpenTransparencyPopup()', '_host.UiManager.RequestOpacity()')
    s=s.replace('OpenTransparencyPopup()', '_host.UiManager.RequestOpacity()')
    write(p,s)

# Prune unused one-line private forwarding declarations in the migrated partial provider bridges.
for folder in ['WorkbenchPanels','Navigator','MainMenu','Chrome','Pm4Workbench','Investigation']:
    host=root/f'Workbench/Services/{folder}/{folder}Service.Host.cs'
    if not host.exists():
        host=next((root/f'Workbench/Services/{folder}').glob('*.Host.cs'),None)
    if host is None:continue
    siblings='\n'.join(f.read_text(encoding='utf-8-sig') for f in host.parent.glob('*.cs') if f!=host)
    s=host.read_text(encoding='utf-8-sig')
    lines=[]
    for line in s.splitlines(True):
        if re.match(r'    private .*=>',line):
            prefix=line.split('=>')[0]
            m=re.search(r'\b(\w+)\s*(?:\([^\n]*\))?\s*$',prefix)
            if m and not re.search(r'\b'+m[1]+r'\b',siblings):continue
        lines.append(line)
    host.write_text(''.join(lines),encoding='utf-8')

# Removed editor export/matching bridges have no page consumer now.
p='Workbench/Pages/ViewerAppContext.cs';s=read(p)
s=re.sub(r'^    void (?:DrawPm4Exports|OpenPm4Workbench)\(\);\n','',s,flags=re.M)
write(p,s)
print('Navigation routes and settings reset migrated; obsolete forwarding bridges removed.')
