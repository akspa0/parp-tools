using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Vlm;

namespace WoWViewer;

// Existing host access only; chrome layout and popup bridges are intentionally absent.
internal sealed partial class ViewerChromeService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string _folderInputBuf => ref _host.FolderInputBuf;
    private ref string _lastGameFolderPath => ref _host.LastGameFolderPath;
    private ref MdxFile? _loadedMdx => ref _host.LoadedMdx;
    private ref WmoV14ToV17Converter.WmoV14Data? _loadedWmo => ref _host.LoadedWmo;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ViewerSettingsService _settings => _host.Settings;
    private ref SceneCursorRenderer? _sceneCursorRenderer => ref _host.SceneCursorRenderer;
    private ref bool _showFolderInput => ref _host.ShowFolderInput;
    private ref bool _standaloneWmoGroupLabelsAllEnabled => ref _host.StandaloneWmoGroupLabelsAllEnabled;
    private ref bool _standaloneWmoGroupOverlayEnabled => ref _host.StandaloneWmoGroupOverlayEnabled;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TerrainControlsPanelService _terrainControlsPanel => _host.TerrainControlsPanel;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref bool _wantOpenFile => ref _host.WantOpenFile;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private string GetWorkspaceSaveStatusSummary() => _host.GetWorkspaceSaveStatusSummary();
    private string GetWorkspaceTargetSummary() => _host.GetWorkspaceTargetSummary();
}
