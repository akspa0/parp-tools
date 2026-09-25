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
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using WowViewer.Core.Runtime.World;
using System;
using static WoWViewer.ViewerApp;
using static WoWViewer.InvestigationService;

namespace WoWViewer;

// ViewerChromeService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class ViewerChromeService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string _folderInputBuf => ref _host.FolderInputBuf;
    private ref string _lastGameFolderPath => ref _host.LastGameFolderPath;
    private ref float _leftSidebarWidth => ref _host.LeftSidebarWidth;
    private ref string? _loadedFileName => ref _host.LoadedFileName;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref MdxFile? _loadedMdx => ref _host.LoadedMdx;
    private ref WmoV14ToV17Converter.WmoV14Data? _loadedWmo => ref _host.LoadedWmo;
    private ModelInspectorPanelService _modelInspector => _host.ModelInspector;
    private Pm4WorkbenchService _pm4Workbench => _host.Pm4Workbench;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ref float _rightSidebarWidth => ref _host.RightSidebarWidth;
    private ViewerSettingsService _settings => _host.Settings;
    private ShellLayoutService _shellLayout => _host.ShellLayout;
    private ref bool _showFolderInput => ref _host.ShowFolderInput;
    private ref bool _showSettingsWindow => ref _host.ShowSettingsWindow;
    private ref bool _standaloneWmoGroupLabelsAllEnabled => ref _host.StandaloneWmoGroupLabelsAllEnabled;
    private ref bool _standaloneWmoGroupOverlayEnabled => ref _host.StandaloneWmoGroupOverlayEnabled;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TerrainControlsPanelService _terrainControlsPanel => _host.TerrainControlsPanel;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref bool _useDockspaceUi => ref _host.UseDockspaceUi;
    private ref bool _useTabUi => ref _host.UseTabUi;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref bool _wantOpenFile => ref _host.WantOpenFile;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void DrawVisualInvestigationModeButton(VisualInvestigationMode mode, string label, string tooltip) => _host.DrawVisualInvestigationModeButton(mode, label, tooltip);
    private float GetTopChromeHeight() => _host.GetTopChromeHeight();
    private string GetWorkspaceSaveStatusSummary() => _host.GetWorkspaceSaveStatusSummary();
    private string GetWorkspaceTargetSummary() => _host.GetWorkspaceTargetSummary();
}
