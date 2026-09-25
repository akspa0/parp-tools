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
using WowViewer.Core.Runtime.World.Passes;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// InvestigationService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class InvestigationService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private NavigatorPanelService _navigatorPanel => _host.NavigatorPanel;
    private ref int _selectedObjectIndex => ref _host.SelectedObjectIndex;
    private ref string _selectedObjectInfo => ref _host.SelectedObjectInfo;
    private ref string _selectedObjectType => ref _host.SelectedObjectType;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private TerrainQueryService _terrainQuery => _host.TerrainQuery;
    private ref bool _useTabUi => ref _host.UseTabUi;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref bool _wlLayerListIsolationEnabled => ref _host.WlLayerListIsolationEnabled;
    private ref string _wlLayerSelectedBodyKey => ref _host.WlLayerSelectedBodyKey;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void DrawUnifiedInspectorContent() => _host.DrawUnifiedInspectorContent();
    private void OpenWorkbenchTab(WorkbenchTab topTab, int bottomIndex = -1) => _host.OpenWorkbenchTab(topTab, bottomIndex);
    private void OpenWorkbenchTab(ModelBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void OpenWorkbenchTab(WorldBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void OpenWorkbenchTab(ToolsBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void OpenWorkbenchTab(UtilitiesBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void SetEditorWorkspaceTask(EditorWorkspaceTask task) => _host.SetEditorWorkspaceTask(task);
}
