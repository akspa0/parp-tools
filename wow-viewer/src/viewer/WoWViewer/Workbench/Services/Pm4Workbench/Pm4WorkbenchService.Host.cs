using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
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

// Pm4WorkbenchService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class Pm4WorkbenchService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref FixedBottomDrawerTab _activeBottomDrawerTab => ref _host.ActiveBottomDrawerTab;
    private ref int _activePm4TabIndex => ref _host.ActivePm4TabIndex;
    private ref Camera _camera => ref _host.Camera;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref Pm4ObjectMatchObject? _hoveredPm4ObjectMatch => ref _host.HoveredPm4ObjectMatch;
    private ref int _hoveredPm4ObjectMatchCacheMaxMatches => ref _host.HoveredPm4ObjectMatchCacheMaxMatches;
    private ref (int tileX, int tileY, uint ck24, int objectPart)? _hoveredPm4ObjectMatchKey => ref _host.HoveredPm4ObjectMatchKey;
    private ref FixedBottomDrawerTab? _pendingRightSidebarSection => ref _host.PendingRightSidebarSection;
    private ref int _pm4ObjectMatchMaxMatchesPerObject => ref _host.Pm4ObjectMatchMaxMatchesPerObject;
    private ref Vector3 _pm4SavedOverlayRotationDegrees => ref _host.Pm4SavedOverlayRotationDegrees;
    private ref Vector3 _pm4SavedOverlayScale => ref _host.Pm4SavedOverlayScale;
    private ref Vector3 _pm4SavedOverlayTranslation => ref _host.Pm4SavedOverlayTranslation;
    private ref Dictionary<string, Pm4WmoMatchEntry> _pm4WmoMatchEntries => ref _host.Pm4WmoMatchEntries;
    private ref Pm4WmoMatchStore? _pm4WmoMatchStore => ref _host.Pm4WmoMatchStore;
    private Dictionary<string, SavedPm4ObjectMatchSelection> _savedPm4ObjectMatches => _host.SavedPm4ObjectMatches;
    private ViewerSettingsService _settings => _host.Settings;
    private ShellLayoutService _shellLayout => _host.ShellLayout;
    private ref bool _showPerfWindow => ref _host.ShowPerfWindow;
    private ref bool _showRightSidebar => ref _host.ShowRightSidebar;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorkspaceMode _workspaceMode => ref _host.WorkspaceMode;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void OpenWorkbenchTab(WorkbenchTab topTab, int bottomIndex = -1) => _host.OpenWorkbenchTab(topTab, bottomIndex);
    private void OpenWorkbenchTab(ModelBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void OpenWorkbenchTab(WorldBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void OpenWorkbenchTab(ToolsBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void OpenWorkbenchTab(UtilitiesBottomTab tab) => _host.OpenWorkbenchTab(tab);
    private void SetEditorWorkspaceTask(EditorWorkspaceTask task) => _host.SetEditorWorkspaceTask(task);
}
