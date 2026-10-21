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
using WowViewer.Core.Runtime.PromoVideo;
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
using WoWViewer.UI;
using WowViewer.Core.Runtime.World.Minimap;
using System.IO;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Runtime.World.Passes;
using static WoWViewer.ViewerApp;
using static WoWViewer.MinimapAndStatusService;

namespace WoWViewer;

// NavigatorPanelService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class NavigatorPanelService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private CascAhdrSourceService _cascAhdrSource => _host.CascAhdrSource;
    private ref string _currentAreaName => ref _host.CurrentAreaName;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref List<MapDefinition> _discoveredMaps => ref _host.DiscoveredMaps;
    private ref string _extensionFilter => ref _host.ExtensionFilter;
    private ref List<string> _filteredFiles => ref _host.FilteredFiles;
    private ref string? _loadedFileName => ref _host.LoadedFileName;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref MapListSortMode _mapListSortMode => ref _host.MapListSortMode;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private ref string _searchFilter => ref _host.SearchFilter;
    private ref int _selectedFileIndex => ref _host.SelectedFileIndex;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TaxiAndAreaPoiSelectionService _taxiAndAreaPoi => _host.TaxiAndAreaPoi;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private WdlPreviewService _wdlPreview => _host.WdlPreview;
    private ref WdlPreviewCacheService? _wdlPreviewCacheService => ref _host.WdlPreviewCacheService;
    private WorldLoaderService _worldLoader => _host.WorldLoader;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void ClearSelectedWlLiquidBody(bool clearListIsolation) => _host.ClearSelectedWlLiquidBody(clearListIsolation);
    private void DrawMapSortModeSelector(string id) => _host.DrawMapSortModeSelector(id);
    private void RefreshSelectedWorldObjectInfo() => _host.RefreshSelectedWorldObjectInfo();
}
