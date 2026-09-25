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

// CascAhdrSourceService host bridge. Its member types are declared in ViewerApp.cs, so this file carries
// ViewerApp.cs's using directives; the moved members keep their original file's usings unchanged.
internal sealed partial class CascAhdrSourceService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref AreaTableService? _areaTableService => ref _host.AreaTableService;
    private ref AssetCatalogView? _catalogView => ref _host.CatalogView;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref DBCD.Providers.IDBCProvider? _dbcProvider => ref _host.DbcProvider;
    private ref string? _dbdDir => ref _host.DbdDir;
    private ref List<MapDefinition> _discoveredMaps => ref _host.DiscoveredMaps;
    private HashSet<string> _loggedStandaloneMissingSkinPaths => _host.LoggedStandaloneMissingSkinPaths;
    private ProjectOutputService _projectOutput => _host.ProjectOutput;
    private Dictionary<string, string?> _standaloneSkinPathCache => _host.StandaloneSkinPathCache;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref ReplaceableTextureResolver? _texResolver => ref _host.TexResolver;
    private WdlPreviewService _wdlPreview => _host.WdlPreview;
    private WorldLoaderService _worldLoader => _host.WorldLoader;
}
