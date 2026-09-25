using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.Editor;
using WowViewer.Core.Editor.Bridge;
using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.Editor.Plugins;
using WowViewer.Core.Editor.Session;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Reconciliation;
using WowViewer.Core.PM4.Services;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Terrain;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
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
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// EditorPanelsService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class EditorPanelsService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref EditorHost? _editorHost => ref _host.EditorHost;
    private ref EditorSession? _editorSession => ref _host.EditorSession;
    private ref EditorWorkspaceTask _editorWorkspaceTask => ref _host.EditorWorkspaceTask;
    private ref TerrainTileScope _mapGlbScope => ref _host.MapGlbScope;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private PlacementEditService _placementEditing => _host.PlacementEditing;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ref bool _showSynthesizedMinimapExportDialog => ref _host.ShowSynthesizedMinimapExportDialog;
    private ref string _statusMessage => ref _host.StatusMessage;
    private SynthesizedMinimapExportService _synthesizedMinimapExport => _host.SynthesizedMinimapExport;
    private ref TerrainExportKind _terrainExportKind => ref _host.TerrainExportKind;
    private ref TerrainImportKind _terrainImportKind => ref _host.TerrainImportKind;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref bool _wantExportGlb => ref _host.WantExportGlb;
    private ref bool _wantExportGlbCollision => ref _host.WantExportGlbCollision;
    private ref bool _wantExportMapGlbTiles => ref _host.WantExportMapGlbTiles;
    private ref bool _wantTerrainExport => ref _host.WantTerrainExport;
    private ref bool _wantTerrainImport => ref _host.WantTerrainImport;
    private WorkspacesService _workspaces => _host.Workspaces;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void EnsureEditorHost() => _host.EnsureEditorHost();
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();
}
