using System.Numerics;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using ImGuiNET;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Capture;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Runtime.Marketing;
using WoWViewer.Terrain.Vlm;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Catalog;
using WoWViewer.Population;
using Silk.NET.Input;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.Maps;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// CaptureAutomationService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class CaptureAutomationService
{
    // Host bridge (same names as the former ViewerApp members).
    private ArchaeologyPanelService _archaeologyPanel => _host.ArchaeologyPanel;
    private ref bool _archeologyApplyToNextCapture => ref _host.ArcheologyApplyToNextCapture;
    private ref bool _archeologyApplyToVideoRecording => ref _host.ArcheologyApplyToVideoRecording;
    private ref bool _archeologyPlaybackActive => ref _host.ArcheologyPlaybackActive;
    private ref Camera _camera => ref _host.Camera;
    private CameraPathsService _cameraPaths => _host.CameraPaths;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private DatasetExportDialogsService _datasetExportDialogs => _host.DatasetExportDialogs;
    private ref string? _dbcBuild => ref _host.DbcBuild;
    private ref float _fovDegrees => ref _host.FovDegrees;
    private ref GL _gl => ref _host.Gl;
    private ref bool _hideUiChrome => ref _host.HideUiChrome;
    private ref string? _lastWorldSceneWdtPath => ref _host.LastWorldSceneWdtPath;
    private ref int _mkHarvestViewerValidationCompleted => ref _host.MkHarvestViewerValidationCompleted;
    private ref int _mkHarvestViewerValidationFailed => ref _host.MkHarvestViewerValidationFailed;
    private NavigatorPanelService _navigatorPanel => _host.NavigatorPanel;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ref MapDefinition? _selectedMapForPreview => ref _host.SelectedMapForPreview;
    private ShellLayoutService _shellLayout => _host.ShellLayout;
    private ref bool _showCaptureAutomationWindow => ref _host.ShowCaptureAutomationWindow;
    private StartupAutomationService _startupAutomation => _host.StartupAutomation;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private TerrainQueryService _terrainQuery => _host.TerrainQuery;
    private ref ReplaceableTextureResolver? _texResolver => ref _host.TexResolver;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref IWindow _window => ref _host.Window;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
}
