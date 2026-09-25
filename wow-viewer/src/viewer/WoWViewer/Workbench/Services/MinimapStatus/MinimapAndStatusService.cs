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

/// <summary>
/// Minimap window and status bar content: minimap rendering controls, fullscreen minimap, status text.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class MinimapAndStatusService
{
    private readonly IViewerAppHost _host;

    internal MinimapAndStatusService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref Camera _camera => ref _host.Camera;
    private ref WowViewer.Core.World.AreaLookupResult? _currentAreaLookup => ref _host.CurrentAreaLookup;
    private ref double _currentFps => ref _host.CurrentFps;
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref List<MapDefinition> _discoveredMaps => ref _host.DiscoveredMaps;
    private ref string _folderInputBuf => ref _host.FolderInputBuf;
    private ref bool _fullscreenMinimap => ref _host.FullscreenMinimap;
    private ref string _lastGameFolderPath => ref _host.LastGameFolderPath;
    private ref Vector2 _minimapPanOffset => ref _host.MinimapPanOffset;
    private ref MinimapRenderer? _minimapRenderer => ref _host.MinimapRenderer;
    private ref float _minimapZoom => ref _host.MinimapZoom;
    private ref int _pendingMinimapTeleportClickCount => ref _host.PendingMinimapTeleportClickCount;
    private ref (int tileX, int tileY)? _pendingMinimapTeleportTile => ref _host.PendingMinimapTeleportTile;
    private ref float _rightSidebarWidth => ref _host.RightSidebarWidth;
    private ShellLayoutService _shellLayout => _host.ShellLayout;
    private ref bool _showFolderInput => ref _host.ShowFolderInput;
    private ref bool _showMinimapWindow => ref _host.ShowMinimapWindow;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref bool _useDockspaceUi => ref _host.UseDockspaceUi;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref bool _wantOpenFile => ref _host.WantOpenFile;
    private ref WorldScene? _worldScene => ref _host.WorldScene;

    private bool _minimapDragging = false;
    private DateTime _pendingMinimapTeleportLastClickUtc = DateTime.MinValue;
}
