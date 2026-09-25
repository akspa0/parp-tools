using System;
using ImGuiNET;
using WoWViewer.Terrain;
using WoWViewer.Rendering;
using WowViewer.Core.Maps;
using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
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
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// SettingsWindowService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class SettingsWindowService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref CameraHudRig? _cameraHudRig => ref _host.CameraHudRig;
    private ref float _cameraSpeed => ref _host.CameraSpeed;
    private DatasetCatalogService _datasetCatalog => _host.DatasetCatalog;
    private ref float _fovDegrees => ref _host.FovDegrees;
    private RenderQualityService _renderQuality => _host.RenderQuality;
    private ref SceneCursorRenderer? _sceneCursorRenderer => ref _host.SceneCursorRenderer;
    private ViewerSettingsService _settings => _host.Settings;
    private ref bool _showMinimapWindow => ref _host.ShowMinimapWindow;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref float _uiFontScale => ref _host.UiFontScale;
    private ref bool _useTabUi => ref _host.UseTabUi;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
}
