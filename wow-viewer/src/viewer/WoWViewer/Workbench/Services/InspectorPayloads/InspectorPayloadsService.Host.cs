using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WowViewer.Core.Runtime.World.Inspection;
using WowViewer.Core.Wmo;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using System.Diagnostics;
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
using WowViewer.Core.IO.Converters;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// InspectorPayloadsService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class InspectorPayloadsService
{
    // Host bridge (same names as the former ViewerApp members).
    private InvestigationService _investigation => _host.Investigation;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ModelInspectorPanelService _modelInspector => _host.ModelInspector;
    private NavigatorPanelService _navigatorPanel => _host.NavigatorPanel;
    private Pm4WorkbenchService _pm4Workbench => _host.Pm4Workbench;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ref string _selectedObjectType => ref _host.SelectedObjectType;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TerrainInspectionPanelService _terrainInspection => _host.TerrainInspection;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref string _wlLayerSelectedBodyKey => ref _host.WlLayerSelectedBodyKey;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
}
