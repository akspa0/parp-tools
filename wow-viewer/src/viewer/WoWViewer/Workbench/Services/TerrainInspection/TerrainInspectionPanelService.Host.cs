using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.Runtime.World.Inspection;
using WowViewer.Core.World;
using WoWViewer.Terrain;
using WoWViewer.Rendering;
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
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using WowViewer.Core.Runtime.World;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// TerrainInspectionPanelService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class TerrainInspectionPanelService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref AreaTableService? _areaTableService => ref _host.AreaTableService;
    private ref Camera _camera => ref _host.Camera;
    private ref string _currentAreaName => ref _host.CurrentAreaName;
    private ref int _currentMapId => ref _host.CurrentMapId;
    private ref int _selectedObjectIndex => ref _host.SelectedObjectIndex;
    private ref string _selectedObjectInfo => ref _host.SelectedObjectInfo;
    private ref string _selectedObjectType => ref _host.SelectedObjectType;
    private ref string _statusMessage => ref _host.StatusMessage;
    private TaxiAndAreaPoiSelectionService _taxiAndAreaPoi => _host.TaxiAndAreaPoi;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private TerrainQueryService _terrainQuery => _host.TerrainQuery;
    private TerrainWeakSignalRestoreService _terrainWeakSignalRestore => _host.TerrainWeakSignalRestore;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void ClearSelectedWlLiquidBody(bool clearListIsolation) => _host.ClearSelectedWlLiquidBody(clearListIsolation);
}
