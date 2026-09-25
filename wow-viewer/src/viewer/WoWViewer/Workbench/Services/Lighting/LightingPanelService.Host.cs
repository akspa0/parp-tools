using System;
using System.Numerics;
using ImGuiNET;
using WoWViewer.Terrain;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
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
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using WoWViewer.UI;
using WowViewer.Core.Runtime.World.Minimap;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// LightingPanelService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class LightingPanelService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void DrawAuthoritativeFogControls(bool showDescription = true) => _host.DrawAuthoritativeFogControls(showDescription);
    private void FocusCameraOnLitLight(int lightIndex, bool closeFullscreenAfterFocus) => _host.FocusCameraOnLitLight(lightIndex, closeFullscreenAfterFocus);
}
