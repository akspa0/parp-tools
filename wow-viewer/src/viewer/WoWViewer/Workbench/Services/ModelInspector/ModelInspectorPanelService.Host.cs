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
using WowViewer.Core.Runtime.World.Passes;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ModelInspectorPanelService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class ModelInspectorPanelService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref bool _autoFrameModelOnLoad => ref _host.AutoFrameModelOnLoad;
    private ref Camera _camera => ref _host.Camera;
    private ref bool _hasExplicitWmoMliqRotationOverride => ref _host.HasExplicitWmoMliqRotationOverride;
    private HashSet<int> _highlightedStandaloneWmoGroupIndices => _host.HighlightedStandaloneWmoGroupIndices;
    private ref int _hoveredStandaloneWmoGroupIndex => ref _host.HoveredStandaloneWmoGroupIndex;
    private ref string? _lastVirtualPath => ref _host.LastVirtualPath;
    private ref string? _loadedFilePath => ref _host.LoadedFilePath;
    private ref string _modelInfo => ref _host.ModelInfo;
    private StandaloneModelLoaderService _modelLoader => _host.ModelLoader;
    private ref ISceneRenderer? _renderer => ref _host.Renderer;
    private ref int _selectedStandaloneWmoGroupIndex => ref _host.SelectedStandaloneWmoGroupIndex;
    private SqlSpawnStreamingService _sqlSpawnStreaming => _host.SqlSpawnStreaming;
    private ref bool _standaloneWmoGroupLabelsAllEnabled => ref _host.StandaloneWmoGroupLabelsAllEnabled;
    private ref bool _standaloneWmoGroupOverlayEnabled => ref _host.StandaloneWmoGroupOverlayEnabled;
    private ref bool _standaloneWmoOverlayIncludeHiddenGroups => ref _host.StandaloneWmoOverlayIncludeHiddenGroups;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private void DrawAssetPathActions(string label, string assetPath, string idSuffix) => _host.DrawAssetPathActions(label, assetPath, idSuffix);
    private void DrawToolbarPopupButton(string label, string summary, string popupId, Action drawContent) => _host.DrawToolbarPopupButton(label, summary, popupId, drawContent);
    private void FramePoint(Vector3 target, float radius = 2f) => _host.FramePoint(target, radius);
    private void NormalizeStandaloneWmoGroupSelection(WmoRenderer wmoRenderer) => _host.NormalizeStandaloneWmoGroupSelection(wmoRenderer);
    private void ToggleStandaloneWmoGroupHighlight(int renderGroupIndex) => _host.ToggleStandaloneWmoGroupHighlight(renderGroupIndex);
}
