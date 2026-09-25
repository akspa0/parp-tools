using System.Numerics;
using ImGuiNET;
using WoWViewer.Rendering;
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
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// WmoGroupsPanelService host bridge. Bridged member types are declared across ViewerApp partial files, so this
// file carries those files' using directives; the moved members keep their original usings.
internal sealed partial class WmoGroupsPanelService
{
    // Host bridge (same names as the former ViewerApp members).
    private ref Terrain.BoundingBoxRenderer? _editorOverlayBb => ref _host.EditorOverlayBb;
    private ref GL _gl => ref _host.Gl;
    private HashSet<int> _highlightedStandaloneWmoGroupIndices => _host.HighlightedStandaloneWmoGroupIndices;
    private ref int _hoveredStandaloneWmoGroupIndex => ref _host.HoveredStandaloneWmoGroupIndex;
    private ref float _lastMouseX => ref _host.LastMouseX;
    private ref float _lastMouseY => ref _host.LastMouseY;
    private ref int _selectedStandaloneWmoGroupIndex => ref _host.SelectedStandaloneWmoGroupIndex;
    private ShellLayoutService _shellLayout => _host.ShellLayout;
    private ref bool _standaloneWmoGroupLabelsAllEnabled => ref _host.StandaloneWmoGroupLabelsAllEnabled;
    private ref bool _standaloneWmoGroupOverlayEnabled => ref _host.StandaloneWmoGroupOverlayEnabled;
    private ref bool _standaloneWmoOverlayIncludeHiddenGroups => ref _host.StandaloneWmoOverlayIncludeHiddenGroups;
}
