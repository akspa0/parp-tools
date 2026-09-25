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
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Synthesized minimap export dialog and job.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class SynthesizedMinimapExportService
{
    private readonly IViewerAppHost _host;

    internal SynthesizedMinimapExportService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ProjectOutputService _projectOutput => _host.ProjectOutput;
    private ref bool _showSynthesizedMinimapExportDialog => ref _host.ShowSynthesizedMinimapExportDialog;

    private string _synthesizedMinimapClientRoot = string.Empty;
    private string _synthesizedMinimapMapName = string.Empty;
    private string _synthesizedMinimapOutputDirectory = string.Empty;
    private float _synthesizedMinimapTimeHours = 12f;
    private int _synthesizedMinimapHour = 12;
    private int _synthesizedMinimapMinute;
    private int _synthesizedMinimapResolution = 256;
    private bool _synthesizedMinimapEmitTiles = true;
    private bool _synthesizedMinimapEmitWholeMap = true;
    private bool _synthesizedMinimapIncludeWmos;
    private bool _synthesizedMinimapBakeMcsh;
    private bool _synthesizedMinimapCastShadows = true;
    private bool _synthesizedMinimapRunning;
    private bool _synthesizedMinimapDone;
    private string? _synthesizedMinimapError;
    private readonly List<string> _synthesizedMinimapLog = new();
    private bool _synthesizedMinimapScrollToBottom;
}
