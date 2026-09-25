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
/// Terrain analysis: hidden-terrain candidates, tile previews/similarity, global bounds and analysis exports.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed partial class TerrainAnalysisService
{
    private readonly IViewerAppHost _host;

    internal TerrainAnalysisService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref GL _gl => ref _host.Gl;
    private ProjectOutputService _projectOutput => _host.ProjectOutput;
    private ref TerrainAnalysisPreviewTexture? _terrainAnalysisAlphaTexture => ref _host.TerrainAnalysisAlphaTexture;
    private ref TerrainAnalysisPreviewTexture? _terrainAnalysisGlobalTexture => ref _host.TerrainAnalysisGlobalTexture;
    private List<TerrainHiddenTileCandidate> _terrainAnalysisHiddenCandidates => _host.TerrainAnalysisHiddenCandidates;
    private ref int _terrainAnalysisHiddenSelectedIndex => ref _host.TerrainAnalysisHiddenSelectedIndex;
    private ref string _terrainAnalysisHiddenStatus => ref _host.TerrainAnalysisHiddenStatus;
    private ref TerrainAnalysisPreviewTexture? _terrainAnalysisLocalTexture => ref _host.TerrainAnalysisLocalTexture;
    private ref (int tileX, int tileY)? _terrainAnalysisPreviewCompareTile => ref _host.TerrainAnalysisPreviewCompareTile;
    private ref float? _terrainAnalysisPreviewSimilarity => ref _host.TerrainAnalysisPreviewSimilarity;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private TerrainTileIoService _terrainTileIo => _host.TerrainTileIo;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();

    private (int tileX, int tileY)? _terrainAnalysisPreviewTile;
    private float _terrainAnalysisPreviewTileMin;
    private float _terrainAnalysisPreviewTileMax;
    private float _terrainAnalysisPreviewVisibilityRatio;
    private float _terrainAnalysisPreviewAmplification = 1f;
    private float _terrainAnalysisGlobalMin;
    private float _terrainAnalysisGlobalMax;
    private int _terrainAnalysisGlobalTileCount;
    private TerrainTileScope _terrainAnalysisGlobalScope = TerrainTileScope.LoadedTiles;
    private bool _terrainAnalysisHasGlobalBounds;
    private bool _terrainAnalysisFollowCameraTile = true;
    private string _terrainAnalysisStatus = string.Empty;
    private int _terrainAnalysisHiddenCompareOffsetX;
    private int _terrainAnalysisHiddenCompareOffsetY = 2;
    private float _terrainAnalysisHiddenMinSimilarity = 0.85f;
    private float _terrainAnalysisHiddenMaxVisibilityRatio = 0.05f;
    private int _terrainAnalysisHiddenMaxResults = 24;
    private TerrainTileScope _terrainAnalysisHiddenScope = TerrainTileScope.LoadedTiles;
}
