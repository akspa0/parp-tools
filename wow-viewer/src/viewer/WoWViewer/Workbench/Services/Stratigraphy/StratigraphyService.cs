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
/// Terrain stratigraphy analysis: per-tile and all-loaded-tiles analysis, and exporting analysed stratigraphy tiles.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class StratigraphyService
{
    private readonly IViewerAppHost _host;

    internal StratigraphyService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private DataSourceSessionService _dataSourceSession => _host.DataSourceSession;
    private ref string _statusMessage => ref _host.StatusMessage;
    private Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyTileAnalysis> _stratigraphyTileAnalyses => _host.StratigraphyTileAnalyses;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref string _terrainWeakSignalRestoreStatus => ref _host.TerrainWeakSignalRestoreStatus;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();

    private string _stratigraphySaveOutputDirectory = string.Empty;

    internal void AnalyzeActiveCameraTileStratigraphy()
    {
        var cameraTile = GetCameraTile();
        int tileX = cameraTile.tileX;
        int tileY = cameraTile.tileY;

        IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
        if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
            chunks = result.Chunks;
        else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var vlmResult))
            chunks = vlmResult.Chunks;

        if (chunks == null || chunks.Count == 0)
        {
            _terrainWeakSignalRestoreStatus = $"Tile ({tileY}, {tileX}) is not currently loaded.";
            return;
        }

        var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
        float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);

        var holeMasks = new ushort[256];
        for (int i = 0; i < Math.Min(chunks.Count, 256); i++)
            holeMasks[i] = (ushort)chunks[i].HoleMask;

        string tileName = $"tile_{tileX}_{tileY}";
        var analysis = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyLevelAnalyzer.AnalyzeTile(lattice257, holeMasks, tileX, tileY, tileName);
        _stratigraphyTileAnalyses[(tileX, tileY)] = analysis;

        _terrainWeakSignalRestoreStatus = $"Tile ({tileY}, {tileX}) analyzed: {analysis.DominantStratum}, {analysis.TotalSurvivingLevels:N0} levels, {analysis.SqueezedChunkCount} squeezed chunks, {analysis.HoledChunkCount} dev mesh chunks.";
    }

    internal void AnalyzeAllLoadedTilesStratigraphy()
    {
        var loadedTiles = new HashSet<(int tileX, int tileY)>();
        if (_terrainManager != null)
        {
            foreach (var key in _terrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }
        if (_vlmTerrainManager != null)
        {
            foreach (var key in _vlmTerrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }

        int count = 0;
        int squeezedTotal = 0;
        int holedTotal = 0;

        foreach (var (tileX, tileY) in loadedTiles)
        {
            IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
            if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tileX, tileY, out var result))
                chunks = result.Chunks;
            else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var vlmResult))
                chunks = vlmResult.Chunks;

            if (chunks == null || chunks.Count == 0) continue;

            var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);

            var holeMasks = new ushort[256];
            for (int i = 0; i < Math.Min(chunks.Count, 256); i++)
                holeMasks[i] = (ushort)chunks[i].HoleMask;

            var analysis = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.StratigraphyLevelAnalyzer.AnalyzeTile(lattice257, holeMasks, tileX, tileY, $"tile_{tileX}_{tileY}");
            _stratigraphyTileAnalyses[(tileX, tileY)] = analysis;

            count++;
            squeezedTotal += analysis.SqueezedChunkCount;
            holedTotal += analysis.HoledChunkCount;
        }

        _terrainWeakSignalRestoreStatus = $"Analyzed {count} loaded tile(s): {squeezedTotal} squeezed chunks, {holedTotal} dev mesh chunks across scene.";
    }

    internal void OpenStratigraphySaveDialog()
    {
        string initial = string.IsNullOrEmpty(_stratigraphySaveOutputDirectory)
            ? Directory.GetCurrentDirectory()
            : _stratigraphySaveOutputDirectory;

        ImGuiPathPicker.Instance.Open(
            "Select Output Directory to Save Restored ADT / WDT Tiles",
            pickFolder: true,
            initialPath: initial,
            filterExtension: null,
            selectedPath =>
            {
                if (!string.IsNullOrEmpty(selectedPath))
                {
                    _stratigraphySaveOutputDirectory = selectedPath;
                    ExportLoadedStratigraphyTiles(selectedPath);
                }
            });
    }

    private void ExportLoadedStratigraphyTiles(string outputDir)
    {
        if (string.IsNullOrWhiteSpace(outputDir)) return;
        Directory.CreateDirectory(outputDir);

        var loadedTiles = new HashSet<(int tileX, int tileY)>();
        if (_terrainManager != null)
        {
            foreach (var key in _terrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }
        if (_vlmTerrainManager != null)
        {
            foreach (var key in _vlmTerrainManager.LoadedTiles)
                loadedTiles.Add(key);
        }

        string mapName = _terrainManager?.MapName ?? _dataSourceSession.GetCurrentSessionMapName() ?? "CustomMap";
        string outputMapDir = Path.Combine(outputDir, "World", "Maps", mapName);
        Directory.CreateDirectory(outputMapDir);

        int exported = 0;
        foreach (var (tx, ty) in loadedTiles)
        {
            IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
            if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tx, ty, out var result))
                chunks = result.Chunks;
            else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tx, ty, out var vlmResult))
                chunks = vlmResult.Chunks;

            if (chunks == null || chunks.Count == 0) continue;

            var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
            float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);

            string outAdtPath = Path.Combine(outputMapDir, $"{mapName}_{tx}_{ty}.adt");
            float[] flat = WowViewer.Core.IO.Maps.StratigraphyTileExporter.FlattenHeights257(lattice257);

            var blankAdt = WowViewer.Core.IO.Maps.BlankAdtFactory.CreateBlank(mapName, tx, ty);
            WowViewer.Core.IO.Maps.LkAdtWriter.Write(outAdtPath, blankAdt);
            WowViewer.Core.IO.Maps.AdtTerrainWriter.Write(outAdtPath, outAdtPath, flat);
            exported++;
        }

        // Also write companion modified WDL file
        try
        {
            var wdlDict = new Dictionary<(int tileX, int tileY), WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlTileData>();
            foreach (var (tx, ty) in loadedTiles)
            {
                IReadOnlyList<Terrain.TerrainChunkData>? chunks = null;
                if (_terrainManager != null && _terrainManager.TryGetTileLoadResult(tx, ty, out var result))
                    chunks = result.Chunks;
                else if (_vlmTerrainManager != null && _vlmTerrainManager.TryGetTileLoadResult(tx, ty, out var vlmResult))
                    chunks = vlmResult.Chunks;

                if (chunks == null || chunks.Count == 0) continue;
                var tileHeightmap = Export.TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                float[,] lattice257 = WowViewer.Core.IO.Maps.StratigraphyTileExporter.ExpandHeights257(tileHeightmap.Heights);
                wdlDict[(tx, ty)] = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlFileWriter.FromLattice257(lattice257);
            }

            if (wdlDict.Count > 0)
            {
                byte[] wdlBytes = WowViewer.Core.Runtime.World.Terrain.Stratigraphy.WdlFileWriter.Write(wdlDict);
                string outWdlPath = Path.Combine(outputMapDir, $"{mapName}.wdl");
                File.WriteAllBytes(outWdlPath, wdlBytes);
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"Failed to write companion WDL: {ex.Message}");
        }

        _terrainWeakSignalRestoreStatus = $"Successfully exported {exported} restored tile(s) and companion WDL to '{outputMapDir}'.";
        _statusMessage = $"Exported {exported} restored stratigraphy tiles + WDL.";
    }
}
