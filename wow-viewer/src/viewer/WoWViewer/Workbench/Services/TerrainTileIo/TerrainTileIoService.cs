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
/// Terrain tile import/export: GLB tiles, terrain export/import, tile-scope selection, Alpha atlas/chunk, MCCV and 257 heightmap round-trips.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class TerrainTileIoService
{
    private readonly IViewerAppHost _host;

    internal TerrainTileIoService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref TerrainTileScope _mapGlbScope => ref _host.MapGlbScope;
    private ref Md5TranslateIndex? _md5Index => ref _host.Md5Index;
    private ref bool _showAlphaFolderImportScope => ref _host.ShowAlphaFolderImportScope;
    private ref bool _showHeightmapFolderImportScope => ref _host.ShowHeightmapFolderImportScope;
    private ref bool _showMccvFolderImportScope => ref _host.ShowMccvFolderImportScope;
    private ref string _statusMessage => ref _host.StatusMessage;
    private ref TerrainExportKind _terrainExportKind => ref _host.TerrainExportKind;
    private ref TerrainImportKind _terrainImportKind => ref _host.TerrainImportKind;
    private ref TerrainManager? _terrainManager => ref _host.TerrainManager;
    private ref int _terrainTileRangeEndX => ref _host.TerrainTileRangeEndX;
    private ref int _terrainTileRangeEndY => ref _host.TerrainTileRangeEndY;
    private ref int _terrainTileRangeStartX => ref _host.TerrainTileRangeStartX;
    private ref int _terrainTileRangeStartY => ref _host.TerrainTileRangeStartY;
    private ref TerrainTileScope _terrainTileScope => ref _host.TerrainTileScope;
    private TerrainWeakSignalRestoreService _terrainWeakSignalRestore => _host.TerrainWeakSignalRestore;
    private ref VlmTerrainManager? _vlmTerrainManager => ref _host.VlmTerrainManager;
    private (int tileX, int tileY) GetCameraTile() => _host.GetCameraTile();

    private string _terrainImportFolder = "";
    private string _terrainCustomTilesText = "";

    internal void RunMapGlbTilesExport()
    {
        if (_terrainManager == null)
        {
            _statusMessage = "No terrain loaded.";
            return;
        }

        if (_dataSource == null)
        {
            _statusMessage = "No data source loaded (required to export textures/models).";
            return;
        }

        var tiles = GetTileScopeList(_mapGlbScope);
        if (tiles.Count == 0)
        {
            _statusMessage = "No tiles in scope.";
            return;
        }

        string outDir = Path.Combine(ExportDir, "map_glb", _terrainManager.MapName);
        Directory.CreateDirectory(outDir);

        int exported = 0;
        foreach (var (tileX, tileY) in tiles)
        {
            string outPath = Path.Combine(outDir, $"{_terrainManager.MapName}_{tileX:D2}_{tileY:D2}.glb");
            MapGlbExporter.ExportTile(_terrainManager, _dataSource, _md5Index, tileX, tileY, outPath, includePlacements: true);
            exported++;
        }

        _statusMessage = $"Exported {exported} tile GLB(s) to: {outDir}";
    }

    internal void RunTerrainExport()
    {
        try
        {
            switch (_terrainExportKind)
            {
                case TerrainExportKind.AlphaCurrentTileAtlas:
                    ExportAlphaCurrentTileAtlas();
                    break;
                case TerrainExportKind.AlphaCurrentTileChunksFolder:
                    ExportAlphaCurrentTileChunksFolder();
                    break;
                case TerrainExportKind.AlphaLoadedTilesFolder:
                    ExportAlphaTilesFolder(TerrainTileScope.LoadedTiles);
                    break;
                case TerrainExportKind.AlphaWholeMapFolder:
                    ExportAlphaTilesFolder(TerrainTileScope.WholeMap);
                    break;
                case TerrainExportKind.Heightmap257CurrentTilePerTile:
                    ExportHeightmap257CurrentTilePerTile();
                    break;
                case TerrainExportKind.Heightmap257LoadedTilesFolderPerTile:
                    ExportHeightmap257TilesFolderPerTile(TerrainTileScope.LoadedTiles);
                    break;
                case TerrainExportKind.Heightmap257WholeMapFolderPerMap:
                    ExportHeightmap257TilesFolderPerMap();
                    break;
                case TerrainExportKind.MccvCurrentTilePng:
                    ExportMccvCurrentTilePng();
                    break;
                case TerrainExportKind.MccvLoadedTilesFolder:
                    ExportMccvTilesFolder(TerrainTileScope.LoadedTiles);
                    break;
                case TerrainExportKind.MccvWholeMapFolder:
                    ExportMccvTilesFolder(TerrainTileScope.WholeMap);
                    break;
            }
        }
        catch (Exception ex)
        {
            _statusMessage = $"Terrain export failed: {ex.Message}";
        }
        finally
        {
            _terrainExportKind = TerrainExportKind.None;
        }
    }

    internal void RunTerrainImport()
    {
        try
        {
            switch (_terrainImportKind)
            {
                case TerrainImportKind.AlphaFolder:
                    BeginAlphaFolderImport();
                    break;
                case TerrainImportKind.Heightmap257Folder:
                    BeginHeightmapFolderImport();
                    break;
                case TerrainImportKind.MccvFolder:
                    BeginMccvFolderImport();
                    break;
            }
        }
        catch (Exception ex)
        {
            _statusMessage = $"Terrain import failed: {ex.Message}";
        }
        finally
        {
            _terrainImportKind = TerrainImportKind.None;
        }
    }

    private static bool TryParseTileCoordsFromFileName(string filePath, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;
        string name = Path.GetFileNameWithoutExtension(filePath);

        var matches = Regex.Matches(name, @"\d+");
        if (matches.Count < 2)
            return false;

        var candidates = new List<int>(matches.Count);
        foreach (Match m in matches)
        {
            if (int.TryParse(m.Value, out int v) && v >= 0 && v < 64)
                candidates.Add(v);
        }

        if (candidates.Count < 2)
            return false;

        tileX = candidates[^2];
        tileY = candidates[^1];
        return true;
    }

    private static IEnumerable<(int tileX, int tileY)> ParseCustomTileList(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            yield break;

        var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var line in lines)
        {
            var parts = line.Split(new[] { ',', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;
            if (!int.TryParse(parts[0], out int x)) continue;
            if (!int.TryParse(parts[1], out int y)) continue;
            if ((uint)x >= 64u || (uint)y >= 64u) continue;
            yield return (x, y);
        }
    }

    internal void GetTerrainTileRange(out int startX, out int startY, out int endX, out int endY)
    {
        startX = Math.Clamp(Math.Min(_terrainTileRangeStartX, _terrainTileRangeEndX), 0, 63);
        startY = Math.Clamp(Math.Min(_terrainTileRangeStartY, _terrainTileRangeEndY), 0, 63);
        endX = Math.Clamp(Math.Max(_terrainTileRangeStartX, _terrainTileRangeEndX), 0, 63);
        endY = Math.Clamp(Math.Max(_terrainTileRangeStartY, _terrainTileRangeEndY), 0, 63);
    }

    private IEnumerable<(int tileX, int tileY)> EnumerateTerrainTileRange()
    {
        GetTerrainTileRange(out int startX, out int startY, out int endX, out int endY);
        for (int tileY = startY; tileY <= endY; tileY++)
        {
            for (int tileX = startX; tileX <= endX; tileX++)
                yield return (tileX, tileY);
        }
    }

    private void DrawTerrainTileScopeSelector(string idSuffix, bool includeCurrentTile)
    {
        int scope = (int)_terrainTileScope;
        TerrainTileScope previousScope = _terrainTileScope;
        if (includeCurrentTile)
            ImGui.RadioButton($"Current tile##{idSuffix}", ref scope, (int)TerrainTileScope.CurrentTile);
        ImGui.RadioButton($"Loaded tiles##{idSuffix}", ref scope, (int)TerrainTileScope.LoadedTiles);
        ImGui.RadioButton($"Whole map##{idSuffix}", ref scope, (int)TerrainTileScope.WholeMap);
        ImGui.RadioButton($"Custom list##{idSuffix}", ref scope, (int)TerrainTileScope.CustomList);
        ImGui.RadioButton($"Row/Column range##{idSuffix}", ref scope, (int)TerrainTileScope.RectRange);
        _terrainTileScope = (TerrainTileScope)scope;
        bool restoreScopeChanged = previousScope != _terrainTileScope;

        if (_terrainTileScope == TerrainTileScope.CustomList)
        {
            ImGui.TextDisabled("One tile per line: x y (or x,y)");
            if (ImGui.InputTextMultiline($"##customTiles_{idSuffix}", ref _terrainCustomTilesText, 8192, new Vector2(480, 160)))
                restoreScopeChanged = true;
        }
        else if (_terrainTileScope == TerrainTileScope.RectRange)
        {
            int startX = _terrainTileRangeStartX;
            int startY = _terrainTileRangeStartY;
            int endX = _terrainTileRangeEndX;
            int endY = _terrainTileRangeEndY;
            if (ImGui.InputInt($"Column Start##{idSuffix}", ref startX))
            {
                _terrainTileRangeStartX = Math.Clamp(startX, 0, 63);
                restoreScopeChanged = true;
            }
            if (ImGui.InputInt($"Row Start##{idSuffix}", ref startY))
            {
                _terrainTileRangeStartY = Math.Clamp(startY, 0, 63);
                restoreScopeChanged = true;
            }
            if (ImGui.InputInt($"Column End##{idSuffix}", ref endX))
            {
                _terrainTileRangeEndX = Math.Clamp(endX, 0, 63);
                restoreScopeChanged = true;
            }
            if (ImGui.InputInt($"Row End##{idSuffix}", ref endY))
            {
                _terrainTileRangeEndY = Math.Clamp(endY, 0, 63);
                restoreScopeChanged = true;
            }

            GetTerrainTileRange(out int normalizedStartX, out int normalizedStartY, out int normalizedEndX, out int normalizedEndY);
            int width = normalizedEndX - normalizedStartX + 1;
            int height = normalizedEndY - normalizedStartY + 1;
            ImGui.TextDisabled($"Range: columns {normalizedStartX}..{normalizedEndX}, rows {normalizedStartY}..{normalizedEndY} ({width * height} tile(s)).");
        }

        if (restoreScopeChanged)
            _terrainWeakSignalRestore.MarkTerrainWeakSignalRestoreDirty();
    }

    internal IReadOnlyList<(int tileX, int tileY)> GetTileScopeList(TerrainTileScope scope)
    {
        if (scope == TerrainTileScope.CurrentTile)
        {
            var cam = GetCameraTile();
            return new List<(int, int)> { cam };
        }

        if (scope == TerrainTileScope.CustomList)
            return ParseCustomTileList(_terrainCustomTilesText).Distinct().ToList();

        if (scope == TerrainTileScope.RectRange)
            return EnumerateTerrainTileRange().ToList();

        if (_terrainManager != null)
        {
            if (scope == TerrainTileScope.LoadedTiles)
                return _terrainManager.LoadedTiles.ToList();

            if (scope == TerrainTileScope.WholeMap)
                return _terrainManager.Adapter.ExistingTiles.Select(idx => (idx / 64, idx % 64)).ToList();
        }

        if (_vlmTerrainManager != null)
        {
            if (scope == TerrainTileScope.LoadedTiles)
                return _vlmTerrainManager.Loader.TileCoords
                    .Where(t => _vlmTerrainManager.IsTileLoaded(t.tileX, t.tileY))
                    .ToList();

            if (scope == TerrainTileScope.WholeMap)
                return _vlmTerrainManager.Loader.TileCoords.ToList();
        }

        return new List<(int, int)>();
    }

    internal IReadOnlyList<WoWViewer.Terrain.TerrainChunkData>? LoadTileChunksForExport(int tileX, int tileY)
    {
        if (_terrainManager != null)
        {
            return _terrainManager.GetOrLoadTileLoadResult(tileX, tileY).Chunks;
        }

        if (_vlmTerrainManager != null)
        {
            if (_vlmTerrainManager.TryGetTileLoadResult(tileX, tileY, out var tile))
                return tile.Chunks;

            if (_vlmTerrainManager.Loader.TileCoords.Contains((tileX, tileY)))
                return _vlmTerrainManager.Loader.LoadTile(tileX, tileY).Chunks;
        }

        return null;
    }

    private void ExportAlphaCurrentTileAtlas()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        string defaultName = $"tile_{tx}_{ty}_alpha.png";
        ImGuiPathPicker.Instance.Open(
            "Save Alpha Mask Atlas",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".png",
            picked =>
            {
                if (string.IsNullOrEmpty(picked))
                    return;

                using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
                using (var fs = File.Create(picked))
                    atlas.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                _statusMessage = $"Exported: {picked}";
            },
            defaultName);
    }

    private void ExportAlphaCurrentTileChunksFolder()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        ImGuiPathPicker.Instance.Open(
            "Select output folder for chunk alpha masks",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder))
                    return;

                using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
                var chunkImages = TerrainImageIo.BuildAlphaChunkImagesFromAtlas(atlas);
                foreach (var kvp in chunkImages)
                {
                    var (cx, cy) = kvp.Key;
                    string path = Path.Combine(folder, $"tile_{tx}_{ty}_chunk_{cx}_{cy}_alpha.png");
                    using (var fs = File.Create(path))
                        kvp.Value.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                    kvp.Value.Dispose();
                }

                _statusMessage = $"Exported chunks: {folder}";
            });
    }

    private void ExportAlphaTilesFolder(TerrainTileScope scope)
    {
        ImGuiPathPicker.Instance.Open(
            "Select output folder for tile alpha atlases",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder))
                    return;

                var tiles = GetTileScopeList(scope);

                int written = 0;
                foreach (var (tx, ty) in tiles)
                {
                    var chunks = LoadTileChunksForExport(tx, ty);
                    if (chunks == null) continue;

                    using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
                    string path = Path.Combine(folder, $"tile_{tx}_{ty}_alpha.png");
                    using (var fs = File.Create(path))
                        atlas.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                    written++;
                }

                _statusMessage = $"Exported {written} tiles: {folder}";
            });
    }

    private void BeginAlphaFolderImport()
    {
        ImGuiPathPicker.Instance.Open(
            "Select folder containing tile alpha atlases",
            pickFolder: true,
            initialPath: null,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
                    return;

                _terrainImportFolder = folder;
                _showAlphaFolderImportScope = true;
            });
    }

    internal void DrawAlphaFolderImportScopeDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Import Alpha Masks", ref _showAlphaFolderImportScope, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.Text("Apply imported alpha masks to:");
        ImGui.Separator();
        DrawTerrainTileScopeSelector("AlphaImport", includeCurrentTile: true);

        ImGui.Separator();
        if (ImGui.Button("Import"))
        {
            ApplyAlphaFolderImport(_terrainImportFolder, _terrainTileScope);
            _terrainImportFolder = "";
            _showAlphaFolderImportScope = false;
        }
        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _terrainImportFolder = "";
            _showAlphaFolderImportScope = false;
        }

        ImGui.End();
    }

    private void ApplyAlphaFolderImport(string folder, TerrainTileScope scope)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        var targets = new HashSet<(int tileX, int tileY)>(GetTileScopeList(scope));
        if (targets.Count == 0)
        {
            _statusMessage = "No target tiles selected.";
            return;
        }

        if (scope == TerrainTileScope.WholeMap && _terrainManager != null)
            _terrainManager.LoadAllTiles();

        var renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
        {
            _statusMessage = "No terrain renderer.";
            return;
        }

        int applied = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.png"))
        {
            if (!TryParseTileCoordsFromFileName(file, out int tx, out int ty))
                continue;

            if (!targets.Contains((tx, ty)))
                continue;

            if (_terrainManager != null && !_terrainManager.IsTileLoaded(tx, ty))
                continue;
            if (_vlmTerrainManager != null && !_vlmTerrainManager.IsTileLoaded(tx, ty))
                continue;

            using var atlas = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(file);
            var alphaShadow = TerrainImageIo.DecodeAlphaShadowArrayFromAtlas(atlas);
            renderer.ReplaceTileAlphaShadowArray(tx, ty, alphaShadow);
            applied++;
        }

        _statusMessage = $"Imported alpha masks for {applied} tiles.";
    }

    private void ExportMccvCurrentTilePng()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        string defaultName = $"tile_{tx}_{ty}_mccv.png";
        ImGuiPathPicker.Instance.Open(
            "Save MCCV Tile PNG",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".png",
            picked =>
            {
                if (string.IsNullOrEmpty(picked))
                    return;

                using var image = TerrainMccvIo.BuildTileImage(chunks);
                using (var fs = File.Create(picked))
                    image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

                _statusMessage = $"Exported: {picked}";
            },
            defaultName);
    }

    private void ExportMccvTilesFolder(TerrainTileScope scope)
    {
        ImGuiPathPicker.Instance.Open(
            "Select output folder for tile MCCV PNGs",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder))
                    return;

                var tiles = GetTileScopeList(scope);

                int written = 0;
                foreach (var (tx, ty) in tiles)
                {
                    var chunks = LoadTileChunksForExport(tx, ty);
                    if (chunks == null)
                        continue;

                    using var image = TerrainMccvIo.BuildTileImage(chunks);
                    string path = Path.Combine(folder, $"tile_{tx}_{ty}_mccv.png");
                    using (var fs = File.Create(path))
                        image.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                    written++;
                }

                _statusMessage = $"Exported {written} MCCV tiles: {folder}";
            });
    }

    private void BeginMccvFolderImport()
    {
        ImGuiPathPicker.Instance.Open(
            "Select folder containing tile MCCV PNGs",
            pickFolder: true,
            initialPath: null,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
                    return;

                _terrainImportFolder = folder;
                _showMccvFolderImportScope = true;
            });
    }

    internal void DrawMccvFolderImportScopeDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Import MCCV", ref _showMccvFolderImportScope, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.Text("Apply imported MCCV to:");
        ImGui.Separator();
        DrawTerrainTileScopeSelector("MccvImport", includeCurrentTile: true);

        ImGui.Separator();
        ImGui.TextDisabled("PNG channels preserve raw MCCV bytes in file order for VLM/tooling compatibility.");
        if (ImGui.Button("Import"))
        {
            ApplyMccvFolderImport(_terrainImportFolder, _terrainTileScope);
            _terrainImportFolder = "";
            _showMccvFolderImportScope = false;
        }
        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _terrainImportFolder = "";
            _showMccvFolderImportScope = false;
        }

        ImGui.End();
    }

    private void ApplyMccvFolderImport(string folder, TerrainTileScope scope)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        var targets = new HashSet<(int tileX, int tileY)>(GetTileScopeList(scope));
        if (targets.Count == 0)
        {
            _statusMessage = "No target tiles selected.";
            return;
        }

        if (scope == TerrainTileScope.WholeMap && _terrainManager != null)
            _terrainManager.LoadAllTiles();

        int applied = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.png"))
        {
            if (!TryParseTileCoordsFromFileName(file, out int tx, out int ty))
                continue;
            if (!targets.Contains((tx, ty)))
                continue;

            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null)
                continue;

            using var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgba32>(file);
            var newChunks = TerrainMccvIo.ApplyTileImageToChunks(chunks, image);
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(tx, ty, newChunks);
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tx, ty, newChunks);

            applied++;
        }

        _statusMessage = $"Imported MCCV for {applied} tiles.";
    }

    private void ExportHeightmap257CurrentTilePerTile()
    {
        var (tx, ty) = GetCameraTile();
        var chunks = LoadTileChunksForExport(tx, ty);
        if (chunks == null)
        {
            _statusMessage = $"No tile data available for ({tx},{ty}).";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        string defaultName = $"tile_{tx}_{ty}_height_257.png";
        ImGuiPathPicker.Instance.Open(
            "Save Heightmap (257x257 L16)",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".png",
            picked =>
            {
                if (string.IsNullOrEmpty(picked))
                    return;

                var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, tile.MinHeight, tile.MaxHeight);
                using (var fs = File.Create(picked))
                    img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

                var meta = new HeightmapMetadata
                {
                    MinHeight = tile.MinHeight,
                    MaxHeight = tile.MaxHeight,
                    Normalization = "per_tile",
                };
                string jsonPath = Path.ChangeExtension(picked, ".json");
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));

                _statusMessage = $"Exported: {picked}";
            },
            defaultName);
    }

    private void ExportHeightmap257TilesFolderPerTile(TerrainTileScope scope)
    {
        ImGuiPathPicker.Instance.Open(
            "Select output folder for tile heightmaps",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder))
                    return;

                var tiles = GetTileScopeList(scope);

                int written = 0;
                foreach (var (tx, ty) in tiles)
                {
                    var chunks = LoadTileChunksForExport(tx, ty);
                    if (chunks == null) continue;

                    var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                    using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, tile.MinHeight, tile.MaxHeight);
                    string pngPath = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.png");
                    using (var fs = File.Create(pngPath))
                        img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());

                    var meta = new HeightmapMetadata
                    {
                        MinHeight = tile.MinHeight,
                        MaxHeight = tile.MaxHeight,
                        Normalization = "per_tile",
                    };
                    string jsonPath = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.json");
                    File.WriteAllText(jsonPath, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
                    written++;
                }

                _statusMessage = $"Exported {written} tiles: {folder}";
            });
    }

    private void ExportHeightmap257TilesFolderPerMap()
    {
        ImGuiPathPicker.Instance.Open(
            "Select output folder for map-normalized tile heightmaps",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder))
                    return;

                var tiles = GetTileScopeList(TerrainTileScope.WholeMap);
                if (tiles.Count == 0)
                {
                    _statusMessage = "No tiles available.";
                    return;
                }

                float gMin = float.MaxValue;
                float gMax = float.MinValue;
                foreach (var (tx, ty) in tiles)
                {
                    var chunks = LoadTileChunksForExport(tx, ty);
                    if (chunks == null) continue;
                    var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                    if (tile.MinHeight < gMin) gMin = tile.MinHeight;
                    if (tile.MaxHeight > gMax) gMax = tile.MaxHeight;
                }
                if (gMin == float.MaxValue || gMax == float.MinValue)
                {
                    gMin = 0f;
                    gMax = 0f;
                }

                var mapMeta = new HeightmapMetadata
                {
                    MinHeight = gMin,
                    MaxHeight = gMax,
                    Normalization = "per_map",
                };
                string mapJson = Path.Combine(folder, "heightmap_257_map.json");
                File.WriteAllText(mapJson, JsonSerializer.Serialize(mapMeta, new JsonSerializerOptions { WriteIndented = true }));

                int written = 0;
                foreach (var (tx, ty) in tiles)
                {
                    var chunks = LoadTileChunksForExport(tx, ty);
                    if (chunks == null) continue;

                    var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                    using var img = TerrainHeightmapIo.EncodeL16(tile.Heights, gMin, gMax);
                    string pngPath = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.png");
                    using (var fs = File.Create(pngPath))
                        img.Save(fs, new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                    written++;
                }

                _statusMessage = $"Exported {written} tiles (per-map): {folder}";
            });
    }

    private void BeginHeightmapFolderImport()
    {
        ImGuiPathPicker.Instance.Open(
            "Select folder containing tile heightmaps",
            pickFolder: true,
            initialPath: null,
            filterExtension: null,
            folder =>
            {
                if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
                    return;

                _terrainImportFolder = folder;
                _showHeightmapFolderImportScope = true;
            });
    }

    internal void DrawHeightmapFolderImportScopeDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(520, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Import Heightmaps", ref _showHeightmapFolderImportScope, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.Text("Apply imported heightmaps to:");
        ImGui.Separator();
        DrawTerrainTileScopeSelector("HeightImport", includeCurrentTile: true);

        ImGui.Separator();
        if (ImGui.Button("Import"))
        {
            ApplyHeightmapFolderImport(_terrainImportFolder, _terrainTileScope);
            _terrainImportFolder = "";
            _showHeightmapFolderImportScope = false;
        }
        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _terrainImportFolder = "";
            _showHeightmapFolderImportScope = false;
        }

        ImGui.End();
    }

    private void ApplyHeightmapFolderImport(string folder, TerrainTileScope scope)
    {
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder))
            return;

        var targets = new HashSet<(int tileX, int tileY)>(GetTileScopeList(scope));
        if (targets.Count == 0)
        {
            _statusMessage = "No target tiles selected.";
            return;
        }

        HeightmapMetadata? mapMeta = null;
        string mapMetaPath = Path.Combine(folder, "heightmap_257_map.json");
        if (File.Exists(mapMetaPath))
        {
            try
            {
                mapMeta = JsonSerializer.Deserialize<HeightmapMetadata>(File.ReadAllText(mapMetaPath));
            }
            catch
            {
                mapMeta = null;
            }
        }

        int applied = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.png"))
        {
            if (!TryParseTileCoordsFromFileName(file, out int tx, out int ty))
                continue;
            if (!targets.Contains((tx, ty)))
                continue;

            if (_terrainManager != null && !_terrainManager.IsTileLoaded(tx, ty))
                continue;
            if (_vlmTerrainManager != null && !_vlmTerrainManager.IsTileLoaded(tx, ty))
                continue;

            HeightmapMetadata? meta = null;
            string perTileJson = Path.Combine(folder, $"tile_{tx}_{ty}_height_257.json");
            if (File.Exists(perTileJson))
            {
                try
                {
                    meta = JsonSerializer.Deserialize<HeightmapMetadata>(File.ReadAllText(perTileJson));
                }
                catch
                {
                    meta = null;
                }
            }

            meta ??= mapMeta;
            if (meta == null)
                continue;

            using var img = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.L16>(file);
            var tileHeights = TerrainHeightmapIo.DecodeL16(img, meta.MinHeight, meta.MaxHeight);

            var chunks = LoadTileChunksForExport(tx, ty);
            if (chunks == null)
                continue;

            var newChunks = TerrainHeightmapIo.ApplyHeightmap257ToChunks(chunks, tileHeights);
            if (_terrainManager != null)
                _terrainManager.ReplaceTileChunksAndRebuild(tx, ty, newChunks);
            else
                _vlmTerrainManager?.ReplaceTileChunksAndRebuild(tx, ty, newChunks);

            applied++;
        }

        _statusMessage = $"Imported heightmaps for {applied} tiles.";
    }
}
