using System.Text.Json;
using System.Text.RegularExpressions;
using ImGuiNET;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;
using WoWViewer.Export;
using WoWViewer.Terrain;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;
using static WoWViewer.ChunkEditService;

namespace WoWViewer;

/// <summary>
/// Opt-in UI orchestration above the existing tile, image and rebuild services.
/// No format, renderer or camera behavior is changed here. Imports are session-only.
/// </summary>
internal sealed class TerrainFileOperationsService
{
    private readonly IViewerAppHost _host;
    private TerrainTileScope _importScope = TerrainTileScope.CurrentTile;

    internal TerrainFileOperationsService(IViewerAppHost host) => _host = host;

    public void DrawContent()
    {
        bool hasTerrain = _host.TerrainManager != null || _host.VlmTerrainManager != null;
        if (SharedUiWidgets.SectionHeader("Terrain Exports", defaultOpen: true, id: "TerrainFileExports"))
        {
            ExportButton("Alpha: current tile atlas", TerrainExportKind.AlphaCurrentTileAtlas, hasTerrain);
            ExportButton("Alpha: current tile chunks", TerrainExportKind.AlphaCurrentTileChunksFolder, hasTerrain);
            ExportButton("Alpha: loaded tiles", TerrainExportKind.AlphaLoadedTilesFolder, hasTerrain);
            ExportButton("Alpha: whole map", TerrainExportKind.AlphaWholeMapFolder, hasTerrain);
            ExportButton("Heightmap: current tile (257x257 + JSON)", TerrainExportKind.Heightmap257CurrentTilePerTile, hasTerrain);
            ExportButton("Heightmap: loaded tiles (per-tile)", TerrainExportKind.Heightmap257LoadedTilesFolderPerTile, hasTerrain);
            ExportButton("Heightmap: whole map (per-map)", TerrainExportKind.Heightmap257WholeMapFolderPerMap, hasTerrain);
            ExportButton("MCCV: current tile", TerrainExportKind.MccvCurrentTilePng, hasTerrain);
            ExportButton("MCCV: loaded tiles", TerrainExportKind.MccvLoadedTilesFolder, hasTerrain);
            ExportButton("MCCV: whole map", TerrainExportKind.MccvWholeMapFolder, hasTerrain);
        }

        if (SharedUiWidgets.SectionHeader("Terrain Imports", defaultOpen: true, id: "TerrainFileImports"))
        {
            ImGui.TextWrapped("Applies to this session only; does not save ADT/WDT files. Alpha masks replace the live renderer preview. Heightmaps and MCCV rebuild the session's tile chunks.");
            int scope = (int)_importScope;
            ImGui.RadioButton("Current tile##terrain_import_scope", ref scope, (int)TerrainTileScope.CurrentTile);
            ImGui.RadioButton("Loaded tiles##terrain_import_scope", ref scope, (int)TerrainTileScope.LoadedTiles);
            ImGui.RadioButton("Whole map##terrain_import_scope", ref scope, (int)TerrainTileScope.WholeMap);
            _importScope = (TerrainTileScope)scope;
            ImportButton("Apply Alpha Atlases Folder...", TerrainImportKind.AlphaFolder, hasTerrain);
            ImportButton("Apply Heightmaps Folder...", TerrainImportKind.Heightmap257Folder, hasTerrain);
            ImportButton("Apply MCCV Folder...", TerrainImportKind.MccvFolder, hasTerrain);
        }

        if (!hasTerrain)
            ImGui.TextDisabled("Load terrain to use file operations.");
        SharedUiWidgets.StatusReadout("File operation", _host.StatusMessage);
    }

    private void ExportButton(string label, TerrainExportKind kind, bool enabled)
    {
        if (SharedUiWidgets.ActionButton(label, enabled))
            RequestExport(kind);
    }

    private void ImportButton(string label, TerrainImportKind kind, bool enabled)
    {
        if (SharedUiWidgets.ActionButton(label, enabled))
            RequestImport(kind);
    }

    internal void RequestExport(TerrainExportKind kind)
    {
        try
        {
            if (!Enum.IsDefined(kind) || kind == TerrainExportKind.None)
                throw new NotSupportedException($"Unsupported terrain export: {kind}");
            bool singleFile = kind is TerrainExportKind.AlphaCurrentTileAtlas
                or TerrainExportKind.Heightmap257CurrentTilePerTile or TerrainExportKind.MccvCurrentTilePng;
            TerrainTileScope scope = kind switch
            {
                TerrainExportKind.AlphaLoadedTilesFolder or TerrainExportKind.Heightmap257LoadedTilesFolderPerTile
                    or TerrainExportKind.MccvLoadedTilesFolder => TerrainTileScope.LoadedTiles,
                TerrainExportKind.AlphaWholeMapFolder or TerrainExportKind.Heightmap257WholeMapFolderPerMap
                    or TerrainExportKind.MccvWholeMapFolder => TerrainTileScope.WholeMap,
                _ => TerrainTileScope.CurrentTile,
            };
            var tiles = _host.TerrainTileIo.GetTileScopeList(scope).ToArray();
            if (tiles.Length == 0)
                throw new InvalidOperationException("No terrain tiles in scope.");
            var terrain = _host.TerrainManager;
            var vlm = _host.VlmTerrainManager;
            Directory.CreateDirectory(ExportDir);
            string suffix = kind == TerrainExportKind.AlphaCurrentTileAtlas ? "alpha"
                : kind == TerrainExportKind.MccvCurrentTilePng ? "mccv" : "height_257";
            _host.StatusMessage = $"Select output for {kind}; cancel leaves files unchanged.";
            ImGuiPathPicker.Instance.Open(
                $"Export {kind}",
                singleFile ? ImGuiPathPickerMode.SaveFile : ImGuiPathPickerMode.OpenFolder,
                ExportDir, singleFile ? ".png" : null,
                path => FileOperationReporter.Run($"Terrain export {kind}", path, () =>
                {
                    RequireSameTerrain(terrain, vlm);
                    return Export(kind, tiles, path, singleFile);
                }, message => _host.StatusMessage = message),
                singleFile ? $"tile_{tiles[0].tileX}_{tiles[0].tileY}_{suffix}.png" : null);
        }
        catch (Exception ex)
        {
            _host.StatusMessage = $"Terrain export {kind} failed: {ex.Message}";
        }
    }

    private int Export(TerrainExportKind kind, (int tileX, int tileY)[] tiles, string path, bool singleFile)
    {
        bool alpha = kind is TerrainExportKind.AlphaCurrentTileAtlas or TerrainExportKind.AlphaCurrentTileChunksFolder
            or TerrainExportKind.AlphaLoadedTilesFolder or TerrainExportKind.AlphaWholeMapFolder;
        bool mccv = kind is TerrainExportKind.MccvCurrentTilePng or TerrainExportKind.MccvLoadedTilesFolder
            or TerrainExportKind.MccvWholeMapFolder;
        bool mapHeight = kind == TerrainExportKind.Heightmap257WholeMapFolderPerMap;
        if (mapHeight && tiles.Any(tile => File.Exists(Path.Combine(path, $"tile_{tile.tileX}_{tile.tileY}_height_257.json"))))
            throw new InvalidOperationException("Per-tile metadata already exists here. Choose a clean output folder for map-normalized heightmaps.");
        float min = float.MaxValue, max = float.MinValue;
        if (mapHeight)
        {
            foreach (var (x, y) in tiles)
            {
                var tile = TerrainHeightmapIo.BuildTileHeightmap257(RequireChunks(x, y));
                min = Math.Min(min, tile.MinHeight);
                max = Math.Max(max, tile.MaxHeight);
            }
        }

        int written = 0;
        foreach (var (x, y) in tiles)
        {
            var chunks = RequireChunks(x, y);
            if (alpha)
            {
                using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(chunks);
                if (kind == TerrainExportKind.AlphaCurrentTileChunksFolder)
                {
                    var images = TerrainImageIo.BuildAlphaChunkImagesFromAtlas(atlas);
                    try
                    {
                        foreach (var entry in images)
                        {
                            SavePng(entry.Value, Path.Combine(path, $"tile_{x}_{y}_chunk_{entry.Key.chunkX}_{entry.Key.chunkY}_alpha.png"));
                            written++;
                        }
                    }
                    finally
                    {
                        foreach (var image in images.Values) image.Dispose();
                    }
                }
                else
                {
                    SavePng(atlas, singleFile ? path : Path.Combine(path, $"tile_{x}_{y}_alpha.png"));
                    written++;
                }
            }
            else if (mccv)
            {
                using var image = TerrainMccvIo.BuildTileImage(chunks);
                SavePng(image, singleFile ? path : Path.Combine(path, $"tile_{x}_{y}_mccv.png"));
                written++;
            }
            else
            {
                var tile = TerrainHeightmapIo.BuildTileHeightmap257(chunks);
                var meta = new HeightmapMetadata
                {
                    MinHeight = mapHeight ? min : tile.MinHeight,
                    MaxHeight = mapHeight ? max : tile.MaxHeight,
                    Normalization = mapHeight ? "per_map" : "per_tile",
                };
                using var image = TerrainHeightmapIo.EncodeL16(tile.Heights, meta.MinHeight, meta.MaxHeight);
                string png = singleFile ? path : Path.Combine(path, $"tile_{x}_{y}_height_257.png");
                SavePng(image, png);
                string json = mapHeight ? Path.Combine(path, "heightmap_257_map.json") : Path.ChangeExtension(png, ".json");
                File.WriteAllText(json, JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
                FileOperationReporter.RequireWrittenFile(json);
                written++;
            }
        }
        return written;
    }

    internal void RequestImport(TerrainImportKind kind)
    {
        if (kind is not (TerrainImportKind.AlphaFolder or TerrainImportKind.Heightmap257Folder or TerrainImportKind.MccvFolder))
        {
            _host.StatusMessage = $"Terrain import failed: unsupported operation {kind}.";
            return;
        }
        var terrain = _host.TerrainManager;
        var vlm = _host.VlmTerrainManager;
        var scope = _importScope;
        var targets = _host.TerrainTileIo.GetTileScopeList(scope).ToHashSet();
        if (targets.Count == 0)
        {
            _host.StatusMessage = "Terrain import failed: no target tiles selected.";
            return;
        }
        _host.StatusMessage = $"Select {kind} input folder; applies to {targets.Count} tile(s) in this session only.";
        ImGuiPathPicker.Instance.Open($"Import {kind} (session only)", true, ExportDir, null,
            path => FileOperationReporter.Run($"Terrain import {kind} (session only)", path, () =>
            {
                RequireSameTerrain(terrain, vlm);
                if (!Directory.Exists(path)) throw new DirectoryNotFoundException(path);
                if (scope == TerrainTileScope.WholeMap && terrain != null) terrain.LoadAllTiles();
                return Import(kind, path, targets);
            }, message => _host.StatusMessage = message));
    }

    private int Import(TerrainImportKind kind, string folder, HashSet<(int tileX, int tileY)> targets)
    {
        string suffix = kind switch
        {
            TerrainImportKind.AlphaFolder => "alpha",
            TerrainImportKind.Heightmap257Folder => "height_257",
            TerrainImportKind.MccvFolder => "mccv",
            _ => throw new NotSupportedException($"Unsupported terrain import: {kind}"),
        };
        int applied = 0;
        foreach (string file in Directory.EnumerateFiles(folder, $"tile_*_{suffix}.png"))
        {
            // Only the artifact names emitted by the existing tile exporters, not arbitrary PNGs.
            var match = Regex.Match(Path.GetFileName(file), $@"^tile_(\d+)_(\d+)_{suffix}\.png$", RegexOptions.IgnoreCase);
            if (!match.Success || !int.TryParse(match.Groups[1].Value, out int x)
                || !int.TryParse(match.Groups[2].Value, out int y) || !targets.Contains((x, y))) continue;
            if (_host.TerrainManager != null && !_host.TerrainManager.IsTileLoaded(x, y))
                throw new InvalidOperationException($"Target tile ({x},{y}) is not loaded; load it before importing.");
            if (_host.VlmTerrainManager != null && _host.TerrainManager == null && !_host.VlmTerrainManager.IsTileLoaded(x, y))
                throw new InvalidOperationException($"Target tile ({x},{y}) is not loaded; load it before importing.");

            if (kind == TerrainImportKind.AlphaFolder)
            {
                var renderer = _host.TerrainManager?.Renderer ?? _host.VlmTerrainManager?.Renderer
                    ?? throw new InvalidOperationException("No terrain renderer available.");
                using var atlas = Image.Load<Rgba32>(file);
                renderer.ReplaceTileAlphaShadowArray(x, y, TerrainImageIo.DecodeAlphaShadowArrayFromAtlas(atlas));
            }
            else
            {
                var chunks = RequireChunks(x, y);
                IReadOnlyList<TerrainChunkData> replacement;
                if (kind == TerrainImportKind.MccvFolder)
                {
                    using var image = Image.Load<Rgba32>(file);
                    replacement = TerrainMccvIo.ApplyTileImageToChunks(chunks, image);
                }
                else
                {
                    string json = Path.ChangeExtension(file, ".json");
                    if (!File.Exists(json)) json = Path.Combine(folder, "heightmap_257_map.json");
                    var meta = JsonSerializer.Deserialize<HeightmapMetadata>(File.ReadAllText(json))
                        ?? throw new InvalidDataException($"Missing heightmap metadata: {json}");
                    if (!float.IsFinite(meta.MinHeight) || !float.IsFinite(meta.MaxHeight) || meta.MaxHeight < meta.MinHeight)
                        throw new InvalidDataException($"Invalid height range: {json}");
                    using var image = Image.Load<L16>(file);
                    replacement = TerrainHeightmapIo.ApplyHeightmap257ToChunks(chunks,
                        TerrainHeightmapIo.DecodeL16(image, meta.MinHeight, meta.MaxHeight));
                }
                if (_host.TerrainManager != null) _host.TerrainManager.ReplaceTileChunksAndRebuild(x, y, replacement);
                else if (_host.VlmTerrainManager != null) _host.VlmTerrainManager.ReplaceTileChunksAndRebuild(x, y, replacement);
                else throw new InvalidOperationException("No terrain manager available to apply the import.");
            }
            applied++;
        }
        return applied;
    }

    private IReadOnlyList<TerrainChunkData> RequireChunks(int x, int y) =>
        _host.TerrainTileIo.LoadTileChunksForExport(x, y) is { Count: > 0 } chunks ? chunks
            : throw new InvalidOperationException($"No tile data available for ({x},{y}).");

    private void RequireSameTerrain(TerrainManager? terrain, object? vlm)
    {
        if (!ReferenceEquals(terrain, _host.TerrainManager) || !ReferenceEquals(vlm, _host.VlmTerrainManager))
            throw new InvalidOperationException("The map changed while the picker was open. Select the operation again.");
    }

    private static void SavePng(Image image, string path)
    {
        using (var stream = File.Create(path)) image.Save(stream, new PngEncoder());
        FileOperationReporter.RequireWrittenFile(path);
    }
}
