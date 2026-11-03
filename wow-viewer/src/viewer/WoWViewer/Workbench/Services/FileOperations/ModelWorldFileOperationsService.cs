using ImGuiNET;
using WowViewer.Core.IO.Converters;
using WowViewer.Core.IO.Mdx;
using WoWViewer.Export;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WoWViewer.UI;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>Destination-owned GLB actions. The shared picker owns paths; existing exporters own formats.</summary>
internal sealed class ModelWorldFileOperationsService
{
    private readonly IViewerAppHost _host;
    private TerrainTileScope _worldScope = TerrainTileScope.CurrentTile;
    private bool _combineMultiTileGlb = true;
    private bool _bareCoordFilenames = true;

    internal ModelWorldFileOperationsService(IViewerAppHost host) => _host = host;

    public void DrawModelExportContent()
    {
        if (!SharedUiWidgets.SectionHeader("Model Export", defaultOpen: true, id: "ModelFileExport")) return;
        bool canExport = _host.LoadedFilePath != null && _host.Renderer != null;
        if (SharedUiWidgets.ActionButton("Export Model GLB...", canExport)) RequestModelExport(false);
        bool wmo = canExport && (_host.LoadedWmo != null ||
            string.Equals(Path.GetExtension(_host.LoadedFilePath), ".wmo", StringComparison.OrdinalIgnoreCase));
        if (SharedUiWidgets.ActionButton("Export WMO Collision GLB...", wmo)) RequestModelExport(true);
        SharedUiWidgets.StatusReadout("File operation", _host.StatusMessage);
    }

    public void DrawWorldExportContent()
    {
        if (!SharedUiWidgets.SectionHeader("World Export", defaultOpen: true, id: "WorldFileExport")) return;
        int scope = (int)_worldScope;
        ImGui.RadioButton("Current tile##world_export", ref scope, (int)TerrainTileScope.CurrentTile);
        ImGui.RadioButton("Loaded tiles##world_export", ref scope, (int)TerrainTileScope.LoadedTiles);
        ImGui.RadioButton("Whole map##world_export", ref scope, (int)TerrainTileScope.WholeMap);
        _worldScope = (TerrainTileScope)scope;

        if (_worldScope != TerrainTileScope.CurrentTile)
        {
            ImGui.Checkbox("Combine into single scene GLB##world_export_combine", ref _combineMultiTileGlb);
        }
        ImGui.Checkbox("Use XX_YY filename format (e.g. 26_39.glb)##world_export_bare", ref _bareCoordFilenames);

        bool canExport = _host.TerrainManager != null;
        if (SharedUiWidgets.ActionButton("Export Terrain + Objects GLB...", canExport)) RequestWorldExport(_worldScope, false);
        if (SharedUiWidgets.ActionButton("Export Terrain Collision GLB...", canExport)) RequestWorldExport(_worldScope, true);
        ImGui.TextDisabled("Collision export contains terrain only; object collision is not included.");
        SharedUiWidgets.StatusReadout("File operation", _host.StatusMessage);
    }

    internal void RequestModelExport(bool collision)
    {
        string? source = _host.LoadedFilePath;
        if (source == null || _host.Renderer == null)
        {
            _host.StatusMessage = "Model GLB export failed: no standalone model loaded.";
            return;
        }
        try
        {
            Directory.CreateDirectory(ExportDir);
            var renderer = _host.Renderer;
            _host.StatusMessage = "Select GLB output; cancel leaves files unchanged.";
            ImGuiPathPicker.Instance.Open("Export Model GLB", ImGuiPathPickerMode.SaveFile, ExportDir, ".glb",
                path => FileOperationReporter.Run("Model GLB export", path, () =>
                {
                    if (source != _host.LoadedFilePath || !ReferenceEquals(renderer, _host.Renderer))
                        throw new InvalidOperationException("The model changed while the picker was open.");
                    ExportModel(source, path, collision);
                    FileOperationReporter.RequireWrittenFile(path);
                    return 1;
                }, message => _host.StatusMessage = message),
                Path.ChangeExtension(Path.GetFileName(source), collision ? ".collision.glb" : ".glb"));
        }
        catch (Exception ex)
        {
            _host.StatusMessage = $"Model GLB export failed: {ex.Message}";
        }
    }

    private void ExportModel(string source, string path, bool collision)
    {
        string dir = Path.GetDirectoryName(source) ?? ".";
        string ext = Path.GetExtension(source).ToLowerInvariant();
        if (collision)
        {
            var wmo = _host.LoadedWmo;
            if (wmo == null && ext == ".wmo") wmo = new WmoV14ToV17Converter().ParseWmoV14(source);
            if (wmo == null) throw new NotSupportedException("Standalone collision export supports WMO only.");
            GlbExporter.ExportWmoCollision(wmo, dir, path);
        }
        else if (_host.LoadedWmo != null)
            GlbExporter.ExportWmoWithDoodads(_host.LoadedWmo, dir, path, _host.DataSource);
        else if (_host.LoadedMdx != null)
            GlbExporter.ExportMdx(_host.LoadedMdx, dir, path, _host.DataSource);
        else if (ext is ".mdx" or ".mdl")
            GlbExporter.ExportMdx(MdxFile.Load(source), dir, path, _host.DataSource);
        else if (ext == ".m2")
        {
            byte[]? skin = null;
            foreach (string candidate in WarcraftNetM2Adapter.BuildSkinCandidates(source))
                if (File.Exists(candidate)) { skin = File.ReadAllBytes(candidate); break; }
            byte[] converted = new Transfer.M2ToMdxConverter().ConvertToBytes(File.ReadAllBytes(source), skin, null)
                ?? throw new InvalidDataException("The existing M2 converter produced no MDX output.");
            using var stream = new MemoryStream(converted);
            GlbExporter.ExportMdx(MdxFile.Load(stream), dir, path, _host.DataSource);
        }
        else if (ext == ".wmo")
            GlbExporter.ExportWmoWithDoodads(new WmoV14ToV17Converter().ParseWmoV14(source), dir, path, _host.DataSource);
        else
            throw new NotSupportedException($"GLB export is not supported for {ext}.");
    }

    internal void RequestWorldExport(TerrainTileScope scope, bool collision)
    {
        try
        {
            var terrain = _host.TerrainManager ?? throw new InvalidOperationException("No terrain loaded.");
            var source = _host.DataSource;
            var tiles = _host.TerrainTileIo.GetTileScopeList(scope).ToArray();
            if (tiles.Length == 0) throw new InvalidOperationException("No tiles in scope.");
            Directory.CreateDirectory(ExportDir);

            bool single = scope == TerrainTileScope.CurrentTile;
            bool combine = !single && _combineMultiTileGlb;

            if (combine)
            {
                _host.StatusMessage = $"Select combined GLB output for {tiles.Length} tile(s); cancel leaves files unchanged.";
                string defaultCombinedName = $"{terrain.MapName}_loaded_{tiles.Length}tiles{(collision ? ".collision" : "")}.glb";

                ImGuiPathPicker.Instance.Open("Export World GLB (Combined)", ImGuiPathPickerMode.SaveFile,
                    ExportDir, ".glb",
                    path => FileOperationReporter.Run("World GLB export", path, () =>
                    {
                        if (!ReferenceEquals(terrain, _host.TerrainManager) || !ReferenceEquals(source, _host.DataSource))
                            throw new InvalidOperationException("The world changed while the picker was open.");
                        MapGlbExporter.ExportTilesCombined(terrain, source, _host.Md5Index, tiles, path, includePlacements: !collision);
                        FileOperationReporter.RequireWrittenFile(path);
                        return 1;
                    }, message => _host.StatusMessage = message),
                    defaultCombinedName);
                return;
            }

            _host.StatusMessage = $"Select GLB output for {tiles.Length} tile(s); cancel leaves files unchanged.";
            string singleDefaultName = _bareCoordFilenames
                ? $"{tiles[0].tileY:D2}_{tiles[0].tileX:D2}{(collision ? ".collision" : "")}.glb"
                : $"{terrain.MapName}_{tiles[0].tileY:D2}_{tiles[0].tileX:D2}{(collision ? ".collision" : "")}.glb";

            ImGuiPathPicker.Instance.Open("Export World GLB", single ? ImGuiPathPickerMode.SaveFile : ImGuiPathPickerMode.OpenFolder,
                ExportDir, single ? ".glb" : null,
                path => FileOperationReporter.Run("World GLB export", path, () =>
                {
                    if (!ReferenceEquals(terrain, _host.TerrainManager) || !ReferenceEquals(source, _host.DataSource))
                        throw new InvalidOperationException("The world changed while the picker was open.");
                    int written = 0;
                    foreach (var (x, y) in tiles)
                    {
                        // Note: Blizzard tile coordinate format is XX_YY where XX is column (y) and YY is row (x).
                        string fileName = _bareCoordFilenames
                            ? $"{y:D2}_{x:D2}{(collision ? ".collision" : "")}.glb"
                            : $"{terrain.MapName}_{y:D2}_{x:D2}{(collision ? ".collision" : "")}.glb";
                        string output = single ? path : Path.Combine(path, fileName);
                        MapGlbExporter.ExportTile(terrain, source, _host.Md5Index, x, y, output, includePlacements: !collision);
                        FileOperationReporter.RequireWrittenFile(output);
                        written++;
                    }
                    return written;
                }, message => _host.StatusMessage = message),
                single ? singleDefaultName : null);
        }
        catch (Exception ex)
        {
            _host.StatusMessage = $"World GLB export failed: {ex.Message}";
        }
    }
}
