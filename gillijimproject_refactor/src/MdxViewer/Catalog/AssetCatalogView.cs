using System.Numerics;
using ImGuiNET;
using MdxViewer.DataSources;
using MdxViewer.Rendering;
using Silk.NET.OpenGL;

namespace MdxViewer.Catalog;

/// <summary>
/// ImGui panel for browsing, searching, and exporting NPC and GameObject assets
/// from the alpha-core database. Supports individual and batch export.
/// </summary>
public class AssetCatalogView
{
    private readonly GL _gl;

    // Alpha-core root path (contains etc/databases/world/world.sql and etc/databases/dbc/dbc.sql)
    private string _alphaCoreRoot = "";
    private string _outputDir = "asset_catalog_output";

    // State
    private bool _isConnected;
    private string _connectionStatus = "Not connected";
    private List<AssetCatalogEntry> _allEntries = new();
    private List<AssetCatalogEntry> _filteredEntries = new();
    private AlphaCoreDbReader? _dbReader;
    private AssetExporter? _exporter;

    // Filter/search
    private string _searchText = "";
    private bool _showCreatures = true;
    private bool _showGameObjects = true;
    private bool _showOnlyWithModel = true;
    private bool _showOnlyWithSpawns = false;
    private int _selectedIndex = -1;

    // Export state
    private bool _isExporting;
    private int _exportProgress;
    private int _exportTotal;
    private string _exportStatus = "";
    private CancellationTokenSource? _exportCts;

    // Data source for model loading
    private IDataSource? _dataSource;
    private ReplaceableTextureResolver? _texResolver;

    public bool IsVisible { get; set; } = false;

    public AssetCatalogView(GL gl)
    {
        _gl = gl;
    }

    public void SetDataSource(IDataSource? dataSource, ReplaceableTextureResolver? texResolver = null)
    {
        _dataSource = dataSource;
        _texResolver = texResolver;
    }

    public void Draw()
    {
        if (!IsVisible) return;

        ImGui.SetNextWindowSize(new Vector2(700, 600), ImGuiCond.FirstUseEver);
        bool visible = IsVisible;
        if (!ImGui.Begin("Asset Catalog", ref visible))
        {
            IsVisible = visible;
            ImGui.End();
            return;
        }
        IsVisible = visible;

        DrawConnectionPanel();
        ImGui.Separator();

        if (_isConnected && _allEntries.Count > 0)
        {
            DrawFilterBar();
            ImGui.Separator();
            DrawEntryList();
            ImGui.Separator();
            DrawDetailPanel();
            ImGui.Separator();
            DrawExportPanel();
        }

        ImGui.End();
    }

    private void DrawConnectionPanel()
    {
        if (ImGui.CollapsingHeader("Data Source", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.InputTextWithHint("##acroot", "Path to alpha-core root folder...", ref _alphaCoreRoot, 1024);
            ImGui.SameLine();
            ImGui.TextDisabled("(?)");
            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.Text("Point to the alpha-core repo root.");
                ImGui.Text("It should contain: etc/databases/world/world.sql");
                ImGui.Text("                   etc/databases/dbc/dbc.sql");
                ImGui.EndTooltip();
            }

            if (!_isConnected)
            {
                if (ImGui.Button("Load from SQL Dumps"))
                {
                    _ = ConnectAndLoadAsync();
                }
            }
            else
            {
                ImGui.TextColored(new Vector4(0, 1, 0, 1), "Loaded");
                ImGui.SameLine();
                if (ImGui.SmallButton("Reload"))
                    _ = ConnectAndLoadAsync();
            }

            ImGui.TextWrapped(_connectionStatus);
        }
    }

    private void DrawFilterBar()
    {
        ImGui.InputTextWithHint("##search", "Search by name...", ref _searchText, 256);
        ImGui.SameLine();
        if (ImGui.Button("Clear")) _searchText = "";

        ImGui.Checkbox("NPCs", ref _showCreatures);
        ImGui.SameLine();
        ImGui.Checkbox("GameObjects", ref _showGameObjects);
        ImGui.SameLine();
        ImGui.Checkbox("Has Model", ref _showOnlyWithModel);
        ImGui.SameLine();
        ImGui.Checkbox("Has Spawns", ref _showOnlyWithSpawns);

        ApplyFilters();

        ImGui.Text($"Showing {_filteredEntries.Count} / {_allEntries.Count} entries");
    }

    private void DrawEntryList()
    {
        float listHeight = ImGui.GetContentRegionAvail().Y * 0.45f;
        if (ImGui.BeginChild("EntryList", new Vector2(0, listHeight), true))
        {
            // Simple loop — ImGui handles scrolling via BeginChild
            for (int i = 0; i < _filteredEntries.Count; i++)
            {
                var entry = _filteredEntries[i];
                string icon = entry.Type == AssetType.Creature ? "[NPC]" : "[GO]";
                string modelIndicator = entry.ModelPath != null ? "" : " (no model)";
                string label = $"{icon} [{entry.EntryId}] {entry.Name}{modelIndicator}";

                bool isSelected = _selectedIndex == i;
                if (ImGui.Selectable(label, isSelected))
                    _selectedIndex = i;

                if (ImGui.IsItemHovered())
                {
                    ImGui.BeginTooltip();
                    ImGui.Text($"{entry.TypeLabel}");
                    if (entry.Type == AssetType.Creature)
                    {
                        ImGui.Text($"Level: {entry.LevelMin}-{entry.LevelMax}");
                        if (!string.IsNullOrEmpty(entry.Subname))
                            ImGui.Text($"Title: {entry.Subname}");
                    }
                    ImGui.Text($"Display ID: {entry.DisplayId}");
                    ImGui.Text($"Model: {entry.ModelPath ?? "none"}");
                    ImGui.Text($"Spawns: {entry.Spawns.Count}");
                    ImGui.EndTooltip();
                }
            }
        }
        ImGui.EndChild();
    }

    private void DrawDetailPanel()
    {
        if (_selectedIndex < 0 || _selectedIndex >= _filteredEntries.Count)
        {
            ImGui.TextDisabled("Select an entry to view details");
            return;
        }

        var entry = _filteredEntries[_selectedIndex];

        ImGui.Text($"{entry.TypeLabel}: {entry.Name}");
        if (!string.IsNullOrEmpty(entry.Subname))
            ImGui.TextColored(new Vector4(0.7f, 0.7f, 0.3f, 1), $"<{entry.Subname}>");

        ImGui.Columns(2, "details", false);
        ImGui.SetColumnWidth(0, 150);

        ImGui.Text("Entry ID:"); ImGui.NextColumn(); ImGui.Text($"{entry.EntryId}"); ImGui.NextColumn();
        ImGui.Text("Display ID:"); ImGui.NextColumn(); ImGui.Text($"{entry.DisplayId}"); ImGui.NextColumn();
        ImGui.Text("Model:"); ImGui.NextColumn();
        ImGui.TextWrapped(entry.ModelPath ?? "(none)"); ImGui.NextColumn();
        ImGui.Text("Scale:"); ImGui.NextColumn(); ImGui.Text($"{entry.EffectiveScale:F2}"); ImGui.NextColumn();

        if (entry.Type == AssetType.Creature)
        {
            ImGui.Text("Level:"); ImGui.NextColumn();
            ImGui.Text($"{entry.LevelMin} - {entry.LevelMax}"); ImGui.NextColumn();
            ImGui.Text("Rank:"); ImGui.NextColumn();
            string rankName = entry.Rank switch { 0 => "Normal", 1 => "Elite", 2 => "Rare Elite", 3 => "Boss", 4 => "Rare", _ => $"{entry.Rank}" };
            ImGui.Text(rankName); ImGui.NextColumn();
            ImGui.Text("Faction:"); ImGui.NextColumn(); ImGui.Text($"{entry.Faction}"); ImGui.NextColumn();
            ImGui.Text("NPC Flags:"); ImGui.NextColumn(); ImGui.Text($"0x{entry.NpcFlags:X}"); ImGui.NextColumn();
        }
        else
        {
            ImGui.Text("GO Type:"); ImGui.NextColumn(); ImGui.Text($"{entry.GameObjectType}"); ImGui.NextColumn();
            ImGui.Text("Flags:"); ImGui.NextColumn(); ImGui.Text($"0x{entry.Flags:X}"); ImGui.NextColumn();
        }

        ImGui.Text("Spawns:"); ImGui.NextColumn(); ImGui.Text($"{entry.Spawns.Count}"); ImGui.NextColumn();

        ImGui.Columns(1);

        // Spawn list (first 10)
        if (entry.Spawns.Count > 0 && ImGui.TreeNode($"Spawn Locations ({entry.Spawns.Count})"))
        {
            int showCount = Math.Min(entry.Spawns.Count, 10);
            for (int i = 0; i < showCount; i++)
            {
                var s = entry.Spawns[i];
                ImGui.Text($"  Map {s.MapId}: ({s.X:F1}, {s.Y:F1}, {s.Z:F1}) o={s.Orientation:F2}");
            }
            if (entry.Spawns.Count > 10)
                ImGui.TextDisabled($"  ... and {entry.Spawns.Count - 10} more");
            ImGui.TreePop();
        }

        // Texture variations
        if (entry.TextureVariations.Length > 0 && ImGui.TreeNode("Texture Variations"))
        {
            foreach (var tex in entry.TextureVariations)
                ImGui.Text($"  {tex}");
            ImGui.TreePop();
        }

        // Export single
        if (ImGui.Button("Export This Entry"))
        {
            ExportSingle(entry);
        }
    }

    private void DrawExportPanel()
    {
        if (ImGui.CollapsingHeader("Batch Export"))
        {
            ImGui.InputText("Output Directory", ref _outputDir, 512);

            if (_isExporting)
            {
                ImGui.ProgressBar((float)_exportProgress / Math.Max(_exportTotal, 1),
                    new Vector2(-1, 0), $"{_exportProgress}/{_exportTotal}");
                ImGui.Text(_exportStatus);
                if (ImGui.Button("Cancel"))
                    _exportCts?.Cancel();
            }
            else
            {
                int withModel = _filteredEntries.Count(e => e.ModelPath != null);
                ImGui.Text($"Filtered entries: {_filteredEntries.Count} ({withModel} with models)");

                if (ImGui.Button($"Export All Filtered ({_filteredEntries.Count})"))
                {
                    _ = ExportBatchAsync(_filteredEntries);
                }
                ImGui.SameLine();
                if (ImGui.Button($"Export Only With Models ({withModel})"))
                {
                    var withModels = _filteredEntries.Where(e => e.ModelPath != null).ToList();
                    _ = ExportBatchAsync(withModels);
                }
            }

            if (!string.IsNullOrEmpty(_exportStatus) && !_isExporting)
                ImGui.TextWrapped(_exportStatus);
        }
    }

    private void ApplyFilters()
    {
        _filteredEntries = _allEntries.Where(e =>
        {
            if (e.Type == AssetType.Creature && !_showCreatures) return false;
            if (e.Type == AssetType.GameObject && !_showGameObjects) return false;
            if (_showOnlyWithModel && e.ModelPath == null) return false;
            if (_showOnlyWithSpawns && e.Spawns.Count == 0) return false;
            if (!string.IsNullOrEmpty(_searchText))
            {
                bool nameMatch = e.Name.Contains(_searchText, StringComparison.OrdinalIgnoreCase);
                bool subMatch = e.Subname?.Contains(_searchText, StringComparison.OrdinalIgnoreCase) ?? false;
                bool idMatch = e.EntryId.ToString().Contains(_searchText);
                if (!nameMatch && !subMatch && !idMatch) return false;
            }
            return true;
        }).ToList();

        // Clamp selection
        if (_selectedIndex >= _filteredEntries.Count)
            _selectedIndex = _filteredEntries.Count - 1;
    }

    private async Task ConnectAndLoadAsync()
    {
        if (string.IsNullOrWhiteSpace(_alphaCoreRoot))
        {
            _connectionStatus = "Please enter the alpha-core root path.";
            return;
        }

        _connectionStatus = "Validating...";
        try
        {
            _dbReader = new AlphaCoreDbReader(_alphaCoreRoot);
            var (ok, msg) = _dbReader.Validate();
            if (!ok)
            {
                _connectionStatus = msg;
                return;
            }

            _connectionStatus = "Parsing SQL dumps (this may take a moment)...";
            await Task.Yield(); // let UI update
            _allEntries = await _dbReader.LoadAllAsync();
            _isConnected = true;

            int creatures = _allEntries.Count(e => e.Type == AssetType.Creature);
            int gos = _allEntries.Count(e => e.Type == AssetType.GameObject);
            int withModel = _allEntries.Count(e => e.ModelPath != null);
            int withSpawns = _allEntries.Count(e => e.Spawns.Count > 0);
            _connectionStatus = $"Loaded {creatures} NPCs + {gos} GameObjects ({withModel} with models, {withSpawns} with spawns)";

            ApplyFilters();
        }
        catch (Exception ex)
        {
            _connectionStatus = $"Error: {ex.Message}";
        }
    }

    private void ExportSingle(AssetCatalogEntry entry)
    {
        if (_exporter == null)
            _exporter = new AssetExporter(_gl, _dataSource, _texResolver);

        var (jsonPath, glbPath, ssPath) = _exporter.ExportEntry(entry, _outputDir);
        string results = $"Exported {entry.Name}:";
        if (jsonPath != null) results += $"\n  JSON: {jsonPath}";
        if (glbPath != null) results += $"\n  GLB: {glbPath}";
        if (ssPath != null) results += $"\n  Screenshot: {ssPath}";
        if (jsonPath == null && glbPath == null) results += "\n  (no output)";
        _exportStatus = results;
    }

    private async Task ExportBatchAsync(IReadOnlyList<AssetCatalogEntry> entries)
    {
        if (_isExporting) return;
        _isExporting = true;
        _exportCts = new CancellationTokenSource();
        _exportProgress = 0;
        _exportTotal = entries.Count;

        if (_exporter == null)
            _exporter = new AssetExporter(_gl, _dataSource, _texResolver);

        try
        {
            var result = await _exporter.ExportBatchAsync(entries, _outputDir,
                (current, total, name) =>
                {
                    _exportProgress = current;
                    _exportTotal = total;
                    _exportStatus = $"[{current}/{total}] {name}";
                },
                _exportCts.Token);

            _exportStatus = $"Batch complete: {result}";
        }
        catch (Exception ex)
        {
            _exportStatus = $"Batch failed: {ex.Message}";
        }
        finally
        {
            _isExporting = false;
            _exportCts?.Dispose();
            _exportCts = null;
        }
    }
}
