using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.Media.Imaging;
using Avalonia;
using Avalonia.Input;
using Avalonia.Layout;

namespace WoWRollback.Gui;

public partial class MainWindow : Window
{
    private readonly string _cacheRoot;
    private readonly string _presetsRoot;
    private string _currentMap = string.Empty;

    private readonly Dictionary<string, List<TileLayerRow>> _rowsByMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<TileEntry>> _baseM2 = new();
    private readonly Dictionary<string, List<TileEntry>> _baseWmo = new();
    private readonly Dictionary<string, List<TileEntry>> _customM2 = new();
    private readonly Dictionary<string, List<TileEntry>> _customWmo = new();
    private readonly HashSet<string> _selectedTiles = new(StringComparer.OrdinalIgnoreCase);
    private string _focusedTileKey = string.Empty;
    private bool _isDragging = false;
    private (int x, int y) _dragStartCell;
    private HashSet<string>? _dragBase;
    private SelMode _dragMode = SelMode.Replace;

    private enum SelMode { Replace, Add, Remove }

    private sealed class TileLayerRow
    {
        public int TileX; public int TileY; public string Type = ""; public int Layer; public int Min; public int Max; public int Count;
    }
    private sealed class TileEntry
    {
        public string Type = ""; public int Layer; public int Min; public int Max; public int Count; public bool Enabled = true;
    }

    public MainWindow(string cacheRoot, string presetsRoot)
    {
        _cacheRoot = cacheRoot;
        _presetsRoot = presetsRoot;
        InitializeComponent();
        this.Opened += OnOpened;
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
        var rescan = this.FindControl<Button>("RescanBtn");
        if (rescan != null) rescan.Click += RescanBtn_Click;
        var save = this.FindControl<Button>("SavePresetBtn");
        if (save != null) save.Click += SavePresetBtn_Click;
        var load = this.FindControl<Button>("LoadPresetBtn");
        if (load != null) load.Click += LoadPresetBtn_Click;
        var openPresets = this.FindControl<Button>("OpenPresetsBtn");
        if (openPresets != null) openPresets.Click += OpenPresetsBtn_Click;
        var openCache = this.FindControl<Button>("OpenCacheBtn");
        if (openCache != null) openCache.Click += OpenCacheBtn_Click;

        // Build tab hookups
        var buildOut = this.FindControl<TextBox>("BuildOutBox");
        if (buildOut != null) buildOut.Text = Path.Combine(_cacheRoot, "..", "output");
        var pickOut = this.FindControl<Button>("PickBuildOutBtn"); if (pickOut != null) pickOut.Click += PickBuildOutBtn_Click;
        var saveSel = this.FindControl<Button>("SaveSelectionPresetBtn"); if (saveSel != null) saveSel.Click += SaveSelectionPresetBtn_Click;
        var prep = this.FindControl<Button>("PrepareBtn"); if (prep != null) prep.Click += PrepareBtn_Click;
        var openOut = this.FindControl<Button>("OpenOutBtn"); if (openOut != null) openOut.Click += OpenOutBtn_Click;
        var genBaseline = this.FindControl<Button>("GenBaselineBtn"); if (genBaseline != null) genBaseline.Click += GenBaselineBtn_Click;

        // Filters tab hookups
        var analyzeBtn = this.FindControl<Button>("AnalyzeFiltersBtn"); if (analyzeBtn != null) analyzeBtn.Click += AnalyzeFiltersBtn_Click;
        var whitelistBtn = this.FindControl<Button>("WhitelistBtn"); if (whitelistBtn != null) whitelistBtn.Click += (s,e)=>MarkFilters(true);
        var blacklistBtn = this.FindControl<Button>("BlacklistBtn"); if (blacklistBtn != null) blacklistBtn.Click += (s,e)=>MarkFilters(false);
        var clearSelBtn = this.FindControl<Button>("ClearFilterSelBtn"); if (clearSelBtn != null) clearSelBtn.Click += (s,e)=>{ var lb=this.FindControl<ListBox>("FiltersList"); if(lb?.SelectedItems is System.Collections.IList sel) sel.Clear(); else if(lb!=null) lb.SelectedIndex=-1; };
        var saveFiltersBtn = this.FindControl<Button>("SaveFiltersBtn"); if (saveFiltersBtn != null) saveFiltersBtn.Click += SaveFiltersBtn_Click;
    }

    private async void PickBuildOutBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Select Output Folder", AllowMultiple = false });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("BuildOutBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private void AppendBuildLog(string line)
    {
        var log = this.FindControl<TextBox>("BuildLogBox"); if (log == null) return;
        log.Text += (string.IsNullOrEmpty(log.Text) ? string.Empty : Environment.NewLine) + line;
    }

    private async void SaveSelectionPresetBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var pBox = this.FindControl<TextBox>("PresetsBox");
            var dir = pBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(dir)) dir = _presetsRoot;
            Directory.CreateDirectory(dir!);
            var name = "selection-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + ".json";
            var path = Path.Combine(dir!, name);

            var map = _currentMap;
            var selection = _selectedTiles.Count > 0 ? _selectedTiles.ToArray() : new[] { this.FindControl<ComboBox>("TileSelectBox")?.SelectedItem as string ?? "" };
            var tilesObj = selection.Where(s => !string.IsNullOrWhiteSpace(s)).ToDictionary(k => k, k => new { custom = new { m2 = Array.Empty<object>(), wmo = Array.Empty<object>() } });
            var preset = new
            {
                dataset = "default",
                global = new { m2 = Array.Empty<object>(), wmo = Array.Empty<object>() },
                maps = new Dictionary<string, object> { [map] = new { tiles = tilesObj } }
            };
            File.WriteAllText(path, JsonSerializer.Serialize(preset, new JsonSerializerOptions { WriteIndented = true }));
            AppendBuildLog($"Saved selection preset: {path}");
        }
        catch (Exception ex) { await ShowMessage("Error", ex.Message); }
    }

    private void PrepareBtn_Click(object? sender, RoutedEventArgs e)
    {
        AppendBuildLog("Prepare Layers: queued (placeholder). CLI integration to be wired.");
    }

    private void OpenOutBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var box = this.FindControl<TextBox>("BuildOutBox"); var path = box?.Text?.Trim();
            if (!string.IsNullOrWhiteSpace(path))
            {
                Directory.CreateDirectory(path);
                var psi = new System.Diagnostics.ProcessStartInfo { FileName = Path.GetFullPath(path), UseShellExecute = true };
                System.Diagnostics.Process.Start(psi);
            }
        }
        catch { }
    }

    private static double Percentile(IReadOnlyList<int> sorted, double p)
    {
        if (sorted.Count == 0) return 0;
        var pos = (sorted.Count - 1) * p;
        var lo = (int)Math.Floor(pos); var hi = (int)Math.Ceiling(pos);
        if (lo == hi) return sorted[lo];
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
    }

    private void GenBaselineBtn_Click(object? sender, RoutedEventArgs e)
    {
        var maps = GetMapDirs().Select(m => m.Map).ToList();
        var mapsDict = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);

        List<object> curM2 = new List<object>();
        List<object> curWmo = new List<object>();

        foreach (var m in maps)
        {
            EnsureMapLoaded(m);
            var mrows = _rowsByMap.GetValueOrDefault(m) ?? new List<TileLayerRow>();
            if (mrows.Count == 0) continue;

            List<object> BuildRankedFor(string type)
            {
                bool isM2 = string.Equals(type, "M2", StringComparison.OrdinalIgnoreCase);
                var ranked = mrows
                    .Where(r => isM2 ? string.Equals(r.Type, "M2", StringComparison.OrdinalIgnoreCase) : !string.Equals(r.Type, "M2", StringComparison.OrdinalIgnoreCase))
                    .GroupBy(r => TileKey(r.TileX, r.TileY))
                    .SelectMany(g => g.OrderBy(x => x.Min).Select((x, idx) => new { rank = idx, x.Min, x.Max }))
                    .ToList();

                var byRank = ranked
                    .GroupBy(x => x.rank)
                    .Select(g => new { layer = g.Key, min = (int)Math.Round(g.Average(z => (double)z.Min)), max = (int)Math.Round(g.Average(z => (double)z.Max)) })
                    .OrderBy(o => o.layer)
                    .Cast<object>()
                    .ToList();
                return byRank;
            }

            var m2 = BuildRankedFor("M2");
            var wmo = BuildRankedFor("WMO");
            if (string.Equals(m, _currentMap, StringComparison.OrdinalIgnoreCase)) { curM2 = m2; curWmo = wmo; }
            mapsDict[m] = new { baseline = new { m2 = m2, wmo = wmo } };
        }

        var baselineM2 = curM2;
        var baselineWmo = curWmo;
        var pBox = this.FindControl<TextBox>("PresetsBox"); var dir = pBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(dir)) dir = _presetsRoot;
        Directory.CreateDirectory(dir!);
        var name = $"baseline-ranked-multi-{DateTime.Now:yyyyMMdd-HHmmss}.json";
        var path = Path.Combine(dir!, name);
        var preset = new
        {
            dataset = "default",
            global = new { baseline = new { m2 = baselineM2, wmo = baselineWmo } },
            maps = mapsDict
        };
        File.WriteAllText(path, JsonSerializer.Serialize(preset, new JsonSerializerOptions { WriteIndented = true }));
        AppendBuildLog($"Generated baseline preset for {mapsDict.Count} map(s): {path}");
    }

    private sealed class FilterItem
    {
        public string Prefix { get; set; } = string.Empty;
        public int Count { get; set; }
        public string Label => $"{Prefix} ({Count})";
        public bool Whitelisted { get; set; }
        public bool Blacklisted { get; set; }
    }

    private List<FilterItem> _filterItems = new();

    private void AnalyzeFiltersBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
            var cacheBox = this.FindControl<TextBox>("CacheBox"); var root = cacheBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(root)) root = _cacheRoot;
            var placements = Path.Combine(root!, map, "placements.csv"); if (!File.Exists(placements)) { AppendBuildLog($"No placements.csv for {map}"); return; }
            var counts = new Dictionary<string,int>(StringComparer.OrdinalIgnoreCase);
            foreach (var line in File.ReadLines(placements).Skip(1))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(',');
                if (parts.Length < 5) continue;
                var asset = parts[4];
                var prefix = string.IsNullOrWhiteSpace(asset) ? "Unknown" : string.Join('/', (asset.Replace('\\','/').Split('/', StringSplitOptions.RemoveEmptyEntries).Take(2)));
                if (string.IsNullOrWhiteSpace(prefix)) prefix = "Unknown";
                counts[prefix] = counts.GetValueOrDefault(prefix) + 1;
            }
            _filterItems = counts.OrderByDescending(kv=>kv.Value).Select(kv => new FilterItem { Prefix = kv.Key, Count = kv.Value }).ToList();
            var list = this.FindControl<ListBox>("FiltersList"); if (list != null) { list.ItemsSource = _filterItems.Select(i => i.Label).ToList(); }
            AppendBuildLog($"Analyzed filters: {_filterItems.Count} categories");
        }
        catch (Exception ex) { AppendBuildLog($"Analyze failed: {ex.Message}"); }
    }

    private void MarkFilters(bool whitelist)
    {
        var lb = this.FindControl<ListBox>("FiltersList"); if (lb == null) return;
        var selected = lb.SelectedItems?.Cast<string>().ToList() ?? new List<string>();
        if (selected.Count == 0) return;
        foreach (var s in selected)
        {
            var idx = _filterItems.FindIndex(i => i.Label == s);
            if (idx >= 0)
            {
                if (whitelist) { _filterItems[idx].Whitelisted = true; _filterItems[idx].Blacklisted = false; }
                else { _filterItems[idx].Blacklisted = true; _filterItems[idx].Whitelisted = false; }
            }
        }
        AppendBuildLog($"Marked {selected.Count} {(whitelist?"whitelisted":"blacklisted")} categories");
    }

    private async void SaveFiltersBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var wl = _filterItems.Where(i=>i.Whitelisted).Select(i=>i.Prefix).ToArray();
            var bl = _filterItems.Where(i=>i.Blacklisted).Select(i=>i.Prefix).ToArray();
            var pBox = this.FindControl<TextBox>("PresetsBox"); var dir = pBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(dir)) dir = _presetsRoot;
            Directory.CreateDirectory(dir!);
            var name = $"filters-{_currentMap}-{DateTime.Now:yyyyMMdd-HHmmss}.json";
            var path = Path.Combine(dir!, name);
            var preset = new { dataset = "default", global = new { filters = new { whitelist = wl, blacklist = bl } }, maps = new Dictionary<string, object>() };
            File.WriteAllText(path, JsonSerializer.Serialize(preset, new JsonSerializerOptions { WriteIndented = true }));
            AppendBuildLog($"Saved filters preset: {path}");
        }
        catch (Exception ex) { await ShowMessage("Error", ex.Message); }
    }

    private void OnOpened(object? sender, EventArgs e)
    {
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        if (cacheBox != null) cacheBox.Text = _cacheRoot;
        var presetsBox = this.FindControl<TextBox>("PresetsBox");
        if (presetsBox != null) presetsBox.Text = _presetsRoot;
        Directory.CreateDirectory(_cacheRoot);
        Directory.CreateDirectory(_presetsRoot);
        RefreshMaps();
        InitLayersTab();
    }

    private void RescanBtn_Click(object? sender, RoutedEventArgs e)
    {
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        var cache = cacheBox?.Text?.Trim() ?? _cacheRoot;
        if (!string.IsNullOrWhiteSpace(cache)) Directory.CreateDirectory(cache);
        RefreshMaps();
        InitLayersTab();
    }

    private async void SavePresetBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var nameBox = this.FindControl<TextBox>("PresetNameBox");
            var name = nameBox?.Text?.Trim();
            if (string.IsNullOrWhiteSpace(name)) name = "Preset" + DateTime.Now.ToString("yyyyMMddHHmmss");
            var pBox = this.FindControl<TextBox>("PresetsBox");
            var presetsDir = pBox?.Text?.Trim();
            if (string.IsNullOrWhiteSpace(presetsDir)) presetsDir = _presetsRoot;
            presetsDir = Path.GetFullPath(presetsDir);
            var path = Path.Combine(presetsDir, name + ".json");

            var maps = GetMapDirs();
            var preset = new
            {
                dataset = "default",
                global = new { m2 = Array.Empty<object>(), wmo = Array.Empty<object>() },
                maps = maps.ToDictionary(m => m.Map, _ => new { tiles = new Dictionary<string, object>() })
            };
            Directory.CreateDirectory(presetsDir);
            File.WriteAllText(path, JsonSerializer.Serialize(preset, new JsonSerializerOptions { WriteIndented = true }));
            await ShowMessage("Preset saved", path);
        }
        catch (Exception ex)
        {
            await ShowMessage("Error", ex.Message);
        }
    }

    private async void LoadPresetBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var options = new FilePickerOpenOptions
            {
                Title = "Open Preset",
                AllowMultiple = false,
                FileTypeFilter = new List<FilePickerFileType>
                {
                    new FilePickerFileType("JSON") { Patterns = new List<string> { "*.json" } }
                }
            };
            var results = await this.StorageProvider.OpenFilePickerAsync(options);
            if (results == null || results.Count == 0) return;
            var file = results[0];
            await using var stream = await file.OpenReadAsync();
            using var reader = new StreamReader(stream);
            var json = await reader.ReadToEndAsync();
            using var doc = JsonDocument.Parse(json);
            await ShowMessage("Preset loaded", file.Name);
        }
        catch (Exception ex)
        {
            await ShowMessage("Error", ex.Message);
        }
    }

    private void OpenPresetsBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var pBox = this.FindControl<TextBox>("PresetsBox");
            var path = pBox?.Text?.Trim();
            if (string.IsNullOrWhiteSpace(path)) path = _presetsRoot;
            path = Path.GetFullPath(path);
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);
            var psi = new System.Diagnostics.ProcessStartInfo { FileName = path, UseShellExecute = true };
            System.Diagnostics.Process.Start(psi);
        }
        catch { }
    }

    private void OpenCacheBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var cBox = this.FindControl<TextBox>("CacheBox");
            var path = cBox?.Text?.Trim();
            if (string.IsNullOrWhiteSpace(path)) path = _cacheRoot;
            path = Path.GetFullPath(path);
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);
            var psi = new System.Diagnostics.ProcessStartInfo { FileName = path, UseShellExecute = true };
            System.Diagnostics.Process.Start(psi);
        }
        catch { }
    }

    private async System.Threading.Tasks.Task ShowMessage(string title, string message)
    {
        var dlg = new Window
        {
            Width = 400,
            Height = 160,
            Title = title,
            Content = new TextBlock { Text = message, Margin = new Avalonia.Thickness(12) }
        };
        await dlg.ShowDialog(this);
    }

    private void RefreshMaps()
    {
        var rows = GetMapDirs();
        var grid = this.FindControl<DataGrid>("MapsGrid");
        if (grid != null) grid.ItemsSource = rows;
    }

    private List<MapRow> GetMapDirs()
    {
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        var cache = cacheBox?.Text?.Trim();
        var root = string.IsNullOrWhiteSpace(cache) ? _cacheRoot : cache;
        var result = new List<MapRow>();
        if (!Directory.Exists(root)) return result;
        foreach (var dir in Directory.EnumerateDirectories(root))
        {
            var map = Path.GetFileName(dir);
            var tileCsv = Path.Combine(dir, "tile_layers.csv");
            var layersJson = Path.Combine(dir, "layers.json");
            int rows = 0;
            if (File.Exists(tileCsv))
            {
                try { rows = File.ReadLines(tileCsv).Skip(1).Count(); } catch { rows = 0; }
            }
            result.Add(new MapRow
            {
                Map = map,
                Path = dir,
                HasTileLayers = File.Exists(tileCsv) ? "yes" : "no",
                HasLayersJson = File.Exists(layersJson) ? "yes" : "no",
                TileLayerRows = rows
            });
        }
        return result.OrderBy(r => r.Map, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public class MapRow
    {
        public string Map { get; set; } = string.Empty;
        public string Path { get; set; } = string.Empty;
        public string HasTileLayers { get; set; } = "no";
        public string HasLayersJson { get; set; } = "no";
        public int TileLayerRows { get; set; }
    }

    // ===== Layers tab logic =====
    private void InitLayersTab()
    {
        var mapCombo = this.FindControl<ComboBox>("LayersMapCombo");
        if (mapCombo == null) return;
        var maps = GetMapDirs();
        mapCombo.ItemsSource = maps.Select(m => m.Map).OrderBy(s => s, StringComparer.OrdinalIgnoreCase).ToList();
        mapCombo.SelectionChanged += (_, __) => { OnMapSelected(); };
        if (mapCombo.ItemCount > 0 && mapCombo.SelectedIndex < 0) mapCombo.SelectedIndex = 0;

        var tileCombo = this.FindControl<ComboBox>("TileSelectBox");
        if (tileCombo != null)
            tileCombo.SelectionChanged += (_, __) => { OnTileComboSelectionChanged(); };

        var addBtn = this.FindControl<Button>("AddRangeBtn");
        if (addBtn != null) addBtn.Click += AddRangeBtn_Click;
        var clearBtn = this.FindControl<Button>("ClearTileCustomBtn");
        if (clearBtn != null) clearBtn.Click += ClearTileCustomBtn_Click;
        var gridPanel = this.FindControl<Panel>("TileGridPanel");
        if (gridPanel != null)
        {
            gridPanel.PointerPressed += TileGrid_PointerPressed;
            gridPanel.PointerMoved += TileGrid_PointerMoved;
            gridPanel.PointerReleased += TileGrid_PointerReleased;
        }
        var heatToggle = this.FindControl<CheckBox>("HeatmapToggle");
        if (heatToggle != null)
        {
            heatToggle.Checked += (_, __) => RenderTileGrid();
            heatToggle.Unchecked += (_, __) => RenderTileGrid();
        }
    }

    private void OnTileComboSelectionChanged()
    {
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        _focusedTileKey = val;
        _selectedTiles.Clear();
        _selectedTiles.Add(val);
        RenderTileLayers(); RenderMinimap(); RenderTileCustom(); RenderTileGrid();
    }

    private void OnMapSelected()
    {
        var mapCombo = this.FindControl<ComboBox>("LayersMapCombo");
        if (mapCombo == null) return;
        var map = mapCombo.SelectedItem as string;
        if (string.IsNullOrWhiteSpace(map)) return;
        _currentMap = map;
        EnsureMapLoaded(map);
        PopulateTileCombo(map);
        RenderTileGrid();
        RenderTileLayers();
        RenderMinimap();
        RenderTileCustom();
    }

    private void EnsureMapLoaded(string map)
    {
        if (_rowsByMap.ContainsKey(map)) return;
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        var root = cacheBox?.Text?.Trim();
        if (string.IsNullOrWhiteSpace(root)) root = _cacheRoot;
        var dir = Path.Combine(root!, map);
        var csv = Path.Combine(dir, "tile_layers.csv");
        var rows = new List<TileLayerRow>();
        if (File.Exists(csv))
        {
            foreach (var line in File.ReadLines(csv).Skip(1))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(',');
                // Format: map,tile_x,tile_y,type,layer,range_start,range_end,count (8 columns)
                // Some variants may omit the 'map' column (7 columns). Detect dynamically.
                if (parts.Length < 7) continue;
                int off = parts.Length - 7; // 1 when map column present, 0 otherwise
                int.TryParse(parts[off + 0], out var tx);
                int.TryParse(parts[off + 1], out var ty);
                var type = parts[off + 2].Trim();
                int.TryParse(parts[off + 3], out var layer);
                int.TryParse(parts[off + 4], out var min);
                int.TryParse(parts[off + 5], out var max);
                int.TryParse(parts[off + 6], out var count);
                rows.Add(new TileLayerRow { TileX = tx, TileY = ty, Type = type, Layer = layer, Min = min, Max = max, Count = count });
            }
        }
        _rowsByMap[map] = rows;
        // Reset per-tile state caches
        _baseM2.Clear(); _baseWmo.Clear(); _customM2.Clear(); _customWmo.Clear();
    }

    private static string TileKey(int x, int y) => $"{x},{y}";

    private void PopulateTileCombo(string map)
    {
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox");
        if (tileCombo == null) return;
        var keys = new SortedSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var r in _rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>())
            keys.Add(TileKey(r.TileX, r.TileY));
        var list = keys.ToList();
        tileCombo.ItemsSource = list;
        if (list.Count > 0) tileCombo.SelectedIndex = 0;
        RenderTileGrid();
    }

    private void EnsureTileBaseState(string map, int x, int y)
    {
        var key = TileKey(x, y);
        if (!_baseM2.ContainsKey(key))
        {
            var rows = (_rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>()).Where(r => r.TileX == x && r.TileY == y).ToList();
            _baseM2[key] = rows.Where(r => string.Equals(r.Type, "M2", StringComparison.OrdinalIgnoreCase))
                .OrderBy(r => r.Layer).Select(r => new TileEntry { Type = "M2", Layer = r.Layer, Min = r.Min, Max = r.Max, Count = r.Count, Enabled = true }).ToList();
            _baseWmo[key] = rows.Where(r => !string.Equals(r.Type, "M2", StringComparison.OrdinalIgnoreCase))
                .OrderBy(r => r.Layer).Select(r => new TileEntry { Type = "WMO", Layer = r.Layer, Min = r.Min, Max = r.Max, Count = r.Count, Enabled = true }).ToList();
            _customM2[key] = _customM2.GetValueOrDefault(key) ?? new List<TileEntry>();
            _customWmo[key] = _customWmo.GetValueOrDefault(key) ?? new List<TileEntry>();
        }
    }

    private void RenderTileLayers()
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        var header = this.FindControl<TextBlock>("LayersHeader");
        var host = this.FindControl<StackPanel>("TileLayersList"); if (host == null) return;
        host.Children.Clear();

        if (_selectedTiles.Count > 1)
        {
            if (header != null) header.Text = $"Selection Layers ({_selectedTiles.Count})";
            var rows = _rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>();
            var sel = new HashSet<string>(_selectedTiles, StringComparer.OrdinalIgnoreCase);
            var grouped = rows.Where(r => sel.Contains(TileKey(r.TileX, r.TileY)))
                .GroupBy(r => new { r.Type, r.Layer })
                .Select(g => new { g.Key.Type, g.Key.Layer, Min = g.Min(x => x.Min), Max = g.Max(x => x.Max), Avg = g.Average(x => (double)x.Count) })
                .OrderBy(g => g.Type).ThenBy(g => g.Layer).ToList();
            foreach (var g in grouped)
            {
                var btn = new Button { Content = $"[{g.Type}] L{g.Layer}: {g.Min}-{g.Max} (~{g.Avg:0.0})" };
                btn.Margin = new Thickness(0, 2, 0, 2);
                btn.Click += (_, __) =>
                {
                    var minBox = this.FindControl<TextBox>("AddRangeMinBox");
                    var maxBox = this.FindControl<TextBox>("AddRangeMaxBox");
                    var typeBox = this.FindControl<ComboBox>("AddRangeTypeBox");
                    if (minBox != null) minBox.Text = g.Min.ToString();
                    if (maxBox != null) maxBox.Text = g.Max.ToString();
                    if (typeBox != null)
                    {
                        for (int i = 0; i < typeBox.ItemCount; i++)
                        {
                            if ((typeBox.Items![i] as ComboBoxItem)?.Content?.ToString().Equals(g.Type, StringComparison.OrdinalIgnoreCase) == true)
                            { typeBox.SelectedIndex = i; break; }
                        }
                    }
                };
                host.Children.Add(btn);
            }
            return;
        }

        if (header != null) header.Text = "Per‑Tile Layers";
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var parts = val.Split(','); if (parts.Length != 2) return;
        if (!int.TryParse(parts[0], out var x)) return; if (!int.TryParse(parts[1], out var y)) return;
        EnsureTileBaseState(map, x, y);
        var key = TileKey(x, y);
        void addRow(TileEntry r)
        {
            var cb = new CheckBox { IsChecked = r.Enabled, Content = $"[{r.Type}] L{r.Layer}: {r.Min}-{r.Max} ({r.Count})" };
            cb.Margin = new Thickness(0, 2, 0, 2);
            cb.Checked += (_, __) => r.Enabled = true;
            cb.Unchecked += (_, __) => r.Enabled = false;
            host.Children.Add(cb);
        }
        foreach (var r in _baseM2[key]) addRow(r);
        foreach (var r in _baseWmo[key]) addRow(r);
    }

    private void RenderTileCustom()
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var parts = val.Split(','); if (parts.Length != 2) return;
        if (!int.TryParse(parts[0], out var x)) return; if (!int.TryParse(parts[1], out var y)) return;
        EnsureTileBaseState(map, x, y);
        var key = TileKey(x, y);
        var host = this.FindControl<StackPanel>("TileCustomList"); if (host == null) return;
        host.Children.Clear();
        int idx = 0;
        foreach (var e in _customM2[key].Concat(_customWmo[key]))
        {
            var panel = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
            var cb = new CheckBox { IsChecked = e.Enabled, Content = $"[Custom {e.Type}] {e.Min}-{e.Max}" };
            var del = new Button { Content = "✕", Width = 24 };
            int localIdx = idx++;
            del.Click += (_, __) => { if (e.Type == "M2") _customM2[key].Remove(e); else _customWmo[key].Remove(e); RenderTileCustom(); };
            cb.Checked += (_, __) => e.Enabled = true; cb.Unchecked += (_, __) => e.Enabled = false;
            panel.Children.Add(cb); panel.Children.Add(del);
            host.Children.Add(panel);
        }
    }

    private void AddRangeBtn_Click(object? sender, RoutedEventArgs e)
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var parts = val.Split(','); if (parts.Length != 2) return;
        if (!int.TryParse(parts[0], out var x)) return; if (!int.TryParse(parts[1], out var y)) return;
        EnsureTileBaseState(map, x, y);
        var key = TileKey(x, y);

        var minBox = this.FindControl<TextBox>("AddRangeMinBox");
        var maxBox = this.FindControl<TextBox>("AddRangeMaxBox");
        var typeBox = this.FindControl<ComboBox>("AddRangeTypeBox");
        if (minBox == null || maxBox == null || typeBox == null) return;
        if (!int.TryParse(minBox.Text?.Trim(), out var min)) return;
        if (!int.TryParse(maxBox.Text?.Trim(), out var max)) return;
        var type = (typeBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "M2";
        var entry = new TileEntry { Type = type, Layer = -1, Min = min, Max = max, Count = 0, Enabled = true };
        if (string.Equals(type, "M2", StringComparison.OrdinalIgnoreCase)) _customM2[key].Add(entry); else _customWmo[key].Add(entry);
        RenderTileCustom();
    }

    private void ClearTileCustomBtn_Click(object? sender, RoutedEventArgs e)
    {
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var key = val;
        _customM2[key] = new List<TileEntry>();
        _customWmo[key] = new List<TileEntry>();
        RenderTileCustom();
    }

    private void RenderMinimap()
    {
        var img = this.FindControl<Image>("MinimapImage"); if (img == null) return;
        img.Source = null;
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var parts = val.Split(','); if (parts.Length != 2) return;
        if (!int.TryParse(parts[0], out var x)) return; if (!int.TryParse(parts[1], out var y)) return;
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        var root = cacheBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(root)) root = _cacheRoot;
        var dir = Path.Combine(root!, map, "minimap");
        if (!Directory.Exists(dir)) return;
        // Try to find <map>_x_y.png anywhere under minimap
        var pattern = $"{map}_{x}_{y}.png";
        string? found = Directory.EnumerateFiles(dir, pattern, SearchOption.AllDirectories).FirstOrDefault();
        if (found == null) return;
        try { img.Source = new Bitmap(found); } catch { }
    }

    private void RenderTileGrid()
    {
        var panel = this.FindControl<Panel>("TileGridPanel"); if (panel == null) return;
        panel.Children.Clear();
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        // Presence from CSV
        var rows = _rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>();
        var present = new HashSet<(int x, int y)>();
        foreach (var r in rows) present.Add((r.TileX, r.TileY));
        // Presence from minimap files
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        var root = cacheBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(root)) root = _cacheRoot;
        var miniDir = Path.Combine(root!, map, "minimap");
        if (Directory.Exists(miniDir))
        {
            try
            {
                foreach (var f in Directory.EnumerateFiles(miniDir, "*.png", SearchOption.AllDirectories))
                {
                    var name = Path.GetFileNameWithoutExtension(f);
                    if (string.IsNullOrWhiteSpace(name)) continue;
                    var parts = name.Split('_');
                    if (parts.Length >= 3 && parts[0].Equals(map, StringComparison.OrdinalIgnoreCase)
                        && int.TryParse(parts[^2], out var fx)
                        && int.TryParse(parts[^1], out var fy))
                    {
                        present.Add((fx, fy));
                    }
                }
            }
            catch { }
        }

        // Heatmap setup
        var heatToggle = this.FindControl<CheckBox>("HeatmapToggle");
        bool heat = heatToggle?.IsChecked == true;
        var earliestByTile = new Dictionary<(int x, int y), int>();
        int gMin = 0, gMax = 0;
        if (heat)
        {
            try
            {
                foreach (var g in rows.GroupBy(r => (r.TileX, r.TileY)))
                {
                    var earliest = g.Min(x => x.Min);
                    earliestByTile[(g.Key.TileX, g.Key.TileY)] = earliest;
                }
                if (earliestByTile.Count > 0)
                {
                    gMin = earliestByTile.Values.Min();
                    gMax = earliestByTile.Values.Max();
                }
            }
            catch { heat = false; }
        }

        static Avalonia.Media.Color Lerp(Avalonia.Media.Color a, Avalonia.Media.Color b, double t)
        {
            if (t < 0) t = 0; if (t > 1) t = 1;
            byte A = (byte)(a.A + (b.A - a.A) * t);
            byte R = (byte)(a.R + (b.R - a.R) * t);
            byte G = (byte)(a.G + (b.G - a.G) * t);
            byte B = (byte)(a.B + (b.B - a.B) * t);
            return new Avalonia.Media.Color(A, R, G, B);
        }
        static Avalonia.Media.Color HeatColor(double t)
        {
            // 0 -> blue, 0.5 -> yellow, 1 -> red
            var blue = Avalonia.Media.Colors.DodgerBlue;
            var yellow = Avalonia.Media.Colors.Gold;
            var red = Avalonia.Media.Colors.Red;
            return t < 0.5 ? Lerp(blue, yellow, t * 2) : Lerp(yellow, red, (t - 0.5) * 2);
        }

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var key = (x, y);
                var has = present.Contains(key);
                Avalonia.Media.Color cellColor;
                if (heat && has && earliestByTile.TryGetValue(key, out var val) && gMax > gMin)
                {
                    var t = (double)(val - gMin) / (double)(gMax - gMin);
                    cellColor = HeatColor(t);
                }
                else
                {
                    var color = has ? 0xFF2E7D32 : 0xFF2B2B2B;
                    cellColor = new Avalonia.Media.Color((byte)((color>>24)&0xFF),(byte)((color>>16)&0xFF),(byte)((color>>8)&0xFF),(byte)(color&0xFF));
                }
                var b = new Border
                {
                    Background = new Avalonia.Media.SolidColorBrush(cellColor),
                    BorderBrush = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Colors.Black),
                    BorderThickness = new Thickness(1),
                    Margin = new Thickness(0.5),
                    Tag = $"{x},{y}"
                };
                // selection border styling will be applied below
                // no per-cell handlers; we handle selection on panel-level for marquee
                if (!string.IsNullOrEmpty(_focusedTileKey) && string.Equals(b.Tag as string, _focusedTileKey, StringComparison.OrdinalIgnoreCase))
                {
                    b.BorderBrush = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Colors.Yellow);
                    b.BorderThickness = new Thickness(2);
                }
                else if (_selectedTiles.Contains(b.Tag as string ?? string.Empty))
                {
                    b.BorderBrush = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Colors.DodgerBlue);
                    b.BorderThickness = new Thickness(2);
                }
                panel.Children.Add(b);
            }
        }
        // Initial highlight
        var tileComboInit = this.FindControl<ComboBox>("TileSelectBox");
        var valInit = tileComboInit?.SelectedItem as string;
        if (!string.IsNullOrWhiteSpace(valInit)) { _focusedTileKey = valInit!; _selectedTiles.Clear(); _selectedTiles.Add(valInit!); }
    }

    private void HighlightTile(string val)
    {
        _focusedTileKey = val;
        _selectedTiles.Clear();
        _selectedTiles.Add(val);
        RenderTileGrid();
    }

    private (int x, int y) TileFromPointer(Panel panel, PointerEventArgs e)
    {
        var p = e.GetPosition(panel);
        var cw = Math.Max(1.0, panel.Bounds.Width / 64.0);
        var ch = Math.Max(1.0, panel.Bounds.Height / 64.0);
        var xi = (int)(p.X / cw); var yi = (int)(p.Y / ch);
        if (xi < 0) xi = 0; if (xi > 63) xi = 63; if (yi < 0) yi = 0; if (yi > 63) yi = 63;
        return (xi, yi);
    }

    private void ApplySelectionRect((int x, int y) cur)
    {
        int minx = Math.Min(_dragStartCell.x, cur.x), maxx = Math.Max(_dragStartCell.x, cur.x);
        int miny = Math.Min(_dragStartCell.y, cur.y), maxy = Math.Max(_dragStartCell.y, cur.y);
        var rect = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int yy = miny; yy <= maxy; yy++)
            for (int xx = minx; xx <= maxx; xx++)
                rect.Add(TileKey(xx, yy));

        HashSet<string> next;
        if (_dragMode == SelMode.Replace)
            next = rect;
        else if (_dragMode == SelMode.Add)
            next = new HashSet<string>((_dragBase ?? new HashSet<string>()), StringComparer.OrdinalIgnoreCase) { };
        else
            next = new HashSet<string>((_dragBase ?? new HashSet<string>()), StringComparer.OrdinalIgnoreCase);

        if (_dragMode == SelMode.Add)
            foreach (var k in rect) next.Add(k);
        else if (_dragMode == SelMode.Remove)
            foreach (var k in rect) next.Remove(k);

        _selectedTiles.Clear(); foreach (var k in next) _selectedTiles.Add(k);
        _focusedTileKey = TileKey(cur.x, cur.y);

        // Sync dropdown
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox");
        if (tileCombo != null)
        {
            var seq = (tileCombo.ItemsSource as System.Collections.IEnumerable) ?? Array.Empty<object>();
            int idx = 0; foreach (var it in seq) { if ((it as string) == _focusedTileKey) { tileCombo.SelectedIndex = idx; break; } idx++; }
        }

        RenderTileGrid(); RenderTileLayers(); RenderMinimap(); RenderTileCustom();
    }

    private void TileGrid_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        var panel = sender as Panel; if (panel == null) return;
        _isDragging = true;
        _dragStartCell = TileFromPointer(panel, e);
        _dragBase = new HashSet<string>(_selectedTiles, StringComparer.OrdinalIgnoreCase);
        _dragMode = (e.KeyModifiers & KeyModifiers.Control) != 0 ? SelMode.Add : (e.KeyModifiers & KeyModifiers.Shift) != 0 ? SelMode.Remove : SelMode.Replace;
        e.Pointer.Capture(panel);
        ApplySelectionRect(_dragStartCell);
    }

    private void TileGrid_PointerMoved(object? sender, PointerEventArgs e)
    {
        if (!_isDragging) return;
        var panel = sender as Panel; if (panel == null) return;
        var cur = TileFromPointer(panel, e);
        ApplySelectionRect(cur);
    }

    private void TileGrid_PointerReleased(object? sender, PointerReleasedEventArgs e)
    {
        if (!_isDragging) return;
        _isDragging = false;
        e.Pointer.Capture(null);
    }
}
