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
            tileCombo.SelectionChanged += (_, __) => { RenderTileLayers(); RenderMinimap(); RenderTileCustom(); };

        var addBtn = this.FindControl<Button>("AddRangeBtn");
        if (addBtn != null) addBtn.Click += AddRangeBtn_Click;
        var clearBtn = this.FindControl<Button>("ClearTileCustomBtn");
        if (clearBtn != null) clearBtn.Click += ClearTileCustomBtn_Click;
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
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var parts = val.Split(','); if (parts.Length != 2) return;
        if (!int.TryParse(parts[0], out var x)) return; if (!int.TryParse(parts[1], out var y)) return;
        EnsureTileBaseState(map, x, y);
        var key = TileKey(x, y);
        var host = this.FindControl<StackPanel>("TileLayersList"); if (host == null) return;
        host.Children.Clear();
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

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                var key = (x, y);
                var has = present.Contains(key);
                var color = has ? 0xFF2E7D32 : 0xFF2B2B2B;
                var b = new Border
                {
                    Background = new Avalonia.Media.SolidColorBrush(new Avalonia.Media.Color((byte)((color>>24)&0xFF),(byte)((color>>16)&0xFF),(byte)((color>>8)&0xFF),(byte)(color&0xFF))),
                    BorderBrush = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Colors.Black),
                    BorderThickness = new Thickness(1),
                    Margin = new Thickness(0.5),
                    Tag = $"{x},{y}"
                };
                b.PointerPressed += (s, e) =>
                {
                    var val = (s as Border)?.Tag as string; if (string.IsNullOrWhiteSpace(val)) return;
                    var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
                    // Set selection to the clicked tile
                    var seq = (tileCombo.ItemsSource as System.Collections.IEnumerable) ?? Array.Empty<object>();
                    int idx = 0; foreach (var it in seq)
                    {
                        var sv = it as string; if (sv == val) { tileCombo.SelectedIndex = idx; break; }
                        idx++;
                    }
                    HighlightTile(val);
                    RenderTileLayers(); RenderMinimap(); RenderTileCustom();
                };
                panel.Children.Add(b);
            }
        }
        // Initial highlight
        var tileComboInit = this.FindControl<ComboBox>("TileSelectBox");
        var valInit = tileComboInit?.SelectedItem as string; if (!string.IsNullOrWhiteSpace(valInit)) HighlightTile(valInit!);
    }

    private void HighlightTile(string val)
    {
        var panel = this.FindControl<Panel>("TileGridPanel"); if (panel == null) return;
        foreach (var child in panel.Children)
        {
            if (child is Border b)
            {
                var isSel = string.Equals(b.Tag as string, val, StringComparison.OrdinalIgnoreCase);
                b.BorderBrush = new Avalonia.Media.SolidColorBrush(isSel ? Avalonia.Media.Colors.Yellow : Avalonia.Media.Colors.Black);
                b.BorderThickness = new Thickness(isSel ? 2 : 1);
            }
        }
    }
}
