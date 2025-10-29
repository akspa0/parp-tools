using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Text;
using System.Diagnostics;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.Media.Imaging;
using Avalonia;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Controls.Primitives;
using Avalonia.Threading;
using WoWRollback.DbcModule;

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
    private const bool _areasUiEnabled = false; // TEMP: disable AreaID UI/overlay
    private int _timeCenter = 0; // current time slider center
    private int _globalSliceMin = 0, _globalSliceMax = 0; // global time slice

    // Area grouping state (per-map)
    private sealed class AreaData
    {
        public Dictionary<string, int> AreaByTile { get; } = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<int, HashSet<string>> TilesByArea { get; } = new();
        public Dictionary<int, HashSet<int>> ChildrenByParent { get; } = new();
        public Dictionary<int, HashSet<string>> TilesByParent { get; } = new();
        public Dictionary<int, string> AreaName { get; } = new();
        public Dictionary<int, string> ParentName { get; } = new();
        public SortedDictionary<int, string> ParentLabel { get; } = new();
        public Dictionary<int, SortedDictionary<int, string>> SubzoneLabelByParent { get; } = new();
    }

    private void UpdateTimeLabel()
    {
        var bandBox = this.FindControl<ComboBox>("BandSizeBox");
        int band = 128;
        var selItem = bandBox?.SelectedItem as ComboBoxItem;
        if (selItem != null && int.TryParse(selItem.Content?.ToString(), out var parsed) && parsed > 0) band = parsed;
        int half = band / 2;
        _globalSliceMin = _timeCenter - half;
        _globalSliceMax = _timeCenter + half;
        var label = this.FindControl<TextBlock>("TimeLabel");
        if (label != null) label.Text = $"Range: {_globalSliceMin}–{_globalSliceMax}";
        var btn = this.FindControl<Button>("BaselineTimeSliceBtn");
        if (btn != null) btn.Content = $"Only Range: {_globalSliceMin}–{_globalSliceMax}";
    }

    private void UpdateTimeRangeFromSelection()
    {
        var tSlider = this.FindControl<Slider>("TimeSlider"); if (tSlider == null) return;
        var stats = this.FindControl<TextBlock>("TimeStats");
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) { tSlider.IsEnabled = false; if (stats!=null) stats.Text = string.Empty; return; }
        var rows = _rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>();
        IEnumerable<TileLayerRow> scope;
        if (_selectedTiles.Count > 0)
        {
            var sel = new HashSet<string>(_selectedTiles, StringComparer.OrdinalIgnoreCase);
            scope = rows.Where(r => sel.Contains(TileKey(r.TileX, r.TileY)));
        }
        else scope = rows;
        if (!scope.Any()) { tSlider.IsEnabled = false; if (stats!=null) stats.Text = string.Empty; return; }
        int min = scope.Min(r => r.Min);
        int max = scope.Max(r => r.Max);
        if (min >= max) { tSlider.IsEnabled = false; if (stats!=null) stats.Text = string.Empty; return; }
        tSlider.IsEnabled = true;
        tSlider.Minimum = min;
        tSlider.Maximum = max;
        // keep value within bounds
        if (tSlider.Value < tSlider.Minimum || tSlider.Value > tSlider.Maximum)
            tSlider.Value = Math.Clamp(_timeCenter == 0 ? min : _timeCenter, (int)tSlider.Minimum, (int)tSlider.Maximum);
        _timeCenter = (int)Math.Round(tSlider.Value);
        UpdateTimeLabel();
        // Removed: ApplyTimeBandForCurrent();
    }



    private void BaselineEnableAllBtn_Click(object? sender, RoutedEventArgs e)
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        foreach (var kstr in _selectedTiles.ToArray())
        {
            var parts = kstr.Split(','); if (parts.Length != 2) continue;
            if (!int.TryParse(parts[0], out var x)) continue; if (!int.TryParse(parts[1], out var y)) continue;
            EnsureTileBaseState(map, x, y);
            var key = TileKey(x, y);
            foreach (var e2 in _baseM2[key]) e2.Enabled = true;
            foreach (var e2 in _baseWmo[key]) e2.Enabled = true;
        }
        RenderTileLayers();
    }

    private void BaselineDisableAllBtn_Click(object? sender, RoutedEventArgs e)
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        foreach (var kstr in _selectedTiles.ToArray())
        {
            var parts = kstr.Split(','); if (parts.Length != 2) continue;
            if (!int.TryParse(parts[0], out var x)) continue; if (!int.TryParse(parts[1], out var y)) continue;
            EnsureTileBaseState(map, x, y);
            var key = TileKey(x, y);
            foreach (var e2 in _baseM2[key]) e2.Enabled = false;
            foreach (var e2 in _baseWmo[key]) e2.Enabled = false;
        }
        RenderTileLayers();
    }

    private void BaselineTimeSliceBtn_Click(object? sender, RoutedEventArgs e)
    {
        ApplyGlobalSlice();
    }

    private void ApplyGlobalSlice()
    {
        var stats = this.FindControl<TextBlock>("TimeStats");
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) { if (stats != null) stats.Text = string.Empty; return; }
        if (_selectedTiles.Count == 0) { if (stats != null) stats.Text = string.Empty; return; }
        int tilesInSlice = 0, sinks = 0;

        foreach (var kstr in _selectedTiles.ToArray())
        {
            var parts = kstr.Split(','); if (parts.Length != 2) continue;
            if (!int.TryParse(parts[0], out var x)) continue; if (!int.TryParse(parts[1], out var y)) continue;
            EnsureTileBaseState(map, x, y);
            var k = TileKey(x, y);
            bool anyOverlapM2 = false, anyOverlapWmo = false;
            foreach (var e in _baseM2[k]) { bool hit = e.Min <= _globalSliceMax && e.Max >= _globalSliceMin; e.Enabled = hit; anyOverlapM2 |= hit; }
            foreach (var e in _baseWmo[k]) { bool hit = e.Min <= _globalSliceMax && e.Max >= _globalSliceMin; e.Enabled = hit; anyOverlapWmo |= hit; }
            if (anyOverlapM2 || anyOverlapWmo) tilesInSlice++;
            if (!anyOverlapM2 && !anyOverlapWmo) sinks++;
        }
        if (stats != null) stats.Text = $"Applied {_globalSliceMin}–{_globalSliceMax} · tiles in slice: {tilesInSlice} · sinks: {sinks}";
        // RenderTileLayers(); // Hidden
    }

    private static string BuildPrepareLayersArgs(string cliProj, string clientRoot, string outRoot, string mapsArg, DataSourcePayload p)
    {
        var sb = new StringBuilder();
        sb.Append($"run --project \"{cliProj}\" -- prepare-layers --client-root \"{clientRoot}\" --out \"{outRoot}\" --maps {mapsArg}");
        if (!string.IsNullOrWhiteSpace(p.DbdDir)) sb.Append($" --dbd-dir \"{p.DbdDir}\"");
        if (!string.IsNullOrWhiteSpace(p.DbcDir)) sb.Append($" --dbc-dir \"{p.DbcDir}\"");
        if (!string.IsNullOrWhiteSpace(p.Build)) sb.Append($" --build \"{p.Build}\"");
        if (!string.IsNullOrWhiteSpace(p.LkClient)) sb.Append($" --lk-client-path \"{p.LkClient}\"");
        if (!string.IsNullOrWhiteSpace(p.LkDbcDir)) sb.Append($" --lk-dbc-dir \"{p.LkDbcDir}\"");
        return sb.ToString();
    }

    private static string BuildAlphaToLkArgs(string cliProj, string wdtPath, int maxUniqueId, string alphaOut, string lkOut, string? lkClient, string? lkDbcDir)
    {
        var sb = new StringBuilder();
        sb.Append($"run --project \"{cliProj}\" -- alpha-to-lk --input \"{wdtPath}\" --max-uniqueid {maxUniqueId} --bury-depth -5000.0 --out \"{alphaOut}\"");
        if (!string.IsNullOrWhiteSpace(lkOut)) sb.Append($" --lk-out \"{lkOut}\"");
        if (!string.IsNullOrWhiteSpace(lkClient)) sb.Append($" --lk-client-path \"{lkClient}\"");
        if (!string.IsNullOrWhiteSpace(lkDbcDir)) sb.Append($" --lk-dbc-dir \"{lkDbcDir}\"");
        return sb.ToString();
    }

    private static string? TryFindWdtPath(string datasetRoot, string map)
    {
        try
        {
            var cands = new[]
            {
                Path.Combine(datasetRoot, "World", "Maps", map, map + ".wdt"),
                Path.Combine(datasetRoot, "tree", "World", "Maps", map, map + ".wdt"),
                Path.Combine(datasetRoot, "World", map, map + ".wdt"),
                Path.Combine(datasetRoot, "tree", "World", map, map + ".wdt")
            };
            foreach (var p in cands) if (File.Exists(p)) return p;
            try
            {
                var found = Directory.EnumerateFiles(datasetRoot, map + ".wdt", SearchOption.AllDirectories)
                    .Where(p => p.EndsWith(Path.DirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase))
                    .Take(1).ToList();
                if (found.Count > 0) return found[0];
            }
            catch { }
        }
        catch { }
        return null;
    }

    private async void RecompileMapBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) { await ShowMessage("Info", "Select a map first."); return; }
            var cacheBox = this.FindControl<TextBox>("CacheBox"); var cacheRoot = cacheBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(cacheRoot)) cacheRoot = _cacheRoot;
            var datasetRoot = TryGetDatasetRootFromCache(cacheRoot!);
            if (string.IsNullOrWhiteSpace(datasetRoot))
            {
                var rootBox = this.FindControl<TextBox>("DataRootBox"); var fallback = rootBox?.Text?.Trim();
                if (!string.IsNullOrWhiteSpace(fallback)) datasetRoot = fallback;
            }
            if (string.IsNullOrWhiteSpace(datasetRoot)) { await ShowMessage("Info", "Set Data Sources root and build cache first."); return; }

            var wdt = TryFindWdtPath(datasetRoot!, map);
            if (string.IsNullOrWhiteSpace(wdt) || !File.Exists(wdt!)) { await ShowMessage("Error", "Could not locate WDT for map."); return; }

            var threshold = _globalSliceMax;
            var outRoot = Path.GetFullPath(Path.Combine(_cacheRoot, "..", "output", map, DateTime.Now.ToString("yyyyMMdd-HHmmss")));
            var alphaOut = Path.Combine(outRoot, "alpha_out");
            var lkOut = Path.Combine(outRoot, "lk_adts");
            Directory.CreateDirectory(alphaOut);
            Directory.CreateDirectory(lkOut);

            var p = BuildDataSourcePayload();
            var cliProj = ResolveProjectCsproj("WoWRollback", "WoWRollback.Cli");
            if (string.IsNullOrWhiteSpace(cliProj) || !File.Exists(cliProj)) { await ShowMessage("Error", "CLI project not found."); return; }

            var sessionObj = new
            {
                map,
                wdt = wdt,
                sliceCenter = _timeCenter,
                sliceWidth = (this.FindControl<ComboBox>("BandSizeBox")?.SelectedItem as ComboBoxItem)?.Content?.ToString(),
                rangeMin = _globalSliceMin,
                rangeMax = _globalSliceMax,
                threshold = threshold,
                lkClient = p.LkClient,
                lkDbcDir = p.LkDbcDir
            };
            File.WriteAllText(Path.Combine(outRoot, "session.json"), JsonSerializer.Serialize(sessionObj, new JsonSerializerOptions { WriteIndented = true }));

            var args = BuildAlphaToLkArgs(cliProj, wdt!, threshold, alphaOut, lkOut, p.LkClient, p.LkDbcDir);
            File.WriteAllText(Path.Combine(outRoot, "commands.txt"), args);

            ShowLoading("Recompiling map...");
            var psi = new ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, ev) => { if (!string.IsNullOrEmpty(ev.Data)) AppendBuildLog(ev.Data!); };
            proc.ErrorDataReceived += (_, ev) => { if (!string.IsNullOrEmpty(ev.Data)) AppendBuildLog(ev.Data!); };
            AppendBuildLog($"[recompile] Starting: {args}");
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            await proc.WaitForExitAsync();
            AppendBuildLog($"[recompile] Exit code: {proc.ExitCode}");
            HideLoading();
            if (proc.ExitCode == 0)
            {
                try { var psiOpen = new System.Diagnostics.ProcessStartInfo { FileName = outRoot, UseShellExecute = true }; System.Diagnostics.Process.Start(psiOpen); } catch { }
                await ShowMessage("Done", outRoot);
            }
            else
            {
                await ShowMessage("Recompile failed", $"Exit code: {proc.ExitCode}");
            }
        }
        catch (Exception ex) { HideLoading(); await ShowMessage("Error", ex.Message); }
    }
    private readonly Dictionary<string, AreaData> _areasByMap = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string?> _wmoByMap = new(StringComparer.OrdinalIgnoreCase);
    private static string _lastWmoDebug = string.Empty;

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
        var refreshPresets = this.FindControl<Button>("RefreshPresetsBtn");
        if (refreshPresets != null) refreshPresets.Click += (_, __) => RefreshPresetsList();
        var openCache = this.FindControl<Button>("OpenCacheBtn");
        if (openCache != null) openCache.Click += OpenCacheBtn_Click;

        // Area grouping controls
        var parentAreaCombo = this.FindControl<ComboBox>("ParentAreaCombo");
        if (parentAreaCombo != null) parentAreaCombo.SelectionChanged += (_, __) => OnParentAreaChanged();
        var selParentBtn = this.FindControl<Button>("SelectParentBtn"); if (selParentBtn != null) selParentBtn.Click += SelectParentBtn_Click;
        var selSubsBtn = this.FindControl<Button>("SelectSubzonesBtn"); if (selSubsBtn != null) selSubsBtn.Click += SelectSubzonesBtn_Click;
        var clearAreaSelBtn = this.FindControl<Button>("ClearAreaSelBtn"); if (clearAreaSelBtn != null) clearAreaSelBtn.Click += (_, __) => { _selectedTiles.Clear(); RenderTileGrid(); RenderTileLayers(); RenderMinimap(); };

        // Build tab hookups
        var buildOut = this.FindControl<TextBox>("BuildOutBox");
        if (buildOut != null) buildOut.Text = Path.Combine(_cacheRoot, "..", "output");
        var pickOut = this.FindControl<Button>("PickBuildOutBtn"); if (pickOut != null) pickOut.Click += PickBuildOutBtn_Click;
        var saveSel = this.FindControl<Button>("SaveSelectionPresetBtn"); if (saveSel != null) saveSel.Click += SaveSelectionPresetBtn_Click;
        var prep = this.FindControl<Button>("PrepareBtn"); if (prep != null) prep.Click += PrepareBtn_Click;
        var openOut = this.FindControl<Button>("OpenOutBtn"); if (openOut != null) openOut.Click += OpenOutBtn_Click;
        var genBaseline = this.FindControl<Button>("GenBaselineBtn"); if (genBaseline != null) genBaseline.Click += GenBaselineBtn_Click;

        // Data Sources tab hookups
        var dsLoose = this.FindControl<RadioButton>("DsTypeLoose"); if (dsLoose != null) dsLoose.IsCheckedChanged += (_, __) => UpdateCascVisibility();
        var dsInstall = this.FindControl<RadioButton>("DsTypeInstall"); if (dsInstall != null) dsInstall.IsCheckedChanged += (_, __) => UpdateCascVisibility();
        var dsCasc = this.FindControl<RadioButton>("DsTypeCasc"); if (dsCasc != null) dsCasc.IsCheckedChanged += (_, __) => UpdateCascVisibility();
        var pickRoot = this.FindControl<Button>("PickDataRootBtn"); if (pickRoot != null) pickRoot.Click += PickDataRootBtn_Click;
        var pickDbd = this.FindControl<Button>("PickDataDbdBtn"); if (pickDbd != null) pickDbd.Click += PickDataDbdBtn_Click;
        var pickDbc = this.FindControl<Button>("PickDataDbcBtn"); if (pickDbc != null) pickDbc.Click += PickDataDbcBtn_Click;
        var pickList = this.FindControl<Button>("PickDataListfileBtn"); if (pickList != null) pickList.Click += PickDataListfileBtn_Click;
        var pickLkClient = this.FindControl<Button>("PickLkClientBtn"); if (pickLkClient != null) pickLkClient.Click += PickLkClientBtn_Click;
        var pickLkDbc = this.FindControl<Button>("PickLkDbcBtn"); if (pickLkDbc != null) pickLkDbc.Click += PickLkDbcBtn_Click;
        var dsPrev = this.FindControl<Button>("DataPreviewBtn"); if (dsPrev != null) dsPrev.Click += DataPreviewBtn_Click;
        var dsLoad = this.FindControl<Button>("DataLoadBtn"); if (dsLoad != null) dsLoad.Click += DataLoadBtn_Click;
        var saveDefaults = this.FindControl<Button>("SaveDataDefaultsBtn"); if (saveDefaults != null) saveDefaults.Click += SaveDataDefaultsBtn_Click;
        var loadDefaults = this.FindControl<Button>("LoadDataDefaultsBtn"); if (loadDefaults != null) loadDefaults.Click += LoadDataDefaultsBtn_Click;

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

    private async void PickDataRootBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Select Data Root", AllowMultiple = false });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("DataRootBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private async void PickDataDbdBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Select DBD Directory", AllowMultiple = false });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("DataDbdBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private async void PickDataDbcBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Select DBC Directory", AllowMultiple = false });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("DataDbcBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private async void PickDataListfileBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select CASC Listfile",
            AllowMultiple = false,
            FileTypeFilter = new List<FilePickerFileType> { new FilePickerFileType("CSV / TXT") { Patterns = new List<string> { "*.csv", "*.txt" } } }
        });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("DataListfileBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private async void PickLkClientBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Select LK Client Root", AllowMultiple = false });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("LkClientBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private async void PickLkDbcBtn_Click(object? sender, RoutedEventArgs e)
    {
        var res = await this.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = "Select LK DBFilesClient Directory", AllowMultiple = false });
        if (res != null && res.Count > 0)
        {
            var box = this.FindControl<TextBox>("LkDbcBox"); if (box != null) box.Text = res[0].Path.LocalPath;
        }
    }

    private void AppendBuildLog(string line)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var log = this.FindControl<TextBox>("BuildLogBox"); if (log == null) return;
            log.Text += (string.IsNullOrEmpty(log.Text) ? string.Empty : Environment.NewLine) + line;
            try { log.CaretIndex = log.Text?.Length ?? 0; } catch { }
        });
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
            var tilesObj = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
            foreach (var key in selection.Where(s => !string.IsNullOrWhiteSpace(s)))
            {
                var m2 = (_customM2.GetValueOrDefault(key) ?? new List<TileEntry>())
                    .Select(e => new { min = e.Min, max = e.Max })
                    .Cast<object>()
                    .ToList();
                var wmo = (_customWmo.GetValueOrDefault(key) ?? new List<TileEntry>())
                    .Select(e => new { min = e.Min, max = e.Max })
                    .Cast<object>()
                    .ToList();
                tilesObj[key] = new { custom = new { m2 = m2, wmo = wmo } };
            }
            var preset = new
            {
                dataset = "default",
                global = new { m2 = Array.Empty<object>(), wmo = Array.Empty<object>() },
                maps = new Dictionary<string, object> { [map] = new { tiles = tilesObj } }
            };
            File.WriteAllText(path, JsonSerializer.Serialize(preset, new JsonSerializerOptions { WriteIndented = true }));
            AppendBuildLog($"Saved selection preset: {path}");
            RefreshPresetsList();
        }
        catch (Exception ex) { await ShowMessage("Error", ex.Message); }
    }

    private async void PrepareBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            ShowLoading("Preparing per-map layers (this may take a while)...");
            var p = BuildDataSourcePayload();
            var buildOutBox = this.FindControl<TextBox>("BuildOutBox");
            var outRoot = this.FindControl<TextBox>("CacheBox")?.Text?.Trim();
            var root = p.Root;
            if (string.IsNullOrWhiteSpace(root)) { AppendBuildLog("[prepare] Please set Root."); return; }

            if (string.Equals(p.Type, "casc", StringComparison.OrdinalIgnoreCase))
            {
                AppendBuildLog("[prepare] CASC pipeline not implemented yet in GUI.");
                return;
            }

            var cliProj = ResolveProjectCsproj("WoWRollback", "WoWRollback.Cli");
            if (string.IsNullOrWhiteSpace(cliProj) || !File.Exists(cliProj))
            {
                AppendBuildLog("[prepare] Could not locate WoWRollback.Cli.csproj.");
                return;
            }

            var (_, method, resolvedVer, _) = DiscoverMapFolders(p);
            // Ensure output root is version-scoped
            if (string.IsNullOrWhiteSpace(outRoot))
            {
                var verLabel = string.IsNullOrWhiteSpace(resolvedVer) ? "unknown" : resolvedVer;
                outRoot = Path.Combine(_cacheRoot, verLabel);
                var cacheBox = this.FindControl<TextBox>("CacheBox"); if (cacheBox != null) cacheBox.Text = outRoot;
            }
            Directory.CreateDirectory(outRoot!);
            var mapsArg = "all"; // process all Alpha WDTs discovered
            AppendBuildLog($"[prepare] Processing all maps (discovery: {method}); out={outRoot}");

            var psi = new ProcessStartInfo
            {
                FileName = "dotnet",
                Arguments = BuildPrepareLayersArgs(cliProj, root!, outRoot!, mapsArg, p),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, ev) => { if (!string.IsNullOrEmpty(ev.Data)) AppendBuildLog(ev.Data!); };
            proc.ErrorDataReceived += (_, ev) => { if (!string.IsNullOrEmpty(ev.Data)) AppendBuildLog(ev.Data!); };
            AppendBuildLog("[prepare] Starting CLI...");
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            await proc.WaitForExitAsync();
            AppendBuildLog($"[prepare] Exit code: {proc.ExitCode}");

            RefreshMaps();
            InitLayersTab();
            OnMapSelected();
            if (proc.ExitCode == 0)
            {
                AppendBuildLog("[prepare] Prepare complete.");
                var tabs = this.FindControl<TabControl>("MainTabs");
                if (tabs != null) tabs.SelectedIndex = 1; // Layers
            }
        }
        catch (Exception ex)
        {
            AppendBuildLog("[prepare] ERROR: " + ex.Message);
        }
        finally { HideLoading(); }
    }

    private static string ResolveProjectCsproj(string folder, string projectName)
    {
        try
        {
            var start = new DirectoryInfo(AppContext.BaseDirectory);
            for (var dir = start; dir != null; dir = dir.Parent)
            {
                var csproj = Path.Combine(dir.FullName, folder, projectName, projectName + ".csproj");
                if (File.Exists(csproj)) return csproj;
            }
        }
        catch { }
        return Path.Combine(folder, projectName);
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

    // ===== Data Sources logic =====
    private void PopulateDataVersions()
    {
        var combo = this.FindControl<ComboBox>("DataVersionCombo"); if (combo == null) return;
        combo.ItemsSource = new[] { "(auto)", "0.5.3", "0.5.5", "0.6.0", "3.3.5.12340" };
        if (combo.SelectedIndex < 0) combo.SelectedIndex = 0;
    }

    private void UpdateCascVisibility()
    {
        var row = this.FindControl<Grid>("CascListfileRow"); if (row == null) return;
        var casc = this.FindControl<RadioButton>("DsTypeCasc");
        row.IsVisible = casc?.IsChecked == true;
    }

    private sealed class DataSourcePayload
    {
        public string Type = "loose"; // loose | install | casc
        public string? Root;
        public string? Version;
        public string? Build; // full build string for DBCD (e.g., 0.5.3.3368)
        public string? DbdDir;
        public string? DbcDir;
        public string? Listfile;
        public string? OutputDir;
        public string? LkClient;
        public string? LkDbcDir;
    }

    private DataSourcePayload BuildDataSourcePayload()
    {
        var type = (this.FindControl<RadioButton>("DsTypeCasc")?.IsChecked == true) ? "casc" : (this.FindControl<RadioButton>("DsTypeInstall")?.IsChecked == true ? "install" : "loose");
        var root = this.FindControl<TextBox>("DataRootBox")?.Text?.Trim();
        var verSel = this.FindControl<ComboBox>("DataVersionCombo");
        var ver = verSel?.SelectedItem as string;
        if (!string.IsNullOrWhiteSpace(ver) && ver!.Equals("(auto)", StringComparison.OrdinalIgnoreCase)) ver = null;
        var build = this.FindControl<TextBox>("DataBuildBox")?.Text?.Trim();
        var dbd = this.FindControl<TextBox>("DataDbdBox")?.Text?.Trim();
        var dbc = this.FindControl<TextBox>("DataDbcBox")?.Text?.Trim();
        var list = this.FindControl<TextBox>("DataListfileBox")?.Text?.Trim();
        var lkClient = this.FindControl<TextBox>("LkClientBox")?.Text?.Trim();
        var lkDbc = this.FindControl<TextBox>("LkDbcBox")?.Text?.Trim();
        var cacheBox = this.FindControl<TextBox>("CacheBox");
        var outDir = cacheBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(outDir)) outDir = _cacheRoot;
        return new DataSourcePayload { Type = type, Root = root, Version = ver, Build = build, DbdDir = dbd, DbcDir = dbc, Listfile = list, OutputDir = outDir, LkClient = lkClient, LkDbcDir = lkDbc };
    }

    private void DataPreviewBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var p = BuildDataSourcePayload();
            var notes = new List<string>();
            if (!string.IsNullOrWhiteSpace(p.Root) && p.Type == "casc")
            {
                if (File.Exists(Path.Combine(p.Root!, ".build.info"))) notes.Add(".build.info detected");
                if (string.IsNullOrWhiteSpace(p.Listfile)) notes.Add("CASC listfile not set");
            }

            var (maps, method, resolvedVer, err) = DiscoverMapFolders(p);
            if (!string.IsNullOrWhiteSpace(err)) notes.Add(err!);
            var sample = maps.Take(10).ToList();
            var buildUsed = ResolveBuildTag(resolvedVer ?? p.Version ?? string.Empty, p.Build, p.Root);
            var obj = new { type = p.Type, version = p.Version, build = p.Build, buildUsed, resolvedVersion = resolvedVer, method, mapsCount = maps.Count, sampleMaps = sample, notes };
            var box = this.FindControl<TextBox>("DataPreviewBox"); if (box != null) box.Text = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true });
        }
        catch (Exception ex)
        {
            var box = this.FindControl<TextBox>("DataPreviewBox"); if (box != null) box.Text = ex.Message;
        }
    }

    private void AppendDataLoadLog(string line)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var box = this.FindControl<TextBox>("DataLoadLogBox"); if (box == null) return;
            box.Text += (string.IsNullOrEmpty(box.Text) ? string.Empty : Environment.NewLine) + line;
            try { box.CaretIndex = box.Text?.Length ?? 0; } catch { }
        });
    }

    private void ShowLoading(string message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            var overlay = this.FindControl<Border>("LoadingOverlay");
            var text = this.FindControl<TextBlock>("LoadingText");
            if (text != null) text.Text = message;
            if (overlay != null) overlay.IsVisible = true;
        });
    }

    private void HideLoading()
    {
        Dispatcher.UIThread.Post(() =>
        {
            var overlay = this.FindControl<Border>("LoadingOverlay");
            if (overlay != null) overlay.IsVisible = false;
        });
    }

    private async void DataLoadBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            ShowLoading("Discovering maps and initializing cache...");
            var p = BuildDataSourcePayload();
            AppendDataLoadLog($"[Load] Type={p.Type} Root={p.Root}");
            if (string.IsNullOrWhiteSpace(p.OutputDir)) p.OutputDir = _cacheRoot;

            var (mapFolders, method, resolvedVer, err) = DiscoverMapFolders(p);
            if (!string.IsNullOrWhiteSpace(err)) AppendDataLoadLog($"[Load] Note: {err}");
            if (mapFolders.Count == 0) { AppendDataLoadLog("[Load] No maps discovered"); return; }

            // Use version-scoped cache directory
            var verLabel = string.IsNullOrWhiteSpace(resolvedVer) ? "unknown" : resolvedVer;
            p.OutputDir = Path.Combine(p.OutputDir!, verLabel);
            var cacheBoxCtl = this.FindControl<TextBox>("CacheBox"); if (cacheBoxCtl != null) cacheBoxCtl.Text = p.OutputDir;
            Directory.CreateDirectory(p.OutputDir!);
            // If no explicit version was set, persist the resolved version into the manifest
            if (string.IsNullOrWhiteSpace(p.Version)) p.Version = resolvedVer;
            foreach (var map in mapFolders)
            {
                var outMap = Path.Combine(p.OutputDir!, map);
                Directory.CreateDirectory(outMap);
                File.WriteAllText(Path.Combine(outMap, "tile_layers.csv"), "tile_x,tile_y,type,layer,min,max,count\n");
                File.WriteAllText(Path.Combine(outMap, "layers.json"), "{\n  \"layers\": []\n}\n");
                AppendDataLoadLog($"[Load] Initialized cache for {map}");
            }
            WriteSourcesJson(p, method, resolvedVer, mapFolders);

            AppendDataLoadLog("[Load] Done");
            WriteDatasetManifest(p);
            RefreshMaps();
            InitLayersTab();
            OnMapSelected();
            // Move to Build tab for the next step in the flow
            var tabs = this.FindControl<TabControl>("MainTabs");
            if (tabs != null) tabs.SelectedIndex = 2; // Build

            // Auto-prepare if checked
            var auto = this.FindControl<CheckBox>("AutoPrepareAfterLoadCheck");
            if (auto?.IsChecked == true)
            {
                PrepareBtn_Click(null!, null!);
            }
        }
        catch (Exception ex)
        {
            AppendDataLoadLog("ERROR: " + ex.Message);
        }
        finally { HideLoading(); }
    }

    private string GetGuiDefaultsPath()
    {
        try
        {
            var repo = Environment.CurrentDirectory;
            var path = Path.Combine(repo, "WoWRollback", "work", "gui-data-sources.json");
            return Path.GetFullPath(path);
        }
        catch { return Path.Combine(_cacheRoot, "..", "gui-data-sources.json"); }
    }

    private void SaveDataDefaultsBtn_Click(object? sender, RoutedEventArgs e)
    {
        try
        {
            var p = BuildDataSourcePayload();
            var auto = this.FindControl<CheckBox>("AutoPrepareAfterLoadCheck");
            var obj = new
            {
                type = p.Type,
                root = p.Root,
                version = p.Version,
                build = p.Build,
                dbdDir = p.DbdDir,
                dbcDir = p.DbcDir,
                listfile = p.Listfile,
                lkClient = p.LkClient,
                lkDbcDir = p.LkDbcDir,
                autoPrepare = auto?.IsChecked == true
            };
            var json = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true });
            var path = GetGuiDefaultsPath();
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, json);
            AppendDataLoadLog($"Saved defaults: {path}");
        }
        catch (Exception ex) { AppendDataLoadLog("Save defaults failed: " + ex.Message); }
    }

    private void LoadDataDefaultsBtn_Click(object? sender, RoutedEventArgs e)
    {
        TryLoadGuiDefaults(showMessage: true);
    }

    private void TryLoadGuiDefaults(bool showMessage = false)
    {
        try
        {
            var path = GetGuiDefaultsPath();
            if (!File.Exists(path)) { if (showMessage) AppendDataLoadLog("No saved defaults"); return; }
            var json = File.ReadAllText(path);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var type = root.TryGetProperty("type", out var t) ? t.GetString() : "loose";
            var r = root.TryGetProperty("root", out var rEl) ? rEl.GetString() : null;
            var version = root.TryGetProperty("version", out var vEl) ? vEl.GetString() : null;
            var build = root.TryGetProperty("build", out var bEl) ? bEl.GetString() : null;
            var dbd = root.TryGetProperty("dbdDir", out var dEl) ? dEl.GetString() : null;
            var dbc = root.TryGetProperty("dbcDir", out var cEl) ? cEl.GetString() : null;
            var list = root.TryGetProperty("listfile", out var lEl) ? lEl.GetString() : null;
            var lkClient = root.TryGetProperty("lkClient", out var lcEl) ? lcEl.GetString() : null;
            var lkDbc = root.TryGetProperty("lkDbcDir", out var ldEl) ? ldEl.GetString() : null;
            var auto = root.TryGetProperty("autoPrepare", out var aEl) && aEl.ValueKind == JsonValueKind.True;

            var casc = this.FindControl<RadioButton>("DsTypeCasc");
            var install = this.FindControl<RadioButton>("DsTypeInstall");
            var loose = this.FindControl<RadioButton>("DsTypeLoose");
            if (string.Equals(type, "casc", StringComparison.OrdinalIgnoreCase)) { if (casc != null) casc.IsChecked = true; }
            else if (string.Equals(type, "install", StringComparison.OrdinalIgnoreCase)) { if (install != null) install.IsChecked = true; }
            else { if (loose != null) loose.IsChecked = true; }

            var rootBox = this.FindControl<TextBox>("DataRootBox"); if (rootBox != null) rootBox.Text = r ?? string.Empty;
            var buildBox = this.FindControl<TextBox>("DataBuildBox"); if (buildBox != null) buildBox.Text = build ?? string.Empty;
            var dbdBox = this.FindControl<TextBox>("DataDbdBox"); if (dbdBox != null) dbdBox.Text = dbd ?? string.Empty;
            var dbcBox = this.FindControl<TextBox>("DataDbcBox"); if (dbcBox != null) dbcBox.Text = dbc ?? string.Empty;
            var listBox = this.FindControl<TextBox>("DataListfileBox"); if (listBox != null) listBox.Text = list ?? string.Empty;
            var lkClientBox = this.FindControl<TextBox>("LkClientBox"); if (lkClientBox != null) lkClientBox.Text = lkClient ?? string.Empty;
            var lkDbcBox = this.FindControl<TextBox>("LkDbcBox"); if (lkDbcBox != null) lkDbcBox.Text = lkDbc ?? string.Empty;
            var autoChk = this.FindControl<CheckBox>("AutoPrepareAfterLoadCheck"); if (autoChk != null) autoChk.IsChecked = auto;

            var combo = this.FindControl<ComboBox>("DataVersionCombo");
            if (combo != null)
            {
                if (string.IsNullOrWhiteSpace(version)) combo.SelectedIndex = 0; // (auto)
                else
                {
                    var items = combo.Items?.Cast<string>().ToList() ?? new List<string>();
                    var idx = items.FindIndex(s => string.Equals(s, version, StringComparison.OrdinalIgnoreCase));
                    if (idx >= 0) combo.SelectedIndex = idx; else combo.SelectedIndex = 0;
                }
            }

            UpdateCascVisibility();
            if (showMessage) AppendDataLoadLog($"Loaded defaults: {path}");
        }
        catch (Exception ex) { AppendDataLoadLog("Load defaults failed: " + ex.Message); }
    }

    private void WriteDatasetManifest(DataSourcePayload p)
    {
        try
        {
            var path = Path.Combine(p.OutputDir ?? _cacheRoot, ".dataset.json");
            var obj = new
            {
                type = p.Type,
                root = p.Root,
                version = p.Version,
                build = p.Build,
                dbdDir = p.DbdDir,
                dbcDir = p.DbcDir,
                listfile = p.Listfile,
                generatedAt = DateTimeOffset.UtcNow.ToString("o")
            };
            File.WriteAllText(path, JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { }
    }

    private (List<string> Maps, string Method, string ResolvedVersion, string? Error) DiscoverMapFolders(DataSourcePayload p)
    {
        var maps = new List<string>();
        string method = "none";
        string? error = null;
        var resolved = p.Version;
        if (string.IsNullOrWhiteSpace(resolved) && !string.IsNullOrWhiteSpace(p.Root)) resolved = InferVersionFromPath(p.Root!);

        // Try Map.dbc discovery when DBD + DBC + version available
        try
        {
            if (!string.IsNullOrWhiteSpace(p.DbdDir) && Directory.Exists(p.DbdDir!))
            {
                string? dbc = null;
                if (!string.IsNullOrWhiteSpace(p.DbcDir) && Directory.Exists(p.DbcDir!)) dbc = p.DbcDir;
                else if (!string.IsNullOrWhiteSpace(p.Root))
                {
                    var d1 = Path.Combine(p.Root!, "DBFilesClient"); if (Directory.Exists(d1)) dbc = d1;
                    var d2 = Path.Combine(p.Root!, "dbc"); if (dbc == null && Directory.Exists(d2)) dbc = d2;
                }
                if (!string.IsNullOrWhiteSpace(dbc) && !string.IsNullOrWhiteSpace(resolved))
                {
                    var reader = new MapDbcReader(p.DbdDir!);
                    // Resolve build tag: prefer user-provided, else detect from .build.info, else known fallback for version
                    var buildTag = ResolveBuildTag(resolved!, p.Build, p.Root);
                    var res = reader.ReadMaps(buildTag, dbc!);
                    if (res.Success)
                    {
                        maps = res.Maps.Select(m => m.Folder).Where(s => !string.IsNullOrWhiteSpace(s)).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
                        method = "Map.dbc";
                        if (string.IsNullOrWhiteSpace(p.Build)) p.Build = buildTag; // persist auto-detected build for manifests/UI
                        return (maps, method, resolved!, null);
                    }
                    else
                    {
                        error = res.ErrorMessage ?? "Map.dbc read failed";
                    }
                }
            }
        }
        catch (Exception ex)
        {
            error = ex.Message;
        }

        // Fallback: file system folder scan
        try
        {
            if (!string.IsNullOrWhiteSpace(p.Root))
            {
                var mapsDir = FindMapsDir(p.Root!);
                if (mapsDir != null && Directory.Exists(mapsDir))
                {
                    maps = Directory.EnumerateDirectories(mapsDir).Select(Path.GetFileName).Where(s => !string.IsNullOrWhiteSpace(s)).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
                    method = "folder-scan";
                }
            }
        }
        catch (Exception ex)
        {
            error = error ?? ex.Message;
        }

        return (maps, method, resolved ?? string.Empty, error);
    }

    private static string ResolveBuildTag(string resolvedVersion, string? userBuild, string? root)
    {
        if (!string.IsNullOrWhiteSpace(userBuild))
        {
            var ub = userBuild!.Trim();
            // If user typed just a numeric build ID, combine with version
            if (Regex.IsMatch(ub, "^\\d{3,6}$")) return string.IsNullOrWhiteSpace(resolvedVersion) ? ub : (resolvedVersion + "." + ub);
            // If user typed a full version or version.build, accept as-is
            if (Regex.IsMatch(ub, "^\\d+\\.\\d+(\\.\\d+){0,2}$")) return ub;
            // Otherwise, fall through to detection/fallback
        }
        var detected = TryDetectBuildFromRoot(root);
        if (!string.IsNullOrWhiteSpace(detected)) return detected!;
        var fallback = AutoBuildForVersion(resolvedVersion);
        return string.IsNullOrWhiteSpace(fallback) ? resolvedVersion : fallback!;
    }

    private static string? TryDetectBuildFromRoot(string? root)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(root)) return null;
            var candidates = new[]
            {
                Path.Combine(root!, ".build.info"),
                Path.Combine(root!, "Data", ".build.info")
            };
            foreach (var p in candidates)
            {
                if (!File.Exists(p)) continue;
                var lines = File.ReadAllLines(p);
                if (lines.Length == 0) continue;
                // Heuristic parse: try pipe/space/semicolon separated
                var header = lines[0];
                var delim = header.Contains('|') ? '|' : (header.Contains(';') ? ';' : ' ');
                var cols = header.Split(delim, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                int idxBuildId = Array.FindIndex(cols, c => c.Equals("BuildId", StringComparison.OrdinalIgnoreCase) || c.Equals("Build ID", StringComparison.OrdinalIgnoreCase) || c.Equals("BuildID", StringComparison.OrdinalIgnoreCase));
                int idxVersion = Array.FindIndex(cols, c => c.Equals("Version", StringComparison.OrdinalIgnoreCase) || c.Equals("VersionsName", StringComparison.OrdinalIgnoreCase) || c.Equals("VersionName", StringComparison.OrdinalIgnoreCase));
                for (int i = 1; i < lines.Length; i++)
                {
                    var parts = lines[i].Split(delim, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                    if (idxVersion >= 0 && idxVersion < parts.Length)
                    {
                        var ver = parts[idxVersion];
                        // normalize (strip trailing letters like 'a')
                        ver = Regex.Replace(ver, @"[^0-9\.]+$", "");
                        string? buildId = null;
                        if (idxBuildId >= 0 && idxBuildId < parts.Length) buildId = parts[idxBuildId];
                        if (!string.IsNullOrWhiteSpace(ver))
                        {
                            if (!string.IsNullOrWhiteSpace(buildId) && Regex.IsMatch(buildId, "^\\d{4,6}$")) return ver + "." + buildId;
                            return ver; // at least return version
                        }
                    }
                }
            }
        }
        catch { }
        return null;
    }

    private static string? AutoBuildForVersion(string? version)
    {
        if (string.IsNullOrWhiteSpace(version)) return null;
        return version switch
        {
            "0.5.3" => "0.5.3.3368",
            "0.5.5" => "0.5.5.3494",
            "0.6.0" => "0.6.0.3694",
            "3.3.5" => "3.3.5.12340",
            _ => null
        };
    }

    private void WriteSourcesJson(DataSourcePayload p, string method, string resolvedVersion, IReadOnlyList<string> folders)
    {
        try
        {
            var path = Path.Combine(p.OutputDir ?? _cacheRoot, "sources.json");
            var obj = new { method, resolvedVersion, folders };
            File.WriteAllText(path, JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { }
    }

    private static string? InferVersionFromPath(string path)
    {
        try
        {
            var segments = path.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries);
            foreach (var seg in segments.Reverse())
            {
                var m = Regex.Match(seg, "^(?<v>\\d+\\.\\d+(\\.\\d+)?)");
                if (m.Success) return m.Groups["v"].Value;
            }
        }
        catch { }
        return null;
    }

    private static string? FindMapsDir(string root)
    {
        try
        {
            var d1 = Path.Combine(root, "World", "Maps"); if (Directory.Exists(d1)) return d1;
            var d2 = Path.Combine(root, "tree", "World", "Maps"); if (Directory.Exists(d2)) return d2;
        }
        catch { }
        return null;
    }

    private static string? TryGetDatasetRootFromCache(string cacheRoot)
    {
        try
        {
            var path = Path.Combine(cacheRoot, ".dataset.json");
            if (!File.Exists(path)) return null;
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            var r = root.TryGetProperty("root", out var rEl) ? rEl.GetString() : null;
            return string.IsNullOrWhiteSpace(r) ? null : r;
        }
        catch { return null; }
    }

    private static string? TryExtractGlobalWmo(string datasetRoot, string map)
    {
        try
        {
            var dbg = new StringBuilder();
            dbg.AppendLine($"datasetRoot={datasetRoot}");
            dbg.AppendLine($"map={map}");
            var cands = new[]
            {
                Path.Combine(datasetRoot, "World", "Maps", map, map + ".wdt"),
                Path.Combine(datasetRoot, "tree", "World", "Maps", map, map + ".wdt"),
                Path.Combine(datasetRoot, "World", map, map + ".wdt"),
                Path.Combine(datasetRoot, "tree", "World", map, map + ".wdt")
            };
            string? wdt = cands.FirstOrDefault(File.Exists);
            if (wdt == null)
            {
                // Last resort: search recursively for <map>.wdt anywhere under dataset root
                try
                {
                    var candidates = Directory.EnumerateFiles(datasetRoot, map + ".wdt", SearchOption.AllDirectories)
                        .Where(p => p.EndsWith(Path.DirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase))
                        .Take(1)
                        .ToList();
                    if (candidates.Count > 0) wdt = candidates[0];
                }
                catch { }
                if (wdt == null) { _lastWmoDebug = dbg.AppendLine("wdt=(not found)").ToString(); return null; }
            }
            dbg.AppendLine($"wdt={wdt}");

            // Preferred: proper chunk parse (supports Alpha MONM and retail MWMO)
            var parsed = TryParseWdtGlobalWmo(wdt, dbg);
            if (!string.IsNullOrWhiteSpace(parsed)) { _lastWmoDebug = dbg.AppendLine($"wmo={parsed}").ToString(); return parsed; }

            // Fallback: heuristic ASCII scan
            var bytes = File.ReadAllBytes(wdt);
            var list = new List<string>();
            var sb = new StringBuilder();
            void flush()
            {
                if (sb.Length >= 4) list.Add(sb.ToString());
                sb.Clear();
            }
            foreach (var b in bytes)
            {
                if (b == 0) { flush(); continue; }
                char ch = (char)b;
                if (ch >= 32 && ch <= 126)
                {
                    sb.Append(ch);
                }
                else
                {
                    flush();
                }
            }
            flush();
            var wmoFallback = list.FirstOrDefault(s => s.IndexOf(".wmo", StringComparison.OrdinalIgnoreCase) >= 0) ?? list.FirstOrDefault(s => !string.IsNullOrWhiteSpace(s));
            _lastWmoDebug = dbg.AppendLine($"fallback-strings={list.Count}").AppendLine($"wmo={wmoFallback ?? "(none)"}").ToString();
            return wmoFallback;
        }
        catch { return null; }
    }

    private static string? TryParseWdtGlobalWmo(string wdtPath, StringBuilder dbg)
    {
        try
        {
            using var fs = new FileStream(wdtPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: false);
            string? firstName = null;
            var chunks = new List<string>();
            while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                var idBytes = br.ReadBytes(4);
                if (idBytes.Length < 4) break;
                var fourCC = Encoding.ASCII.GetString(idBytes).ToUpperInvariant();
                int size = br.ReadInt32(); // little-endian
                if (size < 0 || br.BaseStream.Position + size > br.BaseStream.Length) return null;
                if (chunks.Count < 12) chunks.Add(fourCC + ":" + size);

                bool isWmoNames = fourCC == "MWMO" || fourCC == "MONM" || fourCC == "OMWM" || fourCC == "MNOM"; // include reversed aliases defensively
                if (!isWmoNames)
                {
                    // skip payload + padding to 4-byte boundary
                    long skip = size;
                    int pad = size & 3;
                    if (pad != 0) skip += (4 - pad);
                    br.BaseStream.Seek(skip, SeekOrigin.Current);
                    continue;
                }

                var payload = br.ReadBytes(size);
                // Split zero-terminated ASCII strings
                int start = 0;
                for (int i = 0; i <= payload.Length; i++)
                {
                    bool isEnd = (i == payload.Length) || payload[i] == 0;
                    if (isEnd)
                    {
                        if (i > start)
                        {
                            var s = Encoding.ASCII.GetString(payload, start, i - start);
                            if (!string.IsNullOrWhiteSpace(s))
                            {
                                if (firstName == null) firstName = s;
                                if (s.IndexOf(".wmo", StringComparison.OrdinalIgnoreCase) >= 0)
                                    return s;
                            }
                        }
                        start = i + 1;
                    }
                }

                // align to 4-byte boundary after payload
                int padAfter = size & 3;
                if (padAfter != 0)
                {
                    br.BaseStream.Seek(4 - padAfter, SeekOrigin.Current);
                }

                // If we didn't return, continue scanning
            }
            if (chunks.Count > 0) dbg.AppendLine($"chunks={string.Join(", ", chunks)}");
            if (!string.IsNullOrWhiteSpace(firstName)) dbg.AppendLine($"firstName={firstName}");
            else dbg.AppendLine("firstName=(none)");
            if (!string.IsNullOrWhiteSpace(firstName)) return firstName;
        }
        catch { }
        return null;
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
        RefreshPresetsList();
        PopulateDataVersions();
        UpdateCascVisibility();
        // Auto-load saved defaults if present
        TryLoadGuiDefaults();
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
            string? path = null;
            var list = this.FindControl<ListBox>("PresetsList");
            var pBox = this.FindControl<TextBox>("PresetsBox");
            var dir = pBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(dir)) dir = _presetsRoot;
            if (list != null && list.SelectedItem is string name && !string.IsNullOrWhiteSpace(name))
            {
                path = Path.Combine(dir!, name);
            }
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
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
                path = file.Path.LocalPath;
            }
            var json = await System.IO.File.ReadAllTextAsync(path!);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var map = _currentMap;
            if (string.IsNullOrWhiteSpace(map)) { await ShowMessage("Info", "Select a map first."); return; }

            // Apply tiles selection and custom ranges
            if (root.TryGetProperty("maps", out var mapsEl) && mapsEl.ValueKind == JsonValueKind.Object && mapsEl.TryGetProperty(map, out var mapEl))
            {
                if (mapEl.TryGetProperty("tiles", out var tilesEl) && tilesEl.ValueKind == JsonValueKind.Object)
                {
                    _selectedTiles.Clear();
                    foreach (var tileProp in tilesEl.EnumerateObject())
                    {
                        var key = tileProp.Name;
                        _selectedTiles.Add(key);
                        _customM2[key] = new List<TileEntry>();
                        _customWmo[key] = new List<TileEntry>();
                        if (tileProp.Value.ValueKind == JsonValueKind.Object && tileProp.Value.TryGetProperty("custom", out var customEl))
                        {
                            if (customEl.TryGetProperty("m2", out var m2El) && m2El.ValueKind == JsonValueKind.Array)
                            {
                                foreach (var elem in m2El.EnumerateArray())
                                {
                                    if (elem.TryGetProperty("min", out var minEl) && elem.TryGetProperty("max", out var maxEl) && minEl.TryGetInt32(out var min) && maxEl.TryGetInt32(out var max))
                                        _customM2[key].Add(new TileEntry { Type = "M2", Layer = -1, Min = min, Max = max, Enabled = true });
                                }
                            }
                            if (customEl.TryGetProperty("wmo", out var wmoEl) && wmoEl.ValueKind == JsonValueKind.Array)
                            {
                                foreach (var elem in wmoEl.EnumerateArray())
                                {
                                    if (elem.TryGetProperty("min", out var minEl) && elem.TryGetProperty("max", out var maxEl) && minEl.TryGetInt32(out var min) && maxEl.TryGetInt32(out var max))
                                        _customWmo[key].Add(new TileEntry { Type = "WMO", Layer = -1, Min = min, Max = max, Enabled = true });
                                }
                            }
                        }
                    }
                }

                // Apply baseline to selection (or all tiles if none selected)
                if (mapEl.TryGetProperty("baseline", out var baselineEl) && baselineEl.ValueKind == JsonValueKind.Object)
                {
                    var targetKeys = _selectedTiles.Count > 0
                        ? new HashSet<string>(_selectedTiles, StringComparer.OrdinalIgnoreCase)
                        : new HashSet<string>(((IEnumerable<string>)((_rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>())
                            .Select(r => TileKey(r.TileX, r.TileY)).Distinct())), StringComparer.OrdinalIgnoreCase);

                    static bool Intersects(int a1, int a2, int b1, int b2) => Math.Max(a1, b1) <= Math.Min(a2, b2);
                    var m2Ranges = new List<(int min, int max)>();
                    var wmoRanges = new List<(int min, int max)>();
                    if (baselineEl.TryGetProperty("m2", out var m2El) && m2El.ValueKind == JsonValueKind.Array)
                        foreach (var elem in m2El.EnumerateArray()) if (elem.TryGetProperty("min", out var minEl) && elem.TryGetProperty("max", out var maxEl) && minEl.TryGetInt32(out var min) && maxEl.TryGetInt32(out var max)) m2Ranges.Add((min, max));
                    if (baselineEl.TryGetProperty("wmo", out var wmoEl) && wmoEl.ValueKind == JsonValueKind.Array)
                        foreach (var elem in wmoEl.EnumerateArray()) if (elem.TryGetProperty("min", out var minEl) && elem.TryGetProperty("max", out var maxEl) && minEl.TryGetInt32(out var min) && maxEl.TryGetInt32(out var max)) wmoRanges.Add((min, max));

                    foreach (var key in targetKeys)
                    {
                        var parts = key.Split(','); if (parts.Length != 2) continue;
                        if (!int.TryParse(parts[0], out var x)) continue; if (!int.TryParse(parts[1], out var y)) continue;
                        EnsureTileBaseState(map, x, y);
                        foreach (var r in _baseM2[key]) r.Enabled = m2Ranges.Any(b => Intersects(r.Min, r.Max, b.min, b.max));
                        foreach (var r in _baseWmo[key]) r.Enabled = wmoRanges.Any(b => Intersects(r.Min, r.Max, b.min, b.max));
                    }
                }
            }

            // Fallback to global.baseline if no map baseline and selection exists
            else if (root.TryGetProperty("global", out var globalEl)
                     && globalEl.ValueKind == JsonValueKind.Object
                     && globalEl.TryGetProperty("baseline", out var baselineEl)
                     && baselineEl.ValueKind == JsonValueKind.Object)
            {
                var targetKeys = _selectedTiles.Count > 0
                    ? new HashSet<string>(_selectedTiles, StringComparer.OrdinalIgnoreCase)
                    : new HashSet<string>();
                if (targetKeys.Count > 0)
                {
                    static bool Intersects(int a1, int a2, int b1, int b2) => Math.Max(a1, b1) <= Math.Min(a2, b2);
                    var m2Ranges = new List<(int min, int max)>();
                    var wmoRanges = new List<(int min, int max)>();
                    if (baselineEl.TryGetProperty("m2", out var m2El) && m2El.ValueKind == JsonValueKind.Array)
                        foreach (var elem in m2El.EnumerateArray()) if (elem.TryGetProperty("min", out var minEl) && elem.TryGetProperty("max", out var maxEl) && minEl.TryGetInt32(out var min) && maxEl.TryGetInt32(out var max)) m2Ranges.Add((min, max));
                    if (baselineEl.TryGetProperty("wmo", out var wmoEl) && wmoEl.ValueKind == JsonValueKind.Array)
                        foreach (var elem in wmoEl.EnumerateArray()) if (elem.TryGetProperty("min", out var minEl) && elem.TryGetProperty("max", out var maxEl) && minEl.TryGetInt32(out var min) && maxEl.TryGetInt32(out var max)) wmoRanges.Add((min, max));

                    foreach (var key in targetKeys)
                    {
                        var parts = key.Split(','); if (parts.Length != 2) continue;
                        if (!int.TryParse(parts[0], out var x)) continue; if (!int.TryParse(parts[1], out var y)) continue;
                        EnsureTileBaseState(map, x, y);
                        foreach (var r in _baseM2[key]) r.Enabled = m2Ranges.Any(b => Intersects(r.Min, r.Max, b.min, b.max));
                        foreach (var r in _baseWmo[key]) r.Enabled = wmoRanges.Any(b => Intersects(r.Min, r.Max, b.min, b.max));
                    }
                }
            }

            RenderTileGrid();
            RenderTileLayers();
            RenderTileCustom();
            RenderMinimap();
            await ShowMessage("Preset loaded", System.IO.Path.GetFileName(path));
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

    private void RefreshPresetsList()
    {
        try
        {
            var pBox = this.FindControl<TextBox>("PresetsBox");
            var dir = pBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(dir)) dir = _presetsRoot;
            Directory.CreateDirectory(dir!);
            var files = Directory.EnumerateFiles(dir!, "*.json", SearchOption.TopDirectoryOnly)
                .Select(Path.GetFileName)
                .Where(n => !string.IsNullOrWhiteSpace(n))
                .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                .ToList();
            var lb = this.FindControl<ListBox>("PresetsList");
            if (lb != null) lb.ItemsSource = files;
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
        var datasetRoot = TryGetDatasetRootFromCache(root);

        static bool IsMapDir(string d)
        {
            return File.Exists(Path.Combine(d, "tile_layers.csv")) || File.Exists(Path.Combine(d, "layers.json"));
        }

        // First pass: collect immediate map directories
        var immediate = Directory.EnumerateDirectories(root).ToList();
        var mapsHere = new List<string>();
        foreach (var dir in immediate)
        {
            if (IsMapDir(dir)) mapsHere.Add(dir);
        }

        // If no direct map outputs found, look one level deeper (handles version folder like "0.5.3")
        if (mapsHere.Count == 0)
        {
            foreach (var dir in immediate)
            {
                foreach (var sub in Directory.EnumerateDirectories(dir))
                {
                    if (IsMapDir(sub)) mapsHere.Add(sub);
                }
            }
        }

        foreach (var dir in mapsHere)
        {
            var map = Path.GetFileName(dir);
            var tileCsv = Path.Combine(dir, "tile_layers.csv");
            var layersJson = Path.Combine(dir, "layers.json");
            int rows = 0;
            if (File.Exists(tileCsv))
            {
                try { rows = File.ReadLines(tileCsv).Skip(1).Count(); } catch { rows = 0; }
            }

            if (rows == 0 && !string.IsNullOrWhiteSpace(datasetRoot))
            {
                var c1 = Path.Combine(datasetRoot!, "World", "Maps", map, map + ".wdt");
                var c2 = Path.Combine(datasetRoot!, "tree", "World", "Maps", map, map + ".wdt");
                var c3 = Path.Combine(datasetRoot!, "World", map, map + ".wdt");
                var c4 = Path.Combine(datasetRoot!, "tree", "World", map, map + ".wdt");
                var hasWdt = File.Exists(c1) || File.Exists(c2) || File.Exists(c3) || File.Exists(c4);
                if (!hasWdt)
                {
                    try
                    {
                        hasWdt = Directory.EnumerateFiles(datasetRoot!, map + ".wdt", SearchOption.AllDirectories)
                            .Any(p => p.EndsWith(Path.DirectorySeparatorChar + map + ".wdt", StringComparison.OrdinalIgnoreCase));
                    }
                    catch { }
                }
                if (!hasWdt)
                {
                    AppendBuildLog($"[maps] Excluding {map}: no WDT found under dataset root");
                    continue;
                }
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
            heatToggle.IsCheckedChanged += (_, __) => RenderTileGrid();
        }
        var bandBox = this.FindControl<ComboBox>("BandSizeBox");
        if (bandBox != null)
        {
            if (bandBox.ItemCount > 0 && bandBox.SelectedIndex < 0) bandBox.SelectedIndex = 1; // default 128
            bandBox.SelectionChanged += (_, __) => UpdateTimeLabel();
        }
        var tSlider = this.FindControl<Slider>("TimeSlider");
        if (tSlider != null)
        {
            tSlider.PropertyChanged += (s, e) =>
            {
                if (e.Property == RangeBase.ValueProperty)
                {
                    _timeCenter = (int)Math.Round(tSlider.Value);
                    UpdateTimeLabel();
                }
            };
        }
        var blAll = this.FindControl<Button>("BaselineEnableAllBtn"); if (blAll != null) blAll.Click += BaselineEnableAllBtn_Click;
        var blNone = this.FindControl<Button>("BaselineDisableAllBtn"); if (blNone != null) blNone.Click += BaselineDisableAllBtn_Click;
        var blSlice = this.FindControl<Button>("BaselineTimeSliceBtn"); if (blSlice != null) blSlice.Click += BaselineTimeSliceBtn_Click;
        var recompile = this.FindControl<Button>("RecompileMapBtn"); if (recompile != null) recompile.Click += RecompileMapBtn_Click;
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
        // Force fresh reload when switching or re-selecting a map
        _rowsByMap.Remove(map);
        _areasByMap.Remove(map);
        _wmoByMap.Remove(map);
        _baseM2.Clear(); _baseWmo.Clear(); _customM2.Clear(); _customWmo.Clear();
        EnsureMapLoaded(map);
        PopulateTileCombo(map);
        RefreshAreaUi(map);
        RenderTileGrid();
        RenderTileLayers();
        RenderMinimap();
        RenderTileCustom();
    }

    private void LoadAreasForMap(string map)
    {
        try
        {
            var cacheBox = this.FindControl<TextBox>("CacheBox");
            var root = cacheBox?.Text?.Trim(); if (string.IsNullOrWhiteSpace(root)) root = _cacheRoot;
            var areasCsv = Path.Combine(root!, map, "areas.csv");
            if (!File.Exists(areasCsv)) { _areasByMap.Remove(map); return; }

            var data = new AreaData();
            foreach (var line in File.ReadLines(areasCsv).Skip(1))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var p = line.Split(','); if (p.Length < 6) continue;
                if (!int.TryParse(p[0], out var tx)) continue;
                if (!int.TryParse(p[1], out var ty)) continue;
                if (!int.TryParse(p[2], out var aid)) continue;
                var aname = p[3];
                int.TryParse(p[4], out var pid);
                var pname = p[5];
                var key = TileKey(tx, ty);
                data.AreaByTile[key] = aid;
                if (!data.TilesByArea.TryGetValue(aid, out var setA)) data.TilesByArea[aid] = setA = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                setA.Add(key);
                if (aid > 0 && !string.IsNullOrWhiteSpace(aname)) data.AreaName[aid] = aname;
                if (pid > 0)
                {
                    if (!data.ChildrenByParent.TryGetValue(pid, out var kids)) data.ChildrenByParent[pid] = kids = new HashSet<int>();
                    kids.Add(aid);
                    if (!data.TilesByParent.TryGetValue(pid, out var setP)) data.TilesByParent[pid] = setP = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    setP.Add(key);
                    if (!string.IsNullOrWhiteSpace(pname)) data.ParentName[pid] = pname;
                }
            }
            // Build labels
            foreach (var kv in data.ChildrenByParent)
            {
                var pid = kv.Key;
                var plabel = (data.ParentName.GetValueOrDefault(pid) ?? $"Parent {pid}") + $" ({pid})";
                data.ParentLabel[pid] = plabel;
                var subDict = new SortedDictionary<int, string>();
                foreach (var cid in kv.Value)
                {
                    var clabel = (data.AreaName.GetValueOrDefault(cid) ?? $"Area {cid}") + $" ({cid})";
                    subDict[cid] = clabel;
                }
                data.SubzoneLabelByParent[pid] = subDict;
            }

            _areasByMap[map] = data;
        }
        catch { _areasByMap.Remove(map); }
    }

    private void RefreshAreaUi(string map)
    {
        var parentCombo = this.FindControl<ComboBox>("ParentAreaCombo");
        var subsList = this.FindControl<ListBox>("SubzonesList");
        var addBox = this.FindControl<CheckBox>("AreaAddModeBox");
        var selParentBtn = this.FindControl<Button>("SelectParentBtn");
        var selSubsBtn = this.FindControl<Button>("SelectSubzonesBtn");
        var clearBtn = this.FindControl<Button>("ClearAreaSelBtn");
        if (!_areasByMap.TryGetValue(map, out var data))
        {
            if (parentCombo != null) { parentCombo.ItemsSource = Array.Empty<string>(); parentCombo.IsEnabled = false; }
            if (subsList != null) { subsList.ItemsSource = Array.Empty<string>(); subsList.IsEnabled = false; }
            if (addBox != null) addBox.IsEnabled = false;
            if (selParentBtn != null) selParentBtn.IsEnabled = false;
            if (selSubsBtn != null) selSubsBtn.IsEnabled = false;
            if (clearBtn != null) clearBtn.IsEnabled = false;
            PopulateAreasLegend(map);
            UpdateTimeRangeFromSelection();
            return;
        }
        if (parentCombo != null)
        {
            parentCombo.ItemsSource = data.ParentLabel.Values.ToList();
            parentCombo.IsEnabled = data.ParentLabel.Count > 0;
            if (parentCombo.ItemCount > 0 && parentCombo.SelectedIndex < 0) parentCombo.SelectedIndex = 0;
        }
        if (subsList != null) { subsList.ItemsSource = Array.Empty<string>(); subsList.IsEnabled = true; }
        if (addBox != null) addBox.IsEnabled = true;
        if (selParentBtn != null) selParentBtn.IsEnabled = true;
        if (selSubsBtn != null) selSubsBtn.IsEnabled = true;
        if (clearBtn != null) clearBtn.IsEnabled = true;
        OnParentAreaChanged();
    }

    private void OnParentAreaChanged()
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        if (!_areasByMap.TryGetValue(map, out var data)) return;
        var parentCombo = this.FindControl<ComboBox>("ParentAreaCombo");
        var subsList = this.FindControl<ListBox>("SubzonesList");
        if (parentCombo == null || subsList == null) return;
        var sel = parentCombo.SelectedItem as string;
        if (string.IsNullOrWhiteSpace(sel)) { subsList.ItemsSource = Array.Empty<string>(); return; }
        // Extract parent id from label suffix " (id)"
        int pid = -1; var idx = sel.LastIndexOf('('); var idx2 = sel.LastIndexOf(')');
        if (idx >= 0 && idx2 > idx) int.TryParse(sel.Substring(idx + 1, idx2 - idx - 1), out pid);
        if (pid <= 0) { subsList.ItemsSource = Array.Empty<string>(); return; }
        var list = data.SubzoneLabelByParent.GetValueOrDefault(pid)?.Values?.ToList() ?? new List<string>();
        subsList.ItemsSource = list;
    }

    private void SelectParentBtn_Click(object? sender, RoutedEventArgs e)
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        if (!_areasByMap.TryGetValue(map, out var data)) return;
        var parentCombo = this.FindControl<ComboBox>("ParentAreaCombo");
        var addMode = this.FindControl<CheckBox>("AreaAddModeBox");
        if (parentCombo == null) return;
        var sel = parentCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(sel)) return;
        int pid = -1; var idx = sel.LastIndexOf('('); var idx2 = sel.LastIndexOf(')');
        if (idx >= 0 && idx2 > idx) int.TryParse(sel.Substring(idx + 1, idx2 - idx - 1), out pid);
        if (pid <= 0) return;
        var tiles = data.TilesByParent.GetValueOrDefault(pid) ?? new HashSet<string>();
        if (addMode?.IsChecked != true) _selectedTiles.Clear();
        foreach (var k in tiles) _selectedTiles.Add(k);
        if (tiles.Count > 0) _focusedTileKey = tiles.First();
        RenderTileGrid(); RenderTileLayers(); RenderMinimap();
    }

    private void SelectSubzonesBtn_Click(object? sender, RoutedEventArgs e)
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        if (!_areasByMap.TryGetValue(map, out var data)) return;
        var parentCombo = this.FindControl<ComboBox>("ParentAreaCombo");
        var subsList = this.FindControl<ListBox>("SubzonesList");
        var addMode = this.FindControl<CheckBox>("AreaAddModeBox");
        if (parentCombo == null || subsList == null) return;
        var sel = parentCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(sel)) return;
        int pid = -1; var idx = sel.LastIndexOf('('); var idx2 = sel.LastIndexOf(')');
        if (idx >= 0 && idx2 > idx) int.TryParse(sel.Substring(idx + 1, idx2 - idx - 1), out pid);
        if (pid <= 0) return;
        var chosen = subsList.SelectedItems?.Cast<string>().ToList() ?? new List<string>();
        var wantAreas = new HashSet<int>();
        foreach (var label in chosen)
        {
            int id = -1; var i1 = label.LastIndexOf('('); var i2 = label.LastIndexOf(')');
            if (i1 >= 0 && i2 > i1) int.TryParse(label.Substring(i1 + 1, i2 - i1 - 1), out id);
            if (id > 0) wantAreas.Add(id);
        }
        var tiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var aid in wantAreas)
        {
            foreach (var k in data.TilesByArea.GetValueOrDefault(aid) ?? new HashSet<string>()) tiles.Add(k);
        }
        if (addMode?.IsChecked != true) _selectedTiles.Clear();
        foreach (var k in tiles) _selectedTiles.Add(k);
        if (tiles.Count > 0) _focusedTileKey = tiles.First();
        RenderTileGrid(); RenderTileLayers(); RenderMinimap();
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

        // Fallback: if no tile rows, try <map>_tile_layers.csv and normalize
        if (rows.Count == 0)
        {
            var mapCsv = Path.Combine(dir, map + "_tile_layers.csv");
            if (File.Exists(mapCsv))
            {
                var fb = new List<TileLayerRow>();
                foreach (var line in File.ReadLines(mapCsv).Skip(1))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = line.Split(',');
                    if (parts.Length < 7) continue;
                    int off = parts.Length - 7;
                    int.TryParse(parts[off + 0], out var tx);
                    int.TryParse(parts[off + 1], out var ty);
                    var type = parts[off + 2].Trim();
                    int.TryParse(parts[off + 3], out var layer);
                    int.TryParse(parts[off + 4], out var min);
                    int.TryParse(parts[off + 5], out var max);
                    int.TryParse(parts[off + 6], out var count);
                    fb.Add(new TileLayerRow { TileX = tx, TileY = ty, Type = type, Layer = layer, Min = min, Max = max, Count = count });
                }
                if (fb.Count > 0)
                {
                    rows = fb;
                    try { File.Copy(mapCsv, csv, overwrite: true); } catch { }
                    AppendBuildLog($"[layers] Using fallback {Path.GetFileName(mapCsv)} ({fb.Count} rows)");
                }
            }
        }

        AppendBuildLog($"[layers] Loaded {rows.Count} tile rows for {map}");
        _rowsByMap[map] = rows;
        // Reset per-tile state caches
        _baseM2.Clear(); _baseWmo.Clear(); _customM2.Clear(); _customWmo.Clear();

        // If no tile rows, try to resolve global WMO from the map's WDT
        if (rows.Count == 0)
        {
            var datasetRoot = TryGetDatasetRootFromCache(root!);
            var wmo = datasetRoot != null ? TryExtractGlobalWmo(datasetRoot!, map) : null;
            _wmoByMap[map] = wmo;
            // Emit debug if available
            if (!string.IsNullOrWhiteSpace(_lastWmoDebug))
            {
                foreach (var line in _lastWmoDebug.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries))
                    AppendBuildLog("[WDT] " + line);
            }
            else
            {
                if (datasetRoot == null)
                {
                    AppendBuildLog($"[WDT] Dataset root unknown for cache '{root}'. Cannot verify {map}.wdt; layers are empty");
                }
                else if (string.IsNullOrWhiteSpace(wmo))
                {
                    AppendBuildLog($"[WDT] {map}: No WDT or no MWMO names found; treating as empty");
                }
            }
        }

        // Load area group data if available
        LoadAreasForMap(map);
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
        RenderTileGrid(); PopulateAreasLegend(map); UpdateTimeRangeFromSelection();
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

        var rows = _rowsByMap.GetValueOrDefault(map) ?? new List<TileLayerRow>();
        if (rows.Count == 0)
        {
            if (header != null) header.Text = "WMO Map";
            var wmo = _wmoByMap.GetValueOrDefault(map) ?? "(no MWMO found)";
            var tb = new TextBlock { Text = $"Global WMO: {System.IO.Path.GetFileName(wmo) ?? wmo}", Margin = new Thickness(0,2,0,2) };
            host.Children.Add(tb);
            return;
        }

        if (_selectedTiles.Count > 1)
        {
            if (header != null) header.Text = $"Selection Layers ({_selectedTiles.Count})";
            var sel = new HashSet<string>(_selectedTiles, StringComparer.OrdinalIgnoreCase);
            // Helper rounding for bands
            static int RoundDown(int v, int step) => v - (v % step);
            static int RoundUp(int v, int step) { var m = v % step; return m == 0 ? v : v + (step - m); }
            int Band = 128; // configurable via BandSizeBox
            try
            {
                var bandBox2 = this.FindControl<ComboBox>("BandSizeBox");
                var selItem = bandBox2?.SelectedItem as ComboBoxItem;
                if (selItem != null && int.TryParse(selItem.Content?.ToString(), out var parsed) && parsed > 0) Band = parsed;
            }
            catch { Band = 128; }

            // Build per-tile band summaries, then aggregate across selection
            var tiles = sel.Select(k => k).ToList();
            var bands = new Dictionary<(string Type, int BMin, int BMax), (double SumCount, int Samples, int TilesHave, int TilesEnabled)>(
                StringComparer.OrdinalIgnoreCase as IEqualityComparer<(string,int,int)> ?? EqualityComparer<(string,int,int)>.Default);

            foreach (var keyStr in tiles)
            {
                var partsSel = (keyStr ?? string.Empty).Split(',');
                if (partsSel.Length != 2) continue;
                if (!int.TryParse(partsSel[0], out var sx)) continue; if (!int.TryParse(partsSel[1], out var sy)) continue;
                EnsureTileBaseState(map, sx, sy);
                var k = TileKey(sx, sy);
                var perTile = new Dictionary<(string Type, int BMin, int BMax), (double Sum, int Cnt, bool AnyEnabled)>(
                    StringComparer.OrdinalIgnoreCase as IEqualityComparer<(string,int,int)> ?? EqualityComparer<(string,int,int)>.Default);

                void Accum(List<TileEntry> list, string type)
                {
                    foreach (var e in list)
                    {
                        var bmin = RoundDown(e.Min, Band);
                        var bmax = RoundUp(e.Max, Band);
                        var kk = (type, bmin, bmax);
                        if (!perTile.TryGetValue(kk, out var v)) v = (0, 0, false);
                        v.Sum += Math.Max(0, e.Count);
                        v.Cnt += 1;
                        v.AnyEnabled = v.AnyEnabled || e.Enabled;
                        perTile[kk] = v;
                    }
                }

                Accum(_baseM2.GetValueOrDefault(k) ?? new List<TileEntry>(), "M2");
                Accum(_baseWmo.GetValueOrDefault(k) ?? new List<TileEntry>(), "WMO");

                foreach (var kv in perTile)
                {
                    if (!bands.TryGetValue(kv.Key, out var g)) g = (0, 0, 0, 0);
                    g.SumCount += kv.Value.Sum;
                    g.Samples += kv.Value.Cnt;
                    g.TilesHave += 1;
                    if (kv.Value.AnyEnabled) g.TilesEnabled += 1;
                    bands[kv.Key] = g;
                }
            }

            foreach (var kv in bands.OrderBy(k => k.Key.Type).ThenBy(k => k.Key.BMin).ThenBy(k => k.Key.BMax))
            {
                var type = kv.Key.Type; var bmin = kv.Key.BMin; var bmax = kv.Key.BMax;
                var avg = kv.Value.Samples > 0 ? kv.Value.SumCount / kv.Value.Samples : 0.0;
                var allTiles = tiles.Count;
                bool initiallyChecked = (kv.Value.TilesEnabled == allTiles); // strict: present and enabled in all tiles
                var cb = new CheckBox { IsChecked = initiallyChecked, Content = $"[{type}] {bmin}-{bmax} (~{avg:0.0})" };
                cb.Margin = new Thickness(0, 2, 0, 2);
                cb.Checked += (_, __) => ApplyBand(true, type, bmin, bmax);
                cb.Unchecked += (_, __) => ApplyBand(false, type, bmin, bmax);
                host.Children.Add(cb);
            }

            void ApplyBand(bool on, string type, int bmin, int bmax)
            {
                foreach (var keyStr in tiles)
                {
                    var partsSel = (keyStr ?? string.Empty).Split(',');
                    if (partsSel.Length != 2) continue;
                    if (!int.TryParse(partsSel[0], out var sx)) continue; if (!int.TryParse(partsSel[1], out var sy)) continue;
                    EnsureTileBaseState(map, sx, sy);
                    var k = TileKey(sx, sy);
                    var list = string.Equals(type, "M2", StringComparison.OrdinalIgnoreCase) ? _baseM2[k] : _baseWmo[k];
                    bool found = false;
                    foreach (var e in list)
                    {
                        // overlap with band
                        if (e.Min <= bmax && e.Max >= bmin)
                        {
                            e.Enabled = on;
                            found = true;
                        }
                    }
                    if (!found)
                    {
                        var cust = string.Equals(type, "M2", StringComparison.OrdinalIgnoreCase) ? _customM2[k] : _customWmo[k];
                        var ex = cust.FirstOrDefault(c => c.Layer == -1 && c.Min == bmin && c.Max == bmax);
                        if (ex != null) ex.Enabled = on;
                        else cust.Add(new TileEntry { Type = type, Layer = -1, Min = bmin, Max = bmax, Count = 0, Enabled = on });
                    }
                }
                RenderTileLayers(); RenderTileCustom();
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
        foreach (var entry in _customM2[key].Concat(_customWmo[key]))
        {
            var panel = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
            var cb = new CheckBox { IsChecked = entry.Enabled, Content = $"[Custom {entry.Type}] {entry.Min}-{entry.Max}" };
            var del = new Button { Content = "✕", Width = 24 };
            int localIdx = idx++;
            del.Click += (_, __) => { if (entry.Type == "M2") _customM2[key].Remove(entry); else _customWmo[key].Remove(entry); RenderTileCustom(); };
            cb.Checked += (_, __) => entry.Enabled = true; cb.Unchecked += (_, __) => entry.Enabled = false;
            panel.Children.Add(cb); panel.Children.Add(del);
            host.Children.Add(panel);
        }
    }

    private void AddRangeBtn_Click(object? sender, RoutedEventArgs e)
    {
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        // Build target keys: all selected tiles if present, otherwise the focused one
        var targets = _selectedTiles.Count > 0 ? _selectedTiles.ToArray() : new[] { val };

        var minBox = this.FindControl<TextBox>("AddRangeMinBox");
        var maxBox = this.FindControl<TextBox>("AddRangeMaxBox");
        var typeBox = this.FindControl<ComboBox>("AddRangeTypeBox");
        if (minBox == null || maxBox == null || typeBox == null) return;
        if (!int.TryParse(minBox.Text?.Trim(), out var min)) return;
        if (!int.TryParse(maxBox.Text?.Trim(), out var max)) return;
        var type = (typeBox.SelectedItem as ComboBoxItem)?.Content?.ToString() ?? "M2";
        var entry = new TileEntry { Type = type, Layer = -1, Min = min, Max = max, Count = 0, Enabled = true };
        foreach (var key in targets)
        {
            if (string.IsNullOrWhiteSpace(key)) continue;
            var parts = key.Split(','); if (parts.Length != 2) continue;
            if (!int.TryParse(parts[0], out var tx)) continue; if (!int.TryParse(parts[1], out var ty)) continue;
            EnsureTileBaseState(map, tx, ty);
            if (string.Equals(type, "M2", StringComparison.OrdinalIgnoreCase))
                _customM2[key].Add(new TileEntry { Type = entry.Type, Layer = entry.Layer, Min = entry.Min, Max = entry.Max, Count = 0, Enabled = true });
            else
                _customWmo[key].Add(new TileEntry { Type = entry.Type, Layer = entry.Layer, Min = entry.Min, Max = entry.Max, Count = 0, Enabled = true });
        }
        RenderTileCustom();
    }

    private void ClearTileCustomBtn_Click(object? sender, RoutedEventArgs e)
    {
        var tileCombo = this.FindControl<ComboBox>("TileSelectBox"); if (tileCombo == null) return;
        var val = tileCombo.SelectedItem as string; if (string.IsNullOrWhiteSpace(val)) return;
        var targets = _selectedTiles.Count > 0 ? _selectedTiles.ToArray() : new[] { val };
        foreach (var key in targets)
        {
            _customM2[key] = new List<TileEntry>();
            _customWmo[key] = new List<TileEntry>();
        }
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

                // Areas overlay: blend parent area color if available (disabled via flag)
                if (_areasUiEnabled && _areasByMap.TryGetValue(map, out var aData))
                {
                    var tkey = TileKey(key.x, key.y);
                    // Resolve parent id for this tile if any
                    int parentId = -1;
                    foreach (var kv in aData.TilesByParent)
                    {
                        if (kv.Value.Contains(tkey)) { parentId = kv.Key; break; }
                    }
                    if (parentId > 0)
                    {
                        static Avalonia.Media.Color ColorFromParent(int pid)
                        {
                            // deterministic pastel palette from id bits
                            byte r = (byte)(128 + ((pid * 73) & 0x7F));
                            byte g = (byte)(128 + ((pid * 151) & 0x7F));
                            byte b = (byte)(128 + ((pid * 197) & 0x7F));
                            return new Avalonia.Media.Color(0xFF, r, g, b);
                        }
                        static Avalonia.Media.Color Blend(Avalonia.Media.Color baseC, Avalonia.Media.Color over, double alpha)
                        {
                            if (alpha < 0) alpha = 0; if (alpha > 1) alpha = 1;
                            byte r = (byte)(baseC.R * (1 - alpha) + over.R * alpha);
                            byte g = (byte)(baseC.G * (1 - alpha) + over.G * alpha);
                            byte b = (byte)(baseC.B * (1 - alpha) + over.B * alpha);
                            return new Avalonia.Media.Color(0xFF, r, g, b);
                        }
                        var over = ColorFromParent(parentId);
                        cellColor = Blend(cellColor, over, 0.35);
                    }
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
        // Initial highlight (only if nothing selected yet)
        var tileComboInit = this.FindControl<ComboBox>("TileSelectBox");
        var valInit = tileComboInit?.SelectedItem as string;
        if (_selectedTiles.Count == 0 && !string.IsNullOrWhiteSpace(valInit)) { _focusedTileKey = valInit!; _selectedTiles.Clear(); _selectedTiles.Add(valInit!); }
    }

    private void PopulateAreasLegend(string map)
    {
        if (!_areasUiEnabled) return;
        var host = this.FindControl<StackPanel>("AreasLegend"); if (host == null) return;
        host.Children.Clear();
        if (!_areasByMap.TryGetValue(map, out var data)) return;
        // Top 8 parents by tile count
        var items = data.TilesByParent.Select(kv => new { Id = kv.Key, Count = kv.Value.Count, Name = data.ParentName.GetValueOrDefault(kv.Key) ?? ("Parent " + kv.Key) })
            .OrderByDescending(x => x.Count).Take(8).ToList();
        foreach (var it in items)
        {
            var panel = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
            var swatch = new Border { Width = 14, Height = 14, BorderBrush = new Avalonia.Media.SolidColorBrush(Avalonia.Media.Colors.Black), BorderThickness = new Thickness(1) };
            // same color function as grid overlay
            byte r = (byte)(128 + ((it.Id * 73) & 0x7F));
            byte g = (byte)(128 + ((it.Id * 151) & 0x7F));
            byte b = (byte)(128 + ((it.Id * 197) & 0x7F));
            swatch.Background = new Avalonia.Media.SolidColorBrush(new Avalonia.Media.Color(0xFF, r, g, b));
            var btn = new Button { Content = $"{it.Name} ({it.Id}) · {it.Count}", Margin = new Thickness(4,0,0,0) };
            btn.Click += (_, __) =>
            {
                var add = this.FindControl<CheckBox>("AreaAddModeBox");
                if (add?.IsChecked != true) _selectedTiles.Clear();
                foreach (var k in data.TilesByParent.GetValueOrDefault(it.Id) ?? new HashSet<string>()) _selectedTiles.Add(k);
                if ((_selectedTiles?.Count ?? 0) > 0) _focusedTileKey = _selectedTiles.First();
                RenderTileGrid(); RenderTileLayers(); RenderMinimap();
            };
            panel.Children.Add(swatch); panel.Children.Add(btn);
            host.Children.Add(panel);
        }
    }

    private void HighlightTile(string val)
    {
        _focusedTileKey = val;
        _selectedTiles.Clear();
        _selectedTiles.Add(val);
        RenderTileGrid(); UpdateTimeRangeFromSelection();
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

        RenderTileGrid(); RenderTileLayers(); RenderMinimap(); RenderTileCustom(); UpdateTimeRangeFromSelection();
    }

    private void TileGrid_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        var panel = sender as Panel; if (panel == null) return;
        var map = _currentMap; if (string.IsNullOrWhiteSpace(map)) return;
        var cell = TileFromPointer(panel, e);
        // Alt-click = select entire parent area of the clicked tile (if available)
        if (_areasUiEnabled && (e.KeyModifiers & KeyModifiers.Alt) != 0 && _areasByMap.TryGetValue(map, out var data))
        {
            string tkey = TileKey(cell.x, cell.y);
            int parentId = -1;
            foreach (var kv in data.TilesByParent) { if (kv.Value.Contains(tkey)) { parentId = kv.Key; break; } }
            if (parentId > 0)
            {
                var add = this.FindControl<CheckBox>("AreaAddModeBox");
                if (add?.IsChecked != true) _selectedTiles.Clear();
                foreach (var k in data.TilesByParent.GetValueOrDefault(parentId) ?? new HashSet<string>()) _selectedTiles.Add(k);
                _focusedTileKey = tkey;
                RenderTileGrid(); RenderTileLayers(); RenderMinimap();
                return;
            }
        }

        _isDragging = true;
        _dragStartCell = cell;
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
