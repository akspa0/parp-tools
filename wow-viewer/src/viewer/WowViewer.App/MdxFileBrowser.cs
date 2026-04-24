using System.Numerics;
using ImGuiNET;
using WowViewer.Core.IO.Files;

namespace WowViewer.App;

internal enum AssetFileBrowserFilter
{
    SupportedAssets = 0,
    M2 = 1,
    Mdx = 2,
    Wmo = 3,
}

internal sealed class AssetFileBrowserState
{
    private IArchiveCatalog? _catalog;
    private List<string> _allFiles = [];
    private List<string> _filteredFiles = [];
    private string _searchFilter = string.Empty;
    private AssetFileBrowserFilter _filter = AssetFileBrowserFilter.Mdx;
    private int _selectedIndex = -1;
    private bool _catalogLoaded;
    private string _lastError = string.Empty;
    private string _currentArchiveRoot = string.Empty;
    private string _currentLooseOverlayRoot = string.Empty;

    private static readonly (AssetFileBrowserFilter Value, string Label)[] Filters =
    [
        (AssetFileBrowserFilter.SupportedAssets, "Supported Assets"),
        (AssetFileBrowserFilter.M2, "M2 (.m2)"),
        (AssetFileBrowserFilter.Mdx, "MDX (.mdx)"),
        (AssetFileBrowserFilter.Wmo, "WMO (.wmo, .wmo.mpq)"),
    ];

    public string? SelectedFilePath => _selectedIndex >= 0 && _selectedIndex < _filteredFiles.Count
        ? _filteredFiles[_selectedIndex]
        : null;

    public bool TryLoadCatalog(string archiveRoot, string? looseOverlayRoot)
    {
        try
        {
            ClearCatalog();

            _currentArchiveRoot = archiveRoot?.Trim() ?? string.Empty;
            _currentLooseOverlayRoot = looseOverlayRoot?.Trim() ?? string.Empty;

            HashSet<string> mergedFiles = new(StringComparer.OrdinalIgnoreCase);

            string? cacheKey = WowViewerArchiveBootstrap.ResolveArchiveListfileCacheKey(null, _currentArchiveRoot);
            string? cacheDirectory = WowViewerArchiveBootstrap.ResolveDefaultArchiveListfileCacheDirectory();
            ArchiveListfileCacheManifest? cacheManifest = !string.IsNullOrWhiteSpace(cacheKey) && !string.IsNullOrWhiteSpace(cacheDirectory)
                ? ArchiveListfileCache.TryRead(cacheDirectory, cacheKey)
                : null;

            if (cacheManifest is not null)
            {
                foreach (string file in cacheManifest.AllEntries)
                    mergedFiles.Add(file.Replace('\\', '/'));
            }

            if (!string.IsNullOrWhiteSpace(_currentArchiveRoot))
            {
                bool needLiveCatalog = cacheManifest is null;
                if (needLiveCatalog)
                {
                    var factory = new MpqArchiveCatalogFactory();
                    _catalog = factory.Create();
                    ArchiveCatalogBootstrapper.Bootstrap(
                        _catalog,
                        [_currentArchiveRoot],
                        WowViewerArchiveBootstrap.CreateBootstrapOptions(null, _currentArchiveRoot));

                    foreach (string file in _catalog.GetAllKnownFiles())
                        mergedFiles.Add(file.Replace('\\', '/'));
                }
            }

            foreach (string file in VirtualAssetOverlayResolver.EnumerateLooseVirtualFiles(_currentLooseOverlayRoot))
                mergedFiles.Add(file.Replace('\\', '/'));

            _allFiles = mergedFiles.ToList();
            _catalogLoaded = true;
            RefreshFilteredFiles();
            _lastError = string.Empty;
            return true;
        }
        catch (Exception ex)
        {
            _lastError = $"Failed to load catalog: {ex.Message}";
            _catalogLoaded = false;
            return false;
        }
    }

    public void ClearCatalog()
    {
        _catalog?.Dispose();
        _catalog = null;
        _allFiles.Clear();
        _filteredFiles.Clear();
        _selectedIndex = -1;
        _catalogLoaded = false;
    }

    public void SetFilter(AssetFileBrowserFilter filter)
    {
        if (_filter != filter)
        {
            _filter = filter;
            RefreshFilteredFiles();
        }
    }

    public void SetSearchFilter(string filter)
    {
        if (_searchFilter != filter)
        {
            _searchFilter = filter;
            RefreshFilteredFiles();
        }
    }

    public void SetSelectedIndex(int index)
    {
        _selectedIndex = index;
    }

    private void RefreshFilteredFiles()
    {
        _filteredFiles = _allFiles
            .Where(f => MatchesFilter(f, _filter))
            .Where(f => string.IsNullOrWhiteSpace(_searchFilter) ||
                        f.Contains(_searchFilter, StringComparison.OrdinalIgnoreCase) ||
                        Path.GetFileName(f).Contains(_searchFilter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
        _selectedIndex = -1;
    }

    public void Draw(string clientRoot, Action onFileSelected)
    {
        string normalizedArchiveRoot = clientRoot?.Trim() ?? string.Empty;
        bool sourceChanged = !string.Equals(normalizedArchiveRoot, _currentArchiveRoot, StringComparison.OrdinalIgnoreCase);

        bool catalogLoaded = _catalogLoaded;
        if (ImGui.Checkbox("Catalog Loaded", ref catalogLoaded))
        {
            if (catalogLoaded && !_catalogLoaded)
                TryLoadCatalog(normalizedArchiveRoot, _currentLooseOverlayRoot);
            else if (!catalogLoaded && _catalogLoaded)
                ClearCatalog();
        }

        if (_catalogLoaded && sourceChanged)
            TryLoadCatalog(normalizedArchiveRoot, _currentLooseOverlayRoot);

        if (!_catalogLoaded)
        {
            ImGui.TextWrapped("Load a game client catalog to browse files.");
            return;
        }

        ImGui.Separator();

        if (ImGui.BeginCombo("Type Filter", GetFilterLabel(_filter)))
        {
            foreach (var (value, label) in Filters)
            {
                if (ImGui.Selectable(label, _filter == value))
                    SetFilter(value);
            }
            ImGui.EndCombo();
        }

        // Search filter
        string search = _searchFilter;
        if (ImGui.InputText("Search", ref search, 256))
            SetSearchFilter(search);

        ImGui.Text($"{_filteredFiles.Count} files");

        // File list
        float remainingH = ImGui.GetContentRegionAvail().Y - 60f;
        if (ImGui.BeginChild("FileList", new Vector2(0, remainingH), true))
        {
            float rowHeight = ImGui.GetTextLineHeightWithSpacing();
            GetVisibleListRange(_filteredFiles.Count, rowHeight, out int startIndex, out int endIndex);

            if (startIndex > 0)
                ImGui.Dummy(new Vector2(0, startIndex * rowHeight));

            for (int i = startIndex; i < endIndex; i++)
            {
                var file = _filteredFiles[i];
                var displayName = Path.GetFileName(file);
                bool selected = i == _selectedIndex;

                if (ImGui.Selectable(displayName, selected, ImGuiSelectableFlags.AllowDoubleClick))
                {
                    _selectedIndex = i;
                    if (ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                        onFileSelected();
                }

                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip(file);
            }

            if (endIndex < _filteredFiles.Count)
                ImGui.Dummy(new Vector2(0, (_filteredFiles.Count - endIndex) * rowHeight));

            ImGui.EndChild();
        }

        // Selection info and load button
        if (ImGui.Button("Load Selected", new Vector2(-1, 0)))
            onFileSelected();

        if (!string.IsNullOrEmpty(_lastError))
            ImGui.TextColored(new Vector4(1f, 0f, 0f, 1f), _lastError);
    }

    private static bool MatchesFilter(string path, AssetFileBrowserFilter filter)
    {
        return filter switch
        {
            AssetFileBrowserFilter.SupportedAssets => path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(".wmo.mpq", StringComparison.OrdinalIgnoreCase),
            AssetFileBrowserFilter.M2 => path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase),
            AssetFileBrowserFilter.Mdx => path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase),
            AssetFileBrowserFilter.Wmo => path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(".wmo.mpq", StringComparison.OrdinalIgnoreCase),
            _ => false,
        };
    }

    private static string GetFilterLabel(AssetFileBrowserFilter filter)
    {
        foreach (var (value, label) in Filters)
        {
            if (value == filter)
                return label;
        }

        return Filters[0].Label;
    }

    public void SetLooseOverlayRoot(string? looseOverlayRoot)
    {
        string normalized = looseOverlayRoot?.Trim() ?? string.Empty;
        if (string.Equals(_currentLooseOverlayRoot, normalized, StringComparison.OrdinalIgnoreCase))
            return;

        _currentLooseOverlayRoot = normalized;
        if (_catalogLoaded)
            TryLoadCatalog(_currentArchiveRoot, _currentLooseOverlayRoot);
    }

    private static void GetVisibleListRange(int itemCount, float rowHeight, out int startIndex, out int endIndex)
    {
        float clipY = ImGui.GetScrollY();
        float clipH = ImGui.GetWindowSize().Y;

        int firstVisible = Math.Min(itemCount, Math.Max(0, (int)(clipY / rowHeight)));
        int lastVisible = Math.Min(itemCount, (int)((clipY + clipH) / rowHeight) + 1);

        startIndex = firstVisible;
        endIndex = lastVisible;
    }
}

internal static class FileBrowserEx
{
    public static bool DrawAssetFileBrowser(
        string label,
        ref bool browserOpen,
        string clientRoot,
        string? looseOverlayRoot,
        AssetFileBrowserState state,
        AssetFileBrowserFilter filter,
        Action<string> onFileSelected)
    {
        bool result = false;
        ImGui.SetNextWindowSize(new Vector2(500, 400), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin(label, ref browserOpen))
        {
            ImGui.End();
            return result;
        }

        ImGui.Text($"Client Root: {clientRoot}");
        if (!string.IsNullOrWhiteSpace(looseOverlayRoot))
            ImGui.Text($"Loose Overlay: {Path.GetFullPath(looseOverlayRoot)}");
        ImGui.Separator();

        state.SetFilter(filter);
        state.SetLooseOverlayRoot(looseOverlayRoot);

        state.Draw(clientRoot, () =>
        {
            if (state.SelectedFilePath != null)
            {
                onFileSelected(state.SelectedFilePath);
                result = true;
            }
        });

        ImGui.End();
        return result;
    }
}
