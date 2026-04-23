using System.Numerics;
using System.Runtime.InteropServices;
using ImGuiNET;
using WowViewer.Core.IO.Files;

namespace WowViewer.App;

internal sealed class MdxFileBrowserState
{
    private IArchiveCatalog? _catalog;
    private List<string> _allFiles = [];
    private List<string> _filteredFiles = [];
    private string _searchFilter = string.Empty;
    private string _extensionFilter = ".mdx";
    private int _selectedIndex = -1;
    private bool _catalogLoaded;
    private string _lastError = string.Empty;
    private string _currentArchiveRoot = string.Empty;
    private string _currentLooseOverlayRoot = string.Empty;

    private static readonly (string Value, string Label)[] ExtensionFilters =
    [
        (".mdx", ".mdx"),
        (".m2", ".m2"),
        (".wmo", ".wmo"),
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

            if (!string.IsNullOrWhiteSpace(_currentArchiveRoot))
            {
                var factory = new MpqArchiveCatalogFactory();
                _catalog = factory.Create();
                ArchiveCatalogBootstrapper.Bootstrap(_catalog, [_currentArchiveRoot], WowViewerArchiveBootstrap.CreateBootstrapOptions());

                foreach (string file in _catalog.GetAllKnownFiles())
                    mergedFiles.Add(file.Replace('\\', '/'));
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

    public void SetExtensionFilter(string ext)
    {
        if (_extensionFilter != ext)
        {
            _extensionFilter = ext;
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
            .Where(f => f.EndsWith(_extensionFilter, StringComparison.OrdinalIgnoreCase))
            .Where(f => string.IsNullOrWhiteSpace(_searchFilter) ||
                        Path.GetFileName(f).Contains(_searchFilter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToList();
        _selectedIndex = -1;
    }

    public void Draw(string archiveRoot, Action onFileSelected)
    {
        string normalizedArchiveRoot = archiveRoot?.Trim() ?? string.Empty;
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
            ImGui.TextWrapped("Load an archive catalog to browse files.");
            return;
        }

        ImGui.Separator();

        // Extension filter dropdown
        if (ImGui.BeginCombo("Type Filter", _extensionFilter))
        {
            foreach (var (value, label) in ExtensionFilters)
            {
                if (ImGui.Selectable(label, _extensionFilter == value))
                    SetExtensionFilter(value);
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
    public static bool DrawMdxFileBrowser(
        string label,
        ref bool browserOpen,
        string archiveRoot,
        string? looseOverlayRoot,
        MdxFileBrowserState state,
        Action<string> onFileSelected)
    {
        bool result = false;
        ImGui.SetNextWindowSize(new Vector2(500, 400), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin(label, ref browserOpen))
        {
            ImGui.End();
            return result;
        }

        ImGui.Text($"Archive Root: {archiveRoot}");
        if (!string.IsNullOrWhiteSpace(looseOverlayRoot))
            ImGui.Text($"Loose Overlay: {Path.GetFullPath(looseOverlayRoot)}");
        ImGui.Separator();

        state.SetLooseOverlayRoot(looseOverlayRoot);

        state.Draw(archiveRoot, () =>
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
