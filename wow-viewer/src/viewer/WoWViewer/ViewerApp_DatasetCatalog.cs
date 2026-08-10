using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Maps;

namespace WoWViewer;

public partial class ViewerApp
{
    private readonly List<DatasetVersionCatalogEntry> _datasetVersions = new();
    private string _datasetCatalogRoot = Path.Combine(OutputDir, "datasets");
    private string _selectedDatasetVersionRoot = string.Empty;
    private string _activeDatasetVersionRoot = string.Empty;
    private bool _wantSelectDatasetCatalogRoot;

    private void DrawDatasetVersionSettingsContent()
    {
        ImGui.TextDisabled("Dataset versions are separate from the game client and the secondary client-map overlay.");
        ImGui.InputText("Catalog root", ref _datasetCatalogRoot, 1024);
        ImGui.SameLine();
        if (ImGui.SmallButton("Browse...##dataset_catalog_root"))
            _wantSelectDatasetCatalogRoot = true;

        if (ImGui.Button("Refresh dataset catalog##dataset_catalog_refresh"))
        {
            RefreshDatasetCatalog();
            SaveViewerSettings();
        }

        if (_datasetVersions.Count == 0)
        {
            ImGui.TextDisabled("No renderable project or recognized Zarr roots found.");
            ImGui.TextDisabled("Control NPZ and experiment folders are intentionally excluded.");
            return;
        }

        int selectedIndex = FindDatasetVersionIndex(_selectedDatasetVersionRoot);
        string preview = selectedIndex >= 0
            ? _datasetVersions[selectedIndex].DisplayName
            : "Choose dataset version...";

        if (ImGui.BeginCombo("Dataset version", preview))
        {
            for (int i = 0; i < _datasetVersions.Count; i++)
            {
                DatasetVersionCatalogEntry entry = _datasetVersions[i];
                bool selected = i == selectedIndex;
                if (ImGui.Selectable($"{entry.DisplayName}##dataset_version_{i}", selected))
                {
                    _selectedDatasetVersionRoot = entry.RootPath;
                    SaveViewerSettings();
                }

                if (selected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        selectedIndex = FindDatasetVersionIndex(_selectedDatasetVersionRoot);
        if (selectedIndex < 0)
            return;

        DatasetVersionCatalogEntry selectedEntry = _datasetVersions[selectedIndex];
        ImGui.TextDisabled($"Source: {selectedEntry.SourceKind}  Map: {selectedEntry.MapName ?? "unknown"}");
        ImGui.TextDisabled($"Tiles: {selectedEntry.TileCount}  Renderable: {(selectedEntry.Renderable ? "yes" : "no")}");
        ImGui.TextWrapped($"Signals: {string.Join(", ", selectedEntry.Signals)}");

        if (!string.IsNullOrWhiteSpace(selectedEntry.Diagnostic))
            ImGui.TextWrapped($"Diagnostic: {selectedEntry.Diagnostic}");

        bool isActive = string.Equals(_activeDatasetVersionRoot, selectedEntry.RootPath, StringComparison.OrdinalIgnoreCase);
        if (isActive)
            ImGui.TextColored(new Vector4(0.4f, 0.9f, 0.4f, 1f), "Active dataset version");

        if (ImGui.Button("Activate selected dataset##dataset_activate", new Vector2(210f, 0f)))
            ActivateDatasetVersion(selectedEntry);
    }

    private void RefreshDatasetCatalog()
    {
        _datasetVersions.Clear();
        if (string.IsNullOrWhiteSpace(_datasetCatalogRoot))
            return;

        try
        {
            _datasetVersions.AddRange(DatasetVersionCatalog.Discover(_datasetCatalogRoot));
            if (FindDatasetVersionIndex(_selectedDatasetVersionRoot) < 0 && _datasetVersions.Count > 0)
                _selectedDatasetVersionRoot = _datasetVersions[0].RootPath;

            _statusMessage = $"Dataset catalog refreshed: {_datasetVersions.Count} recognized root(s).";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Dataset catalog refresh failed: {ex.Message}";
        }
    }

    private void ActivateDatasetVersion(DatasetVersionCatalogEntry entry)
    {
        if (!entry.Renderable)
        {
            _statusMessage = $"Dataset '{entry.DisplayName}' is summary-only: {entry.Diagnostic}";
            return;
        }

        Vector3 savedCameraPosition = _camera.Position;
        float savedYaw = _camera.Yaw;
        float savedPitch = _camera.Pitch;

        try
        {
            // Preflight before disposing the current dataset-backed renderer. This keeps a bad
            // catalog entry from replacing a working session with an empty one.
            _ = new Terrain.VlmProjectLoader(entry.RootPath);
            LoadVlmProject(entry.RootPath);

            if (_vlmTerrainManager == null)
                return;

            _camera.Position = savedCameraPosition;
            _camera.Yaw = savedYaw;
            _camera.Pitch = savedPitch;
            _selectedDatasetVersionRoot = entry.RootPath;
            _activeDatasetVersionRoot = entry.RootPath;
            SaveViewerSettings();
            _statusMessage = $"Activated dataset version: {entry.DisplayName}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"Dataset activation failed: {ex.Message}";
        }
    }

    private int FindDatasetVersionIndex(string root)
    {
        if (string.IsNullOrWhiteSpace(root))
            return -1;

        return _datasetVersions.FindIndex(entry =>
            string.Equals(entry.RootPath, root, StringComparison.OrdinalIgnoreCase));
    }
}
