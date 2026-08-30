using System.Numerics;
using ImGuiNET;
using WowViewer.Core.IO.Maps;
using WoWViewer.DataSources;
using WoWViewer.Terrain;

namespace WoWViewer;

/// <summary>
/// Partial class containing client selection and lightweight viewer dialogs.
/// </summary>
public partial class ViewerApp
{
    private void DrawFolderInputDialog()
    {
        if (!_showFolderInput) return;
        _showFolderInput = false;

        string initial = string.IsNullOrEmpty(_folderInputBuf)
            ? (Directory.Exists(@"H:\CLIENTS") ? @"H:\CLIENTS" : Directory.GetCurrentDirectory())
            : _folderInputBuf;

        ImGuiPathPicker.Instance.Open(
            "Select WoW game folder (containing Data/ with MPQs)",
            pickFolder: true,
            initialPath: initial,
            filterExtension: null,
            selectedPath =>
            {
                if (!string.IsNullOrEmpty(selectedPath) && Directory.Exists(selectedPath))
                {
                    _folderInputBuf = selectedPath;
                    PrepareBuildSelectionDialog(selectedPath);
                }
            });
    }

    private void DrawBuildSelectionDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(560, 220), ImGuiCond.FirstUseEver);
        bool open = _showBuildSelectionDialog;
        if (!ImGui.Begin("Select Client Build", ref open, ImGuiWindowFlags.NoCollapse))
        {
            ImGui.End();
            _showBuildSelectionDialog = open;
            if (!_showBuildSelectionDialog)
                _pendingGameFolderPath = null;
            return;
        }

        _showBuildSelectionDialog = open;
        if (!_showBuildSelectionDialog)
            _pendingGameFolderPath = null;

        ImGui.TextWrapped("Explicit client version selection is required before loading MPQs. Path hints only preselect the most likely build.");
        if (!string.IsNullOrWhiteSpace(_pendingGameFolderPath))
            ImGui.TextWrapped($"Folder: {_pendingGameFolderPath}");
        if (!string.IsNullOrWhiteSpace(_buildSelectionHint))
            ImGui.TextDisabled(_buildSelectionHint);

        ImGui.Separator();

        if (_clientBuildOptions.Count == 0)
        {
            ImGui.TextWrapped("No build profiles are available. Ensure WoWDBDefs/definitions/Map.dbd exists, or rely on the built-in fallback list.");
            if (ImGui.Button("Cancel"))
            {
                _pendingGameFolderPath = null;
                _showBuildSelectionDialog = false;
                _buildSelectionHint = null;
            }

            ImGui.End();
            return;
        }

        _selectedBuildOptionIndex = Math.Clamp(_selectedBuildOptionIndex, 0, _clientBuildOptions.Count - 1);
        string preview = _clientBuildOptions[_selectedBuildOptionIndex].Label;
        ImGui.InputTextWithHint("##build_filter", "Filter by build or family", ref _buildSelectionFilter, 128);

        if (ImGui.BeginCombo("Client version family", preview))
        {
            for (int i = 0; i < _clientBuildOptions.Count; i++)
            {
                if (!string.IsNullOrWhiteSpace(_buildSelectionFilter))
                {
                    string filter = _buildSelectionFilter.Trim();
                    bool matches = _clientBuildOptions[i].Label.Contains(filter, StringComparison.OrdinalIgnoreCase)
                        || _clientBuildOptions[i].BuildVersion.Contains(filter, StringComparison.OrdinalIgnoreCase);
                    if (!matches)
                        continue;
                }

                bool isSelected = i == _selectedBuildOptionIndex;
                if (ImGui.Selectable(_clientBuildOptions[i].Label, isSelected))
                    _selectedBuildOptionIndex = i;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        ImGui.TextDisabled($"Selected build: {_clientBuildOptions[_selectedBuildOptionIndex].BuildVersion}");

        if (ImGui.Button("Load MPQs"))
        {
            if (!string.IsNullOrWhiteSpace(_pendingGameFolderPath) && Directory.Exists(_pendingGameFolderPath))
            {
                string selectedPath = _pendingGameFolderPath;
                string buildVersion = _clientBuildOptions[_selectedBuildOptionIndex].BuildVersion;
                _pendingGameFolderPath = null;
                _showBuildSelectionDialog = false;
                _buildSelectionHint = null;
                LoadMpqDataSource(selectedPath, null, buildVersion);
            }
            else
            {
                _statusMessage = "Game folder is missing or no longer accessible.";
            }
        }

        ImGui.SameLine();
        if (ImGui.Button("Cancel"))
        {
            _pendingGameFolderPath = null;
            _showBuildSelectionDialog = false;
            _buildSelectionHint = null;
        }

        ImGui.End();
    }

    private void PrepareBuildSelectionDialog(string selectedPath)
    {
        _pendingGameFolderPath = selectedPath;
        _buildSelectionFilter = string.Empty;
        RefreshClientBuildOptions();

        if (_clientBuildOptions.Count == 0)
        {
            _selectedBuildOptionIndex = 0;
            _buildSelectionHint = "No build profiles available from Map.dbd.";
            _showBuildSelectionDialog = true;
            return;
        }

        if (BuildVersionCatalog.TryInferBuildIndexFromPath(_clientBuildOptions, selectedPath, out int inferredIndex))
        {
            _selectedBuildOptionIndex = inferredIndex;
            _buildSelectionHint = $"Path hint matched build {_clientBuildOptions[inferredIndex].BuildVersion}. Confirm before loading.";
        }
        else
        {
            _selectedBuildOptionIndex = Math.Clamp(_selectedBuildOptionIndex, 0, _clientBuildOptions.Count - 1);
            _buildSelectionHint = "No clear build token found in the folder path. Select the client build manually.";
        }

        _showBuildSelectionDialog = true;
    }

    private void RefreshClientBuildOptions()
    {
        string? previouslySelected = null;
        if (_clientBuildOptions.Count > 0)
        {
            int currentIndex = Math.Clamp(_selectedBuildOptionIndex, 0, _clientBuildOptions.Count - 1);
            previouslySelected = _clientBuildOptions[currentIndex].BuildVersion;
        }

        _clientBuildOptions.Clear();

        string? dbdDir = ResolveDbdDefinitionsDir();
        if (!string.IsNullOrWhiteSpace(dbdDir))
            _clientBuildOptions.AddRange(BuildVersionCatalog.LoadOptionsFromMapDbd(dbdDir));

        if (_clientBuildOptions.Count == 0)
            _clientBuildOptions.AddRange(FallbackClientBuildOptions);

        _selectedBuildOptionIndex = FindBuildOptionIndex(previouslySelected);
    }

    private static string? ResolveDbdDefinitionsDir()
    {
        string[] dbdSearchPaths =
        {
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "lib", "WoWDBDefs", "definitions"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "definitions"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "WoWDBDefs", "definitions"),
        };

        foreach (var path in dbdSearchPaths)
        {
            var resolved = Path.GetFullPath(path);
            if (Directory.Exists(resolved) && File.Exists(Path.Combine(resolved, "Map.dbd")))
                return resolved;
        }

        return null;
    }

    private int FindBuildOptionIndex(string? buildVersion)
    {
        if (string.IsNullOrWhiteSpace(buildVersion) || _clientBuildOptions.Count == 0)
            return 0;

        for (int i = 0; i < _clientBuildOptions.Count; i++)
        {
            if (string.Equals(_clientBuildOptions[i].BuildVersion, buildVersion, StringComparison.OrdinalIgnoreCase))
                return i;
        }

        return 0;
    }

    private void QueueKnownGoodClientAction(string gamePath, string? buildVersion, bool attachLooseFolder)
    {
        _pendingKnownGoodClientPath = gamePath;
        _pendingKnownGoodClientBuildVersion = buildVersion;
        _pendingKnownGoodClientAttachLooseFolder = attachLooseFolder;
    }

    private void SaveCurrentGameFolderAsKnownGoodBase()
    {
        if (_dataSource is not MpqDataSource mpqDataSource)
        {
            _statusMessage = "Load a base MPQ game folder before saving it as a known-good client path.";
            return;
        }

        AddOrUpdateKnownGoodClientPath(mpqDataSource.GamePath, _dbcBuild);
        SaveViewerSettings();
        _statusMessage = $"Saved known-good client path: {mpqDataSource.GamePath}";
    }

    private void AddOrUpdateKnownGoodClientPath(string gamePath, string? buildVersion)
    {
        string normalizedPath = Path.GetFullPath(gamePath);
        string displayName = BuildKnownGoodClientDisplayName(normalizedPath, buildVersion);

        int existingIndex = _knownGoodClientPaths.FindIndex(entry =>
            string.Equals(entry.Path, normalizedPath, StringComparison.OrdinalIgnoreCase));

        var entry = new KnownGoodClientPath
        {
            Name = displayName,
            Path = normalizedPath,
            BuildVersion = string.IsNullOrWhiteSpace(buildVersion) ? null : buildVersion
        };

        if (existingIndex >= 0)
            _knownGoodClientPaths[existingIndex] = entry;
        else
            _knownGoodClientPaths.Add(entry);

        _knownGoodClientPaths = _knownGoodClientPaths
            .OrderBy(client => client.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private void ForgetKnownGoodClientPath(string gamePath)
    {
        int removed = _knownGoodClientPaths.RemoveAll(entry =>
            string.Equals(entry.Path, gamePath, StringComparison.OrdinalIgnoreCase));

        if (removed > 0)
        {
            SaveViewerSettings();
            _statusMessage = $"Removed known-good client path: {gamePath}";
        }
    }

    private static string BuildKnownGoodClientDisplayName(string gamePath, string? buildVersion)
    {
        string folderName = Path.GetFileName(Path.TrimEndingDirectorySeparator(gamePath));
        if (string.IsNullOrWhiteSpace(folderName))
            folderName = gamePath;

        return string.IsNullOrWhiteSpace(buildVersion)
            ? folderName
            : $"{folderName} [{buildVersion}]";
    }

    private static string BuildKnownGoodClientTooltip(KnownGoodClientPath knownClient)
    {
        return string.IsNullOrWhiteSpace(knownClient.BuildVersion)
            ? knownClient.Path
            : $"{knownClient.Path}\nBuild: {knownClient.BuildVersion}";
    }

    private void DrawListfileInputDialog()
    {
        // No longer needed — listfile is auto-downloaded
        _showListfileInput = false;
    }

    private string _rosettaDatastorePathInput = "output/rosetta-datastore.zarr";
    private RosettaObjectLibrary? _rosettaLoadedLibrary = null;
    private string? _rosettaLoadedDatastorePath = null;
    private int _selectedDatastoreBuildIndex = 0;
    private int _selectedDatastoreMapIndex = 0;
    private int _selectedDatastoreBaseClientIndex = 0;
    private int _selectedDiffCompareBuildIndex = 0;
    private RosettaBuildDiff? _activeBuildDiff = null;

    private static string FormatBuildDisplayName(string rawBuildId)
    {
        if (string.IsNullOrWhiteSpace(rawBuildId))
            return "Unknown Build";

        string cleaned = rawBuildId.Replace('_', '.');
        if (cleaned.StartsWith("0.5.3", StringComparison.OrdinalIgnoreCase))
            return $"Alpha 0.5.3 (Build 3368)";
        if (cleaned.StartsWith("1.0.0", StringComparison.OrdinalIgnoreCase))
            return $"Vanilla 1.0.0 (Build 3980)";
        if (cleaned.StartsWith("1.12.1", StringComparison.OrdinalIgnoreCase))
            return $"Classic 1.12.1 (Build 5875)";
        if (cleaned.StartsWith("2.4.3", StringComparison.OrdinalIgnoreCase))
            return $"TBC 2.4.3 (Build 8606)";
        if (cleaned.StartsWith("3.3.5", StringComparison.OrdinalIgnoreCase))
            return $"Wrath 3.3.5a (Build 12340)";

        return rawBuildId;
    }

    private void DrawRosettaDatastoreDialog()
    {
        ImGui.SetNextWindowSize(new Vector2(740, 620), ImGuiCond.FirstUseEver);
        bool open = _showRosettaDatastoreDialog;
        if (!ImGui.Begin("Load from Rosetta Datastore", ref open, ImGuiWindowFlags.NoCollapse))
        {
            ImGui.End();
            _showRosettaDatastoreDialog = open;
            return;
        }

        _showRosettaDatastoreDialog = open;

        ImGui.TextColored(new Vector4(1f, 0.85f, 0.3f, 1f), "Unified Multi-Version Zarr Datastore Loader");
        ImGui.TextWrapped("Load calibration maps, terrain, and object placements directly from the Zarr v3 datastore over any active client base, with automatic cross-era format shifting (.mdx <-> .m2).");
        ImGui.Separator();

        // Datastore Directory Path
        ImGui.InputText("Datastore Path", ref _rosettaDatastorePathInput, 512);
        ImGui.SameLine();
        if (ImGui.Button("Browse..."))
        {
            ImGuiPathPicker.Instance.Open(
                "Select Rosetta Datastore Directory",
                pickFolder: true,
                initialPath: _rosettaDatastorePathInput,
                filterExtension: null,
                picked =>
                {
                    if (!string.IsNullOrWhiteSpace(picked))
                        _rosettaDatastorePathInput = picked;
                });
        }

        string fullDatastorePath = Path.IsPathRooted(_rosettaDatastorePathInput)
            ? _rosettaDatastorePathInput
            : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, _rosettaDatastorePathInput));

        if (!Directory.Exists(fullDatastorePath))
        {
            string repoOutput = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", _rosettaDatastorePathInput));
            if (Directory.Exists(repoOutput))
                fullDatastorePath = repoOutput;
        }

        if (_rosettaLoadedLibrary == null || !string.Equals(_rosettaLoadedDatastorePath, fullDatastorePath, StringComparison.OrdinalIgnoreCase))
        {
            if (Directory.Exists(fullDatastorePath) && File.Exists(Path.Combine(fullDatastorePath, "global_assets", "catalog.parquet")))
            {
                try
                {
                    _rosettaLoadedLibrary = RosettaObjectLibrary.Open(fullDatastorePath);
                    _rosettaLoadedDatastorePath = fullDatastorePath;
                    _selectedDatastoreBuildIndex = 0;
                    _selectedDatastoreMapIndex = 0;
                    _activeBuildDiff = null;
                }
                catch (Exception ex)
                {
                    ImGui.TextColored(new Vector4(1f, 0.4f, 0.4f, 1f), $"Failed to open datastore: {ex.Message}");
                }
            }
            else
            {
                _rosettaLoadedLibrary = null;
                _rosettaLoadedDatastorePath = null;
            }
        }

        if (_rosettaLoadedLibrary == null)
        {
            ImGui.Spacing();
            ImGui.TextColored(new Vector4(1f, 0.8f, 0.3f, 1f), $"No valid Zarr datastore found at: {fullDatastorePath}");
            ImGui.TextDisabled("Run 'wowviewer-inspect rosetta-generate --emit-zarr' to create a datastore.");
            ImGui.End();
            return;
        }

        ImGui.TextColored(new Vector4(0.4f, 1f, 0.4f, 1f), $"Datastore Active: {_rosettaLoadedLibrary.TotalUniqueAssets:N0} unique deduplicated assets across {_rosettaLoadedLibrary.Builds.Count} registered builds");
        ImGui.Separator();

        var builds = _rosettaLoadedLibrary.Builds;
        if (builds.Count == 0)
        {
            ImGui.TextDisabled("No versioned builds found in datastore.");
            ImGui.End();
            return;
        }

        _selectedDatastoreBuildIndex = Math.Clamp(_selectedDatastoreBuildIndex, 0, builds.Count - 1);
        string currentBuild = builds[_selectedDatastoreBuildIndex];
        var maps = _rosettaLoadedLibrary.GetMaps(currentBuild);
        _selectedDatastoreMapIndex = Math.Clamp(_selectedDatastoreMapIndex, 0, Math.Max(0, maps.Count - 1));
        string currentMap = maps.Count > 0 ? maps[_selectedDatastoreMapIndex] : "None";
        var buildMeta = _rosettaLoadedLibrary.GetBuildMetadata(currentBuild);

        // ── Visual Two-Card Selection Layout ──────────────────────────────
        float halfWidth = (ImGui.GetContentRegionAvail().X - 10f) * 0.5f;

        // Card 1: Data Version & Map Selection
        ImGui.BeginChild("##data_version_card", new Vector2(halfWidth, 190), true);
        ImGui.TextColored(new Vector4(1f, 0.8f, 0.2f, 1f), "1. DATA VERSION (Map Source)");
        ImGui.Separator();

        string dataVersionPreview = $"{FormatBuildDisplayName(currentBuild)}";
        if (ImGui.BeginCombo("Build##data_version", dataVersionPreview))
        {
            for (int i = 0; i < builds.Count; i++)
            {
                bool isSelected = i == _selectedDatastoreBuildIndex;
                string label = $"{FormatBuildDisplayName(builds[i])}##opt_{i}";
                if (ImGui.Selectable(label, isSelected))
                {
                    _selectedDatastoreBuildIndex = i;
                    _selectedDatastoreMapIndex = 0;
                    _activeBuildDiff = null;
                }
                if (isSelected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        if (maps.Count > 0)
        {
            if (ImGui.BeginCombo("Map##data_map", currentMap))
            {
                for (int i = 0; i < maps.Count; i++)
                {
                    bool isSelected = i == _selectedDatastoreMapIndex;
                    if (ImGui.Selectable(maps[i], isSelected))
                        _selectedDatastoreMapIndex = i;
                    if (isSelected) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
        }
        else
        {
            ImGui.TextDisabled("No maps available for this build.");
        }

        if (buildMeta != null)
        {
            ImGui.Spacing();
            ImGui.TextDisabled($"Placements: {buildMeta.TotalPlacements:N0}");
            ImGui.TextDisabled($"Unique Assets: {buildMeta.TotalUniqueAssets:N0}");
            ImGui.TextDisabled($"Tiles: {buildMeta.TilesWritten:N0}");
        }

        ImGui.EndChild();

        ImGui.SameLine();

        // Card 2: Base Game Version & Asset Source
        ImGui.BeginChild("##base_game_card", new Vector2(halfWidth, 190), true);
        ImGui.TextColored(new Vector4(0.3f, 0.85f, 1f, 1f), "2. BASE GAME VERSION (Asset Source)");
        ImGui.Separator();

        var clientOptions = new List<string>();
        string activeClientLabel = _dataSource != null
            ? $"Current Active Client: {_dataSource.Name} [{_dbcBuild ?? "unknown"}]"
            : "Current Active Client: (No MPQs loaded)";
        clientOptions.Add(activeClientLabel);

        foreach (var kg in _knownGoodClientPaths)
            clientOptions.Add($"{kg.Name}");

        _selectedDatastoreBaseClientIndex = Math.Clamp(_selectedDatastoreBaseClientIndex, 0, clientOptions.Count - 1);
        if (ImGui.BeginCombo("Base Client##base_client", clientOptions[_selectedDatastoreBaseClientIndex]))
        {
            for (int i = 0; i < clientOptions.Count; i++)
            {
                bool isSelected = i == _selectedDatastoreBaseClientIndex;
                if (ImGui.Selectable(clientOptions[i], isSelected))
                    _selectedDatastoreBaseClientIndex = i;
                if (isSelected) ImGui.SetItemDefaultFocus();
            }
            ImGui.EndCombo();
        }

        ImGui.Spacing();
        ImGui.TextColored(new Vector4(0.4f, 1f, 0.4f, 1f), "Cross-Era Shifting: ACTIVE");
        ImGui.TextWrapped(".mdx, .mdl, and .m2 extensions shift automatically to match the active base client's model files.");
        ImGui.EndChild();

        ImGui.Spacing();
        ImGui.Separator();

        // ── Cross-Build Diff Analysis Section ─────────────────────────────
        if (ImGui.CollapsingHeader("Cross-Build Diff Analysis", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.Text("Compare selected Data Version against another build in the datastore:");
            _selectedDiffCompareBuildIndex = Math.Clamp(_selectedDiffCompareBuildIndex, 0, builds.Count - 1);
            string comparePreview = FormatBuildDisplayName(builds[_selectedDiffCompareBuildIndex]);

            if (ImGui.BeginCombo("Compare With##diff_target", comparePreview))
            {
                for (int i = 0; i < builds.Count; i++)
                {
                    bool isSelected = i == _selectedDiffCompareBuildIndex;
                    string label = $"{FormatBuildDisplayName(builds[i])}##diff_opt_{i}";
                    if (ImGui.Selectable(label, isSelected))
                    {
                        _selectedDiffCompareBuildIndex = i;
                        _activeBuildDiff = null;
                    }
                    if (isSelected) ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }

            ImGui.SameLine();
            if (ImGui.Button("Run Diff"))
            {
                _activeBuildDiff = _rosettaLoadedLibrary.ComputeBuildDiff(currentBuild, builds[_selectedDiffCompareBuildIndex]);
            }

            if (_activeBuildDiff != null)
            {
                ImGui.Spacing();
                ImGui.TextColored(new Vector4(0.4f, 0.8f, 1f, 1f),
                    $"Diff Summary: +{_activeBuildDiff.AddedCount:N0} Added | -{_activeBuildDiff.RemovedCount:N0} Removed | {_activeBuildDiff.FormatMigratedCount:N0} Format Migrated (.mdx <-> .m2) | {_activeBuildDiff.GeometryModifiedCount:N0} Modified | {_activeBuildDiff.IdenticalCount:N0} Identical");
            }
        }

        ImGui.Spacing();
        ImGui.Separator();
        ImGui.Spacing();

        // ── Primary Action Button ─────────────────────────────────────────
        if (maps.Count > 0)
        {
            if (ImGui.Button($"Load '{currentMap}' Directly from Zarr Datastore (3D)", new Vector2(380, 36)))
            {
                // If a different base client was selected, switch active client
                if (_selectedDatastoreBaseClientIndex > 0 && _selectedDatastoreBaseClientIndex - 1 < _knownGoodClientPaths.Count)
                {
                    var chosenClient = _knownGoodClientPaths[_selectedDatastoreBaseClientIndex - 1];
                    QueueKnownGoodClientAction(chosenClient.Path, chosenClient.BuildVersion, attachLooseFolder: false);
                }

                // Load directly from Zarr datastore!
                LoadRosettaDatastoreTerrain(_rosettaLoadedLibrary, currentBuild, currentMap);
                _showRosettaDatastoreDialog = false;
            }
        }

        ImGui.End();
    }
}
