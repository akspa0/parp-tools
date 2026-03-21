using System.Numerics;
using ImGuiNET;
using MdxViewer.DataSources;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Partial class containing client selection and lightweight viewer dialogs.
/// </summary>
public partial class ViewerApp
{
    private void DrawFolderInputDialog()
    {
        if (!_showFolderInput) return;

        // Use WinForms folder browser for native experience
        _showFolderInput = false;

        string? selectedPath = ShowFolderDialogSTA(
            "Select WoW game folder (containing Data/ with MPQs)",
            initialDir: string.IsNullOrEmpty(_folderInputBuf) ? null : _folderInputBuf,
            showNewFolderButton: false);

        if (!string.IsNullOrEmpty(selectedPath) && Directory.Exists(selectedPath))
        {
            _folderInputBuf = selectedPath;
            PrepareBuildSelectionDialog(selectedPath);
        }
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
}