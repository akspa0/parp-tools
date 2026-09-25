using System.Diagnostics;
using System.Numerics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.IO.Mdx;
using WoWViewer.DataSources;
using WoWViewer.Export;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Catalog;
using WoWViewer.Capture;
using WoWViewer.Population;
using WoWViewer.Terrain;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.OpenGL.Extensions.ImGui;
using Silk.NET.Windowing;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WoWViewer.Terrain.Vlm;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.M2Chunked;
using WowViewer.Core.IO.M2Era1121;
using WowViewer.Core.M2;
using WowViewer.Core.Runtime.M2;
using WowViewer.Core.Runtime.Marketing;
using WowViewer.Core.Runtime.World.Visibility;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Converters;
using WoWViewer.Workbench;
using CoreMdxCollisionSummary = WowViewer.Core.Mdx.MdxCollisionSummary;
using CoreMdxGeometryFile = WowViewer.Core.Mdx.MdxGeometryFile;
using CoreMdxSummary = WowViewer.Core.Mdx.MdxSummary;
using CorePm4DocumentReader = WowViewer.Core.PM4.Services.Pm4ResearchReader;
using Pm4CoordinateService = WowViewer.Core.PM4.Services.Pm4CoordinateService;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

/// <summary>
/// Placement editing: selected-placement edit controls, staged edits, save targets and the save queue that writes placement files.
/// Extracted verbatim from <see cref="ViewerApp"/> (Epic 251 U-01). World and app state it
/// needs comes only through <see cref="IViewerAppHost"/>; the bridge members below keep the
/// names the moved code used inside ViewerApp, so no moved body was edited.
/// </summary>
internal sealed class PlacementEditService
{
    private readonly IViewerAppHost _host;

    internal PlacementEditService(IViewerAppHost host)
    {
        _host = host;
    }

    // Host bridge (same names as the former ViewerApp members).
    private ref IDataSource? _dataSource => ref _host.DataSource;
    private ref string _editorProjectOutputDir => ref _host.EditorProjectOutputDir;
    private ref string _selectedPlacementSaveStatus => ref _host.SelectedPlacementSaveStatus;
    private ref string? _selectedPlacementSaveTargetPath => ref _host.SelectedPlacementSaveTargetPath;
    private ref WorldScene? _worldScene => ref _host.WorldScene;
    private string EnsureEditorProjectOutputDirectory(bool forceNew = false) => _host.EnsureEditorProjectOutputDirectory(forceNew);
    private string GetProjectOutputRootDirectory() => _host.GetProjectOutputRootDirectory();
    private void RefreshSelectedWorldObjectInfo() => _host.RefreshSelectedWorldObjectInfo();


    private readonly record struct PlacementEditKey(Terrain.ObjectType ObjectType, int TileX, int TileY, int EntryIndex, int UniqueId);

    private sealed class StagedPlacementEdit
    {
        public PlacementEditKey Key { get; init; }
        public string SourcePath { get; init; } = string.Empty;
        public Vector3 OriginalPosition { get; set; }
        public Vector3 EditedPosition { get; set; }
        public Vector3? EditedRotation { get; set; }
        public float? EditedScale { get; set; }
        public bool Deleted { get; set; }
    }
    private Terrain.ObjectType _selectedPlacementEditType = Terrain.ObjectType.None;
    private int _selectedPlacementEditUniqueId = -1;
    private int _selectedPlacementEditTileX = -1;
    private int _selectedPlacementEditTileY = -1;
    private int _selectedPlacementEditEntryIndex = -1;
    private Vector3 _selectedPlacementOriginalPosition;
    private Vector3 _selectedPlacementEditedPosition;
    private bool _selectedPlacementDirty;
    private string? _selectedPlacementSourcePath;
    private readonly Dictionary<PlacementEditKey, StagedPlacementEdit> _stagedPlacementEdits = new();
    private readonly Dictionary<string, string> _placementSaveTargetsBySourcePath = new(StringComparer.OrdinalIgnoreCase);

    private bool IsProjectManagedOutputPath(string? outputPath)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
            return false;

        string fullPath = Path.GetFullPath(outputPath);
        string rootPath = GetProjectOutputRootDirectory();
        return fullPath.StartsWith(rootPath, StringComparison.OrdinalIgnoreCase);
    }

    private string BuildProjectManagedPlacementOutputPath(string sourcePath)
    {
        string normalizedSourcePath = sourcePath.Replace('/', '\\').TrimStart('\\');
        return Path.Combine(EnsureEditorProjectOutputDirectory(), "lk-split", normalizedSourcePath);
    }

    internal void RefreshProjectManagedPlacementTargets()
    {
        foreach (string sourcePath in _stagedPlacementEdits.Values
            .Select(edit => edit.SourcePath)
            .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!_placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? targetPath)
                || string.IsNullOrWhiteSpace(targetPath)
                || IsProjectManagedOutputPath(targetPath))
            {
                _placementSaveTargetsBySourcePath[sourcePath] = BuildProjectManagedPlacementOutputPath(sourcePath);
            }
        }

        if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath)
            && (string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                || IsProjectManagedOutputPath(_selectedPlacementSaveTargetPath)))
        {
            _selectedPlacementSaveTargetPath = BuildProjectManagedPlacementOutputPath(_selectedPlacementSourcePath);
        }
    }

    internal void DrawSelectedPlacementEditControls()
    {
        SyncSelectedPlacementEditState();

        ImGui.Separator();
        ImGui.Text("Selected Placement Move");
        ImGui.TextDisabled("Translation-only save for existing ADT MDDF/MODF placements, grouped by source ADT when multiple moves are staged.");

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            DrawPlacementSaveQueueActions(includeCurrentSourceSave: false);
            ImGui.TextDisabled(_selectedPlacementSaveStatus);
            return;
        }

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        bool editable = selected.HasTileCoordinate && selected.PlacementEntryIndex >= 0
            && _worldScene.SelectedObjectType is Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo;

        if (!editable)
        {
            DrawPlacementSaveQueueActions(includeCurrentSourceSave: false);
            ImGui.TextDisabled(_selectedPlacementSaveStatus);
            return;
        }

        ImGui.TextDisabled($"Tile ({selected.TileX}, {selected.TileY})  Entry {selected.PlacementEntryIndex}  UniqueId {selected.UniqueId}");

        Vector3 editedPosition = _selectedPlacementEditedPosition;
        if (ImGui.InputFloat3("Placement Position", ref editedPosition, "%.3f"))
        {
            if (!EnsureSelectedPlacementSourcePath(out _, out string sourceError))
            {
                _selectedPlacementSaveStatus = sourceError;
            }
            else if (_worldScene.TryUpdateSelectedPlacementPosition(editedPosition, out string error))
            {
                _selectedPlacementEditedPosition = editedPosition;
                _selectedPlacementDirty = !PositionsNearlyEqual(_selectedPlacementEditedPosition, _selectedPlacementOriginalPosition);
                if (_selectedPlacementDirty)
                {
                    UpsertSelectedPlacementEdit();
                    _selectedPlacementSaveStatus = BuildSelectedPlacementSaveStatus();
                }
                else
                {
                    RemoveSelectedPlacementEdit();
                    _selectedPlacementSaveStatus = HasPendingPlacementEdits()
                        ? "Preview matches the source tile placement. Other staged placement moves remain pending."
                        : "Preview matches the source tile placement position.";
                }

                RefreshSelectedWorldObjectInfo();
            }
            else
            {
                _selectedPlacementSaveStatus = error;
            }
        }

        if (_selectedPlacementDirty && ImGui.Button("Reset Preview"))
        {
            if (_worldScene.TryUpdateSelectedPlacementPosition(_selectedPlacementOriginalPosition, out string error))
            {
                _selectedPlacementEditedPosition = _selectedPlacementOriginalPosition;
                _selectedPlacementDirty = false;
                RemoveSelectedPlacementEdit();
                _selectedPlacementSaveStatus = HasPendingPlacementEdits()
                    ? "Preview reset to the source tile placement position. Other staged placement moves remain pending."
                    : "Preview reset to the source tile placement position.";
                RefreshSelectedWorldObjectInfo();
            }
            else
            {
                _selectedPlacementSaveStatus = error;
            }
        }

        string targetLabel = string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
            ? "No save path selected."
            : _selectedPlacementSaveTargetPath!;
        ImGui.TextWrapped($"Save target: {targetLabel}");
        ImGui.TextDisabled("Writes an ADT copy to disk. The loaded source files are not overwritten in place.");

        if (ImGui.Button("Choose Save Path"))
            ChooseSelectedPlacementSavePath();

        DrawPlacementSaveQueueActions(includeCurrentSourceSave: true);

        ImGui.TextDisabled(_selectedPlacementSaveStatus);
    }

    private void SyncSelectedPlacementEditState()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            ResetSelectedPlacementEditState("Select a tile-backed world object to stage a translation-only save.");
            return;
        }

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        if (!selected.HasTileCoordinate || selected.PlacementEntryIndex < 0)
        {
            ResetSelectedPlacementEditState("The selected object is not backed by a writable ADT tile placement.");
            return;
        }

        Terrain.ObjectType selectedType = _worldScene.SelectedObjectType;
        if (selectedType is not (Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo))
        {
            ResetSelectedPlacementEditState("Only MDDF and MODF tile placements are supported by the current save seam.");
            return;
        }

        bool sameSelection = _selectedPlacementEditType == selectedType
            && _selectedPlacementEditUniqueId == selected.UniqueId
            && _selectedPlacementEditTileX == selected.TileX
            && _selectedPlacementEditTileY == selected.TileY
            && _selectedPlacementEditEntryIndex == selected.PlacementEntryIndex;

        if (sameSelection)
            return;

        _selectedPlacementEditType = selectedType;
        _selectedPlacementEditUniqueId = selected.UniqueId;
        _selectedPlacementEditTileX = selected.TileX;
        _selectedPlacementEditTileY = selected.TileY;
        _selectedPlacementEditEntryIndex = selected.PlacementEntryIndex;
        PlacementEditKey key = CreatePlacementEditKey(selectedType, selected);
        if (_stagedPlacementEdits.TryGetValue(key, out StagedPlacementEdit? stagedEdit))
        {
            _selectedPlacementOriginalPosition = stagedEdit.OriginalPosition;
            _selectedPlacementEditedPosition = stagedEdit.EditedPosition;
            _selectedPlacementDirty = !PositionsNearlyEqual(stagedEdit.EditedPosition, stagedEdit.OriginalPosition);
            _selectedPlacementSourcePath = stagedEdit.SourcePath;
        }
        else
        {
            _selectedPlacementOriginalPosition = selected.PlacementPosition;
            _selectedPlacementEditedPosition = selected.PlacementPosition;
            _selectedPlacementDirty = false;
            _selectedPlacementSourcePath = null;
        }

        _selectedPlacementSaveTargetPath = null;
        if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath)
            && _placementSaveTargetsBySourcePath.TryGetValue(_selectedPlacementSourcePath, out string? stagedTarget)
            && !string.IsNullOrWhiteSpace(stagedTarget))
        {
            _selectedPlacementSaveTargetPath = stagedTarget;
        }
        else if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath))
        {
            _selectedPlacementSaveTargetPath = BuildProjectManagedPlacementOutputPath(_selectedPlacementSourcePath);
        }

        _selectedPlacementSaveStatus = _selectedPlacementDirty
            ? BuildSelectedPlacementSaveStatus()
            : string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                ? "Adjust the selected placement to stage a dirty ADT source. A timestamped project output folder will be created for the save target."
                : "Ready to stage placement moves for this ADT source.";
    }

    private void ResetSelectedPlacementEditState(string status)
    {
        _selectedPlacementEditType = Terrain.ObjectType.None;
        _selectedPlacementEditUniqueId = -1;
        _selectedPlacementEditTileX = -1;
        _selectedPlacementEditTileY = -1;
        _selectedPlacementEditEntryIndex = -1;
        _selectedPlacementOriginalPosition = Vector3.Zero;
        _selectedPlacementEditedPosition = Vector3.Zero;
        _selectedPlacementDirty = false;
        _selectedPlacementSourcePath = null;
        _selectedPlacementSaveTargetPath = null;
        _selectedPlacementSaveStatus = HasPendingPlacementEdits()
            ? $"{status} {GetPendingPlacementEditCount()} staged move(s) across {GetPendingPlacementSourceCount()} ADT source(s) remain pending."
            : status;
    }

    private void ChooseSelectedPlacementSavePath()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return;

        if (!EnsureSelectedPlacementSourcePath(out string sourcePath, out string error))
        {
            _selectedPlacementSaveStatus = error;
            return;
        }

        string initialDir = Environment.CurrentDirectory;
        string defaultFileName = $"placement_{DateTime.Now:yyyyMMdd_HHmmss}.adt";

        defaultFileName = Path.GetFileName(sourcePath);

        string projectManagedTargetPath = BuildProjectManagedPlacementOutputPath(sourcePath);
        if (string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath) || IsProjectManagedOutputPath(_selectedPlacementSaveTargetPath))
            _selectedPlacementSaveTargetPath = projectManagedTargetPath;

        if (!string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath))
        {
            string? existingDir = Path.GetDirectoryName(_selectedPlacementSaveTargetPath);
            if (!string.IsNullOrWhiteSpace(existingDir) && Directory.Exists(existingDir))
                initialDir = existingDir;
            defaultFileName = Path.GetFileName(_selectedPlacementSaveTargetPath);
        }
        else if (_worldScene.TryGetSelectedPlacementWritablePath(out string? writablePath) && !string.IsNullOrWhiteSpace(writablePath))
        {
            string? writableDir = Path.GetDirectoryName(writablePath);
            if (!string.IsNullOrWhiteSpace(writableDir) && Directory.Exists(writableDir))
                initialDir = writableDir;
            defaultFileName = Path.GetFileName(writablePath);
        }

        ImGuiPathPicker.Instance.Open(
            "Save moved ADT placement as",
            ImGuiPathPickerMode.SaveFile,
            initialDir,
            ".adt",
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                _selectedPlacementSaveTargetPath = picked;
                _placementSaveTargetsBySourcePath[sourcePath] = picked;
                int pendingForSource = GetPendingPlacementCountForSource(sourcePath);
                _selectedPlacementSaveStatus = pendingForSource > 0
                    ? $"Ready to save {pendingForSource} staged placement move(s) from {Path.GetFileName(sourcePath)} to {picked}."
                    : $"Default save target for {Path.GetFileName(sourcePath)} set to {picked}.";
            },
            defaultFileName);
    }

    private void SaveSelectedPlacementEdit()
    {
        if (!EnsureSelectedPlacementSourcePath(out string sourcePath, out string error))
        {
            _selectedPlacementSaveStatus = error;
            return;
        }

        SaveStagedPlacementEdits(sourcePath);
    }

    internal void DrawPlacementSaveQueueActions(bool includeCurrentSourceSave)
    {
        int pendingEditCount = GetPendingPlacementEditCount();
        int pendingSourceCount = GetPendingPlacementSourceCount();

        if (pendingEditCount <= 0)
            return;

        ImGui.Separator();
        ImGui.Text($"{pendingEditCount} staged placement move(s) across {pendingSourceCount} ADT source(s).");

        string? currentSourcePath = null;
        int currentSourcePendingCount = 0;
        if (includeCurrentSourceSave && TryGetSelectedPlacementSourcePathForQueue(out string sourcePath))
        {
            currentSourcePath = sourcePath;
            currentSourcePendingCount = GetPendingPlacementCountForSource(sourcePath);
        }

        if (includeCurrentSourceSave)
        {
            if (currentSourcePendingCount <= 0)
                ImGui.BeginDisabled();
            if (ImGui.Button("Save Current Source") && currentSourcePath != null)
                SaveStagedPlacementEdits(currentSourcePath);
            if (currentSourcePendingCount <= 0)
                ImGui.EndDisabled();

            ImGui.SameLine();
        }

        if (ImGui.Button("Save All Pending"))
            SaveStagedPlacementEdits();

        if (ImGui.CollapsingHeader("Pending Dirty Sources", ImGuiTreeNodeFlags.DefaultOpen))
        {
            foreach ((string pendingSourcePath, int editCount, string? targetPath) in EnumeratePendingPlacementSourceSummaries())
            {
                ImGui.TextWrapped($"{editCount} move(s): {pendingSourcePath}");
                ImGui.TextDisabled(string.IsNullOrWhiteSpace(targetPath)
                    ? "Output: choose an .adt path before save."
                    : $"Output: {targetPath}");
            }
        }
    }

    private void SaveStagedPlacementEdits(string? sourcePathFilter = null)
    {
        if (_dataSource == null)
        {
            _selectedPlacementSaveStatus = "Placement save failed: no data source is loaded.";
            return;
        }

        List<(string SourcePath, List<StagedPlacementEdit> Edits)> groups = BuildPendingPlacementSaveGroups(sourcePathFilter);
        if (groups.Count == 0)
        {
            _selectedPlacementSaveStatus = string.IsNullOrWhiteSpace(sourcePathFilter)
                ? "No staged placement moves to save."
                : "No staged placement moves are pending for the selected ADT source.";
            return;
        }

        List<string> missingTargets = new();
        foreach ((string sourcePath, _) in groups)
        {
            if (!_placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? outputPath) || string.IsNullOrWhiteSpace(outputPath))
                missingTargets.Add(sourcePath);
        }

        if (missingTargets.Count > 0)
        {
            string missingSummary = missingTargets.Count == 1
                ? missingTargets[0]
                : $"{missingTargets.Count} ADT sources";
            _selectedPlacementSaveStatus = $"Choose an output .adt path before saving pending placement moves for {missingSummary}.";
            return;
        }

        var savedKeys = new List<PlacementEditKey>();
        int savedSourceCount = 0;
        int savedEditCount = 0;

        try
        {
            foreach ((string sourcePath, List<StagedPlacementEdit> edits) in groups)
            {
                string outputPath = _placementSaveTargetsBySourcePath[sourcePath];
                byte[]? sourceBytes = File.Exists(outputPath)
                    ? File.ReadAllBytes(outputPath)
                    : _dataSource.ReadFile(sourcePath);
                if (sourceBytes == null)
                    throw new InvalidOperationException($"The source ADT could not be read from the current data source: {sourcePath}");

                // Apply every staged edit through the library-first placement editor, which rebuilds
                // only the placement/name-table chunks and preserves all other bytes.
                var placementEdits = new List<AdtPlacementEdit>(edits.Count);
                foreach (StagedPlacementEdit edit in edits)
                {
                    AdtPlacementKind kind = edit.Key.ObjectType == Terrain.ObjectType.Wmo
                        ? AdtPlacementKind.WorldModel
                        : AdtPlacementKind.Model;

                    if (edit.Deleted)
                    {
                        placementEdits.Add(new AdtPlacementDeleteEdit(kind, edit.Key.EntryIndex, edit.Key.UniqueId));
                        continue;
                    }

                    placementEdits.Add(new AdtPlacementMoveEdit(kind, edit.Key.EntryIndex, edit.Key.UniqueId, edit.EditedPosition));
                    if (edit.EditedRotation.HasValue)
                        placementEdits.Add(new AdtPlacementRotateEdit(kind, edit.Key.EntryIndex, edit.Key.UniqueId, edit.EditedRotation.Value));
                    if (edit.EditedScale.HasValue)
                        placementEdits.Add(new AdtPlacementScaleEdit(kind, edit.Key.EntryIndex, edit.Key.UniqueId, edit.EditedScale.Value));
                }

                byte[] updatedBytes = AdtPlacementEditor.Apply(sourceBytes, sourcePath, placementEdits).Bytes;

                string? outputDirectory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrWhiteSpace(outputDirectory))
                    Directory.CreateDirectory(outputDirectory);

                File.WriteAllBytes(outputPath, updatedBytes);

                savedSourceCount++;
                savedEditCount += edits.Count;
                foreach (StagedPlacementEdit edit in edits)
                    savedKeys.Add(edit.Key);
            }

            foreach (PlacementEditKey key in savedKeys)
                _stagedPlacementEdits.Remove(key);

            if (TryGetSelectedPlacementKey(out PlacementEditKey selectedKey) && savedKeys.Contains(selectedKey))
            {
                _selectedPlacementOriginalPosition = _selectedPlacementEditedPosition;
                _selectedPlacementDirty = false;
            }

            _selectedPlacementSaveStatus = BuildPlacementSaveCompletionStatus(groups, savedEditCount, savedSourceCount);
        }
        catch (Exception ex)
        {
            _selectedPlacementSaveStatus = $"Placement save failed: {ex.Message}";
            return;
        }

        SyncSelectedPlacementEditState();
        RefreshSelectedWorldObjectInfo();
    }

    private string BuildPlacementSaveCompletionStatus(
        IReadOnlyList<(string SourcePath, List<StagedPlacementEdit> Edits)> groups,
        int savedEditCount,
        int savedSourceCount)
    {
        List<string> outputPaths = groups
            .Select(group => _placementSaveTargetsBySourcePath[group.SourcePath])
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (outputPaths.Count == 1)
        {
            return $"Saved {savedEditCount} staged placement move(s) across {savedSourceCount} ADT source(s) to {outputPaths[0]}. Source ADTs were left untouched.";
        }

        if (outputPaths.All(IsProjectManagedOutputPath) && !string.IsNullOrWhiteSpace(_editorProjectOutputDir))
        {
            string projectOutputDir = Path.Combine(_editorProjectOutputDir, "lk-split");
            return $"Saved {savedEditCount} staged placement move(s) across {savedSourceCount} ADT source(s) into {projectOutputDir}. Source ADTs were left untouched.";
        }

        string previewPaths = string.Join("; ", outputPaths.Take(2));
        if (outputPaths.Count > 2)
            previewPaths += $"; +{outputPaths.Count - 2} more";

        return $"Saved {savedEditCount} staged placement move(s) across {savedSourceCount} ADT source(s). Output ADT copies: {previewPaths}. Source ADTs were left untouched.";
    }

    private List<(string SourcePath, List<StagedPlacementEdit> Edits)> BuildPendingPlacementSaveGroups(string? sourcePathFilter)
    {
        var grouped = new Dictionary<string, List<StagedPlacementEdit>>(StringComparer.OrdinalIgnoreCase);

        foreach (StagedPlacementEdit edit in _stagedPlacementEdits.Values)
        {
            if (!string.IsNullOrWhiteSpace(sourcePathFilter)
                && !string.Equals(edit.SourcePath, sourcePathFilter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!grouped.TryGetValue(edit.SourcePath, out List<StagedPlacementEdit>? edits))
            {
                edits = new List<StagedPlacementEdit>();
                grouped.Add(edit.SourcePath, edits);
            }

            edits.Add(edit);
        }

        return grouped
            .OrderBy(entry => entry.Key, StringComparer.OrdinalIgnoreCase)
            .Select(entry => (entry.Key, entry.Value))
            .ToList();
    }

    private IEnumerable<(string SourcePath, int EditCount, string? TargetPath)> EnumeratePendingPlacementSourceSummaries()
    {
        foreach ((string sourcePath, List<StagedPlacementEdit> edits) in BuildPendingPlacementSaveGroups(sourcePathFilter: null))
        {
            _placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? targetPath);
            yield return (sourcePath, edits.Count, targetPath);
        }
    }

    private bool EnsureSelectedPlacementSourcePath(out string sourcePath, out string error)
    {
        sourcePath = _selectedPlacementSourcePath ?? string.Empty;
        error = string.Empty;

        if (!string.IsNullOrWhiteSpace(sourcePath))
            return true;

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            error = "No tile-backed world object is selected.";
            return false;
        }

        if (!_worldScene.TryGetSelectedPlacementSourceData(out sourcePath, out _))
        {
            error = "The selected placement source ADT could not be read from the current data source.";
            return false;
        }

        _selectedPlacementSourcePath = sourcePath;
        if (!string.IsNullOrWhiteSpace(sourcePath))
        {
            if (!_placementSaveTargetsBySourcePath.TryGetValue(sourcePath, out string? savedTarget)
                || string.IsNullOrWhiteSpace(savedTarget)
                || IsProjectManagedOutputPath(savedTarget))
            {
                savedTarget = BuildProjectManagedPlacementOutputPath(sourcePath);
                _placementSaveTargetsBySourcePath[sourcePath] = savedTarget;
            }

            if (string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath) || IsProjectManagedOutputPath(_selectedPlacementSaveTargetPath))
                _selectedPlacementSaveTargetPath = savedTarget;
        }

        return true;
    }

    /// <summary>
    /// Upserts one authored placement edit (move/rotate/scale/delete) into the shared staged-save
    /// queue from any authoring surface. Repeated edits to the same row collapse into the latest
    /// state; earlier field values are preserved so a rotate after a move keeps both.
    /// </summary>
    internal void StageAuthoringPlacementEdit(
        Terrain.ObjectType objectType,
        ObjectInstance selected,
        string sourcePath,
        Vector3? position = null,
        Vector3? rotation = null,
        float? scale = null,
        bool delete = false)
    {
        PlacementEditKey key = CreatePlacementEditKey(objectType, selected);
        if (!_stagedPlacementEdits.TryGetValue(key, out StagedPlacementEdit? edit))
        {
            edit = new StagedPlacementEdit
            {
                Key = key,
                SourcePath = sourcePath,
                OriginalPosition = selected.PlacementPosition,
                EditedPosition = selected.PlacementPosition,
            };
            _stagedPlacementEdits[key] = edit;
        }

        if (position.HasValue)
            edit.EditedPosition = position.Value;
        if (rotation.HasValue)
            edit.EditedRotation = rotation.Value;
        if (scale.HasValue)
            edit.EditedScale = scale.Value;
        edit.Deleted = delete;

        if (!string.IsNullOrWhiteSpace(sourcePath) && !_placementSaveTargetsBySourcePath.ContainsKey(sourcePath))
            _placementSaveTargetsBySourcePath[sourcePath] = BuildProjectManagedPlacementOutputPath(sourcePath);
    }

    private void UpsertSelectedPlacementEdit()
    {
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return;

        if (!EnsureSelectedPlacementSourcePath(out string sourcePath, out _))
            return;

        PlacementEditKey key = CreatePlacementEditKey(_worldScene.SelectedObjectType, _worldScene.SelectedInstance.Value);
        _stagedPlacementEdits[key] = new StagedPlacementEdit
        {
            Key = key,
            SourcePath = sourcePath,
            OriginalPosition = _selectedPlacementOriginalPosition,
            EditedPosition = _selectedPlacementEditedPosition,
        };

        if (!string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath))
            _placementSaveTargetsBySourcePath[sourcePath] = _selectedPlacementSaveTargetPath!;
    }

    private void RemoveSelectedPlacementEdit()
    {
        if (!TryGetSelectedPlacementKey(out PlacementEditKey key))
            return;

        _stagedPlacementEdits.Remove(key);
    }

    private bool TryGetSelectedPlacementKey(out PlacementEditKey key)
    {
        key = default;
        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
            return false;

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        Terrain.ObjectType selectedType = _worldScene.SelectedObjectType;
        if (!selected.HasTileCoordinate || selected.PlacementEntryIndex < 0 || selectedType is not (Terrain.ObjectType.Mdx or Terrain.ObjectType.Wmo))
            return false;

        key = CreatePlacementEditKey(selectedType, selected);
        return true;
    }

    private bool TryGetSelectedPlacementSourcePathForQueue(out string sourcePath)
    {
        sourcePath = string.Empty;
        if (!_selectedPlacementDirty && !HasPendingPlacementEdits())
            return false;

        return EnsureSelectedPlacementSourcePath(out sourcePath, out _);
    }

    private PlacementEditKey CreatePlacementEditKey(Terrain.ObjectType objectType, ObjectInstance selected)
    {
        return new PlacementEditKey(objectType, selected.TileX, selected.TileY, selected.PlacementEntryIndex, selected.UniqueId);
    }

    private bool HasPendingPlacementEdits()
    {
        return _stagedPlacementEdits.Count > 0;
    }

    internal int GetPendingPlacementEditCount()
    {
        return _stagedPlacementEdits.Count;
    }

    internal int GetPendingPlacementSourceCount()
    {
        return _stagedPlacementEdits.Values
            .Select(edit => edit.SourcePath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Count();
    }

    private int GetPendingPlacementCountForSource(string sourcePath)
    {
        int count = 0;
        foreach (StagedPlacementEdit edit in _stagedPlacementEdits.Values)
        {
            if (string.Equals(edit.SourcePath, sourcePath, StringComparison.OrdinalIgnoreCase))
                count++;
        }

        return count;
    }

    internal int GetPendingPlacementSourceCountMissingTargets()
    {
        int count = 0;
        foreach ((string sourcePath, _, string? targetPath) in EnumeratePendingPlacementSourceSummaries())
        {
            if (string.IsNullOrWhiteSpace(targetPath))
                count++;
        }

        return count;
    }

    private string BuildSelectedPlacementSaveStatus()
    {
        int pendingForSource = 0;
        if (!string.IsNullOrWhiteSpace(_selectedPlacementSourcePath))
            pendingForSource = GetPendingPlacementCountForSource(_selectedPlacementSourcePath);

        if (pendingForSource > 0)
        {
            return string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                ? $"Preview updated. {pendingForSource} staged placement move(s) are pending for this ADT source. Choose an output .adt path before saving."
                : $"Preview updated. {pendingForSource} staged placement move(s) are pending for this ADT source.";
        }

        return string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
            ? "Preview updated. A timestamped project output folder will be used unless you choose a different .adt path."
            : "Preview updated. Save writes a translated copy into the active project output folder unless you override the target.";
    }

    private static bool PositionsNearlyEqual(Vector3 left, Vector3 right)
    {
        return Vector3.DistanceSquared(left, right) < 0.0001f;
    }
}
