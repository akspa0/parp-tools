using System.Numerics;
using ImGuiNET;
using MdxViewer.Rendering;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Partial class containing viewer/editor workspace shell helpers.
/// </summary>
public partial class ViewerApp
{
    private static string GetWorkspaceModeLabel(WorkspaceMode mode)
    {
        return mode == WorkspaceMode.Viewer ? "Viewer" : "Editor";
    }

    private static string GetEditorWorkspaceTaskLabel(EditorWorkspaceTask task)
    {
        return task switch
        {
            EditorWorkspaceTask.Terrain => "Terrain",
            EditorWorkspaceTask.Objects => "Objects",
            EditorWorkspaceTask.Pm4Evidence => "PM4 Evidence",
            EditorWorkspaceTask.Inspect => "Inspect",
            EditorWorkspaceTask.Publish => "Publish",
            _ => "Unknown",
        };
    }

    private void SetWorkspaceMode(WorkspaceMode mode)
    {
        _workspaceMode = mode;
        _showLeftSidebar = true;
        _showRightSidebar = true;

        if (mode == WorkspaceMode.Editor && !HasWorldEditingContext())
            _editorWorkspaceTask = EditorWorkspaceTask.Inspect;
    }

    private void SetEditorWorkspaceTask(EditorWorkspaceTask task)
    {
        if (_workspaceMode != WorkspaceMode.Editor)
            SetWorkspaceMode(WorkspaceMode.Editor);

        _editorWorkspaceTask = task;
    }

    private bool HasWorldEditingContext()
    {
        return _worldScene != null || _terrainManager != null || _vlmTerrainManager != null;
    }

    private bool HasTerrainEditingContext()
    {
        return _terrainManager != null || _vlmTerrainManager != null;
    }

    private bool IsEditorTaskAvailable(EditorWorkspaceTask task)
    {
        return task switch
        {
            EditorWorkspaceTask.Terrain => HasTerrainEditingContext(),
            EditorWorkspaceTask.Objects => _worldScene != null,
            EditorWorkspaceTask.Pm4Evidence => _worldScene != null,
            EditorWorkspaceTask.Inspect => true,
            EditorWorkspaceTask.Publish => HasWorldEditingContext() || _renderer != null,
            _ => false,
        };
    }

    private void DrawWorkspaceToolbarControls()
    {
        ImGui.TextDisabled("Workspace");
        ImGui.SameLine();
        if (ImGui.RadioButton("Viewer", _workspaceMode == WorkspaceMode.Viewer))
            SetWorkspaceMode(WorkspaceMode.Viewer);

        ImGui.SameLine();
        if (ImGui.RadioButton("Editor", _workspaceMode == WorkspaceMode.Editor))
            SetWorkspaceMode(WorkspaceMode.Editor);

        if (_workspaceMode == WorkspaceMode.Editor)
        {
            ImGui.SameLine();
            ImGui.TextColored(new Vector4(0.5f, 0.5f, 0.5f, 1f), "|");
            ImGui.SameLine();
            ImGui.TextDisabled("Task");
            ImGui.SameLine();
            ImGui.SetNextItemWidth(170f);
            if (ImGui.BeginCombo("##EditorWorkspaceTask", GetEditorWorkspaceTaskLabel(_editorWorkspaceTask)))
            {
                foreach (EditorWorkspaceTask task in Enum.GetValues<EditorWorkspaceTask>())
                {
                    bool isAvailable = IsEditorTaskAvailable(task);
                    if (!isAvailable)
                        ImGui.BeginDisabled();

                    bool isSelected = task == _editorWorkspaceTask;
                    if (ImGui.Selectable(GetEditorWorkspaceTaskLabel(task), isSelected))
                        SetEditorWorkspaceTask(task);

                    if (isSelected)
                        ImGui.SetItemDefaultFocus();

                    if (!isAvailable)
                        ImGui.EndDisabled();
                }

                ImGui.EndCombo();
            }
        }
    }

    private void DrawEditorWorkspaceNavigator(bool hasWorldLoaded)
    {
        ImGui.SetNextItemOpen(true, ImGuiCond.Once);
        if (ImGui.CollapsingHeader("Editor Workspace", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled("One app, shared services, explicit editor tasks.");
            ImGui.TextDisabled($"Current task: {GetEditorWorkspaceTaskLabel(_editorWorkspaceTask)}");
            ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
            ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
            ImGui.Separator();

            foreach (EditorWorkspaceTask task in Enum.GetValues<EditorWorkspaceTask>())
            {
                bool isAvailable = IsEditorTaskAvailable(task);
                if (!isAvailable)
                    ImGui.BeginDisabled();

                bool isSelected = task == _editorWorkspaceTask;
                if (ImGui.Selectable(GetEditorWorkspaceTaskLabel(task), isSelected))
                    SetEditorWorkspaceTask(task);

                if (ImGui.IsItemHovered(ImGuiHoveredFlags.AllowWhenDisabled))
                {
                    ImGui.BeginTooltip();
                    ImGui.TextDisabled(GetEditorWorkspaceTooltip(task));
                    ImGui.EndTooltip();
                }

                if (!isAvailable)
                    ImGui.EndDisabled();
            }
        }

        if (hasWorldLoaded)
        {
            ImGui.SetNextItemOpen(true, ImGuiCond.Once);
            if (ImGui.CollapsingHeader("World Overview", ImGuiTreeNodeFlags.DefaultOpen))
                DrawWorldOverviewContent();
        }

        ImGui.SetNextItemOpen(!hasWorldLoaded || _editorWorkspaceTask == EditorWorkspaceTask.Inspect, ImGuiCond.Once);
        if (_showFileBrowser && ImGui.CollapsingHeader("Map / Assets"))
            DrawFileBrowserContent();

        ImGui.SetNextItemOpen(!hasWorldLoaded, ImGuiCond.Once);
        if (_discoveredMaps.Count > 0 && ImGui.CollapsingHeader("World Maps"))
            DrawMapDiscoveryContent();
    }

    private void DrawEditorWorkspaceInspector()
    {
        ImGui.TextColored(new Vector4(0.75f, 0.88f, 1f, 1f), $"{GetWorkspaceModeLabel(_workspaceMode)} Workspace");
        ImGui.SameLine();
        ImGui.TextDisabled(GetEditorWorkspaceTaskLabel(_editorWorkspaceTask));
        ImGui.TextDisabled($"Target: {GetWorkspaceTargetSummary()}");
        ImGui.TextDisabled($"Save: {GetWorkspaceSaveStatusSummary()}");
        ImGui.Separator();

        switch (_editorWorkspaceTask)
        {
            case EditorWorkspaceTask.Terrain:
                DrawEditorTerrainWorkspace();
                break;
            case EditorWorkspaceTask.Objects:
                DrawEditorObjectsWorkspace();
                break;
            case EditorWorkspaceTask.Pm4Evidence:
                DrawEditorPm4Workspace();
                break;
            case EditorWorkspaceTask.Inspect:
                DrawEditorInspectWorkspace();
                break;
            case EditorWorkspaceTask.Publish:
                DrawEditorPublishWorkspace();
                break;
        }
    }

    private void DrawEditorTerrainWorkspace()
    {
        TerrainRenderer? renderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (renderer == null)
        {
            ImGui.TextWrapped("Load a terrain-backed world to use terrain editing tools.");
            return;
        }

        ImGui.TextWrapped("Terrain actions are live-scene only in the current viewer. Use this workspace to make target and save status explicit.");

        if (ImGui.CollapsingHeader("Chunk Clipboard", ImGuiTreeNodeFlags.DefaultOpen))
            DrawChunkClipboardContent(renderer);

        ImGui.Separator();
        ImGui.Text("Terrain Import / Export");
        if (ImGui.Button("Export Alpha Atlas"))
        {
            _terrainExportKind = TerrainExportKind.AlphaCurrentTileAtlas;
            _wantTerrainExport = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Export Heightmap"))
        {
            _terrainExportKind = TerrainExportKind.Heightmap257CurrentTilePerTile;
            _wantTerrainExport = true;
        }

        if (ImGui.Button("Import Alpha Folder"))
        {
            _terrainImportKind = TerrainImportKind.AlphaFolder;
            _wantTerrainImport = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Import Heightmaps"))
        {
            _terrainImportKind = TerrainImportKind.Heightmap257Folder;
            _wantTerrainImport = true;
        }

        ImGui.Separator();
        DrawTerrainControlsContent();
    }

    private void DrawEditorObjectsWorkspace()
    {
        if (_worldScene == null)
        {
            ImGui.TextWrapped("Load a world scene to inspect object selection, archaeology, and world-population tools.");
            return;
        }

        ImGui.TextWrapped("This is the first regrouping slice. The current world-object tools still come from the existing inspector surface, but they now live under an explicit object task.");
        DrawWorldObjectsContentCore();
    }

    private void DrawEditorPm4Workspace()
    {
        if (_worldScene == null)
        {
            ImGui.TextWrapped("Load a world scene to inspect PM4 overlay, matches, and correlation.");
            return;
        }

        ImGui.TextWrapped("PM4 stays evidence-first here. This workspace groups overlay tuning, selected-object inspection, graph, and correlation without implying save ownership.");
        ImGui.Separator();
        DrawPm4WorkbenchInspector();
    }

    private void DrawEditorInspectWorkspace()
    {
        ImGui.TextWrapped("Use Navigator for map and asset browse. This task keeps read-only inspection, camera, and utility panels together.");

        ImGui.SetNextItemOpen(!string.IsNullOrEmpty(_modelInfo), ImGuiCond.Once);
        if (_showModelInfo && ImGui.CollapsingHeader("Model Info", ImGuiTreeNodeFlags.DefaultOpen))
            DrawModelInfoContent();

        ImGui.SetNextItemOpen(true, ImGuiCond.Once);
        if (ImGui.CollapsingHeader("Camera", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.SliderFloat("Camera Speed", ref _cameraSpeed, 1f, 500f, "%.0f");
            ImGui.Text("Hold Shift for 5x boost");
            ImGui.SliderFloat("FOV", ref _fovDegrees, 20f, 90f, "%.0f°");
        }

        ImGui.Separator();
        ImGui.Text("Utility Panels");
        if (ImGui.Button(_showMinimapWindow ? "Hide Minimap" : "Show Minimap"))
            _showMinimapWindow = !_showMinimapWindow;

        ImGui.SameLine();
        if (ImGui.Button(_showLogViewer ? "Hide Log Viewer" : "Show Log Viewer"))
            _showLogViewer = !_showLogViewer;

        if (ImGui.Button(_showPerfWindow ? "Hide Perf" : "Show Perf"))
            _showPerfWindow = !_showPerfWindow;

        ImGui.SameLine();
        if (ImGui.Button(_showRenderQualityWindow ? "Hide Render Quality" : "Show Render Quality"))
            _showRenderQualityWindow = !_showRenderQualityWindow;
    }

    private void DrawEditorPublishWorkspace()
    {
        ImGui.TextWrapped("Save and publish are explicit here. Current MdxViewer support is still narrow: translation-only saves for selected existing ADT object placements, plus export and capture. General map save and terrain persistence are still not implemented.");
        ImGui.TextDisabled(GetWorkspaceSaveStatusSummary());
        ImGui.Separator();

        if (ImGui.Button("Capture Current (No UI)"))
            QueueCurrentCameraCapture(includeUi: false);

        ImGui.SameLine();
        if (ImGui.Button("Capture Current (With UI)"))
            QueueCurrentCameraCapture(includeUi: true);

        if (ImGui.Button("Open Capture Automation"))
            _showCaptureAutomationWindow = true;

        ImGui.Separator();
        ImGui.Text("Export");
        if (ImGui.Button("Export GLB"))
            _wantExportGlb = true;

        ImGui.SameLine();
        if (ImGui.Button("Export GLB Collision"))
            _wantExportGlbCollision = true;

        if (HasTerrainEditingContext())
        {
            if (ImGui.Button("Export Terrain Alpha"))
            {
                _terrainExportKind = TerrainExportKind.AlphaCurrentTileAtlas;
                _wantTerrainExport = true;
            }

            ImGui.SameLine();
            if (ImGui.Button("Export Terrain Heightmap"))
            {
                _terrainExportKind = TerrainExportKind.Heightmap257CurrentTilePerTile;
                _wantTerrainExport = true;
            }
        }
    }

    private static string GetEditorWorkspaceTooltip(EditorWorkspaceTask task)
    {
        return task switch
        {
            EditorWorkspaceTask.Terrain => "Chunk clipboard, terrain import/export, and live terrain controls.",
            EditorWorkspaceTask.Objects => "Selection, archaeology, world population, and object-facing scene tools.",
            EditorWorkspaceTask.Pm4Evidence => "Overlay tuning, selected PM4 object inspection, graph, and correlation.",
            EditorWorkspaceTask.Inspect => "Read-only model, camera, map, and utility inspection tools.",
            EditorWorkspaceTask.Publish => "Capture and export surfaces. No map save pipeline yet.",
            _ => string.Empty,
        };
    }

    private string GetWorkspaceTargetSummary()
    {
        if (_worldScene?.HasSelectedPm4Object == true && _worldScene.SelectedPm4ObjectKey.HasValue)
        {
            var selectedPm4 = _worldScene.SelectedPm4ObjectKey.Value;
            return $"PM4 CK24 0x{selectedPm4.ck24:X6} part {selectedPm4.objectPart}";
        }

        if (!string.IsNullOrEmpty(_selectedObjectType) && _selectedObjectIndex >= 0)
            return $"{_selectedObjectType} #{_selectedObjectIndex}";

        if (_selectedChunks.Count > 0)
            return $"{_selectedChunks.Count} selected chunk(s)";

        if (_terrainManager != null || _vlmTerrainManager != null || _worldScene != null)
        {
            int tileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
            int tileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
            return $"Camera tile ({tileX}, {tileY})";
        }

        if (!string.IsNullOrWhiteSpace(_loadedFileName))
            return _loadedFileName!;

        return "No active target";
    }

    private string GetWorkspaceSaveStatusSummary()
    {
        return _workspaceMode switch
        {
            WorkspaceMode.Viewer => "Read-only viewer workspace.",
            WorkspaceMode.Editor when _selectedPlacementDirty && !string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                => $"1 pending selected placement move. Target: {_selectedPlacementSaveTargetPath}",
            WorkspaceMode.Editor when _selectedPlacementDirty
                => "1 pending selected placement move. Choose an output .adt path to save.",
            WorkspaceMode.Editor when HasWorldEditingContext() && !string.IsNullOrWhiteSpace(_selectedPlacementSaveTargetPath)
                => $"Selected placement save target: {_selectedPlacementSaveTargetPath}. No general map save pipeline yet.",
            WorkspaceMode.Editor when HasWorldEditingContext()
                => "Selected placement save available for translation-only ADT object moves. No general map save pipeline yet.",
            WorkspaceMode.Editor => "Editor workspace loaded without a world save target.",
            _ => "Unknown save state.",
        };
    }
}