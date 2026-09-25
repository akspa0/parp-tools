using System.Numerics;
using System.Diagnostics;
using System.Text.Json;
using ImGuiNET;
using WoWViewer.DataSources;
using WoWViewer.Workbench;
using WoWViewer.Logging;
using WoWViewer.Rendering;
using WoWViewer.Terrain;
using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Passes;
using WowViewer.Core.Runtime.World.Visibility;
using WoWViewer.Population;
using WoWViewer.UI;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ArchaeologyPanelService: members moved from ViewerApp_Sidebars.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ArchaeologyPanelService
{

    // ── Tool windows extracted from right sidebar ──────────────────────

    internal void DrawUniqueIdArchaeologyWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(420f, 360f), ImGuiCond.FirstUseEver);
        if (ImGui.Begin("UniqueId Archaeology", ref _showUniqueIdArchaeologyWindow))
        {
            DrawUniqueIdArchaeologyContent();
        }
        ImGui.End();
    }

    private void DrawUniqueIdArchaeologyContent()
    {
        // Legacy entry point (was used by the floating window). Now delegates
        // to the per-sub-tab dispatch. Kept so the floating window still works
        // when _useTabUi = false.
        if (_worldScene == null)
            return;

        int cameraTileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int cameraTileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
        _worldScene.SetUniqueIdFilterTile(cameraTileX, cameraTileY);

        DrawArcheologyRangeSubTab();
        ImGui.Separator();
        DrawArcheologyLayersSubTab();
        ImGui.Separator();
        DrawArcheologyPlaybackSubTab();
    }

    private void DrawArcheologyRangeSubTab()
    {
        int cameraTileX = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.X) / WoWConstants.ChunkSize);
        int cameraTileY = (int)MathF.Floor((WoWConstants.MapOrigin - _camera.Position.Y) / WoWConstants.ChunkSize);
        _worldScene!.SetUniqueIdFilterTile(cameraTileX, cameraTileY);

        ImGui.TextDisabled("Filter by UniqueId range. The 'Camera Tile' scope uses the tile the camera is currently in.");
        ImGui.Spacing();

        bool uniqueIdFilterEnabled = _worldScene!.UniqueIdFilterEnabled;
        if (ImGui.Checkbox("Filter UniqueId Range", ref uniqueIdFilterEnabled))
        {
            _worldScene.UniqueIdFilterEnabled = uniqueIdFilterEnabled;
            _settings.SaveViewerSettings();
        }

        ImGui.SameLine();
        UniqueIdVisibilityScope currentScope = _worldScene.UniqueIdVisibilityScope;
        string scopeLabel = currentScope == UniqueIdVisibilityScope.PerMap ? "Per-Map" : "Camera Tile";
        if (ImGui.BeginCombo("##UniqueIdScope", scopeLabel))
        {
            if (ImGui.Selectable("Per-Map", currentScope == UniqueIdVisibilityScope.PerMap))
            {
                _worldScene.UniqueIdVisibilityScope = UniqueIdVisibilityScope.PerMap;
                _archeologyScopeIndex = 0;
                _settings.SaveViewerSettings();
            }
            if (ImGui.Selectable("Camera Tile", currentScope == UniqueIdVisibilityScope.CameraTile))
            {
                _worldScene.UniqueIdVisibilityScope = UniqueIdVisibilityScope.CameraTile;
                _archeologyScopeIndex = 1;
                _settings.SaveViewerSettings();
            }
            ImGui.EndCombo();
        }

        // On world load, apply sticky range if set.
        if (_archeologyMinUniqueId >= 0 && _archeologyMaxUniqueId >= _archeologyMinUniqueId)
        {
            if (_worldScene.UniqueIdFilterMin != _archeologyMinUniqueId || _worldScene.UniqueIdFilterMax != _archeologyMaxUniqueId)
            {
                _worldScene.SetUniqueIdFilterRange(_archeologyMinUniqueId, _archeologyMaxUniqueId);
            }
        }

        if (_worldScene.TryGetUniqueIdFilterRange(out int minUniqueId, out int maxUniqueId, out int instanceCount))
        {
            ImGui.Spacing();
            int configuredMin = _worldScene.UniqueIdFilterMin;
            int configuredMax = _worldScene.UniqueIdFilterMax;
            int visibleMin = configuredMin >= minUniqueId ? Math.Min(configuredMin, maxUniqueId) : minUniqueId;
            int visibleMax = configuredMax >= minUniqueId ? Math.Max(configuredMin, configuredMax) : maxUniqueId;

            bool changed = false;
            if (ImGui.SliderInt("Visible Range Start", ref visibleMin, minUniqueId, maxUniqueId))
            {
                _worldScene.SetUniqueIdFilterRange(visibleMin, visibleMax);
                _worldScene.UniqueIdFilterEnabled = true;
                if (_archeologyPlaybackActive)
                    StopArcheologyPlayback(restoreRange: false);
                changed = true;
            }

            if (ImGui.SliderInt("Visible Range End", ref visibleMax, minUniqueId, maxUniqueId))
            {
                _worldScene.SetUniqueIdFilterRange(visibleMin, visibleMax);
                _worldScene.UniqueIdFilterEnabled = true;
                if (_archeologyPlaybackActive)
                    StopArcheologyPlayback(restoreRange: false);
                changed = true;
            }

            if (changed)
            {
                _archeologyMinUniqueId = visibleMin;
                _archeologyMaxUniqueId = visibleMax;
                _settings.SaveViewerSettings();
            }

            string status = _worldScene.UniqueIdFilterEnabled
                ? $"Scoped placements: {instanceCount}  Range: {minUniqueId}..{maxUniqueId}  Visible range: {visibleMin}..{visibleMax}"
                : $"Scoped placements: {instanceCount}  Range: {minUniqueId}..{maxUniqueId}  Selected visible range: {visibleMin}..{visibleMax} (filter off)";
            ImGui.TextDisabled(status);
        }
        else
        {
            ImGui.TextDisabled("No scoped placements with positive UniqueIds are currently available.");
        }

        if (ImGui.SmallButton("Reset UniqueId Filter"))
        {
            _worldScene.ResetUniqueIdFilter();
            _archeologyMinUniqueId = -1;
            _archeologyMaxUniqueId = -1;
            _settings.SaveViewerSettings();
        }
    }

    private void DrawArcheologyLayersSubTab()
    {
        ImGui.TextDisabled("Detected layers (consecutive gap analysis of uniqueId sequence).");
        ImGui.Spacing();

        IReadOnlyList<UniqueIdArchaeologyLayer> detectedLayers = _worldScene!.GetUniqueIdArchaeologyLayers();
        if (detectedLayers.Count == 0)
        {
            ImGui.TextDisabled("No UniqueId data available for the current scope.");
            return;
        }

        if (ImGui.BeginTable("##UniqueIdArcheologyLayers", 4, ImGuiTableFlags.Borders | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp | ImGuiTableFlags.ScrollX))
        {
            ImGui.TableSetupColumn("Layer", ImGuiTableColumnFlags.WidthFixed, 64f);
            ImGui.TableSetupColumn("Range", ImGuiTableColumnFlags.WidthFixed, 180f);
            ImGui.TableSetupColumn("Summary", ImGuiTableColumnFlags.WidthStretch);
            ImGui.TableSetupColumn("", ImGuiTableColumnFlags.WidthFixed, 80f);
            ImGui.TableHeadersRow();

            for (int i = 0; i < detectedLayers.Count; i++)
            {
                UniqueIdArchaeologyLayer layer = detectedLayers[i];
                ImGui.TableNextRow();
                ImGui.TableNextColumn();
                ImGui.TextUnformatted($"#{layer.LayerNumber}");
                ImGui.TableNextColumn();
                ImGui.TextUnformatted($"{layer.MinUniqueId}..{layer.MaxUniqueId}");
                ImGui.TableNextColumn();
                ImGui.TextUnformatted($"{layer.PlacementCount} placements ({layer.WmoCount} WMO, {layer.MdxCount} M2)");
                ImGui.TableNextColumn();
                if (ImGui.SmallButton($"Show##uid_layer_{i}"))
                {
                    _worldScene.SetUniqueIdFilterRange(layer.MinUniqueId, layer.MaxUniqueId);
                    _worldScene.UniqueIdFilterEnabled = true;
                }
            }
            ImGui.EndTable();
        }
    }

    private void DrawArcheologyPlaybackSubTab()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to use playback.");
            return;
        }

        ImGui.TextDisabled("Playback animates 'Visible Range End' from min to max at the configured speed.");
        ImGui.Spacing();

        // Capture speed reference
        if (ImGui.SliderFloat("Speed (uniqueIds/sec)", ref _archeologyPlaybackSpeed, 1f, 5000f, "%.0f"))
            _settings.SaveViewerSettings();

        ImGui.SameLine();
        if (ImGui.Checkbox("Loop", ref _archeologyPlaybackLoop))
            _settings.SaveViewerSettings();

        ImGui.Spacing();

        DrawArcheologyPlaybackTransportControls();

        ImGui.Spacing();

        // Status
        if (_worldScene.TryGetUniqueIdFilterRange(out int minId, out int maxId, out int count))
        {
            int currentMax = _worldScene.UniqueIdFilterMax;
            int remaining = Math.Max(0, maxId - currentMax);
            float secondsAtCurrentSpeed = _archeologyPlaybackSpeed > 0
                ? remaining / _archeologyPlaybackSpeed
                : float.PositiveInfinity;

            string status = _archeologyPlaybackActive
                ? $"Playing — end advancing at {_archeologyPlaybackSpeed:F0}/s. Remaining: {remaining} uniqueIds (~{secondsAtCurrentSpeed:F1}s)."
                : $"Stopped. End at {currentMax}, max {maxId}. Range: {minId}..{maxId}.";
            ImGui.TextDisabled(status);
        }
        else
        {
            ImGui.TextDisabled("No scoped placements to play.");
        }
    }

    private void DrawArcheologyPlaybackTransportControls()
    {
        if (_archeologyPlaybackActive)
        {
            if (ImGui.Button("Pause##archeology"))
            {
                _archeologyPlaybackActive = false;
                _statusMessage = "Archeology playback paused.";
            }
            ImGui.SameLine();
            if (ImGui.Button("Stop##archeology"))
            {
                StopArcheologyPlayback(restoreRange: true);
            }
        }
        else
        {
            if (ImGui.Button("Play##archeology"))
            {
                StartArcheologyPlayback();
            }
            ImGui.SameLine();
            if (ImGui.Button("Stop##archeology"))
            {
                StopArcheologyPlayback(restoreRange: true);
            }
        }
    }

    internal void StartArcheologyPlayback()
    {
        if (_worldScene == null) return;
        if (!_worldScene.TryGetUniqueIdFilterRange(out int minId, out int maxId, out _)) return;

        // Save current state so Stop can restore.
        _archeologyPlaybackRestoreMin = _worldScene.UniqueIdFilterMin;
        _archeologyPlaybackRestoreMax = _worldScene.UniqueIdFilterMax;
        _archeologyPlaybackRestoreFilter = _worldScene.UniqueIdFilterEnabled;
        _archeologyPlaybackAccumulator = 0;
        _archeologyPlaybackActive = true;
        _worldScene.UniqueIdFilterEnabled = true;
        _statusMessage = "Archeology playback started.";
    }

    internal void StopArcheologyPlayback(bool restoreRange)
    {
        _archeologyPlaybackActive = false;
        _archeologyPlaybackAccumulator = 0;
        if (restoreRange && _worldScene != null && _archeologyPlaybackRestoreMin >= 0)
        {
            _worldScene.SetUniqueIdFilterRange(_archeologyPlaybackRestoreMin, _archeologyPlaybackRestoreMax);
            _worldScene.UniqueIdFilterEnabled = _archeologyPlaybackRestoreFilter;
        }
        _archeologyPlaybackRestoreMin = -1;
        _archeologyPlaybackRestoreMax = -1;
        _statusMessage = "Archeology playback stopped.";
    }

    private void DrawArcheologyCaptureSubTab()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to use capture integration.");
            return;
        }
        ImGui.TextDisabled("Apply archeology playback to next capture / video recording.");
        ImGui.Spacing();
        ImGui.TextWrapped("Capture automation integration: when 'Apply to next capture' is enabled, the next capture batch will advance 'Visible Range End' per shot. When 'Apply to video recording' is enabled, the video recording session will start playback and capture progression at real-time speed.");
        ImGui.Spacing();

        bool applyToNextCapture = _archeologyApplyToNextCapture;
        if (ImGui.Checkbox("Apply to next capture", ref applyToNextCapture))
        {
            _archeologyApplyToNextCapture = applyToNextCapture;
            _settings.SaveViewerSettings();
        }

        bool applyToVideo = _archeologyApplyToVideoRecording;
        if (ImGui.Checkbox("Apply to video recording", ref applyToVideo))
        {
            _archeologyApplyToVideoRecording = applyToVideo;
            _settings.SaveViewerSettings();
        }

        if (_archeologyApplyToVideoRecording)
        {
            ImGui.SliderFloat("Video playback speed##archeology", ref _archeologyPlaybackSpeed, 1f, 5000f, "%.0f");
        }

        if (ImGui.Button("Apply playback to next capture"))
        {
            _archeologyApplyToNextCapture = true;
            _settings.SaveViewerSettings();
            _statusMessage = "Archeology playback will apply to the next queued capture.";
        }
        if (ImGui.IsItemHovered())
            ImGui.SetTooltip("Enables the same one-shot playback option used by Capture Automation.");

        ImGui.Spacing();
        ImGui.TextDisabled($"Next capture: {(_archeologyApplyToNextCapture ? "playback active" : "no playback")}");
        ImGui.TextDisabled($"Video recording: {(_archeologyApplyToVideoRecording ? $"playback @ {_archeologyPlaybackSpeed:F0}/s" : "no playback")}");
    }

    internal void DrawArchaeologyWorkbenchSubTabContent()
    {
        switch (_activeBottomTabIndex)
        {
            case 0:
                // New Weak Signal & Temporal Stratigraphy Tooling
                DrawTerrainControlsAdjustmentWeakSignalContent();
                break;
            case 1:
                // UniqueId Timeline Range
                if (_worldScene == null)
                {
                    ImGui.TextDisabled("Load a world to filter UniqueId range.");
                    return;
                }
                DrawArcheologyRangeSubTab();
                break;
            case 2:
                // Layers & Provenance
                if (_worldScene == null)
                {
                    ImGui.TextDisabled("Load a world to inspect archaeology layers.");
                    return;
                }
                DrawArcheologyLayersSubTab();
                break;
            case 3:
                // Playback & Capture
                if (_worldScene == null)
                {
                    ImGui.TextDisabled("Load a world to use playback & capture.");
                    return;
                }
                DrawArcheologyPlaybackSubTab();
                ImGui.Separator();
                DrawArcheologyCaptureSubTab();
                ImGui.Separator();
                ImGui.SeparatorText("Camera Paths & Video Capture");
                DrawCapturePanelContent();
                break;
            case 4:
                // PM4 Archaeological Analysis
                DrawPm4SubTabContent();
                break;
            case 5:
                // Cartography (Spec 222 integration)
                DrawArchaeologyCartographyContent();
                break;
            default:
                DrawTerrainControlsAdjustmentWeakSignalContent();
                break;
        }
    }

    /// <summary>
    /// Draws the tab strip for a nested sub-tab level and returns the selected index.
    /// The parent strip in DrawWorkbenchContent only covers the first level, so without this
    /// a nested section can never select anything but its parent's index.
    /// </summary>
    private int DrawNestedSubTabStrip(string id, string[] labels, int activeIndex)
    {
        if (labels.Length == 0)
            return 0;

        return DrawPageCombo(id, labels, activeIndex);
    }






    // ── PM4 sub-tab content ────────────────────────────────────────────────
    private void DrawPm4SubTabContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to use the PM4 tab.");
            return;
        }

        _activePm4TabIndex = DrawNestedSubTabStrip(
            "##Pm4SubTabs", WorkbenchNavigator.GetPm4BottomTabLabels(), _activePm4TabIndex);

        switch ((Pm4BottomTab)_activePm4TabIndex)
        {
            case Pm4BottomTab.Overlay:
                _pm4Workbench.DrawPm4OverlayWorkbenchContent();
                break;
            case Pm4BottomTab.Selection:
                _pm4Workbench.DrawPm4SelectionWorkbenchContent();
                break;
            case Pm4BottomTab.Correlation:
                _pm4Workbench.DrawPm4WmoCorrelationContent();
                break;
            case Pm4BottomTab.Info:
                _pm4Workbench.DrawPm4InfoPanelContent();
                break;
            case Pm4BottomTab.Reconcile:
                DrawReconciliationPanel();
                break;
            case Pm4BottomTab.Alignment:
                _pm4Workbench.DrawPm4AlignmentContent();
                break;
            case Pm4BottomTab.Outliner:
                _pm4Workbench.DrawPm4Outliner();
                break;
        }
    }

    // ── Archeology sub-tab content ──────────────────────────────────────────
    internal void DrawArcheologySubTabContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world to use the Archeology tab.");
            return;
        }

        if (_archeologyPlaybackActive)
        {
            ImGui.TextColored(new Vector4(0.95f, 0.75f, 0.25f, 1f), "Playback is active");
            ImGui.SameLine();
            DrawArcheologyPlaybackTransportControls();
            ImGui.Separator();
        }

        _activeArcheologyTabIndex = DrawNestedSubTabStrip(
            "##ArcheologySubTabs", WorkbenchNavigator.GetArcheologyBottomTabLabels(), _activeArcheologyTabIndex);

        switch ((ArcheologyBottomTab)_activeArcheologyTabIndex)
        {
            case ArcheologyBottomTab.Range:
                DrawArcheologyRangeSubTab();
                break;
            case ArcheologyBottomTab.Layers:
                DrawArcheologyLayersSubTab();
                break;
            case ArcheologyBottomTab.Playback:
                DrawArcheologyPlaybackSubTab();
                break;
            case ArcheologyBottomTab.Capture:
                DrawArcheologyCaptureSubTab();
                break;
            case ArcheologyBottomTab.Stratigraphy:
                DrawTemporalStratigraphySubTab();
                break;
        }
    }

    internal void DrawArchaeologyQuickSection()
    {
        ImGui.Spacing();
        ImGui.Text("Visual Investigation");
        ImGui.Separator();
        DrawVisualInvestigationToolbox(showWorldObjectRangeControls: false);

        ImGui.Spacing();
        ImGui.Text("Phase Layers");
        ImGui.Separator();
        int phaseCount = _terrainManager?.PhaseLayers.Count ?? 0;
        ImGui.TextDisabled(phaseCount == 0 ? "No active phase layers (base map only)." : $"{phaseCount} active phase layer(s) configured.");
        if (ImGui.Button("Manage Layers & Provenance##Quick"))
        {
            OpenWorkbenchTab(WorkbenchTab.Archaeology, 2);
        }

        ImGui.Spacing();
        ImGui.Text("Archaeological Launchers");
        ImGui.Separator();
        if (ImGui.Button("Cartography (Spec 222)##Quick"))
        {
            OpenWorkbenchTab(WorkbenchTab.Archaeology, 5);
        }
        ImGui.SameLine();
        if (ImGui.Button("UniqueId Timeline##Quick"))
        {
            OpenWorkbenchTab(WorkbenchTab.Archaeology, 1);
        }

        if (ImGui.Button("Synthesized Minimap Export...##Quick"))
        {
            PrepareSynthesizedMinimapExportDialogInputs();
            _showSynthesizedMinimapExportDialog = true;
        }
    }
}
