using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using MdxViewer.Logging;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Partial class containing PM4 alignment and viewer utility windows.
/// </summary>
public partial class ViewerApp
{
    private void OpenPm4Workbench(Pm4WorkbenchTab tab)
    {
        _showRightSidebar = true;
        _forceOpenPm4WorkbenchInspector = true;
        _pendingPm4WorkbenchTab = tab;
    }

    private void DrawPm4WorkbenchInspector()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("PM4 workbench becomes available once a world scene is loaded.");
            return;
        }

        ImGui.TextDisabled("Hover stays lightweight. Click a PM4 object to inspect its matches, graph, and correlation here.");
        ImGui.SetNextItemOpen(true, ImGuiCond.Once);
        DrawPm4GlossarySummary();

        if (!ImGui.BeginTabBar("##Pm4WorkbenchTabs"))
            return;

        ImGuiTabItemFlags overlayFlags = _pendingPm4WorkbenchTab == Pm4WorkbenchTab.Overlay
            ? ImGuiTabItemFlags.SetSelected
            : ImGuiTabItemFlags.None;
        bool overlayTabOpen = true;
        if (ImGui.BeginTabItem("Overlay", ref overlayTabOpen, overlayFlags))
        {
            DrawPm4OverlayWorkbenchContent();
            ImGui.EndTabItem();
        }

        ImGuiTabItemFlags selectionFlags = _pendingPm4WorkbenchTab == Pm4WorkbenchTab.Selection
            ? ImGuiTabItemFlags.SetSelected
            : ImGuiTabItemFlags.None;
        bool selectionTabOpen = true;
        if (ImGui.BeginTabItem("Selection", ref selectionTabOpen, selectionFlags))
        {
            DrawPm4SelectionWorkbenchContent();
            ImGui.EndTabItem();
        }

        ImGuiTabItemFlags correlationFlags = _pendingPm4WorkbenchTab == Pm4WorkbenchTab.Correlation
            ? ImGuiTabItemFlags.SetSelected
            : ImGuiTabItemFlags.None;
        bool correlationTabOpen = true;
        if (ImGui.BeginTabItem("Correlation", ref correlationTabOpen, correlationFlags))
        {
            DrawPm4CorrelationInspectorContent();
            ImGui.EndTabItem();
        }

        _pendingPm4WorkbenchTab = null;
        ImGui.EndTabBar();
    }

    private void DrawPm4GlossarySummary()
    {
        if (!ImGui.CollapsingHeader("PM4 Glossary / Evidence"))
            return;

        ImGui.TextWrapped("The PM4 workbench mixes raw chunk names, viewer aliases, and viewer-generated structure. Not every label here is a proven native PM4 field name.");
        ImGui.BulletText("CK24: viewer alias for the packed MSUR field at 0x1C. Type = high byte, ObjId = low 16 bits.");
        ImGui.BulletText("part / ObjectPartId: viewer-generated split id. MdxViewer assigns it during the current overlay build after CK24 grouping, dominant MSLK grouping, optional MDOS split, then optional connectivity split. It is not a raw PM4 field.");
        ImGui.BulletText("MSLK Group: dominant MSLK.GroupObjectId seen in the current viewer object. Strong grouping hint, not final proof of identity.");
        ImGui.BulletText("Linked MPRL refs: position-reference rows attached to the current viewer object or its dominant link family. Used as placement evidence.");
        ImGui.BulletText("Group / Attr / Mdos: dominant MSUR values across the currently selected viewer object. Useful for debugging, not guaranteed unique or authoritative.");
        ImGui.BulletText("PM4 Graph: the viewer's current decomposition of the selected object, not a literal raw node graph stored in PM4.");
        ImGui.BulletText("Match uid: nearby MODF/MDDF placement candidate id. It is not a PM4-native object id.");
    }

    private void DrawPm4OverlayWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        bool showPm4Overlay = _worldScene.ShowPm4Overlay;
        if (ImGui.Checkbox("PM4 Overlay", ref showPm4Overlay))
            _worldScene.ShowPm4Overlay = showPm4Overlay;

        ImGui.SameLine();
        if (ImGui.Button("Reload PM4"))
            _worldScene.ReloadPm4Overlay();

        ImGui.SameLine();
        if (ImGui.Button("Save Overlay Align"))
            SaveCurrentPm4Alignment();

        bool showPm4Solid = _worldScene.ShowPm4SolidOverlay;
        if (ImGui.Checkbox("PM4 Solid Fill", ref showPm4Solid))
            _worldScene.ShowPm4SolidOverlay = showPm4Solid;

        ImGui.SameLine();
        bool pm4IgnoreDepth = _worldScene.Pm4OverlayIgnoreDepth;
        if (ImGui.Checkbox("PM4 X-Ray", ref pm4IgnoreDepth))
            _worldScene.Pm4OverlayIgnoreDepth = pm4IgnoreDepth;

        ImGui.SameLine();
        bool showPm4Bounds = _worldScene.ShowPm4ObjectBounds;
        if (ImGui.Checkbox("PM4 Bounds", ref showPm4Bounds))
            _worldScene.ShowPm4ObjectBounds = showPm4Bounds;

        bool showPm4Refs = _worldScene.ShowPm4PositionRefs;
        if (ImGui.Checkbox("PM4 MPRL Refs", ref showPm4Refs))
            _worldScene.ShowPm4PositionRefs = showPm4Refs;

        ImGui.SameLine();
        bool showPm4Centroids = _worldScene.ShowPm4ObjectCentroids;
        if (ImGui.Checkbox("PM4 Centroids", ref showPm4Centroids))
            _worldScene.ShowPm4ObjectCentroids = showPm4Centroids;

        ImGui.SameLine();
        bool pm4FlipAllObjY = _worldScene.Pm4FlipAllObjectsY;
        if (ImGui.Checkbox("Mirror PM4 N/S", ref pm4FlipAllObjY))
            _worldScene.Pm4FlipAllObjectsY = pm4FlipAllObjY;

        bool showType40 = _worldScene.ShowPm4Type40;
        if (ImGui.Checkbox("CK24 0x40", ref showType40))
            _worldScene.ShowPm4Type40 = showType40;

        ImGui.SameLine();
        bool showType80 = _worldScene.ShowPm4Type80;
        if (ImGui.Checkbox("CK24 0x80", ref showType80))
            _worldScene.ShowPm4Type80 = showType80;

        ImGui.SameLine();
        bool showTypeOther = _worldScene.ShowPm4TypeOther;
        if (ImGui.Checkbox("CK24 Other", ref showTypeOther))
            _worldScene.ShowPm4TypeOther = showTypeOther;

        Pm4OverlayColorMode colorMode = _worldScene.Pm4ColorMode;
        if (ImGui.BeginCombo("PM4 Color", GetPm4ColorModeLabel(colorMode)))
        {
            foreach (Pm4OverlayColorMode mode in Enum.GetValues<Pm4OverlayColorMode>())
            {
                bool isSelected = mode == colorMode;
                if (ImGui.Selectable(GetPm4ColorModeLabel(mode), isSelected))
                    _worldScene.Pm4ColorMode = mode;
                if (isSelected)
                    ImGui.SetItemDefaultFocus();
            }

            ImGui.EndCombo();
        }

        bool splitCk24Connectivity = _worldScene.Pm4SplitCk24ByConnectivity;
        if (ImGui.Checkbox("Split CK24 by Connectivity", ref splitCk24Connectivity))
        {
            _worldScene.Pm4SplitCk24ByConnectivity = splitCk24Connectivity;
            _worldScene.ReloadPm4Overlay();
        }

        bool splitCk24ByMdos = _worldScene.Pm4SplitCk24ByMdos;
        if (ImGui.Checkbox("Split CK24 by MdosIndex", ref splitCk24ByMdos))
        {
            _worldScene.Pm4SplitCk24ByMdos = splitCk24ByMdos;
            _worldScene.ReloadPm4Overlay();
        }

        if (_worldScene.IsPm4Loading)
            ImGui.TextColored(new Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"PM4 loading... {_worldScene.Pm4Status}");
        else if (_worldScene.Pm4LoadAttempted)
            ImGui.TextDisabled($"PM4: {_worldScene.Pm4LoadedFiles}/{_worldScene.Pm4TotalFiles} files, {_worldScene.Pm4VisibleObjectCount}/{_worldScene.Pm4ObjectCount} objects, {_worldScene.Pm4VisibleLineCount}/{_worldScene.Pm4LineCount} lines, {_worldScene.Pm4VisibleTriangleCount}/{_worldScene.Pm4TriangleCount} tris, {_worldScene.Pm4VisiblePositionRefCount}/{_worldScene.Pm4PositionRefCount} refs");
        else
            ImGui.TextDisabled("Toggle PM4 Overlay to lazy-load navmesh debug data.");

        if (_worldScene.Pm4LoadAttempted)
            ImGui.TextDisabled($"Status: {_worldScene.Pm4Status}");

        ImGui.TextDisabled($"Overlay Align: T=({_worldScene.Pm4OverlayTranslation.X:F2}, {_worldScene.Pm4OverlayTranslation.Y:F2}, {_worldScene.Pm4OverlayTranslation.Z:F2}) Rot=({_worldScene.Pm4OverlayRotationDegrees.X:F2}, {_worldScene.Pm4OverlayRotationDegrees.Y:F2}, {_worldScene.Pm4OverlayRotationDegrees.Z:F2})° S=({_worldScene.Pm4OverlayScale.X:F3}, {_worldScene.Pm4OverlayScale.Y:F3}, {_worldScene.Pm4OverlayScale.Z:F3})");

        DrawPm4ColorLegend("WorkbenchOverlay");
    }

    private void DrawPm4SelectionWorkbenchContent()
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.HasSelectedPm4Object || !_worldScene.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to inspect one object at a time.");
            DrawPm4ObjectCollectionSummary("WorkbenchSelection");
            if (ImGui.Button("Dump PM4 Objects JSON"))
                ExportPm4ObjectsJson();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 OBJ Set"))
                ExportPm4ObjectsObjSet();
            return;
        }

        int requestedMatches = _pm4ObjectMatchMaxMatchesPerObject;
        ImGui.SetNextItemWidth(130f);
        if (ImGui.SliderInt("Top Matches", ref requestedMatches, 3, 5))
            _pm4ObjectMatchMaxMatchesPerObject = Math.Clamp(requestedMatches, 3, 5);

        ImGui.SameLine();
        if (ImGui.Button("Open Advanced Align"))
            _showPm4AlignmentWindow = true;

        ImGui.SameLine();
        if (ImGui.Button("Save Overlay Align"))
            SaveCurrentPm4Alignment();

        if (ImGui.CollapsingHeader("Selected PM4", ImGuiTreeNodeFlags.DefaultOpen))
        {
            var selectedPm4 = _worldScene.SelectedPm4ObjectKey.Value;
            ImGui.Text($"tile ({selectedPm4.tileX}, {selectedPm4.tileY}) CK24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart}");
            ImGui.TextDisabled("part = viewer-generated split id from the current overlay build, not a raw PM4 field.");

            if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
            {
                ImGui.TextDisabled($"Type=0x{debugInfo.Ck24Type:X2} ObjId={debugInfo.Ck24ObjectId} Surfaces={debugInfo.SurfaceCount}");
                ImGui.TextDisabled($"Group=0x{debugInfo.DominantGroupKey:X2} Attr=0x{debugInfo.DominantAttributeMask:X2} Mdos={debugInfo.DominantMdosIndex} AvgH={debugInfo.AverageSurfaceHeight:F2}");
                ImGui.TextDisabled($"MSLKGroup=0x{debugInfo.LinkGroupObjectId:X8} Linked MPRL refs={debugInfo.LinkedPositionRefCount}");
            }

            ImGui.TextDisabled($"Tile layer align: T=({_worldScene.SelectedPm4Ck24LayerTranslation.X:F2}, {_worldScene.SelectedPm4Ck24LayerTranslation.Y:F2}, {_worldScene.SelectedPm4Ck24LayerTranslation.Z:F2}) Rot=({_worldScene.SelectedPm4Ck24LayerRotationDegrees.X:F2}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Y:F2}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Z:F2})° S=({_worldScene.SelectedPm4Ck24LayerScale.X:F3}, {_worldScene.SelectedPm4Ck24LayerScale.Y:F3}, {_worldScene.SelectedPm4Ck24LayerScale.Z:F3})");
            ImGui.TextDisabled($"Object align: T=({_worldScene.SelectedPm4ObjectTranslation.X:F2}, {_worldScene.SelectedPm4ObjectTranslation.Y:F2}, {_worldScene.SelectedPm4ObjectTranslation.Z:F2}) Rot=({_worldScene.SelectedPm4ObjectRotationDegrees.X:F2}, {_worldScene.SelectedPm4ObjectRotationDegrees.Y:F2}, {_worldScene.SelectedPm4ObjectRotationDegrees.Z:F2})° S=({_worldScene.SelectedPm4ObjectScale.X:F3}, {_worldScene.SelectedPm4ObjectScale.Y:F3}, {_worldScene.SelectedPm4ObjectScale.Z:F3})");

            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.ClearPm4ObjectSelection();
            ImGui.SameLine();
            if (ImGui.Button("Dump PM4 Objects JSON"))
                ExportPm4ObjectsJson();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 OBJ Set"))
                ExportPm4ObjectsObjSet();
        }

        if (ImGui.CollapsingHeader("Match Suggestions", ImGuiTreeNodeFlags.DefaultOpen))
            DrawPm4SelectedObjectMatchSuggestions("WorkbenchSelectedPm4", compact: false);

        DrawSelectedPm4ObjectGraph("WorkbenchSelectedObject");
    }

    private void DrawPm4CorrelationInspectorContent()
    {
        if (_worldScene == null)
            return;

        EnsurePm4WmoCorrelationReportLoaded();

        int requestedMatches = _pm4WmoCorrelationMaxMatchesPerPlacement;
        ImGui.SetNextItemWidth(90f);
        if (ImGui.InputInt("Max Matches", ref requestedMatches))
        {
            _pm4WmoCorrelationMaxMatchesPerPlacement = Math.Clamp(requestedMatches, 1, 32);
            RefreshPm4WmoCorrelationReport();
        }

        ImGui.SameLine();
        if (ImGui.Button("Refresh"))
            RefreshPm4WmoCorrelationReport();

        ImGui.SameLine();
        if (ImGui.Button("Dump JSON"))
            ExportPm4WmoCorrelationJson();

        ImGui.SameLine();
        if (ImGui.Checkbox("Only Near", ref _pm4WmoCorrelationNearOnly))
        {
            if (_selectedPm4WmoCorrelationPlacementIndex >= 0)
                _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        ImGui.SetNextItemWidth(-1f);
        ImGui.InputTextWithHint("##Pm4WmoCorrelationFilterWorkbench", "Filter model name or path", ref _pm4WmoCorrelationModelFilter, 256);

        if (_pm4WmoCorrelationReport == null)
        {
            ImGui.TextDisabled("No PM4/WMO correlation report is loaded.");
            return;
        }

        Pm4WmoCorrelationReport report = _pm4WmoCorrelationReport;
        ImGui.TextDisabled($"Generated {report.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss} UTC | placements {report.Summary.WmoPlacementCount}, resolved WMO meshes {report.Summary.WmoMeshResolvedCount}, PM4 objects {report.Summary.Pm4ObjectCount}");
        ImGui.TextDisabled($"Candidates {report.Summary.PlacementsWithCandidates}/{report.Summary.WmoPlacementCount}, near {report.Summary.PlacementsWithNearCandidates}, PM4 status: {report.Pm4Status}");

        string filter = _pm4WmoCorrelationModelFilter.Trim();
        var filteredPlacements = report.Placements
            .Select((placement, index) => new { placement, index })
            .Where(entry => !_pm4WmoCorrelationNearOnly || entry.placement.Pm4NearCandidateCount > 0)
            .Where(entry => string.IsNullOrWhiteSpace(filter)
                || entry.placement.ModelName.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelPath.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelKey.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(entry => entry.placement.Pm4Matches.Count > 0 ? entry.placement.Pm4Matches[0].FootprintOverlapRatio : 0f)
            .ThenBy(entry => entry.placement.ModelName, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (filteredPlacements.Count == 0)
        {
            ImGui.TextDisabled("No placements matched the current filter.");
            return;
        }

        if (!filteredPlacements.Any(entry => entry.index == _selectedPm4WmoCorrelationPlacementIndex))
        {
            _selectedPm4WmoCorrelationPlacementIndex = filteredPlacements[0].index;
            _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        float leftWidth = MathF.Min(360f, ImGui.GetContentRegionAvail().X * 0.44f);
        if (ImGui.BeginChild("##Pm4WmoPlacementListWorkbench", new Vector2(leftWidth, 360f), true))
        {
            for (int i = 0; i < filteredPlacements.Count; i++)
            {
                var entry = filteredPlacements[i];
                Pm4WmoCorrelationPlacement placement = entry.placement;
                bool selected = entry.index == _selectedPm4WmoCorrelationPlacementIndex;
                string label = $"[{placement.TileX},{placement.TileY}] {placement.ModelName}##Pm4WmoPlacementWorkbench{entry.index}";
                if (ImGui.Selectable(label, selected))
                {
                    _selectedPm4WmoCorrelationPlacementIndex = entry.index;
                    _selectedPm4WmoCorrelationMatchIndex = 0;
                }

                if (placement.Pm4Matches.Count > 0)
                {
                    Pm4WmoCorrelationMatch best = placement.Pm4Matches[0];
                    ImGui.TextDisabled($"best CK24=0x{best.Ck24:X6} part={best.ObjectPartId} overlap={best.FootprintOverlapRatio:F2} dist={best.FootprintDistance:F1}");
                }
                else
                {
                    ImGui.TextDisabled("No PM4 candidates in the current tile neighborhood.");
                }

                ImGui.Separator();
            }
        }
        ImGui.EndChild();

        ImGui.SameLine();

        if (ImGui.BeginChild("##Pm4WmoPlacementDetailsWorkbench", Vector2.Zero, true))
        {
            Pm4WmoCorrelationPlacement placement = report.Placements[_selectedPm4WmoCorrelationPlacementIndex];
            ImGui.TextWrapped($"{placement.ModelName} (tile {placement.TileX},{placement.TileY}, uid {placement.UniqueId})");
            ImGui.TextDisabled(placement.ModelPath);

            if (placement.Pm4Matches.Count > 0)
            {
                Pm4WmoCorrelationMatch selectedMatch = placement.Pm4Matches[Math.Clamp(_selectedPm4WmoCorrelationMatchIndex, 0, placement.Pm4Matches.Count - 1)];

                if (ImGui.Button("Select PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);

                ImGui.SameLine();
                if (ImGui.Button("Frame PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.Button("Frame Pair"))
                {
                    Vector3 boundsMin = Vector3.Min(placement.WorldBoundsMin, selectedMatch.BoundsMin);
                    Vector3 boundsMax = Vector3.Max(placement.WorldBoundsMax, selectedMatch.BoundsMax);
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);
                    FocusCameraOnBounds(boundsMin, boundsMax);
                }
            }

            ImGui.Separator();
            ImGui.TextDisabled($"Placement pos: ({placement.PlacementPosition.X:F2}, {placement.PlacementPosition.Y:F2}, {placement.PlacementPosition.Z:F2})");
            ImGui.TextDisabled($"World bounds min: ({placement.WorldBoundsMin.X:F2}, {placement.WorldBoundsMin.Y:F2}, {placement.WorldBoundsMin.Z:F2})");
            ImGui.TextDisabled($"World bounds max: ({placement.WorldBoundsMax.X:F2}, {placement.WorldBoundsMax.Y:F2}, {placement.WorldBoundsMax.Z:F2})");

            ImGui.Separator();
            ImGui.Text($"PM4 matches ({placement.Pm4Matches.Count}/{placement.Pm4CandidateCount} shown, near={placement.Pm4NearCandidateCount})");

            for (int matchIndex = 0; matchIndex < placement.Pm4Matches.Count; matchIndex++)
            {
                Pm4WmoCorrelationMatch match = placement.Pm4Matches[matchIndex];
                bool selected = matchIndex == _selectedPm4WmoCorrelationMatchIndex;
                string label = $"CK24 0x{match.Ck24:X6} part {match.ObjectPartId}##Pm4WmoMatchWorkbench{matchIndex}";
                if (ImGui.Selectable(label, selected))
                    _selectedPm4WmoCorrelationMatchIndex = matchIndex;

                ImGui.TextDisabled($"footprint overlap={match.FootprintOverlapRatio:F3} area={match.FootprintAreaRatio:F3} dist={match.FootprintDistance:F2}");
                ImGui.TextDisabled($"planar gap={match.PlanarGap:F2} vertical gap={match.VerticalGap:F2} center={match.CenterDistance:F2}");
                ImGui.Separator();
            }
        }
        ImGui.EndChild();
    }

    private void DrawPerfWindow()
    {
        ImGui.SetNextWindowSize(new Vector2(360, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("Perf", ref _showPerfWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        var terrainRenderer = _terrainManager?.Renderer ?? _vlmTerrainManager?.Renderer;
        if (terrainRenderer == null)
        {
            ImGui.Text("No terrain loaded.");
            ImGui.End();
            return;
        }

        ImGui.Text($"Chunks: {terrainRenderer.ChunksRendered} rendered, {terrainRenderer.ChunksCulled} culled");
        ImGui.TextDisabled("Stats are for the last terrain Render() call.");

        ImGui.End();
    }

    private void DrawPm4AlignmentWindow()
    {
        if (_worldScene == null)
        {
            _showPm4AlignmentWindow = false;
            return;
        }

        ImGui.SetNextWindowSize(new Vector2(430, 0), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("PM4 Alignment", ref _showPm4AlignmentWindow, ImGuiWindowFlags.AlwaysAutoResize))
        {
            ImGui.End();
            return;
        }

        ImGui.TextWrapped("PM4 alignment is now tile-local for the selected CK24 bucket plus object-local for the selected part. Select one PM4 object, then adjust the tile CK24 block or the object block.");
        ImGui.TextDisabled("Global PM4 overlay transforms are no longer edited in this window.");
        ImGui.TextDisabled("Use Overlay > Flip All Obj Y for map-wide Y mirror correction.");

        ImGui.Text("Translation Step:");
        if (ImGui.RadioButton("0.5u", MathF.Abs(_pm4TranslationStepUnits - 0.5f) < 0.001f))
            _pm4TranslationStepUnits = 0.5f;
        ImGui.SameLine();
        if (ImGui.RadioButton("1u", MathF.Abs(_pm4TranslationStepUnits - 1f) < 0.001f))
            _pm4TranslationStepUnits = 1f;
        ImGui.SameLine();
        if (ImGui.RadioButton("10u", MathF.Abs(_pm4TranslationStepUnits - 10f) < 0.001f))
            _pm4TranslationStepUnits = 10f;
        ImGui.SameLine();
        if (ImGui.RadioButton("100u", MathF.Abs(_pm4TranslationStepUnits - 100f) < 0.001f))
            _pm4TranslationStepUnits = 100f;
        ImGui.SameLine();
        if (ImGui.RadioButton("533.333u", MathF.Abs(_pm4TranslationStepUnits - 533.3333f) < 0.01f))
            _pm4TranslationStepUnits = 533.3333f;

        ImGui.Text("Rotation Step:");
        if (ImGui.RadioButton("1 deg", MathF.Abs(_pm4RotationStepDegrees - 1f) < 0.001f))
            _pm4RotationStepDegrees = 1f;
        ImGui.SameLine();
        if (ImGui.RadioButton("5 deg", MathF.Abs(_pm4RotationStepDegrees - 5f) < 0.001f))
            _pm4RotationStepDegrees = 5f;
        ImGui.SameLine();
        if (ImGui.RadioButton("15 deg", MathF.Abs(_pm4RotationStepDegrees - 15f) < 0.001f))
            _pm4RotationStepDegrees = 15f;
        ImGui.SameLine();
        if (ImGui.RadioButton("45 deg", MathF.Abs(_pm4RotationStepDegrees - 45f) < 0.001f))
            _pm4RotationStepDegrees = 45f;
        ImGui.SameLine();
        if (ImGui.RadioButton("90 deg", MathF.Abs(_pm4RotationStepDegrees - 90f) < 0.001f))
            _pm4RotationStepDegrees = 90f;

        ImGui.Text("Scale Step:");
        if (ImGui.RadioButton("0.01", MathF.Abs(_pm4ScaleStepUnits - 0.01f) < 0.0001f))
            _pm4ScaleStepUnits = 0.01f;
        ImGui.SameLine();
        if (ImGui.RadioButton("0.1", MathF.Abs(_pm4ScaleStepUnits - 0.1f) < 0.0001f))
            _pm4ScaleStepUnits = 0.1f;
        ImGui.SameLine();
        if (ImGui.RadioButton("0.25", MathF.Abs(_pm4ScaleStepUnits - 0.25f) < 0.0001f))
            _pm4ScaleStepUnits = 0.25f;
        ImGui.SameLine();
        if (ImGui.RadioButton("1.0", MathF.Abs(_pm4ScaleStepUnits - 1f) < 0.0001f))
            _pm4ScaleStepUnits = 1f;

        ImGui.Separator();

        if (!_worldScene.HasSelectedPm4Object || !_worldScene.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to pick an object.");
            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.ClearPm4ObjectSelection();
            ImGui.SameLine();
            if (ImGui.Button("Dump PM4 Objects JSON"))
                ExportPm4ObjectsJson();
            ImGui.SameLine();
            if (ImGui.Button("Export PM4 OBJ Set"))
                ExportPm4ObjectsObjSet();
            ImGui.SameLine();
            if (ImGui.Button("PM4 Object Match"))
            {
                _showPm4ObjectMatchWindow = true;
                EnsurePm4ObjectMatchReportLoaded();
            }
            ImGui.SameLine();
            if (ImGui.Button("Dump PM4/WMO Correlation JSON"))
                ExportPm4WmoCorrelationJson();
            ImGui.SameLine();
            if (ImGui.Button("PM4/WMO Panel"))
            {
                _showPm4WmoCorrelationWindow = true;
                EnsurePm4WmoCorrelationReportLoaded();
            }
            ImGui.End();
            return;
        }

        var selectedPm4 = _worldScene.SelectedPm4ObjectKey.Value;
        uint? selectedLayerCk24 = _worldScene.SelectedPm4RawCk24;
        Vector3 selectedObjectTranslation = _worldScene.SelectedPm4ObjectTranslation;
        Vector3 selectedObjectRotation = _worldScene.SelectedPm4ObjectRotationDegrees;
        Vector3 selectedObjectScale = _worldScene.SelectedPm4ObjectScale;
        Vector3 selectedLayerTranslation = _worldScene.SelectedPm4Ck24LayerTranslation;
        Vector3 selectedLayerRotation = _worldScene.SelectedPm4Ck24LayerRotationDegrees;
        Vector3 selectedLayerScale = _worldScene.SelectedPm4Ck24LayerScale;
        bool translationChanged = false;
        bool rotationChanged = false;
        bool scaleChanged = false;
        bool layerTranslationChanged = false;
        bool layerRotationChanged = false;
        bool layerScaleChanged = false;

        ImGui.Text($"Selected: tile ({selectedPm4.tileX}, {selectedPm4.tileY}) CK24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart}");
        if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            ImGui.TextDisabled($"Type=0x{debugInfo.Ck24Type:X2} ObjId={debugInfo.Ck24ObjectId} Surfaces={debugInfo.SurfaceCount}");
            ImGui.TextDisabled($"Group=0x{debugInfo.DominantGroupKey:X2} Attr=0x{debugInfo.DominantAttributeMask:X2} Mdos={debugInfo.DominantMdosIndex} AvgH={debugInfo.AverageSurfaceHeight:F2}");
            ImGui.TextDisabled($"Part={debugInfo.ObjectPartId} MSLKGroup=0x{debugInfo.LinkGroupObjectId:X8}");
            ImGui.TextDisabled($"Linked MPRL refs={debugInfo.LinkedPositionRefCount}");
            if (debugInfo.LinkedPositionRefSummary.TotalCount > 0)
            {
                if (debugInfo.LinkedPositionRefSummary.HasNormalHeadings)
                {
                    ImGui.TextDisabled(
                        $"MPRL normal={debugInfo.LinkedPositionRefSummary.NormalCount} term={debugInfo.LinkedPositionRefSummary.TerminatorCount} floors={debugInfo.LinkedPositionRefSummary.FloorMin}..{debugInfo.LinkedPositionRefSummary.FloorMax}");
                    ImGui.TextDisabled(
                        $"MPRL heading={debugInfo.LinkedPositionRefSummary.HeadingMinDegrees:F2}..{debugInfo.LinkedPositionRefSummary.HeadingMaxDegrees:F2} mean={debugInfo.LinkedPositionRefSummary.HeadingMeanDegrees:F2} deg");
                }
                else
                {
                    ImGui.TextDisabled(
                        $"MPRL normal={debugInfo.LinkedPositionRefSummary.NormalCount} term={debugInfo.LinkedPositionRefSummary.TerminatorCount}");
                }
            }
            ImGui.TextDisabled($"Planar: swap={debugInfo.SwapPlanarAxes} invertU={debugInfo.InvertU} invertV={debugInfo.InvertV} windingFlip={debugInfo.InvertsWinding}");
        }

        if (selectedLayerCk24.HasValue && _worldScene.TryGetSelectedPm4Ck24LayerStats(out int layerTileCount, out int layerObjectCount))
            ImGui.TextDisabled($"Tile CK24 0x{selectedLayerCk24.Value:X6} on ({selectedPm4.tileX}, {selectedPm4.tileY}): {layerObjectCount} parts across {layerTileCount} tile");

        if (_worldScene.TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo researchInfo)
            && ImGui.CollapsingHeader("PM4 Research", ImGuiTreeNodeFlags.DefaultOpen))
        {
            ImGui.TextDisabled($"Source: {Path.GetFileName(researchInfo.SourcePath)}");
            ImGui.TextDisabled($"v{researchInfo.Version} MSLK={researchInfo.MslkCount} MSUR={researchInfo.MsurCount} MSCN={researchInfo.MscnCount} MPRL={researchInfo.MprlCount}");
            ImGui.TextDisabled($"RefIndex mismatches={researchInfo.InvalidRefIndexCount} diagnostics={researchInfo.DiagnosticCount} hypotheses={researchInfo.MatchingCk24HypothesisCount}/{researchInfo.TotalHypothesisCount}");

            if (researchInfo.Diagnostics.Count > 0)
            {
                for (int i = 0; i < researchInfo.Diagnostics.Count; i++)
                    ImGui.TextDisabled($"diag: {researchInfo.Diagnostics[i]}");
            }

            if (researchInfo.TopMatches.Count == 0)
            {
                ImGui.TextDisabled("No raw PM4 hypotheses matched the selected CK24.");
            }
            else
            {
                ImGui.Text("Top raw hypotheses:");
                for (int i = 0; i < researchInfo.TopMatches.Count; i++)
                {
                    Pm4ResearchHypothesisMatch match = researchInfo.TopMatches[i];
                    string headingText = match.MprlHeadingMeanDegrees.HasValue
                        ? $" heading={match.MprlHeadingMeanDegrees.Value:F1} delta={match.HeadingDeltaDegrees?.ToString("F1") ?? "n/a"}"
                        : string.Empty;
                    ImGui.BulletText($"{match.Family}#{match.FamilyObjectIndex} score={match.SimilarityScore:F2} surfaces={match.SurfaceCount} indices={match.TotalIndexCount} mdos={match.MdosCount} groups={match.GroupKeyCount} linkGroups={match.LinkGroupCount} dominant=0x{match.DominantLinkGroupObjectId:X} mode={match.CoordinateMode} planar=(swap={match.PlanarTransform.SwapPlanarAxes},u={match.PlanarTransform.InvertU},v={match.PlanarTransform.InvertV}) yaw={match.FrameYawDegrees:F1}{headingText} linkedMPRL={match.LinkedMprlRefCount}/{match.LinkedMprlInBoundsCount}");
                }
            }
        }

        ImGui.Separator();
        ImGui.Text("Tile CK24 Translation:");

        if (ImGui.Button("Layer X <<"))
        {
            selectedLayerTranslation.X -= _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer X >>"))
        {
            selectedLayerTranslation.X += _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }

        if (ImGui.Button("Layer Y <<"))
        {
            selectedLayerTranslation.Y -= _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Y >>"))
        {
            selectedLayerTranslation.Y += _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }

        if (ImGui.Button("Layer Z <<"))
        {
            selectedLayerTranslation.Z -= _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Z >>"))
        {
            selectedLayerTranslation.Z += _pm4TranslationStepUnits;
            layerTranslationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Tile CK24 Rotation:");

        if (ImGui.Button("Layer Rot X -"))
        {
            selectedLayerRotation.X -= _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot X +"))
        {
            selectedLayerRotation.X += _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }

        if (ImGui.Button("Layer Rot Y -"))
        {
            selectedLayerRotation.Y -= _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Y +"))
        {
            selectedLayerRotation.Y += _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Y +180"))
        {
            selectedLayerRotation.Y += 180f;
            layerRotationChanged = true;
        }

        if (ImGui.Button("Layer Rot Z -"))
        {
            selectedLayerRotation.Z -= _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Z +"))
        {
            selectedLayerRotation.Z += _pm4RotationStepDegrees;
            layerRotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Rot Z +180"))
        {
            selectedLayerRotation.Z += 180f;
            layerRotationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Tile CK24 Scale:");

        if (ImGui.Button("Layer Sx -"))
        {
            selectedLayerScale.X -= _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Sx +"))
        {
            selectedLayerScale.X += _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }

        if (ImGui.Button("Layer Sy -"))
        {
            selectedLayerScale.Y -= _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Sy +"))
        {
            selectedLayerScale.Y += _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }

        if (ImGui.Button("Layer Sz -"))
        {
            selectedLayerScale.Z -= _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Layer Sz +"))
        {
            selectedLayerScale.Z += _pm4ScaleStepUnits;
            layerScaleChanged = true;
        }

        ImGui.Text("Tile CK24 Axis Flips:");
        if (ImGui.Button("Flip Layer X"))
        {
            selectedLayerScale.X = -selectedLayerScale.X;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Layer Y"))
        {
            selectedLayerScale.Y = -selectedLayerScale.Y;
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Layer Z"))
        {
            selectedLayerScale.Z = -selectedLayerScale.Z;
            layerScaleChanged = true;
        }

        ImGui.Text("Tile CK24 Winding:");
        if (ImGui.Button("Wind Tile X"))
        {
            selectedLayerScale.X = ToggleWindingComponent(selectedLayerScale.X);
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Tile Y"))
        {
            selectedLayerScale.Y = ToggleWindingComponent(selectedLayerScale.Y);
            layerScaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Tile Z"))
        {
            selectedLayerScale.Z = ToggleWindingComponent(selectedLayerScale.Z);
            layerScaleChanged = true;
        }

        bool pm4TransformChanged = false;

        if (layerTranslationChanged)
        {
            _worldScene.SelectedPm4Ck24LayerTranslation = selectedLayerTranslation;
            pm4TransformChanged = true;
        }
        if (layerRotationChanged)
        {
            _worldScene.SelectedPm4Ck24LayerRotationDegrees = NormalizeRotationDegrees(selectedLayerRotation);
            pm4TransformChanged = true;
        }
        if (layerScaleChanged)
        {
            _worldScene.SelectedPm4Ck24LayerScale = selectedLayerScale;
            pm4TransformChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Object Translation:");

        if (ImGui.Button("Obj X <<"))
        {
            selectedObjectTranslation.X -= _pm4TranslationStepUnits;
            translationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj X >>"))
        {
            selectedObjectTranslation.X += _pm4TranslationStepUnits;
            translationChanged = true;
        }

        if (ImGui.Button("Obj Y <<"))
        {
            selectedObjectTranslation.Y -= _pm4TranslationStepUnits;
            translationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Y >>"))
        {
            selectedObjectTranslation.Y += _pm4TranslationStepUnits;
            translationChanged = true;
        }

        if (ImGui.Button("Obj Z <<"))
        {
            selectedObjectTranslation.Z -= _pm4TranslationStepUnits;
            translationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Z >>"))
        {
            selectedObjectTranslation.Z += _pm4TranslationStepUnits;
            translationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Object Rotation:");

        if (ImGui.Button("Obj Rot X -"))
        {
            selectedObjectRotation.X -= _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot X +"))
        {
            selectedObjectRotation.X += _pm4RotationStepDegrees;
            rotationChanged = true;
        }

        if (ImGui.Button("Obj Rot Y -"))
        {
            selectedObjectRotation.Y -= _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Y +"))
        {
            selectedObjectRotation.Y += _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Y +180"))
        {
            selectedObjectRotation.Y += 180f;
            rotationChanged = true;
        }

        if (ImGui.Button("Obj Rot Z -"))
        {
            selectedObjectRotation.Z -= _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Z +"))
        {
            selectedObjectRotation.Z += _pm4RotationStepDegrees;
            rotationChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Rot Z +180"))
        {
            selectedObjectRotation.Z += 180f;
            rotationChanged = true;
        }

        ImGui.Separator();
        ImGui.Text("Object Scale:");

        if (ImGui.Button("Obj Sx -"))
        {
            selectedObjectScale.X -= _pm4ScaleStepUnits;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Sx +"))
        {
            selectedObjectScale.X += _pm4ScaleStepUnits;
            scaleChanged = true;
        }

        if (ImGui.Button("Obj Sy -"))
        {
            selectedObjectScale.Y -= _pm4ScaleStepUnits;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Sy +"))
        {
            selectedObjectScale.Y += _pm4ScaleStepUnits;
            scaleChanged = true;
        }

        if (ImGui.Button("Obj Sz -"))
        {
            selectedObjectScale.Z -= _pm4ScaleStepUnits;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Obj Sz +"))
        {
            selectedObjectScale.Z += _pm4ScaleStepUnits;
            scaleChanged = true;
        }

        ImGui.Text("Object Axis Flips:");
        if (ImGui.Button("Flip Obj X"))
        {
            selectedObjectScale.X = -selectedObjectScale.X;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Obj Y"))
        {
            selectedObjectScale.Y = -selectedObjectScale.Y;
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Flip Obj Z"))
        {
            selectedObjectScale.Z = -selectedObjectScale.Z;
            scaleChanged = true;
        }

        ImGui.Text("Object Winding:");
        if (ImGui.Button("Wind Obj X"))
        {
            selectedObjectScale.X = ToggleWindingComponent(selectedObjectScale.X);
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Obj Y"))
        {
            selectedObjectScale.Y = ToggleWindingComponent(selectedObjectScale.Y);
            scaleChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Wind Obj Z"))
        {
            selectedObjectScale.Z = ToggleWindingComponent(selectedObjectScale.Z);
            scaleChanged = true;
        }

        if (translationChanged)
        {
            _worldScene.SelectedPm4ObjectTranslation = selectedObjectTranslation;
            pm4TransformChanged = true;
        }
        if (rotationChanged)
        {
            _worldScene.SelectedPm4ObjectRotationDegrees = NormalizeRotationDegrees(selectedObjectRotation);
            pm4TransformChanged = true;
        }
        if (scaleChanged)
        {
            _worldScene.SelectedPm4ObjectScale = selectedObjectScale;
            pm4TransformChanged = true;
        }

        ImGui.Separator();

        if (ImGui.Button("Reset Layer Move"))
        {
            _worldScene.SelectedPm4Ck24LayerTranslation = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Layer Rot"))
        {
            _worldScene.SelectedPm4Ck24LayerRotationDegrees = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Layer Scale"))
        {
            _worldScene.SelectedPm4Ck24LayerScale = Vector3.One;
            pm4TransformChanged = true;
        }

        if (ImGui.Button("Reset Layer 9DoF"))
        {
            _worldScene.SelectedPm4Ck24LayerTranslation = Vector3.Zero;
            _worldScene.SelectedPm4Ck24LayerRotationDegrees = Vector3.Zero;
            _worldScene.SelectedPm4Ck24LayerScale = Vector3.One;
            pm4TransformChanged = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Print Layer Alignment") && selectedLayerCk24.HasValue)
        {
            Vector3 t = _worldScene.SelectedPm4Ck24LayerTranslation;
            Vector3 r = _worldScene.SelectedPm4Ck24LayerRotationDegrees;
            Vector3 s = _worldScene.SelectedPm4Ck24LayerScale;
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Tile CK24 Align] tile=({selectedPm4.tileX},{selectedPm4.tileY}) ck24=0x{selectedLayerCk24.Value:X6} T=({t.X:F3},{t.Y:F3},{t.Z:F3}) Rot=({r.X:F3},{r.Y:F3},{r.Z:F3}) Scale=({s.X:F4},{s.Y:F4},{s.Z:F4})");
        }

            ImGui.TextDisabled($"Tile Move: ({_worldScene.SelectedPm4Ck24LayerTranslation.X:F3}, {_worldScene.SelectedPm4Ck24LayerTranslation.Y:F3}, {_worldScene.SelectedPm4Ck24LayerTranslation.Z:F3})");
            ImGui.TextDisabled($"Tile Rot: ({_worldScene.SelectedPm4Ck24LayerRotationDegrees.X:F3}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Y:F3}, {_worldScene.SelectedPm4Ck24LayerRotationDegrees.Z:F3}) deg");
            ImGui.TextDisabled($"Tile Scale: ({_worldScene.SelectedPm4Ck24LayerScale.X:F4}, {_worldScene.SelectedPm4Ck24LayerScale.Y:F4}, {_worldScene.SelectedPm4Ck24LayerScale.Z:F4})");

        ImGui.Separator();

        if (ImGui.Button("Reset Obj Move"))
        {
            _worldScene.SelectedPm4ObjectTranslation = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Rot"))
        {
            _worldScene.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Scale"))
        {
            _worldScene.SelectedPm4ObjectScale = Vector3.One;
            pm4TransformChanged = true;
        }

        if (ImGui.Button("Reset Obj 9DoF"))
        {
            _worldScene.SelectedPm4ObjectTranslation = Vector3.Zero;
            _worldScene.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            _worldScene.SelectedPm4ObjectScale = Vector3.One;
            pm4TransformChanged = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Clear PM4 Selection"))
            _worldScene.ClearPm4ObjectSelection();

        if (pm4TransformChanged)
            InvalidatePm4DerivedReports();

        if (ImGui.Button("Dump PM4 Objects JSON"))
            ExportPm4ObjectsJson();
        ImGui.SameLine();
        if (ImGui.Button("Export PM4 OBJ Set"))
            ExportPm4ObjectsObjSet();
        ImGui.SameLine();
        if (ImGui.Button("PM4 Object Match"))
        {
            _showPm4ObjectMatchWindow = true;
            EnsurePm4ObjectMatchReportLoaded();
        }
        ImGui.SameLine();
        if (ImGui.Button("Dump PM4/WMO Correlation JSON"))
            ExportPm4WmoCorrelationJson();
        ImGui.SameLine();
        if (ImGui.Button("PM4/WMO Panel"))
        {
            _showPm4WmoCorrelationWindow = true;
            EnsurePm4WmoCorrelationReportLoaded();
        }
        ImGui.SameLine();
        if (ImGui.Button("Print Obj Alignment"))
        {
            Vector3 t = _worldScene.SelectedPm4ObjectTranslation;
            Vector3 r = _worldScene.SelectedPm4ObjectRotationDegrees;
            Vector3 s = _worldScene.SelectedPm4ObjectScale;
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Obj Align] tile=({selectedPm4.tileX},{selectedPm4.tileY}) ck24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart} T=({t.X:F3},{t.Y:F3},{t.Z:F3}) Rot=({r.X:F3},{r.Y:F3},{r.Z:F3}) Scale=({s.X:F4},{s.Y:F4},{s.Z:F4})");
        }

        ImGui.TextDisabled($"Obj Move: ({_worldScene.SelectedPm4ObjectTranslation.X:F3}, {_worldScene.SelectedPm4ObjectTranslation.Y:F3}, {_worldScene.SelectedPm4ObjectTranslation.Z:F3})");
        ImGui.TextDisabled($"Obj Rot: ({_worldScene.SelectedPm4ObjectRotationDegrees.X:F3}, {_worldScene.SelectedPm4ObjectRotationDegrees.Y:F3}, {_worldScene.SelectedPm4ObjectRotationDegrees.Z:F3}) deg");
        ImGui.TextDisabled($"Obj Scale: ({_worldScene.SelectedPm4ObjectScale.X:F4}, {_worldScene.SelectedPm4ObjectScale.Y:F4}, {_worldScene.SelectedPm4ObjectScale.Z:F4})");

        ImGui.End();
    }

    private void DrawPm4WmoCorrelationWindow()
    {
        if (_worldScene == null)
        {
            _showPm4WmoCorrelationWindow = false;
            _pm4WmoCorrelationReport = null;
            return;
        }

        EnsurePm4WmoCorrelationReportLoaded();

        ImGui.SetNextWindowSize(new Vector2(1120, 720), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("PM4/WMO Correlation", ref _showPm4WmoCorrelationWindow))
        {
            ImGui.End();
            return;
        }

        int requestedMatches = _pm4WmoCorrelationMaxMatchesPerPlacement;
        ImGui.SetNextItemWidth(90f);
        if (ImGui.InputInt("Max Matches", ref requestedMatches))
        {
            _pm4WmoCorrelationMaxMatchesPerPlacement = Math.Clamp(requestedMatches, 1, 32);
            RefreshPm4WmoCorrelationReport();
        }

        ImGui.SameLine();
        if (ImGui.Button("Refresh"))
            RefreshPm4WmoCorrelationReport();

        ImGui.SameLine();
        if (ImGui.Button("Dump JSON"))
            ExportPm4WmoCorrelationJson();

        ImGui.SameLine();
        if (ImGui.Checkbox("Only Near", ref _pm4WmoCorrelationNearOnly))
        {
            if (_selectedPm4WmoCorrelationPlacementIndex >= 0)
                _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        ImGui.SameLine();
        ImGui.SetNextItemWidth(260f);
        ImGui.InputTextWithHint("##Pm4WmoCorrelationFilter", "Filter model name or path", ref _pm4WmoCorrelationModelFilter, 256);

        if (_pm4WmoCorrelationReport == null)
        {
            ImGui.TextDisabled("No PM4/WMO correlation report is loaded.");
            ImGui.End();
            return;
        }

        Pm4WmoCorrelationReport report = _pm4WmoCorrelationReport;
        ImGui.TextDisabled(
            $"Generated {report.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss} UTC | placements {report.Summary.WmoPlacementCount}, resolved WMO meshes {report.Summary.WmoMeshResolvedCount}, PM4 objects {report.Summary.Pm4ObjectCount}");
        ImGui.TextDisabled(
            $"Candidates {report.Summary.PlacementsWithCandidates}/{report.Summary.WmoPlacementCount}, near {report.Summary.PlacementsWithNearCandidates}, PM4 status: {report.Pm4Status}");
        ImGui.Separator();

        string filter = _pm4WmoCorrelationModelFilter.Trim();
        var filteredPlacements = report.Placements
            .Select((placement, index) => new { placement, index })
            .Where(entry => !_pm4WmoCorrelationNearOnly || entry.placement.Pm4NearCandidateCount > 0)
            .Where(entry => string.IsNullOrWhiteSpace(filter)
                || entry.placement.ModelName.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelPath.Contains(filter, StringComparison.OrdinalIgnoreCase)
                || entry.placement.ModelKey.Contains(filter, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(entry => entry.placement.Pm4Matches.Count > 0 ? entry.placement.Pm4Matches[0].FootprintOverlapRatio : 0f)
            .ThenBy(entry => entry.placement.ModelName, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (filteredPlacements.Count == 0)
        {
            ImGui.TextDisabled("No placements matched the current filter.");
            ImGui.End();
            return;
        }

        if (!filteredPlacements.Any(entry => entry.index == _selectedPm4WmoCorrelationPlacementIndex))
        {
            _selectedPm4WmoCorrelationPlacementIndex = filteredPlacements[0].index;
            _selectedPm4WmoCorrelationMatchIndex = 0;
        }

        float leftWidth = MathF.Min(430f, ImGui.GetContentRegionAvail().X * 0.42f);
        if (ImGui.BeginChild("##Pm4WmoPlacementList", new Vector2(leftWidth, 0f), true))
        {
            for (int i = 0; i < filteredPlacements.Count; i++)
            {
                var entry = filteredPlacements[i];
                Pm4WmoCorrelationPlacement placement = entry.placement;
                bool selected = entry.index == _selectedPm4WmoCorrelationPlacementIndex;
                string label = $"[{placement.TileX},{placement.TileY}] {placement.ModelName}##Pm4WmoPlacement{entry.index}";
                if (ImGui.Selectable(label, selected))
                {
                    _selectedPm4WmoCorrelationPlacementIndex = entry.index;
                    _selectedPm4WmoCorrelationMatchIndex = 0;
                }

                ImGui.TextDisabled($"uid={placement.UniqueId} candidates={placement.Pm4CandidateCount} near={placement.Pm4NearCandidateCount}");
                if (placement.Pm4Matches.Count > 0)
                {
                    Pm4WmoCorrelationMatch best = placement.Pm4Matches[0];
                    ImGui.TextDisabled(
                        $"best CK24=0x{best.Ck24:X6} part={best.ObjectPartId} footprint={best.FootprintOverlapRatio:F2} area={best.FootprintAreaRatio:F2} dist={best.FootprintDistance:F1}");
                }
                else
                {
                    ImGui.TextDisabled("No PM4 candidates in the current tile neighborhood.");
                }

                ImGui.Separator();
            }
        }
        ImGui.EndChild();

        ImGui.SameLine();

        if (ImGui.BeginChild("##Pm4WmoPlacementDetails", Vector2.Zero, true))
        {
            Pm4WmoCorrelationPlacement placement = report.Placements[_selectedPm4WmoCorrelationPlacementIndex];
            ImGui.Text($"{placement.ModelName} (tile {placement.TileX},{placement.TileY}, uid {placement.UniqueId})");
            ImGui.TextDisabled(placement.ModelPath);

            if (ImGui.Button("Frame WMO"))
                FocusCameraOnBounds(placement.WorldBoundsMin, placement.WorldBoundsMax);

            if (placement.Pm4Matches.Count > 0)
            {
                Pm4WmoCorrelationMatch selectedMatch = placement.Pm4Matches[Math.Clamp(_selectedPm4WmoCorrelationMatchIndex, 0, placement.Pm4Matches.Count - 1)];

                ImGui.SameLine();
                if (ImGui.Button("Select PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);

                ImGui.SameLine();
                if (ImGui.Button("Frame PM4"))
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.Button("Frame Pair"))
                {
                    Vector3 boundsMin = Vector3.Min(placement.WorldBoundsMin, selectedMatch.BoundsMin);
                    Vector3 boundsMax = Vector3.Max(placement.WorldBoundsMax, selectedMatch.BoundsMax);
                    SelectPm4CorrelationMatch(selectedMatch, frameCamera: false);
                    FocusCameraOnBounds(boundsMin, boundsMax);
                }

                ImGui.SameLine();
                if (ImGui.Button("Snap PM4 XY"))
                    AlignPm4CorrelationMatchToPlacement(placement, selectedMatch, includeZ: false);

                ImGui.SameLine();
                if (ImGui.Button("Snap PM4 XYZ"))
                    AlignPm4CorrelationMatchToPlacement(placement, selectedMatch, includeZ: true);
            }

            ImGui.Separator();
            ImGui.TextDisabled($"Placement pos: ({placement.PlacementPosition.X:F2}, {placement.PlacementPosition.Y:F2}, {placement.PlacementPosition.Z:F2})");
            ImGui.TextDisabled($"Placement rot: ({placement.PlacementRotation.X:F2}, {placement.PlacementRotation.Y:F2}, {placement.PlacementRotation.Z:F2}) scale={placement.PlacementScale:F3}");
            ImGui.TextDisabled($"World bounds min: ({placement.WorldBoundsMin.X:F2}, {placement.WorldBoundsMin.Y:F2}, {placement.WorldBoundsMin.Z:F2})");
            ImGui.TextDisabled($"World bounds max: ({placement.WorldBoundsMax.X:F2}, {placement.WorldBoundsMax.Y:F2}, {placement.WorldBoundsMax.Z:F2})");
            if (placement.AdtPlacement.Found)
                ImGui.TextDisabled($"ADT flags=0x{placement.AdtPlacement.Flags:X4}");
            else
                ImGui.TextDisabled("No raw MODF placement metadata was found for this unique id.");

            if (placement.WmoMesh.Available)
            {
                ImGui.TextDisabled(
                    $"WMO v{placement.WmoMesh.Version}: groups={placement.WmoMesh.GroupCount} verts={placement.WmoMesh.VertexCount} tris={placement.WmoMesh.TriangleCount} batches={placement.WmoMesh.BatchCount}");
                ImGui.TextDisabled(
                    $"Footprint samples={placement.WmoMesh.FootprintSampleCount} hull={placement.WmoMesh.WorldFootprintHullPointCount} area={placement.WmoMesh.WorldFootprintArea:F1}");
            }
            else
            {
                ImGui.TextDisabled("WMO mesh summary is unavailable for this placement.");
            }

            ImGui.Separator();
            ImGui.Text($"PM4 matches ({placement.Pm4Matches.Count}/{placement.Pm4CandidateCount} shown, near={placement.Pm4NearCandidateCount})");

            if (placement.Pm4Matches.Count == 0)
            {
                ImGui.TextDisabled("No PM4 candidate objects are available for this placement.");
            }
            else if (ImGui.BeginChild("##Pm4WmoMatchList", Vector2.Zero, false))
            {
                for (int matchIndex = 0; matchIndex < placement.Pm4Matches.Count; matchIndex++)
                {
                    Pm4WmoCorrelationMatch match = placement.Pm4Matches[matchIndex];
                    bool selected = matchIndex == _selectedPm4WmoCorrelationMatchIndex;
                    string label = $"CK24 0x{match.Ck24:X6} part {match.ObjectPartId}##Pm4WmoMatch{matchIndex}";
                    if (ImGui.Selectable(label, selected))
                        _selectedPm4WmoCorrelationMatchIndex = matchIndex;

                    ImGui.TextDisabled(
                        $"tile=({match.TileX},{match.TileY}) type=0x{match.Ck24Type:X2} objId={match.Ck24ObjectId} sameTile={match.SameTile}");
                    ImGui.TextDisabled(
                        $"footprint overlap={match.FootprintOverlapRatio:F3} area={match.FootprintAreaRatio:F3} dist={match.FootprintDistance:F2}");
                    ImGui.TextDisabled(
                        $"planar gap={match.PlanarGap:F2} vertical gap={match.VerticalGap:F2} center={match.CenterDistance:F2} planar overlap={match.PlanarOverlapRatio:F3}");
                    ImGui.TextDisabled(
                        $"surfaces={match.SurfaceCount} linked refs={match.LinkedPositionRefCount} mdos={match.DominantMdosIndex} avgH={match.AverageSurfaceHeight:F2}");
                    ImGui.Separator();
                }

                ImGui.EndChild();
            }
        }
        ImGui.EndChild();

        ImGui.End();
    }

    private void SaveCurrentPm4Alignment()
    {
        if (_worldScene == null)
            return;

        _pm4SavedOverlayTranslation = _worldScene.Pm4OverlayTranslation;
        _pm4SavedOverlayRotationDegrees = _worldScene.Pm4OverlayRotationDegrees;
        _pm4SavedOverlayScale = _worldScene.Pm4OverlayScale;
        SaveViewerSettings();

        _statusMessage = $"Saved PM4 alignment: T=({_pm4SavedOverlayTranslation.X:F2}, {_pm4SavedOverlayTranslation.Y:F2}, {_pm4SavedOverlayTranslation.Z:F2}) Rot=({_pm4SavedOverlayRotationDegrees.X:F2}, {_pm4SavedOverlayRotationDegrees.Y:F2}, {_pm4SavedOverlayRotationDegrees.Z:F2})° S=({_pm4SavedOverlayScale.X:F3}, {_pm4SavedOverlayScale.Y:F3}, {_pm4SavedOverlayScale.Z:F3})";
    }

    private void ExportPm4ObjectsJson()
    {
        if (_worldScene == null)
            return;

        string defaultName = $"pm4_objects_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save PM4 Objects JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = _worldScene.BuildPm4OverlayInterchangeJson(includeGeometry: true);
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported PM4 objects JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 JSON export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] JSON export failed: {ex}");
        }
    }

    private void ExportPm4ObjectsObjSet()
    {
        if (_worldScene == null)
            return;

        Directory.CreateDirectory(ExportDir);
        string? picked = ShowFolderDialogSTA(
            "Choose a folder for PM4 OBJ export",
            ExportDir,
            showNewFolderButton: true);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            Pm4OfflineObjExportSummary summary = _worldScene.ExportPm4ObjectsAsObjDirectory(picked);
            _statusMessage =
                $"Exported PM4 OBJ set: {summary.ExportedObjectCount} objects across {summary.ExportedTileCount} tiles to {summary.OutputDirectory} (manifest: {summary.ManifestPath}).";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 OBJ export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] OBJ export failed: {ex}");
        }
    }

    private void ExportPm4WmoCorrelationJson()
    {
        if (_worldScene == null)
            return;

        string defaultName = $"pm4_wmo_correlation_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save PM4/WMO Correlation JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = _worldScene.BuildPm4WmoPlacementCorrelationJson();
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported PM4/WMO correlation JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4/WMO correlation export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Correlation] JSON export failed: {ex}");
        }
    }

    private void InvalidatePm4DerivedReports()
    {
        _pm4ObjectMatchReport = null;
        _selectedPm4ObjectMatch = null;
        _selectedPm4ObjectMatchKey = null;
        _selectedPm4ObjectMatchCacheMaxMatches = -1;
        _hoveredPm4ObjectMatch = null;
        _hoveredPm4ObjectMatchKey = null;
        _hoveredPm4ObjectMatchCacheMaxMatches = -1;
        _pm4WmoCorrelationReport = null;
    }

    private void EnsurePm4WmoCorrelationReportLoaded()
    {
        if (_pm4WmoCorrelationReport == null)
            RefreshPm4WmoCorrelationReport();
    }

    private void RefreshPm4WmoCorrelationReport()
    {
        if (_worldScene == null)
            return;

        try
        {
            _pm4WmoCorrelationReport = _worldScene.BuildPm4WmoPlacementCorrelationReport(_pm4WmoCorrelationMaxMatchesPerPlacement);
            if (_pm4WmoCorrelationReport.Placements.Count == 0)
            {
                _selectedPm4WmoCorrelationPlacementIndex = -1;
                _selectedPm4WmoCorrelationMatchIndex = 0;
            }
            else if (_selectedPm4WmoCorrelationPlacementIndex < 0 || _selectedPm4WmoCorrelationPlacementIndex >= _pm4WmoCorrelationReport.Placements.Count)
            {
                _selectedPm4WmoCorrelationPlacementIndex = 0;
                _selectedPm4WmoCorrelationMatchIndex = 0;
            }

            _statusMessage = $"Refreshed PM4/WMO correlation report ({_pm4WmoCorrelationReport.Summary.WmoPlacementCount} placements).";
        }
        catch (Exception ex)
        {
            _pm4WmoCorrelationReport = null;
            _statusMessage = $"PM4/WMO correlation refresh failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Correlation] Report refresh failed: {ex}");
        }
    }

    private void EnsurePm4ObjectMatchReportLoaded()
    {
        if (_pm4ObjectMatchReport == null)
            RefreshPm4ObjectMatchReport();
    }

    private void RefreshPm4ObjectMatchReport()
    {
        if (_worldScene == null)
            return;

        try
        {
            _pm4ObjectMatchReport = _worldScene.BuildPm4ObjectMatchReport(_pm4ObjectMatchMaxMatchesPerObject);
            if (_pm4ObjectMatchReport.Objects.Count == 0)
            {
                _selectedPm4ObjectMatchObjectIndex = -1;
                _selectedPm4ObjectMatchCandidateIndex = 0;
            }
            else if (!TryGetSelectedPm4ObjectMatch(out _)
                && (_selectedPm4ObjectMatchObjectIndex < 0 || _selectedPm4ObjectMatchObjectIndex >= _pm4ObjectMatchReport.Objects.Count))
            {
                _selectedPm4ObjectMatchObjectIndex = 0;
                _selectedPm4ObjectMatchCandidateIndex = 0;
            }

            _statusMessage = $"Refreshed PM4 object match report ({_pm4ObjectMatchReport.Summary.Pm4ObjectCount} PM4 objects).";
        }
        catch (Exception ex)
        {
            _pm4ObjectMatchReport = null;
            _statusMessage = $"PM4 object match refresh failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Object Match] Report refresh failed: {ex}");
        }
    }

    private void DrawPm4ObjectMatchWindow()
    {
        if (_worldScene == null)
        {
            _showPm4ObjectMatchWindow = false;
            _pm4ObjectMatchReport = null;
            return;
        }

        EnsurePm4ObjectMatchReportLoaded();

        ImGui.SetNextWindowSize(new Vector2(1220, 760), ImGuiCond.FirstUseEver);
        if (!ImGui.Begin("PM4 Object Match", ref _showPm4ObjectMatchWindow))
        {
            ImGui.End();
            return;
        }

        int requestedMatches = _pm4ObjectMatchMaxMatchesPerObject;
        ImGui.SetNextItemWidth(110f);
        if (ImGui.SliderInt("Top Matches", ref requestedMatches, 3, 5))
        {
            _pm4ObjectMatchMaxMatchesPerObject = Math.Clamp(requestedMatches, 3, 5);
            RefreshPm4ObjectMatchReport();
        }

        ImGui.SameLine();
        if (ImGui.Button("Refresh"))
            RefreshPm4ObjectMatchReport();

        if (_pm4ObjectMatchReport == null)
        {
            ImGui.TextDisabled("No PM4 object match report is loaded.");
            ImGui.End();
            return;
        }

        Pm4ObjectMatchReport report = _pm4ObjectMatchReport;
        ImGui.TextDisabled(
            $"Generated {report.GeneratedAtUtc:yyyy-MM-dd HH:mm:ss} UTC | PM4 objects {report.Summary.Pm4ObjectCount}, WMO placements {report.Summary.WmoPlacementCount}, M2 placements {report.Summary.M2PlacementCount}");
        ImGui.TextDisabled(
            $"Objects with candidates {report.Summary.ObjectsWithCandidates}/{report.Summary.Pm4ObjectCount}, near {report.Summary.ObjectsWithNearCandidates}, status: {report.Pm4Status}");
        ImGui.TextDisabled("Ranking keeps WMO-mesh priority for non-zero families, but zero/root PM4 objects with linked refs now prefer M2 anchors before the usual tile/anchor/planar fit checks.");
        ImGui.Separator();

        if (!_worldScene.HasSelectedPm4Object)
        {
            ImGui.TextDisabled("Select a PM4 object in the scene to see its top suggested matches.");
            ImGui.End();
            return;
        }

        if (!TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch))
        {
            ImGui.TextDisabled("The selected PM4 object is not present in the current match report. Refresh and try again.");
            ImGui.End();
            return;
        }

        DrawPm4SelectedObjectMatchSuggestions("WindowPm4Match", compact: false);

        ImGui.End();
    }

    private bool TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        if (_worldScene == null || !_worldScene.SelectedPm4ObjectKey.HasValue)
            return false;

        var selectedKey = _worldScene.SelectedPm4ObjectKey.Value;
        if (_selectedPm4ObjectMatch != null
            && _selectedPm4ObjectMatchKey.HasValue
            && _selectedPm4ObjectMatchKey.Value == selectedKey
            && _selectedPm4ObjectMatchCacheMaxMatches == _pm4ObjectMatchMaxMatchesPerObject)
        {
            objectMatch = _selectedPm4ObjectMatch;
            return true;
        }

        if (!_worldScene.TryBuildSelectedPm4ObjectMatch(_pm4ObjectMatchMaxMatchesPerObject, out Pm4ObjectMatchObject selectedMatch))
            return false;

        _selectedPm4ObjectMatch = selectedMatch;
        _selectedPm4ObjectMatchKey = selectedKey;
        _selectedPm4ObjectMatchCacheMaxMatches = _pm4ObjectMatchMaxMatchesPerObject;
        objectMatch = selectedMatch;

        if (_pm4ObjectMatchReport == null)
            return true;

        for (int index = 0; index < _pm4ObjectMatchReport.Objects.Count; index++)
        {
            Pm4ObjectMatchObject candidate = _pm4ObjectMatchReport.Objects[index];
            if (candidate.TileX != selectedKey.tileX
                || candidate.TileY != selectedKey.tileY
                || candidate.Ck24 != selectedKey.ck24
                || candidate.ObjectPartId != selectedKey.objectPart)
            {
                continue;
            }

            _selectedPm4ObjectMatchObjectIndex = index;
            if (_selectedPm4ObjectMatchCandidateIndex < 0 || _selectedPm4ObjectMatchCandidateIndex >= candidate.Candidates.Count)
                _selectedPm4ObjectMatchCandidateIndex = 0;
            return true;
        }

        _selectedPm4ObjectMatchObjectIndex = -1;
        return true;
    }

    private void DrawPm4SelectedObjectMatchSuggestions(string idSuffix, bool compact)
    {
        if (_worldScene == null || !_worldScene.HasSelectedPm4Object)
        {
            ImGui.TextDisabled("Select a PM4 object in the scene to see suggested matches.");
            return;
        }

        if (!TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch))
        {
            ImGui.TextDisabled("No PM4 match report entry is available for the current selection.");
            return;
        }

        int shownCandidateCount = Math.Min(objectMatch.Candidates.Count, Math.Clamp(_pm4ObjectMatchMaxMatchesPerObject, 3, 5));
        bool hasSaved = TryGetSavedPm4ObjectMatch(objectMatch, out SavedPm4ObjectMatchSelection? savedSelection);

        ImGui.Text($"CK24 0x{objectMatch.Ck24:X6} part {objectMatch.ObjectPartId} (tile {objectMatch.TileX},{objectMatch.TileY})");
        ImGui.TextDisabled($"wmo={objectMatch.WmoCandidateCount} m2={objectMatch.M2CandidateCount} near={objectMatch.NearCandidateCount} refs={objectMatch.LinkedPositionRefCount} mslk=0x{objectMatch.LinkGroupObjectId:X8}");

        if (hasSaved && savedSelection != null)
            ImGui.TextColored(new Vector4(0.95f, 0.85f, 0.35f, 1f), $"Saved: {savedSelection.PlacementKind} uid={savedSelection.PlacementUniqueId} {savedSelection.ModelName} [{savedSelection.EvidenceSource}]");
        else
            ImGui.TextDisabled("Saved: none");

        if (!compact)
        {
            if (ImGui.Button($"Frame PM4##{idSuffix}"))
                SelectPm4ObjectMatchObject(objectMatch, frameCamera: true);

            ImGui.SameLine();
            if (ImGui.Button($"Jump To Alignment##{idSuffix}"))
                SelectPm4ObjectMatchObject(objectMatch, frameCamera: false);

            ImGui.SameLine();
            if (ImGui.Button($"Clear Saved Choice##{idSuffix}"))
                ClearSavedPm4ObjectMatch(objectMatch);
        }
        else if (ImGui.SmallButton($"Clear Saved Choice##{idSuffix}"))
        {
            ClearSavedPm4ObjectMatch(objectMatch);
        }

        ImGui.Separator();
        ImGui.Text($"Top matches ({shownCandidateCount}/{objectMatch.CandidateCount})");

        if (shownCandidateCount == 0)
        {
            ImGui.TextDisabled("No placement candidates are available for this PM4 object.");
            return;
        }

        for (int candidateIndex = 0; candidateIndex < shownCandidateCount; candidateIndex++)
        {
            Pm4ObjectMatchCandidate candidate = objectMatch.Candidates[candidateIndex];
            bool isSaved = savedSelection != null
                && string.Equals(savedSelection.PlacementKind, candidate.Kind, StringComparison.OrdinalIgnoreCase)
                && savedSelection.PlacementUniqueId == candidate.UniqueId
                && string.Equals(savedSelection.ModelPath, candidate.ModelPath, StringComparison.OrdinalIgnoreCase);

            ImGui.PushID($"{idSuffix}_{candidateIndex}");
            if (isSaved)
                ImGui.TextColored(new Vector4(0.95f, 0.85f, 0.35f, 1f), $"{candidateIndex + 1}. {candidate.Kind.ToUpperInvariant()} {candidate.ModelName}  [saved]");
            else
                ImGui.TextWrapped($"{candidateIndex + 1}. {candidate.Kind.ToUpperInvariant()} {candidate.ModelName}");

            ImGui.TextDisabled($"{candidate.EvidenceSource} | tile {candidate.TileX},{candidate.TileY} | anchor {candidate.AnchorPlanarGap:F1} | planar {candidate.PlanarGap:F1} | center {candidate.CenterDistance:F1}");
            if (!compact)
                ImGui.TextDisabled(candidate.ModelPath);

            if (ImGui.SmallButton("Frame"))
            {
                _selectedPm4ObjectMatchCandidateIndex = candidateIndex;
                FocusCameraOnBounds(candidate.WorldBoundsMin, candidate.WorldBoundsMax);
            }

            ImGui.SameLine();
            if (ImGui.SmallButton("Save"))
            {
                _selectedPm4ObjectMatchCandidateIndex = candidateIndex;
                SavePm4ObjectMatchSelection(objectMatch, candidate);
            }

            if (!compact)
            {
                ImGui.SameLine();
                if (ImGui.SmallButton("Frame Pair"))
                {
                    _selectedPm4ObjectMatchCandidateIndex = candidateIndex;
                    SelectPm4ObjectMatchObject(objectMatch, frameCamera: false);
                    FocusCameraOnBounds(Vector3.Min(objectMatch.BoundsMin, candidate.WorldBoundsMin), Vector3.Max(objectMatch.BoundsMax, candidate.WorldBoundsMax));
                }
            }

            ImGui.PopID();
            if (candidateIndex + 1 < shownCandidateCount)
                ImGui.Separator();
        }
    }

    private void SelectPm4ObjectMatchObject(Pm4ObjectMatchObject objectMatch, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectPm4Object((objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId)))
        {
            OpenPm4Workbench(Pm4WorkbenchTab.Selection);
            if (frameCamera)
                FocusCameraOnBounds(objectMatch.BoundsMin, objectMatch.BoundsMax);

            _statusMessage = $"Selected PM4 object CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId}.";
        }
        else
        {
            _statusMessage = $"PM4 object CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId} is no longer available.";
        }
    }

    private void SavePm4ObjectMatchSelection(Pm4ObjectMatchObject objectMatch, Pm4ObjectMatchCandidate candidate)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        if (string.IsNullOrWhiteSpace(mapName))
        {
            _statusMessage = "Cannot save PM4 object match: map name is unavailable.";
            return;
        }

        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        _savedPm4ObjectMatches[key] = new SavedPm4ObjectMatchSelection
        {
            MapName = mapName,
            TileX = objectMatch.TileX,
            TileY = objectMatch.TileY,
            Ck24 = objectMatch.Ck24,
            ObjectPartId = objectMatch.ObjectPartId,
            PlacementKind = candidate.Kind,
            PlacementUniqueId = candidate.UniqueId,
            PlacementTileX = candidate.TileX,
            PlacementTileY = candidate.TileY,
            ModelName = candidate.ModelName,
            ModelPath = candidate.ModelPath,
            EvidenceSource = candidate.EvidenceSource,
        };

        SaveViewerSettings();
        _statusMessage = $"Saved PM4 object match: CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId} -> {candidate.Kind} uid={candidate.UniqueId}.";
    }

    private void ClearSavedPm4ObjectMatch(Pm4ObjectMatchObject objectMatch)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        if (_savedPm4ObjectMatches.Remove(key))
        {
            SaveViewerSettings();
            _statusMessage = $"Cleared saved PM4 object match for CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId}.";
        }
    }

    private bool TryGetSavedPm4ObjectMatch(Pm4ObjectMatchObject objectMatch, out SavedPm4ObjectMatchSelection? selection)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        if (_savedPm4ObjectMatches.TryGetValue(key, out SavedPm4ObjectMatchSelection? savedSelection))
        {
            selection = savedSelection;
            return true;
        }

        selection = null;
        return false;
    }

    private static string BuildSavedPm4ObjectMatchKey(string mapName, int tileX, int tileY, uint ck24, int objectPartId)
    {
        return $"{mapName.Trim().ToLowerInvariant()}|{tileX}|{tileY}|{ck24:X6}|{objectPartId}";
    }

    private void SelectPm4CorrelationMatch(Pm4WmoCorrelationMatch match, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
        {
            OpenPm4Workbench(Pm4WorkbenchTab.Selection);
            if (frameCamera)
                FocusCameraOnBounds(match.BoundsMin, match.BoundsMax);

            _statusMessage = $"Selected PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} from correlation panel.";
        }
        else
        {
            _statusMessage = $"PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} is no longer available.";
        }
    }

    private void AlignPm4CorrelationMatchToPlacement(Pm4WmoCorrelationPlacement placement, Pm4WmoCorrelationMatch match, bool includeZ)
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
        {
            _statusMessage = $"PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} is no longer available.";
            return;
        }

        if (!_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            _statusMessage = "PM4 snap failed: selected object debug info is unavailable.";
            return;
        }

        Vector3 placementCenter = (placement.WorldBoundsMin + placement.WorldBoundsMax) * 0.5f;
        Vector3 delta = placementCenter - debugInfo.Center;
        if (!includeZ)
            delta.Z = 0f;

        _worldScene.SelectedPm4ObjectTranslation += delta;
        InvalidatePm4DerivedReports();
        _showPm4AlignmentWindow = true;

        string axes = includeZ ? "XYZ" : "XY";
        _statusMessage =
            $"Snapped PM4 CK24=0x{match.Ck24:X6} part={match.ObjectPartId} to WMO center ({axes}) by ({delta.X:F2}, {delta.Y:F2}, {delta.Z:F2}).";
    }

    private void FocusCameraOnBounds(Vector3 boundsMin, Vector3 boundsMax)
    {
        Vector3 center = (boundsMin + boundsMax) * 0.5f;
        Vector3 extent = Vector3.Max(boundsMax - boundsMin, new Vector3(1f, 1f, 1f));
        float distance = MathF.Max(extent.Length() * 1.35f, 80f);

        _camera.Position = center + new Vector3(distance, 0f, MathF.Max(extent.Z * 0.6f, 30f));
        _camera.Yaw = 180f;
        _camera.Pitch = -18f;
    }

    private void ApplySavedPm4AlignmentToScene()
    {
        if (_worldScene == null)
            return;

        _worldScene.Pm4OverlayTranslation = _pm4SavedOverlayTranslation;
        _worldScene.Pm4OverlayRotationDegrees = _pm4SavedOverlayRotationDegrees;
        _worldScene.Pm4OverlayScale = _pm4SavedOverlayScale;
        InvalidatePm4DerivedReports();
    }

    private static Vector3 NormalizeRotationDegrees(Vector3 rotation)
    {
        return new Vector3(
            NormalizeDegrees(rotation.X),
            NormalizeDegrees(rotation.Y),
            NormalizeDegrees(rotation.Z));
    }

    private static float NormalizeDegrees(float value)
    {
        float wrapped = value % 360f;
        if (wrapped < -180f)
            wrapped += 360f;
        else if (wrapped > 180f)
            wrapped -= 360f;
        return wrapped;
    }

    private static float ToggleWindingComponent(float value)
    {
        float magnitude = MathF.Abs(value);
        if (magnitude < 0.0001f)
            magnitude = 1f;

        return value < 0f ? magnitude : -magnitude;
    }

    private static string GetPm4ColorModeLabel(Pm4OverlayColorMode mode)
    {
        return mode switch
        {
            Pm4OverlayColorMode.Ck24Type => "MSUR._0x1C-derived type byte",
            Pm4OverlayColorMode.Ck24ObjectId => "MSUR._0x1C-derived low16",
            Pm4OverlayColorMode.Ck24Key => "MSUR._0x1C-derived key24",
            Pm4OverlayColorMode.Tile => "Tile",
            Pm4OverlayColorMode.GroupKey => "MSUR._0x00 (alias GroupKey)",
            Pm4OverlayColorMode.AttributeMask => "MSUR._0x02 (alias AttributeMask)",
            Pm4OverlayColorMode.Height => "MSUR._0x10 plane-distance gradient",
            _ => mode.ToString()
        };
    }

    private void DrawPm4ColorLegend(string idSuffix = "")
    {
        if (_worldScene == null || !_worldScene.ShowPm4Overlay)
            return;

        Pm4ColorLegendInfo legend = _worldScene.GetPm4ColorLegend();
        if (!ImGui.CollapsingHeader($"PM4 Color Legend##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        if (!string.IsNullOrWhiteSpace(legend.Description))
            ImGui.TextDisabled(legend.Description);

        if (legend.Entries.Count == 0)
        {
            ImGui.TextDisabled("No loaded PM4 objects for the current legend mode.");
            return;
        }

        for (int i = 0; i < legend.Entries.Count; i++)
        {
            Pm4ColorLegendEntry entry = legend.Entries[i];
            ImGui.ColorButton(
                $"##Pm4LegendColor{idSuffix}_{i}",
                new Vector4(entry.Color, 1f),
                ImGuiColorEditFlags.NoTooltip | ImGuiColorEditFlags.NoDragDrop,
                new Vector2(14f, 14f));
            ImGui.SameLine();
            if (entry.IsSelected)
                ImGui.TextColored(new Vector4(1f, 1f, 0.35f, 1f), $"{entry.Label}  [{entry.ObjectCount}]  selected");
            else if (legend.IsContinuous)
                ImGui.TextUnformatted(entry.Label);
            else
                ImGui.TextUnformatted($"{entry.Label}  [{entry.ObjectCount}]");
        }

        if (legend.IsTruncated)
            ImGui.TextDisabled($"Showing {legend.Entries.Count} of {legend.TotalEntryCount} legend entries.");
    }

    private void DrawSelectedPm4ObjectGraph(string idSuffix = "")
    {
        if (_worldScene == null || !_worldScene.TryGetSelectedPm4ObjectGraphInfo(out Pm4SelectedObjectGraphInfo graph))
            return;

        if (!ImGui.CollapsingHeader($"PM4 Graph##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGui.TextDisabled("Derived from the current overlay build: CK24 root, MSLK-linked groups, optional MDOS split, then connectivity parts.");
        ImGui.TextDisabled("part/ObjectPartId is a viewer-generated split id from that build, not a raw PM4 field.");
        ImGui.TextDisabled("Treat this as viewer structure, not a claim that PM4 stores matching raw graph nodes.");
        ImGui.TextDisabled($"Split flags: MDOS={graph.SplitByMdos} Connectivity={graph.SplitByConnectivity}");
        ImGui.TextDisabled($"Tiles={graph.TileCount} LinkGroups={graph.LinkGroupCount} MdosGroups={graph.MdosGroupCount} Parts={graph.PartCount}");
        ImGui.TextDisabled($"Surfaces={graph.SurfaceCount} Indices={graph.TotalIndexCount} AttrMasks={graph.AttributeMaskCount} GroupKeys={graph.GroupKeyCount}");
        ImGui.TextDisabled("Click a part row to reselect it. Use Frame to move the camera to that exact part.");
        ImGui.TextDisabled("Use the graph Collect buttons as the primary PM4 multi-select path; viewport PM4 picking is not reliable enough.");

        if (ImGui.Button($"Export Graph JSON##{idSuffix}"))
            ExportSelectedPm4GraphJson(graph);
        ImGui.SameLine();
        if (ImGui.Button($"Add Part##{idSuffix}"))
            AddSelectedPm4ObjectToCollection();
        ImGui.SameLine();
        if (ImGui.Button($"Add Merged Group##{idSuffix}"))
            AddSelectedPm4GraphGroupToCollection(graph);
        ImGui.SameLine();
        if (ImGui.Button($"Export Collection JSON##{idSuffix}"))
            ExportPm4ObjectCollectionJson();
        ImGui.SameLine();
        if (ImGui.Button($"Clear Collection##{idSuffix}"))
            ClearPm4ObjectCollection();

        DrawPm4ObjectCollectionSummary(idSuffix);

        ImGuiTreeNodeFlags rootFlags = ImGuiTreeNodeFlags.DefaultOpen;
        if (ImGui.TreeNodeEx($"CK24 0x{graph.Ck24:X6} type=0x{graph.Ck24Type:X2} obj={graph.Ck24ObjectId}##Pm4GraphRoot{idSuffix}", rootFlags))
        {
            for (int linkIndex = 0; linkIndex < graph.LinkGroups.Count; linkIndex++)
            {
                Pm4SelectedObjectGraphLinkNode linkGroup = graph.LinkGroups[linkIndex];
                string linkSummary = $"MSLK 0x{linkGroup.LinkGroupObjectId:X8} parts={linkGroup.PartCount} surfaces={linkGroup.SurfaceCount} indices={linkGroup.TotalIndexCount} linkedMPRL={linkGroup.LinkedPositionRefCount}";
                if (ImGui.TreeNodeEx($"{linkSummary}##Pm4GraphLink{idSuffix}_{linkIndex}", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    if (ImGui.SmallButton($"Collect Link##Pm4GraphCollectLink{idSuffix}_{linkIndex}"))
                        AddPm4LinkGroupToCollection(graph, linkGroup);

                    if (linkGroup.LinkedPositionRefSummary.TotalCount > 0)
                    {
                        ImGui.TextDisabled(
                            $"MPRL normal={linkGroup.LinkedPositionRefSummary.NormalCount} term={linkGroup.LinkedPositionRefSummary.TerminatorCount} floors={linkGroup.LinkedPositionRefSummary.FloorMin}..{linkGroup.LinkedPositionRefSummary.FloorMax}");
                    }

                    for (int mdosIndex = 0; mdosIndex < linkGroup.MdosGroups.Count; mdosIndex++)
                    {
                        Pm4SelectedObjectGraphMdosNode mdosGroup = linkGroup.MdosGroups[mdosIndex];
                        string mdosSummary = $"MDOS {mdosGroup.MdosIndex} parts={mdosGroup.PartCount} surfaces={mdosGroup.SurfaceCount} indices={mdosGroup.TotalIndexCount} attrs={FormatPm4ByteList(mdosGroup.AttributeMasks)} groups={FormatPm4ByteList(mdosGroup.GroupKeys)}";
                        if (ImGui.TreeNodeEx($"{mdosSummary}##Pm4GraphMdos{idSuffix}_{linkIndex}_{mdosIndex}", ImGuiTreeNodeFlags.DefaultOpen))
                        {
                            if (ImGui.SmallButton($"Collect MDOS##Pm4GraphCollectMdos{idSuffix}_{linkIndex}_{mdosIndex}"))
                                AddPm4MdosGroupToCollection(graph, mdosGroup);

                            for (int partIndex = 0; partIndex < mdosGroup.Parts.Count; partIndex++)
                            {
                                Pm4SelectedObjectGraphPartNode part = mdosGroup.Parts[partIndex];
                                ImGuiTreeNodeFlags partFlags = ImGuiTreeNodeFlags.Leaf | ImGuiTreeNodeFlags.NoTreePushOnOpen;
                                if (part.IsSelected)
                                    partFlags |= ImGuiTreeNodeFlags.Selected;

                                string partSummary = $"part={part.ObjectPartId} tile=({part.TileX},{part.TileY}) surfaces={part.SurfaceCount} indices={part.TotalIndexCount} lines={part.LineCount} tris={part.TriangleCount} group=0x{part.DominantGroupKey:X2} attr=0x{part.DominantAttributeMask:X2}";
                                ImGui.TreeNodeEx($"{partSummary}##Pm4GraphPart{idSuffix}_{linkIndex}_{mdosIndex}_{partIndex}", partFlags);
                                if (ImGui.IsItemClicked())
                                    SelectPm4GraphPart((part.TileX, part.TileY, graph.Ck24, part.ObjectPartId), frameCamera: false);
                                ImGui.SameLine();
                                if (ImGui.SmallButton($"Frame##Pm4GraphPartFrame{idSuffix}_{linkIndex}_{mdosIndex}_{partIndex}"))
                                    SelectPm4GraphPart((part.TileX, part.TileY, graph.Ck24, part.ObjectPartId), frameCamera: true);
                                ImGui.SameLine();
                                if (ImGui.SmallButton($"Collect##Pm4GraphPartCollect{idSuffix}_{linkIndex}_{mdosIndex}_{partIndex}"))
                                    TogglePm4ObjectCollectionMembership((part.TileX, part.TileY, graph.Ck24, part.ObjectPartId), reportStatus: true);
                            }

                            ImGui.TreePop();
                        }
                    }

                    ImGui.TreePop();
                }
            }

            ImGui.TreePop();
        }
    }

    private static string FormatPm4ByteList(IReadOnlyList<byte> values)
    {
        if (values.Count == 0)
            return "-";

        return string.Join(", ", values.Select(static value => $"0x{value:X2}"));
    }

    private void SelectPm4GraphPart((int tileX, int tileY, uint ck24, int objectPart) objectKey, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (!_worldScene.SelectPm4Object(objectKey))
        {
            _statusMessage = $"PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart} is no longer available.";
            return;
        }

        OpenPm4Workbench(Pm4WorkbenchTab.Selection);

        if (frameCamera)
        {
            if (_worldScene.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
                FocusCameraOnBounds(debugInfo.BoundsMin, debugInfo.BoundsMax);
        }

        _statusMessage = frameCamera
            ? $"Selected and framed PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart}."
            : $"Selected PM4 graph part CK24=0x{objectKey.ck24:X6} part={objectKey.objectPart}.";
    }

    private void ExportSelectedPm4GraphJson(Pm4SelectedObjectGraphInfo graph)
    {
        string defaultName = $"pm4_graph_ck24_{graph.Ck24:X6}_part_{graph.SelectedObjectPartId:D4}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save Selected PM4 Graph JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = JsonSerializer.Serialize(BuildJsonSafePm4Graph(graph), new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported selected PM4 graph JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 graph export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Graph] JSON export failed: {ex}");
        }
    }

    private void AddSelectedPm4ObjectToCollection()
    {
        if (_worldScene == null || !_worldScene.SelectedPm4ObjectKey.HasValue)
            return;

        TogglePm4ObjectCollectionMembership(_worldScene.SelectedPm4ObjectKey.Value, reportStatus: true, removeIfPresent: false);
    }

    private void AddSelectedPm4GraphGroupToCollection(Pm4SelectedObjectGraphInfo graph)
    {
        var keys = graph.LinkGroups
            .SelectMany(static linkGroup => linkGroup.MdosGroups)
            .SelectMany(static mdosGroup => mdosGroup.Parts)
            .Select(part => (part.TileX, part.TileY, graph.Ck24, part.ObjectPartId));

        int added = AddPm4ObjectsToCollection(keys);
        _statusMessage = added > 0
            ? $"Added {added} PM4 parts from the merged group to the collection."
            : "All parts in the merged group were already in the PM4 collection.";
        SyncPm4CollectionHighlight();
    }

    private void AddPm4LinkGroupToCollection(Pm4SelectedObjectGraphInfo graph, Pm4SelectedObjectGraphLinkNode linkGroup)
    {
        var keys = linkGroup.MdosGroups
            .SelectMany(static mdosGroup => mdosGroup.Parts)
            .Select(part => (part.TileX, part.TileY, graph.Ck24, part.ObjectPartId));

        int added = AddPm4ObjectsToCollection(keys);
        _statusMessage = added > 0
            ? $"Added {added} PM4 parts from MSLK 0x{linkGroup.LinkGroupObjectId:X8} to the collection."
            : $"All PM4 parts from MSLK 0x{linkGroup.LinkGroupObjectId:X8} were already in the collection.";
        SyncPm4CollectionHighlight();
    }

    private void AddPm4MdosGroupToCollection(Pm4SelectedObjectGraphInfo graph, Pm4SelectedObjectGraphMdosNode mdosGroup)
    {
        var keys = mdosGroup.Parts
            .Select(part => (part.TileX, part.TileY, graph.Ck24, part.ObjectPartId));

        int added = AddPm4ObjectsToCollection(keys);
        _statusMessage = added > 0
            ? $"Added {added} PM4 parts from MDOS {mdosGroup.MdosIndex} to the collection."
            : $"All PM4 parts from MDOS {mdosGroup.MdosIndex} were already in the collection.";
        SyncPm4CollectionHighlight();
    }

    private int AddPm4ObjectsToCollection(IEnumerable<(int tileX, int tileY, uint ck24, int objectPart)> keys)
    {
        int added = 0;
        foreach (var key in keys)
        {
            if (_pm4ObjectCollection.Contains(key))
                continue;

            _pm4ObjectCollection.Add(key);
            added++;
        }

        if (added > 0)
            SyncPm4CollectionHighlight();

        return added;
    }

    private bool TogglePm4ObjectCollectionMembership(
        (int tileX, int tileY, uint ck24, int objectPart) key,
        bool reportStatus,
        bool removeIfPresent = true)
    {
        int existingIndex = _pm4ObjectCollection.IndexOf(key);
        if (existingIndex >= 0)
        {
            if (removeIfPresent)
            {
                _pm4ObjectCollection.RemoveAt(existingIndex);
                SyncPm4CollectionHighlight();
                if (reportStatus)
                    _statusMessage = $"Removed PM4 CK24=0x{key.ck24:X6} part={key.objectPart} from the collection.";
                return false;
            }

            if (reportStatus)
                _statusMessage = $"PM4 CK24=0x{key.ck24:X6} part={key.objectPart} is already in the collection.";
            return false;
        }

        _pm4ObjectCollection.Add(key);
        SyncPm4CollectionHighlight();
        if (reportStatus)
            _statusMessage = $"Added PM4 CK24=0x{key.ck24:X6} part={key.objectPart} to the collection.";
        return true;
    }

    private void ClearPm4ObjectCollection()
    {
        _pm4ObjectCollection.Clear();
        SyncPm4CollectionHighlight();
        _statusMessage = "Cleared PM4 object collection.";
    }

    private void SyncPm4CollectionHighlight()
    {
        _worldScene?.SetHighlightedPm4Objects(_pm4ObjectCollection);
    }

    private void DrawPm4ObjectCollectionSummary(string idSuffix)
    {
        PruneMissingPm4CollectionObjects();

        if (!ImGui.CollapsingHeader($"PM4 Collection##{idSuffix}", ImGuiTreeNodeFlags.DefaultOpen))
            return;

        ImGui.TextDisabled($"Parts in collection: {_pm4ObjectCollection.Count}");
        ImGui.TextDisabled("Use this to compare one family against duplicated placements or overlapping copies.");
        ImGui.TextDisabled("Shift+LMB PM4 add is best-effort only; use graph Collect buttons when scene overlap is ambiguous.");

        if (_pm4ObjectCollection.Count == 0)
        {
            ImGui.TextDisabled("No PM4 parts collected yet.");
            return;
        }

        if (ImGui.BeginChild($"Pm4CollectionList##{idSuffix}", new Vector2(0f, 140f), true))
        {
            for (int index = 0; index < _pm4ObjectCollection.Count; index++)
            {
                var key = _pm4ObjectCollection[index];
                bool selected = _worldScene != null
                    && _worldScene.SelectedPm4ObjectKey.HasValue
                    && _worldScene.SelectedPm4ObjectKey.Value == key;

                ImGui.PushID($"Pm4CollectionItem{idSuffix}_{index}");
                if (selected)
                    ImGui.TextColored(new Vector4(1f, 0.95f, 0.35f, 1f), $"{index + 1}. CK24 0x{key.ck24:X6} part={key.objectPart} tile=({key.tileX},{key.tileY})");
                else
                    ImGui.TextUnformatted($"{index + 1}. CK24 0x{key.ck24:X6} part={key.objectPart} tile=({key.tileX},{key.tileY})");

                ImGui.SameLine();
                if (ImGui.SmallButton("Select"))
                    SelectPm4GraphPart(key, frameCamera: false);

                ImGui.SameLine();
                if (ImGui.SmallButton("Frame"))
                    SelectPm4GraphPart(key, frameCamera: true);

                ImGui.SameLine();
                if (ImGui.SmallButton("Remove"))
                {
                    _pm4ObjectCollection.RemoveAt(index);
                    SyncPm4CollectionHighlight();
                    _statusMessage = $"Removed PM4 CK24=0x{key.ck24:X6} part={key.objectPart} from the collection.";
                    ImGui.PopID();
                    break;
                }

                ImGui.PopID();
            }
        }
        ImGui.EndChild();
    }

    private void PruneMissingPm4CollectionObjects()
    {
        if (_worldScene == null)
            return;

        for (int index = _pm4ObjectCollection.Count - 1; index >= 0; index--)
        {
            if (!_worldScene.TryGetPm4ObjectDebugInfo(_pm4ObjectCollection[index], out _))
                _pm4ObjectCollection.RemoveAt(index);
        }

        SyncPm4CollectionHighlight();
    }

    private void ExportPm4ObjectCollectionJson()
    {
        if (_worldScene == null)
            return;

        PruneMissingPm4CollectionObjects();
        if (_pm4ObjectCollection.Count == 0)
        {
            _statusMessage = "PM4 collection export skipped: no collected parts.";
            return;
        }

        string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? "map";
        string defaultName = $"pm4_collection_{mapName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string? picked = ShowSaveFileDialogSTA(
            "Save PM4 Collection JSON",
            "JSON Files (*.json)|*.json|All Files (*.*)|*.*",
            ExportDir,
            defaultName);

        if (string.IsNullOrWhiteSpace(picked))
            return;

        try
        {
            string json = JsonSerializer.Serialize(BuildJsonSafePm4Collection(), new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(picked, json, Encoding.UTF8);
            _statusMessage = $"Exported PM4 collection JSON: {picked}";
        }
        catch (Exception ex)
        {
            _statusMessage = $"PM4 collection export failed: {ex.Message}";
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Collection] JSON export failed: {ex}");
        }
    }

    private object BuildJsonSafePm4Collection()
    {
        if (_worldScene == null)
        {
            return new
            {
                generatedAtUtc = DateTime.UtcNow,
                objectCount = 0,
                objects = Array.Empty<object>()
            };
        }

        var entries = new List<Pm4CollectionExportEntry>(_pm4ObjectCollection.Count);
        foreach (var key in _pm4ObjectCollection)
        {
            if (!_worldScene.TryGetPm4ObjectDebugInfo(key, out Pm4ObjectDebugInfo debugInfo))
                continue;

            _worldScene.TryGetPm4ObjectGroupKey(key, out var mergedGroupKey);
            Vector3 size = debugInfo.BoundsMax - debugInfo.BoundsMin;
            string signature = BuildPm4CollectionSignature(debugInfo, size);
            entries.Add(new Pm4CollectionExportEntry(key, mergedGroupKey, debugInfo, size, signature));
        }

        var signatureGroups = entries
            .GroupBy(static entry => entry.Signature)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key, StringComparer.Ordinal)
            .Select(group => new
            {
                signature = group.Key,
                count = group.Count(),
                ck24 = group.First().DebugInfo.Ck24,
                linkGroupObjectId = group.First().DebugInfo.LinkGroupObjectId,
                members = group.Select(static entry => new
                {
                    tileX = entry.Key.tileX,
                    tileY = entry.Key.tileY,
                    objectPartId = entry.Key.objectPart,
                    center = VectorToArray(entry.DebugInfo.Center)
                }).ToList()
            })
            .ToList();

        Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics> duplicateMetrics = BuildPm4CollectionDuplicateMetrics(entries);
        var stackClusters = BuildPm4CollectionStackClusters(entries, duplicateMetrics);
        string mapName = _terrainManager?.MapName ?? _worldScene.Terrain.MapName ?? string.Empty;

        return new
        {
            generatedAtUtc = DateTime.UtcNow,
            mapName,
            objectCount = entries.Count,
            currentSelection = _worldScene.SelectedPm4ObjectKey.HasValue
                ? new
                {
                    tileX = _worldScene.SelectedPm4ObjectKey.Value.tileX,
                    tileY = _worldScene.SelectedPm4ObjectKey.Value.tileY,
                    ck24 = _worldScene.SelectedPm4ObjectKey.Value.ck24,
                    objectPartId = _worldScene.SelectedPm4ObjectKey.Value.objectPart,
                }
                : null,
            signatureGroupCount = signatureGroups.Count,
            stackClusterCount = stackClusters.Count,
            signatureGroups,
            stackClusters,
            objects = entries.Select(entry => new
            {
                tileX = entry.Key.tileX,
                tileY = entry.Key.tileY,
                ck24 = entry.DebugInfo.Ck24,
                ck24Type = entry.DebugInfo.Ck24Type,
                ck24ObjectId = entry.DebugInfo.Ck24ObjectId,
                objectPartId = entry.Key.objectPart,
                mergedGroupKey = new
                {
                    tileX = entry.GroupKey.tileX,
                    tileY = entry.GroupKey.tileY,
                    ck24 = entry.GroupKey.ck24,
                },
                signature = entry.Signature,
                sameSignatureCount = duplicateMetrics[entry.Key].SameSignatureCount,
                overlapClusterSize = duplicateMetrics[entry.Key].OverlapClusterSize,
                nearestSameSignatureDistance = JsonFiniteOrNull(duplicateMetrics[entry.Key].NearestSameSignatureDistance),
                likelyDuplicateScore = duplicateMetrics[entry.Key].LikelyDuplicateScore,
                linkGroupObjectId = entry.DebugInfo.LinkGroupObjectId,
                linkedPositionRefCount = entry.DebugInfo.LinkedPositionRefCount,
                linkedPositionRefSummary = new
                {
                    totalCount = entry.DebugInfo.LinkedPositionRefSummary.TotalCount,
                    normalCount = entry.DebugInfo.LinkedPositionRefSummary.NormalCount,
                    terminatorCount = entry.DebugInfo.LinkedPositionRefSummary.TerminatorCount,
                    floorMin = entry.DebugInfo.LinkedPositionRefSummary.FloorMin,
                    floorMax = entry.DebugInfo.LinkedPositionRefSummary.FloorMax,
                    headingMinDegrees = JsonFiniteOrNull(entry.DebugInfo.LinkedPositionRefSummary.HeadingMinDegrees),
                    headingMaxDegrees = JsonFiniteOrNull(entry.DebugInfo.LinkedPositionRefSummary.HeadingMaxDegrees),
                    headingMeanDegrees = JsonFiniteOrNull(entry.DebugInfo.LinkedPositionRefSummary.HeadingMeanDegrees)
                },
                surfaceCount = entry.DebugInfo.SurfaceCount,
                dominantGroupKey = entry.DebugInfo.DominantGroupKey,
                dominantAttributeMask = entry.DebugInfo.DominantAttributeMask,
                dominantMdosIndex = entry.DebugInfo.DominantMdosIndex,
                averageSurfaceHeight = JsonFiniteOrNull(entry.DebugInfo.AverageSurfaceHeight),
                boundsMin = VectorToArray(entry.DebugInfo.BoundsMin),
                boundsMax = VectorToArray(entry.DebugInfo.BoundsMax),
                boundsSize = VectorToArray(entry.BoundsSize),
                center = VectorToArray(entry.DebugInfo.Center),
                nearestPositionRefDistance = JsonFiniteOrNull(entry.DebugInfo.NearestPositionRefDistance),
                planar = new
                {
                    swapAxes = entry.DebugInfo.SwapPlanarAxes,
                    invertU = entry.DebugInfo.InvertU,
                    invertV = entry.DebugInfo.InvertV,
                    windingFlip = entry.DebugInfo.InvertsWinding
                }
            }).ToList()
        };
    }

    private static Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics> BuildPm4CollectionDuplicateMetrics(IReadOnlyList<Pm4CollectionExportEntry> entries)
    {
        const float centerTolerance = 2f;
        const float sizeTolerance = 0.5f;
        var metrics = new Dictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics>(entries.Count);

        foreach (var signatureGroup in entries.GroupBy(static entry => entry.Signature))
        {
            List<Pm4CollectionExportEntry> groupEntries = signatureGroup.ToList();
            foreach (Pm4CollectionExportEntry entry in groupEntries)
            {
                float nearestSameSignatureDistance = float.PositiveInfinity;
                int overlapClusterSize = 1;

                for (int i = 0; i < groupEntries.Count; i++)
                {
                    Pm4CollectionExportEntry candidate = groupEntries[i];
                    if (candidate.Key == entry.Key)
                        continue;

                    float distance = Vector3.Distance(entry.DebugInfo.Center, candidate.DebugInfo.Center);
                    if (distance < nearestSameSignatureDistance)
                        nearestSameSignatureDistance = distance;

                    Vector3 sizeDelta = Vector3.Abs(entry.BoundsSize - candidate.BoundsSize);
                    if (distance <= centerTolerance
                        && sizeDelta.X <= sizeTolerance
                        && sizeDelta.Y <= sizeTolerance
                        && sizeDelta.Z <= sizeTolerance)
                    {
                        overlapClusterSize++;
                    }
                }

                if (!float.IsFinite(nearestSameSignatureDistance))
                    nearestSameSignatureDistance = float.NaN;

                int sameSignatureCount = groupEntries.Count;
                float score = sameSignatureCount <= 1
                    ? 0f
                    : MathF.Min(1f,
                        (overlapClusterSize - 1) * 0.45f
                        + (sameSignatureCount - 1) * 0.15f
                        + (float.IsNaN(nearestSameSignatureDistance)
                            ? 0f
                            : MathF.Max(0f, 1f - MathF.Min(nearestSameSignatureDistance, 12f) / 12f) * 0.40f));

                metrics[entry.Key] = new Pm4CollectionDuplicateMetrics(
                    sameSignatureCount,
                    overlapClusterSize,
                    nearestSameSignatureDistance,
                    MathF.Round(score, 3));
            }
        }

        return metrics;
    }

    private static List<object> BuildPm4CollectionStackClusters(
        IReadOnlyList<Pm4CollectionExportEntry> entries,
        IReadOnlyDictionary<(int tileX, int tileY, uint ck24, int objectPart), Pm4CollectionDuplicateMetrics> duplicateMetrics)
    {
        const float centerTolerance = 2f;
        const float sizeTolerance = 0.5f;
        var clusters = new List<object>();

        foreach (var signatureGroup in entries.GroupBy(static entry => entry.Signature))
        {
            List<Pm4CollectionExportEntry> remaining = signatureGroup.ToList();
            while (remaining.Count > 0)
            {
                Pm4CollectionExportEntry seed = remaining[0];
                remaining.RemoveAt(0);

                var cluster = new List<Pm4CollectionExportEntry> { seed };
                for (int index = remaining.Count - 1; index >= 0; index--)
                {
                    Pm4CollectionExportEntry candidate = remaining[index];
                    if (Vector3.Distance(seed.DebugInfo.Center, candidate.DebugInfo.Center) > centerTolerance)
                        continue;

                    Vector3 sizeDelta = Vector3.Abs(seed.BoundsSize - candidate.BoundsSize);
                    if (sizeDelta.X > sizeTolerance || sizeDelta.Y > sizeTolerance || sizeDelta.Z > sizeTolerance)
                        continue;

                    cluster.Add(candidate);
                    remaining.RemoveAt(index);
                }

                if (cluster.Count < 2)
                    continue;

                Vector3 centroid = Vector3.Zero;
                foreach (Pm4CollectionExportEntry entry in cluster)
                    centroid += entry.DebugInfo.Center;
                centroid /= cluster.Count;

                clusters.Add(new
                {
                    signature = seed.Signature,
                    count = cluster.Count,
                    likelyDuplicateScore = cluster.Max(entry => duplicateMetrics[entry.Key].LikelyDuplicateScore),
                    centroid = VectorToArray(centroid),
                    members = cluster.Select(static entry => new
                    {
                        tileX = entry.Key.tileX,
                        tileY = entry.Key.tileY,
                        ck24 = entry.DebugInfo.Ck24,
                        objectPartId = entry.Key.objectPart,
                        center = VectorToArray(entry.DebugInfo.Center)
                    }).ToList()
                });
            }
        }

        return clusters;
    }

    private static string BuildPm4CollectionSignature(Pm4ObjectDebugInfo debugInfo, Vector3 boundsSize)
    {
        return FormattableString.Invariant($"ck24=0x{debugInfo.Ck24:X6}|mslk=0x{debugInfo.LinkGroupObjectId:X8}|surf={debugInfo.SurfaceCount}|g=0x{debugInfo.DominantGroupKey:X2}|a=0x{debugInfo.DominantAttributeMask:X2}|mdos={debugInfo.DominantMdosIndex}|size=({boundsSize.X:F2},{boundsSize.Y:F2},{boundsSize.Z:F2})");
    }

    private static float[] VectorToArray(Vector3 value) => new[] { value.X, value.Y, value.Z };

    private readonly record struct Pm4CollectionExportEntry(
        (int tileX, int tileY, uint ck24, int objectPart) Key,
        (int tileX, int tileY, uint ck24) GroupKey,
        Pm4ObjectDebugInfo DebugInfo,
        Vector3 BoundsSize,
        string Signature);

    private readonly record struct Pm4CollectionDuplicateMetrics(
        int SameSignatureCount,
        int OverlapClusterSize,
        float NearestSameSignatureDistance,
        float LikelyDuplicateScore);

    private static object BuildJsonSafePm4Graph(Pm4SelectedObjectGraphInfo graph)
    {
        return new
        {
            selectedTileX = graph.SelectedTileX,
            selectedTileY = graph.SelectedTileY,
            ck24 = graph.Ck24,
            ck24Type = graph.Ck24Type,
            ck24ObjectId = graph.Ck24ObjectId,
            selectedObjectPartId = graph.SelectedObjectPartId,
            splitByMdos = graph.SplitByMdos,
            splitByConnectivity = graph.SplitByConnectivity,
            tileCount = graph.TileCount,
            linkGroupCount = graph.LinkGroupCount,
            mdosGroupCount = graph.MdosGroupCount,
            partCount = graph.PartCount,
            surfaceCount = graph.SurfaceCount,
            totalIndexCount = graph.TotalIndexCount,
            attributeMaskCount = graph.AttributeMaskCount,
            groupKeyCount = graph.GroupKeyCount,
            linkGroups = graph.LinkGroups.Select(static linkGroup => new
            {
                linkGroupObjectId = linkGroup.LinkGroupObjectId,
                partCount = linkGroup.PartCount,
                surfaceCount = linkGroup.SurfaceCount,
                totalIndexCount = linkGroup.TotalIndexCount,
                linkedPositionRefCount = linkGroup.LinkedPositionRefCount,
                linkedPositionRefSummary = BuildJsonSafeLinkedPositionRefSummary(linkGroup.LinkedPositionRefSummary),
                mdosIndices = linkGroup.MdosIndices,
                attributeMasks = linkGroup.AttributeMasks,
                groupKeys = linkGroup.GroupKeys,
                mdosGroups = linkGroup.MdosGroups.Select(static mdosGroup => new
                {
                    mdosIndex = mdosGroup.MdosIndex,
                    partCount = mdosGroup.PartCount,
                    surfaceCount = mdosGroup.SurfaceCount,
                    totalIndexCount = mdosGroup.TotalIndexCount,
                    attributeMasks = mdosGroup.AttributeMasks,
                    groupKeys = mdosGroup.GroupKeys,
                    parts = mdosGroup.Parts.Select(static part => new
                    {
                        tileX = part.TileX,
                        tileY = part.TileY,
                        objectPartId = part.ObjectPartId,
                        surfaceCount = part.SurfaceCount,
                        totalIndexCount = part.TotalIndexCount,
                        lineCount = part.LineCount,
                        triangleCount = part.TriangleCount,
                        dominantGroupKey = part.DominantGroupKey,
                        dominantAttributeMask = part.DominantAttributeMask,
                        dominantMdosIndex = part.DominantMdosIndex,
                        isSelected = part.IsSelected,
                    }).ToList(),
                }).ToList(),
            }).ToList(),
        };
    }

    private static object BuildJsonSafeLinkedPositionRefSummary(Pm4LinkedPositionRefSummary summary)
    {
        return new
        {
            totalCount = summary.TotalCount,
            normalCount = summary.NormalCount,
            terminatorCount = summary.TerminatorCount,
            floorMin = summary.FloorMin,
            floorMax = summary.FloorMax,
            headingMinDegrees = JsonFiniteOrNull(summary.HeadingMinDegrees),
            headingMaxDegrees = JsonFiniteOrNull(summary.HeadingMaxDegrees),
            headingMeanDegrees = JsonFiniteOrNull(summary.HeadingMeanDegrees),
            hasNormalHeadings = summary.HasNormalHeadings,
        };
    }

    private static float? JsonFiniteOrNull(float value)
    {
        return float.IsFinite(value) ? value : null;
    }
}