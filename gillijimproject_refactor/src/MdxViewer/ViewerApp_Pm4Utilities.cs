using System.Numerics;
using System.Text;
using ImGuiNET;
using MdxViewer.Logging;
using MdxViewer.Terrain;

namespace MdxViewer;

/// <summary>
/// Partial class containing PM4 alignment and viewer utility windows.
/// </summary>
public partial class ViewerApp
{
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

        ImGui.TextWrapped("Object-local PM4 alignment only. Select one PM4 object, then adjust translation, rotation, and scale (including axis flips) on that object only.");
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
        Vector3 selectedObjectTranslation = _worldScene.SelectedPm4ObjectTranslation;
        Vector3 selectedObjectRotation = _worldScene.SelectedPm4ObjectRotationDegrees;
        Vector3 selectedObjectScale = _worldScene.SelectedPm4ObjectScale;
        bool translationChanged = false;
        bool rotationChanged = false;
        bool scaleChanged = false;

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
                    ImGui.BulletText($"{match.Family}#{match.FamilyObjectIndex} score={match.SimilarityScore:F2} surfaces={match.SurfaceCount} indices={match.TotalIndexCount} mdos={match.MdosCount} groups={match.GroupKeyCount} linkedMPRL={match.LinkedMprlRefCount}/{match.LinkedMprlInBoundsCount}");
                }
            }
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

        ImGui.Text("Axis Flips:");
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

        if (translationChanged)
            _worldScene.SelectedPm4ObjectTranslation = selectedObjectTranslation;
        if (rotationChanged)
            _worldScene.SelectedPm4ObjectRotationDegrees = NormalizeRotationDegrees(selectedObjectRotation);
        if (scaleChanged)
            _worldScene.SelectedPm4ObjectScale = selectedObjectScale;

        ImGui.Separator();

        if (ImGui.Button("Reset Obj Move"))
            _worldScene.SelectedPm4ObjectTranslation = Vector3.Zero;
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Rot"))
            _worldScene.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Scale"))
            _worldScene.SelectedPm4ObjectScale = Vector3.One;

        if (ImGui.Button("Reset Obj 9DoF"))
        {
            _worldScene.SelectedPm4ObjectTranslation = Vector3.Zero;
            _worldScene.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            _worldScene.SelectedPm4ObjectScale = Vector3.One;
        }

        ImGui.SameLine();
        if (ImGui.Button("Clear PM4 Selection"))
            _worldScene.ClearPm4ObjectSelection();

        if (ImGui.Button("Dump PM4 Objects JSON"))
            ExportPm4ObjectsJson();
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

    private void SelectPm4CorrelationMatch(Pm4WmoCorrelationMatch match, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
        {
            _showPm4AlignmentWindow = true;
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

    private static string GetPm4ColorModeLabel(Pm4OverlayColorMode mode)
    {
        return mode switch
        {
            Pm4OverlayColorMode.Ck24Type => "CK24 Type",
            Pm4OverlayColorMode.Ck24ObjectId => "CK24 Object Id",
            Pm4OverlayColorMode.Ck24Key => "CK24 Full Key",
            Pm4OverlayColorMode.Tile => "Tile",
            Pm4OverlayColorMode.GroupKey => "MSUR GroupKey",
            Pm4OverlayColorMode.AttributeMask => "MSUR Attr Mask",
            Pm4OverlayColorMode.Height => "Height Gradient",
            _ => mode.ToString()
        };
    }
}