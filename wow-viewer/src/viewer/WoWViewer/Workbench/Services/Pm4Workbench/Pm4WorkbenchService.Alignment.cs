using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Globalization;
using ImGuiNET;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WoWViewer.Workbench;
using MslkEntry = WowViewer.Core.PM4.Models.Pm4MslkEntry;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// Pm4WorkbenchService: alignment and WMO-correlation tabs, PM4 exports, object-match reports and saved match selection.
// Pm4WorkbenchService: members moved from ViewerApp_Pm4Utilities.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
// Original file note: Partial class containing PM4 alignment and viewer utility windows.
internal sealed partial class Pm4WorkbenchService
{

    internal void DrawPm4AlignmentContent()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world scene to adjust PM4 alignment.");
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

        if (!_worldScene.Pm4Overlay.HasSelectedPm4Object || !_worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
        {
            ImGui.TextDisabled("No PM4 object selected. Left-click PM4 geometry to pick an object.");
            if (ImGui.Button("Clear PM4 Selection"))
                _worldScene.Pm4Overlay.ClearPm4ObjectSelection();
            ImGui.SameLine();
            if (ImGui.Button("Open Data I/O"))
                OpenWorkbenchTab(WorkbenchTab.Editor, 2); // Spec 231 D1: exports live on the Data I/O page
            ImGui.SameLine();
            if (ImGui.Button("Reconcile Tile"))
            {
                // The old corpus-wide object-match report froze the render thread on whole-map
                // loads; the Reconcile tab runs the Spec 176 tile-scoped pipeline instead.
                _activePm4TabIndex = (int)Pm4BottomTab.Reconcile;
            }
            ImGui.SameLine();
            if (ImGui.Button("PM4/WMO Panel"))
            {
                _activePm4TabIndex = (int)Pm4BottomTab.Correlation;
                EnsurePm4WmoCorrelationReportLoaded();
            }
            return;
        }

        var selectedPm4 = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value;
        uint? selectedLayerCk24 = _worldScene.Pm4Overlay.SelectedPm4RawCk24;
        Vector3 selectedObjectTranslation = _worldScene.Pm4Overlay.SelectedPm4ObjectTranslation;
        Vector3 selectedObjectRotation = _worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees;
        Vector3 selectedObjectScale = _worldScene.Pm4Overlay.SelectedPm4ObjectScale;
        Vector3 selectedLayerTranslation = _worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation;
        Vector3 selectedLayerRotation = _worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees;
        Vector3 selectedLayerScale = _worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale;
        bool translationChanged = false;
        bool rotationChanged = false;
        bool scaleChanged = false;
        bool layerTranslationChanged = false;
        bool layerRotationChanged = false;
        bool layerScaleChanged = false;

        ImGui.Text($"Selected: tile ({selectedPm4.tileX}, {selectedPm4.tileY}) CK24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart}");
        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            ImGui.TextDisabled($"Type=0x{debugInfo.Ck24Type:X2} ObjId={debugInfo.Ck24ObjectId} Surfaces={debugInfo.SurfaceCount}");
            ImGui.TextDisabled($"Group=0x{debugInfo.DominantGroupKey:X2} Attr=0x{debugInfo.DominantAttributeMask:X2} mscnRef={debugInfo.DominantMscnRefIndex} AvgH={debugInfo.AverageSurfaceHeight:F2}");
            ImGui.TextDisabled($"Part={debugInfo.ObjectPartId} MSLKGroup=0x{debugInfo.LinkGroupObjectId:X8}");
            ImGui.TextDisabled($"Linked MPRL refs={debugInfo.LinkedPositionRefCount}");
            if (debugInfo.DistinctTypeFlags != 0)
            {
                List<string> typeFlagLabels = [];
                for (int bit = 1; bit < 32; bit++)
                {
                    if ((debugInfo.DistinctTypeFlags & (1u << bit)) != 0)
                    {
                        string label = bit switch
                        {
                            0x03 => "m2-top(0x03)",
                            0x10 => "interior-floor(0x10)",
                            0x12 => "exterior-solid(0x12)",
                            _ => $"0x{bit:X2}",
                        };
                        typeFlagLabels.Add(label);
                    }
                }
                ImGui.TextDisabled($"TypeFlags: {string.Join(", ", typeFlagLabels)}");
            }
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

        if (selectedLayerCk24.HasValue && _worldScene.Pm4Overlay.TryGetSelectedPm4Ck24LayerStats(out int layerTileCount, out int layerObjectCount))
            ImGui.TextDisabled($"Tile CK24 0x{selectedLayerCk24.Value:X6} on ({selectedPm4.tileX}, {selectedPm4.tileY}): {layerObjectCount} parts across {layerTileCount} tile");

        if (_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectResearchInfo(out Pm4SelectedObjectResearchInfo researchInfo)
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
                    ImGui.BulletText($"{match.Family}#{match.FamilyObjectIndex} score={match.SimilarityScore:F2} surfaces={match.SurfaceCount} indices={match.TotalIndexCount} mscnRef={match.MscnRefCount} groups={match.GroupKeyCount} linkGroups={match.LinkGroupCount} dominant=0x{match.DominantLinkGroupObjectId:X} mode={match.CoordinateMode} planar=(swap={match.PlanarTransform.SwapPlanarAxes},u={match.PlanarTransform.InvertU},v={match.PlanarTransform.InvertV}) yaw={match.FrameYawDegrees:F1}{headingText} linkedMPRL={match.LinkedMprlRefCount}/{match.LinkedMprlInBoundsCount}");
                }
            }

            if (researchInfo.MshdRawFields != null)
            {
                ImGui.Separator();
                ImGui.TextDisabled(researchInfo.MshdRawFields);
            }

            if (researchInfo.MslkRawEntries.Count > 0)
            {
                ImGui.Separator();
                ImGui.TextDisabled($"MSLK entries for this CK24 ({researchInfo.MslkRawEntries.Count}):");
                for (int i = 0; i < researchInfo.MslkRawEntries.Count && i < 16; i++)
                    ImGui.TextDisabled(researchInfo.MslkRawEntries[i]);
                if (researchInfo.MslkRawEntries.Count > 16)
                    ImGui.TextDisabled($"... and {researchInfo.MslkRawEntries.Count - 16} more");
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
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation = selectedLayerTranslation;
            pm4TransformChanged = true;
        }
        if (layerRotationChanged)
        {
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees = NormalizeRotationDegrees(selectedLayerRotation);
            pm4TransformChanged = true;
        }
        if (layerScaleChanged)
        {
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale = selectedLayerScale;
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
            _worldScene.Pm4Overlay.SelectedPm4ObjectTranslation = selectedObjectTranslation;
            pm4TransformChanged = true;
        }
        if (rotationChanged)
        {
            _worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees = NormalizeRotationDegrees(selectedObjectRotation);
            pm4TransformChanged = true;
        }
        if (scaleChanged)
        {
            _worldScene.Pm4Overlay.SelectedPm4ObjectScale = selectedObjectScale;
            pm4TransformChanged = true;
        }

        ImGui.Separator();

        if (ImGui.Button("Reset Layer Move"))
        {
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Layer Rot"))
        {
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Layer Scale"))
        {
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale = Vector3.One;
            pm4TransformChanged = true;
        }

        if (ImGui.Button("Reset Layer 9DoF"))
        {
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation = Vector3.Zero;
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees = Vector3.Zero;
            _worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale = Vector3.One;
            pm4TransformChanged = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Print Layer Alignment") && selectedLayerCk24.HasValue)
        {
            Vector3 t = _worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation;
            Vector3 r = _worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees;
            Vector3 s = _worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale;
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Tile CK24 Align] tile=({selectedPm4.tileX},{selectedPm4.tileY}) ck24=0x{selectedLayerCk24.Value:X6} T=({t.X:F3},{t.Y:F3},{t.Z:F3}) Rot=({r.X:F3},{r.Y:F3},{r.Z:F3}) Scale=({s.X:F4},{s.Y:F4},{s.Z:F4})");
        }

            ImGui.TextDisabled($"Tile Move: ({_worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation.X:F3}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation.Y:F3}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerTranslation.Z:F3})");
            ImGui.TextDisabled($"Tile Rot: ({_worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees.X:F3}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees.Y:F3}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerRotationDegrees.Z:F3}) deg");
            ImGui.TextDisabled($"Tile Scale: ({_worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale.X:F4}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale.Y:F4}, {_worldScene.Pm4Overlay.SelectedPm4Ck24LayerScale.Z:F4})");

        ImGui.Separator();

        if (ImGui.Button("Reset Obj Move"))
        {
            _worldScene.Pm4Overlay.SelectedPm4ObjectTranslation = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Rot"))
        {
            _worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            pm4TransformChanged = true;
        }
        ImGui.SameLine();
        if (ImGui.Button("Reset Obj Scale"))
        {
            _worldScene.Pm4Overlay.SelectedPm4ObjectScale = Vector3.One;
            pm4TransformChanged = true;
        }

        if (ImGui.Button("Reset Obj 9DoF"))
        {
            _worldScene.Pm4Overlay.SelectedPm4ObjectTranslation = Vector3.Zero;
            _worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees = Vector3.Zero;
            _worldScene.Pm4Overlay.SelectedPm4ObjectScale = Vector3.One;
            pm4TransformChanged = true;
        }

        ImGui.SameLine();
        if (ImGui.Button("Clear PM4 Selection"))
            _worldScene.Pm4Overlay.ClearPm4ObjectSelection();

        if (pm4TransformChanged)
            InvalidatePm4DerivedReports();

        if (ImGui.Button("Open Data I/O"))
            OpenWorkbenchTab(WorkbenchTab.Editor, 2); // Spec 231 D1: exports live on the Data I/O page
        ImGui.SameLine();
        if (ImGui.Button("Reconcile Tile"))
        {
            // The old corpus-wide object-match report froze the render thread on whole-map
            // loads; the Reconcile tab runs the Spec 176 tile-scoped pipeline instead.
            _activePm4TabIndex = (int)Pm4BottomTab.Reconcile;
        }
        ImGui.SameLine();
        if (ImGui.Button("PM4/WMO Panel"))
        {
            _activePm4TabIndex = (int)Pm4BottomTab.Correlation;
            EnsurePm4WmoCorrelationReportLoaded();
        }
        ImGui.SameLine();
        if (ImGui.Button("Print Obj Alignment"))
        {
            Vector3 t = _worldScene.Pm4Overlay.SelectedPm4ObjectTranslation;
            Vector3 r = _worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees;
            Vector3 s = _worldScene.Pm4Overlay.SelectedPm4ObjectScale;
            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Obj Align] tile=({selectedPm4.tileX},{selectedPm4.tileY}) ck24=0x{selectedPm4.ck24:X6} part={selectedPm4.objectPart} T=({t.X:F3},{t.Y:F3},{t.Z:F3}) Rot=({r.X:F3},{r.Y:F3},{r.Z:F3}) Scale=({s.X:F4},{s.Y:F4},{s.Z:F4})");
        }

        ImGui.TextDisabled($"Obj Move: ({_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.X:F3}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Y:F3}, {_worldScene.Pm4Overlay.SelectedPm4ObjectTranslation.Z:F3})");
        ImGui.TextDisabled($"Obj Rot: ({_worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees.X:F3}, {_worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees.Y:F3}, {_worldScene.Pm4Overlay.SelectedPm4ObjectRotationDegrees.Z:F3}) deg");
        ImGui.TextDisabled($"Obj Scale: ({_worldScene.Pm4Overlay.SelectedPm4ObjectScale.X:F4}, {_worldScene.Pm4Overlay.SelectedPm4ObjectScale.Y:F4}, {_worldScene.Pm4Overlay.SelectedPm4ObjectScale.Z:F4})");
    }

    internal void DrawPm4WmoCorrelationContent()
    {
        if (_worldScene == null)
        {
            _pm4WmoCorrelationReport = null;
            ImGui.TextDisabled("Load a world scene to inspect PM4/WMO correlation.");
            return;
        }

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

        ImGui.SameLine();
        ImGui.SetNextItemWidth(260f);
        ImGui.InputTextWithHint("##Pm4WmoCorrelationFilter", "Filter model name or path", ref _pm4WmoCorrelationModelFilter, 256);

        if (_pm4WmoCorrelationReport == null)
        {
            ImGui.TextDisabled("No PM4/WMO correlation report is loaded.");
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
                        $"surfaces={match.SurfaceCount} linked refs={match.LinkedPositionRefCount} mscnRef={match.DominantMscnRefIndex} avgH={match.AverageSurfaceHeight:F2}");
                    ImGui.Separator();
                }

                ImGui.EndChild();
            }
        }
        ImGui.EndChild();
    }

    private void SaveCurrentPm4Alignment()
    {
        if (_worldScene == null)
            return;

        _pm4SavedOverlayTranslation = _worldScene.Pm4Overlay.Pm4OverlayTranslation;
        _pm4SavedOverlayRotationDegrees = _worldScene.Pm4Overlay.Pm4OverlayRotationDegrees;
        _pm4SavedOverlayScale = _worldScene.Pm4Overlay.Pm4OverlayScale;
        _settings.SaveViewerSettings();

        _statusMessage = $"Saved PM4 alignment: T=({_pm4SavedOverlayTranslation.X:F2}, {_pm4SavedOverlayTranslation.Y:F2}, {_pm4SavedOverlayTranslation.Z:F2}) Rot=({_pm4SavedOverlayRotationDegrees.X:F2}, {_pm4SavedOverlayRotationDegrees.Y:F2}, {_pm4SavedOverlayRotationDegrees.Z:F2})° S=({_pm4SavedOverlayScale.X:F3}, {_pm4SavedOverlayScale.Y:F3}, {_pm4SavedOverlayScale.Z:F3})";
    }

    /// <summary>
    /// Spec 231 D1: single authoritative draw site for the PM4 export command set.
    /// The former duplicate clusters in the PM4 selection/transform/info panels are
    /// now "Open Data I/O" links; the commands themselves are drawn only here, on
    /// the Editor > Data I/O page.
    /// </summary>
    internal void DrawPm4ExportCommandSet()
    {
        if (_worldScene == null)
        {
            ImGui.TextDisabled("Load a world with PM4 data to export.");
            return;
        }

        if (ImGui.Button("Dump PM4 Objects JSON"))
            ExportPm4ObjectsJson();
        ImGui.SameLine();
        if (ImGui.Button("Export PM4 OBJ Set"))
            ExportPm4ObjectsObjSet();
        ImGui.SameLine();
        if (ImGui.Button("Export PM4 LLM Bundle"))
            ExportPm4LlmEvidenceBundle();

        if (ImGui.Button("Export Visible PM4 Report"))
            ExportPm4OverlayReport();
        ImGui.SameLine();
        if (ImGui.Button("Dump PM4/WMO Correlation JSON"))
            ExportPm4WmoCorrelationJson();
    }

    internal void ExportPm4ObjectsJson()
    {
        if (_worldScene == null)
            return;

        if (_pm4JsonExportRunning)
        {
            _statusMessage = "A PM4 JSON export is already running.";
            return;
        }

        string defaultName = $"pm4_objects_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        ImGuiPathPicker.Instance.Open(
            "Save PM4 Objects JSON",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".json",
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                // Spec 231 D6: the interchange JSON includes full geometry, which walks
                // every loaded PM4 file — run it off the render thread with a re-entrancy
                // guard so the viewer stays responsive during the dump.
                WorldScene scene = _worldScene;
                _pm4JsonExportRunning = true;
                _statusMessage = "Exporting PM4 objects JSON… (running in background; the viewer stays responsive)";
                _ = Task.Run(() =>
                {
                    try
                    {
                        string json = scene.Pm4Overlay.BuildPm4OverlayInterchangeJson(includeGeometry: true);
                        File.WriteAllText(picked, json, Encoding.UTF8);
                        _statusMessage = $"Exported PM4 objects JSON: {picked}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"PM4 JSON export failed: {ex.Message}";
                        ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] JSON export failed: {ex}");
                    }
                    finally
                    {
                        _pm4JsonExportRunning = false;
                    }
                });
            },
            defaultName);
    }

    /// <summary>True while a background PM4 OBJ export is in flight; guards re-entrancy.</summary>
    private bool _pm4ObjExportRunning;

    /// <summary>
    /// Spec 231 D6: true while a background PM4 JSON dump (objects JSON or PM4/WMO
    /// correlation JSON) is in flight; guards re-entrancy like <see cref="_pm4ObjExportRunning"/>.
    /// </summary>
    private bool _pm4JsonExportRunning;

    internal void ExportPm4ObjectsObjSet()
    {
        if (_worldScene == null)
            return;

        if (_pm4ObjExportRunning)
        {
            _statusMessage = "PM4 OBJ export is already running.";
            return;
        }

        Directory.CreateDirectory(ExportDir);
        ImGuiPathPicker.Instance.Open(
            "Choose a folder for PM4 OBJ export",
            pickFolder: true,
            initialPath: ExportDir,
            filterExtension: null,
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                // The export re-reads and re-decodes every PM4 file on the map and writes one
                // OBJ per object. Run it off the render thread — synchronously it stalled the
                // frame loop for the whole export and the window appeared frozen. The MPQ data
                // source is safe for concurrent reads: terrain streaming already reads it from
                // background workers (MaxConcurrentMpqReads semaphore in TerrainManager).
                WorldScene scene = _worldScene;
                _pm4ObjExportRunning = true;
                _statusMessage = "Exporting PM4 OBJ set… (running in background; the viewer stays responsive)";
                _ = Task.Run(() =>
                {
                    try
                    {
                        Pm4OfflineObjExportSummary summary = scene.Pm4Overlay.ExportPm4ObjectsAsObjDirectory(picked);
                        _statusMessage =
                            $"Exported PM4 OBJ set: {summary.ExportedObjectCount} objects across {summary.ExportedTileCount} tiles to {summary.OutputDirectory} (manifest: {summary.ManifestPath}).";
                        ViewerLog.Info(ViewerLog.Category.Terrain,
                            $"[PM4 Export] OBJ set finished: {summary.ExportedObjectCount} objects, {summary.ExportedTileCount} tiles -> {summary.OutputDirectory}");
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"PM4 OBJ export failed: {ex.Message}";
                        ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] OBJ export failed: {ex}");
                    }
                    finally
                    {
                        _pm4ObjExportRunning = false;
                    }
                });
            });
    }

    private void ExportPm4WmoCorrelationJson()
    {
        if (_worldScene == null)
            return;

        if (_pm4JsonExportRunning)
        {
            _statusMessage = "A PM4 JSON export is already running.";
            return;
        }

        string defaultName = $"pm4_wmo_correlation_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        ImGuiPathPicker.Instance.Open(
            "Save PM4/WMO Correlation JSON",
            ImGuiPathPickerMode.SaveFile,
            ExportDir,
            ".json",
            picked =>
            {
                if (string.IsNullOrWhiteSpace(picked))
                    return;

                // Spec 231 D6: the correlation build walks every placement in the loaded
                // PM4 corpus — background it with a re-entrancy guard like the OBJ export.
                WorldScene scene = _worldScene;
                _pm4JsonExportRunning = true;
                _statusMessage = "Exporting PM4/WMO correlation JSON… (running in background; the viewer stays responsive)";
                _ = Task.Run(() =>
                {
                    try
                    {
                        string json = scene.Pm4Overlay.BuildPm4WmoPlacementCorrelationJson();
                        File.WriteAllText(picked, json, Encoding.UTF8);
                        _statusMessage = $"Exported PM4/WMO correlation JSON: {picked}";
                    }
                    catch (Exception ex)
                    {
                        _statusMessage = $"PM4/WMO correlation export failed: {ex.Message}";
                        ViewerLog.Error(ViewerLog.Category.Terrain, $"[PM4 Export] Correlation export failed: {ex}");
                    }
                    finally
                    {
                        _pm4JsonExportRunning = false;
                    }
                });
            },
            defaultName);
    }

    internal void InvalidatePm4DerivedReports()
    {
        _pm4ObjectMatchReport = null;
        _selectedPm4ObjectMatch = null;
        _selectedPm4ObjectMatchKey = null;
        _selectedPm4ObjectMatchCacheMaxMatches = -1;
        _hoveredPm4ObjectMatch = null;
        _hoveredPm4ObjectMatchKey = null;
        _hoveredPm4ObjectMatchCacheMaxMatches = -1;
        _pm4WmoCorrelationReport = null;
        _pm4OutlineCache = null;
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
            _pm4WmoCorrelationReport = _worldScene.Pm4Overlay.BuildPm4WmoPlacementCorrelationReport(_pm4WmoCorrelationMaxMatchesPerPlacement);
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
            _pm4ObjectMatchReport = _worldScene.Pm4Overlay.BuildPm4ObjectMatchReport(_pm4ObjectMatchMaxMatchesPerObject);
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

    private bool TryGetSelectedPm4ObjectMatch(out Pm4ObjectMatchObject objectMatch)
    {
        objectMatch = null!;

        if (_worldScene == null || !_worldScene.Pm4Overlay.SelectedPm4ObjectKey.HasValue)
            return false;

        var selectedKey = _worldScene.Pm4Overlay.SelectedPm4ObjectKey.Value;
        if (_selectedPm4ObjectMatch != null
            && _selectedPm4ObjectMatchKey.HasValue
            && _selectedPm4ObjectMatchKey.Value == selectedKey
            && _selectedPm4ObjectMatchCacheMaxMatches == _pm4ObjectMatchMaxMatchesPerObject)
        {
            objectMatch = _selectedPm4ObjectMatch;
            return true;
        }

        if (!_worldScene.Pm4Overlay.TryBuildSelectedPm4ObjectMatch(_pm4ObjectMatchMaxMatchesPerObject, out Pm4ObjectMatchObject selectedMatch))
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


    private void SelectPm4ObjectMatchObject(Pm4ObjectMatchObject objectMatch, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.Pm4Overlay.SelectPm4Object((objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId)))
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

        _settings.SaveViewerSettings();
        _statusMessage = $"Saved PM4 object match: CK24=0x{objectMatch.Ck24:X6} part={objectMatch.ObjectPartId} -> {candidate.Kind} uid={candidate.UniqueId}.";
    }

    private void ClearSavedPm4ObjectMatch(Pm4ObjectMatchObject objectMatch)
    {
        string mapName = _terrainManager?.MapName ?? _worldScene?.Terrain.MapName ?? string.Empty;
        string key = BuildSavedPm4ObjectMatchKey(mapName, objectMatch.TileX, objectMatch.TileY, objectMatch.Ck24, objectMatch.ObjectPartId);
        if (_savedPm4ObjectMatches.Remove(key))
        {
            _settings.SaveViewerSettings();
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

    internal static string BuildSavedPm4ObjectMatchKey(string mapName, int tileX, int tileY, uint ck24, int objectPartId)
    {
        return $"{mapName.Trim().ToLowerInvariant()}|{tileX}|{tileY}|{ck24:X6}|{objectPartId}";
    }

    private void SelectPm4CorrelationMatch(Pm4WmoCorrelationMatch match, bool frameCamera)
    {
        if (_worldScene == null)
            return;

        if (_worldScene.Pm4Overlay.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
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

        if (!_worldScene.Pm4Overlay.SelectPm4Object((match.TileX, match.TileY, match.Ck24, match.ObjectPartId)))
        {
            _statusMessage = $"PM4 candidate CK24=0x{match.Ck24:X6} part={match.ObjectPartId} is no longer available.";
            return;
        }

        if (!_worldScene.Pm4Overlay.TryGetSelectedPm4ObjectDebugInfo(out Pm4ObjectDebugInfo debugInfo))
        {
            _statusMessage = "PM4 snap failed: selected object debug info is unavailable.";
            return;
        }

        Vector3 placementCenter = (placement.WorldBoundsMin + placement.WorldBoundsMax) * 0.5f;
        Vector3 delta = placementCenter - debugInfo.Center;
        if (!includeZ)
            delta.Z = 0f;

        _worldScene.Pm4Overlay.SelectedPm4ObjectTranslation += delta;
        InvalidatePm4DerivedReports();
        _activePm4TabIndex = (int)Pm4BottomTab.Alignment;

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

    internal void ApplySavedPm4AlignmentToScene()
    {
        if (_worldScene == null)
            return;

        _worldScene.Pm4Overlay.Pm4OverlayTranslation = _pm4SavedOverlayTranslation;
        _worldScene.Pm4Overlay.Pm4OverlayRotationDegrees = _pm4SavedOverlayRotationDegrees;
        _worldScene.Pm4Overlay.Pm4OverlayScale = _pm4SavedOverlayScale;
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
}
