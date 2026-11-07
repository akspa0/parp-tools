using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using ImGuiNET;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WoWViewer.Terrain;
using WoWViewer.Terrain.Pm4;
using WoWViewer.UI;
using WoWViewer.Workbench.Services.WorkbenchPanels;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

/// <summary>
/// Spec 295 Phase 4–5 (T015, T016, T019, T022): Renders the calibrated -X offset controls,
/// 3D proposal preview trigger, human-in-the-loop review queue (Accept, Reject, Replace with
/// visual library alternatives), opt-in bounded auto-accept, and pairing audit history.
/// </summary>
internal sealed class Pm4ShapeMatchReviewPanelService
{
    private int _selectedAlternativeIndex;

    public void Draw(
        Pm4ShapeMatchWorkbenchService workbench,
        ManualCurationWorkbenchService curationService,
        Pm4OverlayScene overlay,
        IViewerAppHost host,
        WorldScene? worldScene,
        Pm4OverlayObject? selectedHullObj,
        int activeTileX,
        int activeTileY,
        string mapName)
    {
        if (!SharedUiWidgets.SectionHeader("Shape-Match Proposals & Review Queue", defaultOpen: true, id: "Pm4ShapeMatchReviewSection"))
            return;

        workbench.EnsureLibraryInitialized(worldScene);
        ManualCurationProject project = curationService.GetProject(mapName);

        // 1. Calibrated -X Centroid Offset Controls (FR-005, T015, T016)
        Pm4CentroidCalibration activeCal = workbench.ActiveCalibration;
        ImGui.TextColored(UiTokens.Brass, "Centroid Calibration (-X Offset):");
        ImGui.TextDisabled($"Active Offset: ({activeCal.Offset.X:+0.000;-0.000;0.000}, {activeCal.Offset.Y:+0.000;-0.000;0.000}, {activeCal.Offset.Z:+0.000;-0.000;0.000}) yd — {activeCal.Source}");

        bool calEnabled = workbench.CalibrationEnabled;
        if (ImGui.Checkbox("Apply Calibrated -X Offset##Pm4CalibEnable", ref calEnabled))
        {
            workbench.CalibrationEnabled = calEnabled;
            if (calEnabled)
                workbench.SetCustomCalibrationOffset(workbench.CustomCalibrationOffset, worldScene);
            else
                workbench.ZeroCalibration(worldScene);
        }

        if (SharedUiWidgets.ActionButton("Reset Calibration##Pm4CalibReset"))
        {
            workbench.ResetCalibrationToMeasured(worldScene);
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Zero Offset##Pm4CalibZero"))
        {
            workbench.ZeroCalibration(worldScene);
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Invert Offset##Pm4CalibInvert"))
        {
            workbench.InvertCalibration(worldScene);
        }

        // 2. Proposal Generation & 3D Viewport Preview (T014, T015)
        ImGui.Spacing();
        if (SharedUiWidgets.ActionButton("Propose for Selected Hull##Pm4ProposeSelected", selectedHullObj != null))
        {
            if (selectedHullObj != null)
            {
                workbench.AnalyzeHull(worldScene, selectedHullObj, activeTileX, activeTileY, project);
            }
        }
        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Propose for Current Tile##Pm4ProposeTile", worldScene != null))
        {
            if (worldScene != null)
            {
                workbench.AnalyzeTileHulls(worldScene, activeTileX, activeTileY, project);
            }
        }

        Pm4ShapeMatchReviewService review = workbench.Review;
        ImGui.TextDisabled(
            $"Queue: {review.Queue.Count} total | Pending: {review.PendingCount} | Accepted: {review.AcceptedCount} | Replaced: {review.ReplacedCount} | Rejected: {review.RejectedCount} | Unmatched: {review.UnmatchedCount}");

        if (review.Queue.Count > 0)
        {
            if (ImGui.BeginChild("Pm4ReviewQueueList", new Vector2(0f, 150f), true))
            {
                for (int i = 0; i < review.Queue.Count; i++)
                {
                    Pm4ReviewQueueItem item = review.Queue[i];
                    bool isSelected = i == review.SelectedQueueIndex;
                    string candidateName = item.CommittedModelPath != null
                        ? Path.GetFileName(item.CommittedModelPath)
                        : item.ActiveTopCandidate != null
                            ? Path.GetFileName(item.ActiveTopCandidate.ModelPath)
                            : "(unmatched)";
                    string confText = item.ActiveTopCandidate != null
                        ? $"{item.ActiveTopCandidate.Confidence:P0}"
                        : "-";
                    string rowText = $"[{item.Status}] Tile {item.Hull.TileX}_{item.Hull.TileY} CK24 0x{item.Hull.Ck24:X6} (part {item.Hull.ObjectPartId}) → {candidateName} ({confText})##rev_q_{i}";

                    if (ImGui.Selectable(rowText, isSelected))
                    {
                        review.SelectIndex(i);
                    }
                }
            }
            ImGui.EndChild();
        }

        if (review.SelectedItem is { } selectedItem)
        {
            ImGui.Spacing();
            ImGui.TextColored(
                UiTokens.Brass,
                $"Selected Hull: Tile ({selectedItem.Hull.TileX}, {selectedItem.Hull.TileY}) CK24 0x{selectedItem.Hull.Ck24:X6} Part {selectedItem.Hull.ObjectPartId}");
            Vector3 rawC = selectedItem.Proposal.UncalibratedCentroid;
            Vector3 calC = selectedItem.Proposal.CalibratedCentroid;
            ImGui.TextDisabled(
                $"Uncalibrated: ({rawC.X:F2}, {rawC.Y:F2}, {rawC.Z:F2}) → Calibrated: ({calC.X:F2}, {calC.Y:F2}, {calC.Z:F2})");

            string? previewPath = selectedItem.CommittedModelPath ?? selectedItem.ActiveTopCandidate?.ModelPath;
            string previewKind = selectedItem.ActiveTopCandidate?.AssetKind ?? "WMO";

            if (SharedUiWidgets.ActionButton("Preview 3D Proposal##Pm4Preview3dProposal", !string.IsNullOrWhiteSpace(previewPath)))
            {
                if (!string.IsNullOrWhiteSpace(previewPath))
                {
                    workbench.PreviewCandidateOnHull(worldScene, host, selectedItem.HullInput, previewPath, previewKind);
                }
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Clear 3D Preview##Pm4Clear3dPreview", workbench.ActivePreview != null))
            {
                workbench.ClearViewportPreviewInstance(worldScene);
            }

            // 3. Review Actions: Accept, Reject, Replace (FR-007, FR-008, FR-009, FR-010)
            if (SharedUiWidgets.ActionButton("Accept Proposal##Pm4ReviewAccept", selectedItem.ActiveTopCandidate != null))
            {
                if (review.Accept(selectedItem.Hull, project))
                {
                    curationService.SaveProject(mapName);
                    workbench.StatusMessage = $"Accepted {Path.GetFileName(selectedItem.ActiveTopCandidate!.ModelPath)} for CK24 0x{selectedItem.Hull.Ck24:X6}.";
                }
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Reject Proposal##Pm4ReviewReject", selectedItem.ActiveTopCandidate != null))
            {
                string rejectedName = Path.GetFileName(selectedItem.ActiveTopCandidate!.ModelPath);
                if (review.Reject(selectedItem.Hull, project))
                {
                    curationService.SaveProject(mapName);
                    workbench.StatusMessage = $"Rejected {rejectedName} for CK24 0x{selectedItem.Hull.Ck24:X6} (suppressed from top match).";
                }
            }
            ImGui.SameLine();
            if (SharedUiWidgets.ActionButton("Reset Rejected Pairs##Pm4ReviewResetRejects"))
            {
                int cleared = review.ResetRejections(project, selectedItem.Hull);
                curationService.SaveProject(mapName);
                workbench.StatusMessage = $"Reset {cleared} rejected candidate pair(s) for CK24 0x{selectedItem.Hull.Ck24:X6}.";
            }

            // Replace from selected visual library shape or compatible alternatives (FR-008)
            var selectedLibEntry = workbench.Library.SelectedEntry;
            if (SharedUiWidgets.ActionButton("Replace with Selected Library Shape##Pm4ReviewReplaceLib", selectedLibEntry is { IsHealthy: true }))
            {
                if (selectedLibEntry != null &&
                    review.Replace(selectedItem.Hull, selectedLibEntry.PrimaryModelPath, selectedLibEntry.AssetKind, project))
                {
                    curationService.SaveProject(mapName);
                    workbench.StatusMessage = $"Replaced proposal with {selectedLibEntry.DisplayName} and saved curation project.";
                }
            }

            var alternatives = review.GetReplacementAlternatives(selectedItem.HullInput, workbench.Library, project, maxResults: 8);
            if (alternatives.Count > 0)
            {
                _selectedAlternativeIndex = Math.Clamp(_selectedAlternativeIndex, 0, alternatives.Count - 1);
                ImGui.TextDisabled($"Compatible Proportion Alternatives ({alternatives.Count}):");
                for (int a = 0; a < Math.Min(alternatives.Count, 5); a++)
                {
                    var alt = alternatives[a];
                    string altRow = $"Use {alt.Entry.DisplayName} ({alt.Entry.Preview.ProportionSummary}) — Conf {alt.Confidence:P0}##alt_pick_{a}";
                    if (ImGui.Selectable(altRow, a == _selectedAlternativeIndex))
                    {
                        _selectedAlternativeIndex = a;
                        workbench.Library.SelectEntry(alt.Entry.EntryKey);
                        if (review.Replace(selectedItem.Hull, alt.Entry.PrimaryModelPath, alt.Entry.AssetKind, project))
                        {
                            curationService.SaveProject(mapName);
                            workbench.StatusMessage = $"Replaced with proportion-compatible alternative {alt.Entry.DisplayName}.";
                        }
                    }
                }
            }
        }

        // 4. Opt-in Bounded Auto-Accept Mode (T022, SC-006)
        ImGui.Spacing();
        bool autoAccept = review.BoundedAutoAcceptEnabled;
        if (ImGui.Checkbox("Enable Bounded Auto-Accept (This Run Only)##Pm4BoundedAutoToggle", ref autoAccept))
        {
            review.BoundedAutoAcceptEnabled = autoAccept;
        }
        if (review.BoundedAutoAcceptEnabled)
        {
            float threshold = review.BoundedAutoAcceptThreshold;
            if (ImGui.SliderFloat("Auto-Accept Min Confidence##Pm4BoundedAutoThreshold", ref threshold, 0.50f, 0.99f))
            {
                review.BoundedAutoAcceptThreshold = threshold;
            }
            if (SharedUiWidgets.ActionButton("Run Bounded Auto-Accept##Pm4BoundedAutoRun"))
            {
                int count = review.RunBoundedAutoAccept(project, workbench.Batch.LastSummary?.Conflicts);
                if (count > 0)
                    curationService.SaveProject(mapName);
                workbench.StatusMessage = $"Bounded auto-accept committed {count} high-confidence non-conflicting proposal(s).";
            }
        }

        // 5. Audit History (FR-011, SC-003)
        if (SharedUiWidgets.SectionHeader("Pairing Audit History", defaultOpen: false, id: "Pm4ShapeAuditHistorySection"))
        {
            var history = project.ShapeMatchAuditHistory;
            if (history == null || history.Count == 0)
            {
                ImGui.TextDisabled("No pairing decisions recorded yet for this map.");
            }
            else
            {
                ImGui.TextDisabled($"{history.Count} decision(s) persisted in {mapName}.manual-curation.json:");
                for (int i = history.Count - 1; i >= Math.Max(0, history.Count - 15); i--)
                {
                    ManualShapeMatchAuditEntry entry = history[i];
                    string propName = Path.GetFileName(entry.ProposedModelPath);
                    string chosenName = entry.ChosenModelPath != null ? Path.GetFileName(entry.ChosenModelPath) : "-";
                    ImGui.BulletText(
                        $"[{entry.Decision}] {entry.Hull.TileX}_{entry.Hull.TileY} 0x{entry.Hull.Ck24:X6}: proposed={propName}, chosen={chosenName}, conf={entry.Confidence:P0}, prov={entry.Provenance} ({entry.TimestampUtc:HH:mm:ss})");
                }
            }
        }

        if (!string.IsNullOrWhiteSpace(workbench.StatusMessage))
        {
            SharedUiWidgets.StatusReadout("Shape Match", workbench.StatusMessage);
        }
    }
}

