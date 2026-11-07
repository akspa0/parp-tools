using System;
using System.IO;
using System.Numerics;
using System.Threading;
using ImGuiNET;
using WowViewer.Core.Maps;
using WoWViewer.Terrain;
using WoWViewer.UI;
using WoWViewer.Workbench.Services.WorkbenchPanels;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

/// <summary>
/// Spec 295 Phase 6 (T024): Batch shape-match analysis panel with tile-range selection,
/// progress bar, cancellation preserving accepted decisions (FR-014), reconciled summary
/// counts (Matched, Low-Confidence, Unmatched, Conflicts, SC-004), and JSON+Markdown export (FR-017).
/// </summary>
internal sealed class Pm4ShapeMatchBatchPanelService
{
    private int _minTileX;
    private int _maxTileX = 63;
    private int _minTileY;
    private int _maxTileY = 63;
    private CancellationTokenSource? _activeCts;

    public void Draw(
        Pm4ShapeMatchWorkbenchService workbench,
        ManualCurationWorkbenchService curationService,
        WorldScene? worldScene,
        string mapName)
    {
        if (!SharedUiWidgets.SectionHeader("Batch Shape-Match Analysis & Audit Export", defaultOpen: false, id: "Pm4ShapeMatchBatchSection"))
            return;

        workbench.EnsureLibraryInitialized(worldScene);
        Pm4ShapeMatchBatchService batch = workbench.Batch;
        ManualCurationProject project = curationService.GetProject(mapName);

        ImGui.SliderInt("Batch Min Tile X##Pm4BatchMinX", ref _minTileX, 0, 63);
        ImGui.SliderInt("Batch Max Tile X##Pm4BatchMaxX", ref _maxTileX, 0, 63);
        ImGui.SliderInt("Batch Min Tile Y##Pm4BatchMinY", ref _minTileY, 0, 63);
        ImGui.SliderInt("Batch Max Tile Y##Pm4BatchMaxY", ref _maxTileY, 0, 63);

        if (_minTileX > _maxTileX) _maxTileX = _minTileX;
        if (_minTileY > _maxTileY) _maxTileY = _minTileY;

        if (SharedUiWidgets.ActionButton("Run Batch Shape Analysis##Pm4BatchRun", worldScene != null && !batch.IsRunning))
        {
            _activeCts?.Dispose();
            _activeCts = new CancellationTokenSource();

            var tiles = workbench.CollectLoadedBatchTiles(worldScene);
            var scope = new Pm4BatchTileScope(mapName, _minTileX, _maxTileX, _minTileY, _maxTileY);
            Pm4BatchAnalysisSummary summary = batch.RunBatch(
                scope,
                tiles,
                workbench.Library.GetHealthyModelCandidates(),
                project,
                workbench.Review,
                workbench.ActiveCalibration,
                cancellationToken: _activeCts.Token);

            workbench.StatusMessage =
                $"Batch complete ({summary.ProcessedTilesCount}/{summary.TotalTilesCount} tiles): Matched={summary.MatchedCount}, LowConf={summary.LowConfidenceCount}, Unmatched={summary.UnmatchedCount}, Conflicts={summary.ConflictCount}.";
        }

        ImGui.SameLine();
        if (SharedUiWidgets.ActionButton("Cancel Batch Run##Pm4BatchCancel", batch.IsRunning))
        {
            _activeCts?.Cancel();
        }

        ImGui.ProgressBar(
            batch.ProgressFraction,
            new Vector2(-1f, 0f),
            $"{batch.ProcessedTilesCount} / {batch.TotalTilesCount} tiles");

        if (batch.LastSummary is { } last)
        {
            ImGui.Spacing();
            ImGui.TextColored(
                last.WasCancelled ? UiTokens.Brass : UiTokens.Success,
                last.WasCancelled ? "Batch Status: Cancelled (Accepted decisions preserved)" : "Batch Status: Completed");
            ImGui.BulletText($"Total Hulls: {last.TotalHulls}");
            ImGui.BulletText($"Matched: {last.MatchedCount} (Preserved Accepted: {last.PreservedAcceptedCount})");
            ImGui.BulletText($"Low-Confidence: {last.LowConfidenceCount}");
            ImGui.BulletText($"Unmatched: {last.UnmatchedCount}");
            ImGui.BulletText($"Conflicts: {last.ConflictCount}");

            if (last.Conflicts.Count > 0)
            {
                ImGui.TextColored(UiTokens.Brass, $"Detected Conflicts ({last.Conflicts.Count}):");
                for (int c = 0; c < Math.Min(last.Conflicts.Count, 5); c++)
                {
                    var conflict = last.Conflicts[c];
                    float zA = conflict.FirstHull.PlacementZ;
                    float zB = conflict.SecondHull.PlacementZ;
                    string tagA = $"[{conflict.FirstHull.TileX}_{conflict.FirstHull.TileY}] Obj #{conflict.FirstHull.ObjectPartId} (Z={zA:F1}yd)";
                    string tagB = $"[{conflict.SecondHull.TileX}_{conflict.SecondHull.TileY}] Obj #{conflict.SecondHull.ObjectPartId} (Z={zB:F1}yd)";
                    ImGui.BulletText(
                        $"{tagA} vs {tagB} claiming {Path.GetFileName(conflict.ModelPath)} ({conflict.DistanceBetweenHulls:F1} yds apart)");
                }
            }

            if (SharedUiWidgets.ActionButton("Export Batch Summary (JSON + MD)##Pm4BatchExport"))
            {
                string outputDir = Path.Combine(AppContext.BaseDirectory, "output", "projects", "pm4_pairings");
                var (jsonPath, mdPath) = batch.ExportSummaryFiles(last, outputDir, $"{mapName}-pm4-shape-match-summary");
                workbench.StatusMessage = $"Exported batch summary to {Path.GetFileName(jsonPath)} and {Path.GetFileName(mdPath)}.";
            }

            if (!string.IsNullOrWhiteSpace(batch.LastExportJsonPath))
            {
                ImGui.TextDisabled($"Last Export: {batch.LastExportJsonPath}");
            }
        }
    }
}

