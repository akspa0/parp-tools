using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Threading;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

public sealed record Pm4BatchTileScope(
    string MapName,
    int MinTileX,
    int MaxTileX,
    int MinTileY,
    int MaxTileY);

public sealed record Pm4BatchTileInput(
    int TileX,
    int TileY,
    IReadOnlyList<Pm4HullShapeInput> Hulls);

public sealed record Pm4BatchSummaryRow(
    int TileX,
    int TileY,
    uint Ck24,
    int ObjectPartId,
    uint LinkGroupObjectId,
    float PlacementZ,
    string Classification,
    string? ProposedModelPath,
    string? ChosenModelPath,
    string? AssetKind,
    float Confidence,
    float ShapeDistance,
    Vector3 CalibratedCentroid,
    string DecisionProvenance,
    bool HasConflict);

public sealed record Pm4BatchAnalysisSummary(
    string MapName,
    Pm4BatchTileScope Scope,
    bool WasCancelled,
    int ProcessedTilesCount,
    int TotalTilesCount,
    int TotalHulls,
    int MatchedCount,
    int LowConfidenceCount,
    int UnmatchedCount,
    int ConflictCount,
    int PreservedAcceptedCount,
    Pm4CentroidCalibration Calibration,
    IReadOnlyList<Pm4ProposalConflict> Conflicts,
    IReadOnlyList<Pm4BatchSummaryRow> Rows);

/// <summary>
/// Spec 295 Phase 6 (T023, T025): UI-independent batch shape-match analysis, cancellation with
/// preservation of already-accepted decisions (FR-014), conflict tracking (FR-015), and
/// machine/human-readable audit summary export (FR-017, SC-004).
/// </summary>
public sealed class Pm4ShapeMatchBatchService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public bool IsRunning { get; private set; }
    public bool IsCancelled { get; private set; }
    public int ProcessedTilesCount { get; private set; }
    public int TotalTilesCount { get; private set; }
    public float ProgressFraction => TotalTilesCount <= 0 ? 0f : Math.Clamp((float)ProcessedTilesCount / TotalTilesCount, 0f, 1f);
    public Pm4BatchAnalysisSummary? LastSummary { get; private set; }
    public string? LastExportJsonPath { get; private set; }
    public string? LastExportMarkdownPath { get; private set; }

    public Pm4BatchAnalysisSummary RunBatch(
        Pm4BatchTileScope scope,
        IReadOnlyList<Pm4BatchTileInput> tiles,
        IReadOnlyList<Pm4ModelShapeCandidate> candidates,
        ManualCurationProject project,
        Pm4ShapeMatchReviewService? reviewService = null,
        Pm4CentroidCalibration? calibration = null,
        Pm4ShapeMatchOptions? options = null,
        CancellationToken cancellationToken = default,
        Func<int, bool>? shouldCancelAfterTileIndex = null)
    {
        ArgumentNullException.ThrowIfNull(scope);
        ArgumentNullException.ThrowIfNull(tiles);
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(project);

        var filteredTiles = tiles
            .Where(t => t.TileX >= scope.MinTileX && t.TileX <= scope.MaxTileX &&
                        t.TileY >= scope.MinTileY && t.TileY <= scope.MaxTileY)
            .OrderBy(static t => t.TileY)
            .ThenBy(static t => t.TileX)
            .ToArray();

        IsRunning = true;
        IsCancelled = false;
        ProcessedTilesCount = 0;
        TotalTilesCount = filteredTiles.Length;

        var activeCalibration = calibration ?? Pm4CentroidCalibration.None;
        var analyzedPairs = new List<(Pm4HullShapeInput HullInput, Pm4ShapeMatchProposal Proposal)>();

        for (int i = 0; i < filteredTiles.Length; i++)
        {
            if (cancellationToken.IsCancellationRequested ||
                (shouldCancelAfterTileIndex != null && shouldCancelAfterTileIndex(i)))
            {
                IsCancelled = true;
                break;
            }

            Pm4BatchTileInput tile = filteredTiles[i];
            if (tile.Hulls.Count > 0)
            {
                Pm4ShapeBatchResult tileBatch = Pm4ShapeMatchAnalysisService.AnalyzeBatch(
                    tile.Hulls,
                    candidates,
                    activeCalibration,
                    options);

                for (int h = 0; h < tile.Hulls.Count; h++)
                {
                    analyzedPairs.Add((tile.Hulls[h], tileBatch.Proposals[h]));
                }
            }

            ProcessedTilesCount = i + 1;
        }

        IsRunning = false;

        // Enqueue analyzed hulls into review service without overwriting already-accepted/replaced decisions (FR-014, SC-006)
        reviewService?.EnqueueProposals(analyzedPairs, project);

        IReadOnlyList<Pm4ProposalConflict> allConflicts = Pm4ShapeMatchAnalysisService.FindHullConflicts(
            analyzedPairs.Select(static p => p.Proposal).ToArray(),
            (options ?? new Pm4ShapeMatchOptions()).ConflictDistance);

        var conflictHullKeys = new HashSet<(int TileX, int TileY, uint Ck24, int ObjectPartId)>();
        foreach (var c in allConflicts)
        {
            conflictHullKeys.Add((c.FirstHull.TileX, c.FirstHull.TileY, c.FirstHull.Ck24, c.FirstHull.ObjectPartId));
            conflictHullKeys.Add((c.SecondHull.TileX, c.SecondHull.TileY, c.SecondHull.Ck24, c.SecondHull.ObjectPartId));
        }

        var rows = new List<Pm4BatchSummaryRow>(analyzedPairs.Count);
        int matchedCount = 0;
        int lowConfidenceCount = 0;
        int unmatchedCount = 0;
        int preservedAcceptedCount = 0;

        var addedHullIds = new HashSet<Pm4HullIdentity>();

        foreach (var (hullInput, proposal) in analyzedPairs)
        {
            Pm4HullIdentity hullId = new(
                hullInput.TileX,
                hullInput.TileY,
                hullInput.Ck24,
                hullInput.ObjectPartId,
                hullInput.LinkGroupObjectId);
            addedHullIds.Add(hullId);

            bool hasConflict = conflictHullKeys.Contains((hullId.TileX, hullId.TileY, hullId.Ck24, hullId.ObjectPartId));

            float placementZ = hullInput.PlacementZ;

            if (ManualCurationProjectPolicy.TryGetActiveShapeMatchDecision(project, hullId, out var existingDecision) &&
                existingDecision != null)
            {
                preservedAcceptedCount++;
                matchedCount++;
                rows.Add(new Pm4BatchSummaryRow(
                    TileX: hullId.TileX,
                    TileY: hullId.TileY,
                    Ck24: hullId.Ck24,
                    ObjectPartId: hullId.ObjectPartId,
                    LinkGroupObjectId: hullId.LinkGroupObjectId,
                    PlacementZ: placementZ,
                    Classification: "Matched",
                    ProposedModelPath: existingDecision.ProposedModelPath,
                    ChosenModelPath: existingDecision.ChosenModelPath ?? existingDecision.ProposedModelPath,
                    AssetKind: existingDecision.AssetKind,
                    Confidence: existingDecision.Confidence,
                    ShapeDistance: existingDecision.ShapeDistance,
                    CalibratedCentroid: existingDecision.CalibratedCentroid,
                    DecisionProvenance: $"{existingDecision.Decision}:{existingDecision.Provenance}",
                    HasConflict: hasConflict));
                continue;
            }

            if (proposal.IsUnmatched || proposal.TopCandidate == null)
            {
                unmatchedCount++;
                rows.Add(new Pm4BatchSummaryRow(
                    TileX: hullId.TileX,
                    TileY: hullId.TileY,
                    Ck24: hullId.Ck24,
                    ObjectPartId: hullId.ObjectPartId,
                    LinkGroupObjectId: hullId.LinkGroupObjectId,
                    PlacementZ: placementZ,
                    Classification: "Unmatched",
                    ProposedModelPath: null,
                    ChosenModelPath: null,
                    AssetKind: null,
                    Confidence: 0f,
                    ShapeDistance: float.PositiveInfinity,
                    CalibratedCentroid: proposal.CalibratedCentroid,
                    DecisionProvenance: "Pending:Unmatched",
                    HasConflict: hasConflict));
            }
            else if (proposal.IsLowConfidence)
            {
                lowConfidenceCount++;
                rows.Add(new Pm4BatchSummaryRow(
                    TileX: hullId.TileX,
                    TileY: hullId.TileY,
                    Ck24: hullId.Ck24,
                    ObjectPartId: hullId.ObjectPartId,
                    LinkGroupObjectId: hullId.LinkGroupObjectId,
                    PlacementZ: placementZ,
                    Classification: "LowConfidence",
                    ProposedModelPath: proposal.TopCandidate.ModelPath,
                    ChosenModelPath: null,
                    AssetKind: proposal.TopCandidate.AssetKind,
                    Confidence: proposal.Confidence,
                    ShapeDistance: proposal.TopCandidate.ShapeDistance,
                    CalibratedCentroid: proposal.CalibratedCentroid,
                    DecisionProvenance: "Pending:LowConfidence",
                    HasConflict: hasConflict));
            }
            else
            {
                matchedCount++;
                rows.Add(new Pm4BatchSummaryRow(
                    TileX: hullId.TileX,
                    TileY: hullId.TileY,
                    Ck24: hullId.Ck24,
                    ObjectPartId: hullId.ObjectPartId,
                    LinkGroupObjectId: hullId.LinkGroupObjectId,
                    PlacementZ: placementZ,
                    Classification: "Matched",
                    ProposedModelPath: proposal.TopCandidate.ModelPath,
                    ChosenModelPath: null,
                    AssetKind: proposal.TopCandidate.AssetKind,
                    Confidence: proposal.Confidence,
                    ShapeDistance: proposal.TopCandidate.ShapeDistance,
                    CalibratedCentroid: proposal.CalibratedCentroid,
                    DecisionProvenance: "Pending:Proposed",
                    HasConflict: hasConflict));
            }
        }

        // If cancelled mid-run, also include any already-accepted/replaced decisions on unvisited tiles in the scope so they remain accounted for and preserved (FR-014, SC-004).
        if (project.ShapeMatchDecisions != null)
        {
            foreach (var decision in project.ShapeMatchDecisions)
            {
                if (decision.Decision == ManualShapeMatchDecisionKind.Rejected)
                    continue;

                if (decision.Hull.TileX < scope.MinTileX || decision.Hull.TileX > scope.MaxTileX ||
                    decision.Hull.TileY < scope.MinTileY || decision.Hull.TileY > scope.MaxTileY)
                {
                    continue;
                }

                if (!addedHullIds.Add(decision.Hull))
                    continue;

                preservedAcceptedCount++;
                matchedCount++;
                rows.Add(new Pm4BatchSummaryRow(
                    TileX: decision.Hull.TileX,
                    TileY: decision.Hull.TileY,
                    Ck24: decision.Hull.Ck24,
                    ObjectPartId: decision.Hull.ObjectPartId,
                    LinkGroupObjectId: decision.Hull.LinkGroupObjectId,
                    PlacementZ: decision.CalibratedCentroid.Z,
                    Classification: "Matched",
                    ProposedModelPath: decision.ProposedModelPath,
                    ChosenModelPath: decision.ChosenModelPath ?? decision.ProposedModelPath,
                    AssetKind: decision.AssetKind,
                    Confidence: decision.Confidence,
                    ShapeDistance: decision.ShapeDistance,
                    CalibratedCentroid: decision.CalibratedCentroid,
                    DecisionProvenance: $"{decision.Decision}:{decision.Provenance}",
                    HasConflict: false));
            }
        }

        LastSummary = new Pm4BatchAnalysisSummary(
            MapName: scope.MapName,
            Scope: scope,
            WasCancelled: IsCancelled,
            ProcessedTilesCount: ProcessedTilesCount,
            TotalTilesCount: TotalTilesCount,
            TotalHulls: rows.Count,
            MatchedCount: matchedCount,
            LowConfidenceCount: lowConfidenceCount,
            UnmatchedCount: unmatchedCount,
            ConflictCount: allConflicts.Count,
            PreservedAcceptedCount: preservedAcceptedCount,
            Calibration: activeCalibration,
            Conflicts: allConflicts,
            Rows: rows);

        return LastSummary;
    }

    public static string SerializeJsonSummary(Pm4BatchAnalysisSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);
        return JsonSerializer.Serialize(summary, JsonOptions);
    }

    public static string FormatMarkdownSummary(Pm4BatchAnalysisSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);
        var sb = new StringBuilder();
        sb.AppendLine(CultureInfo.InvariantCulture, $"# PM4 Shape-Match Batch Summary — {summary.MapName}");
        sb.AppendLine();
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Scope**: Tiles `[{summary.Scope.MinTileX}..{summary.Scope.MaxTileX}, {summary.Scope.MinTileY}..{summary.Scope.MaxTileY}]`");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Status**: {(summary.WasCancelled ? "Cancelled (Accepted decisions preserved)" : "Completed")}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Tiles Processed**: {summary.ProcessedTilesCount} / {summary.TotalTilesCount}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Total Hulls**: {summary.TotalHulls}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Matched**: {summary.MatchedCount}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Low-Confidence**: {summary.LowConfidenceCount}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Unmatched**: {summary.UnmatchedCount}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Conflicts**: {summary.ConflictCount}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Preserved Accepted Decisions**: {summary.PreservedAcceptedCount}");
        sb.AppendLine(CultureInfo.InvariantCulture, $"- **Calibration**: `({summary.Calibration.Offset.X:F3}, {summary.Calibration.Offset.Y:F3}, {summary.Calibration.Offset.Z:F3})` ({summary.Calibration.Source})");
        sb.AppendLine();
        sb.AppendLine("## Hull Pairing Rows");
        sb.AppendLine();
        sb.AppendLine("| Tile | Part | Lnk | CK24 | Base Z (yd) | Classification | Proposed Candidate | Chosen Asset | Confidence | Distance | Provenance | Conflict |");
        sb.AppendLine("|---|---:|---:|---:|---:|---|---|---|---:|---:|---|---|");
        foreach (var row in summary.Rows)
        {
            string dist = float.IsInfinity(row.ShapeDistance)
                ? "inf"
                : row.ShapeDistance.ToString("F3", CultureInfo.InvariantCulture);
            sb.AppendLine(
                CultureInfo.InvariantCulture,
                $"| `{row.TileX}_{row.TileY}` | {row.ObjectPartId} | {row.LinkGroupObjectId} | `0x{row.Ck24:X6}` | {row.PlacementZ:F1} | {row.Classification} | `{row.ProposedModelPath ?? "-"}` | `{row.ChosenModelPath ?? "-"}` | {row.Confidence:F3} | {dist} | `{row.DecisionProvenance}` | {(row.HasConflict ? "Yes" : "No")} |");
        }
        return sb.ToString();
    }

    public (string JsonPath, string MarkdownPath) ExportSummaryFiles(
        Pm4BatchAnalysisSummary summary,
        string outputDirectory,
        string? baseFileName = null)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputDirectory);

        Directory.CreateDirectory(outputDirectory);
        string safeMap = string.IsNullOrWhiteSpace(summary.MapName) ? "map" : summary.MapName.Trim();
        string stem = string.IsNullOrWhiteSpace(baseFileName)
            ? $"{safeMap}-pm4-shape-match-summary"
            : baseFileName.Trim();

        string jsonPath = Path.Combine(outputDirectory, stem + ".json");
        string mdPath = Path.Combine(outputDirectory, stem + ".md");

        File.WriteAllText(jsonPath, SerializeJsonSummary(summary));
        File.WriteAllText(mdPath, FormatMarkdownSummary(summary));

        LastExportJsonPath = jsonPath;
        LastExportMarkdownPath = mdPath;
        return (jsonPath, mdPath);
    }
}

