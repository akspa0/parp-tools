using System.Numerics;
using System.Text.Json;
using ImGuiNET;
using WowViewer.Core.Editor;
using WowViewer.Core.Editor.Bridge;
using WowViewer.Core.Editor.Eras;
using WowViewer.Core.Editor.Logging;
using WowViewer.Core.Editor.Operations;
using WowViewer.Core.Editor.Plugins;
using WowViewer.Core.Editor.Session;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Reconciliation;
using WowViewer.Core.PM4.Services;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;
using WowViewer.Core.IO.Terrain;
using static WoWViewer.ViewerApp;

namespace WoWViewer;

// ArchaeologyPanelService: members moved from ViewerApp_Editor.cs; this file keeps that file's using directives so every
// name in the moved code resolves exactly as it did there.
internal sealed partial class ArchaeologyPanelService
{

    // Reconciliation preview state (transient, owned by the plugin surface, never written during preview).
    private IReadOnlyList<ReconciliationProposal> _reconciliationProposals = [];
    private readonly Dictionary<string, ReviewDisposition> _reconciliationDecisions = new(StringComparer.Ordinal);
    private string _reconciliationStatus = "Load a PM4 guide and its paired Museum ADT to preview alignment proposals.";
    private string _reconciliationOutputDir = string.Empty;
    private string _reconciliationBuildFingerprint = string.Empty;
    private string? _reconciliationSourceHash;
    private string? _reconciliationGuideHash;

    // Per-proposal source mapping captured during preview so apply can write every accepted action
    // (including clones, which carry no ExistingPlacement) back to the ADT it was previewed against.
    private readonly Dictionary<string, string> _reconciliationSourceAdtByProposalId = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _reconciliationSourceHashByAdt = new(StringComparer.OrdinalIgnoreCase);

    // Review-list UI state: the docked sidebar cannot be relied on to scroll, so the proposal list
    // is filtered and paginated instead.
    private int _reconciliationListFilter;
    private int _reconciliationListPage;
    private const int ReconciliationListPageSize = 50;

    private static readonly JsonSerializerOptions ReconciliationJsonOptions = new() { WriteIndented = true };


    private void DrawReconciliationPanel()
    {
        ImGui.Text("PM4/Museum Reconciliation");
        ImGui.TextDisabled("Processes every PM4 tile on disk for the current map against its placement ADTs — not just tiles streamed around the camera. Nothing is written during preview.");

        // Everything is discerned from the loaded session — the user never types or browses a path.
        // The whole map's PM4 corpus on disk IS the guide; placement ADTs are derived per tile.
        string? mapName = _dataSourceSession.GetCurrentSessionMapName();
        string? mapDirectory = _dataSourceSession.TryResolveCurrentMapDirectory(preferLooseOverlay: true);

        if (string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(mapDirectory))
        {
            ImGui.TextDisabled("Load a map to reconcile. Its PM4 tiles and their placement ADTs are detected automatically from the map directory.");
            return;
        }

        if (!TryCollectReconciliationPairs(mapName, mapDirectory!, out List<(int TileX, int TileY, string Pm4Path, string AdtPath)> resolved) || resolved.Count == 0)
        {
            ImGui.TextDisabled($"No PM4/placement-ADT pairs found on disk under {mapDirectory}.");
            return;
        }

        ImGui.TextDisabled($"Map: {mapName}  |  {resolved.Count} PM4 tile(s) with placement ADTs detected.");
        foreach ((int tileX, int tileY, string pm4Path, string adtPath) in resolved)
            ImGui.TextDisabled($"  ({tileX},{tileY})  {Path.GetFileName(pm4Path)}  +  {Path.GetFileName(adtPath)}");

        // Output folder: auto-generate a timestamped project folder; never ask the user.
        string effectiveOutputDir = string.IsNullOrWhiteSpace(_reconciliationOutputDir)
            ? _projectOutput.DescribeEditorProjectOutputDirectory()
            : _reconciliationOutputDir;
        ImGui.TextDisabled($"Will write to: {effectiveOutputDir}");
        if (ImGui.SmallButton("New output folder"))
        {
            _reconciliationOutputDir = _projectOutput.EnsureEditorProjectOutputDirectory(forceNew: true);
            _reconciliationStatus = $"Created new output folder: {_reconciliationOutputDir}";
        }

        ImGui.InputText("Build fingerprint", ref _reconciliationBuildFingerprint, 128);

        if (ImGui.Button("Preview"))
        {
            RunReconciliationPreview();
        }

        ImGui.SameLine();
        if (ImGui.Button("Clear"))
        {
            _reconciliationProposals = [];
            _reconciliationDecisions.Clear();
            _reconciliationSourceHash = null;
            _reconciliationGuideHash = null;
            _reconciliationSourceAdtByProposalId.Clear();
            _reconciliationSourceHashByAdt.Clear();
            _reconciliationStatus = "Preview cleared.";
        }

        ImGui.TextDisabled(_reconciliationStatus);

        if (_reconciliationProposals.Count == 0)
            return;

        int reviewable = _reconciliationProposals.Count(p => p.Status == ProposalStatus.ReviewRequired);
        int conflicts = _reconciliationProposals.Count(p => p.Status == ProposalStatus.Conflict);
        int alreadyAligned = _reconciliationProposals.Count(p => p.Status == ProposalStatus.AlreadyAligned);
        int accepted = _reconciliationProposals.Count(p =>
            _reconciliationDecisions.TryGetValue(p.ProposalId, out ReviewDisposition d) && d == ReviewDisposition.Accept);

        ImGui.Separator();
        ImGui.Text(
            $"{_reconciliationProposals.Count} proposal(s): {reviewable} reviewable, {conflicts} conflict, {alreadyAligned} already aligned, {accepted} accepted.");

        // Bulk review: every reviewable proposal needs an explicit decision, so offer one click for
        // the common case instead of hundreds of individual headers.
        if (reviewable > 0)
        {
            if (ImGui.Button("Accept all reviewable"))
            {
                foreach (ReconciliationProposal proposal in _reconciliationProposals)
                {
                    if (proposal.Status == ProposalStatus.ReviewRequired)
                        _reconciliationDecisions[proposal.ProposalId] = ReviewDisposition.Accept;
                }
            }

            ImGui.SameLine();
        }

        if (_reconciliationDecisions.Count > 0 && ImGui.Button("Clear decisions"))
            _reconciliationDecisions.Clear();

        ImGui.Separator();

        // The proposals list can be thousands of rows and the docked sidebar does not reliably
        // scroll, so review is driven by filters plus pagination instead of raw scrolling.
        var filtered = new List<ReconciliationProposal>(_reconciliationProposals.Count);
        foreach (ReconciliationProposal proposal in _reconciliationProposals)
        {
            bool include = _reconciliationListFilter switch
            {
                1 => proposal.Status == ProposalStatus.ReviewRequired,
                2 => proposal.Status == ProposalStatus.Conflict,
                3 => proposal.Status == ProposalStatus.AlreadyAligned,
                _ => true,
            };
            if (include)
                filtered.Add(proposal);
        }

        if (ImGui.SmallButton("All##reconfilter"))
            _reconciliationListFilter = 0;
        ImGui.SameLine();
        if (ImGui.SmallButton($"Reviewable ({reviewable})##reconfilter"))
            _reconciliationListFilter = 1;
        ImGui.SameLine();
        if (ImGui.SmallButton($"Conflicts ({conflicts})##reconfilter"))
            _reconciliationListFilter = 2;
        ImGui.SameLine();
        if (ImGui.SmallButton($"Aligned ({alreadyAligned})##reconfilter"))
            _reconciliationListFilter = 3;

        int pageCount = Math.Max(1, (filtered.Count + ReconciliationListPageSize - 1) / ReconciliationListPageSize);
        if (_reconciliationListPage >= pageCount)
            _reconciliationListPage = pageCount - 1;

        ImGui.BeginDisabled(_reconciliationListPage == 0);
        if (ImGui.SmallButton("< Prev##reconpage"))
            _reconciliationListPage--;
        ImGui.EndDisabled();
        ImGui.SameLine();
        ImGui.TextDisabled($"page {_reconciliationListPage + 1}/{pageCount} — {filtered.Count} shown");
        ImGui.SameLine();
        ImGui.BeginDisabled(_reconciliationListPage >= pageCount - 1);
        if (ImGui.SmallButton("Next >##reconpage"))
            _reconciliationListPage++;
        ImGui.EndDisabled();

        int pageStart = _reconciliationListPage * ReconciliationListPageSize;
        int pageEnd = Math.Min(filtered.Count, pageStart + ReconciliationListPageSize);

        ImGui.BeginChild("##ReconciliationProposals", new Vector2(0, -ImGui.GetFrameHeightWithSpacing()), border: false);
        for (int index = pageStart; index < pageEnd; index++)
        {
            DrawReconciliationProposal(filtered[index]);
        }

        ImGui.EndChild();

        ImGui.BeginDisabled(accepted == 0);
        if (ImGui.Button($"Apply Accepted ({accepted})"))
        {
            ApplyAcceptedReconciliation();
        }
        ImGui.EndDisabled();
    }

    /// <summary>One path row: editable text plus an in-app Browse button (no native dialogs).</summary>

    private void RunReconciliationPreview()
    {
        _reconciliationStatus = "Preview requires loaded PM4 tiles and their placement ADTs.";
        if (!TryResolveReconciliationPairs(out List<(int TileX, int TileY, string Pm4Path, string AdtPath)> pairs))
            return;

        try
        {
            string fingerprint = string.IsNullOrWhiteSpace(_reconciliationBuildFingerprint)
                ? "unspecified"
                : _reconciliationBuildFingerprint;
            string mapName = _dataSourceSession.GetCurrentSessionMapName() ?? "map";

            var allProposals = new List<ReconciliationProposal>();
            int totalSegments = 0;
            int totalPlacements = 0;
            int totalCorpus = 0;
            _reconciliationSourceAdtByProposalId.Clear();
            _reconciliationSourceHashByAdt.Clear();

            foreach ((int tileX, int tileY, string pm4Path, string adtPath) in pairs)
            {
                // 1. Parse the real PM4 guide into object segments (existing segment builder owner).
                IReadOnlyList<Pm4BuiltObjectSegment> segments = Pm4ObjectSegmentBuilder.Build(pm4Path);
                if (segments.Count == 0)
                    continue;

                // 2. Read the placement catalog and fingerprint the source bytes.
                byte[] museumBytes = File.ReadAllBytes(adtPath);
                AdtPlacementCatalog catalog = AdtPlacementReader.Read(adtPath);
                _reconciliationSourceHashByAdt[adtPath] = ReconciliationApplyService.ComputeSha256Hex(museumBytes);

                // 3. Score segments against the labelled Museum self-corpus (no archive access).
                var corpus = Pm4ReconciliationInputAdapter.BuildSelfCorpusReferences(catalog, $"{tileX}_{tileY}", fingerprint);
                IReadOnlyList<Pm4SegmentMatchResult> matchResults = Pm4AssetMatchScorer.ScoreSegments(segments, corpus);

                // 4. Build guide observations, placement snapshots, and candidate lists.
                var guides = Pm4ReconciliationInputAdapter.BuildGuideObservations(
                    matchResults, pm4Path, fingerprint, mapName, tileX, tileY);
                var snapshots = Pm4ReconciliationInputAdapter.BuildPlacementSnapshots(
                    catalog, mapName, tileX, tileY, fingerprint);
                var candidatesByGuide = Pm4ReconciliationInputAdapter.BuildCandidatesByGuideId(matchResults);

                // 5. Build the proposal set (side-effect free), mapping every proposal — including
                // clones, which carry no ExistingPlacement — back to this tile's placement ADT.
                IReadOnlyList<ReconciliationProposal> tileProposals =
                    Pm4ReconciliationEngine.BuildTileProposals(guides, snapshots, candidatesByGuide);
                foreach (ReconciliationProposal proposal in tileProposals)
                    _reconciliationSourceAdtByProposalId[proposal.ProposalId] = adtPath;
                allProposals.AddRange(tileProposals);

                totalSegments += segments.Count;
                totalPlacements += snapshots.Count;
                totalCorpus += corpus.Count;
            }

            _reconciliationProposals = allProposals;
            _reconciliationDecisions.Clear();

            // 6. Record source fingerprints so apply can refuse stale previews (first pair).
            (_, _, string firstPm4, string firstAdt) = pairs[0];
            _reconciliationSourceHash = _reconciliationSourceHashByAdt.TryGetValue(firstAdt, out string? firstHash)
                ? firstHash
                : ReconciliationApplyService.ComputeSha256Hex(File.ReadAllBytes(firstAdt));
            _reconciliationGuideHash = ReconciliationApplyService.ComputeSha256Hex(File.ReadAllBytes(firstPm4));

            _reconciliationStatus =
                $"Previewed {_reconciliationProposals.Count} proposal(s) from {totalSegments} PM4 object(s) across {pairs.Count} tile(s) against {totalPlacements} placement(s) ({totalCorpus} self-corpus assets).";
        }
        catch (Exception ex)
        {
            _reconciliationStatus = $"Preview failed: {ex.Message}";
        }
    }

    /// <summary>Resolves every PM4/placement-ADT pair on disk for the current map (no user input,
    /// no dependence on which tiles the scene has streamed around the camera).</summary>
    private bool TryResolveReconciliationPairs(out List<(int TileX, int TileY, string Pm4Path, string AdtPath)> pairs)
    {
        pairs = [];
        string? mapName = _dataSourceSession.GetCurrentSessionMapName();
        string? mapDirectory = _dataSourceSession.TryResolveCurrentMapDirectory(preferLooseOverlay: true);

        if (string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(mapDirectory))
        {
            _reconciliationStatus = "Load a map so its directory can be located for reconciliation.";
            return false;
        }

        if (!TryCollectReconciliationPairs(mapName, mapDirectory!, out pairs) || pairs.Count == 0)
        {
            _reconciliationStatus = $"No PM4/placement-ADT pairs found on disk under {mapDirectory}.";
            return false;
        }

        return true;
    }

    /// <summary>
    /// Enumerates every <c>{map}_*.pm4</c> file in the map directory and pairs it with its
    /// placement ADT (split <c>_obj0.adt</c> preferred, monolithic root ADT as fallback). This
    /// covers the entire map at once rather than only the tiles the viewer has loaded.
    /// </summary>
    private static bool TryCollectReconciliationPairs(
        string mapName,
        string mapDirectory,
        out List<(int TileX, int TileY, string Pm4Path, string AdtPath)> pairs)
    {
        pairs = [];
        var seenTiles = new HashSet<(int, int)>();

        foreach (string pm4Path in Directory.EnumerateFiles(mapDirectory, $"{mapName}_*.pm4"))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
                continue;

            if (!seenTiles.Add((tileX, tileY)))
                continue;

            if (TryResolvePlacementAdtPath(mapDirectory, mapName, tileX, tileY, out string adtPath))
                pairs.Add((tileX, tileY, pm4Path, adtPath));
        }

        pairs.Sort(static (a, b) => (a.TileY, a.TileX).CompareTo((b.TileY, b.TileX)));
        return pairs.Count > 0;
    }

    /// <summary>
    /// Resolves the placement ADT for one tile: prefers the split-format <c>_obj0.adt</c> when it
    /// exists, otherwise falls back to the monolithic root <c>{map}_{y}_{x}.adt</c>, which carries
    /// its own MDDF/MODF rows. Both forms are supported by AdtPlacementReader/Editor.
    /// </summary>
    private static bool TryResolvePlacementAdtPath(string mapDirectory, string mapName, int tileX, int tileY, out string adtPath)
    {
        string obj0Path = Path.Combine(mapDirectory, $"{mapName}_{tileY}_{tileX}_obj0.adt");
        if (File.Exists(obj0Path))
        {
            adtPath = obj0Path;
            return true;
        }

        adtPath = Path.Combine(mapDirectory, $"{mapName}_{tileY}_{tileX}.adt");
        return File.Exists(adtPath);
    }

    private void DrawReconciliationProposal(ReconciliationProposal proposal)
    {
        string action = proposal.Action switch
        {
            ReconciliationAction.Align => "Align",
            ReconciliationAction.Substitute => "Substitute",
            ReconciliationAction.Clone => "Clone",
            _ => "?",
        };

        bool isAccepted = _reconciliationDecisions.TryGetValue(proposal.ProposalId, out ReviewDisposition d) && d == ReviewDisposition.Accept;
        bool isRejected = _reconciliationDecisions.TryGetValue(proposal.ProposalId, out ReviewDisposition d2) && d2 == ReviewDisposition.Reject;
        string decisionMark = isAccepted ? "[ACCEPTED] " : isRejected ? "[rejected] " : string.Empty;

        double positionResidual = proposal.Residual.GetValueOrDefault("position");
        string summary = $"{decisionMark}{action,-10} {proposal.Status,-14} conf {proposal.Confidence:F2}  id {proposal.ProposalId[..8]}";
        if (proposal.Residual.Count > 0)
            summary += $"  Δpos {positionResidual:F1}";

        ImGui.PushID(proposal.ProposalId);

        bool canDecide = proposal.Status == ProposalStatus.ReviewRequired;
        if (canDecide)
        {
            if (ImGui.SmallButton(isAccepted ? "Accepted" : "Accept"))
                _reconciliationDecisions[proposal.ProposalId] = ReviewDisposition.Accept;
            ImGui.SameLine();
            if (ImGui.SmallButton(isRejected ? "Rejected" : "Reject"))
                _reconciliationDecisions[proposal.ProposalId] = ReviewDisposition.Reject;
            ImGui.SameLine();
        }
        else if (proposal.Status == ProposalStatus.AlreadyAligned)
        {
            ImGui.TextDisabled("[ok] ");
            ImGui.SameLine();
        }
        else
        {
            ImGui.TextDisabled("[!]    ");
            ImGui.SameLine();
        }

        ImGui.TextUnformatted(summary);

        if (proposal.Status == ProposalStatus.Conflict || proposal.Candidate != null)
        {
            string? detail = proposal.Candidate is not null
                ? $"candidate {proposal.Candidate.AssetPath}"
                : string.Join("; ", proposal.Evidence.Where(e => e.Signal.StartsWith("association-competitor", StringComparison.Ordinal)).Select(e => e.Source));
            if (!string.IsNullOrWhiteSpace(detail))
                ImGui.TextDisabled($"        {detail}");
        }

        if (ImGui.TreeNode("Details"))
        {
            ImGui.TextDisabled($"Guide: {proposal.Guide.GuideId}");
            ImGui.TextDisabled($"Proposed position: ({proposal.ProposedPosition.X:F1}, {proposal.ProposedPosition.Y:F1}, {proposal.ProposedPosition.Z:F1})");

            if (proposal.Residual.Count > 0)
            {
                ImGui.Text("Residuals:");
                foreach ((string key, double value) in proposal.Residual)
                    ImGui.TextDisabled($"  {key}: {value:F3}");
            }

            if (proposal.Evidence.Count > 0)
            {
                ImGui.Text("Evidence:");
                foreach (ReconciliationEvidence evidence in proposal.Evidence)
                    ImGui.TextDisabled($"  {evidence.Signal}: {evidence.Value:F3} ({evidence.Source})");
            }

            ImGui.TreePop();
        }

        ImGui.PopID();
    }

    private void ApplyAcceptedReconciliation()
    {
        // The reconcile tab does not route through DrawEditorContent, so the editor host may never
        // have been created when this panel is the first editor surface the user touches.
        EnsureEditorHost();
        if (_editorSession == null)
        {
            _reconciliationStatus = "Editor session could not be initialized; cannot apply reconciliation edits.";
            return;
        }

        var accepted = _reconciliationProposals
            .Where(p => _reconciliationDecisions.TryGetValue(p.ProposalId, out ReviewDisposition d) && d == ReviewDisposition.Accept)
            .ToList();

        if (accepted.Count == 0)
        {
            _reconciliationStatus = "No accepted proposals to apply.";
            return;
        }

        if (_reconciliationSourceHashByAdt.Count == 0)
        {
            _reconciliationStatus = "Run a preview first; the apply validates against the preview's source fingerprints.";
            return;
        }

        try
        {
            // Auto-generate a timestamped project output folder when none is set, so the user never
            // has to pick one. A custom folder (via Browse) is honored when provided.
            string outputDir = string.IsNullOrWhiteSpace(_reconciliationOutputDir)
                ? _projectOutput.EnsureEditorProjectOutputDirectory(forceNew: false)
                : _reconciliationOutputDir;
            if (string.IsNullOrWhiteSpace(outputDir))
            {
                _reconciliationStatus = "No output directory configured.";
                return;
            }

            Directory.CreateDirectory(outputDir);
            _editorSession.OutputDirectory = outputDir;

            // Group accepted proposals by their previewed source ADT. The mapping was captured at
            // preview time per tile, so clone proposals (which carry no ExistingPlacement) land in
            // the same group as the align/substitute proposals of their tile.
            var bySource = accepted
                .GroupBy(
                    p => _reconciliationSourceAdtByProposalId.TryGetValue(p.ProposalId, out string? mapped)
                        ? mapped
                        : p.ExistingPlacement?.SourcePath ?? string.Empty,
                    StringComparer.OrdinalIgnoreCase)
                .Where(g => !string.IsNullOrWhiteSpace(g.Key))
                .ToList();

            if (bySource.Count == 0)
            {
                _reconciliationStatus = "Accepted proposals could not be mapped to a source ADT; re-run the preview.";
                return;
            }

            int appliedCount = 0;
            foreach (IGrouping<string, ReconciliationProposal> group in bySource)
            {
                string sourceAdt = group.Key;
                byte[] sourceBytes = File.ReadAllBytes(sourceAdt);
                string expectedHash = _reconciliationSourceHashByAdt.TryGetValue(sourceAdt, out string? previewedHash)
                    ? previewedHash
                    : ReconciliationApplyService.ComputeSha256Hex(sourceBytes);

                (AdtPlacementEditResult result, ReconciliationProvenanceReport report) =
                    ReconciliationApplyService.ApplyAccepted(
                        group.ToList(),
                        sourceBytes,
                        sourceAdt,
                        expectedHash,
                        sourceAdt,
                        _reconciliationGuideHash,
                        string.IsNullOrWhiteSpace(_reconciliationBuildFingerprint) ? "unspecified" : _reconciliationBuildFingerprint);

                // Resolve through the session so protected roots and container paths are refused.
                string fullPath = Path.GetFullPath(Path.Combine(outputDir, Path.GetFileName(sourceAdt)));
                string outputPath = _editorSession.ResolveOutputPath(fullPath);
                string reportPath = outputPath + ".reconciliation.json";

                byte[]? priorBytes = File.Exists(outputPath) ? File.ReadAllBytes(outputPath) : null;
                string reportJson = JsonSerializer.Serialize(report, ReconciliationJsonOptions);

                var operation = new ReconciliationApplyOperation(
                    $"apply-{report.BatchId}",
                    "pm4.reconciliation",
                    sourceAdt,
                    outputPath,
                    reportPath,
                    priorBytes,
                    result.Bytes,
                    reportJson);

                MaterializeReconciliationOutput(operation);
                _editorSession.RecordApplied(operation);
                appliedCount += group.Count();
            }

            _reconciliationStatus =
                $"Applied {appliedCount} accepted proposal(s) across {bySource.Count} ADT file(s). Output written under {outputDir}.";
        }
        catch (Exception ex)
        {
            _reconciliationStatus = $"Apply failed: {ex.Message}";
        }
    }

    /// <summary>
    /// Materializes a reconciliation operation's output state (forward or reverse) through the
    /// session's write guards. Undo restores the prior output bytes — or removes the output and
    /// sidecar when the apply created them.
    /// </summary>
    internal void MaterializeReconciliationOutput(ReconciliationApplyOperation operation)
    {
        _editorSession?.GuardWritablePath(operation.OutputPath);

        if (operation.BytesToWrite is null)
        {
            if (File.Exists(operation.OutputPath))
                File.Delete(operation.OutputPath);
            if (File.Exists(operation.ReportPath))
                File.Delete(operation.ReportPath);
            return;
        }

        string? directory = Path.GetDirectoryName(operation.OutputPath);
        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllBytes(operation.OutputPath, operation.BytesToWrite);

        if (operation.ReportExists)
            File.WriteAllText(operation.ReportPath, operation.ReportJson);
        else if (File.Exists(operation.ReportPath))
            File.Delete(operation.ReportPath);
    }
}
