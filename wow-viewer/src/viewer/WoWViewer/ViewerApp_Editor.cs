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

namespace WoWViewer;

/// <summary>
/// The Editor destination (Spec 166) and the PM4/Museum reconciliation surface (Spec 176 Phases 3–4).
/// This is a thin host shell: the plugin host, bridge, session, and reconciliation engine all live in
/// core. The viewer only adapts live scene state into the bridge snapshot and draws the plugin surface.
/// </summary>
public partial class ViewerApp
{
    private EditorHost? _editorHost;
    private EditorSession? _editorSession;
    private EditorSceneReaderAdapter? _editorSceneReader;
    private EditorLogAdapter? _editorLog;

    // Reconciliation preview state (transient, owned by the plugin surface, never written during preview).
    private IReadOnlyList<ReconciliationProposal> _reconciliationProposals = [];
    private readonly Dictionary<string, ReviewDisposition> _reconciliationDecisions = new(StringComparer.Ordinal);
    private string _reconciliationStatus = "Load a PM4 guide and its paired Museum ADT to preview alignment proposals.";
    private string _reconciliationPm4Path = string.Empty;
    private string _reconciliationMuseumPath = string.Empty;
    private string _reconciliationOutputDir = string.Empty;
    private string _reconciliationBuildFingerprint = string.Empty;
    private string? _reconciliationSourceHash;
    private string? _reconciliationGuideHash;

    // Per-proposal source mapping captured during preview so apply can write every accepted action
    // (including clones, which carry no ExistingPlacement) back to the ADT it was previewed against.
    private readonly Dictionary<string, string> _reconciliationSourceAdtByProposalId = new(StringComparer.Ordinal);
    private readonly Dictionary<string, string> _reconciliationSourceHashByAdt = new(StringComparer.OrdinalIgnoreCase);

    private static readonly JsonSerializerOptions ReconciliationJsonOptions = new() { WriteIndented = true };

    private void EnsureEditorHost()
    {
        if (_editorHost != null)
            return;

        _editorLog = new EditorLogAdapter();
        EditorBuildVersion build = EditorBuildVersion.TryParse(_dbcBuild, out EditorBuildVersion parsed)
            ? parsed
            : EditorBuildVersion.Parse("0.0.0");

        _editorHost = new EditorHost(build, _editorLog);
        _editorHost.Register(new ReferenceEditorPlugin());

        _editorSceneReader = new EditorSceneReaderAdapter(this);
        _editorSession = new EditorSession(new EditorApplierAdapter(this), _editorLog, _editorProjectOutputDir);
    }

    private void DrawEditorContent()
    {
        EnsureEditorHost();
        if (_editorHost == null || _editorSession == null)
        {
            ImGui.Text("Editor host failed to initialize.");
            return;
        }

        ImGui.Text("Editor");
        ImGui.TextDisabled("Plugin host for editing capabilities. Plugins are listed below; activate one to draw its surface.");

        ImGui.Separator();

        // Plugin catalog
        foreach (EditorPluginCatalogEntry entry in _editorHost.Catalog)
        {
            bool isActive = _editorHost.ActivePlugin?.Descriptor.Id == entry.Id;
            string label = entry.Availability.IsAvailable
                ? $"{entry.DisplayName}##{entry.Id}"
                : $"{entry.DisplayName} (unavailable)##{entry.Id}";

            if (ImGui.Selectable(label, isActive))
            {
                if (entry.Availability.IsAvailable)
                    _editorHost.Activate(entry.Id);
            }

            if (!entry.Availability.IsAvailable)
                ImGui.TextDisabled($"  {entry.Availability.UnavailableReason}");
            else if (entry.State == EditorPluginState.Faulted)
                ImGui.TextDisabled($"  Faulted: {entry.FaultMessage}");
        }

        ImGui.Separator();

        // Session controls
        ImGui.Text("Session");
        if (ImGui.Button("Undo") && _editorSession.CanUndo)
            _editorSession.Undo();
        ImGui.SameLine();
        if (ImGui.Button("Redo") && _editorSession.CanRedo)
            _editorSession.Redo();
        ImGui.SameLine();
        if (ImGui.Button("Save All") && _editorSession.HasUnsavedChanges)
            _editorSession.SaveAll();

        ImGui.TextDisabled(_editorSession.HasUnsavedChanges
            ? $"Unsaved changes: {string.Join(", ", _editorSession.DirtyPlugins)}"
            : "No unsaved changes.");

        ImGui.Separator();

        // Placement authoring panel (Spec 175)
        DrawPlacementAuthoringPanel();

        ImGui.Separator();

        // PM4/Museum reconciliation lives in ONE place: Experimental > PM4 > Reconcile, next to the
        // other PM4 surfaces. Drawing it here too duplicated the panel and its preview state.
        ImGui.TextDisabled("PM4/Museum reconciliation: see Experimental > PM4 > Reconcile.");
    }

    private void DrawPlacementAuthoringPanel()
    {
        ImGui.Text("Placement Authoring");
        ImGui.TextDisabled("Move/rotate/scale/delete selected ADT placements. Edits join the existing staged-placement save queue.");

        // The save queue (per-source output targets, Save Current Source / Save All Pending) is the
        // ONE save path for authored placement edits — shared with Scene > Placements.
        DrawPlacementSaveQueueActions(includeCurrentSourceSave: true);

        ImGui.Separator();

        if (_worldScene == null || !_worldScene.SelectedInstance.HasValue)
        {
            ImGui.TextDisabled("Select a tile-backed world object to edit its placement.");
            return;
        }

        ObjectInstance selected = _worldScene.SelectedInstance.Value;
        if (!selected.HasTileCoordinate || selected.PlacementEntryIndex < 0)
        {
            ImGui.TextDisabled("The selected object is not backed by a writable ADT placement.");
            return;
        }

        if (!_worldScene.TryGetSelectedPlacementSourceData(out string sourcePath, out _))
        {
            ImGui.TextDisabled("The selected placement source ADT could not be resolved from the current data source.");
            return;
        }

        ImGui.TextDisabled($"Tile ({selected.TileX}, {selected.TileY})  Entry {selected.PlacementEntryIndex}  UniqueId {selected.UniqueId}");
        ImGui.TextDisabled($"Source: {sourcePath}");

        Vector3 position = selected.PlacementPosition;
        if (ImGui.InputFloat3("Position", ref position, "%.3f"))
        {
            if (_worldScene.TryUpdateSelectedPlacementPosition(position, out string error))
            {
                StageAuthoringPlacementEdit(_worldScene.SelectedObjectType, selected, sourcePath, position: position);
                RecordAuthoringSessionOperation(
                    new PlacementMoveOperation(
                        $"move-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
                        "placement.authoring",
                        sourcePath,
                        AuthoringPlacementKind(),
                        selected.PlacementEntryIndex,
                        selected.UniqueId,
                        selected.PlacementPosition,
                        position));
            }
            else
            {
                ImGui.TextDisabled(error);
            }
        }

        Vector3 rotation = selected.PlacementRotation;
        if (ImGui.InputFloat3("Rotation", ref rotation, "%.3f"))
        {
            StageAuthoringPlacementEdit(_worldScene.SelectedObjectType, selected, sourcePath, rotation: rotation);
            RecordAuthoringSessionOperation(
                new PlacementRotateOperation(
                    $"rotate-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
                    "placement.authoring",
                    sourcePath,
                    AuthoringPlacementKind(),
                    selected.PlacementEntryIndex,
                    selected.UniqueId,
                    selected.PlacementRotation,
                    rotation));
        }

        float scale = selected.PlacementScale;
        if (ImGui.InputFloat("Scale", ref scale, 0.01f))
        {
            StageAuthoringPlacementEdit(_worldScene.SelectedObjectType, selected, sourcePath, scale: scale);
            RecordAuthoringSessionOperation(
                new PlacementScaleOperation(
                    $"scale-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
                    "placement.authoring",
                    sourcePath,
                    AuthoringPlacementKind(),
                    selected.PlacementEntryIndex,
                    selected.UniqueId,
                    selected.PlacementScale,
                    scale));
        }

        if (ImGui.Button("Delete Placement"))
        {
            StageAuthoringPlacementEdit(_worldScene.SelectedObjectType, selected, sourcePath, delete: true);
            RecordAuthoringSessionOperation(
                new PlacementDeleteOperation(
                    $"delete-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
                    "placement.authoring",
                    sourcePath,
                    AuthoringPlacementKind(),
                    selected.PlacementEntryIndex,
                    selected.UniqueId));
        }
    }

    private AdtPlacementKind AuthoringPlacementKind()
        => _worldScene?.SelectedObjectType == Terrain.ObjectType.Wmo
            ? AdtPlacementKind.WorldModel
            : AdtPlacementKind.Model;

    private void RecordAuthoringSessionOperation(EditorOperation operation)
        => _editorSession?.RecordApplied(operation);


    private void DrawReconciliationPanel()
    {
        ImGui.Text("PM4/Museum Reconciliation");
        ImGui.TextDisabled("Processes the PM4 tiles already loaded in the scene against their placement ADTs. Nothing is written during preview.");

        // Everything is discerned from the loaded scene — the user never types or browses a path.
        // The loaded PM4 tiles ARE the guide; their placement ADTs are derived from the map name.
        string? mapName = GetCurrentSessionMapName();
        string? mapDirectory = TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        IReadOnlyList<(int TileX, int TileY)> loadedPm4Tiles = _worldScene?.LoadedPm4Tiles ?? [];

        if (string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(mapDirectory) || loadedPm4Tiles.Count == 0)
        {
            ImGui.TextDisabled("Load a map with PM4 overlay tiles to reconcile. The loaded PM4 tiles and their placement ADTs are detected automatically.");
            return;
        }

        // Resolve the placement ADT for each loaded PM4 tile (first number bounds Y, unpadded).
        var resolved = new List<(int TileX, int TileY, string Pm4Path, string AdtPath)>();
        foreach ((int tileX, int tileY) in loadedPm4Tiles)
        {
            string pm4Path = Path.Combine(mapDirectory, $"{mapName}_{tileY:D2}_{tileX:D2}.pm4");
            if (!File.Exists(pm4Path))
                continue;
            if (TryResolvePlacementAdtPath(mapDirectory, mapName, tileX, tileY, out string adtPath))
                resolved.Add((tileX, tileY, pm4Path, adtPath));
        }

        if (resolved.Count == 0)
        {
            ImGui.TextDisabled($"No PM4/placement-ADT pairs found on disk for the {loadedPm4Tiles.Count} loaded PM4 tile(s) under {mapDirectory}.");
            return;
        }

        ImGui.TextDisabled($"Map: {mapName}  |  {resolved.Count} PM4 tile(s) with placement ADTs detected.");
        foreach ((int tileX, int tileY, string pm4Path, string adtPath) in resolved)
            ImGui.TextDisabled($"  ({tileX},{tileY})  {Path.GetFileName(pm4Path)}  +  {Path.GetFileName(adtPath)}");

        // Output folder: auto-generate a timestamped project folder; never ask the user.
        string effectiveOutputDir = string.IsNullOrWhiteSpace(_reconciliationOutputDir)
            ? DescribeEditorProjectOutputDirectory()
            : _reconciliationOutputDir;
        ImGui.TextDisabled($"Will write to: {effectiveOutputDir}");
        if (ImGui.SmallButton("New output folder"))
        {
            _reconciliationOutputDir = EnsureEditorProjectOutputDirectory(forceNew: true);
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

        ImGui.BeginChild("##ReconciliationProposals", new Vector2(0, -ImGui.GetFrameHeightWithSpacing()), border: false);
        foreach (ReconciliationProposal proposal in _reconciliationProposals)
        {
            DrawReconciliationProposal(proposal);
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
    private void DrawReconciliationPathField(
        string label,
        Func<string> getPath,
        Action<string> setPath,
        string pickerTitle,
        bool pickFolder,
        string? filterExtension)
    {
        string path = getPath();
        ImGui.InputText(label, ref path, 512);
        setPath(path);

        ImGui.SameLine();
        if (ImGui.SmallButton($"Browse##{label}"))
        {
            ImGuiPathPicker.Instance.Open(
                pickerTitle,
                pickFolder,
                path,
                filterExtension,
                picked => setPath(picked));
        }
    }

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
            string mapName = GetCurrentSessionMapName() ?? "map";

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

    /// <summary>Resolves the placement ADT for every loaded PM4 tile from the scene (no user input).</summary>
    private bool TryResolveReconciliationPairs(out List<(int TileX, int TileY, string Pm4Path, string AdtPath)> pairs)
    {
        pairs = [];
        string? mapName = GetCurrentSessionMapName();
        string? mapDirectory = TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        IReadOnlyList<(int TileX, int TileY)> loadedPm4Tiles = _worldScene?.LoadedPm4Tiles ?? [];

        if (string.IsNullOrWhiteSpace(mapName) || string.IsNullOrWhiteSpace(mapDirectory) || loadedPm4Tiles.Count == 0)
        {
            _reconciliationStatus = "Load a map with PM4 overlay tiles to reconcile.";
            return false;
        }

        foreach ((int tileX, int tileY) in loadedPm4Tiles)
        {
            string pm4Path = Path.Combine(mapDirectory, $"{mapName}_{tileY:D2}_{tileX:D2}.pm4");
            if (!File.Exists(pm4Path))
                continue;
            if (TryResolvePlacementAdtPath(mapDirectory, mapName, tileX, tileY, out string adtPath))
                pairs.Add((tileX, tileY, pm4Path, adtPath));
        }

        if (pairs.Count == 0)
        {
            _reconciliationStatus = $"No PM4/placement-ADT pairs found on disk for the {loadedPm4Tiles.Count} loaded PM4 tile(s).";
            return false;
        }

        return true;
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
                ? EnsureEditorProjectOutputDirectory(forceNew: false)
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

    /// <summary>Adapts the live scene into the editor bridge snapshot.</summary>
    private sealed class EditorSceneReaderAdapter : IEditorSceneReader
    {
        private readonly ViewerApp _app;

        public EditorSceneReaderAdapter(ViewerApp app) => _app = app;

        public EditorSceneSnapshot Capture()
        {
            var selection = new List<EditorSelectionEntry>();
            if (_app._worldScene?.SelectedInstance is ObjectInstance selected)
            {
                selection.Add(new EditorSelectionEntry(
                    _app._worldScene.SelectedObjectType == Terrain.ObjectType.Wmo ? EditorSelectionKind.WorldModel : EditorSelectionKind.Model,
                    _app.GetCurrentCaptureMapName(),
                    selected.TileX,
                    selected.TileY,
                    selected.PlacementEntryIndex,
                    selected.UniqueId,
                    selected.ModelPath,
                    selected.PlacementPosition));
            }

            return new EditorSceneSnapshot(
                _app.GetCurrentCaptureMapName(),
                new EditorCamera(_app._camera.Position, _app._camera.Forward, Vector3.UnitZ),
                [],
                selection);
        }
    }

    /// <summary>Applies an editor operation to the live scene and source file.</summary>
    private sealed class EditorApplierAdapter : IEditorOperationApplier
    {
        private readonly ViewerApp _app;

        public EditorApplierAdapter(ViewerApp app) => _app = app;

        public void Apply(EditorOperation operation)
        {
            switch (operation)
            {
                case ReconciliationApplyOperation reconciliation:
                    _app.MaterializeReconciliationOutput(reconciliation);
                    break;
                case PlacementMoveOperation move:
                    _app._worldScene?.TryUpdateSelectedPlacementPosition(move.NewPosition, out _);
                    break;
                default:
                    break;
            }
        }
    }

    /// <summary>Routes editor log lines to the viewer's existing log surface.</summary>
    private sealed class EditorLogAdapter : IEditorLog
    {
        public void Trace(string message) => ViewerLog.Trace($"[Editor] {message}");
        public void Info(string message) => ViewerLog.Info(ViewerLog.Category.General, $"[Editor] {message}");
        public void Warn(string message) => ViewerLog.Important(ViewerLog.Category.General, $"[Editor] {message}");
        public void Error(string message, Exception? exception = null)
            => ViewerLog.Error(ViewerLog.Category.General, $"[Editor] {message}{(exception is null ? "" : $" | {exception.Message}")}");
    }
}
