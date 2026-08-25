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

        // PM4/Museum reconciliation (Spec 176 Phases 3-4)
        DrawReconciliationPanel();
    }

    private void DrawPlacementAuthoringPanel()
    {
        ImGui.Text("Placement Authoring");
        ImGui.TextDisabled("Move/rotate/scale/add/delete selected ADT placements. Changes are staged as undoable editor operations.");

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

        ImGui.TextDisabled($"Tile ({selected.TileX}, {selected.TileY})  Entry {selected.PlacementEntryIndex}  UniqueId {selected.UniqueId}");

        Vector3 position = selected.PlacementPosition;
        if (ImGui.InputFloat3("Position", ref position, "%.3f"))
        {
            if (_worldScene.TryUpdateSelectedPlacementPosition(position, out string error))
            {
                StagePlacementMove(selected, position);
            }
            else
            {
                ImGui.TextDisabled(error);
            }
        }

        Vector3 rotation = selected.PlacementRotation;
        if (ImGui.InputFloat3("Rotation", ref rotation, "%.3f"))
        {
            StagePlacementRotate(selected, rotation);
        }

        float scale = selected.PlacementScale;
        if (ImGui.InputFloat("Scale", ref scale, 0.01f))
        {
            StagePlacementScale(selected, scale);
        }

        if (ImGui.Button("Delete Placement"))
        {
            StagePlacementDelete(selected);
        }
    }

    private void StagePlacementMove(ObjectInstance selected, Vector3 newPosition)
    {
        if (_editorSession == null)
            return;

        AdtPlacementKind kind = _worldScene?.SelectedObjectType == Terrain.ObjectType.Wmo
            ? AdtPlacementKind.WorldModel
            : AdtPlacementKind.Model;

        var operation = new PlacementMoveOperation(
            $"move-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
            "placement.authoring",
            selected.ModelPath,
            kind,
            selected.PlacementEntryIndex,
            selected.UniqueId,
            selected.PlacementPosition,
            newPosition);

        _editorSession.RecordApplied(operation);
    }

    private void StagePlacementRotate(ObjectInstance selected, Vector3 newRotation)
    {
        if (_editorSession == null)
            return;

        AdtPlacementKind kind = _worldScene?.SelectedObjectType == Terrain.ObjectType.Wmo
            ? AdtPlacementKind.WorldModel
            : AdtPlacementKind.Model;

        var operation = new PlacementRotateOperation(
            $"rotate-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
            "placement.authoring",
            selected.ModelPath,
            kind,
            selected.PlacementEntryIndex,
            selected.UniqueId,
            selected.PlacementRotation,
            newRotation);

        _editorSession.RecordApplied(operation);
    }

    private void StagePlacementScale(ObjectInstance selected, float newScale)
    {
        if (_editorSession == null)
            return;

        AdtPlacementKind kind = _worldScene?.SelectedObjectType == Terrain.ObjectType.Wmo
            ? AdtPlacementKind.WorldModel
            : AdtPlacementKind.Model;

        var operation = new PlacementScaleOperation(
            $"scale-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
            "placement.authoring",
            selected.ModelPath,
            kind,
            selected.PlacementEntryIndex,
            selected.UniqueId,
            selected.PlacementScale,
            newScale);

        _editorSession.RecordApplied(operation);
    }

    private void StagePlacementDelete(ObjectInstance selected)
    {
        if (_editorSession == null)
            return;

        AdtPlacementKind kind = _worldScene?.SelectedObjectType == Terrain.ObjectType.Wmo
            ? AdtPlacementKind.WorldModel
            : AdtPlacementKind.Model;

        var operation = new PlacementDeleteOperation(
            $"delete-{selected.UniqueId}-{DateTime.UtcNow.Ticks}",
            "placement.authoring",
            selected.ModelPath,
            kind,
            selected.PlacementEntryIndex,
            selected.UniqueId);

        _editorSession.RecordApplied(operation);
    }

    /// <summary>
    /// Prefills the reconciliation paths from the live scene: the current session map directory and
    /// camera tile. PM4 filenames are <c><map>_<Ytile:D2>_<Xtile:D2>.pm4</c> while
    /// the placement ADT is <c><map>_<Ytile>_<Xtile>_obj0.adt</c> — the measured
    /// filename convention on <see cref="Pm4CoordinateService"/> (first number bounds Y).
    /// </summary>
    internal void PrefillReconciliationPathsFromScene()
    {
        bool museumMissing = string.IsNullOrWhiteSpace(_reconciliationMuseumPath) || !File.Exists(_reconciliationMuseumPath);
        bool pm4Missing = string.IsNullOrWhiteSpace(_reconciliationPm4Path) || !File.Exists(_reconciliationPm4Path);
        if (!museumMissing && !pm4Missing)
            return;

        string? mapName = GetCurrentSessionMapName();
        if (string.IsNullOrWhiteSpace(mapName))
            return;

        string? mapDirectory = TryResolveCurrentMapDirectory(preferLooseOverlay: true);
        if (string.IsNullOrWhiteSpace(mapDirectory) || !Directory.Exists(mapDirectory))
            return;

        (int tileX, int tileY)? cameraTile = _worldScene?.GetPm4CameraTile();
        if (cameraTile is null)
            return;

        if (museumMissing)
        {
            string museumPath = Path.Combine(mapDirectory, $"{mapName}_{cameraTile.Value.tileY}_{cameraTile.Value.tileX}_obj0.adt");
            if (File.Exists(museumPath))
                _reconciliationMuseumPath = museumPath;
        }

        if (pm4Missing)
        {
            string pm4Path = Path.Combine(mapDirectory, $"{mapName}_{cameraTile.Value.tileY:D2}_{cameraTile.Value.tileX:D2}.pm4");
            if (File.Exists(pm4Path))
                _reconciliationPm4Path = pm4Path;
        }
    }

    private void DrawReconciliationPanel()
    {
        ImGui.Text("PM4/Museum Reconciliation");
        ImGui.TextDisabled("Preview alignment/substitution/clone proposals from a PM4 guide against the Museum placement catalog. Nothing is written during preview.");

        ImGui.SameLine();
        if (ImGui.Button("Use Camera Tile"))
        {
            _reconciliationPm4Path = string.Empty;
            _reconciliationMuseumPath = string.Empty;
            PrefillReconciliationPathsFromScene();
            _reconciliationStatus = File.Exists(_reconciliationPm4Path) && File.Exists(_reconciliationMuseumPath)
                ? "Prefilled paths for the camera tile."
                : "No PM4/_obj0 pair found on disk for the camera tile.";
        }

        ImGui.InputText("PM4 guide", ref _reconciliationPm4Path, 512);
        ImGui.InputText("Museum ADT", ref _reconciliationMuseumPath, 512);
        ImGui.InputText("Output dir", ref _reconciliationOutputDir, 512);
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
            _reconciliationStatus = "Preview cleared.";
        }

        ImGui.TextDisabled(_reconciliationStatus);

        if (_reconciliationProposals.Count == 0)
            return;

        ImGui.Separator();
        ImGui.Text($"{_reconciliationProposals.Count} proposal(s)");

        foreach (ReconciliationProposal proposal in _reconciliationProposals)
        {
            DrawReconciliationProposal(proposal);
        }

        ImGui.Separator();
        if (ImGui.Button("Apply Accepted"))
        {
            ApplyAcceptedReconciliation();
        }
    }

    private void RunReconciliationPreview()
    {
        _reconciliationStatus = "Preview requires a PM4 guide and a Museum ADT.";
        if (string.IsNullOrWhiteSpace(_reconciliationPm4Path) || string.IsNullOrWhiteSpace(_reconciliationMuseumPath))
            return;

        if (!File.Exists(_reconciliationPm4Path) || !File.Exists(_reconciliationMuseumPath))
        {
            _reconciliationStatus = "One of the configured paths does not exist.";
            return;
        }

        try
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(_reconciliationPm4Path, out int tileX, out int tileY))
            {
                _reconciliationStatus = "Could not parse tile coordinates from the PM4 guide filename.";
                return;
            }

            // 1. Parse the real PM4 guide into object segments (existing segment builder owner).
            IReadOnlyList<Pm4BuiltObjectSegment> segments = Pm4ObjectSegmentBuilder.Build(_reconciliationPm4Path);
            if (segments.Count == 0)
            {
                _reconciliationStatus = "The PM4 guide produced no usable object segments (no MSUR surfaces with indexable geometry).";
                return;
            }

            // 2. Read the Museum placement catalog and fingerprint the source bytes.
            byte[] museumBytes = File.ReadAllBytes(_reconciliationMuseumPath);
            AdtPlacementCatalog catalog = AdtPlacementReader.Read(_reconciliationMuseumPath);

            string fingerprint = string.IsNullOrWhiteSpace(_reconciliationBuildFingerprint)
                ? "unspecified"
                : _reconciliationBuildFingerprint;
            string mapName = Path.GetFileNameWithoutExtension(_reconciliationMuseumPath);

            // 3. Score segments against the labelled Museum self-corpus (no archive access).
            var corpus = Pm4ReconciliationInputAdapter.BuildSelfCorpusReferences(catalog, $"{tileX}_{tileY}", fingerprint);
            IReadOnlyList<Pm4SegmentMatchResult> matchResults = Pm4AssetMatchScorer.ScoreSegments(segments, corpus);

            // 4. Build guide observations, placement snapshots, and candidate lists.
            var guides = Pm4ReconciliationInputAdapter.BuildGuideObservations(
                matchResults, _reconciliationPm4Path, fingerprint, mapName, tileX, tileY);
            var snapshots = Pm4ReconciliationInputAdapter.BuildPlacementSnapshots(
                catalog, mapName, tileX, tileY, fingerprint);
            var candidatesByGuide = Pm4ReconciliationInputAdapter.BuildCandidatesByGuideId(matchResults);

            // 5. Build the proposal set (side-effect free).
            _reconciliationProposals = Pm4ReconciliationEngine.BuildTileProposals(guides, snapshots, candidatesByGuide);
            _reconciliationDecisions.Clear();

            // 6. Record source fingerprints so apply can refuse stale previews.
            _reconciliationSourceHash = ReconciliationApplyService.ComputeSha256Hex(museumBytes);
            _reconciliationGuideHash = ReconciliationApplyService.ComputeSha256Hex(File.ReadAllBytes(_reconciliationPm4Path));

            _reconciliationStatus =
                $"Previewed {_reconciliationProposals.Count} proposal(s) from {segments.Count} PM4 object(s) against {snapshots.Count} placement(s) ({corpus.Count} self-corpus assets).";
        }
        catch (Exception ex)
        {
            _reconciliationStatus = $"Preview failed: {ex.Message}";
        }
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

        string status = proposal.Status switch
        {
            ProposalStatus.ReviewRequired => "Review required",
            ProposalStatus.Conflict => "Conflict",
            ProposalStatus.Unsupported => "Unsupported",
            _ => proposal.Status.ToString(),
        };

        if (ImGui.CollapsingHeader($"{action} {proposal.ProposalId} ({status})"))
        {
            ImGui.TextDisabled($"Guide: {proposal.Guide.GuideId}");
            ImGui.TextDisabled($"Proposed position: ({proposal.ProposedPosition.X:F1}, {proposal.ProposedPosition.Y:F1}, {proposal.ProposedPosition.Z:F1})");
            ImGui.TextDisabled($"Confidence: {proposal.Confidence:F3}");

            if (proposal.Candidate != null)
                ImGui.TextDisabled($"Candidate: {proposal.Candidate.AssetPath} (score {proposal.Candidate.Score:F3})");

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

            bool canAccept = proposal.Status == ProposalStatus.ReviewRequired;
            if (canAccept)
            {
                if (ImGui.Button($"Accept##{proposal.ProposalId}"))
                    _reconciliationDecisions[proposal.ProposalId] = ReviewDisposition.Accept;
                ImGui.SameLine();
                if (ImGui.Button($"Reject##{proposal.ProposalId}"))
                    _reconciliationDecisions[proposal.ProposalId] = ReviewDisposition.Reject;
            }
            else
            {
                ImGui.TextDisabled("This proposal cannot be accepted automatically.");
            }
        }
    }

    private void ApplyAcceptedReconciliation()
    {
        if (_editorSession == null)
            return;

        var accepted = _reconciliationProposals
            .Where(p => _reconciliationDecisions.TryGetValue(p.ProposalId, out ReviewDisposition d) && d == ReviewDisposition.Accept)
            .ToList();

        if (accepted.Count == 0)
        {
            _reconciliationStatus = "No accepted proposals to apply.";
            return;
        }

        if (_reconciliationSourceHash == null)
        {
            _reconciliationStatus = "Run a preview first; the apply validates against the preview's source fingerprints.";
            return;
        }

        try
        {
            // Stage the edited bytes and the provenance report in memory; the service refuses a
            // stale source, an empty batch, or any proposal that is not actionable.
            byte[] sourceBytes = File.ReadAllBytes(_reconciliationMuseumPath);
            (AdtPlacementEditResult result, ReconciliationProvenanceReport report) =
                ReconciliationApplyService.ApplyAccepted(
                    accepted,
                    sourceBytes,
                    _reconciliationMuseumPath,
                    _reconciliationSourceHash,
                    _reconciliationPm4Path,
                    _reconciliationGuideHash,
                    string.IsNullOrWhiteSpace(_reconciliationBuildFingerprint) ? "unspecified" : _reconciliationBuildFingerprint);

            string outputDir = string.IsNullOrWhiteSpace(_reconciliationOutputDir)
                ? _editorProjectOutputDir
                : _reconciliationOutputDir;
            if (string.IsNullOrWhiteSpace(outputDir))
            {
                _reconciliationStatus = "No output directory configured.";
                return;
            }

            // Resolve through the session so protected roots and container paths are refused.
            string fullPath = Path.GetFullPath(Path.Combine(outputDir, Path.GetFileName(_reconciliationMuseumPath)));
            string outputPath = _editorSession.ResolveOutputPath(fullPath);
            string reportPath = outputPath + ".reconciliation.json";

            byte[]? priorBytes = File.Exists(outputPath) ? File.ReadAllBytes(outputPath) : null;
            string reportJson = JsonSerializer.Serialize(report, ReconciliationJsonOptions);

            var operation = new ReconciliationApplyOperation(
                $"reconciliation-{report.BatchId}",
                "pm4.reconciliation",
                _reconciliationMuseumPath,
                outputPath,
                reportPath,
                priorBytes,
                result.Bytes,
                reportJson);

            MaterializeReconciliationOutput(operation);
            _editorSession.RecordApplied(operation);

            _reconciliationStatus =
                $"Applied {accepted.Count} accepted proposal(s) to {outputPath} (report: {Path.GetFileName(reportPath)}).";
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