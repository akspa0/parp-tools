using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;

namespace WoWViewer.Workbench.Services.Pm4Workbench;

public enum Pm4ReviewQueueStatus
{
    Pending,
    Accepted,
    Rejected,
    Replaced,
    Unmatched,
}

public sealed record Pm4ReviewQueueItem(
    Pm4HullIdentity Hull,
    Pm4HullShapeInput HullInput,
    Pm4ShapeMatchProposal Proposal,
    Pm4ShapeCandidate? ActiveTopCandidate,
    Pm4ReviewQueueStatus Status,
    string? CommittedModelPath,
    string? Provenance,
    DateTimeOffset? DecidedUtc);

/// <summary>
/// Spec 295 Phase 5 (T018, T022): UI-independent human-in-the-loop review queue for PM4 shape proposals.
/// Enforces SC-006 (no automatic pairing is committed without an explicit operator decision unless
/// bounded auto-accept mode is deliberately enabled for that run).
/// </summary>
public sealed class Pm4ShapeMatchReviewService
{
    private readonly List<Pm4ReviewQueueItem> _queue = [];

    public IReadOnlyList<Pm4ReviewQueueItem> Queue => _queue;
    public int SelectedQueueIndex { get; private set; } = -1;

    /// <summary>
    /// Spec 295 T022 / SC-006: Bounded auto-accept is opt-in per run and defaults to false.
    /// </summary>
    public bool BoundedAutoAcceptEnabled { get; set; }
    public float BoundedAutoAcceptThreshold { get; set; } = 0.85f;

    public Pm4ReviewQueueItem? SelectedItem =>
        SelectedQueueIndex >= 0 && SelectedQueueIndex < _queue.Count ? _queue[SelectedQueueIndex] : null;

    public int PendingCount => _queue.Count(static q => q.Status == Pm4ReviewQueueStatus.Pending);
    public int AcceptedCount => _queue.Count(static q => q.Status == Pm4ReviewQueueStatus.Accepted);
    public int RejectedCount => _queue.Count(static q => q.Status == Pm4ReviewQueueStatus.Rejected);
    public int ReplacedCount => _queue.Count(static q => q.Status == Pm4ReviewQueueStatus.Replaced);
    public int UnmatchedCount => _queue.Count(static q => q.Status == Pm4ReviewQueueStatus.Unmatched);

    public void SelectIndex(int index)
    {
        if (_queue.Count == 0)
        {
            SelectedQueueIndex = -1;
            return;
        }
        SelectedQueueIndex = Math.Clamp(index, 0, _queue.Count - 1);
    }

    public void ClearQueue()
    {
        _queue.Clear();
        SelectedQueueIndex = -1;
    }

    /// <summary>
    /// Enqueues or refreshes proposals without committing any pairing to <paramref name="project"/> (SC-006).
    /// Respects already-persisted decisions and suppresses rejected candidate-hull pairs (FR-010).
    /// </summary>
    public void EnqueueProposals(
        IEnumerable<(Pm4HullShapeInput HullInput, Pm4ShapeMatchProposal Proposal)> analyzedHulls,
        ManualCurationProject project)
    {
        ArgumentNullException.ThrowIfNull(analyzedHulls);
        ArgumentNullException.ThrowIfNull(project);

        foreach (var (hullInput, proposal) in analyzedHulls)
        {
            Pm4HullIdentity hull = ToHullIdentity(hullInput);
            Pm4ReviewQueueItem item = BuildQueueItem(hull, hullInput, proposal, project);

            int existingIdx = _queue.FindIndex(q => q.Hull == hull);
            if (existingIdx >= 0)
                _queue[existingIdx] = item;
            else
                _queue.Add(item);
        }

        if (SelectedQueueIndex < 0 && _queue.Count > 0)
            SelectedQueueIndex = 0;
    }

    public bool Accept(
        Pm4HullIdentity hull,
        ManualCurationProject project,
        DateTimeOffset? timestampUtc = null,
        string provenance = "Manual",
        string? notes = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        int idx = _queue.FindIndex(q => q.Hull == hull);
        if (idx < 0)
            return false;

        Pm4ReviewQueueItem item = _queue[idx];
        if (item.ActiveTopCandidate is not { } candidate)
            return false;

        DateTimeOffset now = timestampUtc ?? DateTimeOffset.UtcNow;
        var decision = new ManualShapeMatchDecisionRecord(
            Hull: hull,
            ProposedModelPath: candidate.ModelPath,
            ChosenModelPath: candidate.ModelPath,
            AssetKind: candidate.AssetKind,
            Decision: ManualShapeMatchDecisionKind.Accepted,
            Confidence: candidate.Confidence,
            ShapeDistance: candidate.ShapeDistance,
            CalibratedCentroid: item.Proposal.CalibratedCentroid,
            Provenance: provenance,
            DecidedUtc: now);

        if (!ManualCurationProjectPolicy.RecordShapeMatchDecision(project, decision, notes))
            return false;

        _queue[idx] = item with
        {
            Status = Pm4ReviewQueueStatus.Accepted,
            CommittedModelPath = candidate.ModelPath,
            Provenance = provenance,
            DecidedUtc = now
        };
        return true;
    }

    /// <summary>
    /// Rejects the active top candidate for <paramref name="hull"/> (FR-007, FR-010).
    /// Suppresses the rejected pair and promotes the next non-rejected candidate if one exists.
    /// </summary>
    public bool Reject(
        Pm4HullIdentity hull,
        ManualCurationProject project,
        DateTimeOffset? timestampUtc = null,
        string provenance = "Manual",
        string? notes = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        int idx = _queue.FindIndex(q => q.Hull == hull);
        if (idx < 0)
            return false;

        Pm4ReviewQueueItem item = _queue[idx];
        Pm4ShapeCandidate? candidate = item.ActiveTopCandidate ?? item.Proposal.TopCandidate;
        if (candidate == null)
            return false;

        DateTimeOffset now = timestampUtc ?? DateTimeOffset.UtcNow;
        var decision = new ManualShapeMatchDecisionRecord(
            Hull: hull,
            ProposedModelPath: candidate.ModelPath,
            ChosenModelPath: null,
            AssetKind: candidate.AssetKind,
            Decision: ManualShapeMatchDecisionKind.Rejected,
            Confidence: candidate.Confidence,
            ShapeDistance: candidate.ShapeDistance,
            CalibratedCentroid: item.Proposal.CalibratedCentroid,
            Provenance: provenance,
            DecidedUtc: now);

        if (!ManualCurationProjectPolicy.RecordShapeMatchDecision(project, decision, notes))
            return false;

        // Re-evaluate next non-rejected candidate (FR-010)
        Pm4ShapeCandidate? nextCandidate = item.Proposal.Candidates
            .FirstOrDefault(c => !ManualCurationProjectPolicy.IsCandidateRejectedForHull(project, hull, c.ModelPath));

        _queue[idx] = item with
        {
            ActiveTopCandidate = nextCandidate,
            Status = nextCandidate != null ? Pm4ReviewQueueStatus.Pending : Pm4ReviewQueueStatus.Rejected,
            CommittedModelPath = null,
            Provenance = provenance,
            DecidedUtc = now
        };
        return true;
    }

    public bool Replace(
        Pm4HullIdentity hull,
        string replacementModelPath,
        string replacementAssetKind,
        ManualCurationProject project,
        DateTimeOffset? timestampUtc = null,
        string provenance = "Manual",
        string? notes = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        if (string.IsNullOrWhiteSpace(replacementModelPath))
            return false;

        int idx = _queue.FindIndex(q => q.Hull == hull);
        if (idx < 0)
            return false;

        Pm4ReviewQueueItem item = _queue[idx];
        string proposedPath = item.ActiveTopCandidate?.ModelPath
            ?? item.Proposal.TopCandidate?.ModelPath
            ?? replacementModelPath;
        float confidence = item.ActiveTopCandidate?.Confidence ?? 1.0f;
        float distance = item.ActiveTopCandidate?.ShapeDistance ?? 0f;

        DateTimeOffset now = timestampUtc ?? DateTimeOffset.UtcNow;
        var decision = new ManualShapeMatchDecisionRecord(
            Hull: hull,
            ProposedModelPath: proposedPath,
            ChosenModelPath: replacementModelPath,
            AssetKind: string.IsNullOrWhiteSpace(replacementAssetKind) ? "WMO" : replacementAssetKind,
            Decision: ManualShapeMatchDecisionKind.Replaced,
            Confidence: confidence,
            ShapeDistance: distance,
            CalibratedCentroid: item.Proposal.CalibratedCentroid,
            Provenance: provenance,
            DecidedUtc: now);

        if (!ManualCurationProjectPolicy.RecordShapeMatchDecision(project, decision, notes))
            return false;

        _queue[idx] = item with
        {
            Status = Pm4ReviewQueueStatus.Replaced,
            CommittedModelPath = replacementModelPath,
            Provenance = provenance,
            DecidedUtc = now
        };
        return true;
    }

    public int ResetRejections(ManualCurationProject project, Pm4HullIdentity? hull = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        int removed = ManualCurationProjectPolicy.ResetRejectedShapeMatchDecisions(project, hull);
        for (int i = 0; i < _queue.Count; i++)
        {
            if (hull == null || _queue[i].Hull == hull.Value)
                _queue[i] = BuildQueueItem(_queue[i].Hull, _queue[i].HullInput, _queue[i].Proposal, project);
        }
        return removed;
    }

    /// <summary>
    /// Spec 295 T022 / SC-006: Commits high-confidence non-conflicting pending proposals ONLY when
    /// <see cref="BoundedAutoAcceptEnabled"/> is explicitly enabled for this run.
    /// </summary>
    public int RunBoundedAutoAccept(
        ManualCurationProject project,
        IReadOnlyList<Pm4ProposalConflict>? conflicts = null,
        DateTimeOffset? timestampUtc = null)
    {
        ArgumentNullException.ThrowIfNull(project);
        if (!BoundedAutoAcceptEnabled)
            return 0;

        var conflictedKeys = new HashSet<(int TileX, int TileY, uint Ck24, int ObjectPartId)>();
        if (conflicts != null)
        {
            foreach (var c in conflicts)
            {
                conflictedKeys.Add((c.FirstHull.TileX, c.FirstHull.TileY, c.FirstHull.Ck24, c.FirstHull.ObjectPartId));
                conflictedKeys.Add((c.SecondHull.TileX, c.SecondHull.TileY, c.SecondHull.Ck24, c.SecondHull.ObjectPartId));
            }
        }

        int accepted = 0;
        for (int i = 0; i < _queue.Count; i++)
        {
            Pm4ReviewQueueItem item = _queue[i];
            if (item.Status != Pm4ReviewQueueStatus.Pending || item.ActiveTopCandidate is not { } candidate)
                continue;

            if (item.Proposal.IsUnmatched || item.Proposal.IsLowConfidence)
                continue;

            if (candidate.Confidence < BoundedAutoAcceptThreshold)
                continue;

            if (conflictedKeys.Contains((item.Hull.TileX, item.Hull.TileY, item.Hull.Ck24, item.Hull.ObjectPartId)))
                continue;

            if (Accept(
                    item.Hull,
                    project,
                    timestampUtc,
                    provenance: $"BoundedAutoAccept(>={BoundedAutoAcceptThreshold:P0})",
                    notes: "Explicit bounded auto-accept run"))
            {
                accepted++;
            }
        }

        return accepted;
    }

    /// <summary>
    /// Spec 295 FR-008: Returns ranked replacement alternatives from the visual library filtered by compatible footprint proportions.
    /// </summary>
    public IReadOnlyList<(Pm4ShapeLibraryVisualEntry Entry, float ShapeDistance, float Confidence)> GetReplacementAlternatives(
        Pm4HullShapeInput hullInput,
        Pm4ShapeLibraryVisualService library,
        ManualCurationProject? project = null,
        float maxPlanarAspectRatioDelta = 1.5f,
        int maxResults = 12)
    {
        ArgumentNullException.ThrowIfNull(hullInput);
        ArgumentNullException.ThrowIfNull(library);

        Vector3 hullDims = hullInput.Descriptor.BoundingBoxDimensions;
        float hullLen = MathF.Max(MathF.Max(hullDims.X, hullDims.Y), 0.1f);
        float hullWid = MathF.Max(MathF.Min(hullDims.X, hullDims.Y), 0.1f);
        float hullAspect = hullLen / hullWid;
        Pm4HullIdentity hullId = ToHullIdentity(hullInput);
        var criteria = new Pm4ShapeMatchingCriteria(DimensionRatioTolerance: 1.5f);

        var ranked = new List<(Pm4ShapeLibraryVisualEntry Entry, float ShapeDistance, float Confidence)>();
        foreach (var entry in library.AllEntries)
        {
            if (!entry.IsHealthy || entry.AssetKind.Equals("PM4Hull", StringComparison.OrdinalIgnoreCase))
                continue;

            if (project != null && ManualCurationProjectPolicy.IsCandidateRejectedForHull(project, hullId, entry.PrimaryModelPath))
                continue;

            float aspectDelta = MathF.Abs(entry.Preview.PlanarAspectRatio - hullAspect);
            if (aspectDelta > maxPlanarAspectRatioDelta)
                continue;

            float distance = Pm4DirectModelMatcher.CompareShapes(hullInput.Descriptor, entry.Descriptor, criteria);
            if (float.IsInfinity(distance))
            {
                // Still allow proportion-compatible manual replacement with fallback distance
                Vector3 dimDiff = entry.Dimensions - new Vector3(hullLen, hullWid, MathF.Max(hullDims.Z, 0.1f));
                distance = dimDiff.Length() / MathF.Max(hullLen, 1f);
            }

            float confidence = 1f / (1f + MathF.Max(0f, distance));
            ranked.Add((entry, distance, confidence));
        }

        return ranked
            .OrderBy(static r => r.ShapeDistance)
            .ThenByDescending(static r => r.Entry.IsCurated)
            .Take(Math.Max(1, maxResults))
            .ToArray();
    }

    public static Pm4HullIdentity ToHullIdentity(Pm4HullShapeInput input) =>
        new(input.TileX, input.TileY, input.Ck24, input.ObjectPartId, input.LinkGroupObjectId);

    private static Pm4ReviewQueueItem BuildQueueItem(
        Pm4HullIdentity hull,
        Pm4HullShapeInput hullInput,
        Pm4ShapeMatchProposal proposal,
        ManualCurationProject project)
    {
        if (ManualCurationProjectPolicy.TryGetActiveShapeMatchDecision(project, hull, out var activeDecision) &&
            activeDecision != null)
        {
            Pm4ReviewQueueStatus status = activeDecision.Decision == ManualShapeMatchDecisionKind.Replaced
                ? Pm4ReviewQueueStatus.Replaced
                : Pm4ReviewQueueStatus.Accepted;

            Pm4ShapeCandidate? chosenCandidate = proposal.Candidates
                .FirstOrDefault(c => string.Equals(c.ModelPath, activeDecision.ChosenModelPath, StringComparison.OrdinalIgnoreCase))
                ?? proposal.TopCandidate;

            return new Pm4ReviewQueueItem(
                hull,
                hullInput,
                proposal,
                chosenCandidate,
                status,
                activeDecision.ChosenModelPath ?? activeDecision.ProposedModelPath,
                activeDecision.Provenance,
                activeDecision.DecidedUtc);
        }

        Pm4ShapeCandidate? nonRejectedTop = proposal.Candidates
            .FirstOrDefault(c => !ManualCurationProjectPolicy.IsCandidateRejectedForHull(project, hull, c.ModelPath));

        if (proposal.Candidates.Count == 0 || proposal.IsUnmatched)
        {
            return new Pm4ReviewQueueItem(
                hull,
                hullInput,
                proposal,
                null,
                Pm4ReviewQueueStatus.Unmatched,
                null,
                null,
                null);
        }

        if (nonRejectedTop == null)
        {
            return new Pm4ReviewQueueItem(
                hull,
                hullInput,
                proposal,
                null,
                Pm4ReviewQueueStatus.Rejected,
                null,
                "Manual",
                null);
        }

        return new Pm4ReviewQueueItem(
            hull,
            hullInput,
            proposal,
            nonRejectedTop,
            Pm4ReviewQueueStatus.Pending,
            null,
            null,
            null);
    }
}

