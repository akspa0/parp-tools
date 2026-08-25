using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace WowViewer.Core.PM4.Reconciliation;

/// <summary>
/// Deterministic, side-effect-free proposal engine (Spec 176 Phase 1). It consumes PM4 guide
/// observations, the actual ADT placement snapshot, and the existing corpus matcher's ranked
/// candidates, and emits align/substitute/clone proposals with explicit status. It never writes a
/// file, never mutates input, and never grants mutation rights from a score or from proximity.
/// </summary>
/// <summary>
/// Tolerances for associating an existing placement with a guide observation. A placement is
/// associated when its position lies inside the guide's placement-space bounds expanded by these
/// tolerances. These are starting defaults, not tuned thresholds; the values actually used are
/// recorded in the proposal evidence so nothing is hidden (Spec 176 research decision 3).
/// </summary>
public sealed record ReconciliationAssociationOptions(
    float HorizontalTolerance = 10f,
    float VerticalTolerance = 50f);

public static class Pm4ReconciliationEngine
{
    public const double DefaultAmbiguityWindow = 0.03;

    /// <summary>Position residual (world units) at or below which an aligned placement is reported
    /// as <see cref="ProposalStatus.AlreadyAligned"/> instead of proposing a no-op move.</summary>
    public const double AlreadyAlignedResidualThreshold = 0.5;

    /// <summary>
    /// Maps a position residual to a display confidence: 1.0 at zero residual, falling off with a
    /// 25-unit scale. This is a proximity-derived SORTING signal only — it never grants mutation
    /// rights and is always accompanied by the raw residual in the evidence (FR-012).
    /// </summary>
    public static double ConfidenceFromResidual(double residual)
        => Math.Exp(-Math.Max(0, residual) / 25.0);

    public static IReadOnlyList<ReconciliationProposal> BuildProposals(
        Pm4GuideObservation guide,
        PlacementSnapshot? existing,
        IReadOnlyList<ReconciliationCandidate> candidates,
        double ambiguityWindow = DefaultAmbiguityWindow)
    {
        ArgumentNullException.ThrowIfNull(guide);
        ArgumentNullException.ThrowIfNull(candidates);

        if (existing is null)
        {
            return [BuildCloneProposal(guide, candidates, ambiguityWindow)];
        }

        bool kindCompatible = guide.ExpectedAssetKind == ExpectedAssetKind.Unknown
            || guide.ExpectedAssetKind == existing.Identity.Kind;

        if (kindCompatible)
        {
            return [BuildAlignProposal(guide, existing)];
        }

        return [BuildSubstituteProposal(guide, existing, candidates, ambiguityWindow)];
    }

    /// <summary>
    /// Builds the full proposal set for one tile preview: associates each guide observation with
    /// the placements whose positions fall inside its (tolerance-expanded) bounds, then delegates
    /// to <see cref="BuildProposals"/> per guide. Association is deterministic and honest:
    /// <list type="bullet">
    /// <item>no associated placement → the clone path (missing object);</item>
    /// <item>exactly one → the align/substitute path, decided by kind compatibility;</item>
    /// <item>more than one → a <see cref="ProposalStatus.Conflict"/> proposal naming every
    /// competing placement; no automatic decision is offered (FR-014).</item>
    /// </list>
    /// </summary>
    public static IReadOnlyList<ReconciliationProposal> BuildTileProposals(
        IReadOnlyList<Pm4GuideObservation> guides,
        IReadOnlyList<PlacementSnapshot> placements,
        IReadOnlyDictionary<string, IReadOnlyList<ReconciliationCandidate>> candidatesByGuideId,
        ReconciliationAssociationOptions? associationOptions = null,
        double ambiguityWindow = DefaultAmbiguityWindow)
    {
        ArgumentNullException.ThrowIfNull(guides);
        ArgumentNullException.ThrowIfNull(placements);
        ArgumentNullException.ThrowIfNull(candidatesByGuideId);

        ReconciliationAssociationOptions options = associationOptions ?? new ReconciliationAssociationOptions();
        var proposals = new List<ReconciliationProposal>(guides.Count);

        foreach (Pm4GuideObservation guide in guides)
        {
            IReadOnlyList<ReconciliationCandidate> candidates =
                candidatesByGuideId.TryGetValue(guide.Guide.GuideId, out IReadOnlyList<ReconciliationCandidate>? found)
                    ? found
                    : [];

            List<PlacementSnapshot> associated = AssociatePlacements(guide, placements, options);

            if (associated.Count == 0)
            {
                proposals.AddRange(BuildProposals(guide, existing: null, candidates, ambiguityWindow));
                continue;
            }

            if (associated.Count == 1)
            {
                PlacementSnapshot existing = associated[0];

                // An existing placement already at the guide position is a result, not work:
                // report it as already aligned instead of proposing a no-op move.
                if (guide.ExpectedAssetKind == ExpectedAssetKind.Unknown
                    || guide.ExpectedAssetKind == existing.Identity.Kind)
                {
                    double positionResidual = Vector3.Distance(guide.Position, existing.Position);
                    if (positionResidual <= AlreadyAlignedResidualThreshold)
                    {
                        proposals.Add(BuildAlreadyAlignedProposal(guide, existing));
                        continue;
                    }
                }

                proposals.AddRange(BuildProposals(guide, existing, candidates, ambiguityWindow));
                continue;
            }

            proposals.Add(BuildAssociationConflictProposal(guide, associated, options));
        }

        return proposals;
    }

    private static List<PlacementSnapshot> AssociatePlacements(
        Pm4GuideObservation guide,
        IReadOnlyList<PlacementSnapshot> placements,
        ReconciliationAssociationOptions options)
    {
        float minX = guide.BoundsMin.X - options.HorizontalTolerance;
        float maxX = guide.BoundsMax.X + options.HorizontalTolerance;
        float minY = guide.BoundsMin.Y - options.HorizontalTolerance;
        float maxY = guide.BoundsMax.Y + options.HorizontalTolerance;
        float minZ = guide.BoundsMin.Z - options.VerticalTolerance;
        float maxZ = guide.BoundsMax.Z + options.VerticalTolerance;

        var associated = new List<PlacementSnapshot>();
        foreach (PlacementSnapshot placement in placements)
        {
            Vector3 p = placement.Position;
            if (p.X >= minX && p.X <= maxX && p.Y >= minY && p.Y <= maxY && p.Z >= minZ && p.Z <= maxZ)
                associated.Add(placement);
        }

        // Deterministic ordering regardless of catalog order: by kind then unique id.
        associated.Sort(static (a, b) =>
        {
            int kind = a.Identity.Kind.CompareTo(b.Identity.Kind);
            return kind != 0 ? kind : a.Identity.UniqueId.CompareTo(b.Identity.UniqueId);
        });

        return associated;
    }

    private static ReconciliationProposal BuildAssociationConflictProposal(
        Pm4GuideObservation guide,
        IReadOnlyList<PlacementSnapshot> competing,
        ReconciliationAssociationOptions options)
    {
        var evidence = new List<ReconciliationEvidence>(guide.Evidence)
        {
            new("association-ambiguous", competing.Count,
                $"{competing.Count} placements lie inside the guide bounds (tolerances H={options.HorizontalTolerance:0.#}, V={options.VerticalTolerance:0.#}); none is selected automatically"),
        };

        foreach (PlacementSnapshot placement in competing)
            evidence.Add(new ReconciliationEvidence(
                "association-competitor",
                placement.Identity.UniqueId,
                $"{placement.Identity.Kind} entry {placement.Identity.EntryIndex} '{placement.Identity.AssetPath}'"));

        var residual = new Dictionary<string, double>();
        return new ReconciliationProposal(
            BuildProposalId(guide, existing: null, ReconciliationAction.Align, candidate: null, guide.Position),
            guide.Guide,
            ExistingPlacement: null,
            Current: null,
            ReconciliationAction.Align,
            Candidate: null,
            ProposedPosition: guide.Position,
            ProposedRotation: Vector3.Zero,
            ProposedScale: 1f,
            residual,
            Confidence: 0.0,
            evidence,
            ProposalStatus.Conflict);
    }

    private static ReconciliationProposal BuildAlreadyAlignedProposal(Pm4GuideObservation guide, PlacementSnapshot existing)
    {
        var evidence = new List<ReconciliationEvidence>(guide.Evidence)
        {
            new("already-aligned", 1.0, $"existing placement sits within {AlreadyAlignedResidualThreshold:0.#} world units of the guide position"),
        };

        return new ReconciliationProposal(
            BuildProposalId(guide, existing.Identity, ReconciliationAction.Align, candidate: null, existing.Position),
            guide.Guide,
            existing.Identity,
            existing,
            ReconciliationAction.Align,
            Candidate: null,
            ProposedPosition: existing.Position,
            ProposedRotation: existing.Rotation,
            ProposedScale: existing.Scale,
            new Dictionary<string, double>(),
            Confidence: 1.0,
            evidence,
            ProposalStatus.AlreadyAligned);
    }

    private static ReconciliationProposal BuildAlignProposal(Pm4GuideObservation guide, PlacementSnapshot existing)
    {
        double positionResidual = Vector3.Distance(guide.Position, existing.Position);

        var residual = new Dictionary<string, double> { ["position"] = positionResidual };
        if (guide.HeightSignal is not null)
            residual["height"] = Math.Abs(guide.HeightSignal.Value - existing.Position.Z);

        var evidence = new List<ReconciliationEvidence>(guide.Evidence)
        {
            new("placement-kind", 1.0, "guide ExpectedAssetKind matches existing placement kind"),
            new("position-residual", positionResidual, "distance between guide and existing placement"),
            new("alignment-confidence", ConfidenceFromResidual(positionResidual), "display-only sorting signal derived from the position residual (25-unit scale); never grants mutation rights"),
        };

        return new ReconciliationProposal(
            BuildProposalId(guide, existing.Identity, ReconciliationAction.Align, candidate: null, guide.Position),
            guide.Guide,
            existing.Identity,
            existing,
            ReconciliationAction.Align,
            Candidate: null,
            ProposedPosition: guide.Position,
            ProposedRotation: existing.Rotation,
            ProposedScale: existing.Scale,
            residual,
            Confidence: ConfidenceFromResidual(positionResidual),
            evidence,
            ProposalStatus.ReviewRequired);
    }

    private static ReconciliationProposal BuildSubstituteProposal(
        Pm4GuideObservation guide,
        PlacementSnapshot existing,
        IReadOnlyList<ReconciliationCandidate> candidates,
        double ambiguityWindow)
    {
        (ReconciliationCandidate? candidate, CandidateStatus selection) = SelectCandidate(candidates, ambiguityWindow);

        var evidence = new List<ReconciliationEvidence>(guide.Evidence)
        {
            new("placement-kind-mismatch", 1.0, "existing placement kind differs from guide ExpectedAssetKind"),
        };

        if (selection == CandidateStatus.Matched && candidate is not null)
        {
            evidence.Add(new("substitution-candidate", candidate.Score, candidate.AssetPath));
            return new ReconciliationProposal(
                BuildProposalId(guide, existing.Identity, ReconciliationAction.Substitute, candidate, guide.Position),
                guide.Guide,
                existing.Identity,
                existing,
                ReconciliationAction.Substitute,
                candidate,
                ProposedPosition: guide.Position,
                ProposedRotation: existing.Rotation,
                ProposedScale: existing.Scale,
                Residual: new Dictionary<string, double>(),
                Confidence: Clamp01(candidate.Score),
                evidence,
                ProposalStatus.ReviewRequired);
        }

        evidence.Add(new("substitution-ambiguous", 1.0, selection == CandidateStatus.Ambiguous ? "top candidates within ambiguity window" : "no matching candidate"));
        return new ReconciliationProposal(
            BuildProposalId(guide, existing.Identity, ReconciliationAction.Substitute, candidate: null, guide.Position),
            guide.Guide,
            existing.Identity,
            existing,
            ReconciliationAction.Substitute,
            Candidate: null,
            ProposedPosition: guide.Position,
            ProposedRotation: existing.Rotation,
            ProposedScale: existing.Scale,
            Residual: new Dictionary<string, double>(),
            Confidence: 0.0,
            evidence,
            ProposalStatus.Conflict);
    }

    private static ReconciliationProposal BuildCloneProposal(
        Pm4GuideObservation guide,
        IReadOnlyList<ReconciliationCandidate> candidates,
        double ambiguityWindow)
    {
        (ReconciliationCandidate? candidate, CandidateStatus selection) = SelectCandidate(candidates, ambiguityWindow);

        var evidence = new List<ReconciliationEvidence>(guide.Evidence)
        {
            new("missing-placement", 1.0, "no existing Museum placement corresponds to this guide object"),
        };

        if (selection == CandidateStatus.Matched && candidate is not null)
        {
            evidence.Add(new("clone-candidate", candidate.Score, candidate.AssetPath));
            return new ReconciliationProposal(
                BuildProposalId(guide, existing: null, ReconciliationAction.Clone, candidate, guide.Position),
                guide.Guide,
                ExistingPlacement: null,
                Current: null,
                ReconciliationAction.Clone,
                candidate,
                ProposedPosition: guide.Position,
                ProposedRotation: Vector3.Zero,
                ProposedScale: 1f,
                Residual: new Dictionary<string, double>(),
                Confidence: Clamp01(candidate.Score),
                evidence,
                ProposalStatus.ReviewRequired);
        }

        evidence.Add(new("clone-ambiguous", 1.0, selection == CandidateStatus.Ambiguous ? "top candidates within ambiguity window" : "no matching candidate"));
        return new ReconciliationProposal(
            BuildProposalId(guide, existing: null, ReconciliationAction.Clone, candidate: null, guide.Position),
            guide.Guide,
            ExistingPlacement: null,
            Current: null,
            ReconciliationAction.Clone,
            Candidate: null,
            ProposedPosition: guide.Position,
            ProposedRotation: Vector3.Zero,
            ProposedScale: 1f,
            Residual: new Dictionary<string, double>(),
            Confidence: 0.0,
            evidence,
            ProposalStatus.Conflict);
    }

    private static (ReconciliationCandidate? Candidate, CandidateStatus Selection) SelectCandidate(
        IReadOnlyList<ReconciliationCandidate> candidates,
        double ambiguityWindow)
    {
        ReconciliationCandidate[] eligible = candidates
            .Where(c => c.Status != CandidateStatus.Ineligible)
            .OrderBy(c => c.Rank)
            .ThenByDescending(c => c.Score)
            .ToArray();

        if (eligible.Length == 0)
            return (null, CandidateStatus.Unresolved);

        ReconciliationCandidate top = eligible[0];
        if (top.Status == CandidateStatus.Unresolved)
            return (null, CandidateStatus.Unresolved);

        if (top.Status == CandidateStatus.Ambiguous)
            return (null, CandidateStatus.Ambiguous);

        // A second-ranked candidate within the ambiguity window means the result is ambiguous, so
        // no automatic decision is proposed.
        if (eligible.Length >= 2)
        {
            ReconciliationCandidate second = eligible[1];
            if (second.Status != CandidateStatus.Ineligible && (top.Score - second.Score) < ambiguityWindow)
                return (null, CandidateStatus.Ambiguous);
        }

        return (top, CandidateStatus.Matched);
    }

    private static double Clamp01(double value)
        => Math.Clamp(value, 0.0, 1.0);

    private static string BuildProposalId(
        Pm4GuideObservation guide,
        PlacementIdentity? existing,
        ReconciliationAction action,
        ReconciliationCandidate? candidate,
        System.Numerics.Vector3 position)
    {
        var builder = new StringBuilder();
        builder.Append(guide.Guide.GuideId).Append('|');
        builder.Append(guide.Guide.SourcePath).Append('|');
        builder.Append(action).Append('|');
        builder.Append(existing is null ? "-" : $"{existing.Kind}:{existing.EntryIndex}:{existing.UniqueId}").Append('|');
        builder.Append(candidate?.AssetId ?? "-").Append('|');
        builder.Append(position.X.ToString("R")).Append(',').Append(position.Y.ToString("R")).Append(',').Append(position.Z.ToString("R"));

        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        return Convert.ToHexString(hash.AsSpan(0, 16)).ToLowerInvariant();
    }
}