using System.Numerics;
using WowViewer.Core.PM4.Reconciliation;

namespace WowViewer.Core.Editor.Tests.Reconciliation;

public class ReconciliationTileProposalTests
{
    private static readonly Vector3 GuideCenter = new(16000f, 15000f, 300f);

    private static Pm4GuideObservation Guide(ExpectedAssetKind kind = ExpectedAssetKind.Model)
        => new(
            new Pm4GuideIdentity("guide-1", "dev_01_00.pm4", "3.3.5.12340", "development", 0, 1),
            GuideCenter,
            GuideCenter - new Vector3(5f, 5f, 20f),
            GuideCenter + new Vector3(5f, 5f, 20f),
            [new Vector3(GuideCenter.X, GuideCenter.Y, 0f)],
            HeightSignal: 300d,
            kind,
            "pm4-segment-signals/v1",
            [new ReconciliationEvidence("pm4-segment-surfaces", 1, "test")]);

    private static PlacementSnapshot Placement(
        int uniqueId,
        Vector3? position = null,
        ExpectedAssetKind kind = ExpectedAssetKind.Model,
        int entryIndex = 0)
        => new(
            new PlacementIdentity("synthetic_1_0_obj0.adt", "development", 1, 0, kind, entryIndex, uniqueId, "foo.mdx", "3.3.5.12340"),
            position ?? GuideCenter,
            Vector3.Zero,
            1f);

    private static ReconciliationCandidate Candidate(string id, double score)
        => new(id, $"{id}.asset", ExpectedAssetKind.WorldModel, 0, score,
            new Dictionary<string, double> { ["geom"] = score }, [], CandidateStatus.Matched);

    [Fact]
    public void Placement_inside_guide_bounds_produces_align()
    {
        var proposals = Pm4ReconciliationEngine.BuildTileProposals(
            [Guide()],
            [Placement(77)],
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>());

        ReconciliationProposal proposal = Assert.Single(proposals);
        Assert.Equal(ReconciliationAction.Align, proposal.Action);
        Assert.Equal(ProposalStatus.ReviewRequired, proposal.Status);
        Assert.Equal(GuideCenter, proposal.ProposedPosition);
        Assert.True(proposal.Residual.ContainsKey("position"));
    }

    [Fact]
    public void Missing_placement_produces_clone_conflict_without_candidates()
    {
        var proposals = Pm4ReconciliationEngine.BuildTileProposals(
            [Guide()],
            [],
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>());

        ReconciliationProposal proposal = Assert.Single(proposals);
        Assert.Equal(ReconciliationAction.Clone, proposal.Action);
        Assert.Equal(ProposalStatus.Conflict, proposal.Status);
        Assert.Null(proposal.Candidate);
    }

    [Fact]
    public void Missing_placement_with_unique_candidate_produces_reviewable_clone()
    {
        var proposals = Pm4ReconciliationEngine.BuildTileProposals(
            [Guide()],
            [],
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>
            {
                ["guide-1"] = [Candidate("wmo:donor", 0.85d)],
            });

        ReconciliationProposal proposal = Assert.Single(proposals);
        Assert.Equal(ReconciliationAction.Clone, proposal.Action);
        Assert.Equal(ProposalStatus.ReviewRequired, proposal.Status);
        Assert.NotNull(proposal.Candidate);
        Assert.Equal("wmo:donor", proposal.Candidate!.AssetId);
    }

    [Fact]
    public void Kind_mismatch_inside_bounds_produces_substitute()
    {
        var proposals = Pm4ReconciliationEngine.BuildTileProposals(
            [Guide(ExpectedAssetKind.WorldModel)],
            [Placement(77, kind: ExpectedAssetKind.Model)],
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>
            {
                ["guide-1"] = [Candidate("wmo:a", 0.9d)],
            });

        ReconciliationProposal proposal = Assert.Single(proposals);
        Assert.Equal(ReconciliationAction.Substitute, proposal.Action);
        Assert.Equal(ProposalStatus.ReviewRequired, proposal.Status);
        Assert.Equal("wmo:a", proposal.Candidate!.AssetId);
    }

    [Fact]
    public void Two_placements_inside_bounds_is_an_association_conflict()
    {
        var proposals = Pm4ReconciliationEngine.BuildTileProposals(
            [Guide()],
            [Placement(77), Placement(78, entryIndex: 1)],
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>());

        ReconciliationProposal proposal = Assert.Single(proposals);
        Assert.Equal(ProposalStatus.Conflict, proposal.Status);

        int competitorCount = proposal.Evidence.Count(e => e.Signal == "association-competitor");
        Assert.Equal(2, competitorCount);
        Assert.Contains(proposal.Evidence, e => e.Signal == "association-ambiguous");
    }

    [Fact]
    public void Placement_outside_tolerated_bounds_is_not_associated()
    {
        Vector3 farAway = GuideCenter + new Vector3(500f, 0f, 0f);
        var proposals = Pm4ReconciliationEngine.BuildTileProposals(
            [Guide()],
            [Placement(77, position: farAway)],
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>());

        ReconciliationProposal proposal = Assert.Single(proposals);
        // Not associated -> treated as a missing object (clone path), not as an align.
        Assert.Equal(ReconciliationAction.Clone, proposal.Action);
    }

    [Fact]
    public void Tile_proposals_are_deterministic()
    {
        PlacementSnapshot[] placements = [Placement(77)];
        Pm4GuideObservation[] guides = [Guide()];

        IReadOnlyDictionary<string, IReadOnlyList<ReconciliationCandidate>> noCandidates =
            new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>();
        var first = Pm4ReconciliationEngine.BuildTileProposals(guides, placements, noCandidates);
        var second = Pm4ReconciliationEngine.BuildTileProposals(guides, placements, noCandidates);

        ReconciliationProposal a = Assert.Single(first);
        ReconciliationProposal b = Assert.Single(second);
        Assert.Equal(a.ProposalId, b.ProposalId);
    }
}
