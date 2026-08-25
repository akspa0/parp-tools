using System.Numerics;
using WowViewer.Core.PM4.Reconciliation;

namespace WowViewer.Core.Editor.Tests.Reconciliation;

public class Pm4ReconciliationEngineTests
{
    private static Pm4GuideObservation Guide(ExpectedAssetKind kind = ExpectedAssetKind.Model)
        => new(
            new Pm4GuideIdentity("guide-1", "dev_01.pm4", "3.3.5.12340", "development", 1, 0),
            new Vector3(16000f, 15000f, 300f),
            new Vector3(15000f, 14000f, 0f),
            new Vector3(17000f, 16000f, 400f),
            [new Vector3(16000f, 15000f, 0f)],
            HeightSignal: 300.0,
            kind,
            "2026-08",
            [new ReconciliationEvidence("height", 300.0, "surface-end")]);

    private static PlacementSnapshot Placement(ExpectedAssetKind kind = ExpectedAssetKind.Model, Vector3? position = null)
        => new(
            new PlacementIdentity("dev_1_0_obj0.adt", "development", 1, 0, kind, 0, 77, "foo.mdx", "3.3.5.12340"),
            position ?? new Vector3(16010f, 15000f, 300f),
            Vector3.Zero,
            1f);

    private static ReconciliationCandidate Candidate(string id, double score, int rank = 0, CandidateStatus status = CandidateStatus.Matched)
        => new(id, id, ExpectedAssetKind.Model, rank, score, new Dictionary<string, double> { ["geom"] = score }, [], status);

    [Fact]
    public void Align_when_kind_matches()
    {
        var proposal = Assert.Single(Pm4ReconciliationEngine.BuildProposals(Guide(), Placement(), []));

        Assert.Equal(ReconciliationAction.Align, proposal.Action);
        Assert.Equal(ProposalStatus.ReviewRequired, proposal.Status);
        Assert.Equal(new Vector3(16000f, 15000f, 300f), proposal.ProposedPosition);
        Assert.True(proposal.Residual.ContainsKey("position"));
    }

    [Fact]
    public void Substitute_unique_candidate_when_kind_mismatches()
    {
        var proposal = Assert.Single(Pm4ReconciliationEngine.BuildProposals(
            Guide(ExpectedAssetKind.WorldModel),
            Placement(ExpectedAssetKind.Model),
            [Candidate("wmo_a", 0.9)]));

        Assert.Equal(ReconciliationAction.Substitute, proposal.Action);
        Assert.Equal(ProposalStatus.ReviewRequired, proposal.Status);
        Assert.NotNull(proposal.Candidate);
        Assert.Equal("wmo_a", proposal.Candidate!.AssetId);
    }

    [Fact]
    public void Substitute_is_conflict_when_ambiguous()
    {
        var proposal = Assert.Single(Pm4ReconciliationEngine.BuildProposals(
            Guide(ExpectedAssetKind.WorldModel),
            Placement(ExpectedAssetKind.Model),
            [Candidate("wmo_a", 0.91), Candidate("wmo_b", 0.90)]));

        Assert.Equal(ProposalStatus.Conflict, proposal.Status);
        Assert.Null(proposal.Candidate);
    }

    [Fact]
    public void Clone_unique_candidate_when_no_placement()
    {
        var proposal = Assert.Single(Pm4ReconciliationEngine.BuildProposals(
            Guide(),
            existing: null,
            [Candidate("src_a", 0.85)]));

        Assert.Equal(ReconciliationAction.Clone, proposal.Action);
        Assert.Equal(ProposalStatus.ReviewRequired, proposal.Status);
        Assert.Null(proposal.ExistingPlacement);
        Assert.Equal("src_a", proposal.Candidate!.AssetId);
    }

    [Fact]
    public void Clone_is_conflict_when_no_candidate()
    {
        var proposal = Assert.Single(Pm4ReconciliationEngine.BuildProposals(Guide(), existing: null, []));

        Assert.Equal(ReconciliationAction.Clone, proposal.Action);
        Assert.Equal(ProposalStatus.Conflict, proposal.Status);
        Assert.Null(proposal.Candidate);
    }

    [Fact]
    public void Proposal_ids_are_deterministic()
    {
        var first = Pm4ReconciliationEngine.BuildProposals(Guide(), Placement(), []);
        var second = Pm4ReconciliationEngine.BuildProposals(Guide(), Placement(), []);

        Assert.Equal(first[0].ProposalId, second[0].ProposalId);
    }
}