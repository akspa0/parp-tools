using WowViewer.Core.Runtime.World.Physics;

namespace WowViewer.Core.Tests;

/// <summary>Spec 214 — evidence-gated era resolution and solver-independent admission policy.</summary>
public sealed class PhysicsRuntimePolicyTests
{
    [Fact]
    public void EraResolver_EnablesOnlyMeasuredMopBuild()
    {
        PhysicsEraDecision decision = PhysicsEraProfileResolver.Resolve("5.0.1.15464");

        Assert.Equal(PhysicsAvailability.Enabled, decision.Availability);
        Assert.True(decision.CanAdmitCandidates);
        Assert.Equal("5.0.1.15464", decision.Evidence.BuildIdentity);
        Assert.Contains("workstream-atmosphere-501-ghidra.md", decision.Evidence.EvidenceSource);
    }

    [Fact]
    public void EraResolver_DisablesMeasuredAlphaBuild()
    {
        PhysicsEraDecision decision = PhysicsEraProfileResolver.Resolve("0.5.3.3368");

        Assert.Equal(PhysicsAvailability.Disabled, decision.Availability);
        Assert.False(decision.CanAdmitCandidates);
        Assert.Equal("0.5.3.3368", decision.Evidence.BuildIdentity);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("5.0.1")]
    [InlineData("5.0.1.16000")]
    [InlineData("4.0.0.12025")]
    public void EraResolver_FailsClosedForUnmeasuredBuilds(string? build)
    {
        PhysicsEraDecision decision = PhysicsEraProfileResolver.Resolve(build);

        Assert.Equal(PhysicsAvailability.Unknown, decision.Availability);
        Assert.False(decision.CanAdmitCandidates);
        Assert.NotEmpty(decision.Diagnostic);
        Assert.NotEmpty(decision.Evidence.ProfileId);
        Assert.NotEmpty(decision.Evidence.BuildIdentity);
        Assert.NotEmpty(decision.Evidence.EvidenceSource);
    }

    [Fact]
    public void AdmissionPolicy_DisabledBudgetReturnsReasonForEveryCandidate()
    {
        PhysicsEraDecision era = PhysicsEraProfileResolver.Resolve("5.0.1.15464");
        PhysicsCandidate[] candidates = [new("near", 2), new("far", 20)];

        IReadOnlyList<PhysicsAdmissionDecision> decisions = PhysicsAdmissionPolicy.Evaluate(
            candidates,
            era,
            new PhysicsBudget(isEnabled: false, cullDistance: 10, maximumActiveObjects: 1));

        Assert.Equal(2, decisions.Count);
        Assert.All(decisions, static decision => Assert.Equal(PhysicsAdmissionReason.PhysicsDisabled, decision.Reason));
        Assert.All(decisions, static decision => Assert.Equal(PhysicsAvailability.Enabled, decision.Availability));
        Assert.All(decisions, static decision => Assert.NotEmpty(decision.Diagnostic));
    }

    [Theory]
    [InlineData("0.5.3.3368", PhysicsAdmissionReason.KnownDisabledBuild, PhysicsAvailability.Disabled)]
    [InlineData("5.0.1.16000", PhysicsAdmissionReason.UnknownBuild, PhysicsAvailability.Unknown)]
    public void AdmissionPolicy_NonEnabledEraFailsClosed(
        string build,
        PhysicsAdmissionReason expectedReason,
        PhysicsAvailability expectedAvailability)
    {
        PhysicsEraDecision era = PhysicsEraProfileResolver.Resolve(build);

        PhysicsAdmissionDecision decision = Assert.Single(PhysicsAdmissionPolicy.Evaluate(
            [new PhysicsCandidate("flag", 1)],
            era,
            new PhysicsBudget(isEnabled: true, cullDistance: 10, maximumActiveObjects: 1)));

        Assert.Equal(expectedReason, decision.Reason);
        Assert.Equal(expectedAvailability, decision.Availability);
        Assert.Equal(era.Evidence, decision.Evidence);
        Assert.Equal(era.Diagnostic, decision.Diagnostic);
    }

    [Fact]
    public void AdmissionPolicy_CullsBeyondDistanceAndDefersOverCapacity()
    {
        PhysicsEraDecision era = PhysicsEraProfileResolver.Resolve("5.0.1.15464");
        PhysicsCandidate[] candidates =
        [
            new("far", 11),
            new("third", 3),
            new("first", 1),
            new("second", 2),
        ];

        IReadOnlyList<PhysicsAdmissionDecision> decisions = PhysicsAdmissionPolicy.Evaluate(
            candidates,
            era,
            new PhysicsBudget(isEnabled: true, cullDistance: 10, maximumActiveObjects: 2));

        Assert.Equal(PhysicsAdmissionReason.BeyondCullDistance, decisions[0].Reason);
        Assert.Equal(PhysicsAdmissionReason.ActiveObjectBudgetExhausted, decisions[1].Reason);
        Assert.Equal(PhysicsAdmissionReason.Admitted, decisions[2].Reason);
        Assert.Equal(PhysicsAdmissionReason.Admitted, decisions[3].Reason);
        Assert.All(decisions, decision => Assert.Equal(era.Evidence, decision.Evidence));
        Assert.All(decisions, static decision => Assert.Equal(PhysicsAvailability.Enabled, decision.Availability));
        Assert.All(decisions, static decision => Assert.NotEmpty(decision.Diagnostic));
    }

    [Fact]
    public void AdmissionPolicy_IsDeterministicAcrossInputOrder()
    {
        PhysicsEraDecision era = PhysicsEraProfileResolver.Resolve("5.0.1.15464");
        var budget = new PhysicsBudget(isEnabled: true, cullDistance: 10, maximumActiveObjects: 2);
        PhysicsCandidate[] forward = [new("b", 1), new("a", 1), new("priority", 9, Priority: -1)];
        PhysicsCandidate[] reverse = forward.Reverse().ToArray();

        string[] admittedForward = PhysicsAdmissionPolicy.Evaluate(forward, era, budget)
            .Where(static decision => decision.IsAdmitted)
            .Select(static decision => decision.StableId)
            .Order()
            .ToArray();
        string[] admittedReverse = PhysicsAdmissionPolicy.Evaluate(reverse, era, budget)
            .Where(static decision => decision.IsAdmitted)
            .Select(static decision => decision.StableId)
            .Order()
            .ToArray();

        Assert.Equal(["a", "priority"], admittedForward);
        Assert.Equal(admittedForward, admittedReverse);
    }

    [Fact]
    public void Budget_RejectsInvalidValues()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PhysicsBudget(true, float.NaN, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PhysicsBudget(true, -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PhysicsBudget(true, 1, -1));
    }

    [Fact]
    public void AdmissionPolicy_RejectsInvalidCandidatesInsteadOfDroppingThem()
    {
        PhysicsEraDecision era = PhysicsEraProfileResolver.Resolve("5.0.1.16000");
        var budget = new PhysicsBudget(isEnabled: false, cullDistance: 10, maximumActiveObjects: 1);

        Assert.Throws<ArgumentException>(() => PhysicsAdmissionPolicy.Evaluate(
            [new PhysicsCandidate("", 1)], era, budget));
        Assert.Throws<ArgumentOutOfRangeException>(() => PhysicsAdmissionPolicy.Evaluate(
            [new PhysicsCandidate("bad-distance", float.PositiveInfinity)], era, budget));
    }

    [Fact]
    public void AdmissionPolicy_RejectsDuplicateStableIds()
    {
        PhysicsEraDecision era = PhysicsEraProfileResolver.Resolve("5.0.1.15464");
        var budget = new PhysicsBudget(isEnabled: true, cullDistance: 10, maximumActiveObjects: 1);

        Assert.Throws<ArgumentException>(() => PhysicsAdmissionPolicy.Evaluate(
            [new PhysicsCandidate("duplicate", 1), new PhysicsCandidate("duplicate", 2)],
            era,
            budget));
    }

    [Fact]
    public void AdmissionPolicy_RejectsUnknownAvailabilityValues()
    {
        var era = new PhysicsEraDecision(
            (PhysicsAvailability)int.MaxValue,
            new PhysicsEvidence("forged", "5.0.1.15464", "test"),
            "forged");
        var budget = new PhysicsBudget(isEnabled: true, cullDistance: 10, maximumActiveObjects: 1);

        Assert.Throws<ArgumentOutOfRangeException>(() => PhysicsAdmissionPolicy.Evaluate(
            [new PhysicsCandidate("flag", 1)],
            era,
            budget));
    }
}
