using WoWViewer;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class MapsActivationPolicyTests
{
    [Fact]
    public void InitialState_HasZeroGenerationsAndNoDiscovery()
    {
        var policy = new MapsActivationPolicy();
        Assert.Equal(0, policy.CommittedGeneration);
        Assert.Equal(0, policy.ConsumedGeneration);
        Assert.Null(policy.ActiveSourceName);
        Assert.Null(policy.LastCompletedDiscovery);
    }

    [Fact]
    public void FirstActivation_IncrementsGenerationMonotonically_AndConsumesOnceOnSuccess()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation(@"H:\CLIENTS\WoW_1.12.1");

        Assert.Equal(1, gen);
        Assert.Equal(1, policy.CommittedGeneration);
        Assert.Equal(0, policy.ConsumedGeneration);
        Assert.Equal(@"H:\CLIENTS\WoW_1.12.1", policy.ActiveSourceName);
        Assert.Null(policy.LastCompletedDiscovery);

        // Discovery completes successfully with 15 maps
        var outcome = new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Success, 15);
        bool shouldOpen = policy.TryConsumeAutoOpen(outcome);

        Assert.True(shouldOpen);
        Assert.Equal(1, policy.ConsumedGeneration);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(15, policy.LastCompletedDiscovery.Value.MapCount);

        // Subsequent call in the same generation does not auto-open again
        bool secondAttempt = policy.TryConsumeAutoOpen(outcome);
        Assert.False(secondAttempt);
        Assert.Equal(1, policy.ConsumedGeneration);
    }

    [Fact]
    public void ReplacementActivation_AdvancesGeneration_ClearsPriorDiscovery_AndConsumesForNewGeneration()
    {
        var policy = new MapsActivationPolicy();
        long gen1 = policy.CommitActivation("Client1");
        policy.TryConsumeAutoOpen(new DiscoveryOutcome(gen1, DiscoveryOutcomeStatus.Success, 10));
        Assert.Equal(1, policy.ConsumedGeneration);

        // Replace client
        long gen2 = policy.CommitActivation("Client2");
        Assert.Equal(2, gen2);
        Assert.Equal(2, policy.CommittedGeneration);
        Assert.Equal(1, policy.ConsumedGeneration); // not yet consumed for gen 2
        Assert.Equal("Client2", policy.ActiveSourceName);
        Assert.Null(policy.LastCompletedDiscovery); // prior discovery cleared immediately

        // Complete discovery for gen 2
        var outcome2 = new DiscoveryOutcome(gen2, DiscoveryOutcomeStatus.Success, 42);
        bool shouldOpen2 = policy.TryConsumeAutoOpen(outcome2);

        Assert.True(shouldOpen2);
        Assert.Equal(2, policy.ConsumedGeneration);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(42, policy.LastCompletedDiscovery.Value.MapCount);
    }

    [Fact]
    public void MapsAlreadyOpen_MarkConsumed_PreventsSubsequentAutoOpenInSameGeneration()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation("Client1");

        // User explicitly opened maps before discovery finished
        policy.MarkConsumed();
        Assert.Equal(gen, policy.ConsumedGeneration);

        // When discovery finishes, auto-open is suppressed because it's already consumed
        var outcome = new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Success, 5);
        bool shouldOpen = policy.TryConsumeAutoOpen(outcome);

        Assert.False(shouldOpen);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(5, policy.LastCompletedDiscovery.Value.MapCount);
    }

    [Fact]
    public void CloseThenOrdinaryRefresh_DoesNotReopenMaps()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation("Client1");
        bool firstOpen = policy.TryConsumeAutoOpen(new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Success, 8));
        Assert.True(firstOpen);
        Assert.Equal(1, policy.ConsumedGeneration);

        // User closes the window (UI state Open = false, policy generation unchanged).
        // Then an ordinary refresh runs (e.g. RefreshDiscoveredMaps without CommitActivation).
        var refreshOutcome = new DiscoveryOutcome(policy.CommittedGeneration, DiscoveryOutcomeStatus.Success, 8);
        bool refreshOpen = policy.TryConsumeAutoOpen(refreshOutcome);

        Assert.False(refreshOpen);
        Assert.Equal(1, policy.ConsumedGeneration);
    }

    [Fact]
    public void ZeroMaps_SuccessfulEmptyClient_AutoOpensOnceWithEmptyStatus()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation("EmptyWc3Client");

        var outcome = new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Empty, 0);
        Assert.True(outcome.IsSuccess);

        bool shouldOpen = policy.TryConsumeAutoOpen(outcome);
        Assert.True(shouldOpen);
        Assert.Equal(gen, policy.ConsumedGeneration);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(0, policy.LastCompletedDiscovery.Value.MapCount);
        Assert.Equal(DiscoveryOutcomeStatus.Empty, policy.LastCompletedDiscovery.Value.Status);

        // Further attempts in same generation return false
        Assert.False(policy.TryConsumeAutoOpen(outcome));
    }

    [Fact]
    public void Failure_DoesNotAdvanceCommittedGeneration_AndDoesNotAutoOpen()
    {
        var policy = new MapsActivationPolicy();
        // If client load throws before CommitActivation, generation stays 0
        Assert.Equal(0, policy.CommittedGeneration);

        // Any outcome reported for generation 0 or without active client fails
        var outcome = new DiscoveryOutcome(0, DiscoveryOutcomeStatus.Success, 10);
        Assert.False(policy.TryConsumeAutoOpen(outcome));
        Assert.Equal(0, policy.ConsumedGeneration);
    }

    [Fact]
    public void Cancellation_DoesNotAutoOpenMaps()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation("Client1");

        var cancelOutcome = new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Cancelled, 0, "Cancelled by user");
        Assert.False(cancelOutcome.IsSuccess);

        bool shouldOpen = policy.TryConsumeAutoOpen(cancelOutcome);
        Assert.False(shouldOpen);
        Assert.Equal(0, policy.ConsumedGeneration);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(DiscoveryOutcomeStatus.Cancelled, policy.LastCompletedDiscovery.Value.Status);
    }

    [Fact]
    public void DiscoveryError_DoesNotAutoOpenMaps_AndRecordsErrorOutcome()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation("CorruptedClient");

        var errorOutcome = new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Error, 0, "DBC corrupted");
        Assert.False(errorOutcome.IsSuccess);

        bool shouldOpen = policy.TryConsumeAutoOpen(errorOutcome);
        Assert.False(shouldOpen);
        Assert.Equal(0, policy.ConsumedGeneration);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(DiscoveryOutcomeStatus.Error, policy.LastCompletedDiscovery.Value.Status);
        Assert.Equal("DBC corrupted", policy.LastCompletedDiscovery.Value.ErrorMessage);
    }

    [Fact]
    public void OutOfOrderStaleCompletion_IsDroppedAndNeverPresentsPreviousClientData()
    {
        var policy = new MapsActivationPolicy();
        long gen1 = policy.CommitActivation("Client1");
        long gen2 = policy.CommitActivation("Client2"); // gen 2 replaces gen 1 before gen 1 finished

        Assert.Equal(2, policy.CommittedGeneration);

        // Gen 1 finishes late
        var staleGen1Outcome = new DiscoveryOutcome(gen1, DiscoveryOutcomeStatus.Success, 100);
        bool staleConsumed = policy.TryConsumeAutoOpen(staleGen1Outcome);

        Assert.False(staleConsumed);
        Assert.Null(policy.LastCompletedDiscovery); // Stale outcome did not overwrite null
        Assert.Equal(0, policy.ConsumedGeneration);

        // Gen 2 completes
        var gen2Outcome = new DiscoveryOutcome(gen2, DiscoveryOutcomeStatus.Success, 25);
        bool gen2Consumed = policy.TryConsumeAutoOpen(gen2Outcome);

        Assert.True(gen2Consumed);
        Assert.Equal(2, policy.ConsumedGeneration);
        Assert.NotNull(policy.LastCompletedDiscovery);
        Assert.Equal(25, policy.LastCompletedDiscovery.Value.MapCount);
    }

    [Fact]
    public void Reset_ClearsAllPolicyState()
    {
        var policy = new MapsActivationPolicy();
        long gen = policy.CommitActivation("Client1");
        policy.TryConsumeAutoOpen(new DiscoveryOutcome(gen, DiscoveryOutcomeStatus.Success, 5));

        policy.Reset();

        Assert.Equal(0, policy.CommittedGeneration);
        Assert.Equal(0, policy.ConsumedGeneration);
        Assert.Null(policy.ActiveSourceName);
        Assert.Null(policy.LastCompletedDiscovery);
    }
}
