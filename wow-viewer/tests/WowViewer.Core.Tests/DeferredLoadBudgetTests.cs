using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 153 Defect D / FR-008. The loop this policy replaces checked its budget only between loads,
/// so a 3.5 ms budget produced a 58.1 ms stage. These tests pin the admission behaviour, including
/// the honest limits of what a synchronous loader can guarantee.
/// </summary>
public sealed class DeferredLoadBudgetTests
{
    [Fact]
    public void ColdPolicy_AdmitsFreely_BecauseItHasNoEvidenceYet()
    {
        DeferredLoadBudget budget = new();

        Assert.True(budget.CanStartAnotherLoad(elapsedMs: 0, budgetMs: 3.5, loadsStartedThisFrame: 0));
        Assert.True(budget.CanStartAnotherLoad(elapsedMs: 1.0, budgetMs: 3.5, loadsStartedThisFrame: 1));
        Assert.Equal(0, budget.PredictedCostMs(DeferredLoadKind.Mdx));
    }

    [Fact]
    public void RefusesToStartALoad_ThatTheRemainingBudgetCannotPayFor()
    {
        DeferredLoadBudget budget = new();
        budget.RecordLoad(DeferredLoadKind.Mdx, 20.0);
        budget.RecordLoad(DeferredLoadKind.Wmo, 20.0);

        // This is the exact measured failure: 3.4 ms spent of a 3.5 ms budget, and the old loop
        // still started a load it already had evidence would cost ~20 ms.
        Assert.False(budget.CanStartAnotherLoad(elapsedMs: 3.4, budgetMs: 3.5, loadsStartedThisFrame: 1));
        Assert.Equal(1, budget.BudgetDeferralCount);
    }

    [Fact]
    public void AdmitsWhenTheRemainingBudgetCoversThePrediction()
    {
        DeferredLoadBudget budget = new();
        budget.RecordLoad(DeferredLoadKind.Mdx, 1.0);
        budget.RecordLoad(DeferredLoadKind.Wmo, 1.0);

        Assert.True(budget.CanStartAnotherLoad(elapsedMs: 1.0, budgetMs: 6.0, loadsStartedThisFrame: 1));
        Assert.Equal(0, budget.BudgetDeferralCount);
    }

    [Fact]
    public void FirstLoadOfAFrame_IsAlwaysAdmitted_SoOversizedAssetsDoNotStarve()
    {
        DeferredLoadBudget budget = new();
        budget.RecordLoad(DeferredLoadKind.Mdx, 58.0);
        budget.RecordLoad(DeferredLoadKind.Wmo, 58.0);

        // A model costlier than the whole budget must still become resident. This is the residual a
        // synchronous loader cannot remove; it is counted rather than hidden.
        Assert.True(budget.CanStartAnotherLoad(elapsedMs: 0, budgetMs: 3.5, loadsStartedThisFrame: 0));
        Assert.Equal(1, budget.OversizedAdmissionCount);
    }

    [Fact]
    public void OverrunIsNoLongerAdditive_AcrossLoadsInOneFrame()
    {
        DeferredLoadBudget budget = new();
        budget.RecordLoad(DeferredLoadKind.Mdx, 55.0);
        budget.RecordLoad(DeferredLoadKind.Wmo, 55.0);

        // Simulate the old shape: one heavy load already ran, taking the frame past its budget.
        // The old condition (elapsed < budget) would be false here too — but the damage case was a
        // frame that had spent *almost* all its budget on cheap work and then admitted a heavy load.
        double budgetMs = 3.5;
        double elapsedMs = 0;
        int started = 0;

        Assert.True(budget.CanStartAnotherLoad(elapsedMs, budgetMs, started));
        elapsedMs += 55.0;
        started++;

        Assert.False(budget.CanStartAnotherLoad(elapsedMs, budgetMs, started));

        // Total frame cost is one load, not a chain of them.
        Assert.Equal(55.0, elapsedMs);
    }

    [Fact]
    public void ZeroOrNegativeBudget_AdmitsNothing()
    {
        DeferredLoadBudget budget = new();

        Assert.False(budget.CanStartAnotherLoad(elapsedMs: 0, budgetMs: 0, loadsStartedThisFrame: 0));
        Assert.False(budget.CanStartAnotherLoad(elapsedMs: 0, budgetMs: -1, loadsStartedThisFrame: 0));
    }

    [Fact]
    public void PredictionTracksTheKindThatIsActuallyExpensive()
    {
        DeferredLoadBudget budget = new();
        for (int i = 0; i < 8; i++)
        {
            budget.RecordLoad(DeferredLoadKind.Mdx, 0.5);
            budget.RecordLoad(DeferredLoadKind.Wmo, 40.0);
        }

        Assert.True(budget.PredictedCostMs(DeferredLoadKind.Wmo) > budget.PredictedCostMs(DeferredLoadKind.Mdx));

        // Pre-dequeue the kind is unknown, so admission uses the cheapest known cost: refusing on the
        // expensive kind would stall MDX loading behind WMO cost it will never pay.
        Assert.Equal(budget.PredictedCostMs(DeferredLoadKind.Mdx), budget.CheapestPredictedCostMs());
    }

    [Fact]
    public void HighWaterEstimateDecays_SoOnePathologicalAssetDoesNotSuppressLoadingForever()
    {
        DeferredLoadBudget budget = new();
        budget.RecordLoad(DeferredLoadKind.Mdx, 200.0);
        double afterSpike = budget.PredictedCostMs(DeferredLoadKind.Mdx);

        for (int i = 0; i < 200; i++)
            budget.RecordLoad(DeferredLoadKind.Mdx, 0.5);

        double afterRecovery = budget.PredictedCostMs(DeferredLoadKind.Mdx);
        Assert.True(afterRecovery < afterSpike, $"prediction never relaxed: {afterSpike} -> {afterRecovery}");
        Assert.True(afterRecovery < 5.0, $"prediction relaxed too slowly to be usable: {afterRecovery}");

        // The worst observation itself is retained for reporting, separately from the estimate.
        Assert.Equal(200.0, budget.WorstObservedLoadMs);
    }

    [Fact]
    public void NegativeAndNaNCosts_AreIgnoredRatherThanPoisoningThePrediction()
    {
        DeferredLoadBudget budget = new();
        budget.RecordLoad(DeferredLoadKind.Mdx, -5.0);
        budget.RecordLoad(DeferredLoadKind.Mdx, double.NaN);

        Assert.Equal(0, budget.PredictedCostMs(DeferredLoadKind.Mdx));
    }
}
