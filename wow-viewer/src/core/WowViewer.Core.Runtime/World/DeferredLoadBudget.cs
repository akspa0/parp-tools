namespace WowViewer.Core.Runtime.World;

/// <summary>Asset families the deferred loader admits, tracked separately because their costs differ by an order of magnitude.</summary>
public enum DeferredLoadKind
{
    Mdx = 0,
    Wmo = 1,
}

/// <summary>
/// Admission policy for per-frame deferred asset loading (Spec 153 Defect D / FR-008).
/// <para>
/// The loop this replaces checked its budget only <em>between</em> loads:
/// <c>while (completed &lt; maxLoads &amp;&amp; elapsed &lt; budget)</c>. With 3.4 ms of a 3.5 ms
/// budget already spent, it would still start another load and discover the cost afterwards —
/// measured at 46.6 ms and 58.1 ms against a 3.5 ms budget.
/// </para>
/// <para>
/// This policy learns what a load of each kind actually costs and refuses to <em>start</em> one that
/// the remaining budget cannot pay for, so the frame stops admitting instead of overshooting
/// additively.
/// </para>
/// <para>
/// <b>What this does not fix.</b> A single synchronous load that is larger than the entire budget
/// still costs what it costs; the policy only guarantees it is never <em>added</em> to time already
/// spent. Removing that residual requires moving decode off the render thread so the budget governs
/// upload alone — Spec 153 Phase 5 step 2, deliberately not attempted here.
/// </para>
/// </summary>
public sealed class DeferredLoadBudget
{
    /// <summary>
    /// Weight of the newest sample in the cost estimate. High enough to react to a zone whose assets
    /// are uniformly heavier, low enough that one cheap cache-warm load does not erase the memory of
    /// an expensive one.
    /// </summary>
    private const double SampleWeight = 0.25;

    /// <summary>
    /// Decay applied to the high-water estimate on every sample. Without it a single pathological
    /// asset would suppress admission permanently; with it the estimate relaxes back toward typical
    /// cost over a few dozen loads.
    /// </summary>
    private const double HighWaterDecay = 0.97;

    private static readonly int KindCount = Enum.GetValues<DeferredLoadKind>().Length;

    private readonly double[] _meanMs;
    private readonly double[] _highWaterMs;
    private readonly int[] _sampleCount;

    public DeferredLoadBudget()
    {
        _meanMs = new double[KindCount];
        _highWaterMs = new double[KindCount];
        _sampleCount = new int[KindCount];
    }

    /// <summary>Loads refused because the remaining budget could not pay for them. They stay queued.</summary>
    public long BudgetDeferralCount { get; private set; }

    /// <summary>
    /// Loads admitted despite being predicted not to fit, because nothing had been loaded yet this
    /// frame. Without this escape an asset costlier than the whole budget would never become
    /// resident. This count is the honest measure of the residual the off-thread decode still owes.
    /// </summary>
    public long OversizedAdmissionCount { get; private set; }

    /// <summary>Largest single load cost observed, across all kinds, since construction.</summary>
    public double WorstObservedLoadMs { get; private set; }

    /// <summary>
    /// Predicted cost of the next load of <paramref name="kind"/>. Zero until a sample exists, so a
    /// cold policy admits freely rather than refusing work it has no evidence about.
    /// </summary>
    public double PredictedCostMs(DeferredLoadKind kind)
        => _sampleCount[(int)kind] == 0 ? 0 : Math.Max(_meanMs[(int)kind], _highWaterMs[(int)kind]);

    /// <summary>
    /// Cheapest load the policy currently believes it could start. Used pre-dequeue, where the kind
    /// of the next item is not yet known: if even the cheapest known load cannot fit, no dequeue can
    /// help. Zero when no kind has samples yet.
    /// </summary>
    public double CheapestPredictedCostMs()
    {
        double cheapest = double.MaxValue;
        for (int kind = 0; kind < KindCount; kind++)
        {
            if (_sampleCount[kind] == 0)
                return 0;

            cheapest = Math.Min(cheapest, PredictedCostMs((DeferredLoadKind)kind));
        }

        return cheapest == double.MaxValue ? 0 : cheapest;
    }

    /// <summary>
    /// Whether another load may be started this frame.
    /// <para>
    /// <paramref name="loadsStartedThisFrame"/> counts loads actually performed, not queue entries
    /// skipped as already-cached — a frame that has paid nothing has its whole budget intact.
    /// </para>
    /// </summary>
    public bool CanStartAnotherLoad(double elapsedMs, double budgetMs, int loadsStartedThisFrame)
    {
        if (budgetMs <= 0)
            return false;

        // Guaranteed progress: the first load of a frame is always admitted. Otherwise an asset
        // that is permanently larger than the budget starves and its content never appears.
        if (loadsStartedThisFrame <= 0)
        {
            if (CheapestPredictedCostMs() > budgetMs)
                OversizedAdmissionCount++;

            return true;
        }

        double remainingMs = budgetMs - elapsedMs;
        if (remainingMs <= 0)
            return false;

        if (remainingMs >= CheapestPredictedCostMs())
            return true;

        BudgetDeferralCount++;
        return false;
    }

    /// <summary>Feed back what a completed load actually cost, so the next admission decision is informed.</summary>
    public void RecordLoad(DeferredLoadKind kind, double costMs)
    {
        if (costMs < 0 || double.IsNaN(costMs))
            return;

        int index = (int)kind;
        _meanMs[index] = _sampleCount[index] == 0
            ? costMs
            : (_meanMs[index] * (1 - SampleWeight)) + (costMs * SampleWeight);
        _highWaterMs[index] = Math.Max(costMs, _highWaterMs[index] * HighWaterDecay);
        _sampleCount[index]++;

        WorstObservedLoadMs = Math.Max(WorstObservedLoadMs, costMs);
    }

    public void ResetCounters()
    {
        BudgetDeferralCount = 0;
        OversizedAdmissionCount = 0;
        WorstObservedLoadMs = 0;
    }
}
