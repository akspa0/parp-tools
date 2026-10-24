namespace WoWViewer;

/// <summary>
/// Status of a map discovery operation for a client data source.
/// </summary>
public enum DiscoveryOutcomeStatus
{
    Success,
    Empty,
    Error,
    Cancelled
}

/// <summary>
/// Immutable record of a map discovery result for a specific client activation generation.
/// </summary>
public readonly record struct DiscoveryOutcome(
    long Generation,
    DiscoveryOutcomeStatus Status,
    int MapCount,
    string? ErrorMessage = null)
{
    public bool IsSuccess => Status is DiscoveryOutcomeStatus.Success or DiscoveryOutcomeStatus.Empty;
}

/// <summary>
/// Pure deterministic state policy governing client activation identity, map discovery lifecycle,
/// and the rule that Maps opens at most once per committed client activation (Spec 286 U2, FR-007..009).
/// </summary>
public sealed class MapsActivationPolicy
{
    public long CommittedGeneration { get; private set; }
    public long ConsumedGeneration { get; private set; }
    public string? ActiveSourceName { get; private set; }
    public DiscoveryOutcome? LastCompletedDiscovery { get; private set; }

    /// <summary>
    /// Commits a new client activation with a strictly monotonic generation ID.
    /// Clears any prior discovery outcome so a previous client's list cannot be presented as new.
    /// </summary>
    public long CommitActivation(string sourceName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceName);
        CommittedGeneration++;
        ActiveSourceName = sourceName;
        LastCompletedDiscovery = null;
        return CommittedGeneration;
    }

    /// <summary>
    /// Evaluates a completed discovery outcome.
    /// Returns true if and only if this outcome matches the current committed generation,
    /// is successful (including successful-empty), and has not yet been consumed to auto-open Maps.
    /// </summary>
    public bool TryConsumeAutoOpen(DiscoveryOutcome outcome)
    {
        if (CommittedGeneration == 0 || outcome.Generation != CommittedGeneration)
            return false;

        LastCompletedDiscovery = outcome;

        if (!outcome.IsSuccess)
            return false;

        if (ConsumedGeneration == CommittedGeneration)
            return false;

        ConsumedGeneration = CommittedGeneration;
        return true;
    }

    /// <summary>
    /// Explicit user action or UI reset can mark the current generation consumed so subsequent
    /// non-activation events do not auto-open the Maps window.
    /// </summary>
    public void MarkConsumed()
    {
        if (CommittedGeneration > 0)
            ConsumedGeneration = CommittedGeneration;
    }

    /// <summary>
    /// Resets policy state to uninitialized (no active client).
    /// </summary>
    public void Reset()
    {
        CommittedGeneration = 0;
        ConsumedGeneration = 0;
        ActiveSourceName = null;
        LastCompletedDiscovery = null;
    }
}
