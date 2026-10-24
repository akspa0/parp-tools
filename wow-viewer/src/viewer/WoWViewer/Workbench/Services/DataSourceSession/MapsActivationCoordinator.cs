namespace WoWViewer;

/// <summary>
/// Coordinates client activation generations, map discovery outcomes, and once-per-activation
/// Maps window opening (Spec 286 U2, FR-007..009).
/// </summary>
internal sealed class MapsActivationCoordinator
{
    private readonly MapsActivationPolicy _policy = new();
    private readonly Action _openMapsWindow;

    internal MapsActivationCoordinator(Action openMapsWindow)
    {
        _openMapsWindow = openMapsWindow ?? throw new ArgumentNullException(nameof(openMapsWindow));
    }

    internal MapsActivationPolicy Policy => _policy;
    internal long CurrentGeneration => _policy.CommittedGeneration;
    internal string? ActiveSourceName => _policy.ActiveSourceName;
    internal DiscoveryOutcome? LastCompletedDiscovery => _policy.LastCompletedDiscovery;

    /// <summary>
    /// Commits a successful data source activation and begins a new generation.
    /// Clears any prior discovery state.
    /// </summary>
    internal long CommitActivation(string sourceName)
    {
        return _policy.CommitActivation(sourceName);
    }

    /// <summary>
    /// Reports the result of a map discovery operation. If it matches the current committed
    /// generation, is successful (including 0 maps), and has not been consumed yet, triggers auto-open.
    /// </summary>
    internal bool NotifyDiscoveryCompleted(long generation, int mapCount, bool isSuccess, string? errorMessage = null)
    {
        DiscoveryOutcomeStatus status = !isSuccess
            ? DiscoveryOutcomeStatus.Error
            : (mapCount == 0 ? DiscoveryOutcomeStatus.Empty : DiscoveryOutcomeStatus.Success);

        var outcome = new DiscoveryOutcome(generation, status, mapCount, errorMessage);
        bool shouldOpen = _policy.TryConsumeAutoOpen(outcome);
        if (shouldOpen)
        {
            _openMapsWindow();
        }
        return shouldOpen;
    }

    /// <summary>
    /// Reports that discovery or client loading was cancelled for this generation.
    /// </summary>
    internal void NotifyDiscoveryCancelled(long generation, string? reason = null)
    {
        var outcome = new DiscoveryOutcome(generation, DiscoveryOutcomeStatus.Cancelled, 0, reason);
        _policy.TryConsumeAutoOpen(outcome);
    }

    /// <summary>
    /// Explicit user action or UI reset to open Maps marks the current generation consumed.
    /// </summary>
    internal void MarkConsumed()
    {
        _policy.MarkConsumed();
    }

    /// <summary>
    /// Resets all activation state.
    /// </summary>
    internal void Reset()
    {
        _policy.Reset();
    }
}
