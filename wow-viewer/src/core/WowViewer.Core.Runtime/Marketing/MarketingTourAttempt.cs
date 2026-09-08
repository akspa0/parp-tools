namespace WowViewer.Core.Runtime.Marketing;

/// <summary>Immutable render-facing state for one currently visible feature-tour beat.</summary>
public readonly record struct FeatureTourPresentation(
    string FeatureId,
    FeatureTourPresentationKind Kind,
    string Title,
    string? Body,
    double StartedAtSeconds,
    double EndsAtSeconds);

/// <summary>Typed launch result; invalid recipes cannot touch playback or capture state.</summary>
public sealed record MarketingTourAttemptStartResult(
    bool IsStarted,
    MarketingTourAttempt? Attempt,
    string? ErrorCode,
    string? Error)
{
    public static MarketingTourAttemptStartResult Rejected(FeatureTourValidationResult validation)
        => new(
            false,
            null,
            "recipe-invalid",
            validation.Errors.Count == 0
                ? "Feature-tour recipe validation failed."
                : validation.Errors[0].Message);
}

/// <summary>
/// Allocation-free-on-steady-frame lifecycle state for an active marketing tour. This class owns
/// recipe timing only; viewer composition supplies path playback, capture, and UI rendering.
/// </summary>
public sealed class MarketingTourAttempt
{
    private readonly FeatureTourRecipe _recipe;
    private readonly double _cameraPathDurationSeconds;
    private int _activeBeatIndex = -1;
    private FeatureTourPresentation? _activePresentation;
    private bool _isCancelled;
    private string? _terminalReason;

    private MarketingTourAttempt(FeatureTourRecipe recipe, double cameraPathDurationSeconds)
    {
        _recipe = recipe;
        _cameraPathDurationSeconds = cameraPathDurationSeconds;
    }

    public FeatureTourRecipe Recipe => _recipe;

    public bool IsCancelled => _isCancelled;

    public string? TerminalReason => _terminalReason;

    public FeatureTourPresentation? ActivePresentation => _activePresentation;

    public static MarketingTourAttemptStartResult TryStart(FeatureTourRecipe? recipe, double cameraPathDurationSeconds)
    {
        FeatureTourValidationResult validation = FeatureTourRecipeValidator.Validate(recipe, cameraPathDurationSeconds);
        if (!validation.IsValid)
            return MarketingTourAttemptStartResult.Rejected(validation);

        return new MarketingTourAttemptStartResult(
            true,
            new MarketingTourAttempt(recipe!, cameraPathDurationSeconds),
            null,
            null);
    }

    /// <summary>
    /// Advances presentation state for the supplied camera-path time. Steady calls that remain in
    /// one beat do not allocate, which keeps this convenience layer out of renderer measurements.
    /// </summary>
    public void Advance(double playbackSeconds)
    {
        if (_isCancelled || !double.IsFinite(playbackSeconds))
            return;

        int nextBeatIndex = FindActiveBeatIndex(Math.Clamp(playbackSeconds, 0, _cameraPathDurationSeconds));
        if (nextBeatIndex == _activeBeatIndex)
            return;

        _activeBeatIndex = nextBeatIndex;
        if (nextBeatIndex < 0)
        {
            _activePresentation = null;
            return;
        }

        FeatureTourBeat beat = _recipe.Beats[nextBeatIndex];
        _activePresentation = new FeatureTourPresentation(
            beat.FeatureId,
            beat.Kind,
            beat.Title,
            beat.Body,
            beat.AtSeconds,
            beat.AtSeconds + beat.DurationSeconds);
    }

    /// <summary>Ends presentation immediately while retaining an explicit reason for a later receipt.</summary>
    public void Cancel(string? reason)
    {
        if (_isCancelled)
            return;

        _isCancelled = true;
        _terminalReason = string.IsNullOrWhiteSpace(reason) ? "cancelled" : reason.Trim();
        _activeBeatIndex = -1;
        _activePresentation = null;
    }

    private int FindActiveBeatIndex(double playbackSeconds)
    {
        for (int index = 0; index < _recipe.Beats.Count; index++)
        {
            FeatureTourBeat beat = _recipe.Beats[index];
            if (playbackSeconds < beat.AtSeconds)
                return -1;
            if (playbackSeconds < beat.AtSeconds + beat.DurationSeconds)
                return index;
        }

        return -1;
    }
}
