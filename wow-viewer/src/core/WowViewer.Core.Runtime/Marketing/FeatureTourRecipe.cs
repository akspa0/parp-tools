namespace WowViewer.Core.Runtime.Marketing;

/// <summary>
/// The visual form a feature-tour beat may request. The first production tour uses a callout;
/// registered-control presentation is deliberately reserved for a later UI registry.
/// </summary>
public enum FeatureTourPresentationKind
{
    Callout = 0,
    RegisteredControl = 1,
}

/// <summary>Capture constraints that make a tour's presentation reproducible.</summary>
public sealed record FeatureTourCaptureSettings(int FramesPerSecond, bool FullFrameWithTourOverlay);

/// <summary>One visible, ordered feature-presentation interval during camera-path playback.</summary>
public sealed record FeatureTourBeat(
    double AtSeconds,
    FeatureTourPresentationKind Kind,
    string FeatureId,
    string Title,
    string? Body,
    double DurationSeconds);

/// <summary>
/// A versioned, path-independent tour definition. Client locations are intentionally absent: the
/// active viewer supplies map/build and a display-safe camera-path identity at launch time.
/// </summary>
public sealed record FeatureTourRecipe(
    string SchemaVersion,
    string Id,
    string DisplayName,
    string Version,
    bool RequiresWarmPath,
    FeatureTourCaptureSettings Capture,
    IReadOnlyList<FeatureTourBeat> Beats)
{
    public const string SchemaV1 = "wow-viewer.feature-tour.v1";
}

/// <summary>One explicit reason a recipe cannot change capture state.</summary>
public sealed record FeatureTourValidationError(string Code, string Message);

/// <summary>Pure validation result used by the viewer and external automation hosts alike.</summary>
public sealed record FeatureTourValidationResult(IReadOnlyList<FeatureTourValidationError> Errors)
{
    public bool IsValid => Errors.Count == 0;

    public static FeatureTourValidationResult Success { get; } = new(Array.Empty<FeatureTourValidationError>());
}

/// <summary>Validates recipe structure before any warmup, playback, or encoder process begins.</summary>
public static class FeatureTourRecipeValidator
{
    public static FeatureTourValidationResult Validate(FeatureTourRecipe? recipe, double cameraPathDurationSeconds)
    {
        if (recipe is null)
        {
            return new FeatureTourValidationResult(
            [
                new FeatureTourValidationError("recipe-missing", "A feature-tour recipe is required."),
            ]);
        }

        var errors = new List<FeatureTourValidationError>();
        if (!string.Equals(recipe.SchemaVersion, FeatureTourRecipe.SchemaV1, StringComparison.Ordinal))
            errors.Add(new("recipe-schema-unsupported", $"Unsupported feature-tour schema '{recipe.SchemaVersion}'."));
        if (!IsSlug(recipe.Id))
            errors.Add(new("recipe-id-invalid", "Recipe id must be a lowercase slug."));
        if (string.IsNullOrWhiteSpace(recipe.DisplayName))
            errors.Add(new("recipe-display-name-missing", "Recipe display name is required."));
        if (string.IsNullOrWhiteSpace(recipe.Version))
            errors.Add(new("recipe-version-missing", "Recipe version is required."));

        if (recipe.Capture is null)
        {
            errors.Add(new("capture-missing", "Feature-tour capture settings are required."));
        }
        else
        {
            if (recipe.Capture.FramesPerSecond is < 12 or > 60)
                errors.Add(new("capture-fps-out-of-range", "Feature-tour FPS must be between 12 and 60."));
            if (!recipe.Capture.FullFrameWithTourOverlay)
                errors.Add(new("capture-must-use-full-frame-overlay", "Feature tours must use the full-frame tour overlay capture tap."));
        }

        if (!double.IsFinite(cameraPathDurationSeconds) || cameraPathDurationSeconds <= 0)
            errors.Add(new("camera-path-duration-invalid", "Camera-path duration must be a positive finite value."));

        if (recipe.Beats is null)
        {
            errors.Add(new("beats-missing", "Feature-tour beats are required."));
            return new FeatureTourValidationResult(errors);
        }

        double previousStart = double.NegativeInfinity;
        double previousEnd = double.NegativeInfinity;
        for (int index = 0; index < recipe.Beats.Count; index++)
        {
            FeatureTourBeat? beat = recipe.Beats[index];
            if (beat is null)
            {
                errors.Add(new("beat-missing", $"Beat {index + 1} is missing."));
                continue;
            }

            if (!double.IsFinite(beat.AtSeconds) || beat.AtSeconds < 0)
                errors.Add(new("beat-time-invalid", $"Beat {index + 1} must have a non-negative finite timestamp."));
            if (!double.IsFinite(beat.DurationSeconds) || beat.DurationSeconds <= 0)
                errors.Add(new("beat-duration-invalid", $"Beat {index + 1} must have a positive finite duration."));
            if (!Enum.IsDefined(beat.Kind))
                errors.Add(new("beat-kind-unsupported", $"Beat {index + 1} has an unsupported presentation kind."));
            if (!IsSlug(beat.FeatureId))
                errors.Add(new("beat-feature-id-invalid", $"Beat {index + 1} feature id must be a lowercase slug."));
            if (string.IsNullOrWhiteSpace(beat.Title))
                errors.Add(new("beat-title-missing", $"Beat {index + 1} title is required."));

            if (double.IsFinite(beat.AtSeconds) && double.IsFinite(cameraPathDurationSeconds)
                && (beat.AtSeconds > cameraPathDurationSeconds || beat.AtSeconds + Math.Max(0, beat.DurationSeconds) > cameraPathDurationSeconds))
            {
                errors.Add(new("beat-outside-camera-path", $"Beat {index + 1} extends beyond the camera-path duration."));
            }

            if (beat.AtSeconds <= previousStart)
                errors.Add(new("beats-not-strictly-ordered", $"Beat {index + 1} must start after the preceding beat."));
            if (beat.AtSeconds < previousEnd)
                errors.Add(new("beats-overlap", $"Beat {index + 1} overlaps the preceding beat."));

            previousStart = beat.AtSeconds;
            previousEnd = Math.Max(previousEnd, beat.AtSeconds + Math.Max(0, beat.DurationSeconds));
        }

        return errors.Count == 0 ? FeatureTourValidationResult.Success : new FeatureTourValidationResult(errors);
    }

    private static bool IsSlug(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || value[0] == '-' || value[^1] == '-')
            return false;

        bool previousWasHyphen = false;
        foreach (char character in value)
        {
            bool isLowercaseLetter = character is >= 'a' and <= 'z';
            bool isDigit = character is >= '0' and <= '9';
            if (isLowercaseLetter || isDigit)
            {
                previousWasHyphen = false;
                continue;
            }

            if (character == '-' && !previousWasHyphen)
            {
                previousWasHyphen = true;
                continue;
            }

            return false;
        }

        return true;
    }
}
