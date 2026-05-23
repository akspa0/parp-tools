namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureBatchResult
{
    public ValidationCaptureBatchResult(
        string mapName,
        string? buildLabel,
        int requestedResolution,
        IReadOnlyList<ValidationCaptureVariantResult> variantResults)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentNullException.ThrowIfNull(variantResults);
        if (variantResults.Any(static result => result is null))
            throw new ArgumentException("Variant results cannot contain null entries.", nameof(variantResults));

        MapName = mapName;
        BuildLabel = buildLabel;
        RequestedResolution = requestedResolution;
        VariantResults = variantResults;
    }

    public string MapName { get; }

    public string? BuildLabel { get; }

    public int RequestedResolution { get; }

    public IReadOnlyList<ValidationCaptureVariantResult> VariantResults { get; }

    public int TotalVariantCount => VariantResults.Count;

    public int SucceededVariantCount => VariantResults.Count(static result => result.Succeeded);

    public int TimedOutVariantCount => VariantResults.Count(static result => result.TimedOut);

    public int FailedVariantCount => VariantResults.Count(static result => !result.Succeeded);
}