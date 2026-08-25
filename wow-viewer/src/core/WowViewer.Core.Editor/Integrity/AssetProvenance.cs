namespace WowViewer.Core.Editor.Integrity;

/// <summary>
/// Provenance that must accompany any written file (Spec 173 FR-006): the source build, the
/// contributing operations, validation results, and any losses a lossy downport path recorded. No
/// client asset bytes are part of provenance.
/// </summary>
public sealed record AssetProvenance
{
    public AssetProvenance(
        string sourceBuild,
        IReadOnlyList<string>? contributingOperations = null,
        IReadOnlyList<AssetValidationResult>? validationResults = null,
        IReadOnlyList<string>? losses = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceBuild);

        SourceBuild = sourceBuild;
        ContributingOperations = contributingOperations ?? [];
        ValidationResults = validationResults ?? [];
        Losses = losses ?? [];
    }

    public string SourceBuild { get; init; }

    public IReadOnlyList<string> ContributingOperations { get; init; }

    public IReadOnlyList<AssetValidationResult> ValidationResults { get; init; }

    public IReadOnlyList<string> Losses { get; init; }

    public string? FilePath { get; init; }

    public bool IsComplete => !string.IsNullOrWhiteSpace(SourceBuild);
}