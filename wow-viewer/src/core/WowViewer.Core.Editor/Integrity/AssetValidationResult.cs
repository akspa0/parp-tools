namespace WowViewer.Core.Editor.Integrity;

/// <summary>
/// The outcome of validating one asset on read. A quarantined asset carries named diagnostics;
/// an unverified asset carries nothing and still blocks a write.
/// </summary>
public sealed record AssetValidationResult
{
    public AssetValidationResult(
        string assetPath,
        ValidationVerdict verdict,
        IReadOnlyList<AssetDiagnostic>? diagnostics = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(assetPath);

        AssetPath = assetPath;
        Verdict = verdict;
        Diagnostics = diagnostics ?? [];
    }

    public string AssetPath { get; init; }

    public ValidationVerdict Verdict { get; init; }

    public IReadOnlyList<AssetDiagnostic> Diagnostics { get; init; }

    public static AssetValidationResult Verified(string path) => new(path, ValidationVerdict.Verified);

    public static AssetValidationResult Unverified(string path) => new(path, ValidationVerdict.Unverified);

    public static AssetValidationResult Quarantined(string path, IReadOnlyList<AssetDiagnostic> diagnostics)
        => new(path, ValidationVerdict.Quarantined, diagnostics);
}