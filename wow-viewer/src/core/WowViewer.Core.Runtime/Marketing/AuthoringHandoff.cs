namespace WowViewer.Core.Runtime.Marketing;

/// <summary>Terminal state carried through receipts and external authoring descriptors.</summary>
public enum MarketingCaptureTerminalOutcome
{
    Completed = 0,
    Degraded = 1,
    Aborted = 2,
    Rejected = 3,
    Failed = 4,
}

/// <summary>Metadata suitable for an external authoring tool and deliberately free of client paths.</summary>
public sealed record AuthoringHandoffProvenance(
    string RecipeId,
    string RecipeVersion,
    string MapName,
    string BuildVersion,
    MarketingCaptureTerminalOutcome TerminalOutcome);

/// <summary>Untrusted absolute/relative artifact inputs supplied by the viewer's receipt layer.</summary>
public sealed record AuthoringHandoffRequest(
    string AttemptId,
    string CapturePath,
    string ReceiptPath,
    IReadOnlyList<string> StillPaths,
    AuthoringHandoffProvenance Provenance);

/// <summary>Versioned, portable descriptor for a future MCP/ComfyUI authoring adapter.</summary>
public sealed record AuthoringHandoff(
    string SchemaVersion,
    string AttemptId,
    string CaptureRelativePath,
    string ReceiptRelativePath,
    IReadOnlyList<string> StillRelativePaths,
    AuthoringHandoffProvenance Provenance)
{
    public const string SchemaV1 = "wow-viewer.authoring-handoff.v1";
}

/// <summary>Typed validation outcome; no failed input mutates local artifacts.</summary>
public sealed record AuthoringHandoffValidationResult(
    bool IsValid,
    AuthoringHandoff? Handoff,
    string? ErrorCode,
    string? Error)
{
    public static AuthoringHandoffValidationResult Failure(string code, string error)
        => new(false, null, code, error);
}

/// <summary>Builds a safe descriptor without contacting external tools or services.</summary>
public static class AuthoringHandoffFactory
{
    public static AuthoringHandoffValidationResult TryCreate(string managedOutputRoot, AuthoringHandoffRequest? request)
    {
        if (request is null)
            return AuthoringHandoffValidationResult.Failure("handoff-request-missing", "An authoring handoff request is required.");
        if (string.IsNullOrWhiteSpace(request.AttemptId))
            return AuthoringHandoffValidationResult.Failure("attempt-id-missing", "A handoff requires an attempt id.");
        if (request.Provenance is null)
            return AuthoringHandoffValidationResult.Failure("handoff-provenance-missing", "A handoff requires capture provenance.");
        if (request.Provenance.TerminalOutcome is not (MarketingCaptureTerminalOutcome.Completed or MarketingCaptureTerminalOutcome.Degraded))
        {
            return AuthoringHandoffValidationResult.Failure(
                "terminal-outcome-not-handoff-eligible",
                "Only completed or degraded captures may be handed to external authoring.");
        }

        if (!MarketingCaptureOutputPolicy.TryResolveManagedArtifact(managedOutputRoot, request.CapturePath, out ManagedMarketingArtifact capture, out string captureError))
            return AuthoringHandoffValidationResult.Failure(MapArtifactError("capture", captureError), "Capture artifact is not inside the managed output root.");
        if (!MarketingCaptureOutputPolicy.TryResolveManagedArtifact(managedOutputRoot, request.ReceiptPath, out ManagedMarketingArtifact receipt, out string receiptError))
            return AuthoringHandoffValidationResult.Failure(MapArtifactError("receipt", receiptError), "Receipt artifact is not inside the managed output root.");

        var stillRelativePaths = new List<string>(request.StillPaths?.Count ?? 0);
        foreach (string? stillPath in request.StillPaths ?? Array.Empty<string>())
        {
            if (!MarketingCaptureOutputPolicy.TryResolveManagedArtifact(managedOutputRoot, stillPath, out ManagedMarketingArtifact still, out string stillError))
                return AuthoringHandoffValidationResult.Failure(MapArtifactError("still", stillError), "Still artifact is not inside the managed output root.");
            stillRelativePaths.Add(still.RelativePath);
        }

        return new AuthoringHandoffValidationResult(
            true,
            new AuthoringHandoff(
                AuthoringHandoff.SchemaV1,
                request.AttemptId,
                capture.RelativePath,
                receipt.RelativePath,
                stillRelativePaths,
                request.Provenance),
            null,
            null);
    }

    private static string MapArtifactError(string artifactName, string policyError)
        => string.Equals(policyError, "artifact-outside-managed-root", StringComparison.Ordinal)
            ? $"{artifactName}-outside-managed-root"
            : $"{artifactName}-path-invalid";
}
