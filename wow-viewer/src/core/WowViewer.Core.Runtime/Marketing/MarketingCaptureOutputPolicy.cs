namespace WowViewer.Core.Runtime.Marketing;

/// <summary>A contained artifact expressed as both a managed-root-relative and resolved path.</summary>
public sealed record ManagedMarketingArtifact(string RelativePath, string FullPath);

/// <summary>
/// Enforces the marketing-capture boundary: evidence and handoff artifact locations remain under
/// the viewer-managed output root, and external descriptors never carry machine-local paths.
/// </summary>
public static class MarketingCaptureOutputPolicy
{
    public static bool TryResolveManagedArtifact(
        string? managedOutputRoot,
        string? artifactPath,
        out ManagedMarketingArtifact artifact,
        out string errorCode)
    {
        artifact = default!;
        errorCode = string.Empty;
        if (string.IsNullOrWhiteSpace(managedOutputRoot))
        {
            errorCode = "managed-root-missing";
            return false;
        }

        if (string.IsNullOrWhiteSpace(artifactPath) || artifactPath.IndexOf('\0') >= 0)
        {
            errorCode = "artifact-path-invalid";
            return false;
        }

        try
        {
            string fullRoot = Path.GetFullPath(managedOutputRoot);
            string fullArtifact = Path.GetFullPath(
                Path.IsPathRooted(artifactPath)
                    ? artifactPath
                    : Path.Combine(fullRoot, artifactPath));
            string relativePath = Path.GetRelativePath(fullRoot, fullArtifact);
            if (IsOutsideRoot(relativePath))
            {
                errorCode = "artifact-outside-managed-root";
                return false;
            }

            if (string.Equals(relativePath, ".", StringComparison.Ordinal))
            {
                errorCode = "artifact-path-invalid";
                return false;
            }

            artifact = new ManagedMarketingArtifact(relativePath, fullArtifact);
            return true;
        }
        catch (Exception exception) when (exception is ArgumentException or NotSupportedException or PathTooLongException)
        {
            errorCode = "artifact-path-invalid";
            return false;
        }
    }

    private static bool IsOutsideRoot(string relativePath)
        => string.Equals(relativePath, "..", StringComparison.Ordinal)
           || relativePath.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
           || relativePath.StartsWith($"..{Path.AltDirectorySeparatorChar}", StringComparison.Ordinal)
           || Path.IsPathRooted(relativePath);
}
