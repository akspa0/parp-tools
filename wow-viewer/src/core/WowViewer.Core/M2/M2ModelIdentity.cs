using System.Globalization;

namespace WowViewer.Core.M2;

public sealed class M2ModelIdentity
{
    public M2ModelIdentity(string requestedPath, string canonicalModelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(requestedPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(canonicalModelPath);

        RequestedPath = NormalizePath(requestedPath);
        CanonicalModelPath = NormalizePath(canonicalModelPath);
        RequestedExtension = Path.GetExtension(RequestedPath);
    }

    public string RequestedPath { get; }

    public string CanonicalModelPath { get; }

    public string RequestedExtension { get; }

    public bool WasCanonicalized => !PathsEqual(RequestedPath, CanonicalModelPath);

    public string BuildSkinPath(int profileIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(profileIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(profileIndex, 99);

        string extension = Path.GetExtension(CanonicalModelPath);
        string basePath = CanonicalModelPath[..^extension.Length];
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{basePath}{profileIndex:D2}.skin");
    }

    public static M2ModelIdentity FromPath(string requestedPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(requestedPath);

        string normalized = NormalizePath(requestedPath);
        string extension = Path.GetExtension(normalized);
        if (!extension.Equals(".m2", StringComparison.OrdinalIgnoreCase)
            && !extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase)
            && !extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"M2 identity requires a .m2, .mdx, or .mdl path. Found '{extension}'.",
                nameof(requestedPath));
        }

        string canonical = normalized[..^extension.Length] + ".m2";
        return new M2ModelIdentity(normalized, canonical);
    }

    public static bool PathsEqual(string left, string right)
    {
        return string.Equals(NormalizePath(left), NormalizePath(right), StringComparison.OrdinalIgnoreCase);
    }

    public static string NormalizePath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return path.Replace('/', '\\');
    }
}