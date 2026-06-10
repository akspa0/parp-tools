using System.Globalization;

namespace WowViewer.Core.Anim;

public static class PathNormalizer
{
    private const string StaleClientsRoot = @"H:\CLIENTS";

    public static string NormalizeForOutput(string absolutePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(absolutePath);

        string forward = absolutePath.Replace('\\', '/');
        return forward.ToLowerInvariant();
    }

    public static void AssertNoStalePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        string normalized = path.Replace('/', '\\');
        if (normalized.Contains(StaleClientsRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                string.Format(
                    CultureInfo.InvariantCulture,
                    "Refusing to use the stale 'H:\\CLIENTS' path '{0}'. Use the staged client root under 'output/tmp/wowarchive-clients/' instead.",
                    path));
        }
    }

    public static string NormalizeAndAssert(string absolutePath)
    {
        AssertNoStalePath(absolutePath);
        return NormalizeForOutput(absolutePath);
    }
}

