using WowViewer.Core.IO.Files;
using System.Text.RegularExpressions;

namespace WowViewer.App;

internal static class WowViewerArchiveBootstrap
{
    public static ArchiveCatalogBootstrapOptions CreateBootstrapOptions(string? buildLabel = null, string? clientRoot = null)
    {
        string? cacheKey = ResolveArchiveListfileCacheKey(buildLabel, clientRoot);
        string? cacheDirectory = string.IsNullOrWhiteSpace(cacheKey)
            ? null
            : ResolveDefaultArchiveListfileCacheDirectory();

        return new ArchiveCatalogBootstrapOptions(
            ExternalListfilePath: ResolveDefaultListfilePath(),
            ListfileCacheKey: cacheKey,
            ListfileCacheDirectoryPath: cacheDirectory);
    }

    public static string? ResolveDefaultListfilePath()
    {
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string[] localCandidates =
        [
            Path.Combine(localAppData, "MdxViewer", "community-listfile-withcapitals.csv"),
            Path.Combine(AppContext.BaseDirectory, "community-listfile-withcapitals.csv"),
            Path.Combine(AppContext.BaseDirectory, "listfile.csv"),
            "community-listfile-withcapitals.csv",
            "listfile.csv",
        ];

        foreach (string candidate in localCandidates)
        {
            if (File.Exists(candidate))
                return Path.GetFullPath(candidate);
        }

        DirectoryInfo? current = new(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")) ||
                File.Exists(Path.Combine(current.FullName, "README.md")))
            {
                string[] rootedCandidates =
                [
                    Path.Combine(current.FullName, "libs", "wowdev", "wow-listfile", "listfile.txt"),
                    Path.Combine(current.FullName, "gillijimproject_refactor", "test_data", "community-listfile-withcapitals.csv"),
                    Path.Combine(current.FullName, "test_data", "community-listfile-withcapitals.csv"),
                ];

                foreach (string candidate in rootedCandidates)
                {
                    if (File.Exists(candidate))
                        return candidate;
                }
            }

            current = current.Parent;
        }

        return null;
    }

    public static string? ResolveDefaultArchiveListfileCacheDirectory()
    {
        DirectoryInfo? current = new(AppContext.BaseDirectory);
        for (int depth = 0; depth < 8 && current is not null; depth++, current = current.Parent)
        {
            if (File.Exists(Path.Combine(current.FullName, "WowViewer.slnx")))
                return Path.Combine(current.FullName, "output", "cache", "archive-listfiles");
        }

        return null;
    }

    public static string? ResolveArchiveListfileCacheKey(string? buildLabel, string? clientRoot)
    {
        string normalizedBuildLabel = buildLabel?.Trim() ?? string.Empty;
        if (!string.IsNullOrWhiteSpace(normalizedBuildLabel))
            return normalizedBuildLabel;

        if (string.IsNullOrWhiteSpace(clientRoot))
            return null;

        string[] segments = Path.GetFullPath(clientRoot)
            .Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            .Where(static segment => !string.IsNullOrWhiteSpace(segment))
            .ToArray();

        foreach (string segment in segments.Reverse())
        {
            if (TryNormalizeBuildToken(segment, out string? token))
                return token;
        }

        return null;
    }

    private static bool TryNormalizeBuildToken(string value, out string? token)
    {
        token = null;
        if (string.IsNullOrWhiteSpace(value))
            return false;

        Match match = Regex.Match(value, "(?<!\\d)(\\d+)[_.](\\d+)[_.](\\d+)[_.](\\d+)(?!\\d)");
        if (!match.Success)
            return false;

        token = string.Join('.', match.Groups.Cast<Group>().Skip(1).Select(static group => group.Value));
        return true;
    }
}