using System.Runtime.InteropServices;

namespace WowViewer.Core.IO.Archive;

public static class WoWArchiveCatalog
{
    private static readonly string StagingRoot = Path.Combine(
        AppContext.BaseDirectory,
        "..", "..", "..", "..", "..", "..",
        "output", "tmp", "wowarchive-clients");

    public static IReadOnlyList<WoWArchiveBuildEntry> Scan(string archiveRootPath)
    {
        if (!Directory.Exists(archiveRootPath))
            throw new DirectoryNotFoundException($"Archive root not found: {archiveRootPath}");

        string[] manifestFiles = Directory.GetFiles(archiveRootPath, "Clients_*.txt", SearchOption.TopDirectoryOnly);
        if (manifestFiles.Length == 0)
            return Array.Empty<WoWArchiveBuildEntry>();

        string latestManifest = manifestFiles.OrderByDescending(f => f).First();

        string resolvedStagingRoot = ResolveStagingRoot();
        string? mountRoot = ResolveMountPath(archiveRootPath);

        var entries = new List<WoWArchiveBuildEntry>();
        var seenVersions = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (string line in File.ReadLines(latestManifest))
        {
            string trimmed = line.Trim();
            if (string.IsNullOrWhiteSpace(trimmed))
                continue;

            if (!TryParseManifestLine(trimmed, out string? buildVersion, out string? platform,
                    out string? locale, out string? eraTag))
                continue;

            string normalizedVersion = buildVersion.StripAltSuffix();
            if (!seenVersions.Add(normalizedVersion))
                continue;

            WoWArchiveBuildStatus status = ResolveStatus(mountRoot, resolvedStagingRoot, trimmed, buildVersion);

            entries.Add(new WoWArchiveBuildEntry(
                BuildVersion: buildVersion,
                Platform: platform,
                Locale: locale,
                Era: WoWArchiveEra.Classify(eraTag),
                InnerPath: trimmed,
                Status: status));
        }

        entries.Sort((a, b) =>
        {
            int eraCompare = string.Compare(a.Era, b.Era, StringComparison.Ordinal);
            if (eraCompare != 0)
                return eraCompare;
            return string.Compare(a.BuildVersion, b.BuildVersion, StringComparison.Ordinal);
        });

        return entries;
    }

    private static bool TryParseManifestLine(string line, out string? buildVersion, out string? platform,
        out string? locale, out string? eraTag)
    {
        buildVersion = null;
        platform = null;
        locale = null;
        eraTag = null;

        string[] parts = line.Split('_');
        if (parts.Length < 5)
            return false;

        eraTag = parts[0];
        buildVersion = parts[^1];
        locale = parts[^2];
        platform = parts[^3];

        return true;
    }

    private static WoWArchiveBuildStatus ResolveStatus(string? mountRoot, string stagingRoot, string line, string buildVersion)
    {
        string stagedBuildDir = buildVersion.Replace('.', '_');
        string stagedPath = Path.Combine(stagingRoot, stagedBuildDir, "World of Warcraft");
        if (Directory.Exists(stagedPath))
            return WoWArchiveBuildStatus.Staged;

        if (mountRoot != null)
        {
            string mountPath = Path.Combine(mountRoot, line, "World of Warcraft");
            if (Directory.Exists(mountPath))
                return WoWArchiveBuildStatus.MountLive;
        }

        return WoWArchiveBuildStatus.Available;
    }

    private static string? ResolveMountPath(string archiveRootPath)
    {
        string mountPath = Path.Combine(archiveRootPath, "Mount");
        if (Directory.Exists(mountPath))
            return mountPath;
        return null;
    }

    private static string ResolveStagingRoot()
    {
        string cwd = AppContext.BaseDirectory;

        for (int i = 0; i < 12 && cwd != null; i++)
        {
            string candidate = Path.Combine(cwd, "output", "tmp", "wowarchive-clients");
            if (Directory.Exists(candidate))
                return candidate;

            string parent = Path.GetDirectoryName(cwd);
            if (parent == null || parent == cwd)
                break;
            cwd = parent;
        }

        return Path.Combine(AppContext.BaseDirectory, "output", "tmp", "wowarchive-clients");
    }
}

file static class BuildVersionExtensions
{
    public static string StripAltSuffix(this string buildVersion)
    {
        int altIndex = buildVersion.IndexOf("-alt", StringComparison.OrdinalIgnoreCase);
        return altIndex > 0 ? buildVersion[..altIndex] : buildVersion;
    }
}
