using WowViewer.Core.IO.Files;

namespace WowViewer.App;

internal static class WowViewerArchiveBootstrap
{
    public static ArchiveCatalogBootstrapOptions CreateBootstrapOptions()
    {
        return new ArchiveCatalogBootstrapOptions(ExternalListfilePath: ResolveDefaultListfilePath());
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
}