using WowViewer.Core.IO.Files;

namespace WowViewer.App;

internal static class VirtualAssetOverlayResolver
{
    public static byte[] ReadVirtualFilePreferLoose(string virtualPath, string archiveRoot, string? looseOverlayRoot)
    {
        if (TryReadLooseVirtualFile(virtualPath, looseOverlayRoot, out byte[]? looseBytes) && looseBytes is not null)
            return looseBytes;

        return ArchiveVirtualFileReader.ReadVirtualFile(virtualPath, [archiveRoot], WowViewerArchiveBootstrap.CreateBootstrapOptions());
    }

    public static bool TryReadLooseVirtualFile(string virtualPath, string? looseOverlayRoot, out byte[]? bytes)
    {
        return TryReadLooseVirtualFile(virtualPath, looseOverlayRoot, out bytes, out _);
    }

    public static bool TryReadLooseVirtualFile(string virtualPath, string? looseOverlayRoot, out byte[]? bytes, out string sourcePath)
    {
        bytes = null;
        sourcePath = string.Empty;

        if (string.IsNullOrWhiteSpace(looseOverlayRoot))
            return false;

        string root = Path.GetFullPath(looseOverlayRoot);
        if (!Directory.Exists(root))
            return false;

        string normalizedVirtualPath = NormalizeVirtualPath(virtualPath);
        string candidate = Path.Combine(root, normalizedVirtualPath.Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(candidate))
            return false;

        bytes = File.ReadAllBytes(candidate);
        sourcePath = Path.GetFullPath(candidate);
        return bytes.Length > 0;
    }

    public static IEnumerable<string> EnumerateLooseVirtualFiles(string? looseOverlayRoot)
    {
        if (string.IsNullOrWhiteSpace(looseOverlayRoot))
            yield break;

        string root = Path.GetFullPath(looseOverlayRoot);
        if (!Directory.Exists(root))
            yield break;

        foreach (string file in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(root, file).Replace('\\', '/');
            if (!string.IsNullOrWhiteSpace(relative))
                yield return relative;
        }
    }

    private static string NormalizeVirtualPath(string path)
    {
        return path.Replace('\\', '/').TrimStart('/');
    }
}
