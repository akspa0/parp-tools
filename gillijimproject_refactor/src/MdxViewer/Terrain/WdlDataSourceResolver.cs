using MdxViewer.DataSources;

namespace MdxViewer.Terrain;

internal static class WdlDataSourceResolver
{
    public static bool HasWdl(IDataSource dataSource, string mapDirectory)
    {
        foreach (string candidate in EnumerateCandidates(mapDirectory))
        {
            if (dataSource.FileExists(candidate))
                return true;

            if (dataSource is not MpqDataSource mpqDataSource)
                continue;

            if (!string.IsNullOrWhiteSpace(mpqDataSource.FindInFileSet(candidate)))
                return true;
        }

        return false;
    }

    public static bool TryReadWdlBytes(IDataSource dataSource, string mapDirectory, out byte[]? wdlBytes, out string? resolvedPath)
    {
        wdlBytes = null;
        resolvedPath = null;

        foreach (string candidate in EnumerateCandidates(mapDirectory))
        {
            wdlBytes = dataSource.ReadFile(candidate);
            if (wdlBytes != null && wdlBytes.Length > 0)
            {
                resolvedPath = candidate;
                return true;
            }

            if (dataSource is not MpqDataSource mpqDataSource)
                continue;

            string? found = mpqDataSource.FindInFileSet(candidate);
            if (string.IsNullOrWhiteSpace(found))
                continue;

            wdlBytes = dataSource.ReadFile(found);
            if (wdlBytes != null && wdlBytes.Length > 0)
            {
                resolvedPath = found;
                return true;
            }
        }

        return false;
    }

    private static IEnumerable<string> EnumerateCandidates(string mapDirectory)
    {
        string basePath = $"World\\Maps\\{mapDirectory}\\{mapDirectory}.wdl";
        yield return basePath;
        yield return basePath + ".mpq";
    }
}