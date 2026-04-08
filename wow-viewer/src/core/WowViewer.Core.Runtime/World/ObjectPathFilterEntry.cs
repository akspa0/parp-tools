namespace WowViewer.Core.Runtime.World;

public readonly record struct ObjectPathFilterEntry(
    string PathPrefix,
    bool AppliesToWmo,
    bool AppliesToMdx)
{
    public static string NormalizePrefix(string pathPrefix)
    {
        if (string.IsNullOrWhiteSpace(pathPrefix))
            return string.Empty;

        return pathPrefix.Trim().Replace('/', '\\').Trim('\\');
    }

    public bool MatchesModelPath(string modelPath)
    {
        string normalizedModelPath = NormalizePrefix(modelPath);
        if (string.IsNullOrWhiteSpace(normalizedModelPath))
            return false;

        bool isWmo = normalizedModelPath.EndsWith(".wmo", System.StringComparison.OrdinalIgnoreCase);
        if (isWmo)
        {
            if (!AppliesToWmo)
                return false;
        }
        else if (!AppliesToMdx)
        {
            return false;
        }

        if (string.Equals(normalizedModelPath, PathPrefix, System.StringComparison.OrdinalIgnoreCase))
            return true;

        if (!normalizedModelPath.StartsWith(PathPrefix, System.StringComparison.OrdinalIgnoreCase))
            return false;

        return normalizedModelPath.Length > PathPrefix.Length && normalizedModelPath[PathPrefix.Length] == '\\';
    }
}