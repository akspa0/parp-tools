namespace WowViewer.Core.Files;

public static class AssetPathTaxonomy
{
    private static readonly string[] TreeKeywords = ["tree", "bush", "shrub", "fern", "plant", "palm", "oak", "pine", "birch"];
    private static readonly string[] RockKeywords = ["rock", "stone", "boulder", "pebble", "cliff", "crag"];
    private static readonly string[] BuildingKeywords = ["building", "house", "hut", "tower", "barrack", "inn", "shop"];
    private static readonly string[] StructureKeywords = ["bridge", "wall", "fence", "gate", "arch", "pillar", "platform"];
    private static readonly string[] DetailKeywords = ["flower", "grass", "mushroom", "twig", "root", "leaf", "stick"];

    public static string Normalize(string? assetPath)
    {
        if (string.IsNullOrWhiteSpace(assetPath))
            return string.Empty;

        return assetPath.Trim().Replace('/', '\\').Trim('\\');
    }

    public static string ClassifyObjectType(string? assetPath)
    {
        string normalizedPath = Normalize(assetPath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return "other";

        string path = normalizedPath.ToLowerInvariant();
        if (ContainsAny(path, TreeKeywords))
            return "tree";
        if (ContainsAny(path, RockKeywords))
            return "rock";
        if (ContainsAny(path, BuildingKeywords))
            return "building";
        if (ContainsAny(path, StructureKeywords))
            return "structure";
        if (ContainsAny(path, DetailKeywords))
            return "detail";
        if (path.Contains(".wmo", StringComparison.Ordinal))
            return "wmo";

        return "other";
    }

    public static AssetPathDescriptor Describe(string? assetPath)
    {
        string normalizedPath = Normalize(assetPath);
        if (string.IsNullOrWhiteSpace(normalizedPath))
            return AssetPathDescriptor.Empty;

        string[] pathSegments = normalizedPath.Split('\\', StringSplitOptions.RemoveEmptyEntries);
        string fileName = pathSegments.Length == 0 ? normalizedPath : pathSegments[^1];
        string[] directorySegments = pathSegments.Length > 1 ? pathSegments[..^1] : [];
        string directoryPath = directorySegments.Length == 0 ? string.Empty : string.Join('\\', directorySegments);
        string extension = Path.GetExtension(fileName);
        string assetKind = GetAssetKind(normalizedPath, extension);
        string objectType = ClassifyObjectType(normalizedPath);
        string rootSegment = directorySegments.Length == 0 ? string.Empty : directorySegments[0];
        string leafCategory = directorySegments.Length == 0 ? string.Empty : directorySegments[^1];
        string categoryKey = string.IsNullOrWhiteSpace(directoryPath)
            ? (!string.IsNullOrWhiteSpace(assetKind) ? assetKind : fileName)
            : directoryPath;
        string hierarchyLabel = directorySegments.Length == 0 ? string.Empty : string.Join(" > ", directorySegments);

        return new AssetPathDescriptor(
            normalizedPath,
            fileName,
            Path.GetFileNameWithoutExtension(fileName),
            extension,
            assetKind,
            objectType,
            rootSegment,
            leafCategory,
            directoryPath,
            categoryKey,
            hierarchyLabel,
            directorySegments);
    }

    private static string GetAssetKind(string normalizedPath, string extension)
    {
        if (normalizedPath.EndsWith(".wmo.mpq", StringComparison.OrdinalIgnoreCase)
            || normalizedPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
        {
            return "WMO";
        }

        if (extension.Equals(".m2", StringComparison.OrdinalIgnoreCase))
            return "M2";
        if (extension.Equals(".mdx", StringComparison.OrdinalIgnoreCase))
            return "MDX";
        if (extension.Equals(".mdl", StringComparison.OrdinalIgnoreCase))
            return "MDL";

        return string.IsNullOrWhiteSpace(extension) ? string.Empty : extension.TrimStart('.').ToUpperInvariant();
    }

    private static bool ContainsAny(string path, string[] keywords)
    {
        foreach (string keyword in keywords)
        {
            if (path.Contains(keyword, StringComparison.Ordinal))
                return true;
        }

        return false;
    }
}

public sealed record AssetPathDescriptor(
    string NormalizedPath,
    string FileName,
    string FileStem,
    string Extension,
    string AssetKind,
    string ObjectType,
    string RootSegment,
    string LeafCategory,
    string DirectoryPath,
    string CategoryKey,
    string HierarchyLabel,
    IReadOnlyList<string> DirectorySegments)
{
    public static AssetPathDescriptor Empty { get; } = new(
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        "other",
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        string.Empty,
        Array.Empty<string>());
}