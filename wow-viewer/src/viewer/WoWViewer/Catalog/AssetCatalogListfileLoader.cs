using System.Text.RegularExpressions;
using WoWViewer.DataSources;

namespace WoWViewer.Catalog;

/// <summary>
/// Listfile-based object source for the asset catalog. Where
/// <see cref="AlphaCoreDbReader"/> enumerates NPCs/GameObjects from the
/// alpha-core SQL, this enumerates every raw world model (M2/MDX/WMO) present
/// in the loaded client listfile, so the catalog can browse/export the whole
/// object set -- not just the SQL-referenced ones.
///
/// WMO group files (<c>*_NNN.wmo</c>) are excluded; only root WMO files are
/// browseable objects.
/// </summary>
public static class AssetCatalogListfileLoader
{
    private static readonly string[] ModelExtensions = { ".m2", ".mdx", ".wmo" };

    // WMO group files are the per-group geometry of a root WMO (e.g.
    // Ironforge_000.wmo); they are not standalone objects and must not be
    // enumerated as catalog entries.
    private static readonly Regex WmoGroupSuffix = new(@"_\d{3}\.wmo$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>
    /// Collect the distinct, sorted root-model virtual paths (M2/MDX/WMO) from
    /// the data source's listfile.
    /// </summary>
    public static IReadOnlyList<string> CollectModelPaths(IDataSource dataSource)
    {
        ArgumentNullException.ThrowIfNull(dataSource);

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var paths = new List<string>();

        foreach (string ext in ModelExtensions)
        {
            foreach (string raw in dataSource.GetFileList(ext))
            {
                if (string.IsNullOrWhiteSpace(raw))
                    continue;
                string normalized = raw.Replace('/', '\\');
                if (ext == ".wmo" && WmoGroupSuffix.IsMatch(normalized))
                    continue;
                if (seen.Add(normalized))
                    paths.Add(normalized);
            }
        }

        paths.Sort(StringComparer.OrdinalIgnoreCase);
        return paths;
    }

    /// <summary>
    /// Build catalog entries for every listfile world model, so they browse,
    /// filter, and export through the same path as SQL entries.
    /// </summary>
    public static List<AssetCatalogEntry> LoadModelEntries(IDataSource dataSource)
    {
        IReadOnlyList<string> paths = CollectModelPaths(dataSource);
        var entries = new List<AssetCatalogEntry>(paths.Count);
        int entryId = 1;
        foreach (string path in paths)
        {
            string name = Path.GetFileNameWithoutExtension(path);
            entries.Add(new AssetCatalogEntry
            {
                EntryId = entryId++,
                Type = AssetType.WorldModel,
                Name = name,
                ModelPath = path,
            });
        }
        return entries;
    }
}
