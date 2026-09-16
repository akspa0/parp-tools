namespace WowViewer.Core.IO.Casc;

/// <summary>
/// Spec 238: bidirectional FileDataID ↔ path map loaded from community listfile CSVs
/// (<c>id;path</c> lines). Path lookups are case-insensitive and treat '/' and '\' alike.
/// </summary>
public sealed class CommunityListfile
{
    private readonly Dictionary<string, uint> _idByPath = new(StringComparer.Ordinal);
    private readonly Dictionary<uint, string> _pathById = [];

    public int Count => _pathById.Count;

    public static CommunityListfile Load(IEnumerable<string> csvPaths)
    {
        var listfile = new CommunityListfile();
        foreach (string csvPath in csvPaths)
        {
            foreach (string line in File.ReadLines(csvPath))
            {
                int separator = line.IndexOf(';');
                if (separator <= 0 || !uint.TryParse(line.AsSpan(0, separator), out uint id))
                    continue;

                string path = line[(separator + 1)..].Trim();
                if (path.Length == 0)
                    continue;

                listfile._pathById[id] = path;
                listfile._idByPath[Normalize(path)] = id;
            }
        }

        return listfile;
    }

    public bool TryGetFileDataId(string path, out uint fileDataId) =>
        _idByPath.TryGetValue(Normalize(path), out fileDataId);

    public string? GetPath(uint fileDataId) =>
        _pathById.TryGetValue(fileDataId, out string? path) ? path : null;

    public IEnumerable<KeyValuePair<uint, string>> Entries => _pathById;

    private static string Normalize(string path) => path.Replace('/', '\\').TrimStart('\\').ToLowerInvariant();
}
