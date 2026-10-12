namespace WoWViewer.Rendering;

/// <summary>
/// Spec 256 P1: answers <see cref="WarcraftNetM2Adapter.FindSkinInFileList"/> without scanning the whole
/// <c>.skin</c> list per model. A file can only score above zero when its base name starts with the model
/// name or its directory equals the model's directory, so the index keeps both groupings and scores just
/// those candidates — with the same scoring function, in the original list order — giving the same result.
/// </summary>
/// <remarks>
/// The index is built only once the same list instance is seen a second time; until then, and whenever a
/// query cannot be narrowed safely (short or non-ASCII model-name prefix), the original scan runs. Data
/// sources that return a new list per call therefore keep the original cost, never more.
/// </remarks>
internal sealed class SkinPathIndex
{
    // Two key lengths: model names of 8+ characters use the long key (small buckets); 3–7 use the short one.
    private const int ShortPrefix = 3;
    private const int LongPrefix = 8;

    private IReadOnlyList<string>? _seenList;
    private int _seenCount;
    private IReadOnlyList<string>? _indexedList;
    private Dictionary<string, List<int>>? _byShortPrefix;
    private Dictionary<string, List<int>>? _byLongPrefix;
    private Dictionary<string, List<int>>? _byDirectory;
    private List<int>? _nonAsciiShortPrefix;
    private List<int>? _nonAsciiLongPrefix;
    private readonly List<int> _candidates = new();

    public string? FindBestSkin(string modelPath, IReadOnlyList<string> files)
    {
        if (files.Count == 0)
            return null;

        if (!ReferenceEquals(files, _indexedList) || files.Count != _seenCount)
        {
            if (!ReferenceEquals(files, _seenList) || files.Count != _seenCount)
            {
                _seenList = files;
                _seenCount = files.Count;
                _indexedList = null;
                return WarcraftNetM2Adapter.FindSkinInFileList(modelPath, files);
            }

            Build(files);
        }

        (string modelName, string modelDir) = WarcraftNetM2Adapter.GetSkinQueryKeys(modelPath);
        bool useLong = modelName.Length >= LongPrefix;
        if (!TryGetPrefixKey(modelName, useLong ? LongPrefix : ShortPrefix, out string? prefixKey))
            return WarcraftNetM2Adapter.FindSkinInFileList(modelPath, files);

        _candidates.Clear();
        Dictionary<string, List<int>> buckets = useLong ? _byLongPrefix! : _byShortPrefix!;
        if (buckets.TryGetValue(prefixKey!, out List<int>? prefixed))
            _candidates.AddRange(prefixed);
        _candidates.AddRange(useLong ? _nonAsciiLongPrefix! : _nonAsciiShortPrefix!);
        if (!string.IsNullOrEmpty(modelDir) && _byDirectory!.TryGetValue(modelDir, out List<int>? sameDirectory))
            _candidates.AddRange(sameDirectory);

        _candidates.Sort();
        string? bestPath = null;
        int bestScore = 0;
        int previous = -1;
        foreach (int index in _candidates)
        {
            if (index == previous)
                continue;

            previous = index;
            int score = WarcraftNetM2Adapter.ScoreSkinCandidate(files[index], modelName, modelDir, out string normalized);
            if (score > bestScore)
            {
                bestScore = score;
                bestPath = normalized;
            }
        }

        return bestPath;
    }

    private void Build(IReadOnlyList<string> files)
    {
        var byShort = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var byLong = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var byDirectory = new Dictionary<string, List<int>>(StringComparer.OrdinalIgnoreCase);
        var nonAsciiShort = new List<int>();
        var nonAsciiLong = new List<int>();
        for (int index = 0; index < files.Count; index++)
        {
            string file = files[index];
            if (!file.EndsWith(".skin", StringComparison.OrdinalIgnoreCase))
                continue;

            string normalized = file.Replace('/', '\\');
            string fileBase = Path.GetFileNameWithoutExtension(normalized).ToLowerInvariant();
            string fileDir = (Path.GetDirectoryName(normalized) ?? string.Empty).ToLowerInvariant();

            AddTo(byDirectory, fileDir, index);

            // A base name shorter than a key cannot start with a model name long enough to use that key.
            if (fileBase.Length >= ShortPrefix)
            {
                if (TryGetPrefixKey(fileBase, ShortPrefix, out string? shortKey))
                    AddTo(byShort, shortKey!, index);
                else
                    nonAsciiShort.Add(index);
            }

            if (fileBase.Length >= LongPrefix)
            {
                if (TryGetPrefixKey(fileBase, LongPrefix, out string? longKey))
                    AddTo(byLong, longKey!, index);
                else
                    nonAsciiLong.Add(index);
            }
        }

        _byShortPrefix = byShort;
        _byLongPrefix = byLong;
        _byDirectory = byDirectory;
        _nonAsciiShortPrefix = nonAsciiShort;
        _nonAsciiLongPrefix = nonAsciiLong;
        _indexedList = files;
    }

    // Upper-cased ASCII prefix. For ASCII, OrdinalIgnoreCase equality is exactly equality of the
    // upper-cased characters, so a StartsWith(OrdinalIgnoreCase) match always shares this key.
    private static bool TryGetPrefixKey(string value, int length, out string? key)
    {
        key = null;
        if (value.Length < length)
            return false;

        Span<char> buffer = stackalloc char[length];
        for (int i = 0; i < length; i++)
        {
            char c = value[i];
            if (c > 0x7F)
                return false;
            buffer[i] = char.ToUpperInvariant(c);
        }

        key = new string(buffer);
        return true;
    }

    private static void AddTo(Dictionary<string, List<int>> map, string key, int index)
    {
        if (!map.TryGetValue(key, out List<int>? list))
        {
            list = new List<int>();
            map.Add(key, list);
        }

        list.Add(index);
    }
}
