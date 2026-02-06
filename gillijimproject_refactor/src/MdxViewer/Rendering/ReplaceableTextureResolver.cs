using DBCD;
using DBCD.Providers;

namespace MdxViewer.Rendering;

/// <summary>
/// Resolves Replaceable texture IDs to actual BLP paths using DBCD (direct DBC reading).
/// Chain: model path → CreatureModelData.ID → CreatureDisplayInfo.ModelID → TextureVariation
/// Replaceable ID 1 = TextureVariation[0], ID 2 = TextureVariation[1], ID 11 = TextureVariation[0] (alias)
/// Supports any WoW version with appropriate DBD definitions and DBC files.
/// </summary>
public class ReplaceableTextureResolver
{
    // ModelID → list of TextureVariation arrays (one per DisplayInfo entry)
    private readonly Dictionary<int, List<string[]>> _displayVariations = new();
    // Model path (lowercase, backslash) → ModelID
    private readonly Dictionary<string, int> _modelPathToId = new();
    // Filename (lowercase, no ext) → ModelID (fallback lookup)
    private readonly Dictionary<string, int> _modelFileNameToId = new();

    private bool _loaded;

    /// <summary>Known build strings for version alias resolution.</summary>
    private static readonly Dictionary<string, string> BuildAliases = new()
    {
        { "0.5.3", "0.5.3.3368" },
        { "0.5.5", "0.5.5.3494" },
        { "0.6.0", "0.6.0.3592" },
        { "3.3.5", "3.3.5.12340" },
    };

    /// <summary>
    /// Load DBC tables directly using DBCD from an IDBCProvider (MPQ or filesystem).
    /// </summary>
    /// <param name="dbcProvider">Provider that reads DBC files (from MPQ or disk)</param>
    /// <param name="dbdDir">Path to WoWDBDefs/definitions directory</param>
    /// <param name="buildOrAlias">Build string or alias (e.g. "0.5.3" or "0.5.3.3368")</param>
    public void LoadFromDBC(IDBCProvider dbcProvider, string dbdDir, string buildOrAlias)
    {
        if (_loaded) return;

        string build = ResolveBuild(buildOrAlias);
        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        try
        {
            LoadCreatureModelData(dbcd, build);
            LoadCreatureDisplayInfo(dbcd, build);
            _loaded = true;
            Console.WriteLine($"[ReplaceableTexResolver] Loaded {_modelPathToId.Count} models, {_displayVariations.Count} display entries from DBC (build {build})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ReplaceableTexResolver] Failed to load DBCs: {ex.Message}");
        }
    }

    private void LoadCreatureModelData(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage storage;
        try
        {
            storage = dbcd.Load("CreatureModelData", build, Locale.EnUS);
        }
        catch
        {
            storage = dbcd.Load("CreatureModelData", build, Locale.None);
        }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int id = key;

            // Get ModelName field — column name varies by version
            string? modelName = TryGetString(row, "ModelName")
                             ?? TryGetString(row, "ModelPath")
                             ?? TryGetString(row, "FileDataID"); // newer versions use FileDataID

            if (string.IsNullOrEmpty(modelName)) continue;

            string normalized = modelName.ToLowerInvariant().Replace('/', '\\');
            _modelPathToId[normalized] = id;

            string fileNameKey = Path.GetFileNameWithoutExtension(normalized);
            _modelFileNameToId.TryAdd(fileNameKey, id);

            count++;
        }
        Console.WriteLine($"[ReplaceableTexResolver] CreatureModelData: {count} entries");
    }

    private void LoadCreatureDisplayInfo(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage storage;
        try
        {
            storage = dbcd.Load("CreatureDisplayInfo", build, Locale.EnUS);
        }
        catch
        {
            storage = dbcd.Load("CreatureDisplayInfo", build, Locale.None);
        }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];

            // ModelID field
            int modelId = TryGetInt(row, "ModelID")
                       ?? TryGetInt(row, "ModelId")
                       ?? 0;
            if (modelId == 0) continue;

            // TextureVariation — can be a single string with pipes, or an array field
            var variations = new List<string>();

            // Try array-style first (TextureVariation_0, TextureVariation_1, TextureVariation_2)
            for (int i = 0; i < 3; i++)
            {
                string? v = TryGetString(row, $"TextureVariation[{i}]")
                         ?? TryGetString(row, $"TextureVariation_{i}");
                variations.Add(v ?? "");
            }

            // If all empty, try single TextureVariation field (pipe-separated)
            if (variations.All(string.IsNullOrEmpty))
            {
                string? texVar = TryGetString(row, "TextureVariation");
                if (!string.IsNullOrEmpty(texVar))
                {
                    var parts = texVar.Split('|', StringSplitOptions.None);
                    variations.Clear();
                    variations.AddRange(parts);
                }
            }

            if (!_displayVariations.ContainsKey(modelId))
                _displayVariations[modelId] = new List<string[]>();
            _displayVariations[modelId].Add(variations.ToArray());
            count++;
        }
        Console.WriteLine($"[ReplaceableTexResolver] CreatureDisplayInfo: {count} entries");
    }

    /// <summary>
    /// Resolve a replaceable texture ID to a BLP path for the given model.
    /// </summary>
    public string? Resolve(string modelPath, uint replaceableId, int displayIndex = 0)
    {
        if (!_loaded) return null;

        int modelId = FindModelId(modelPath);
        if (modelId == 0) return null;

        if (!_displayVariations.TryGetValue(modelId, out var variants) || variants.Count == 0)
            return null;

        if (displayIndex >= variants.Count) displayIndex = 0;
        var texNames = variants[displayIndex];

        // Map replaceable ID to texture variation index
        int varIndex = replaceableId switch
        {
            1 => 0,
            2 => 1,
            3 => 2,
            11 => 0, // Creature Skin 1 alias
            12 => 1, // Creature Skin 2 alias
            13 => 2, // Creature Skin 3 alias
            _ => -1
        };

        if (varIndex < 0 || varIndex >= texNames.Length) return null;

        string texName = texNames[varIndex].Trim();
        if (string.IsNullOrEmpty(texName)) return null;

        // Build full path: model directory + texture name + .blp
        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        string blpPath = Path.Combine(modelDir, texName + ".blp");

        return blpPath;
    }

    /// <summary>Get all display variant count for a model.</summary>
    public int GetVariantCount(string modelPath)
    {
        if (!_loaded) return 0;
        int modelId = FindModelId(modelPath);
        if (modelId == 0 || !_displayVariations.TryGetValue(modelId, out var variants))
            return 0;
        return variants.Count;
    }

    /// <summary>Get display variant description.</summary>
    public string GetVariantDescription(string modelPath, int displayIndex)
    {
        if (!_loaded) return "";
        int modelId = FindModelId(modelPath);
        if (modelId == 0 || !_displayVariations.TryGetValue(modelId, out var variants))
            return "";
        if (displayIndex >= variants.Count) return "";
        return string.Join(", ", variants[displayIndex].Where(s => !string.IsNullOrEmpty(s)));
    }

    public bool IsLoaded => _loaded;

    // --- Helpers ---

    private int FindModelId(string modelPath)
    {
        string normalized = modelPath.ToLowerInvariant().Replace('/', '\\');
        if (_modelPathToId.TryGetValue(normalized, out int modelId))
            return modelId;

        // Fallback: match by filename only
        string fileName = Path.GetFileNameWithoutExtension(normalized);
        if (_modelFileNameToId.TryGetValue(fileName, out modelId))
            return modelId;

        return 0;
    }

    private static string ResolveBuild(string buildOrAlias)
    {
        if (BuildAliases.TryGetValue(buildOrAlias, out var canonical))
            return canonical;
        return buildOrAlias;
    }

    private static string? TryGetString(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is string s) return s;
            return val?.ToString();
        }
        catch { return null; }
    }

    private static int? TryGetInt(dynamic row, string fieldName)
    {
        try
        {
            var val = row[fieldName];
            if (val is int i) return i;
            if (val is uint u) return (int)u;
            if (val is short s) return s;
            if (val is ushort us) return us;
            if (int.TryParse(val?.ToString(), out int parsed)) return parsed;
            return null;
        }
        catch { return null; }
    }
}
