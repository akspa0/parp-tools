using DBCD;
using DBCD.Providers;
using MdxViewer.Logging;

namespace MdxViewer.Rendering;

/// <summary>
/// Resolves Replaceable texture IDs to actual BLP paths using DBCD (direct DBC reading).
/// Loads ALL relevant DBC tables for complete texture resolution:
///   - CreatureModelData.dbc   → model path ↔ ModelID mapping
///   - CreatureDisplayInfo.dbc → ModelID → TextureVariation[3] (creature skins)
///   - CreatureDisplayInfoExtra.dbc → NPC baked textures + item display refs
///   - ItemDisplayInfo.dbc     → item model textures + armor region textures
///
/// Texture type (ReplaceableId) mapping:
///   1  = Creature Skin 1 / Body        → CDI TextureVariation[0]
///   2  = Object Skin / Item            → CDI TextureVariation[1] or ItemDisplayInfo
///   3  = Weapon Blade                  → CDI TextureVariation[2]
///   11 = Creature Skin 1 (explicit)    → CDI TextureVariation[0]
///   12 = Creature Skin 2 (explicit)    → CDI TextureVariation[1]
///   13 = Creature Skin 3 (explicit)    → CDI TextureVariation[2]
/// </summary>
public class ReplaceableTextureResolver
{
    // ModelID → list of TextureVariation arrays (one per DisplayInfo entry)
    private readonly Dictionary<int, List<string[]>> _displayVariations = new();
    // Model path (lowercase, backslash) → ModelID
    private readonly Dictionary<string, int> _modelPathToId = new();
    // ModelID → model path (reverse lookup for building texture paths)
    private readonly Dictionary<int, string> _modelIdToPath = new();
    // Filename (lowercase, no ext) → ModelID (fallback lookup)
    private readonly Dictionary<string, int> _modelFileNameToId = new();
    // ItemDisplayInfo ID → item texture data
    private readonly Dictionary<int, ItemDisplayData> _itemDisplayInfo = new();
    // CreatureDisplayInfoExtra ID → extra display data (bake texture, item refs)
    private readonly Dictionary<int, ExtraDisplayData> _extraDisplayInfo = new();
    // CDI ModelID → ExtraDisplayInfoID (for NPC texture baking)
    private readonly Dictionary<int, int> _modelToExtraDisplayId = new();

    private bool _loaded;

    private record ItemDisplayData(string[] ModelNames, string[] ModelTextures, string[] Textures);
    private record ExtraDisplayData(string BakeName, int[] ItemDisplayIds);

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
            TryLoadCreatureDisplayInfoExtra(dbcd, build);
            TryLoadItemDisplayInfo(dbcd, build);
            _loaded = true;

            ViewerLog.Important(ViewerLog.Category.Dbc, $"=== DBC Texture Resolution Summary (build {build}) ===");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CreatureModelData:          {_modelPathToId.Count} model paths");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CreatureDisplayInfo:        {_displayVariations.Values.Sum(v => v.Count)} display entries for {_displayVariations.Count} unique models");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CreatureDisplayInfoExtra:   {_extraDisplayInfo.Count} entries");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  ItemDisplayInfo:            {_itemDisplayInfo.Count} entries");

            // Log first few TextureVariation samples for debugging
            int sampleCount = 0;
            foreach (var kvp in _displayVariations)
            {
                if (sampleCount >= 5) break;
                var modelPath = _modelIdToPath.GetValueOrDefault(kvp.Key, "?");
                foreach (var vars in kvp.Value)
                {
                    var nonEmpty = vars.Where(s => !string.IsNullOrEmpty(s)).ToArray();
                    if (nonEmpty.Length > 0)
                    {
                        ViewerLog.Debug(ViewerLog.Category.Dbc, $"  Sample: ModelID={kvp.Key} ({Path.GetFileName(modelPath)}) -> [{string.Join(", ", nonEmpty)}]");
                        sampleCount++;
                        break;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Dbc, $"Failed to load DBCs: {ex.Message}");
        }
    }

    private void LoadCreatureModelData(DBCD.DBCD dbcd, string build)
    {
        var storage = LoadDbc(dbcd, "CreatureModelData", build);

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int id = key;

            string? modelName = TryGetString(row, "ModelName")
                             ?? TryGetString(row, "ModelPath")
                             ?? TryGetString(row, "FileDataID");

            if (string.IsNullOrEmpty(modelName)) continue;

            string normalized = modelName.ToLowerInvariant().Replace('/', '\\');
            _modelPathToId[normalized] = id;
            _modelIdToPath[id] = normalized;

            string fileNameKey = Path.GetFileNameWithoutExtension(normalized);
            _modelFileNameToId.TryAdd(fileNameKey, id);

            count++;
        }
        ViewerLog.Info(ViewerLog.Category.Dbc, $"CreatureModelData: {count} entries loaded");
    }

    private void LoadCreatureDisplayInfo(DBCD.DBCD dbcd, string build)
    {
        var storage = LoadDbc(dbcd, "CreatureDisplayInfo", build);

        int count = 0, withTextures = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];

            int modelId = TryGetInt(row, "ModelID") ?? TryGetInt(row, "ModelId") ?? 0;
            if (modelId == 0) continue;

            // Store ExtraDisplayInfoID for NPC texture baking
            int extraId = TryGetInt(row, "ExtendedDisplayInfoID")
                       ?? TryGetInt(row, "ExtraDisplayInfoID") ?? 0;
            if (extraId > 0)
                _modelToExtraDisplayId.TryAdd(modelId, extraId);

            // Read TextureVariation array — DBCD returns string[] for array fields
            var variations = ReadStringArray(row, "TextureVariation", 3);

            if (!_displayVariations.ContainsKey(modelId))
                _displayVariations[modelId] = new List<string[]>();
            _displayVariations[modelId].Add(variations);

            if (variations.Any(s => !string.IsNullOrEmpty(s)))
                withTextures++;
            count++;
        }
        ViewerLog.Info(ViewerLog.Category.Dbc, $"CreatureDisplayInfo: {count} entries, {withTextures} with textures");
    }

    private void TryLoadCreatureDisplayInfoExtra(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage? storage;
        try { storage = LoadDbc(dbcd, "CreatureDisplayInfoExtra", build); }
        catch { ViewerLog.Info(ViewerLog.Category.Dbc, "CreatureDisplayInfoExtra: not available"); return; }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int id = key;

            string bakeName = TryGetString(row, "BakeName") ?? "";

            // NPCItemDisplay is an array of ItemDisplayInfo IDs
            var itemIds = new List<int>();
            for (int i = 0; i < 10; i++)
            {
                int? itemId = TryGetInt(row, $"NPCItemDisplay[{i}]")
                           ?? TryGetInt(row, $"NPCItemDisplay_{i}");
                if (itemId.HasValue && itemId.Value > 0)
                    itemIds.Add(itemId.Value);
            }

            _extraDisplayInfo[id] = new ExtraDisplayData(bakeName, itemIds.ToArray());
            count++;
        }
        ViewerLog.Info(ViewerLog.Category.Dbc, $"CreatureDisplayInfoExtra: {count} entries loaded");
    }

    private void TryLoadItemDisplayInfo(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage? storage;
        try { storage = LoadDbc(dbcd, "ItemDisplayInfo", build); }
        catch { ViewerLog.Info(ViewerLog.Category.Dbc, "ItemDisplayInfo: not available"); return; }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];
            int id = key;

            // ModelName[2] — left/right model paths
            var modelNames = ReadStringArray(row, "ModelName", 2);
            // ModelTexture[2] — left/right model textures
            var modelTextures = ReadStringArray(row, "ModelTexture", 2);
            // Texture[8] — armor region textures
            var textures = ReadStringArray(row, "Texture", 8);

            _itemDisplayInfo[id] = new ItemDisplayData(modelNames, modelTextures, textures);
            count++;
        }
        ViewerLog.Info(ViewerLog.Category.Dbc, $"ItemDisplayInfo: {count} entries loaded");
    }

    /// <summary>
    /// Resolve a replaceable texture ID to a BLP path for the given model.
    /// </summary>
    public string? Resolve(string modelPath, uint replaceableId, int displayIndex = 0)
    {
        if (!_loaded)
        {
            ViewerLog.Debug(ViewerLog.Category.Dbc, $"Resolve called but not loaded! model={modelPath} replId={replaceableId}");
            return null;
        }

        int modelId = FindModelId(modelPath);
        if (modelId == 0)
        {
            ViewerLog.Debug(ViewerLog.Category.Dbc, $"No ModelID for: {modelPath}");
            return null;
        }

        // Try creature TextureVariation first (covers ReplaceableId 1-3 and 11-13)
        string? result = ResolveFromCreatureDisplay(modelId, modelPath, replaceableId, displayIndex);
        if (result != null) return result;

        // Try CreatureDisplayInfoExtra bake texture
        result = ResolveFromExtraDisplay(modelId, replaceableId);
        if (result != null) return result;

        // Try ItemDisplayInfo for NPC equipped items
        result = ResolveFromItemDisplay(modelId, replaceableId);
        if (result != null) return result;

        ViewerLog.Debug(ViewerLog.Category.Dbc, $"Unresolved: ModelID={modelId} replId={replaceableId} ({Path.GetFileName(modelPath)})");
        return null;
    }

    private string? ResolveFromCreatureDisplay(int modelId, string modelPath, uint replaceableId, int displayIndex)
    {
        if (!_displayVariations.TryGetValue(modelId, out var variants) || variants.Count == 0)
            return null;

        if (displayIndex >= variants.Count) displayIndex = 0;
        var texNames = variants[displayIndex];

        // Map replaceable ID to texture variation index
        int varIndex = replaceableId switch
        {
            1 => 0,  // Creature Skin 1 / Body
            2 => 1,  // Creature Skin 2 / Object Skin
            3 => 2,  // Creature Skin 3 / Weapon
            11 => 0, // Creature Skin 1 (explicit)
            12 => 1, // Creature Skin 2 (explicit)
            13 => 2, // Creature Skin 3 (explicit)
            _ => -1
        };

        if (varIndex < 0 || varIndex >= texNames.Length) return null;

        string texName = texNames[varIndex].Trim();
        if (string.IsNullOrEmpty(texName)) return null;

        return BuildTexturePath(texName, modelPath);
    }

    private string? ResolveFromExtraDisplay(int modelId, uint replaceableId)
    {
        if (!_modelToExtraDisplayId.TryGetValue(modelId, out int extraId) || extraId == 0)
            return null;

        if (!_extraDisplayInfo.TryGetValue(extraId, out var extra))
            return null;

        // BakeName is a pre-composited texture for the NPC (replaceableId 1 = body)
        if (replaceableId == 1 && !string.IsNullOrEmpty(extra.BakeName))
            return NormalizePath(extra.BakeName);

        return null;
    }

    private string? ResolveFromItemDisplay(int modelId, uint replaceableId)
    {
        // For NPC models, check if they have equipped items via CreatureDisplayInfoExtra
        if (!_modelToExtraDisplayId.TryGetValue(modelId, out int extraId) || extraId == 0)
            return null;

        if (!_extraDisplayInfo.TryGetValue(extraId, out var extra))
            return null;

        // Try each item display for texture
        foreach (int itemDisplayId in extra.ItemDisplayIds)
        {
            if (!_itemDisplayInfo.TryGetValue(itemDisplayId, out var item))
                continue;

            // ModelTexture[0] is primary texture for the item
            foreach (var tex in item.ModelTextures.Concat(item.Textures))
            {
                if (!string.IsNullOrEmpty(tex))
                    return NormalizePath(tex);
            }
        }

        return null;
    }

    /// <summary>
    /// Build a full BLP path from a texture name and model path.
    /// Handles multiple formats:
    ///   - Full relative path with extension: "Creature\Murloc\Murloc.blp" → use as-is
    ///   - Full relative path without extension: "Creature\Murloc\Murloc" → append .blp
    ///   - Bare filename: "MurlocOrange" → prepend model directory + append .blp
    /// </summary>
    private static string BuildTexturePath(string texName, string modelPath)
    {
        // If the texture name already contains a path separator, it's a full relative path
        if (texName.Contains('\\') || texName.Contains('/'))
        {
            string path = texName.Replace('/', '\\');
            if (!path.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                path += ".blp";
            return path;
        }

        // Bare filename — prepend model directory
        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "";
        string blpName = texName;
        if (!blpName.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
            blpName += ".blp";
        return string.IsNullOrEmpty(modelDir) ? blpName : Path.Combine(modelDir, blpName);
    }

    private static string NormalizePath(string path)
    {
        string result = path.Replace('/', '\\').Trim();
        if (!result.EndsWith(".blp", StringComparison.OrdinalIgnoreCase) &&
            !result.EndsWith(".tga", StringComparison.OrdinalIgnoreCase))
            result += ".blp";
        return result;
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

    private static IDBCDStorage LoadDbc(DBCD.DBCD dbcd, string name, string build)
    {
        try
        {
            return dbcd.Load(name, build, Locale.EnUS);
        }
        catch
        {
            return dbcd.Load(name, build, Locale.None);
        }
    }

    /// <summary>
    /// Read a string array field from a DBCD row.
    /// DBCD represents DBD array fields like TextureVariation[3] as indexable arrays.
    /// Tries multiple access patterns to handle different DBCD versions.
    /// </summary>
    private static string[] ReadStringArray(dynamic row, string fieldName, int expectedCount)
    {
        var result = new string[expectedCount];

        // Pattern 1: DBCD returns the array field directly as string[]
        try
        {
            var val = row[fieldName];
            if (val is string[] arr)
            {
                for (int i = 0; i < Math.Min(arr.Length, expectedCount); i++)
                    result[i] = arr[i] ?? "";
                return result;
            }
            // Could be object[] that needs casting
            if (val is object[] objArr)
            {
                for (int i = 0; i < Math.Min(objArr.Length, expectedCount); i++)
                    result[i] = objArr[i]?.ToString() ?? "";
                return result;
            }
            // Single string value (shouldn't happen for arrays but handle gracefully)
            if (val is string s && !s.StartsWith("System."))
            {
                result[0] = s;
                return result;
            }
        }
        catch { /* Field not accessible as array, try indexed access */ }

        // Pattern 2: Indexed access — FieldName[0], FieldName[1], etc.
        bool anyFound = false;
        for (int i = 0; i < expectedCount; i++)
        {
            string? v = TryGetString(row, $"{fieldName}[{i}]")
                     ?? TryGetString(row, $"{fieldName}_{i}");
            result[i] = v ?? "";
            if (!string.IsNullOrEmpty(v)) anyFound = true;
        }

        // Pattern 3: If nothing found, try pipe-separated single field
        if (!anyFound)
        {
            string? texVar = TryGetString(row, fieldName);
            if (!string.IsNullOrEmpty(texVar) && !texVar.StartsWith("System."))
            {
                var parts = texVar.Split('|', StringSplitOptions.None);
                for (int i = 0; i < Math.Min(parts.Length, expectedCount); i++)
                    result[i] = parts[i];
            }
        }

        return result;
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
