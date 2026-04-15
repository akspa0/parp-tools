using DBCD;
using DBCD.Providers;
using MdxViewer.Catalog;
using MdxViewer.DataSources;
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
    private readonly Dictionary<CharacterSectionKey, CharacterSectionData> _characterSections = new();
    private readonly Dictionary<CharacterVariationKey, int[]> _characterHairGeosets = new();
    private readonly Dictionary<CharacterVariationKey, int[]> _characterFacialHairGeosets = new();
    // ModelID → list of TextureVariation arrays (one per DisplayInfo entry)
    private readonly Dictionary<int, List<string[]>> _displayVariations = new();
    // Exact model path → list of fallback TextureVariation arrays sourced from alpha-core SQL.
    private readonly Dictionary<string, List<string[]>> _fallbackDisplayVariationsByModelPath = new(StringComparer.OrdinalIgnoreCase);
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

    private IDataSource? _dataSource;
    private bool _loaded;

    /// <summary>Set the data source for texture existence validation.</summary>
    public void SetDataSource(IDataSource? dataSource) => _dataSource = dataSource;

    private readonly record struct CharacterSectionKey(int RaceId, int SexId, int BaseSection, int VariationIndex, int ColorIndex);
    private readonly record struct CharacterVariationKey(int RaceId, int SexId, int VariationId);
    private record CharacterSectionData(string[] TextureNames);
    private record ItemDisplayData(string[] ModelNames, string[] ModelTextures, string[] Textures);
    private record ExtraDisplayData(string BakeName, int[] ItemDisplayIds);
    public readonly record struct ReplaceableResolutionCandidate(string Source, string Path, bool Exists);

    private static readonly uint[] DefaultCharacterSelectionGroups =
    {
        0,
        101,
        201,
        301,
        401,
        501,
        702,
        801,
        901,
        1001,
        1101,
        1201,
        1301,
        1401,
        1501,
    };

    /// <summary>Known build strings for version alias resolution.</summary>
    private static readonly Dictionary<string, string> BuildAliases = new()
    {
        { "0.5.3", "0.5.3.3368" },
        { "0.5.5", "0.5.5.3494" },
        { "0.6.0", "0.6.0.3592" },
        { "2.4.3", "2.4.3.8606" },
        { "3.3.5", "3.3.5.12340" },
    };

    private static readonly Dictionary<string, int> CharacterRaceIds = new(StringComparer.OrdinalIgnoreCase)
    {
        ["human"] = 1,
        ["orc"] = 2,
        ["dwarf"] = 3,
        ["nightelf"] = 4,
        ["scourge"] = 5,
        ["undead"] = 5,
        ["tauren"] = 6,
        ["gnome"] = 7,
        ["troll"] = 8,
        ["bloodelf"] = 10,
        ["draenei"] = 11,
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
            TryLoadCharSections(dbcd, build);
            TryLoadCharHairGeosets(dbcd, build);
            TryLoadCharacterFacialHairStyles(dbcd, build);
            LoadCreatureDisplayInfo(dbcd, build);
            TryLoadCreatureDisplayInfoExtra(dbcd, build);
            TryLoadItemDisplayInfo(dbcd, build);
            TryLoadAlphaCoreCreatureDisplayFallback(build);
            _loaded = true;

            ViewerLog.Important(ViewerLog.Category.Dbc, $"=== DBC Texture Resolution Summary (build {build}) ===");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CreatureModelData:          {_modelPathToId.Count} model paths");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CharSections:              {_characterSections.Count} character variants");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CharHairGeosets:           {_characterHairGeosets.Count} character variants");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CharacterFacialHairStyles: {_characterFacialHairGeosets.Count} character variants");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  CreatureDisplayInfo:        {_displayVariations.Values.Sum(v => v.Count)} display entries for {_displayVariations.Count} unique models");
            ViewerLog.Important(ViewerLog.Category.Dbc, $"  AlphaCore fallback:         {_fallbackDisplayVariationsByModelPath.Values.Sum(v => v.Count)} display entries for {_fallbackDisplayVariationsByModelPath.Count} exact model paths");
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

    private void TryLoadCharSections(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage? storage;
        try { storage = LoadDbc(dbcd, "CharSections", build); }
        catch { ViewerLog.Info(ViewerLog.Category.Dbc, "CharSections: not available"); return; }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];

            int raceId = TryGetInt(row, "RaceID") ?? 0;
            int sexId = TryGetInt(row, "SexID") ?? 0;
            int baseSection = TryGetInt(row, "BaseSection") ?? 0;
            int variationIndex = TryGetInt(row, "VariationIndex") ?? 0;
            int colorIndex = TryGetInt(row, "ColorIndex") ?? 0;
            string[] textureNames = ReadStringArray(row, "TextureName", 3);

            if (raceId <= 0 || !textureNames.Any(static name => !string.IsNullOrWhiteSpace(name)))
                continue;

            var keyTuple = new CharacterSectionKey(raceId, sexId, baseSection, variationIndex, colorIndex);
            var incoming = new CharacterSectionData(textureNames);

            if (_characterSections.TryGetValue(keyTuple, out CharacterSectionData? existing))
            {
                int existingCount = existing.TextureNames.Count(static name => !string.IsNullOrWhiteSpace(name));
                int incomingCount = incoming.TextureNames.Count(static name => !string.IsNullOrWhiteSpace(name));
                if (incomingCount <= existingCount)
                    continue;
            }

            _characterSections[keyTuple] = incoming;
            count++;
        }

        ViewerLog.Info(ViewerLog.Category.Dbc, $"CharSections: {count} usable entries loaded ({_characterSections.Count} unique variants)");
    }

    private void TryLoadCharHairGeosets(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage? storage;
        try { storage = LoadDbc(dbcd, "CharHairGeosets", build); }
        catch { ViewerLog.Info(ViewerLog.Category.Dbc, "CharHairGeosets: not available"); return; }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];

            int raceId = TryGetInt(row, "RaceID") ?? 0;
            int sexId = TryGetInt(row, "SexID") ?? 0;
            int variationId = TryGetInt(row, "VariationID") ?? 0;
            int geosetId = TryGetInt(row, "GeosetID") ?? 0;
            int scalpValue = TryGetInt(row, "Showscalp") ?? 0;
            if (raceId <= 0)
                continue;

            int[] groups = new[] { geosetId, scalpValue }
                .Where(static value => value > 0)
                .Distinct()
                .ToArray();
            if (groups.Length == 0)
                continue;

            _characterHairGeosets[new CharacterVariationKey(raceId, sexId, variationId)] = groups;
            count++;
        }

        ViewerLog.Info(ViewerLog.Category.Dbc, $"CharHairGeosets: {count} usable entries loaded ({_characterHairGeosets.Count} unique variants)");
    }

    private void TryLoadCharacterFacialHairStyles(DBCD.DBCD dbcd, string build)
    {
        IDBCDStorage? storage;
        try { storage = LoadDbc(dbcd, "CharacterFacialHairStyles", build); }
        catch { ViewerLog.Info(ViewerLog.Category.Dbc, "CharacterFacialHairStyles: not available"); return; }

        int count = 0;
        foreach (var key in storage.Keys)
        {
            var row = storage[key];

            int raceId = TryGetInt(row, "RaceID") ?? 0;
            int sexId = TryGetInt(row, "SexID") ?? 0;
            int variationId = TryGetInt(row, "VariationID") ?? TryGetInt(row, "VariationId") ?? 0;
            if (raceId <= 0)
                continue;

            int beard = TryGetInt(row, "BeardGeoset") ?? 0;
            int moustache = TryGetInt(row, "MoustacheGeoset") ?? 0;
            int sideburn = TryGetInt(row, "SideburnGeoset") ?? 0;

            int[] groups =
            {
                beard > 0 ? 100 + beard : 0,
                sideburn > 0 ? 200 + sideburn : 0,
                moustache > 0 ? 300 + moustache : 0,
            };

            int[] distinctGroups = groups
                .Where(static value => value > 0)
                .Distinct()
                .ToArray();
            if (distinctGroups.Length == 0)
                continue;

            _characterFacialHairGeosets[new CharacterVariationKey(raceId, sexId, variationId)] = distinctGroups;
            count++;
        }

        ViewerLog.Info(ViewerLog.Category.Dbc, $"CharacterFacialHairStyles: {count} usable entries loaded ({_characterFacialHairGeosets.Count} unique variants)");
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

            string normalized = modelName.ToLowerInvariant().Replace('/', '\\').Trim();
            foreach (string lookupKey in EnumerateCreatureModelLookupKeys(normalized))
                _modelPathToId.TryAdd(lookupKey, id);

            _modelIdToPath[id] = normalized;

            string fileNameKey = Path.GetFileNameWithoutExtension(normalized);
            _modelFileNameToId.TryAdd(fileNameKey, id);

            count++;
        }
        ViewerLog.Info(ViewerLog.Category.Dbc, $"CreatureModelData: {count} entries loaded ({_modelPathToId.Count} path entries)");

        // Dump a few sample paths for debugging
        int samples = 0;
        foreach (var kvp in _modelPathToId)
        {
            if (samples >= 5) break;
            if (kvp.Key.Contains("creature"))
            {
                ViewerLog.Debug(ViewerLog.Category.Dbc, $"  CMD sample: \"{kvp.Key}\" -> ModelID={kvp.Value}");
                samples++;
            }
        }
    }

    private static IEnumerable<string> EnumerateCreatureModelLookupKeys(string normalizedModelName)
    {
        if (string.IsNullOrWhiteSpace(normalizedModelName))
            yield break;

        yield return normalizedModelName;

        string withoutExt = normalizedModelName;
        if (normalizedModelName.EndsWith(".mdx") || normalizedModelName.EndsWith(".mdl"))
        {
            withoutExt = normalizedModelName[..^4];
            yield return withoutExt;
        }
        else
        {
            yield return normalizedModelName + ".mdx";
            yield return normalizedModelName + ".mdl";
        }

        bool hasDirectory = normalizedModelName.Contains('\\') || normalizedModelName.Contains('/');
        if (hasDirectory)
            yield break;

        string modelName = Path.GetFileNameWithoutExtension(normalizedModelName);
        if (string.IsNullOrWhiteSpace(modelName))
            yield break;

        yield return $"creature\\{modelName}\\{modelName}";
        yield return $"creature\\{modelName}\\{modelName}.mdx";
        yield return $"creature\\{modelName}\\{modelName}.mdl";
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

    private void TryLoadAlphaCoreCreatureDisplayFallback(string build)
    {
        if (!build.StartsWith("0.5.3", StringComparison.OrdinalIgnoreCase))
            return;

        string? alphaCoreRoot = ResolveAlphaCoreRoot();
        if (string.IsNullOrWhiteSpace(alphaCoreRoot))
            return;

        try
        {
            using var reader = new AlphaCoreDbReader(alphaCoreRoot);
            var validation = reader.Validate();
            if (!validation.success)
            {
                ViewerLog.Info(ViewerLog.Category.Dbc, $"Alpha-core fallback unavailable: {validation.message}");
                return;
            }

            IReadOnlyList<AssetCatalogEntry> creatures = reader.LoadCreaturesAsync().GetAwaiter().GetResult();
            int added = 0;

            foreach (AssetCatalogEntry creature in creatures)
            {
                if (string.IsNullOrWhiteSpace(creature.ModelPath) || creature.TextureVariations.Length == 0)
                    continue;

                string normalizedPath = creature.ModelPath.Replace('/', '\\').ToLowerInvariant();
                if (!_fallbackDisplayVariationsByModelPath.TryGetValue(normalizedPath, out List<string[]>? variants))
                {
                    variants = new List<string[]>();
                    _fallbackDisplayVariationsByModelPath[normalizedPath] = variants;
                }

                string[] variationSet = BuildTextureVariationSet(creature.TextureVariations);
                if (variants.Any(existing => existing.SequenceEqual(variationSet, StringComparer.OrdinalIgnoreCase)))
                    continue;

                variants.Add(variationSet);
                added++;
            }

            ViewerLog.Info(ViewerLog.Category.Dbc,
                $"Alpha-core fallback: {added} display variants across {_fallbackDisplayVariationsByModelPath.Count} model paths");
        }
        catch (Exception ex)
        {
            ViewerLog.Info(ViewerLog.Category.Dbc, $"Alpha-core fallback load failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Resolve a replaceable texture ID to a BLP path for the given model.
    /// </summary>
    public string? Resolve(string modelPath, uint replaceableId, int displayIndex = 0, int? hairVariationId = null, int? facialHairVariationId = null)
    {
        if (!_loaded)
            return null;

        string normalizedPath = modelPath.ToLowerInvariant().Replace('/', '\\');
        string? characterResult = ResolveFromCharacterSections(normalizedPath, replaceableId, hairVariationId, facialHairVariationId);
        if (characterResult != null)
            return characterResult;

        if (TryGetDisplayVariations(normalizedPath, null, out var fallbackVariants))
        {
            string? fallbackResult = ResolveFromDisplayVariations(0, normalizedPath, replaceableId, displayIndex, fallbackVariants);
            if (fallbackResult != null)
                return fallbackResult;
        }

        int modelId = FindModelId(modelPath);
        if (modelId == 0)
        {
            return null;
        }

        // Try creature TextureVariation first (covers ReplaceableId 1-3 and 11-13)
        string? result = ResolveFromCreatureDisplay(modelId, normalizedPath, replaceableId, displayIndex);
        if (result != null) return result;

        // Try CreatureDisplayInfoExtra bake texture
        result = ResolveFromExtraDisplay(modelId, replaceableId);
        if (result != null) return result;

        // Try ItemDisplayInfo for NPC equipped items
        result = ResolveFromItemDisplay(modelId, replaceableId);
        if (result != null) return result;

        return null;
    }

    public IReadOnlyList<ReplaceableResolutionCandidate> GetReplaceableResolutionCandidates(string modelPath, uint replaceableId, int displayIndex = 0, int? hairVariationId = null, int? facialHairVariationId = null)
    {
        List<ReplaceableResolutionCandidate> candidates = new();
        string normalizedPath = modelPath.ToLowerInvariant().Replace('/', '\\');

        AddCharacterReplaceableCandidates(candidates, normalizedPath, replaceableId, hairVariationId, facialHairVariationId);

        return candidates
            .GroupBy(static candidate => candidate.Path, StringComparer.OrdinalIgnoreCase)
            .Select(static group => group.First())
            .ToArray();
    }

    public IReadOnlyCollection<uint>? GetDefaultCharacterSelectionGroups(string modelPath)
        => GetCharacterSelectionGroups(modelPath, hairVariationId: 0, facialHairVariationId: 0);

    public IReadOnlyCollection<uint>? GetCharacterSelectionGroups(string modelPath, int? hairVariationId = null, int? facialHairVariationId = null)
    {
        string normalizedPath = modelPath.Replace('/', '\\');
        if (!TryParseCharacterModelPath(normalizedPath, out int raceId, out int sexId))
            return null;

        HashSet<uint> groups = new(DefaultCharacterSelectionGroups);

        AddCharacterVariationGroups(groups, _characterHairGeosets, new CharacterVariationKey(raceId, sexId, hairVariationId ?? 0));
        AddCharacterVariationGroups(groups, _characterFacialHairGeosets, new CharacterVariationKey(raceId, sexId, facialHairVariationId ?? 0));

        return groups;
    }

    public IReadOnlyList<int> GetCharacterHairVariationIds(string modelPath)
        => GetCharacterVariationIds(modelPath, _characterHairGeosets);

    public IReadOnlyList<int> GetCharacterFacialHairVariationIds(string modelPath)
        => GetCharacterVariationIds(modelPath, _characterFacialHairGeosets);

    private IReadOnlyList<int> GetCharacterVariationIds(string modelPath, Dictionary<CharacterVariationKey, int[]> source)
    {
        string normalizedPath = modelPath.Replace('/', '\\');
        if (!TryParseCharacterModelPath(normalizedPath, out int raceId, out int sexId))
            return Array.Empty<int>();

        return source.Keys
            .Where(key => key.RaceId == raceId && key.SexId == sexId)
            .Select(key => key.VariationId)
            .Distinct()
            .OrderBy(static value => value)
            .ToArray();
    }

    private static void AddCharacterVariationGroups(HashSet<uint> groups, Dictionary<CharacterVariationKey, int[]> source, CharacterVariationKey key)
    {
        if (!source.TryGetValue(key, out int[]? variationGroups))
            return;

        foreach (int group in variationGroups)
        {
            groups.Add((uint)group);
        }
    }

    private string? ResolveFromCharacterSections(string modelPath, uint replaceableId, int? hairVariationId, int? facialHairVariationId)
    {
        if (!TryParseCharacterModelPath(modelPath, out int raceId, out int sexId))
            return null;

        int resolvedHairVariationId = hairVariationId ?? 0;
        int resolvedFacialHairVariationId = facialHairVariationId ?? 0;

        string? resolved = replaceableId switch
        {
            1 => TryResolveCharacterSectionTexture(modelPath, raceId, sexId, baseSection: 0, variationIndex: 0, colorIndex: 0, textureIndices: new[] { 0 }),
                2 => TryResolveCharacterSkinExtraTexture(modelPath, raceId, sexId, variationIndex: 0, colorIndex: 0),
            8 => TryResolveCharacterSkinExtraTexture(modelPath, raceId, sexId, variationIndex: 0, colorIndex: 0),
            6 => TryResolvePreferredCharacterSectionTexture(modelPath, raceId, sexId, baseSection: 4, variationIndex: resolvedHairVariationId, colorIndex: 0, replaceableId),
            7 => TryResolvePreferredCharacterSectionTexture(modelPath, raceId, sexId, baseSection: 2, variationIndex: resolvedFacialHairVariationId, colorIndex: 0, replaceableId),
            10 => TryResolvePreferredCharacterSectionTexture(modelPath, raceId, sexId, baseSection: 4, variationIndex: resolvedHairVariationId, colorIndex: 0, replaceableId),
            _ => null,
        };

        return resolved ?? ResolveFromCharacterDirectory(modelPath, replaceableId, hairVariationId, facialHairVariationId);
    }

    private string? TryResolveCharacterSkinExtraTexture(string modelPath, int raceId, int sexId, int variationIndex, int colorIndex)
    {
        if (!TryGetCharacterSection(raceId, sexId, 0, variationIndex, colorIndex, out CharacterSectionData section))
            return null;

        foreach (int textureIndex in new[] { 1, 2 })
        {
            string? resolved = TryResolveCharacterTextureCandidate(modelPath, section.TextureNames, textureIndex);
            if (resolved != null)
                return resolved;
        }

        return TryInferCharacterSkinExtraTexture(modelPath, section.TextureNames[0]);
    }

    private string? TryResolveCharacterSectionTexture(string modelPath, int raceId, int sexId, int baseSection, int variationIndex, int colorIndex, int[] textureIndices)
    {
        if (!TryGetCharacterSection(raceId, sexId, baseSection, variationIndex, colorIndex, out CharacterSectionData section))
            return null;

        foreach (int textureIndex in textureIndices)
        {
            string? resolved = TryResolveCharacterTextureCandidate(modelPath, section.TextureNames, textureIndex);
            if (resolved != null)
                return resolved;
        }

        return null;
    }

    private string? TryResolvePreferredCharacterSectionTexture(string modelPath, int raceId, int sexId, int baseSection, int variationIndex, int colorIndex, uint replaceableId)
    {
        foreach ((CharacterSectionData section, _) in EnumerateCharacterSectionFallbacks(raceId, sexId, baseSection, variationIndex, colorIndex))
        {
            foreach (int textureIndex in GetPreferredCharacterTextureIndices(section.TextureNames, replaceableId))
            {
                string? resolved = TryResolveCharacterTextureCandidate(modelPath, section.TextureNames, textureIndex);
                if (resolved != null)
                    return resolved;
            }
        }

        return null;
    }

    private bool TryGetCharacterSection(int raceId, int sexId, int baseSection, int variationIndex, int colorIndex, out CharacterSectionData section)
    {
        if (TryGetCharacterSectionInternal(new CharacterSectionKey(raceId, sexId, baseSection, variationIndex, colorIndex), out section))
            return true;

        if (colorIndex != 0 && TryGetCharacterSectionInternal(new CharacterSectionKey(raceId, sexId, baseSection, variationIndex, 0), out section))
            return true;

        if (variationIndex != 0 && TryGetCharacterSectionInternal(new CharacterSectionKey(raceId, sexId, baseSection, 0, colorIndex), out section))
            return true;

        if (variationIndex != 0 && colorIndex != 0 && TryGetCharacterSectionInternal(new CharacterSectionKey(raceId, sexId, baseSection, 0, 0), out section))
            return true;

        section = null!;
        return false;
    }

    private bool TryGetCharacterSectionInternal(CharacterSectionKey key, out CharacterSectionData section)
    {
        if (_characterSections.TryGetValue(key, out CharacterSectionData? existing) && existing != null)
        {
            section = existing;
            return true;
        }

        section = null!;
        return false;
    }

    private string? TryResolveCharacterTextureCandidate(string modelPath, string[] textureNames, int textureIndex)
    {
        if (textureIndex < 0 || textureIndex >= textureNames.Length)
            return null;

        string texName = textureNames[textureIndex].Trim();
        if (string.IsNullOrEmpty(texName))
            return null;

        string candidate = BuildTexturePath(texName, modelPath);
        if (_dataSource == null || TextureExistsInDataSource(candidate))
            return candidate;

        return null;
    }

    private string? TryInferCharacterSkinExtraTexture(string modelPath, string baseTextureName)
    {
        if (string.IsNullOrWhiteSpace(baseTextureName))
            return null;

        string basePath = BuildTexturePath(baseTextureName, modelPath);
        string extension = Path.GetExtension(basePath);
        if (string.IsNullOrEmpty(extension))
            extension = ".blp";

        string inferred = Path.ChangeExtension(basePath, null) + "_Extra" + extension;
        if (_dataSource == null || TextureExistsInDataSource(inferred))
            return inferred;

        return null;
    }

    private void AddCharacterReplaceableCandidates(List<ReplaceableResolutionCandidate> candidates, string modelPath, uint replaceableId, int? hairVariationId, int? facialHairVariationId)
    {
        if (!TryParseCharacterModelPath(modelPath, out int raceId, out int sexId))
            return;

        int resolvedHairVariationId = hairVariationId ?? 0;
        int resolvedFacialHairVariationId = facialHairVariationId ?? 0;

        switch (replaceableId)
        {
            case 1:
                AddCharacterSectionCandidates(candidates, modelPath, raceId, sexId, baseSection: 0, variationIndex: 0, colorIndex: 0, textureIndices: new[] { 0 }, sourceRoot: "char-section-body");
                break;

            case 2:
                AddCharacterSkinExtraCandidates(candidates, modelPath, raceId, sexId, variationIndex: 0, colorIndex: 0);
                break;

            case 8:
                AddCharacterSkinExtraCandidates(candidates, modelPath, raceId, sexId, variationIndex: 0, colorIndex: 0);
                break;

            case 6:
                AddPreferredCharacterSectionCandidates(candidates, modelPath, raceId, sexId, baseSection: 4, variationIndex: resolvedHairVariationId, colorIndex: 0, replaceableId, sourceRoot: $"char-section-hair[var={resolvedHairVariationId}]");
                break;

            case 7:
                AddPreferredCharacterSectionCandidates(candidates, modelPath, raceId, sexId, baseSection: 2, variationIndex: resolvedFacialHairVariationId, colorIndex: 0, replaceableId, sourceRoot: $"char-section-facial[var={resolvedFacialHairVariationId}]");
                break;

            case 10:
                AddPreferredCharacterSectionCandidates(candidates, modelPath, raceId, sexId, baseSection: 4, variationIndex: resolvedHairVariationId, colorIndex: 0, replaceableId, sourceRoot: $"char-section-mane[var={resolvedHairVariationId}]");
                break;
        }

        AddCharacterDirectoryCandidates(candidates, modelPath, replaceableId, hairVariationId, facialHairVariationId);
    }

    private void AddCharacterSkinExtraCandidates(List<ReplaceableResolutionCandidate> candidates, string modelPath, int raceId, int sexId, int variationIndex, int colorIndex)
    {
        int initialCount = candidates.Count;
        foreach ((CharacterSectionData section, string fallbackLabel) in EnumerateCharacterSectionFallbacks(raceId, sexId, baseSection: 0, variationIndex, colorIndex))
        {
            foreach (int textureIndex in new[] { 1, 2 })
            {
                AddCharacterTextureCandidate(candidates, modelPath, section.TextureNames, textureIndex, $"char-section-extra/{fallbackLabel}");
            }

            if (section.TextureNames.Length > 0 && !string.IsNullOrWhiteSpace(section.TextureNames[0]))
            {
                string? inferred = TryInferCharacterSkinExtraTexture(modelPath, section.TextureNames[0]);
                if (!string.IsNullOrWhiteSpace(inferred))
                {
                    candidates.Add(new ReplaceableResolutionCandidate(
                        $"char-section-extra-inferred/{fallbackLabel}",
                        inferred,
                        TexturePathExists(inferred)));
                }
            }
        }

        if (candidates.Count == initialCount)
        {
            candidates.Add(new ReplaceableResolutionCandidate(
                "char-section-extra/missing-section",
                "<no matching CharSections entry>",
                false));
        }
    }

    private void AddCharacterSectionCandidates(List<ReplaceableResolutionCandidate> candidates, string modelPath, int raceId, int sexId, int baseSection, int variationIndex, int colorIndex, int[] textureIndices, string sourceRoot)
    {
        int initialCount = candidates.Count;
        foreach ((CharacterSectionData section, string fallbackLabel) in EnumerateCharacterSectionFallbacks(raceId, sexId, baseSection, variationIndex, colorIndex))
        {
            foreach (int textureIndex in textureIndices)
            {
                AddCharacterTextureCandidate(candidates, modelPath, section.TextureNames, textureIndex, $"{sourceRoot}/{fallbackLabel}");
            }
        }

        if (candidates.Count == initialCount)
        {
            candidates.Add(new ReplaceableResolutionCandidate(
                $"{sourceRoot}/missing-section",
                "<no matching CharSections entry>",
                false));
        }
    }

    private void AddPreferredCharacterSectionCandidates(List<ReplaceableResolutionCandidate> candidates, string modelPath, int raceId, int sexId, int baseSection, int variationIndex, int colorIndex, uint replaceableId, string sourceRoot)
    {
        int initialCount = candidates.Count;
        foreach ((CharacterSectionData section, string fallbackLabel) in EnumerateCharacterSectionFallbacks(raceId, sexId, baseSection, variationIndex, colorIndex))
        {
            foreach (int textureIndex in GetPreferredCharacterTextureIndices(section.TextureNames, replaceableId))
            {
                AddCharacterTextureCandidate(candidates, modelPath, section.TextureNames, textureIndex, $"{sourceRoot}/{fallbackLabel}/tex{textureIndex}");
            }
        }

        if (candidates.Count == initialCount)
        {
            candidates.Add(new ReplaceableResolutionCandidate(
                $"{sourceRoot}/missing-section",
                "<no preferred CharSections texture>",
                false));
        }
    }

    private IEnumerable<(CharacterSectionData Section, string Label)> EnumerateCharacterSectionFallbacks(int raceId, int sexId, int baseSection, int variationIndex, int colorIndex)
    {
        CharacterSectionKey[] keys =
        {
            new(raceId, sexId, baseSection, variationIndex, colorIndex),
            new(raceId, sexId, baseSection, variationIndex, 0),
            new(raceId, sexId, baseSection, 0, colorIndex),
            new(raceId, sexId, baseSection, 0, 0),
        };

        string[] labels =
        {
            "exact",
            "color=0",
            "variation=0",
            "variation=0,color=0",
        };

        HashSet<CharacterSectionKey> seen = new();
        for (int index = 0; index < keys.Length; index++)
        {
            CharacterSectionKey key = keys[index];
            if (!seen.Add(key))
                continue;

            if (_characterSections.TryGetValue(key, out CharacterSectionData? section) && section != null)
                yield return (section, labels[index]);
        }
    }

    private void AddCharacterTextureCandidate(List<ReplaceableResolutionCandidate> candidates, string modelPath, string[] textureNames, int textureIndex, string source)
    {
        if (textureIndex < 0 || textureIndex >= textureNames.Length)
            return;

        string texName = textureNames[textureIndex].Trim();
        if (string.IsNullOrEmpty(texName))
            return;

        string candidate = BuildTexturePath(texName, modelPath);
        candidates.Add(new ReplaceableResolutionCandidate(source, candidate, TexturePathExists(candidate)));
    }

    private static IEnumerable<int> GetPreferredCharacterTextureIndices(string[] textureNames, uint replaceableId)
    {
        return Enumerable.Range(0, textureNames.Length)
            .Select(index => new { Index = index, Score = ScoreCharacterSectionTextureName(textureNames[index], replaceableId) })
            .Where(entry => entry.Score > 0)
            .OrderByDescending(entry => entry.Score)
            .ThenBy(entry => entry.Index)
            .Select(entry => entry.Index);
    }

    private static int ScoreCharacterSectionTextureName(string textureName, uint replaceableId)
    {
        string name = Path.GetFileNameWithoutExtension(textureName ?? string.Empty).ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(name))
            return 0;

        return replaceableId switch
        {
            6 => ScoreHairLikeTextureName(name),
            7 => ScoreFacialLikeTextureName(name),
            10 => ScoreManeLikeTextureName(name),
            _ => 0,
        };
    }

    private static int ScoreHairLikeTextureName(string name)
    {
        int score = 0;
        if (name.Contains("hair", StringComparison.Ordinal)) score += 100;
        if (name.Contains("scalp", StringComparison.Ordinal)) score += 60;
        if (name.Contains("braid", StringComparison.Ordinal) || name.Contains("pigtail", StringComparison.Ordinal) || name.Contains("ponytail", StringComparison.Ordinal)) score += 40;
        if (name.Contains("facial", StringComparison.Ordinal) || name.Contains("beard", StringComparison.Ordinal) || name.Contains("mustache", StringComparison.Ordinal) || name.Contains("moustache", StringComparison.Ordinal)) score -= 80;
        if (name.Contains("pelvis", StringComparison.Ordinal) || name.Contains("naked", StringComparison.Ordinal) || name.Contains("skin", StringComparison.Ordinal)) score -= 120;
        return score;
    }

    private static int ScoreFacialLikeTextureName(string name)
    {
        int score = 0;
        if (name.Contains("facial", StringComparison.Ordinal) || name.Contains("beard", StringComparison.Ordinal) || name.Contains("mustache", StringComparison.Ordinal) || name.Contains("moustache", StringComparison.Ordinal) || name.Contains("sideburn", StringComparison.Ordinal) || name.Contains("goatee", StringComparison.Ordinal)) score += 100;
        if (name.Contains("hair", StringComparison.Ordinal) || name.Contains("mane", StringComparison.Ordinal)) score -= 60;
        if (name.Contains("pelvis", StringComparison.Ordinal) || name.Contains("naked", StringComparison.Ordinal) || name.Contains("skin", StringComparison.Ordinal)) score -= 120;
        return score;
    }

    private static int ScoreManeLikeTextureName(string name)
    {
        int score = 0;
        if (name.Contains("mane", StringComparison.Ordinal)) score += 100;
        if (name.Contains("hair", StringComparison.Ordinal)) score += 60;
        if (name.Contains("facial", StringComparison.Ordinal) || name.Contains("beard", StringComparison.Ordinal)) score -= 60;
        if (name.Contains("pelvis", StringComparison.Ordinal) || name.Contains("naked", StringComparison.Ordinal) || name.Contains("skin", StringComparison.Ordinal)) score -= 120;
        return score;
    }

    private string? ResolveFromCharacterDirectory(string modelPath, uint replaceableId, int? hairVariationId, int? facialHairVariationId)
    {
        if (_dataSource == null)
            return null;

        return GetCharacterDirectoryResolutionCandidates(modelPath, replaceableId, hairVariationId, facialHairVariationId)
            .FirstOrDefault(static candidate => candidate.Exists)
            .Path;
    }

    private void AddCharacterDirectoryCandidates(List<ReplaceableResolutionCandidate> candidates, string modelPath, uint replaceableId, int? hairVariationId, int? facialHairVariationId)
    {
        int initialCount = candidates.Count;
        foreach (ReplaceableResolutionCandidate candidate in GetCharacterDirectoryResolutionCandidates(modelPath, replaceableId, hairVariationId, facialHairVariationId))
        {
            candidates.Add(candidate);
        }

        if (candidates.Count == initialCount)
        {
            candidates.Add(new ReplaceableResolutionCandidate(
                "char-directory-scan/no-matches",
                "<no matching character directory textures>",
                false));
        }
    }

    private IReadOnlyList<ReplaceableResolutionCandidate> GetCharacterDirectoryResolutionCandidates(string modelPath, uint replaceableId, int? hairVariationId, int? facialHairVariationId)
    {
        List<ReplaceableResolutionCandidate> candidates = new();

        string modelDir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? string.Empty;
        string modelBase = Path.GetFileNameWithoutExtension(modelPath) ?? string.Empty;
        if (string.IsNullOrEmpty(modelDir) || string.IsNullOrEmpty(modelBase))
            return candidates;

        foreach (string candidate in EnumerateCharacterDirectoryCandidates(modelDir, modelBase, replaceableId))
        {
            candidates.Add(new ReplaceableResolutionCandidate("char-directory-explicit", candidate, TexturePathExists(candidate)));
        }

        string modelDirLower = modelDir.ToLowerInvariant();
        string modelBaseLower = modelBase.ToLowerInvariant();
        int? requestedVariationId = replaceableId switch
        {
            6 or 10 => hairVariationId,
            7 => facialHairVariationId,
            _ => null,
        };

        var matches = _dataSource.GetFileList(".blp")
            .Where(path =>
            {
                string normalized = path.Replace('/', '\\').ToLowerInvariant();
                string fileDir = Path.GetDirectoryName(normalized) ?? string.Empty;
                string fileName = Path.GetFileNameWithoutExtension(normalized);
                if (!fileDir.Equals(modelDirLower, StringComparison.Ordinal))
                    return false;

                return replaceableId switch
                {
                    1 => fileName.StartsWith(modelBaseLower + "skin", StringComparison.Ordinal) && !fileName.Contains("extra", StringComparison.Ordinal),
                    8 => fileName.StartsWith(modelBaseLower + "skin", StringComparison.Ordinal) && fileName.Contains("extra", StringComparison.Ordinal),
                    6 => fileName.StartsWith(modelBaseLower, StringComparison.Ordinal) && fileName.Contains("hair", StringComparison.Ordinal) && !fileName.Contains("facial", StringComparison.Ordinal) && !fileName.Contains("skin", StringComparison.Ordinal),
                    7 => fileName.StartsWith(modelBaseLower, StringComparison.Ordinal) && (fileName.Contains("facial", StringComparison.Ordinal) || fileName.Contains("beard", StringComparison.Ordinal) || fileName.Contains("moustache", StringComparison.Ordinal) || fileName.Contains("mustache", StringComparison.Ordinal) || fileName.Contains("sideburn", StringComparison.Ordinal)),
                    10 => fileName.StartsWith(modelBaseLower, StringComparison.Ordinal) && (fileName.Contains("mane", StringComparison.Ordinal) || (fileName.Contains("hair", StringComparison.Ordinal) && !fileName.Contains("facial", StringComparison.Ordinal))),
                    _ => false,
                };
            })
            .Select(path => new
            {
                Path = path.Replace('/', '\\'),
                Score = ScoreCharacterDirectoryMatch(Path.GetFileNameWithoutExtension(path).ToLowerInvariant(), replaceableId, requestedVariationId),
            })
            .OrderByDescending(static candidate => candidate.Score)
            .ThenBy(static candidate => candidate.Path.Length)
            .ToArray();

        foreach (var match in matches)
        {
            candidates.Add(new ReplaceableResolutionCandidate($"char-directory-scan(score={match.Score})", match.Path, TexturePathExists(match.Path)));
        }

        return candidates
            .GroupBy(static candidate => candidate.Path, StringComparer.OrdinalIgnoreCase)
            .Select(static group => group.First())
            .ToArray();
    }

    private static int ScoreCharacterDirectoryMatch(string fileName, uint replaceableId, int? requestedVariationId)
    {
        int score = 0;

        if (replaceableId == 10 && fileName.Contains("mane", StringComparison.Ordinal))
            score += 20;
        if (replaceableId == 6 && fileName.Contains("hair", StringComparison.Ordinal))
            score += 20;
        if (replaceableId == 7 && (fileName.Contains("facial", StringComparison.Ordinal) || fileName.Contains("beard", StringComparison.Ordinal) || fileName.Contains("moustache", StringComparison.Ordinal) || fileName.Contains("mustache", StringComparison.Ordinal) || fileName.Contains("sideburn", StringComparison.Ordinal)))
            score += 20;

        if (requestedVariationId.HasValue)
        {
            string token = requestedVariationId.Value.ToString("00");
            if (fileName.Contains(token, StringComparison.Ordinal))
                score += 10;
        }

        if (fileName.EndsWith("_00", StringComparison.Ordinal))
            score += 2;

        return score;
    }

    private bool TexturePathExists(string texPath)
        => _dataSource == null || TextureExistsInDataSource(texPath);

    private static IEnumerable<string> EnumerateCharacterDirectoryCandidates(string modelDir, string modelBase, uint replaceableId)
    {
        switch (replaceableId)
        {
            case 1:
                yield return Path.Combine(modelDir, modelBase + "Skin00_00.blp");
                yield return Path.Combine(modelDir, modelBase + "Skin.blp");
                break;

            case 8:
                yield return Path.Combine(modelDir, modelBase + "Skin00_00_Extra.blp");
                yield return Path.Combine(modelDir, modelBase + "Skin_Extra.blp");
                yield return Path.Combine(modelDir, modelBase + "SkinExtra.blp");
                break;

            case 6:
                yield return Path.Combine(modelDir, modelBase + "Hair.blp");
                yield return Path.Combine(modelDir, modelBase + "_Hair.blp");
                break;

            case 7:
                yield return Path.Combine(modelDir, modelBase + "FacialHair.blp");
                yield return Path.Combine(modelDir, modelBase + "_FacialHair.blp");
                yield return Path.Combine(modelDir, modelBase + "Facial.blp");
                break;

            case 10:
                yield return Path.Combine(modelDir, modelBase + "Mane.blp");
                yield return Path.Combine(modelDir, modelBase + "_Mane.blp");
                yield return Path.Combine(modelDir, modelBase + "Hair.blp");
                break;
        }
    }

    private static bool TryParseCharacterModelPath(string modelPath, out int raceId, out int sexId)
    {
        raceId = 0;
        sexId = 0;

        string[] segments = modelPath.Split('\\', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (segments.Length < 3 || !segments[0].Equals("character", StringComparison.OrdinalIgnoreCase))
            return false;

        if (!CharacterRaceIds.TryGetValue(segments[1], out raceId))
            return false;

        sexId = segments[2].Equals("female", StringComparison.OrdinalIgnoreCase) ? 1
            : segments[2].Equals("male", StringComparison.OrdinalIgnoreCase) ? 0
            : -1;

        return sexId >= 0;
    }

    /// <summary>
    /// Pick one creature display variant index that best satisfies all replaceable creature slots
    /// used by a model. This keeps ReplaceableIds like 11/12/13 on the same display entry.
    /// </summary>
    public int? SelectBestDisplayIndex(string modelPath, IEnumerable<uint> replaceableIds)
    {
        if (!_loaded)
            return null;

        string normalizedPath = modelPath.ToLowerInvariant().Replace('/', '\\');
        if (TryGetDisplayVariations(normalizedPath, null, out var fallbackVariants))
            return SelectBestDisplayIndexFromVariants(normalizedPath, replaceableIds, fallbackVariants);

        int modelId = FindModelId(modelPath);
        if (modelId == 0)
            return null;

        if (!TryGetDisplayVariations(normalizedPath, modelId, out var variants))
            return null;

        return SelectBestDisplayIndexFromVariants(normalizedPath, replaceableIds, variants);
    }

    private int? SelectBestDisplayIndexFromVariants(string modelPath, IEnumerable<uint> replaceableIds, List<string[]> variants)
    {
        var relevantIds = replaceableIds
            .Distinct()
            .Where(static id => GetCreatureDisplayVariationIndex(id) >= 0)
            .ToArray();
        if (relevantIds.Length == 0)
            return null;

        int bestIndex = -1;
        int bestResolvedCount = -1;
        int bestNamedCount = -1;

        for (int variantIndex = 0; variantIndex < variants.Count; variantIndex++)
        {
            string[] texNames = variants[variantIndex];
            int resolvedCount = 0;
            int namedCount = 0;

            foreach (uint replaceableId in relevantIds)
            {
                string? candidate = BuildCreatureDisplayCandidatePath(modelPath, texNames, replaceableId);
                if (string.IsNullOrEmpty(candidate))
                    continue;

                namedCount++;
                if (TryResolveCreatureDisplayCandidate(modelPath, texNames, replaceableId, out _))
                    resolvedCount++;
            }

            if (resolvedCount > bestResolvedCount
                || (resolvedCount == bestResolvedCount && namedCount > bestNamedCount))
            {
                bestIndex = variantIndex;
                bestResolvedCount = resolvedCount;
                bestNamedCount = namedCount;
            }
        }

        if (bestIndex < 0 || (bestResolvedCount <= 0 && bestNamedCount <= 0))
            return null;

        return bestIndex;
    }

    private string? ResolveFromCreatureDisplay(int modelId, string modelPath, uint replaceableId, int displayIndex)
    {
        if (!TryGetDisplayVariations(modelPath, modelId, out var variants))
            return null;

        return ResolveFromDisplayVariations(modelId, modelPath, replaceableId, displayIndex, variants);
    }

    private string? ResolveFromDisplayVariations(int modelId, string modelPath, uint replaceableId, int displayIndex, List<string[]> variants)
    {
        int varIndex = GetCreatureDisplayVariationIndex(replaceableId);

        if (varIndex < 0) return null;

        // Try the requested display index first, then all others.
        // Validate that the resolved texture actually exists in the data source,
        // because a model can have many CDI entries and displayIndex=0 may be wrong.
        var indicesToTry = new List<int>();
        int startIdx = displayIndex < variants.Count ? displayIndex : 0;
        indicesToTry.Add(startIdx);
        for (int i = 0; i < variants.Count; i++)
        {
            if (i != startIdx) indicesToTry.Add(i);
        }

        string? firstCandidate = null; // first non-empty result (even if not validated)
        string modelDir = (Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? "").ToLowerInvariant();

        foreach (int idx in indicesToTry)
        {
            var texNames = variants[idx];
            string? candidate = BuildCreatureDisplayCandidatePath(modelPath, texNames, replaceableId);
            if (string.IsNullOrEmpty(candidate))
                continue;

            // If we have a data source, validate the texture exists
            if (_dataSource != null)
            {
                if (TryResolveCreatureDisplayCandidate(modelPath, texNames, replaceableId, out string? resolved))
                    return resolved;

                // Remember first candidate as fallback
                firstCandidate ??= candidate;
            }
            else
            {
                // No data source to validate — return first non-empty result
                return candidate;
            }
        }

        // Don't return unvalidated DBC candidates — let caller fall through to directory scan
        if (firstCandidate != null)
        {
            ViewerLog.Debug(ViewerLog.Category.Dbc, $"DBC: no validated texture for ModelID={modelId} ({Path.GetFileName(modelPath)}), replId={replaceableId}, {variants.Count} variants tried");
        }
        return null;
    }

    private bool TryGetDisplayVariations(string normalizedModelPath, int? modelId, out List<string[]> variants)
    {
        if (_fallbackDisplayVariationsByModelPath.TryGetValue(normalizedModelPath, out variants!) && variants.Count > 0)
            return true;

        if (modelId.HasValue && modelId.Value != 0
            && _displayVariations.TryGetValue(modelId.Value, out variants!)
            && variants.Count > 0)
        {
            return true;
        }

        variants = null!;
        return false;
    }

    private static int GetCreatureDisplayVariationIndex(uint replaceableId)
    {
        return replaceableId switch
        {
            1 => 0,
            2 => 1,
            3 => 2,
            11 => 0,
            12 => 1,
            13 => 2,
            _ => -1
        };
    }

    private static string? BuildCreatureDisplayCandidatePath(string modelPath, string[] texNames, uint replaceableId)
    {
        int varIndex = GetCreatureDisplayVariationIndex(replaceableId);
        if (varIndex < 0 || varIndex >= texNames.Length)
            return null;

        string texName = texNames[varIndex].Trim();
        if (string.IsNullOrEmpty(texName))
            return null;

        return BuildTexturePath(texName, modelPath);
    }

    private bool TryResolveCreatureDisplayCandidate(string modelPath, string[] texNames, uint replaceableId, out string? resolvedPath)
    {
        resolvedPath = null;

        string? candidate = BuildCreatureDisplayCandidatePath(modelPath, texNames, replaceableId);
        if (string.IsNullOrEmpty(candidate))
            return false;

        if (_dataSource == null)
        {
            resolvedPath = candidate;
            return true;
        }

        if (TextureExistsInDataSource(candidate))
        {
            resolvedPath = candidate;
            return true;
        }

        int varIndex = GetCreatureDisplayVariationIndex(replaceableId);
        string texName = texNames[varIndex].Trim();
        string modelDir = (Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? string.Empty).ToLowerInvariant();
        if (!texName.Contains('\\') && !texName.Contains('/') && !string.IsNullOrEmpty(modelDir))
        {
            string inModelDir = Path.Combine(modelDir, texName.ToLowerInvariant());
            if (!inModelDir.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                inModelDir += ".blp";

            if (TextureExistsInDataSource(inModelDir))
            {
                resolvedPath = inModelDir;
                return true;
            }
        }

        return false;
    }

    /// <summary>Check if a texture path exists in the data source (case-insensitive, no file read).</summary>
    private bool TextureExistsInDataSource(string texPath)
    {
        if (_dataSource is MpqDataSource mpq)
        {
            return mpq.FileExists(texPath);
        }
        if (_dataSource != null)
        {
            var data = _dataSource.ReadFile(texPath);
            return data != null && data.Length > 0;
        }
        return false;
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
        string normalizedPath = modelPath.ToLowerInvariant().Replace('/', '\\');
        if (_fallbackDisplayVariationsByModelPath.TryGetValue(normalizedPath, out var fallbackVariants))
            return fallbackVariants.Count;

        int modelId = FindModelId(modelPath);
        if (modelId == 0 || !_displayVariations.TryGetValue(modelId, out var variants))
            return 0;
        return variants.Count;
    }

    /// <summary>Get display variant description.</summary>
    public string GetVariantDescription(string modelPath, int displayIndex)
    {
        if (!_loaded) return "";
        string normalizedPath = modelPath.ToLowerInvariant().Replace('/', '\\');
        if (_fallbackDisplayVariationsByModelPath.TryGetValue(normalizedPath, out var fallbackVariants))
        {
            if (displayIndex >= fallbackVariants.Count) return "";
            return string.Join(", ", fallbackVariants[displayIndex].Where(s => !string.IsNullOrEmpty(s)));
        }

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

        // Try without extension (DBC ModelName often omits .mdx)
        string withoutExt = normalized;
        if (withoutExt.EndsWith(".mdx") || withoutExt.EndsWith(".mdl"))
            withoutExt = withoutExt[..^4];
        if (_modelPathToId.TryGetValue(withoutExt, out modelId))
            return modelId;

        // Try with .mdx appended (DBC ModelName sometimes includes extension)
        if (!normalized.EndsWith(".mdx"))
        {
            if (_modelPathToId.TryGetValue(normalized + ".mdx", out modelId))
                return modelId;
        }

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

    private static string[] BuildTextureVariationSet(IReadOnlyList<string> source)
    {
        var result = new string[3];
        for (int i = 0; i < Math.Min(source.Count, result.Length); i++)
            result[i] = source[i]?.Trim() ?? string.Empty;
        return result;
    }

    private static string? ResolveAlphaCoreRoot()
    {
        string[] candidateRoots =
        {
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "external", "alpha-core"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "..", "gillijimproject_refactor", "external", "alpha-core"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "external", "alpha-core"),
        };

        foreach (string candidate in candidateRoots)
        {
            string resolved = Path.GetFullPath(candidate);
            if (Directory.Exists(resolved)
                && File.Exists(Path.Combine(resolved, "etc", "databases", "dbc", "dbc.sql"))
                && File.Exists(Path.Combine(resolved, "etc", "databases", "world", "world.sql")))
            {
                return resolved;
            }
        }

        return null;
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
