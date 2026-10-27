using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Dbc;

namespace WowViewer.Core.IO.Population;

/// <summary>
/// Resolves creature and gameobject display IDs into canonical model paths and scales
/// using CreatureDisplayInfo, CreatureModelData, and GameObjectDisplayInfo tables,
/// matched accurately against client MPQ internal listfiles.
/// </summary>
public sealed class DbcModelResolver : IDbcModelResolver
{
    private readonly Dictionary<int, (string modelPath, float scale, uint? fileDataId, string[]? customTextures)> _creatureDisplays = new();
    private readonly Dictionary<int, (string modelPath, float scale, uint? fileDataId)> _gameObjectDisplays = new();
    private readonly Dictionary<int, string> _mapNames = new();

    private readonly HashSet<string> _availableFiles = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<string>> _filesByBaseName = new(StringComparer.OrdinalIgnoreCase);

    public int CreatureDisplayCount => _creatureDisplays.Count;
    public int GameObjectDisplayCount => _gameObjectDisplays.Count;
    public int MapCount => _mapNames.Count;
    public int AvailableFileCount => _availableFiles.Count;

    public string? PreferredClientVersion { get; set; }

    public void SetAvailableFiles(IEnumerable<string> files)
    {
        ArgumentNullException.ThrowIfNull(files);
        _availableFiles.Clear();
        _filesByBaseName.Clear();

        foreach (var file in files)
        {
            if (string.IsNullOrWhiteSpace(file))
                continue;

            string normalized = file.Trim().Replace('/', '\\');
            _availableFiles.Add(normalized);

            string baseName = Path.GetFileNameWithoutExtension(normalized);
            if (!string.IsNullOrEmpty(baseName))
            {
                if (!_filesByBaseName.TryGetValue(baseName, out var list))
                {
                    list = new List<string>();
                    _filesByBaseName[baseName] = list;
                }
                list.Add(normalized);
            }
        }
    }

    public string GetMapName(int mapId)
    {
        if (_mapNames.TryGetValue(mapId, out string? name))
            return name;

        return PopulationMapNames.GetMapName(mapId);
    }

    public void RegisterMap(int mapId, string mapName)
    {
        if (mapId >= 0 && !string.IsNullOrWhiteSpace(mapName))
            _mapNames[mapId] = mapName;
    }

    public bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale)
    {
        return TryResolveCreature(displayId, out modelPath, out displayScale, out _, out _);
    }

    public bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale, out uint? fileDataId)
    {
        return TryResolveCreature(displayId, out modelPath, out displayScale, out fileDataId, out _);
    }

    public bool TryResolveCreature(int displayId, out string? modelPath, out float displayScale, out uint? fileDataId, out string[]? customTextures)
    {
        if (_creatureDisplays.TryGetValue(displayId, out var entry))
        {
            modelPath = entry.modelPath;
            displayScale = entry.scale;
            fileDataId = entry.fileDataId;
            customTextures = entry.customTextures;
            return true;
        }

        modelPath = null;
        displayScale = 1.0f;
        fileDataId = null;
        customTextures = null;
        return false;
    }

    public bool TryResolveGameObject(int displayId, out string? modelPath, out float displayScale)
    {
        return TryResolveGameObject(displayId, out modelPath, out displayScale, out _);
    }

    public bool TryResolveGameObject(int displayId, out string? modelPath, out float displayScale, out uint? fileDataId)
    {
        if (_gameObjectDisplays.TryGetValue(displayId, out var entry))
        {
            modelPath = entry.modelPath;
            displayScale = entry.scale;
            fileDataId = entry.fileDataId;
            return true;
        }

        modelPath = null;
        displayScale = 1.0f;
        fileDataId = null;
        return false;
    }

    public void RegisterCreature(int displayId, string modelPath, float scale = 1.0f, uint? fileDataId = null, string[]? customTextures = null)
    {
        if (displayId > 0 && !string.IsNullOrWhiteSpace(modelPath))
        {
            string resolved = ResolveModelPath(modelPath, isCreature: true);
            _creatureDisplays[displayId] = (resolved, scale > 0 ? scale : 1.0f, fileDataId, customTextures);
        }
    }

    public void RegisterGameObject(int displayId, string modelPath, float scale = 1.0f, uint? fileDataId = null)
    {
        if (displayId > 0 && !string.IsNullOrWhiteSpace(modelPath))
        {
            string resolved = ResolveModelPath(modelPath, isCreature: false);
            _gameObjectDisplays[displayId] = (resolved, scale > 0 ? scale : 1.0f, fileDataId);
        }
    }

    public string ResolveModelPath(string rawPath, bool isCreature)
    {
        string p = rawPath.Trim().Replace('/', '\\');

        // 1. If internal listfile files are available, match against them:
        if (_availableFiles.Count > 0)
        {
            if (_availableFiles.Contains(p))
                return p;

            // Try swapping extension: .mdx <-> .m2 <-> .mdl
            foreach (var alt in GetAlternatePaths(p))
            {
                if (_availableFiles.Contains(alt))
                    return alt;
            }

            // If not found, look up by base filename in catalog
            string baseName = Path.GetFileNameWithoutExtension(p);
            if (_filesByBaseName.TryGetValue(baseName, out var matches) && matches.Count > 0)
            {
                string prefix = isCreature ? "Creature\\" : "World\\";
                var preferred = matches.FirstOrDefault(m => m.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    ?? matches[0];
                return preferred;
            }
        }

        // 2. Fallback normalization when listfile is not present or model not indexed
        bool hasListfile = _availableFiles.Count > 0;
        bool isVersion335 = string.Equals(PreferredClientVersion, "3.3.5a", StringComparison.OrdinalIgnoreCase);
        bool isVersion053 = string.Equals(PreferredClientVersion, "0.5.3", StringComparison.OrdinalIgnoreCase);

        bool preferM2ForCreature = isVersion335 || (!isVersion053 && (!hasListfile || HasMostlyM2Files()));
        bool preferM2ForGameObject = isVersion335 || (!isVersion053 && (hasListfile && HasMostlyM2Files()));

        return isCreature
            ? NormalizeCreaturePathFallback(p, preferM2ForCreature)
            : NormalizeGameObjectPathFallback(p, preferM2ForGameObject);
    }

    private bool HasMostlyM2Files()
    {
        if (_availableFiles.Count == 0) return true; // Default to modern/3.3.5
        int m2Count = 0;
        int mdxCount = 0;
        foreach (var f in _availableFiles)
        {
            if (f.EndsWith(".m2", StringComparison.OrdinalIgnoreCase)) m2Count++;
            else if (f.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase)) mdxCount++;
        }
        return m2Count >= mdxCount;
    }

    private static IEnumerable<string> GetAlternatePaths(string path)
    {
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".m2";
            yield return path[..^4] + ".mdl";
        }
        else if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^3] + ".mdx";
            yield return path[..^3] + ".mdl";
        }
        else if (path.EndsWith(".mdl", StringComparison.OrdinalIgnoreCase))
        {
            yield return path[..^4] + ".m2";
            yield return path[..^4] + ".mdx";
        }
    }

    private static string NormalizeCreaturePathFallback(string path, bool preferM2)
    {
        string p = path.Trim().Replace('/', '\\');
        if (preferM2)
        {
            if (p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                p = p[..^4] + ".m2";
            else if (!p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                p += ".m2";
        }
        else
        {
            if (p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                p = p[..^3] + ".mdx";
            else if (!p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                p += ".mdx";
        }

        if (!p.StartsWith("Creature\\", StringComparison.OrdinalIgnoreCase) &&
            !p.StartsWith("Character\\", StringComparison.OrdinalIgnoreCase) &&
            !p.Contains('\\'))
        {
            string name = Path.GetFileNameWithoutExtension(p);
            p = $"Creature\\{name}\\{p}";
        }

        return p;
    }

    private static string NormalizeGameObjectPathFallback(string path, bool preferM2)
    {
        string p = path.Trim().Replace('/', '\\');
        if (p.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
            return p;

        if (preferM2)
        {
            if (p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                p = p[..^4] + ".m2";
            else if (!p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                p += ".m2";
        }
        else
        {
            if (p.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                p = p[..^3] + ".mdx";
            else if (!p.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                p += ".mdx";
        }

        return p;
    }

    /// <summary>
    /// Merges entries from another resolver into this resolver.
    /// </summary>
    public void MergeFrom(DbcModelResolver other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (_availableFiles.Count == 0 && other._availableFiles.Count > 0)
        {
            SetAvailableFiles(other._availableFiles);
        }

        foreach (var (id, entry) in other._creatureDisplays)
        {
            _creatureDisplays[id] = entry;
        }
        foreach (var (id, entry) in other._gameObjectDisplays)
        {
            _gameObjectDisplays[id] = entry;
        }
        foreach (var (id, name) in other._mapNames)
        {
            _mapNames[id] = name;
        }
    }

    /// <summary>
    /// Loads DBC files from a path, which can be a directory (loose DBCs or MPQs) or a single MPQ archive.
    /// </summary>
    public static DbcModelResolver LoadFromPath(string path, string? clientVersion = null)
    {
        if (string.IsNullOrWhiteSpace(path))
            return new DbcModelResolver();

        if (File.Exists(path) && path.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                using var catalog = new MpqArchiveCatalog();
                catalog.LoadArchives([Path.GetDirectoryName(path) ?? path]);
                var resolver = LoadFromArchive(catalog);
                if (clientVersion != null) resolver.PreferredClientVersion = clientVersion;
                return resolver;
            }
            catch
            {
                return new DbcModelResolver();
            }
        }

        if (Directory.Exists(path))
        {
            var resolver = LoadFromDirectory(path);
            if (clientVersion != null) resolver.PreferredClientVersion = clientVersion;
            return resolver;
        }

        return new DbcModelResolver();
    }

    /// <summary>
    /// Loads DBC files from loose files in a directory (e.g. DBFilesClient/) or MPQs within the directory.
    /// </summary>
    public static DbcModelResolver LoadFromDirectory(string directory)
    {
        var resolver = new DbcModelResolver();
        if (!Directory.Exists(directory))
            return resolver;

        string cdi = Path.Combine(directory, "CreatureDisplayInfo.dbc");
        string cmd = Path.Combine(directory, "CreatureModelData.dbc");
        string gdi = Path.Combine(directory, "GameObjectDisplayInfo.dbc");
        string map = Path.Combine(directory, "Map.dbc");

        if (!File.Exists(cdi))
            cdi = Path.Combine(directory, "DBFilesClient", "CreatureDisplayInfo.dbc");
        if (!File.Exists(cmd))
            cmd = Path.Combine(directory, "DBFilesClient", "CreatureModelData.dbc");
        if (!File.Exists(gdi))
            gdi = Path.Combine(directory, "DBFilesClient", "GameObjectDisplayInfo.dbc");
        if (!File.Exists(map))
            map = Path.Combine(directory, "DBFilesClient", "Map.dbc");

        string cdie = Path.Combine(directory, "CreatureDisplayInfoExtra.dbc");
        if (!File.Exists(cdie))
            cdie = Path.Combine(directory, "DBFilesClient", "CreatureDisplayInfoExtra.dbc");

        byte[]? cdiBytes = File.Exists(cdi) ? File.ReadAllBytes(cdi) : null;
        byte[]? cdieBytes = File.Exists(cdie) ? File.ReadAllBytes(cdie) : null;
        byte[]? cmdBytes = File.Exists(cmd) ? File.ReadAllBytes(cmd) : null;
        byte[]? gdiBytes = File.Exists(gdi) ? File.ReadAllBytes(gdi) : null;
        byte[]? mapBytes = File.Exists(map) ? File.ReadAllBytes(map) : null;

        // If all 3 essential display DBCs exist loose, load them directly
        if (cdiBytes != null && cmdBytes != null && gdiBytes != null)
        {
            resolver.LoadData(cdiBytes, cmdBytes, gdiBytes, mapBytes, cdieBytes);
            return resolver;
        }

        // If loose display DBCs are missing or incomplete, search for MPQ archives
        try
        {
            using var catalog = new MpqArchiveCatalog();
            catalog.LoadArchives([directory]);
            var archiveResolver = LoadFromArchive(catalog);
            if (archiveResolver.CreatureDisplayCount > 0 || archiveResolver.GameObjectDisplayCount > 0)
            {
                if (mapBytes != null && archiveResolver.MapCount == 0)
                {
                    archiveResolver.LoadData(null, null, null, mapBytes, null);
                }
                return archiveResolver;
            }
        }
        catch
        {
        }

        // Fallback to partial loose data if MPQ search yielded nothing
        if (cdiBytes != null || cmdBytes != null || gdiBytes != null || mapBytes != null || cdieBytes != null)
        {
            resolver.LoadData(cdiBytes, cmdBytes, gdiBytes, mapBytes, cdieBytes);
        }

        return resolver;
    }

    /// <summary>
    /// Loads DBC files from an MPQ / CASC archive catalog.
    /// </summary>
    public static DbcModelResolver LoadFromArchive(IArchiveCatalog catalog)
    {
        var resolver = new DbcModelResolver();
        byte[]? cdiBytes = catalog.ReadFile("DBFilesClient\\CreatureDisplayInfo.dbc")
            ?? catalog.ReadFile("DBFilesClient/CreatureDisplayInfo.dbc")
            ?? catalog.ReadFile("CreatureDisplayInfo.dbc");
        byte[]? cdieBytes = catalog.ReadFile("DBFilesClient\\CreatureDisplayInfoExtra.dbc")
            ?? catalog.ReadFile("DBFilesClient/CreatureDisplayInfoExtra.dbc")
            ?? catalog.ReadFile("CreatureDisplayInfoExtra.dbc");
        byte[]? cmdBytes = catalog.ReadFile("DBFilesClient\\CreatureModelData.dbc")
            ?? catalog.ReadFile("DBFilesClient/CreatureModelData.dbc")
            ?? catalog.ReadFile("CreatureModelData.dbc");
        byte[]? gdiBytes = catalog.ReadFile("DBFilesClient\\GameObjectDisplayInfo.dbc")
            ?? catalog.ReadFile("DBFilesClient/GameObjectDisplayInfo.dbc")
            ?? catalog.ReadFile("GameObjectDisplayInfo.dbc");
        byte[]? mapBytes = catalog.ReadFile("DBFilesClient\\Map.dbc")
            ?? catalog.ReadFile("DBFilesClient/Map.dbc")
            ?? catalog.ReadFile("Map.dbc");

        try
        {
            var internalFiles = catalog.ExtractInternalListfiles();
            if (internalFiles.Count > 0)
            {
                resolver.SetAvailableFiles(internalFiles);
            }
        }
        catch
        {
        }

        resolver.LoadData(cdiBytes, cmdBytes, gdiBytes, mapBytes, cdieBytes);
        return resolver;
    }

    public void LoadData(byte[]? cdiBytes, byte[]? cmdBytes, byte[]? gdiBytes, byte[]? mapBytes = null, byte[]? cdieBytes = null)
    {
        // 1. CreatureModelData: ID(0) -> ModelPath(2)
        var modelDataPaths = new Dictionary<int, string>();
        if (cmdBytes != null && cmdBytes.Length > 0)
        {
            var cmd = DbcReader.Load(cmdBytes);
            for (int r = 0; r < cmd.Rows.Count; r++)
            {
                int id = cmd.GetInt(r, 0);
                // In standard 3.3.5a, ModelPath is field 2
                uint strOffset = cmd.GetUInt(r, 2);
                string path = cmd.GetString(strOffset);
                if (string.IsNullOrWhiteSpace(path) && cmd.Header.RecordSize >= 8)
                {
                    // Fallback check field 1 if 2 was 0
                    uint altOffset = cmd.GetUInt(r, 1);
                    path = cmd.GetString(altOffset);
                }

                if (!string.IsNullOrWhiteSpace(path))
                    modelDataPaths[id] = path;
            }
        }

        // 1b. CreatureDisplayInfoExtra: ID(0) -> BakeName(last field / field 18 in 0.5.3, field 18 in 3.3.5)
        var extraBakedTextures = new Dictionary<int, string>();
        if (cdieBytes != null && cdieBytes.Length > 0)
        {
            var cdie = DbcReader.Load(cdieBytes);
            int lastField = (int)(cdie.Header.RecordSize / 4) - 1;
            for (int r = 0; r < cdie.Rows.Count; r++)
            {
                int extraId = cdie.GetInt(r, 0);
                string bakeName = string.Empty;
                if (lastField > 0)
                {
                    uint strOffset = cdie.GetUInt(r, lastField);
                    bakeName = cdie.GetString(strOffset);
                }

                if (!string.IsNullOrWhiteSpace(bakeName))
                {
                    string bakePath = bakeName.Trim().Replace('/', '\\');
                    if (!bakePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
                        bakePath += ".blp";
                    if (!bakePath.StartsWith("textures\\", StringComparison.OrdinalIgnoreCase))
                        bakePath = Path.Combine("textures\\BakedNpcTextures", bakePath);

                    extraBakedTextures[extraId] = bakePath;
                }
            }
        }

        // 2. CreatureDisplayInfo: ID(0) -> ModelID(1), ExtendedDisplayInfoID(3), Scale(4), TextureVariation[3](6, 7, 8)
        if (cdiBytes != null && cdiBytes.Length > 0)
        {
            var cdi = DbcReader.Load(cdiBytes);
            int fieldCount = (int)(cdi.Header.RecordSize / 4);
            for (int r = 0; r < cdi.Rows.Count; r++)
            {
                int id = cdi.GetInt(r, 0);
                int modelDataId = cdi.GetInt(r, 1);
                int extraId = fieldCount > 3 ? cdi.GetInt(r, 3) : 0;
                float scale = fieldCount > 4 ? cdi.GetFloat(r, 4) : 1.0f;
                if (scale <= 0) scale = 1.0f;

                string[]? customTextures = null;
                if (extraId > 0 && extraBakedTextures.TryGetValue(extraId, out string? bakedPath))
                {
                    customTextures = new[] { bakedPath };
                }
                else if (fieldCount >= 9)
                {
                    // TextureVariation 1, 2, 3
                    string t1 = cdi.GetString(cdi.GetUInt(r, 6));
                    string t2 = cdi.GetString(cdi.GetUInt(r, 7));
                    string t3 = cdi.GetString(cdi.GetUInt(r, 8));

                    if (!string.IsNullOrWhiteSpace(t1) || !string.IsNullOrWhiteSpace(t2) || !string.IsNullOrWhiteSpace(t3))
                    {
                        var variations = new List<string>();
                        if (!string.IsNullOrWhiteSpace(t1)) variations.Add(t1.Trim());
                        if (!string.IsNullOrWhiteSpace(t2)) variations.Add(t2.Trim());
                        if (!string.IsNullOrWhiteSpace(t3)) variations.Add(t3.Trim());
                        if (variations.Count > 0)
                            customTextures = variations.ToArray();
                    }
                }

                if (modelDataPaths.TryGetValue(modelDataId, out var path))
                {
                    RegisterCreature(id, path, scale, fileDataId: null, customTextures: customTextures);
                }
            }
        }

        // 3. GameObjectDisplayInfo: ID(0) -> ModelPath(1)
        if (gdiBytes != null && gdiBytes.Length > 0)
        {
            var gdi = DbcReader.Load(gdiBytes);
            for (int r = 0; r < gdi.Rows.Count; r++)
            {
                int id = gdi.GetInt(r, 0);
                uint strOffset = gdi.GetUInt(r, 1);
                string path = gdi.GetString(strOffset);
                if (!string.IsNullOrWhiteSpace(path))
                {
                    RegisterGameObject(id, path, 1.0f);
                }
            }
        }

        // 4. Map: ID(0) -> Directory(1), MapName_lang(5 in 3.3.5a, 4 in 0.5.3)
        if (mapBytes != null && mapBytes.Length > 0)
        {
            var map = DbcReader.Load(mapBytes);
            for (int r = 0; r < map.Rows.Count; r++)
            {
                int id = map.GetInt(r, 0);
                string dir = map.GetString(map.GetUInt(r, 1));
                string name = dir;
                if (map.Header.FieldCount > 5)
                {
                    string localized = map.GetString(map.GetUInt(r, 5));
                    if (!string.IsNullOrWhiteSpace(localized))
                        name = localized;
                }
                else if (map.Header.FieldCount > 4)
                {
                    string localized = map.GetString(map.GetUInt(r, 4));
                    if (!string.IsNullOrWhiteSpace(localized))
                        name = localized;
                }
                if (id == 451) name = "development";
                if (!string.IsNullOrWhiteSpace(name))
                    RegisterMap(id, name);
            }
        }
    }

    public static string NormalizeCreaturePath(string path)
    {
        return NormalizeCreaturePathFallback(path, preferM2: true);
    }

    public static string NormalizeGameObjectPath(string path)
    {
        return NormalizeGameObjectPathFallback(path, preferM2: true);
    }
}
