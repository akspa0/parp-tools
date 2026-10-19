using System.Text.Json;
using WoWViewer.Catalog;
using WowViewer.Core.Population;

namespace WoWViewer.Population;

public sealed class SqlWorldPopulationService : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        IncludeFields = true
    };

    private readonly string _rootPath;
    private readonly bool _isJsonDataset;
    private readonly WorldEventRegistry _eventRegistry = new();
    private AlphaCoreDbReader? _reader;
    private List<AssetCatalogEntry>? _cachedEntries;
    private readonly Dictionary<int, List<WorldSpawnRecord>> _cachedJsonMapSpawns = new();

    public WorldEventRegistry EventRegistry => _eventRegistry;

    public SqlWorldPopulationService(string rootPath)
    {
        _rootPath = rootPath;
        _isJsonDataset = DetectIsJsonDataset(rootPath);
    }

    public static bool DetectIsJsonDataset(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !Directory.Exists(path))
            return false;

        string manifest = Path.Combine(path, "manifest.json");
        if (File.Exists(manifest))
            return true;

        return Directory.EnumerateFiles(path, "map_*.json").Any();
    }

    public (bool success, string message) Validate()
    {
        if (_isJsonDataset)
        {
            if (!Directory.Exists(_rootPath))
                return (false, $"Population dataset directory not found: {_rootPath}");

            int count = Directory.EnumerateFiles(_rootPath, "map_*.json").Count();
            return (true, $"Pre-extracted population dataset found ({count} map partitions).");
        }

        _reader ??= new AlphaCoreDbReader(_rootPath);
        return _reader.Validate();
    }

    public async Task<IReadOnlyList<WorldSpawnRecord>> LoadMapSpawnsAsync(
        int mapId,
        int maxSpawns,
        bool includeCreatures,
        bool includeGameObjects,
        CancellationToken cancellationToken = default)
    {
        if (mapId < 0)
            return Array.Empty<WorldSpawnRecord>();

        if (!includeCreatures && !includeGameObjects)
            return Array.Empty<WorldSpawnRecord>();

        // Fast path: Pre-extracted JSON partition dataset (< 10 ms load time)
        if (_isJsonDataset)
        {
            return await LoadFromJsonPartitionAsync(mapId, maxSpawns, includeCreatures, includeGameObjects, cancellationToken);
        }

        // Alpha-Core raw SQL dump reader fallback
        _reader ??= new AlphaCoreDbReader(_rootPath);
        _cachedEntries ??= await _reader.LoadAllAsync();

        var result = new List<WorldSpawnRecord>(Math.Max(128, Math.Min(maxSpawns > 0 ? maxSpawns : 8192, 8192)));

        foreach (var entry in _cachedEntries)
        {
            cancellationToken.ThrowIfCancellationRequested();

            bool isCreature = entry.Type == AssetType.Creature;
            bool isGameObject = entry.Type == AssetType.GameObject;

            if (isCreature && !includeCreatures) continue;
            if (isGameObject && !includeGameObjects) continue;
            if (entry.Spawns.Count == 0) continue;

            foreach (var spawn in entry.Spawns)
            {
                if (spawn.MapId != mapId)
                    continue;

                result.Add(new WorldSpawnRecord
                {
                    EntryId = entry.EntryId,
                    SpawnId = spawn.SpawnId,
                    MapId = spawn.MapId,
                    SpawnType = isCreature ? WorldSpawnType.Creature : WorldSpawnType.GameObject,
                    Name = entry.Name,
                    Subname = entry.Subname,
                    ModelPath = entry.ModelPath,
                    Scale = entry.Scale,
                    DisplayScale = entry.DisplayScale,
                    EffectiveScale = entry.EffectiveScale,
                    PositionWow = new System.Numerics.Vector3(spawn.X, spawn.Y, spawn.Z),
                    OrientationWowRadians = spawn.Orientation,
                    Faction = entry.Faction,
                    NpcFlags = entry.NpcFlags,
                    GameObjectType = entry.GameObjectType
                });

                if (maxSpawns > 0 && result.Count >= maxSpawns)
                    return result;
            }
        }

        return result;
    }

    private async Task<IReadOnlyList<WorldSpawnRecord>> LoadFromJsonPartitionAsync(
        int mapId,
        int maxSpawns,
        bool includeCreatures,
        bool includeGameObjects,
        CancellationToken cancellationToken)
    {
        if (!_cachedJsonMapSpawns.TryGetValue(mapId, out var allMapSpawns))
        {
            string mapFile = Path.Combine(_rootPath, $"map_{mapId}.json");
            if (!File.Exists(mapFile))
            {
                // Fallback to other datasets if not found in active version directory
                string parentDir = Path.GetDirectoryName(_rootPath) ?? "";
                string fallback335 = Path.Combine(parentDir, "3.3.5a", $"map_{mapId}.json");
                string fallback053 = Path.Combine(parentDir, "0.5.3", $"map_{mapId}.json");
                if (File.Exists(fallback335))
                    mapFile = fallback335;
                else if (File.Exists(fallback053))
                    mapFile = fallback053;
                else
                    return Array.Empty<WorldSpawnRecord>();
            }

            await using var stream = File.OpenRead(mapFile);
            var mapData = await JsonSerializer.DeserializeAsync<PopulationMapData>(stream, JsonOptions, cancellationToken);
            if (mapData == null || mapData.Spawns.Count == 0)
                return Array.Empty<WorldSpawnRecord>();

            if (!_eventRegistry.IsLoaded)
            {
                await _eventRegistry.LoadAsync(_rootPath, cancellationToken).ConfigureAwait(false);
            }

            allMapSpawns = new List<WorldSpawnRecord>(mapData.Spawns.Count);
            foreach (var s in mapData.Spawns)
            {
                var spawnType = s.SpawnType == PopulationSpawnType.Creature ? WorldSpawnType.Creature : WorldSpawnType.GameObject;
                int evId = 0;
                string? evName = null;
                if (_eventRegistry.TryGetSpawnEvent(s.SpawnId, spawnType, out int foundEvId, out string? foundEvName))
                {
                    evId = foundEvId;
                    evName = foundEvName;
                }

                allMapSpawns.Add(new WorldSpawnRecord
                {
                    EntryId = s.EntryId,
                    SpawnId = s.SpawnId,
                    MapId = s.MapId,
                    SpawnType = spawnType,
                    Name = s.Name,
                    Subname = s.Subname,
                    ModelPath = s.ModelPath,
                    Scale = s.Scale,
                    DisplayScale = s.DisplayScale,
                    EffectiveScale = s.EffectiveScale,
                    PositionWow = s.Position,
                    OrientationWowRadians = s.Orientation,
                    Faction = s.Faction,
                    NpcFlags = s.Flags,
                    GameObjectType = s.GameObjectType,
                    EventId = evId,
                    EventName = evName
                });
            }

            _cachedJsonMapSpawns[mapId] = allMapSpawns;
        }

        var filtered = new List<WorldSpawnRecord>(maxSpawns > 0 ? Math.Min(maxSpawns, allMapSpawns.Count) : allMapSpawns.Count);
        foreach (var s in allMapSpawns)
        {
            if (s.SpawnType == WorldSpawnType.Creature && !includeCreatures) continue;
            if (s.SpawnType == WorldSpawnType.GameObject && !includeGameObjects) continue;

            filtered.Add(s);
            if (maxSpawns > 0 && filtered.Count >= maxSpawns)
                break;
        }

        return filtered;
    }

    public void Dispose()
    {
        _reader?.Dispose();
        _reader = null;
        _cachedEntries = null;
        _cachedJsonMapSpawns.Clear();
    }
}
