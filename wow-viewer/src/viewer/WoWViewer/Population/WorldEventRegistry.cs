using System.Text.Json;
using WoWViewer.Logging;

namespace WoWViewer.Population;

public sealed record WorldEventDefinition(int EventId, int HolidayId, string Name)
{
    public string DisplayName => HolidayId > 0 ? $"{Name} (Holiday {HolidayId})" : $"{Name} (Event {EventId})";
}

/// <summary>
/// Registry for game events and holiday spawn associations sourced from TrinityCore's
/// game_event, game_event_gameobject, and game_event_creature tables.
/// Enables isolating holiday decorations so they do not render on top of each other.
/// </summary>
public sealed class WorldEventRegistry
{
    private sealed class RawEventData
    {
        public Dictionary<string, RawEventInfo>? Events { get; set; }
        public Dictionary<string, int>? GameObjectEvents { get; set; }
        public Dictionary<string, int>? CreatureEvents { get; set; }
    }

    private sealed class RawEventInfo
    {
        public int EventId { get; set; }
        public int HolidayId { get; set; }
        public string Name { get; set; } = "";
    }

    private readonly Dictionary<int, WorldEventDefinition> _eventsById = new();
    private readonly Dictionary<int, int> _gameObjectEvents = new();
    private readonly Dictionary<int, int> _creatureEvents = new();

    public int EventCount => _eventsById.Count;
    public int GameObjectMappingCount => _gameObjectEvents.Count;
    public int CreatureMappingCount => _creatureEvents.Count;

    public bool IsLoaded => _eventsById.Count > 0;
    public IReadOnlyList<WorldEventDefinition> Events => _eventsById.Values.OrderBy(e => e.Name).ToList();

    public async Task LoadAsync(string datasetDir, CancellationToken cancellationToken = default)
    {
        if (IsLoaded)
            return;

        string eventsFile = Path.Combine(datasetDir, "events.json");
        if (!File.Exists(eventsFile))
        {
            // Probe fallback paths
            string fallback = Path.Combine(AppContext.BaseDirectory, "data", "population", "3.3.5a", "events.json");
            if (File.Exists(fallback))
                eventsFile = fallback;
            else
            {
                string devFallback = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "population", "3.3.5a", "events.json"));
                if (File.Exists(devFallback))
                    eventsFile = devFallback;
            }
        }

        if (!File.Exists(eventsFile))
            return;

        try
        {
            await using var stream = File.OpenRead(eventsFile);
            var raw = await JsonSerializer.DeserializeAsync<RawEventData>(stream, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            }, cancellationToken).ConfigureAwait(false);

            if (raw?.Events != null)
            {
                foreach (var (k, v) in raw.Events)
                {
                    if (int.TryParse(k, out int eventId))
                    {
                        _eventsById[eventId] = new WorldEventDefinition(v.EventId, v.HolidayId, v.Name);
                    }
                }
            }

            if (raw?.GameObjectEvents != null)
            {
                foreach (var (k, v) in raw.GameObjectEvents)
                {
                    if (int.TryParse(k, out int guid))
                        _gameObjectEvents[guid] = v;
                }
            }

            if (raw?.CreatureEvents != null)
            {
                foreach (var (k, v) in raw.CreatureEvents)
                {
                    if (int.TryParse(k, out int guid))
                        _creatureEvents[guid] = v;
                }
            }

            ViewerLog.Info(ViewerLog.Category.Terrain,
                $"[WorldEvents] Loaded {_eventsById.Count} events, {_gameObjectEvents.Count} GO mappings, {_creatureEvents.Count} creature mappings from {eventsFile}");
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[WorldEvents] Failed to load events.json: {ex.Message}");
        }
    }

    public bool TryGetSpawnEvent(int spawnId, WorldSpawnType spawnType, out int eventId, out string? eventName)
    {
        eventId = spawnType == WorldSpawnType.GameObject
            ? GetEventIdForGameObject(spawnId)
            : GetEventIdForCreature(spawnId);

        if (eventId > 0 && _eventsById.TryGetValue(eventId, out var def))
        {
            eventName = def.Name;
            return true;
        }

        eventName = null;
        return eventId > 0;
    }

    public static WorldEventRegistry LoadFromDirectory(string datasetDir)
    {
        var registry = new WorldEventRegistry();
        string eventsFile = Path.Combine(datasetDir, "events.json");
        if (!File.Exists(eventsFile))
        {
            // Probe fallback paths
            string fallback = Path.Combine(AppContext.BaseDirectory, "data", "population", "3.3.5a", "events.json");
            if (File.Exists(fallback))
                eventsFile = fallback;
            else
            {
                string devFallback = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "population", "3.3.5a", "events.json"));
                if (File.Exists(devFallback))
                    eventsFile = devFallback;
            }
        }

        if (File.Exists(eventsFile))
        {
            try
            {
                string json = File.ReadAllText(eventsFile);
                var raw = JsonSerializer.Deserialize<RawEventData>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (raw?.Events != null)
                {
                    foreach (var (k, v) in raw.Events)
                    {
                        if (int.TryParse(k, out int eventId))
                        {
                            registry._eventsById[eventId] = new WorldEventDefinition(v.EventId, v.HolidayId, v.Name);
                        }
                    }
                }

                if (raw?.GameObjectEvents != null)
                {
                    foreach (var (k, v) in raw.GameObjectEvents)
                    {
                        if (int.TryParse(k, out int guid))
                            registry._gameObjectEvents[guid] = v;
                    }
                }

                if (raw?.CreatureEvents != null)
                {
                    foreach (var (k, v) in raw.CreatureEvents)
                    {
                        if (int.TryParse(k, out int guid))
                            registry._creatureEvents[guid] = v;
                    }
                }

                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[WorldEvents] Loaded {registry._eventsById.Count} events, {registry._gameObjectEvents.Count} GO mappings, {registry._creatureEvents.Count} creature mappings from {eventsFile}");
            }
            catch (Exception ex)
            {
                ViewerLog.Error(ViewerLog.Category.Terrain, $"[WorldEvents] Failed to load events.json: {ex.Message}");
            }
        }

        return registry;
    }

    public bool TryGetEventForGameObject(int spawnId, out WorldEventDefinition? ev)
    {
        if (_gameObjectEvents.TryGetValue(spawnId, out int eventId) && _eventsById.TryGetValue(eventId, out ev))
            return true;

        ev = null;
        return false;
    }

    public bool TryGetEventForCreature(int spawnId, out WorldEventDefinition? ev)
    {
        if (_creatureEvents.TryGetValue(spawnId, out int eventId) && _eventsById.TryGetValue(eventId, out ev))
            return true;

        ev = null;
        return false;
    }

    public int GetEventIdForGameObject(int spawnId) =>
        _gameObjectEvents.TryGetValue(spawnId, out int eventId) ? eventId : 0;

    public int GetEventIdForCreature(int spawnId) =>
        _creatureEvents.TryGetValue(spawnId, out int eventId) ? eventId : 0;

    public IReadOnlyList<WorldEventDefinition> GetKnownEvents() =>
        _eventsById.Values.OrderBy(e => e.Name).ToList();

    /// <summary>
    /// Checks whether a spawn should be shown under the selected event filter:
    /// - selectedEventId == -1: Show All (Everything rendered, legacy behavior)
    /// - selectedEventId == 0: Base World Only (No holiday/event spawns, clean world)
    /// - selectedEventId > 0: Base World + specific holiday spawns
    /// </summary>
    public bool IsSpawnActive(int spawnId, WorldSpawnType spawnType, int selectedEventId)
    {
        if (selectedEventId == -1)
            return true;

        int spawnEventId = spawnType == WorldSpawnType.GameObject
            ? GetEventIdForGameObject(spawnId)
            : GetEventIdForCreature(spawnId);

        if (selectedEventId == 0)
            return spawnEventId == 0;

        return spawnEventId == 0 || spawnEventId == selectedEventId;
    }
}
