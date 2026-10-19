using System.Numerics;
using System.Text.Json;
using WowViewer.Core.Population;
using WoWViewer.Logging;
using WoWViewer.Population;

namespace WoWViewer.Workbench.Services.Dialogue;

/// <summary>
/// Manages world dialogue, NPC proximity-triggered ambient barks, and server-side quest metadata.
/// Enforces strict version isolation (0.5.3 Alpha-Core only, 3.3.5a TrinityCore only).
/// </summary>
public sealed class WorldDialogueService
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private EntityDialogueFlavorData? _flavorData;
    private string _loadedVersion = string.Empty;
    private readonly Dictionary<int, DateTime> _creatureLastBarkTime = new();
    private DateTime _lastGlobalBarkTime = DateTime.MinValue;

    public bool BarksEnabled { get; set; } = true;
    public float BarkDistanceYards { get; set; } = 40.0f;
    public float PerCreatureCooldownSeconds { get; set; } = 25.0f;
    public float GlobalBarkIntervalSeconds { get; set; } = 3.0f;

    public string LoadedVersion => _loadedVersion;
    public bool HasFlavorData => _flavorData != null;
    public int BarkCount => _flavorData?.CreatureBarks.Count ?? 0;
    public int QuestCount => _flavorData?.Quests.Count ?? 0;

    /// <summary>
    /// Loads the flavor dataset for the given version string ("3.3.5a" or "0.5.3").
    /// </summary>
    public async Task LoadFlavorDataAsync(string version, string datasetRoot, CancellationToken cancellationToken = default)
    {
        if (string.Equals(_loadedVersion, version, StringComparison.OrdinalIgnoreCase) && _flavorData != null)
            return;

        string flavorFile = Path.Combine(datasetRoot, "dialogue_flavor.json");
        if (!File.Exists(flavorFile))
        {
            // Probe default directory structure under data/population/{version}
            string fallback = Path.Combine(AppContext.BaseDirectory, "data", "population", version, "dialogue_flavor.json");
            if (File.Exists(fallback))
                flavorFile = fallback;
            else
            {
                string devFallback = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "data", "population", version, "dialogue_flavor.json"));
                if (File.Exists(devFallback))
                    flavorFile = devFallback;
            }
        }

        if (!File.Exists(flavorFile))
        {
            ViewerLog.Info(ViewerLog.Category.Terrain, $"[Dialogue] Flavor dataset not found for {version} at {flavorFile}");
            return;
        }

        try
        {
            await using var stream = File.OpenRead(flavorFile);
            var data = await JsonSerializer.DeserializeAsync<EntityDialogueFlavorData>(stream, JsonOpts, cancellationToken);
            if (data != null)
            {
                _flavorData = data;
                _loadedVersion = version;
                _creatureLastBarkTime.Clear();
                ViewerLog.Info(ViewerLog.Category.Terrain,
                    $"[Dialogue] Loaded {data.Version} flavor: {data.CreatureBarks.Count} creature barks, {data.Quests.Count} quests, {data.CreatureGossip.Count} gossip entries");
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain, $"[Dialogue] Error loading flavor dataset for {version}: {ex.Message}");
        }
    }

    /// <summary>
    /// Updates proximity checks against nearby NPC spawns and dispatches ambient barks to the chat service.
    /// </summary>
    public void UpdateProximityBarks(
        Vector3 cameraPos,
        IReadOnlyList<WorldSpawnRecord>? activeSpawns,
        FakeChatWindowService chatWindow)
    {
        if (!BarksEnabled || _flavorData == null || activeSpawns == null || activeSpawns.Count == 0)
            return;

        DateTime now = DateTime.UtcNow;
        if ((now - _lastGlobalBarkTime).TotalSeconds < GlobalBarkIntervalSeconds)
            return;

        float maxDistSq = BarkDistanceYards * BarkDistanceYards;

        foreach (var spawn in activeSpawns)
        {
            if (spawn.SpawnType != WorldSpawnType.Creature)
                continue;

            // Check if creature has registered barks or gossip
            bool hasBarks = _flavorData.CreatureBarks.TryGetValue(spawn.EntryId, out var barks) && barks.Count > 0;
            bool hasGossip = _flavorData.CreatureGossip.TryGetValue(spawn.EntryId, out var gossip) && gossip.Count > 0;

            if (!hasBarks && !hasGossip)
                continue;

            // Check per-creature cooldown
            if (_creatureLastBarkTime.TryGetValue(spawn.SpawnId, out var lastTime))
            {
                if ((now - lastTime).TotalSeconds < PerCreatureCooldownSeconds)
                    continue;
            }

            // Proximity check in renderer coordinates
            var pos = SqlSpawnCoordinateConverter.ToRendererPosition(spawn.PositionWow);
            float dx = pos.X - cameraPos.X;
            float dy = pos.Y - cameraPos.Y;
            float dz = pos.Z - cameraPos.Z;
            float distSq = dx * dx + dy * dy + dz * dz;

            if (distSq > maxDistSq)
                continue;

            // Pick line
            string lineText;
            DialogueChatType chatType;

            if (hasBarks)
            {
                var bark = barks![Random.Shared.Next(barks.Count)];
                lineText = bark.Text;
                chatType = bark.Type;
            }
            else
            {
                lineText = gossip![Random.Shared.Next(gossip.Count)];
                chatType = DialogueChatType.Say;
            }

            // Dispatch bark
            chatWindow.PostNpcBark(spawn.Name, lineText, chatType);

            _lastGlobalBarkTime = now;
            _creatureLastBarkTime[spawn.SpawnId] = now;
            break; // One bark per update cycle
        }
    }

    /// <summary>
    /// Returns quests offered by the given creature entry ID.
    /// </summary>
    public IReadOnlyList<QuestTemplateRecord> GetQuestsForCreature(int entryId)
    {
        if (_flavorData == null || !_flavorData.CreatureQuestStarters.TryGetValue(entryId, out var questIds))
            return Array.Empty<QuestTemplateRecord>();

        var list = new List<QuestTemplateRecord>(questIds.Count);
        foreach (int qid in questIds)
        {
            if (_flavorData.Quests.TryGetValue(qid, out var quest))
                list.Add(quest);
        }
        return list;
    }

    /// <summary>
    /// Returns quests offered by the given gameobject entry ID.
    /// </summary>
    public IReadOnlyList<QuestTemplateRecord> GetQuestsForGameObject(int entryId)
    {
        if (_flavorData == null || !_flavorData.GameObjectQuestStarters.TryGetValue(entryId, out var questIds))
            return Array.Empty<QuestTemplateRecord>();

        var list = new List<QuestTemplateRecord>(questIds.Count);
        foreach (int qid in questIds)
        {
            if (_flavorData.Quests.TryGetValue(qid, out var quest))
                list.Add(quest);
        }
        return list;
    }

    /// <summary>
    /// Returns gossip greeting text options for the given creature entry ID.
    /// </summary>
    public IReadOnlyList<string> GetGossipForCreature(int entryId)
    {
        if (_flavorData != null && _flavorData.CreatureGossip.TryGetValue(entryId, out var gossip))
            return gossip;
        return Array.Empty<string>();
    }

    /// <summary>
    /// Returns spoken bark lines for the given creature entry ID.
    /// </summary>
    public IReadOnlyList<DialogueTextRecord> GetBarksForCreature(int entryId)
    {
        if (_flavorData != null && _flavorData.CreatureBarks.TryGetValue(entryId, out var barks))
            return barks;
        return Array.Empty<DialogueTextRecord>();
    }
}
