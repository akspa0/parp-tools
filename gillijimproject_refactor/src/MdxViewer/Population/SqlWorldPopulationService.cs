using MdxViewer.Catalog;

namespace MdxViewer.Population;

public sealed class SqlWorldPopulationService : IDisposable
{
    private readonly string _alphaCoreRoot;
    private AlphaCoreDbReader? _reader;
    private List<AssetCatalogEntry>? _cachedEntries;

    public SqlWorldPopulationService(string alphaCoreRoot)
    {
        _alphaCoreRoot = alphaCoreRoot;
    }

    public (bool success, string message) Validate()
    {
        _reader ??= new AlphaCoreDbReader(_alphaCoreRoot);
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

        _reader ??= new AlphaCoreDbReader(_alphaCoreRoot);
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

    public void Dispose()
    {
        _reader?.Dispose();
        _reader = null;
        _cachedEntries = null;
    }
}
