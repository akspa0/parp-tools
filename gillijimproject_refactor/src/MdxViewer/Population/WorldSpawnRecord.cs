using System.Numerics;

namespace MdxViewer.Population;

public enum WorldSpawnType
{
    Creature,
    GameObject
}

public sealed class WorldSpawnRecord
{
    public int EntryId { get; init; }
    public int SpawnId { get; init; }
    public int MapId { get; init; }
    public WorldSpawnType SpawnType { get; init; }

    public string Name { get; init; } = "";
    public string? Subname { get; init; }
    public string? ModelPath { get; init; }

    public float Scale { get; init; } = 1.0f;
    public float DisplayScale { get; init; } = 1.0f;
    public float EffectiveScale { get; init; } = 1.0f;

    public Vector3 PositionWow { get; init; }
    public float OrientationWowRadians { get; init; }

    public int Faction { get; init; }
    public int NpcFlags { get; init; }
    public int GameObjectType { get; init; }
}
