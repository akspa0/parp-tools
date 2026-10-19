using System.Numerics;

namespace WowViewer.Core.Population;

public enum PopulationSpawnType
{
    Creature = 0,
    GameObject = 1
}

/// <summary>
/// Represents a discrete world spawn record for a creature or gameobject,
/// decoupled from database schemas and ready for high-speed scene instantiation.
/// </summary>
public sealed class PopulationRecord
{
    public int SpawnId { get; init; }
    public int EntryId { get; init; }
    public int MapId { get; init; }
    public PopulationSpawnType SpawnType { get; init; }

    public string Name { get; init; } = string.Empty;
    public string? Subname { get; init; }
    public string? ModelPath { get; init; }

    public float Scale { get; init; } = 1.0f;
    public float DisplayScale { get; init; } = 1.0f;
    public float EffectiveScale { get; init; } = 1.0f;

    public Vector3 Position { get; init; }
    public float Orientation { get; init; }
    public Quaternion? Rotation { get; init; }

    public int Faction { get; init; }
    public int Flags { get; init; }
    public int GameObjectType { get; init; }

    /// <summary>
    /// Indicates whether this spawn was transposed from a prototype/corner development tile
    /// (e.g. Alpha-Core Map 0 corner test fixtures transposed into Map 451).
    /// </summary>
    public bool IsTransposed { get; init; }

    /// <summary>
    /// Lineage or provenance note explaining original placement or transposition.
    /// </summary>
    public string? SourceNote { get; init; }
}
