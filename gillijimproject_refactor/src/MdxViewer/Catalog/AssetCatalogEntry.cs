using System.Numerics;

namespace MdxViewer.Catalog;

/// <summary>
/// Unified data model for an NPC or GameObject from the alpha-core database.
/// Contains resolved model paths, metadata, and spawn information.
/// </summary>
public class AssetCatalogEntry
{
    /// <summary>Template entry ID (creature_template.entry or gameobject_template.entry).</summary>
    public int EntryId { get; init; }

    /// <summary>Whether this is a creature (NPC) or a gameobject.</summary>
    public AssetType Type { get; init; }

    /// <summary>Display name from the template.</summary>
    public string Name { get; init; } = "";

    /// <summary>Subname/title (creatures only, e.g. "Innkeeper", "Blacksmith").</summary>
    public string? Subname { get; init; }

    /// <summary>Primary display ID from the template.</summary>
    public int DisplayId { get; init; }

    /// <summary>All display IDs (creatures can have up to 4).</summary>
    public int[] AllDisplayIds { get; init; } = Array.Empty<int>();

    /// <summary>Resolved model path from DBC chain (e.g. "Creature\Murloc\Murloc.mdx").</summary>
    public string? ModelPath { get; init; }

    /// <summary>Model scale from the template.</summary>
    public float Scale { get; init; } = 1.0f;

    /// <summary>Display scale from CreatureDisplayInfo (creatures) or template (gameobjects).</summary>
    public float DisplayScale { get; init; } = 1.0f;

    /// <summary>Texture variation paths from CreatureDisplayInfo (creatures only).</summary>
    public string[] TextureVariations { get; init; } = Array.Empty<string>();

    /// <summary>Creature level range.</summary>
    public int LevelMin { get; init; }
    public int LevelMax { get; init; }

    /// <summary>Creature rank (0=normal, 1=elite, 2=rare elite, 3=boss, 4=rare).</summary>
    public int Rank { get; init; }

    /// <summary>Creature type (1=beast, 2=dragonkin, 3=demon, etc.).</summary>
    public int CreatureType { get; init; }

    /// <summary>Creature faction ID.</summary>
    public int Faction { get; init; }

    /// <summary>NPC flags (vendor, trainer, quest giver, etc.).</summary>
    public int NpcFlags { get; init; }

    /// <summary>GameObject type (0=door, 1=button, 2=questgiver, 3=chest, etc.).</summary>
    public int GameObjectType { get; init; }

    /// <summary>GameObject flags.</summary>
    public int Flags { get; init; }

    /// <summary>Spawn locations for this entry.</summary>
    public List<SpawnLocation> Spawns { get; init; } = new();

    /// <summary>Whether the model is a WMO (vs MDX).</summary>
    public bool IsWmo => ModelPath != null &&
        ModelPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);

    /// <summary>Effective scale = template scale * display scale.</summary>
    public float EffectiveScale => Scale * DisplayScale;

    /// <summary>Human-readable type label.</summary>
    public string TypeLabel => Type switch
    {
        AssetType.Creature => Rank switch
        {
            1 => "Elite NPC",
            2 => "Rare Elite NPC",
            3 => "Boss",
            4 => "Rare NPC",
            _ => "NPC"
        },
        AssetType.GameObject => GameObjectType switch
        {
            0 => "Door",
            1 => "Button",
            2 => "Quest Giver",
            3 => "Chest",
            5 => "Goober",
            6 => "Transport",
            7 => "Area Damage",
            8 => "Camera",
            10 => "Spell Focus",
            13 => "Guild Bank",
            25 => "Fishinghole",
            _ => $"GameObject (type {GameObjectType})"
        },
        _ => "Unknown"
    };
}

public enum AssetType
{
    Creature,
    GameObject
}

public class SpawnLocation
{
    public int SpawnId { get; init; }
    public int MapId { get; init; }
    public float X { get; init; }
    public float Y { get; init; }
    public float Z { get; init; }
    public float Orientation { get; init; }
}
