namespace WowViewer.Core.Population;

/// <summary>
/// Authentic WoW chat channel types for NPC spoken dialogue, yells, and emotes.
/// Matches TrinityCore / Mangos chat type enumerations.
/// </summary>
public enum DialogueChatType
{
    System = 0,
    Say = 12,
    Yell = 14,
    Emote = 16,
    Whisper = 18,
    BossEmote = 42
}

/// <summary>
/// A single spoken line, shout, emote, or greeting from an NPC.
/// </summary>
public sealed class DialogueTextRecord
{
    public int EntryId { get; init; }
    public string Text { get; init; } = string.Empty;
    public DialogueChatType Type { get; init; } = DialogueChatType.Say;
    public float Probability { get; init; } = 100f;
    public string? Comment { get; init; }
}

/// <summary>
/// Quest metadata and flavor text offered by NPCs or GameObjects.
/// </summary>
public sealed class QuestTemplateRecord
{
    public int QuestId { get; init; }
    public string Title { get; init; } = string.Empty;
    public int QuestLevel { get; init; }
    public int MinLevel { get; init; }
    public string Objectives { get; init; } = string.Empty;
    public string Details { get; init; } = string.Empty;
    public string EndText { get; init; } = string.Empty;
}

/// <summary>
/// Comprehensive version-partitioned flavor data containing creature barks, gossip,
/// quest starter relationships, and quest descriptions.
/// </summary>
public sealed class EntityDialogueFlavorData
{
    public string Version { get; init; } = string.Empty;
    public DateTime GeneratedUtc { get; init; } = DateTime.UtcNow;

    /// <summary>Creature Entry -> Spoken/Yelled lines and ambient barks.</summary>
    public Dictionary<int, List<DialogueTextRecord>> CreatureBarks { get; init; } = new();

    /// <summary>Creature Entry -> Ambient greeting and gossip texts.</summary>
    public Dictionary<int, List<string>> CreatureGossip { get; init; } = new();

    /// <summary>Creature Entry -> List of Quest IDs offered.</summary>
    public Dictionary<int, List<int>> CreatureQuestStarters { get; init; } = new();

    /// <summary>GameObject Entry -> List of Quest IDs offered.</summary>
    public Dictionary<int, List<int>> GameObjectQuestStarters { get; init; } = new();

    /// <summary>Quest ID -> Full Quest template flavor data.</summary>
    public Dictionary<int, QuestTemplateRecord> Quests { get; init; } = new();
}
