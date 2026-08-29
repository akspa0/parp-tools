using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Procedural;

/// <summary>
/// High-level semantic archetype derived from asset folder hierarchy and filename tokens.
/// </summary>
public enum AssetSemanticArchetype
{
    Weapon,
    Armor,
    Item,
    SpellEffect,
    SmallCritter,
    Humanoid,
    Mount,
    Monster,
    GiantBoss,
    SmallDoodad,
    MediumDoodad,
    LargeDoodad,
    Monument,
    Structure,
    Unknown
}

/// <summary>
/// Recommended spatial subdivision cell tier for procedural exhibition planning.
/// </summary>
public enum CellDensityTier
{
    /// <summary>16.66m cell (1/8 chunk) - High density for tiny weapons, items, spell crystals.</summary>
    Micro = 16,
    /// <summary>33.33m cell (1/4 chunk / 1 chunk) - Dense for small props, critters, shields.</summary>
    Small = 33,
    /// <summary>66.66m cell (1/2 chunk / 4 chunks) - Standard for humanoids, mounts, doodads.</summary>
    Medium = 66,
    /// <summary>133.33m cell (16 chunks) - Spacious for large monsters, dragons, small buildings.</summary>
    Large = 133,
    /// <summary>266.66m cell (64 chunks) - Grand for massive WMO structures, keeps, dungeons.</summary>
    Grand = 266,
    /// <summary>533.33m cell (Full tile) - Colossal for mega capital cities.</summary>
    Colossal = 533
}

/// <summary>
/// Detailed semantic classification of a WoW game asset.
/// </summary>
public sealed record SemanticAssetClassification(
    AssetSemanticArchetype Archetype,
    CellDensityTier RecommendedDensityTier,
    float RecommendedScale,
    string SuggestedPavilion,
    bool IsDungeonSetAsset,
    string? DungeonPrefix = null);

/// <summary>
/// Classifies WoW models and world models into semantic archetypes, scale multipliers,
/// and density tiers based on directory paths, filename tokens, and geometric bounding extents.
/// </summary>
public static class SemanticAssetClassifier
{
    private static readonly HashSet<string> WeaponKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "1h", "2h", "bow", "staff", "wand", "dagger", "axe", "sword", "mace", "shield",
        "polearm", "crossbow", "gun", "spear", "glaive", "thrown", "quiver", "ammo",
        "blade", "hammer", "stave", "scepter", "flail", "knuckles", "fist"
    };

    private static readonly HashSet<string> ArmorKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "helm", "head", "shoulder", "pad", "buckle", "belt", "cape", "cloak", "boots",
        "glove", "gauntlet", "bracer", "pant", "robe", "chest", "vest", "shirt", "tabard"
    };

    private static readonly HashSet<string> CritterKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "critter", "rat", "mouse", "frog", "toad", "rabbit", "hare", "squirrel", "bird",
        "parrot", "cockatiel", "cat", "kitten", "dog", "pup", "crab", "beetle", "worm",
        "cockroach", "moth", "snake", "viper", "fish", "prairiedog", "penguin", "chub",
        "skunk", "chicken", "rooster", "turkey", "pig", "sheep", "cow", "fawn", "doe"
    };

    private static readonly HashSet<string> MountKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "mount", "horse", "wolf", "raptor", "ram", "kodo", "tiger", "panther",
        "mechanostrider", "skeletalhorse", "gryphon", "wyvern", "bat", "hippogryph",
        "windrider", "strider", "nightsaber", "warhorse"
    };

    private static readonly HashSet<string> HumanoidKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "human", "orc", "dwarf", "gnome", "nightelf", "tauren", "undead", "scourge",
        "troll", "goblin", "naga", "murloc", "gnoll", "kobold", "centaur", "quillboar",
        "trogg", "harpy", "furbolg", "ogre", "dryad", "keeperofthegrove", "satyr",
        "dragonspawn", "bloodelf", "draenei"
    };

    private static readonly HashSet<string> GiantBossKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "dragon", "drake", "wyrm", "giant", "colossus", "kraken", "hydra", "golem",
        "ragnaros", "nefarian", "onyxia", "cthun", "hakkar", "magmadar", "lucifron",
        "gehennas", "garr", "shazzrah", "sulfuron", "golemagg", "majordomo"
    };

    private static readonly Dictionary<string, string> DungeonPrefixes = new(StringComparer.OrdinalIgnoreCase)
    {
        ["sm_"] = "Scarlet Monastery",
        ["dm_"] = "Dire Maul",
        ["zf_"] = "Zul'Farrak",
        ["st_"] = "Sunken Temple",
        ["strath_"] = "Stratholme",
        ["scholo_"] = "Scholomance",
        ["vc_"] = "Deadmines",
        ["brd_"] = "Blackrock Depths",
        ["brs_"] = "Blackrock Spire",
        ["mc_"] = "Molten Core",
        ["ony_"] = "Onyxia's Lair",
        ["bwl_"] = "Blackwing Lair",
        ["zg_"] = "Zul'Gurub",
        ["aq_"] = "Ahn'Qiraj",
        ["aq20_"] = "Ruins of Ahn'Qiraj",
        ["aq40_"] = "Temple of Ahn'Qiraj",
        ["naxx_"] = "Naxxramas"
    };

    /// <summary>
    /// Classifies an asset by examining its full path, filename tokens, kind, and spatial bounds.
    /// </summary>
    public static SemanticAssetClassification Classify(
        string assetPath,
        Vector3 boundsMin,
        Vector3 boundsMax,
        RosettaAssetKind kind)
    {
        string normPath = (assetPath ?? string.Empty).Replace('\\', '/').ToLowerInvariant();
        string fileName = Path.GetFileNameWithoutExtension(normPath);
        float radius = MathF.Max(
            MathF.Abs(boundsMax.X - boundsMin.X),
            MathF.Max(MathF.Abs(boundsMax.Y - boundsMin.Y), MathF.Abs(boundsMax.Z - boundsMin.Z))) * 0.5f;

        // 1. Check for Contextual Dungeon Set Prefixes
        bool isDungeonPath = normPath.Contains("/dungeon/") || normPath.Contains("/wmo/dungeon/") || normPath.Contains("wmo/");
        string? matchedDungeon = null;
        string? matchedPrefix = null;

        foreach (var (prefix, dungeonName) in DungeonPrefixes)
        {
            if (fileName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) &&
                (isDungeonPath || normPath.Contains(prefix.TrimEnd('_'))))
            {
                matchedDungeon = dungeonName;
                matchedPrefix = prefix.TrimEnd('_');
                break;
            }
        }

        // 2. World Models (WMO)
        if (kind == RosettaAssetKind.WorldModel || normPath.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
        {
            if (radius > 120f)
            {
                return new SemanticAssetClassification(
                    AssetSemanticArchetype.Structure,
                    CellDensityTier.Grand,
                    RecommendedScale: 1.0f,
                    SuggestedPavilion: matchedDungeon != null ? $"{matchedDungeon} Dungeons" : "Grand Architecture",
                    IsDungeonSetAsset: matchedDungeon != null,
                    DungeonPrefix: matchedPrefix);
            }

            return new SemanticAssetClassification(
                AssetSemanticArchetype.Structure,
                CellDensityTier.Large,
                RecommendedScale: 1.0f,
                SuggestedPavilion: matchedDungeon != null ? $"{matchedDungeon} Buildings" : "Pavilions & Buildings",
                IsDungeonSetAsset: matchedDungeon != null,
                DungeonPrefix: matchedPrefix);
        }

        // 3. Weapons & Armory
        if (normPath.Contains("/weapon") || normPath.Contains("/weapons") ||
            (normPath.Contains("item/objectcomponents/") && WeaponKeywords.Any(k => normPath.Contains(k))))
        {
            bool isTinyWeapon = radius < 0.8f || fileName.Contains("dagger") || fileName.Contains("wand") || fileName.Contains("thrown") || fileName.Contains("knife");
            float scale = isTinyWeapon ? 4.0f : 3.0f;
            return new SemanticAssetClassification(
                AssetSemanticArchetype.Weapon,
                CellDensityTier.Micro,
                RecommendedScale: scale,
                SuggestedPavilion: "Grand Armory",
                IsDungeonSetAsset: false);
        }

        // 4. Armor & Apparel
        if (normPath.Contains("/armor") ||
            (normPath.Contains("item/objectcomponents/") && ArmorKeywords.Any(k => normPath.Contains(k))))
        {
            return new SemanticAssetClassification(
                AssetSemanticArchetype.Armor,
                CellDensityTier.Micro,
                RecommendedScale: 3.5f,
                SuggestedPavilion: "Royal Wardrobe",
                IsDungeonSetAsset: false);
        }

        // 5. Items & Spells
        if (normPath.StartsWith("item/") || normPath.Contains("/items/"))
        {
            return new SemanticAssetClassification(
                AssetSemanticArchetype.Item,
                CellDensityTier.Micro,
                RecommendedScale: 3.5f,
                SuggestedPavilion: "Curios & Artifacts",
                IsDungeonSetAsset: false);
        }

        if (normPath.StartsWith("spells/") || normPath.Contains("/spell/"))
        {
            return new SemanticAssetClassification(
                AssetSemanticArchetype.SpellEffect,
                CellDensityTier.Micro,
                RecommendedScale: 3.0f,
                SuggestedPavilion: "Arcane Conservatory",
                IsDungeonSetAsset: false);
        }

        // 6. Living Creatures & Characters
        if (normPath.StartsWith("character/") || normPath.Contains("/character/"))
        {
            return new SemanticAssetClassification(
                AssetSemanticArchetype.Humanoid,
                CellDensityTier.Medium,
                RecommendedScale: 2.0f,
                SuggestedPavilion: "Hall of Champions",
                IsDungeonSetAsset: false);
        }

        if (normPath.StartsWith("creature/") || normPath.Contains("/creature/"))
        {
            if (CritterKeywords.Any(k => normPath.Contains(k)))
            {
                return new SemanticAssetClassification(
                    AssetSemanticArchetype.SmallCritter,
                    CellDensityTier.Small,
                    RecommendedScale: 2.5f,
                    SuggestedPavilion: "Garden Menagerie",
                    IsDungeonSetAsset: false);
            }

            if (MountKeywords.Any(k => normPath.Contains(k)))
            {
                return new SemanticAssetClassification(
                    AssetSemanticArchetype.Mount,
                    CellDensityTier.Medium,
                    RecommendedScale: 2.0f,
                    SuggestedPavilion: "Royal Stables",
                    IsDungeonSetAsset: false);
            }

            if (GiantBossKeywords.Any(k => normPath.Contains(k)) || radius > 15f)
            {
                return new SemanticAssetClassification(
                    AssetSemanticArchetype.GiantBoss,
                    CellDensityTier.Large,
                    RecommendedScale: 1.0f,
                    SuggestedPavilion: "Titan Colosseum",
                    IsDungeonSetAsset: false);
            }

            if (HumanoidKeywords.Any(k => normPath.Contains(k)))
            {
                return new SemanticAssetClassification(
                    AssetSemanticArchetype.Humanoid,
                    CellDensityTier.Medium,
                    RecommendedScale: 2.0f,
                    SuggestedPavilion: "Peoples of Azeroth",
                    IsDungeonSetAsset: false);
            }

            return new SemanticAssetClassification(
                AssetSemanticArchetype.Monster,
                CellDensityTier.Medium,
                RecommendedScale: 2.0f,
                SuggestedPavilion: "Bestiary Conservatory",
                IsDungeonSetAsset: false);
        }

        // 7. World Doodads & Props
        bool hasSmallToken = !isDungeonPath && (
            fileName.EndsWith("_sm", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith("_small", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith("_tiny", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith("_micro", StringComparison.OrdinalIgnoreCase) ||
            fileName.StartsWith("sm_", StringComparison.OrdinalIgnoreCase) ||
            fileName.Contains("_sm_") ||
            fileName.Contains("small") ||
            fileName.Contains("tiny"));

        bool hasLargeToken = (
            fileName.EndsWith("_lg", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith("_large", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith("_huge", StringComparison.OrdinalIgnoreCase) ||
            fileName.EndsWith("_giant", StringComparison.OrdinalIgnoreCase) ||
            fileName.Contains("_lg_") ||
            fileName.Contains("large") ||
            fileName.Contains("huge") ||
            fileName.Contains("giant") ||
            fileName.Contains("monument") ||
            fileName.Contains("statue"));

        if (hasLargeToken || radius > 12f)
        {
            return new SemanticAssetClassification(
                AssetSemanticArchetype.LargeDoodad,
                CellDensityTier.Large,
                RecommendedScale: 1.5f,
                SuggestedPavilion: matchedDungeon != null ? $"{matchedDungeon} Monuments" : "Sculpture Courtyard",
                IsDungeonSetAsset: matchedDungeon != null,
                DungeonPrefix: matchedPrefix);
        }

        if (hasSmallToken || radius <= 1.5f)
        {
            return new SemanticAssetClassification(
                AssetSemanticArchetype.SmallDoodad,
                CellDensityTier.Small,
                RecommendedScale: 2.5f,
                SuggestedPavilion: matchedDungeon != null ? $"{matchedDungeon} Doodads" : "Garden Accessories",
                IsDungeonSetAsset: matchedDungeon != null,
                DungeonPrefix: matchedPrefix);
        }

        return new SemanticAssetClassification(
            AssetSemanticArchetype.MediumDoodad,
            CellDensityTier.Medium,
            RecommendedScale: 2.0f,
            SuggestedPavilion: matchedDungeon != null ? $"{matchedDungeon} Props" : "Botanical Doodads",
            IsDungeonSetAsset: matchedDungeon != null,
            DungeonPrefix: matchedPrefix);
    }
}
