using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text.Json;
using WowViewer.Core.PM4.Footprint;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// A pre-correlated asset entry in the PM4 prior knowledge database.
/// Contains verified or harvested bounds, group frequencies, and neighborhood classification.
/// </summary>
public sealed record PriorKnowledgeAsset(
    string ModelPath,
    string ModelName,
    string AssetKind,
    string Family,
    Vector3 ApproximateBoundsMin,
    Vector3 ApproximateBoundsMax,
    int CorrelatedGroupsCount,
    bool IsCurated,
    float SolvedScale = 1.0f,
    float YawOffset = 0f)
{
    public float Length => MathF.Abs(ApproximateBoundsMax.X - ApproximateBoundsMin.X);
    public float Width => MathF.Abs(ApproximateBoundsMax.Y - ApproximateBoundsMin.Y);
    public float Height => MathF.Abs(ApproximateBoundsMax.Z - ApproximateBoundsMin.Z);

    public Pm4ModelCandidate ToModelCandidate()
    {
        return Pm4ModelCandidate.Create(ModelPath, AssetKind, ApproximateBoundsMin, ApproximateBoundsMax);
    }
}

/// <summary>
/// Built-in pre-populated database of known PM4-to-model candidate pairings and neighborhood models.
/// Indexes curated 3D relationships, persistent taught shape templates, and harvested whole-map probe corpus.
/// Allows live on-the-fly model matching and neighborhood auto-mapping even when models are not resident on the active tile.
/// </summary>
public sealed class Pm4PriorKnowledgeDatabase
{
    private static readonly Lazy<Pm4PriorKnowledgeDatabase> _instance = new(() => new Pm4PriorKnowledgeDatabase());
    public static Pm4PriorKnowledgeDatabase Instance => _instance.Value;

    private readonly Dictionary<string, PriorKnowledgeAsset> _assetsByPath = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, PriorKnowledgeAsset> _assetsByName = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, List<PriorKnowledgeAsset>> _assetsByFamily = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<PriorKnowledgeAsset> _allAssets = [];

    public IReadOnlyList<PriorKnowledgeAsset> AllAssets => _allAssets;
    public IReadOnlyDictionary<string, List<PriorKnowledgeAsset>> AssetsByFamily => _assetsByFamily;

    public Pm4PriorKnowledgeDatabase()
    {
        // 1. Seed with high-confidence architectural assets and curated ground-truth models
        SeedCuratedAssets();

        // 2. Discover and ingest disk files if available in runtime workspace
        TryIngestCuratedRelationshipsJson();
        TryIngestTaughtPairingsJson();
        TryIngestProbeEvidenceJson();
    }

    /// <summary>
    /// Detects the semantic neighborhood or architectural family of a given asset path or model name.
    /// </summary>
    public static string DetectFamily(string modelPathOrName)
    {
        if (string.IsNullOrWhiteSpace(modelPathOrName))
            return "Generic";

        string p = modelPathOrName.ToUpperInvariant();

        if (p.Contains("WINTERTAUREN") || p.Contains("WT_"))
            return "Wintertauren";
        if (p.Contains("TAUREN"))
            return "Tauren";
        if (p.Contains("TUSKARR"))
            return "Tuskarr";
        if (p.Contains("VRYKUL"))
            return "Vrykul";
        if (p.Contains("WINTERGRASP") || p.Contains("WG_"))
            return "Wintergrasp";
        if (p.Contains("IRONDWARF") || p.Contains("DARKIRONDWARF"))
            return "Iron Dwarf";
        if (p.Contains("DWARF") || p.Contains("KHAZMODAN") || p.Contains("IRONFORGE"))
            return "Dwarven";
        if (p.Contains("HUMAN") || p.Contains("ELWYNN") || p.Contains("WESTFALL") || p.Contains("REDRIDGE") || p.Contains("GOLDSHIRE"))
            return "HumanArchitecture";
        if (p.Contains("NIGHTELF") || p.Contains("MOONWELL"))
            return "Night Elf";
        if (p.Contains("BLOODELF") || p.Contains("HIGHELF") || p.Contains("SILVERMOON"))
            return "Blood Elf";
        if (p.Contains("ORC") || p.Contains("DUROTAR") || p.Contains("WINTERORC"))
            return "Orc";
        if (p.Contains("BRIDGE"))
            return "Bridges";
        if (p.Contains("KEEP") || p.Contains("CASTLE"))
            return "Keeps & Castles";
        if (p.Contains("TOWER") || p.Contains("GUARDTOWER"))
            return "Towers";
        if (p.Contains("MINE") || p.Contains("GOLDMINE"))
            return "Mines";
        if (p.Contains("SHIP") || p.Contains("BOAT"))
            return "Ships";
        if (p.Contains("RUIN"))
            return "Ruins";
        if (p.Contains("ROCK") || p.Contains("CLIFF"))
            return "Rocks";
        if (p.Contains("TREE") || p.Contains("LOG") || p.Contains("STUMP") || p.Contains("BUSH"))
            return "Trees & Flora";
        if (p.Contains("BONE"))
            return "Bones";

        return "Generic";
    }

    /// <summary>
    /// Gets all candidate assets from the database matching the given neighborhood or family.
    /// </summary>
    public IReadOnlyList<PriorKnowledgeAsset> GetNeighborhoodAssets(string modelPathOrName)
    {
        string family = DetectFamily(modelPathOrName);
        if (_assetsByFamily.TryGetValue(family, out var list))
            return list;

        return [];
    }

    /// <summary>
    /// Gets all candidate assets from the database for matching against PM4 footprint prisms.
    /// If targetFamily is specified, candidates from that family are returned first.
    /// </summary>
    public List<Pm4ModelCandidate> GetDatabaseCandidates(string? targetFamily = null)
    {
        var candidates = new List<Pm4ModelCandidate>(_allAssets.Count);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Add target family first if requested
        if (!string.IsNullOrWhiteSpace(targetFamily) && _assetsByFamily.TryGetValue(targetFamily, out var familyList))
        {
            for (int i = 0; i < familyList.Count; i++)
            {
                var asset = familyList[i];
                if (seen.Add(asset.ModelPath))
                    candidates.Add(asset.ToModelCandidate());
            }
        }

        // 2. Add curated and high-frequency architectural assets
        for (int i = 0; i < _allAssets.Count; i++)
        {
            var asset = _allAssets[i];
            if (seen.Add(asset.ModelPath))
                candidates.Add(asset.ToModelCandidate());
        }

        return candidates;
    }

    /// <summary>
    /// Checks if a model path is known in the database.
    /// </summary>
    public bool TryGetAsset(string modelPathOrName, out PriorKnowledgeAsset? asset)
    {
        if (_assetsByPath.TryGetValue(modelPathOrName, out asset))
            return true;

        string name = Path.GetFileName(modelPathOrName);
        return _assetsByName.TryGetValue(name, out asset);
    }

    private void AddAsset(PriorKnowledgeAsset asset)
    {
        if (string.IsNullOrWhiteSpace(asset.ModelPath))
            return;

        if (_assetsByPath.TryGetValue(asset.ModelPath, out var existing))
        {
            // Update if newly provided asset is curated or has higher group count
            if (asset.IsCurated || asset.CorrelatedGroupsCount > existing.CorrelatedGroupsCount)
            {
                _assetsByPath[asset.ModelPath] = asset;
                _assetsByName[asset.ModelName] = asset;
                int idx = _allAssets.IndexOf(existing);
                if (idx >= 0) _allAssets[idx] = asset;
                if (_assetsByFamily.TryGetValue(asset.Family, out var fList))
                {
                    int fIdx = fList.IndexOf(existing);
                    if (fIdx >= 0) fList[fIdx] = asset;
                }
            }
            return;
        }

        _assetsByPath[asset.ModelPath] = asset;
        _assetsByName[asset.ModelName] = asset;
        _allAssets.Add(asset);

        if (!_assetsByFamily.TryGetValue(asset.Family, out var list))
        {
            list = [];
            _assetsByFamily[asset.Family] = list;
        }
        list.Add(asset);
    }

    private void SeedCuratedAssets()
    {
        // Wintertauren Architecture & Bridges
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\BUILDINGS\\WINTERTAUREN\\ND_WINTERTAUREN_BRIDGE_01.WMO",
            "WMO", "Wintertauren", new Vector3(-47.5f, -12.5f, 0f), new Vector3(47.5f, 12.5f, 18.0f), 10, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\BUILDINGS\\WINTERTAUREN_INN\\WINTERTAUREN_INN.WMO",
            "WMO", "Wintertauren", new Vector3(-2.1f, -1.5f, 0f), new Vector3(2.1f, 1.5f, 3.9f), 12, yawOffset: 67.5f);
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\BUILDINGS\\WINTERTAUREN_SMOKEHUT\\WINTERTAUREN_SMOKEHUT.WMO",
            "WMO", "Wintertauren", new Vector3(-2.8f, -2.1f, 0f), new Vector3(2.8f, 2.1f, 4.7f), 8, yawOffset: 99.6f);
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\BUILDINGS\\WINTERTAUREN_GUARDTOWER\\WINTERTAUREN_GUARDTOWER.WMO",
            "WMO", "Wintertauren", new Vector3(-6.0f, -6.0f, 0f), new Vector3(6.0f, 6.0f, 28.0f), 7, yawOffset: 0f);
        AddCurated(
            "WORLD\\EXPANSION02\\DOODADS\\WINTERTAUREN\\WT_BENCH_01.M2",
            "M2", "Wintertauren", new Vector3(-1.2f, -0.4f, 0f), new Vector3(1.2f, 0.4f, 0.8f), 15, yawOffset: 0f);
        AddCurated(
            "WORLD\\EXPANSION02\\DOODADS\\WINTERTAUREN\\WT_SLED_02PACKED.M2",
            "M2", "Wintertauren", new Vector3(-1.8f, -0.9f, 0f), new Vector3(1.8f, 0.9f, 1.2f), 8, yawOffset: 0f);

        // Bridges
        AddCurated(
            "WORLD\\WMO\\KALIMDOR\\COLLIDABLEDOODADS\\MULGORE\\TAURENBRIDGE01\\TAURENBRIDGE01.WMO",
            "WMO", "Bridges", new Vector3(-27.5f, -26.6f, 0f), new Vector3(27.5f, 26.6f, 3.58f), 20, yawOffset: -176.6f);
        AddCurated(
            "WORLD\\WMO\\KALIMDOR\\COLLIDABLEDOODADS\\DARKSHORE\\BRIDGE\\DARKSHOREBRIDGE\\BRIAN\\DARKSHOREBRIDGE01.WMO",
            "WMO", "Bridges", new Vector3(-3.4f, -1.8f, 0f), new Vector3(3.4f, 1.8f, 3.34f), 15, yawOffset: -83.5f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\COLLIDABLE DOODADS\\REDRIDGE\\REDRIDGEDAMAGEDBRIDGE\\REDRIDGEDAMAGEDBRIDGE.WMO",
            "WMO", "Bridges", new Vector3(-8.3f, -5.4f, 0f), new Vector3(8.3f, 5.4f, 2.23f), 18, yawOffset: 120.5f);
        AddCurated(
            "WORLD\\WMO\\KHAZMODAN\\COLLIDABLEDOODADS\\WETLANDSFOOTBRIDGE\\WETLANDSFOOTBRIDGE01.WMO",
            "WMO", "Bridges", new Vector3(-3.3f, -2.4f, 0f), new Vector3(3.3f, 2.4f, 2.01f), 22, yawOffset: 90.1f);
        AddCurated(
            "WORLD\\AZEROTH\\DEADWINDPASS\\BUILDINGS\\ROCKBRIDGE\\DEADWINDROCKBRIDGE.M2",
            "M2", "Bridges", new Vector3(-39.3f, -25.6f, 0f), new Vector3(39.3f, 25.6f, 25.3f), 3, yawOffset: -71.1f, solvedScale: 5.45f);

        // Human Architecture, Keeps, Towers
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\GUARDTOWER\\GUARDTOWER.WMO",
            "WMO", "Towers", new Vector3(-11.0f, -9.2f, 0f), new Vector3(11.0f, 9.2f, 22.0f), 168, yawOffset: -31.7f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\BIG_KEEP\\BIG_KEEP.WMO",
            "WMO", "Keeps & Castles", new Vector3(-20.0f, -15.0f, 0f), new Vector3(20.0f, 15.0f, 28.0f), 25, yawOffset: 48.5f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\NSABBEY\\NSABBEY.WMO",
            "WMO", "HumanArchitecture", new Vector3(-18.0f, -18.0f, 0f), new Vector3(18.0f, 18.0f, 23.5f), 15, yawOffset: -44.8f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\HUMANTWOSTORY\\HUMANTWOSTORY.WMO",
            "WMO", "HumanArchitecture", new Vector3(-12.0f, -9.0f, 0f), new Vector3(12.0f, 9.0f, 16.0f), 394, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\HUMAN_FARM\\FARM.WMO",
            "WMO", "HumanArchitecture", new Vector3(-11.0f, -8.0f, 0f), new Vector3(11.0f, 8.0f, 14.0f), 296, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\STABLE\\STABLE.WMO",
            "WMO", "HumanArchitecture", new Vector3(-10.0f, -7.5f, 0f), new Vector3(10.0f, 7.5f, 12.0f), 283, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\KEEPWALL\\WALLPOST01.WMO",
            "WMO", "KeepWall", new Vector3(-5.0f, -5.0f, 0f), new Vector3(5.0f, 5.0f, 18.0f), 179, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\KEEPWALL\\WALLPIECE01.WMO",
            "WMO", "KeepWall", new Vector3(-14.0f, -4.0f, 0f), new Vector3(14.0f, 4.0f, 16.0f), 123, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\AZEROTH\\BUILDINGS\\KEEPWALL\\WALLPIECESTAIRS01.WMO",
            "WMO", "KeepWall", new Vector3(-14.0f, -5.0f, 0f), new Vector3(14.0f, 5.0f, 16.0f), 103, yawOffset: 0f);

        // Tauren & Night Elf
        AddCurated(
            "WORLD\\WMO\\PVP\\BUILDINGS\\TAURENGATE\\PVP_TAURENGATE01.WMO",
            "WMO", "Tauren", new Vector3(-33.3f, -16.6f, 0f), new Vector3(33.3f, 16.6f, 22.0f), 10, yawOffset: 0.01f);
        AddCurated(
            "WORLD\\WMO\\KALIMDOR\\BUILDINGS\\TAURENHUTS\\TAURENHUTD.WMO",
            "WMO", "Tauren", new Vector3(-8.0f, -8.0f, 0f), new Vector3(8.0f, 8.0f, 12.0f), 38, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\KALIMDOR\\BUILDINGS\\TAURENHUTS\\TAURENHUTB.WMO",
            "WMO", "Tauren", new Vector3(-7.5f, -7.5f, 0f), new Vector3(7.5f, 7.5f, 11.0f), 35, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\KALIMDOR\\BUILDINGS\\NIGHTELFMOONWELL\\DSNIGHTELFMOONWELL.WMO",
            "WMO", "Night Elf", new Vector3(-2.5f, -2.5f, 0f), new Vector3(2.5f, 2.5f, 3.4f), 15, yawOffset: -83.5f);

        // Mines, Ships, Dwarven, Tuskarr, Wintergrasp
        AddCurated(
            "WORLD\\WMO\\DUNGEON\\MD_GOLDMINE\\MD_GOLDMINE.WMO",
            "WMO", "Mines", new Vector3(-3.2f, -2.5f, 0f), new Vector3(3.2f, 2.5f, 4.5f), 12, yawOffset: 173.4f);
        AddCurated(
            "WORLD\\WMO\\DUNGEON\\MD_PIRATESHIP\\PIRATESHIP.WMO",
            "WMO", "Ships", new Vector3(-15.0f, -5.0f, 0f), new Vector3(15.0f, 5.0f, 25.0f), 8, yawOffset: -69.8f);
        AddCurated(
            "WORLD\\WMO\\KHAZMODAN\\BUILDINGS\\DWARVEN_GRIFFONAVIARY\\DWARVEN_GRIFFONAVIARY.WMO",
            "WMO", "Dwarven", new Vector3(-5.0f, -2.5f, 0f), new Vector3(5.0f, 2.5f, 26.0f), 10, yawOffset: -85.5f);
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\BUILDINGS\\TUSKARR\\ND_TUSKARR_DESTOYEDHUT02\\ND_TUSKARR_DESTOYEDHUT02.WMO",
            "WMO", "Tuskarr", new Vector3(-7.0f, -6.0f, 0f), new Vector3(7.0f, 6.0f, 10.0f), 97, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\WINTERGRASP\\WG_WALL02D.WMO",
            "WMO", "Wintergrasp", new Vector3(-17.5f, -5.0f, 0f), new Vector3(17.5f, 5.0f, 20.0f), 39, yawOffset: 0f);
        AddCurated(
            "WORLD\\WMO\\NORTHREND\\BUILDINGS\\IRONDWARF\\ND_IRONDWARF_LARGEBUILDING\\ND_IRONDWARF_LARGEBUILDING.WMO",
            "WMO", "Iron Dwarf", new Vector3(-20.0f, -15.0f, 0f), new Vector3(20.0f, 15.0f, 25.0f), 45, yawOffset: 0f);
    }

    private void AddCurated(
        string modelPath, string assetKind, string family, Vector3 min, Vector3 max, int groupCount, float yawOffset = 0f, float solvedScale = 1.0f)
    {
        string name = Path.GetFileName(modelPath);
        AddAsset(new PriorKnowledgeAsset(
            modelPath, name, assetKind, family, min, max, groupCount, IsCurated: true, SolvedScale: solvedScale, YawOffset: yawOffset));
    }

    private static IEnumerable<string> ExpandCandidatePaths(string[] candidatePaths, string repoRelativeSuffix)
    {
        foreach (string path in candidatePaths)
            yield return path;

        foreach (string root in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            if (string.IsNullOrWhiteSpace(root))
                continue;

            DirectoryInfo? dir = new(root);
            while (dir != null)
            {
                yield return Path.Combine(dir.FullName, repoRelativeSuffix);
                yield return Path.Combine(dir.FullName, "wow-viewer", repoRelativeSuffix);
                dir = dir.Parent;
            }
        }
    }

    private void TryIngestCuratedRelationshipsJson()
    {
        string[] candidatePaths = [
            "output/curation/pm4_curated_relationships.json",
            "../output/curation/pm4_curated_relationships.json",
            "../../output/curation/pm4_curated_relationships.json",
            "../../../output/curation/pm4_curated_relationships.json"
        ];

        foreach (var p in ExpandCandidatePaths(candidatePaths, Path.Combine("output", "curation", "pm4_curated_relationships.json")))
        {
            if (File.Exists(p))
            {
                try
                {
                    string json = File.ReadAllText(p);
                    using var doc = JsonDocument.Parse(json);
                    foreach (var elem in doc.RootElement.EnumerateArray())
                    {
                        string path = elem.GetProperty("ModelPath").GetString() ?? "";
                        string kind = elem.GetProperty("AssetKind").GetString() ?? "M2";
                        float len = elem.GetProperty("Length").GetSingle();
                        float wid = elem.GetProperty("Width").GetSingle();
                        float hgt = elem.GetProperty("Height").GetSingle();
                        float scale = elem.TryGetProperty("SolvedScale", out var sp) ? sp.GetSingle() : 1.0f;
                        float yaw = elem.TryGetProperty("YawOffset", out var yp) ? yp.GetSingle() : 0f;

                        Vector3 min = new(-len * 0.5f, -wid * 0.5f, 0f);
                        Vector3 max = new(len * 0.5f, wid * 0.5f, hgt);

                        AddAsset(new PriorKnowledgeAsset(
                            path, Path.GetFileName(path), kind, DetectFamily(path), min, max, 10, IsCurated: true, SolvedScale: scale, YawOffset: yaw));
                    }
                    break;
                }
                catch
                {
                    // Fall back to embedded seed
                }
            }
        }
    }

    private void TryIngestTaughtPairingsJson()
    {
        string[] candidatePaths = [
            "output/curation/pm4_taught_pairings.json",
            "../output/curation/pm4_taught_pairings.json",
            "../../output/curation/pm4_taught_pairings.json"
        ];

        foreach (var p in ExpandCandidatePaths(candidatePaths, Path.Combine("output", "curation", "pm4_taught_pairings.json")))
        {
            if (File.Exists(p))
            {
                try
                {
                    string json = File.ReadAllText(p);
                    using var doc = JsonDocument.Parse(json);
                    foreach (var elem in doc.RootElement.EnumerateArray())
                    {
                        string path = elem.GetProperty("ModelPath").GetString() ?? "";
                        string kind = elem.GetProperty("AssetKind").GetString() ?? "M2";
                        float len = elem.GetProperty("Length").GetSingle();
                        float wid = elem.GetProperty("Width").GetSingle();
                        float hgt = elem.GetProperty("Height").GetSingle();
                        float scale = elem.TryGetProperty("SolvedScale", out var sp) ? sp.GetSingle() : 1.0f;
                        float yaw = elem.TryGetProperty("YawDegrees", out var yp) ? yp.GetSingle() : 0f;

                        Vector3 min = new(-len * 0.5f, -wid * 0.5f, 0f);
                        Vector3 max = new(len * 0.5f, wid * 0.5f, hgt);

                        AddAsset(new PriorKnowledgeAsset(
                            path, Path.GetFileName(path), kind, DetectFamily(path), min, max, 5, IsCurated: true, SolvedScale: scale, YawOffset: yaw));
                    }
                    break;
                }
                catch
                {
                    // Fall back to embedded seed
                }
            }
        }
    }

    private void TryIngestProbeEvidenceJson()
    {
        string[] candidatePaths = [
            "wow-viewer/specs/279-decode-pm4-format/evidence/object-models-v4-2026-10-20.json",
            "specs/279-decode-pm4-format/evidence/object-models-v4-2026-10-20.json",
            "../wow-viewer/specs/279-decode-pm4-format/evidence/object-models-v4-2026-10-20.json",
            "../../wow-viewer/specs/279-decode-pm4-format/evidence/object-models-v4-2026-10-20.json"
        ];

        foreach (var p in ExpandCandidatePaths(candidatePaths, Path.Combine("specs", "279-decode-pm4-format", "evidence", "object-models-v4-2026-10-20.json")))
        {
            if (File.Exists(p))
            {
                try
                {
                    string json = File.ReadAllText(p);
                    using var doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty("Result", out var resElem) &&
                        resElem.TryGetProperty("Models", out var modelsElem))
                    {
                        foreach (var m in modelsElem.EnumerateArray())
                        {
                            string path = m.GetProperty("ModelPath").GetString() ?? "";
                            int groups = m.GetProperty("Groups").GetInt32();
                            if (string.IsNullOrWhiteSpace(path) || groups < 2)
                                continue;

                            bool isWmo = path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase);
                            string kind = isWmo ? "WMO" : "M2";
                            string family = DetectFamily(path);

                            // Synthesize bounds approximation if not already known
                            if (!_assetsByPath.ContainsKey(path))
                            {
                                float defaultPlanar = isWmo ? 20.0f : 4.0f;
                                float defaultHeight = isWmo ? 15.0f : 5.0f;
                                Vector3 min = new(-defaultPlanar * 0.5f, -defaultPlanar * 0.5f, 0f);
                                Vector3 max = new(defaultPlanar * 0.5f, defaultPlanar * 0.5f, defaultHeight);

                                AddAsset(new PriorKnowledgeAsset(
                                    path, Path.GetFileName(path), kind, family, min, max, groups, IsCurated: false));
                            }
                        }
                    }
                    break;
                }
                catch
                {
                    // Ignore and keep seeded assets
                }
            }
        }
    }
}

