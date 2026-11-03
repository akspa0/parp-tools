using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text.Json;
using WoWViewer.Logging;

namespace WoWViewer.Terrain.Pm4;

/// <summary>
/// Project persistence and manifest export for PM4 taught pairings (modeled on CartographyProjectStore).
/// Stores per-map JSON projects under output/projects/pm4_pairings/<mapName>.pairings.json
/// and synchronizes the global curation library under output/curation/pm4_taught_pairings.json.
/// </summary>
internal static class Pm4PairingProjectStore
{
    public static string ProjectsDir { get; set; } =
        Path.Combine(AppContext.BaseDirectory, "output", "projects", "pm4_pairings");

    public static string GlobalCurationPath { get; set; } =
        Path.Combine(AppContext.BaseDirectory, "output", "curation", "pm4_taught_pairings.json");

    public static string ProjectPath(string mapName)
        => Path.Combine(ProjectsDir, $"{SanitizeMapName(mapName)}.pairings.json");

    public static string ManifestPath(string mapName)
        => Path.Combine(ProjectsDir, $"{SanitizeMapName(mapName)}.manifest.json");

    public static bool Exists(string mapName) => File.Exists(ProjectPath(mapName));

    public static string SanitizeMapName(string? mapName)
    {
        if (string.IsNullOrWhiteSpace(mapName)) return "development";
        string clean = Path.GetFileNameWithoutExtension(mapName).Trim();
        return string.IsNullOrEmpty(clean) ? "development" : clean;
    }

    public static string SaveProject(string mapName, IReadOnlyList<TaughtPairing> pairings)
    {
        string cleanMap = SanitizeMapName(mapName);
        Directory.CreateDirectory(ProjectsDir);
        string projectPath = ProjectPath(cleanMap);

        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(pairings, options);
        File.WriteAllText(projectPath, json);

        // Also sync global curation library
        string globalDir = Path.GetDirectoryName(GlobalCurationPath)!;
        Directory.CreateDirectory(globalDir);
        File.WriteAllText(GlobalCurationPath, json);

        // Also generate the project manifest
        SaveManifest(cleanMap, pairings);

        ViewerLog.Important(ViewerLog.Category.Terrain,
            $"[PM4 Project] Saved {pairings.Count} taught pairings to '{projectPath}'.");

        return projectPath;
    }

    public static bool LoadProject(string mapName, out List<TaughtPairing> loadedPairings)
    {
        loadedPairings = new List<TaughtPairing>();
        string cleanMap = SanitizeMapName(mapName);
        string projectPath = ProjectPath(cleanMap);

        string sourcePath = File.Exists(projectPath)
            ? projectPath
            : (File.Exists(GlobalCurationPath) ? GlobalCurationPath : string.Empty);

        if (string.IsNullOrEmpty(sourcePath))
            return false;

        try
        {
            string json = File.ReadAllText(sourcePath);
            var items = JsonSerializer.Deserialize<List<TaughtPairing>>(json);
            if (items == null || items.Count == 0)
                return false;

            bool dirty = false;
            for (int i = 0; i < items.Count; i++)
            {
                var p = items[i];
                if (Pm4DoodadDuplicationService.IsWmoAsset(p.ModelPath, p.AssetKind))
                {
                    if (!string.Equals(p.AssetKind, "WMO", StringComparison.OrdinalIgnoreCase) || p.SolvedScale != 1.0f)
                    {
                        p = p with { AssetKind = "WMO", SolvedScale = 1.0f };
                        dirty = true;
                    }
                }
                loadedPairings.Add(p);
            }

            if (dirty)
            {
                string updatedJson = JsonSerializer.Serialize(loadedPairings, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(sourcePath, updatedJson);
            }

            ViewerLog.Important(ViewerLog.Category.Terrain,
                $"[PM4 Project] Loaded {loadedPairings.Count} taught pairings from '{sourcePath}'.");
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Terrain,
                $"[PM4 Project] Failed loading pairings from '{sourcePath}': {ex.Message}");
            return false;
        }
    }

    public static string SaveManifest(string mapName, IReadOnlyList<TaughtPairing> pairings, string? exportDir = null)
    {
        string cleanMap = SanitizeMapName(mapName);
        string targetDir = exportDir ?? ProjectsDir;
        Directory.CreateDirectory(targetDir);
        string manifestPath = Path.Combine(targetDir, $"{cleanMap}.manifest.json");

        int wmoCount = 0;
        int mdxCount = 0;
        for (int i = 0; i < pairings.Count; i++)
        {
            if (Pm4DoodadDuplicationService.IsWmoAsset(pairings[i].ModelPath, pairings[i].AssetKind))
                wmoCount++;
            else
                mdxCount++;
        }

        var manifest = new Pm4PairingsManifest
        {
            MapName = cleanMap,
            ExportedUtc = DateTime.UtcNow,
            TotalPairings = pairings.Count,
            WmoCount = wmoCount,
            MdxCount = mdxCount,
            Pairings = new List<TaughtPairing>(pairings)
        };

        string json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json);
        return manifestPath;
    }
}

internal sealed class Pm4PairingsManifest
{
    public string MapName { get; set; } = string.Empty;
    public DateTime ExportedUtc { get; set; }
    public int TotalPairings { get; set; }
    public int WmoCount { get; set; }
    public int MdxCount { get; set; }
    public List<TaughtPairing> Pairings { get; set; } = new();
}
