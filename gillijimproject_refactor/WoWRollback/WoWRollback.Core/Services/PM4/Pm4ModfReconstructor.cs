using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WoWRollback.Core.Services.PM4;

/// <summary>
/// Reconstructs MODF (WMO placement) data from PM4 pathfinding geometry.
/// Matches PM4 objects against known WMOs and extracts placement transforms.
/// </summary>
public sealed class Pm4ModfReconstructor
{
    private readonly Pm4WmoGeometryMatcher _matcher = new();

    /// <summary>
    /// Represents a WMO in the reference library with pre-computed geometry stats.
    /// </summary>
    public record WmoReference(
        string WmoPath,
        string CollisionObjPath,
        Pm4WmoGeometryMatcher.GeometryStats Stats);

    /// <summary>
    /// Represents a PM4 object with geometry stats.
    /// </summary>
    public record Pm4Object(
        string Ck24,
        string ObjPath,
        int TileX,
        int TileY,
        Pm4WmoGeometryMatcher.GeometryStats Stats);

    /// <summary>
    /// Reconstructed MODF entry.
    /// </summary>
    public record ModfEntry(
        uint NameId,           // Index into MWMO string table
        uint UniqueId,         // Unique placement ID
        Vector3 Position,      // World position
        Vector3 Rotation,      // Euler angles in degrees
        Vector3 BoundsMin,     // Transformed bounding box
        Vector3 BoundsMax,
        ushort Flags,
        ushort DoodadSet,
        ushort NameSet,
        ushort Scale,          // 1024 = 1.0
        string WmoPath,        // For reference
        string Ck24,           // PM4 object ID
        float MatchConfidence);

    /// <summary>
    /// Result of the reconstruction process.
    /// </summary>
    public record ReconstructionResult(
        List<ModfEntry> ModfEntries,
        List<string> WmoNames,      // MWMO string table
        List<string> UnmatchedPm4Objects,
        Dictionary<string, int> MatchCounts);

    /// <summary>
    /// Build a WMO reference library from extracted collision geometry.
    /// </summary>
    public List<WmoReference> BuildWmoLibrary(string wmoCollisionDir)
    {
        var library = new List<WmoReference>();

        if (!Directory.Exists(wmoCollisionDir))
        {
            Console.WriteLine($"[WARN] WMO collision directory not found: {wmoCollisionDir}");
            return library;
        }

        var legacyCollisionFiles = Directory.GetFiles(wmoCollisionDir, "*_collision.obj", SearchOption.AllDirectories);
        string[] objFiles;

        if (legacyCollisionFiles.Length > 0)
        {
            objFiles = legacyCollisionFiles;
            Console.WriteLine($"[INFO] Building WMO library from {objFiles.Length} legacy collision files (*_collision.obj)...");
        }
        else
        {
            objFiles = Directory.GetFiles(wmoCollisionDir, "*.obj", SearchOption.AllDirectories);
            Console.WriteLine($"[INFO] Building WMO library from {objFiles.Length} OBJ files (per-group/per-flag layout)...");
        }

        foreach (var objPath in objFiles)
        {
            try
            {
                var vertices = _matcher.LoadObjVertices(objPath);
                if (vertices.Count < 10) continue; // Skip tiny/empty files

                var stats = _matcher.ComputeStats(vertices);

                // Derive WMO path from filename / folder.
                // Legacy layout: <name>_collision.obj
                // New layout:   <root>/<WmoName>/<WmoName>_gXYZ_flags_XX.obj
                var fileName = Path.GetFileNameWithoutExtension(objPath);
                var dirName = Path.GetFileName(Path.GetDirectoryName(objPath)?.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) ?? string.Empty);

                string wmoName;
                if (fileName.EndsWith("_collision", StringComparison.OrdinalIgnoreCase))
                {
                    wmoName = fileName.Replace("_collision", "");
                }
                else if (!string.IsNullOrEmpty(dirName))
                {
                    // Prefer folder name when using per-WMO subfolders
                    wmoName = dirName;
                }
                else
                {
                    // Fallback: strip group/flag suffix if present
                    var idx = fileName.IndexOf("_g", StringComparison.OrdinalIgnoreCase);
                    wmoName = idx > 0 ? fileName.Substring(0, idx) : fileName;
                }

                var wmoPath = $"World\\wmo\\{wmoName}.wmo"; // Approximate path

                library.Add(new WmoReference(wmoPath, objPath, stats));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to process {objPath}: {ex.Message}");
            }
        }

        Console.WriteLine($"[INFO] WMO library contains {library.Count} entries");
        return library;
    }

    /// <summary>
    /// Load PM4 objects from PM4FacesTool output.
    /// Recursively finds all ck_instances.csv files and aggregates objects.
    /// </summary>
    public List<Pm4Object> LoadPm4Objects(string pm4FacesOutputDir)
    {
        var objects = new List<Pm4Object>();

        // Look for ALL ck_instances.csv files recursively
        var csvFiles = Directory.GetFiles(pm4FacesOutputDir, "ck_instances.csv", SearchOption.AllDirectories);
        
        if (csvFiles.Length == 0)
        {
            Console.WriteLine($"[WARN] No ck_instances.csv found in {pm4FacesOutputDir}");
            return objects;
        }

        Console.WriteLine($"[INFO] Found {csvFiles.Length} instance files to process...");

        foreach (var instancesCsv in csvFiles)
        {
            var baseDir = Path.GetDirectoryName(instancesCsv)!;
            Console.WriteLine($"[INFO] Loading PM4 objects from {instancesCsv}...");

            foreach (var line in File.ReadLines(instancesCsv).Skip(1)) // Skip header
            {
                var parts = line.Split(',');
                if (parts.Length < 6) continue;

                var ck24 = parts[0].Trim();
                var objRelPath = parts[5].Trim();
                var objPath = Path.Combine(baseDir, objRelPath);

                if (!File.Exists(objPath))
                {
                    // Only warn once per missing file to avoid spam
                    // Console.WriteLine($"[WARN] OBJ not found: {objPath}");
                    continue;
                }

                // Parse tile from path (e.g., objects\t15_37\ck42CBEA_merged.obj)
                int tileX = 0, tileY = 0;
                var pathParts = objRelPath.Split(Path.DirectorySeparatorChar, '/');
                foreach (var p in pathParts)
                {
                    if (p.StartsWith("t") && p.Contains("_"))
                    {
                        var tileParts = p.Substring(1).Split('_');
                        if (tileParts.Length >= 2)
                        {
                            int.TryParse(tileParts[0], out tileX);
                            int.TryParse(tileParts[1], out tileY);
                        }
                        break;
                    }
                }

                try
                {
                    var vertices = _matcher.LoadObjVertices(objPath);
                    if (vertices.Count < 10) continue; // Skip tiny objects

                    var stats = _matcher.ComputeStats(vertices);
                    objects.Add(new Pm4Object(ck24, objPath, tileX, tileY, stats));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Failed to load {objPath}: {ex.Message}");
                }
            }
        }

        Console.WriteLine($"[INFO] Loaded {objects.Count} total PM4 objects");
        return objects;
    }

    /// <summary>
    /// Match a PM4 object against the WMO library.
    /// </summary>
    public (WmoReference? bestMatch, Pm4WmoGeometryMatcher.PlacementTransform? transform) 
        MatchPm4ToWmo(Pm4Object pm4Obj, List<WmoReference> wmoLibrary, float minConfidence = 0.7f)
    {
        WmoReference? bestMatch = null;
        Pm4WmoGeometryMatcher.PlacementTransform? bestTransform = null;
        float bestConfidence = 0;

        foreach (var wmo in wmoLibrary)
        {
            try
            {
                var transform = _matcher.FindAlignment(pm4Obj.Stats, wmo.Stats);
                
                if (transform.MatchConfidence > bestConfidence)
                {
                    bestConfidence = transform.MatchConfidence;
                    bestMatch = wmo;
                    bestTransform = transform;
                }
            }
            catch
            {
                // Skip on error
            }
        }

        if (bestConfidence < minConfidence)
            return (null, null);

        return (bestMatch, bestTransform);
    }

    /// <summary>
    /// Quick match using principal extent similarity (faster than full alignment).
    /// WMOs cannot be scaled, so we compare extents directly at 1:1 scale.
    /// </summary>
    public float QuickMatchScore(Pm4WmoGeometryMatcher.GeometryStats pm4Stats, 
                                  Pm4WmoGeometryMatcher.GeometryStats wmoStats)
    {
        // Compare sorted principal extents
        var pm4Extents = pm4Stats.PrincipalExtents.OrderByDescending(x => x).ToArray();
        var wmoExtents = wmoStats.PrincipalExtents.OrderByDescending(x => x).ToArray();

        if (wmoExtents[0] < 0.001f) return 0;

        // WMOs are always scale 1.0 - compare extents directly
        // Compute normalized difference (how close are the sizes?)
        float diff1 = Math.Abs(pm4Extents[0] - wmoExtents[0]) / Math.Max(pm4Extents[0], wmoExtents[0]);
        float diff2 = Math.Abs(pm4Extents[1] - wmoExtents[1]) / Math.Max(pm4Extents[1], wmoExtents[1]);
        float diff3 = Math.Abs(pm4Extents[2] - wmoExtents[2]) / Math.Max(pm4Extents[2], wmoExtents[2]);

        // Average difference - lower is better
        float avgDiff = (diff1 + diff2 + diff3) / 3;

        // Score based on how close the extents match at 1:1 scale
        // 0% diff = 100% score, 50% diff = 0% score
        float score = Math.Max(0, 1 - avgDiff * 2);

        return score;
    }

    /// <summary>
    /// Reconstruct MODF entries for all PM4 objects in a map.
    /// </summary>
    public ReconstructionResult ReconstructModf(
        string pm4FacesOutputDir, 
        List<WmoReference> wmoLibrary,
        float minConfidence = 0.7f)
    {
        var pm4Objects = LoadPm4Objects(pm4FacesOutputDir);
        return ReconstructModf(pm4Objects, wmoLibrary, minConfidence);
    }

    /// <summary>
    /// Reconstruct MODF entries for a list of PM4 objects.
    /// </summary>
    public ReconstructionResult ReconstructModf(
        List<Pm4Object> pm4Objects, 
        List<WmoReference> wmoLibrary,
        float minConfidence = 0.7f)
    {
        var modfEntries = new List<ModfEntry>();
        var wmoNames = new List<string>();
        var wmoNameToId = new Dictionary<string, uint>();
        var unmatchedObjects = new List<string>();
        var matchCounts = new Dictionary<string, int>();

        const uint UniqueIdBase = 66_000_000;
        uint nextUniqueId = UniqueIdBase;

        Console.WriteLine($"\n[INFO] Matching {pm4Objects.Count} PM4 objects against {wmoLibrary.Count} WMOs...\n");

        int matched = 0;
        int unmatched = 0;

        foreach (var pm4Obj in pm4Objects)
        {
            // Quick pre-filter using extent ratios
            var candidates = wmoLibrary
                .Select(w => (wmo: w, score: QuickMatchScore(pm4Obj.Stats, w.Stats)))
                .Where(x => x.score > 0.5f)
                .OrderByDescending(x => x.score)
                .Take(5) // Top 5 candidates
                .Select(x => x.wmo)
                .ToList();

            if (candidates.Count == 0)
            {
                unmatchedObjects.Add(pm4Obj.Ck24);
                unmatched++;
                continue;
            }

            var (bestMatch, transform) = MatchPm4ToWmo(pm4Obj, candidates, minConfidence);

            if (bestMatch == null || transform == null)
            {
                unmatchedObjects.Add(pm4Obj.Ck24);
                unmatched++;
                continue;
            }

            matched++;

            // Get or create WMO name ID
            if (!wmoNameToId.TryGetValue(bestMatch.WmoPath, out var nameId))
            {
                nameId = (uint)wmoNames.Count;
                wmoNames.Add(bestMatch.WmoPath);
                wmoNameToId[bestMatch.WmoPath] = nameId;
            }

            // Track match counts
            if (!matchCounts.ContainsKey(bestMatch.WmoPath))
                matchCounts[bestMatch.WmoPath] = 0;
            matchCounts[bestMatch.WmoPath]++;

            // Use PM4/server-space placement and bounds; PM4->ADT transform is applied at the CLI layer
            // when exporting via RunPm4ReconstructModf.
            var boundsMin = pm4Obj.Stats.BoundsMin;
            var boundsMax = pm4Obj.Stats.BoundsMax;

            var modf = new ModfEntry(
                NameId: nameId,
                UniqueId: nextUniqueId++,
                Position: transform.Position,
                Rotation: transform.Rotation,
                BoundsMin: boundsMin,
                BoundsMax: boundsMax,
                Flags: 0,
                DoodadSet: 0,
                NameSet: 0,
                Scale: (ushort)(transform.Scale * 1024),
                WmoPath: bestMatch.WmoPath,
                Ck24: pm4Obj.Ck24,
                MatchConfidence: transform.MatchConfidence);

            modfEntries.Add(modf);

            Console.WriteLine($"  {pm4Obj.Ck24} -> {Path.GetFileName(bestMatch.WmoPath)} ({transform.MatchConfidence:P0})");
        }

        Console.WriteLine($"\n[RESULT] Matched: {matched}, Unmatched: {unmatched}");

        return new ReconstructionResult(modfEntries, wmoNames, unmatchedObjects, matchCounts);
    }

    /// <summary>
    /// Export reconstruction results to CSV for analysis.
    /// </summary>
    public void ExportToCsv(ReconstructionResult result, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence");

        foreach (var entry in result.ModfEntries)
        {
            sw.WriteLine(string.Join(",",
                entry.Ck24,
                entry.WmoPath,
                entry.NameId,
                entry.UniqueId,
                entry.Position.X.ToString("F2"),
                entry.Position.Y.ToString("F2"),
                entry.Position.Z.ToString("F2"),
                entry.Rotation.X.ToString("F2"),
                entry.Rotation.Y.ToString("F2"),
                entry.Rotation.Z.ToString("F2"),
                (entry.Scale / 1024f).ToString("F4"),
                entry.MatchConfidence.ToString("F3")));
        }

        Console.WriteLine($"[INFO] Exported {result.ModfEntries.Count} MODF entries to {outputPath}");
    }

    /// <summary>
    /// Export MWMO string table.
    /// </summary>
    public void ExportMwmo(ReconstructionResult result, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("name_id,wmo_path,placement_count");

        for (int i = 0; i < result.WmoNames.Count; i++)
        {
            var path = result.WmoNames[i];
            var count = result.MatchCounts.GetValueOrDefault(path, 0);
            sw.WriteLine($"{i},{path},{count}");
        }

        Console.WriteLine($"[INFO] Exported {result.WmoNames.Count} WMO names to {outputPath}");
    }

    /// <summary>
    /// Export PLACEMENT-FOCUSED verification JSON - just the data we need to verify.
    /// Shows exactly what MODF entries will be written to ADTs.
    /// </summary>
    public void ExportVerificationJson(
        ReconstructionResult result,
        List<Pm4Object> pm4Objects,
        List<WmoReference> wmoLibrary,
        string outputPath)
    {
        // PLACEMENT-FOCUSED: Only include what matters for verification
        var verification = new PlacementVerificationReport
        {
            GeneratedAt = DateTime.UtcNow.ToString("O"),
            Summary = new PlacementSummary
            {
                TotalPm4ObjectsProcessed = pm4Objects.Count,
                TotalWmosInLibrary = wmoLibrary.Count,
                MatchedPlacements = result.ModfEntries.Count,
                UnmatchedObjects = result.UnmatchedPm4Objects.Count,
                UniqueWmosUsed = result.WmoNames.Count,
                TilesWithPlacements = result.ModfEntries
                    .GroupBy(e => GetTileForPosition(e.Position))
                    .Count()
            },
            // MWMO string table - exactly what gets written to ADT
            MwmoChunkData = result.WmoNames.Select((name, idx) => new MwmoChunkEntry
            {
                NameId = idx,
                WmoPath = name,
                PlacementCount = result.MatchCounts.GetValueOrDefault(name, 0)
            }).ToList(),
            // ALL MODF placements - exactly what gets written to ADT
            ModfEntries = result.ModfEntries.Select(e => new ModfChunkEntry
            {
                Pm4ObjectId = e.Ck24,
                WmoPath = e.WmoPath,
                NameId = e.NameId,
                UniqueId = e.UniqueId,
                PositionX = e.Position.X,
                PositionY = e.Position.Y,
                PositionZ = e.Position.Z,
                RotationX = e.Rotation.X,
                RotationY = e.Rotation.Y,
                RotationZ = e.Rotation.Z,
                Scale = e.Scale / 1024f,
                MatchConfidence = e.MatchConfidence,
                TileX = GetTileForPosition(e.Position).X,
                TileY = GetTileForPosition(e.Position).Y
            }).ToList(),
            // Per-tile breakdown for easy verification
            PlacementsByTile = result.ModfEntries
                .GroupBy(e => GetTileForPosition(e.Position))
                .OrderBy(g => g.Key.X).ThenBy(g => g.Key.Y)
                .Select(g => new TilePlacementSummary
                {
                    TileX = g.Key.X,
                    TileY = g.Key.Y,
                    PlacementCount = g.Count(),
                    WmoList = g.Select(e => $"{e.Ck24} -> {Path.GetFileName(e.WmoPath)} ({e.MatchConfidence:P0})").ToList()
                }).ToList(),
            // Unmatched for debugging
            UnmatchedPm4Objects = result.UnmatchedPm4Objects
        };

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var json = JsonSerializer.Serialize(verification, options);
        File.WriteAllText(outputPath, json);

        Console.WriteLine($"\n========================================");
        Console.WriteLine($"[PLACEMENT VERIFICATION] {outputPath}");
        Console.WriteLine($"========================================");
        Console.WriteLine($"  PM4 Objects:       {verification.Summary.TotalPm4ObjectsProcessed}");
        Console.WriteLine($"  Matched:           {verification.Summary.MatchedPlacements}");
        Console.WriteLine($"  Unmatched:         {verification.Summary.UnmatchedObjects}");
        Console.WriteLine($"  Unique WMOs:       {verification.Summary.UniqueWmosUsed}");
        Console.WriteLine($"  Tiles with data:   {verification.Summary.TilesWithPlacements}");
        Console.WriteLine($"========================================\n");
        
        // Also print first few placements as proof
        Console.WriteLine("First 10 placements:");
        foreach (var entry in verification.ModfEntries.Take(10))
        {
            Console.WriteLine($"  {entry.Pm4ObjectId} -> {Path.GetFileName(entry.WmoPath)}");
            Console.WriteLine($"    Position: ({entry.PositionX:F1}, {entry.PositionY:F1}, {entry.PositionZ:F1})");
            Console.WriteLine($"    Tile: ({entry.TileX}, {entry.TileY}), Confidence: {entry.MatchConfidence:P0}");
        }
    }

    private static float[] Vec3ToArray(Vector3 v) => new[] { v.X, v.Y, v.Z };

    /// <summary>
    /// Convert PM4/server-space coordinates (origin near a map corner, range ~[0 .. 64 * TileSize])
    /// into ADT world coordinates as described in ADT_v18 (origin at map center, range
    /// approximately [-17066.66 .. +17066.66]).
    /// </summary>
    private static Vector3 ServerToAdtPosition(Vector3 server)
    {
        const float TileSize = 533.33333f;
        const float HalfMap = TileSize * 32f; // 32 tiles from center to edge

        return new Vector3(
            server.X - HalfMap,
            server.Y - HalfMap,
            server.Z);
    }

    private static (int X, int Y) GetTileForPosition(Vector3 pos)
    {
        const float TileSize = 533.33333f;
        int x = (int)(32 - (pos.X / TileSize));
        int y = (int)(32 - (pos.Y / TileSize));
        return (x, y);
    }

    #region Placement Verification JSON Models

    public class PlacementVerificationReport
    {
        public string GeneratedAt { get; set; } = "";
        public PlacementSummary Summary { get; set; } = new();
        public List<MwmoChunkEntry> MwmoChunkData { get; set; } = new();
        public List<ModfChunkEntry> ModfEntries { get; set; } = new();
        public List<TilePlacementSummary> PlacementsByTile { get; set; } = new();
        public List<string> UnmatchedPm4Objects { get; set; } = new();
    }

    public class PlacementSummary
    {
        public int TotalPm4ObjectsProcessed { get; set; }
        public int TotalWmosInLibrary { get; set; }
        public int MatchedPlacements { get; set; }
        public int UnmatchedObjects { get; set; }
        public int UniqueWmosUsed { get; set; }
        public int TilesWithPlacements { get; set; }
    }

    public class MwmoChunkEntry
    {
        public int NameId { get; set; }
        public string WmoPath { get; set; } = "";
        public int PlacementCount { get; set; }
    }

    public class ModfChunkEntry
    {
        public string Pm4ObjectId { get; set; } = "";
        public string WmoPath { get; set; } = "";
        public uint NameId { get; set; }
        public uint UniqueId { get; set; }
        public float PositionX { get; set; }
        public float PositionY { get; set; }
        public float PositionZ { get; set; }
        public float RotationX { get; set; }
        public float RotationY { get; set; }
        public float RotationZ { get; set; }
        public float Scale { get; set; }
        public float MatchConfidence { get; set; }
        public int TileX { get; set; }
        public int TileY { get; set; }
    }

    public class TilePlacementSummary
    {
        public int TileX { get; set; }
        public int TileY { get; set; }
        public int PlacementCount { get; set; }
        public List<string> WmoList { get; set; } = new();
    }

    #endregion
}
