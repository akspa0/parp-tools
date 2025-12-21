using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using WoWRollback.Core.Services.Archive;

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
        Pm4WmoGeometryMatcher.GeometryStats Stats);

    /// <summary>
    /// Represents an M2 in the reference library.
    /// </summary>
    public record M2Reference(
        uint FileDataId,
        string M2Path,
        string FileName,
        Pm4WmoGeometryMatcher.GeometryStats Stats);

    /// <summary>
    /// Represents a PM4 object with geometry stats.
    /// </summary>
    public record Pm4Object(
        string Ck24,
        string ObjPath,
        int TileX,
        int TileY,
        Pm4WmoGeometryMatcher.GeometryStats Stats,
        List<Vector3>? MscnPoints = null,           // MSCN points for verification
        float? MprlRotationDegrees = null,          // MPRL rotation (0-360 degrees)
        Vector3? MprlPosition = null);              // MPRL placement position

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
        float MatchConfidence,
        int TileX,             // Tile X from PM4 filename
        int TileY);            // Tile Y from PM4 filename

    /// <summary>
    /// Reconstructed MDDF entry (M2 placement).
    /// </summary>
    public record MddfEntry(
        uint NameId,           // Index into MMDX string table
        uint UniqueId,
        Vector3 Position,
        Vector3 Rotation,      // Euler angles
        ushort Scale,          // 1024 = 1.0 (Fixed point)
        ushort Flags,
        string M2Path,
        string Ck24,
        float MatchConfidence,
        int TileX,
        int TileY);

    /// <summary>
    /// Represents a potential match candidate for a PM4 object.
    /// </summary>
    public record MatchCandidate(
        string Pm4Ck24,
        string WmoPath,
        float Confidence,
        Vector3 Position,
        Vector3 Rotation,
        float Scale
    );

    /// <summary>
    /// Result of the reconstruction process.
    /// </summary>
    public record ReconstructionResult(
        List<ModfEntry> ModfEntries,
        List<string> WmoNames,      // MWMO string table
        List<string> UnmatchedPm4Objects,
        Dictionary<string, int> MatchCounts,
        List<MatchCandidate> AllCandidates); // New: All valid candidates

    /// <summary>
    /// Result of M2 reconstruction.
    /// </summary>
    public record MddfReconstructionResult(
        List<MddfEntry> MddfEntries,
        List<string> M2Names,
        List<string> UnmatchedPm4Objects,
        Dictionary<string, int> MatchCounts,
        List<MatchCandidate> AllCandidates);

    /// <summary>
    /// Builds the WMO reference library from the provided game data path.
    /// Extracts, converts, and analyzes WMOs (in-memory) to build fingerprints.
    /// Uses caching to avoid reprocessing.
    /// </summary>
    /// <param name="wmoPathFilter">Optional filter: only include WMOs with paths containing this string (e.g., "Northrend")</param>
    /// <param name="useFullMesh">If true, use all WMO geometry; if false, use only walkable surfaces</param>
    public List<WmoReference> BuildWmoLibrary(string gamePath, string listfilePath, string outputRoot, string? wmoPathFilter = null, bool useFullMesh = false)
    {
        Console.WriteLine("=== Building WMO Reference Library ===\n");
        
        // Cache path includes filter and mesh mode to avoid mixing different library builds
        string cacheKey = $"{wmoPathFilter ?? "all"}_{(useFullMesh ? "full" : "walkable")}";
        string cachePath = Path.Combine(outputRoot, $"wmo_library_cache_{cacheKey.Replace("/", "_").Replace("\\", "_")}.json");

        // 1. Try Cache
        if (File.Exists(cachePath))
        {
            try
            {
                var cached = LoadLibraryCache(cachePath);
                if (cached != null && cached.Count > 0)
                {
                    Console.WriteLine($"[INFO] Loaded {cached.Count} WMOs from cache.");
                    return cached;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Cache load failed: {ex.Message}. Rebuilding.");
            }
        }

        // 2. Build Library (In-Memory)
        Console.WriteLine("Scanning listfile for WMOs...");
        if (!File.Exists(listfilePath)) throw new FileNotFoundException("Listfile not found", listfilePath);

        var allWmos = File.ReadLines(listfilePath)
            .Where(l => l.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase) 
                && !System.Text.RegularExpressions.Regex.IsMatch(l, @"_\d{3}\.wmo$", System.Text.RegularExpressions.RegexOptions.IgnoreCase)) // Exclude group files
            .ToList();

        // Apply path filter if specified
        if (!string.IsNullOrEmpty(wmoPathFilter))
        {
            var originalCount = allWmos.Count;
            allWmos = allWmos.Where(p => p.Contains(wmoPathFilter, StringComparison.OrdinalIgnoreCase)).ToList();
            Console.WriteLine($"Filtered WMOs: {originalCount} -> {allWmos.Count} (containing '{wmoPathFilter}')");
        }

        Console.WriteLine($"Found {allWmos.Count} candidate WMOs in listfile.");
        
        var references = new ConcurrentBag<WmoReference>();
        
        // Find all MPQs in Data directory
        var dataDir = Path.Combine(gamePath, "Data");
        var mpqFiles = Directory.Exists(dataDir) 
            ? Directory.GetFiles(dataDir, "*.MPQ", SearchOption.AllDirectories)
            : Array.Empty<string>();
            
        if (mpqFiles.Length == 0)
        {
             Console.WriteLine($"[WARN] No MPQ files found in {dataDir}. WMOs cannot be extracted.");
             // Return empty or throw? Return empty allows cache to save empty result which is bad.
             // But valid if user pointed to wrong dir.
        }

        using var archiveSource = new MpqArchiveSource(mpqFiles);
        int processed = 0;
        int failed = 0;

        Parallel.ForEach(allWmos, new ParallelOptions { MaxDegreeOfParallelism = 8 }, wmoPath =>
        {
            try
            {
                // Extract WMO bytes
                var wmoBytes = ReadAllBytes(archiveSource, wmoPath);
                if (wmoBytes == null) return;

                // Define group loader for collision extraction
                // The extractor passes the full group path (e.g. "World/wmo/.../root_000.wmo")
                Func<string, byte[]?> groupLoader = (groupPath) =>
                {
                    return ReadAllBytes(archiveSource, groupPath);
                };

                // Extract geometry
                // improved-matching: If NOT using full mesh, we strictly want footprints (floors) for better rotation alignment
                bool onlyFootprint = !useFullMesh; 
                var walkableData = WmoWalkableSurfaceExtractor.ExtractFromBytes(wmoBytes, wmoPath, groupLoader, onlyFootprint);
                
                // Choose vertices based on mesh mode
                var vertices = useFullMesh ? walkableData.AllVertices : walkableData.WalkableVertices;
                if (vertices.Count < 3) return;

                // Compute Stats
                var stats = _matcher.ComputeStats(vertices);
                references.Add(new WmoReference(wmoPath.Replace('/', '\\'), stats)); // Standardize path separator

                System.Threading.Interlocked.Increment(ref processed);
                if (processed % 50 == 0) Console.Write(".");
            }
            catch
            {
                System.Threading.Interlocked.Increment(ref failed);
            }
        });
        
        Console.WriteLine($"\nProcessed {processed} WMOs. ({failed} failed)");
        var refList = references.ToList();

        // 3. Save Cache
        SaveLibraryCache(refList, cachePath);
        
        return refList;
    }

    private byte[]? ReadAllBytes(IArchiveSource source, string path)
    {
        try
        {
            if (source.FileExists(path))
            {
                using var stream = source.OpenFile(path);
                using var ms = new MemoryStream();
                stream.CopyTo(ms);
                return ms.ToArray();
            }
        }
        catch { }
        return null; // File not found or error
    }

    private void SaveLibraryCache(List<WmoReference> references, string path)
    {
        Console.WriteLine("[INFO] Saving WMO library cache...");
        var options = new JsonSerializerOptions { WriteIndented = true, IncludeFields = true };
        var json = JsonSerializer.Serialize(references, options);
        File.WriteAllText(path, json);
    }

    private List<WmoReference>? LoadLibraryCache(string path)
    {
        var json = File.ReadAllText(path);
        var options = new JsonSerializerOptions { IncludeFields = true };
        return JsonSerializer.Deserialize<List<WmoReference>>(json, options);
    }

    public List<M2Reference> LoadM2Library(string cachePath)
    {
        if (!File.Exists(cachePath))
        {
            Console.WriteLine($"[WARN] M2 library cache not found at {cachePath}");
            return new List<M2Reference>();
        }

        try 
        {
            var json = File.ReadAllText(cachePath);
            var options = new JsonSerializerOptions { IncludeFields = true };
            var list = JsonSerializer.Deserialize<List<M2Reference>>(json, options);
            Console.WriteLine($"[INFO] Loaded {list?.Count ?? 0} M2s from library.");
            return list ?? new List<M2Reference>();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] Failed to load M2 library: {ex.Message}");
            return new List<M2Reference>();
        }
    }

    /// <summary>
    /// Load WMO path mapping from a listfile (CSV or line-delimited).
    /// Returns a dictionary of WMO Name (without extension) -> Full Path.
    /// </summary>
    public Dictionary<string, string> LoadWmoPathMapping(string listfilePath)
    {
        var mapping = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (!File.Exists(listfilePath))
        {
            Console.WriteLine($"[WARN] Listfile not found at {listfilePath}. Path resolution will rely on guessing.");
            return mapping;
        }

        Console.WriteLine($"[INFO] Loading WMO path mapping from {listfilePath}...");
        
        foreach (var line in File.ReadLines(listfilePath))
        {
            // Handle both CSV (ID;Path) and simple list (Path) formats
            string path = line;
            if (line.Contains(";"))
            {
                var parts = line.Split(';');
                if (parts.Length > 1)
                    path = parts[1]; // Assume ID;Path
                else
                    path = parts[0];
            }
            else if (line.Contains(",")) // Possible ID,Path
            {
                 // Naive check - if it looks like an integer at start, split
                 var parts = line.Split(',');
                 if (parts.Length > 1 && int.TryParse(parts[0], out _))
                    path = parts[1];
                 else
                    path = parts[0]; // Maybe just a path with comma? Unlikely but safe fallback
            }

            path = path.Trim();
            
            if (path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
            {
                // Key is the filename without extension (e.g. "AltarOfStorms")
                // Value is the full path (e.g. "world/wmo/azeroth/buildings/altarofstorms/altarofstorms.wmo")
                var name = Path.GetFileNameWithoutExtension(path);
                
                // If duplicates exist (same name, different folder), we might overwrite. 
                // For valid WMO referencing, names should generally be unique or context-implied.
                // In strict ADT usage, internal references are by filename, so specific dir doesn't strictly matter 
                // UNLESS names collide. We'll take the first or last valid one.
                if (!mapping.ContainsKey(name))
                {
                    mapping[name] = path.Replace('/', '\\');
                }
            }
        }
        
        Console.WriteLine($"[INFO] Loaded mappings for {mapping.Count} WMOs.");
        return mapping;
    }

    /// <summary>
    /// Load PM4 objects from PM4FacesTool output.
    /// Recursively finds all ck_instances.csv files and aggregates objects.
    /// NOTE: For direct .pm4 file parsing without PM4FacesTool, use
    /// PipelineService.LoadPm4ObjectsFromFiles() instead.
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

                // Parse tile from path (e.g., objects\t15_37\ck42CBEA_merged.obj OR development_15_37\ck...)
                int tileX = 0, tileY = 0;
                var pathParts = objRelPath.Split(Path.DirectorySeparatorChar, '/');
                foreach (var p in pathParts)
                {
                    // Match pattern: *XX_YY where XX and YY are 1-2 digits
                    // Regex is cleaner but manual parsing is fine.
                    // Look for the LAST underscore to find YY, then previous underscore for XX.
                    
                    if (p.Contains('_'))
                    {
                        var tilePathParts = p.Split('_');
                        if (tilePathParts.Length >= 2)
                        {
                            // Try parsing last two parts as numbers
                            if (int.TryParse(tilePathParts[tilePathParts.Length - 2], out int tX) && 
                                int.TryParse(tilePathParts[tilePathParts.Length - 1], out int tY))
                            {
                                tileX = tX;
                                tileY = tY;
                                
                                // Basic validation (0-63)
                                if (tileX >= 0 && tileX <= 63 && tileY >= 0 && tileY <= 63)
                                {
                                    break;
                                }
                            }
                        }
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
        return NormalizeCoordinates(objects);
    }

    // NOTE: LoadPm4ObjectsDirect was removed because WoWRollback.Core cannot
    // reference WoWRollback.PM4Module (circular dependency).
    // Use Pm4ObjectExtractor from PM4Module.Pipeline directly in CLI instead.

    /// <summary>
    /// Normalize PM4 object coordinates to standard WoW global coordinates.
    /// Handles Local->Global conversion by mapping PM4 Global (Corner-Relative, Inverted-X) to WoW Global (Center-Relative).
    /// </summary>
    private List<Pm4Object> NormalizeCoordinates(List<Pm4Object> objects)
    {
        Console.WriteLine("[INFO] Normalizing coordinates: PM4 Global -> WoW Global...");
        var normalized = new List<Pm4Object>();
        const float CenterOffset = 17066.6666f; // 32 * 533.33333f
        
        foreach (var obj in objects)
        {


            // PM4FacesTool output is generally Z-Up (X, Y, Z_Height) but with some transforms applied.
            // Specifically, X is flipped (-X) and coordinates are relative to map corner (0..34133 scale).
            
            var pos = obj.Stats.Centroid;
            var boundsMin = obj.Stats.BoundsMin;
            var boundsMax = obj.Stats.BoundsMax;
            
            // Formula derived:
            // WoW_X = CenterOffset + Input_X (Input X is negative due to flip, so this subtracts magnitude)
            // WoW_Y = CenterOffset - Input_Y
            // WoW_Z = Input_Z (Height)
            
            // We use 533.33333f * 32 as the center offset.
            
            float finalX = CenterOffset + pos.X;
            float finalY = CenterOffset - pos.Y;
            float finalZ = pos.Z; 
            
            var newPos = new Vector3(finalX, finalY, finalZ);
            
            // Transform Bounds:

            // X axis: X' = C + X (Preserves direction if X was -Recast)
            // Y axis: Y' = C - Y (Inverts direction)
            
            // For Y, Min becomes Max and Max becomes Min.
            
            var newBoundsMin = new Vector3(
                CenterOffset + boundsMin.X,
                CenterOffset - boundsMax.Y,
                boundsMin.Z
            );
            
            var newBoundsMax = new Vector3(
                CenterOffset + boundsMax.X,
                CenterOffset - boundsMin.Y,
                boundsMax.Z
            );
            
            // Ensure Min < Max after inversion
            var finalMin = Vector3.Min(newBoundsMin, newBoundsMax);
            var finalMax = Vector3.Max(newBoundsMin, newBoundsMax);
            
            var newStats = obj.Stats with { 
                Centroid = newPos, 
                BoundsMin = finalMin, 
                BoundsMax = finalMax
            };
            
            normalized.Add(obj with { Stats = newStats });
        }
        
        return normalized;
    }

    /// <summary>
    /// Find ALL valid matches for a PM4 object against the WMO library.
    /// Returns best match (if any) and list of all candidates > minConfidence.
    /// </summary>
    public (WmoReference? bestMatch, Pm4WmoGeometryMatcher.PlacementTransform? bestTransform, List<MatchCandidate> candidates) 
        MatchPm4ToWmo(Pm4Object pm4Obj, List<WmoReference> wmoLibrary, float minConfidence = 0.1f)
    {
        WmoReference? bestMatch = null;
        Pm4WmoGeometryMatcher.PlacementTransform? bestTransform = null;
        float bestConfidence = 0;
        string? bestWmoPath = null;
        
        var candidates = new List<MatchCandidate>();

        foreach (var wmo in wmoLibrary)
        {
            try
            {
                // Use simple dimension matching (rotation-agnostic)
                float score = SimpleDimensionMatchScore(pm4Obj.Stats, wmo.Stats);
                
                if (score >= minConfidence)
                {
                    // Use MPRL position/rotation directly instead of computing from geometry
                    var position = pm4Obj.MprlPosition ?? pm4Obj.Stats.Centroid;
                    var rotation = pm4Obj.MprlRotationDegrees.HasValue 
                        ? new Vector3(0, pm4Obj.MprlRotationDegrees.Value, 0)  // MPRL is yaw only
                        : Vector3.Zero;
                    
                    candidates.Add(new MatchCandidate(
                        pm4Obj.Ck24,
                        wmo.WmoPath,
                        score,
                        position,
                        rotation,
                        1.0f  // WMOs are always scale 1.0
                    ));

                    if (score > bestConfidence)
                    {
                        bestConfidence = score;
                        bestMatch = wmo;
                        bestTransform = new Pm4WmoGeometryMatcher.PlacementTransform(
                            position, rotation, 1.0f, score);
                    }
                }
            }
            catch
            {
                // Skip on error
            }
        }

        return (bestMatch, bestTransform, candidates);
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
    /// 3D dimension-based matching - compares sorted XYZ dimensions.
    /// Now includes MSCN wall data for full collision volume.
    /// </summary>
    public float SimpleDimensionMatchScore(Pm4WmoGeometryMatcher.GeometryStats pm4Stats, 
                                            Pm4WmoGeometryMatcher.GeometryStats wmoStats)
    {
        // Compare all 3 dimensions (PM4 now includes MSCN walls for full 3D volume)
        // Sort dimensions to make rotation-agnostic
        var pm4Dims = new[] { pm4Stats.Dimensions.X, pm4Stats.Dimensions.Y, pm4Stats.Dimensions.Z }
            .OrderByDescending(x => x).ToArray();
        var wmoDims = new[] { wmoStats.Dimensions.X, wmoStats.Dimensions.Y, wmoStats.Dimensions.Z }
            .OrderByDescending(x => x).ToArray();

        // Skip tiny objects
        if (wmoDims[0] < 0.5f || pm4Dims[0] < 0.5f) return 0;

        // Compare all 3 sorted dimensions
        float diff1 = Math.Abs(pm4Dims[0] - wmoDims[0]) / Math.Max(pm4Dims[0], wmoDims[0]);
        float diff2 = Math.Abs(pm4Dims[1] - wmoDims[1]) / Math.Max(pm4Dims[1], wmoDims[1]);
        float diff3 = Math.Abs(pm4Dims[2] - wmoDims[2]) / Math.Max(pm4Dims[2], wmoDims[2]);

        // Average difference - lower is better
        float avgDiff = (diff1 + diff2 + diff3) / 3;

        // Score: 0% diff = 100%, 50% diff = 0%
        float score = Math.Max(0, 1 - avgDiff * 2);

        return score;
    }

    /// <summary>
    /// Reconstruct MODF entries for all PM4 objects in a map.
    /// </summary>
    public ReconstructionResult ReconstructModf(
        string pm4FacesOutputDir, 
        List<WmoReference> wmoLibrary,
        float minConfidence = 0.7f,
        uint startingUniqueId = 66_000_000)
    {
        var pm4Objects = LoadPm4Objects(pm4FacesOutputDir);
        return ReconstructModf(pm4Objects, wmoLibrary, minConfidence, startingUniqueId);
    }

    /// <summary>
    /// Reconstruct MODF entries for a list of PM4 objects.
    /// </summary>
    public ReconstructionResult ReconstructModf(
        List<Pm4Object> pm4Objects, 
        List<WmoReference> wmoLibrary,
        float minConfidence = 0.7f,
        uint startingUniqueId = 66_000_000)
    {
        var modfEntries = new List<ModfEntry>();
        var wmoNames = new List<string>();
        var wmoNameToId = new Dictionary<string, uint>();
        var unmatchedObjects = new List<string>();
        var matchCounts = new Dictionary<string, int>();

        uint nextUniqueId = startingUniqueId;

        Console.WriteLine($"\n[INFO] Matching {pm4Objects.Count} PM4 objects against {wmoLibrary.Count} WMOs...\n");

        int matched = 0;
        int unmatched = 0;

        var allCandidates = new List<MatchCandidate>();

        foreach (var pm4Obj in pm4Objects)
        {
            // Quick pre-filter using extent ratios
            var potentialMatches = wmoLibrary
                .Select(w => (wmo: w, score: QuickMatchScore(pm4Obj.Stats, w.Stats)))
                .Where(x => x.score > 0.92f)
                .OrderByDescending(x => x.score)
                .Take(10) // Increased to Top 10 for broader search
                .Select(x => x.wmo)
                .ToList();

            if (potentialMatches.Count == 0)
            {
                unmatchedObjects.Add(pm4Obj.Ck24);
                unmatched++;
                continue;
            }

            // Find all candidates
            var (bestMatch, transform, candidates) = MatchPm4ToWmo(pm4Obj, potentialMatches, minConfidence);
            
            // Add all valid candidates to the global list
            allCandidates.AddRange(candidates);

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

            // Use MPRL data from PM4 when available (this is the ACTUAL placement data!)
            // Fall back to computed values only when MPRL is not found
            var position = pm4Obj.MprlPosition ?? transform.Position;
            var rotation = pm4Obj.MprlRotationDegrees.HasValue 
                ? new Vector3(0, pm4Obj.MprlRotationDegrees.Value, 0)
                : Vector3.Zero;  // No rotation if no MPRL data
            
            bool usedMprl = pm4Obj.MprlPosition.HasValue;
            
            var modf = new ModfEntry(
                NameId: nameId,
                UniqueId: nextUniqueId++,
                Position: position,
                Rotation: rotation,
                BoundsMin: boundsMin,
                BoundsMax: boundsMax,
                Flags: 0,
                DoodadSet: 0,
                NameSet: 0,
                Scale: (ushort)(transform.Scale * 1024),
                WmoPath: bestMatch.WmoPath,
                Ck24: pm4Obj.Ck24,
                MatchConfidence: transform.MatchConfidence,
                TileX: pm4Obj.TileX,
                TileY: pm4Obj.TileY);

            modfEntries.Add(modf);

            // Log best match
            Console.WriteLine($"  {pm4Obj.Ck24} -> {Path.GetFileName(bestMatch.WmoPath)} ({transform.MatchConfidence:P0})");
            // Log other candidates count
            if (candidates.Count > 1)
                Console.WriteLine($"    + {candidates.Count - 1} other candidates");
        }

        Console.WriteLine($"\n[RESULT] Matched: {matched}, Unmatched: {unmatched}, Total Candidates: {allCandidates.Count}");

        return new ReconstructionResult(modfEntries, wmoNames, unmatchedObjects, matchCounts, allCandidates);
    }

    /// <summary>
    /// Reconstruct MDDF entries (M2 placements) for a list of PM4 objects.
    /// </summary>
    public MddfReconstructionResult ReconstructMddf(
        List<Pm4Object> pm4Objects, 
        List<M2Reference> m2Library,
        float minConfidence = 0.7f,
        uint startingUniqueId = 11_000_000)
    {
        var mddfEntries = new List<MddfEntry>();
        var m2Names = new List<string>();
        var m2NameToId = new Dictionary<string, uint>();
        var unmatchedObjects = new List<string>();
        var matchCounts = new Dictionary<string, int>();

        uint nextUniqueId = startingUniqueId; // Distinct from WMO base

        Console.WriteLine($"\n[INFO] Matching {pm4Objects.Count} PM4 objects against {m2Library.Count} M2s...\n");

        int matched = 0;
        int unmatched = 0;
        var allCandidates = new List<MatchCandidate>();

        foreach (var pm4Obj in pm4Objects)
        {
            // Pre-filter: Matches must have similar Aspect Ratios (since scale varies but shape doesn't).
            // M2 QuickMatchScore uses scale-invariant metric.
            
            var potentialMatches = m2Library
                .Select(m => (m2: m, score: QuickMatchScoreM2(pm4Obj.Stats, m.Stats)))
                .Where(x => x.score > 0.85f) // Slightly lower threshold for M2 due to scale var
                .OrderByDescending(x => x.score)
                .Take(20) // Broader search for M2s
                .Select(x => x.m2)
                .ToList();

            if (potentialMatches.Count == 0)
            {
                unmatchedObjects.Add(pm4Obj.Ck24);
                unmatched++;
                continue;
            }

            // Find best match allowing scaling
            var (bestMatch, transform, candidates) = MatchPm4ToM2(pm4Obj, potentialMatches, minConfidence);
            allCandidates.AddRange(candidates);

            if (bestMatch == null || transform == null)
            {
                unmatchedObjects.Add(pm4Obj.Ck24);
                unmatched++;
                continue;
            }

            matched++;

            // M2 Name ID
            if (!m2NameToId.TryGetValue(bestMatch.M2Path, out var nameId))
            {
                nameId = (uint)m2Names.Count;
                m2Names.Add(bestMatch.M2Path);
                m2NameToId[bestMatch.M2Path] = nameId;
            }

            if (!matchCounts.ContainsKey(bestMatch.M2Path)) matchCounts[bestMatch.M2Path] = 0;
            matchCounts[bestMatch.M2Path]++;

            // MDDF Scale is fixed point 1024 = 1.0
            ushort scaleFixed = (ushort)(Math.Clamp(transform.Scale, 0.001f, 10.0f) * 1024);

            var mddf = new MddfEntry(
                NameId: nameId,
                UniqueId: nextUniqueId++,
                Position: ServerToAdtPosition(transform.Position),  // Use original position (tile swap should handle placement)
                Rotation: new Vector3(transform.Rotation.Y, transform.Rotation.Z, transform.Rotation.X), // Map Pitch->RotX, Yaw->RotY, Roll->RotZ
                Scale: scaleFixed,
                Flags: 0,
                M2Path: bestMatch.M2Path,
                Ck24: pm4Obj.Ck24,
                MatchConfidence: transform.MatchConfidence,
                TileX: pm4Obj.TileX,  // Original - no swap
                TileY: pm4Obj.TileY); // Original - no swap

            mddfEntries.Add(mddf);
            Console.WriteLine($"  {pm4Obj.Ck24} -> {bestMatch.FileName} ({transform.MatchConfidence:P0}, Scale: {transform.Scale:F2})");
        }

        Console.WriteLine($"\n[RESULT] Matched M2s: {matched}, Unmatched: {unmatched}");
        return new MddfReconstructionResult(mddfEntries, m2Names, unmatchedObjects, matchCounts, allCandidates);
    }

    private (M2Reference? bestMatch, Pm4WmoGeometryMatcher.PlacementTransform? bestTransform, List<MatchCandidate> candidates) 
        MatchPm4ToM2(Pm4Object pm4Obj, List<M2Reference> library, float minConfidence)
    {
        M2Reference? bestMatch = null;
        Pm4WmoGeometryMatcher.PlacementTransform? bestTransform = null;
        float bestConfidence = 0;
        var candidates = new List<MatchCandidate>();

        foreach (var m2 in library)
        {
            try
            {
                // Use simple aspect ratio matching (M2s can be scaled)
                float score = SimpleM2MatchScore(pm4Obj.Stats, m2.Stats);
                
                if (score >= minConfidence)
                {
                    // Calculate scale from dimension ratio
                    var pm4Dims = new[] { pm4Obj.Stats.Dimensions.X, pm4Obj.Stats.Dimensions.Y, pm4Obj.Stats.Dimensions.Z }
                        .OrderByDescending(x => x).ToArray();
                    var m2Dims = new[] { m2.Stats.Dimensions.X, m2.Stats.Dimensions.Y, m2.Stats.Dimensions.Z }
                        .OrderByDescending(x => x).ToArray();
                    float scale = m2Dims[0] > 0.1f ? pm4Dims[0] / m2Dims[0] : 1.0f;
                    
                    // Use MPRL position/rotation
                    var position = pm4Obj.MprlPosition ?? pm4Obj.Stats.Centroid;
                    var rotation = pm4Obj.MprlRotationDegrees.HasValue 
                        ? new Vector3(0, pm4Obj.MprlRotationDegrees.Value, 0)
                        : Vector3.Zero;
                    
                    candidates.Add(new MatchCandidate(
                        pm4Obj.Ck24,
                        m2.M2Path,
                        score,
                        position,
                        rotation,
                        scale
                    ));

                    if (score > bestConfidence)
                    {
                        bestConfidence = score;
                        bestMatch = m2;
                        bestTransform = new Pm4WmoGeometryMatcher.PlacementTransform(
                            position, rotation, scale, score);
                    }
                }
            }
            catch {}
        }

        return (bestMatch, bestTransform, candidates);
    }

    /// <summary>
    /// Simple M2 matching using aspect ratios (scale-invariant).
    /// </summary>
    private float SimpleM2MatchScore(Pm4WmoGeometryMatcher.GeometryStats pm4Stats, 
                                      Pm4WmoGeometryMatcher.GeometryStats m2Stats)
    {
        // Get sorted dimensions
        var pm4Dims = new[] { pm4Stats.Dimensions.X, pm4Stats.Dimensions.Y, pm4Stats.Dimensions.Z }
            .OrderByDescending(x => x).ToArray();
        var m2Dims = new[] { m2Stats.Dimensions.X, m2Stats.Dimensions.Y, m2Stats.Dimensions.Z }
            .OrderByDescending(x => x).ToArray();

        // Skip tiny objects
        if (m2Dims[0] < 0.1f || pm4Dims[0] < 0.1f) return 0;

        // Normalize to largest = 1.0
        float[] pm4Aspect = { 1.0f, pm4Dims[1] / pm4Dims[0], pm4Dims[2] / pm4Dims[0] };
        float[] m2Aspect = { 1.0f, m2Dims[1] / m2Dims[0], m2Dims[2] / m2Dims[0] };

        // Compare aspect ratios
        float diff1 = Math.Abs(pm4Aspect[1] - m2Aspect[1]);
        float diff2 = Math.Abs(pm4Aspect[2] - m2Aspect[2]);
        float avgDiff = (diff1 + diff2) / 2;

        // Score: 0 diff = 100%, 0.5 diff = 0%
        return Math.Max(0, 1 - avgDiff * 2);
    }

    /// <summary>
    /// Quick match score for M2s (Scale Invariant).
    /// </summary>
    public float QuickMatchScoreM2(Pm4WmoGeometryMatcher.GeometryStats pm4Stats, 
                                   Pm4WmoGeometryMatcher.GeometryStats m2Stats)
    {
        // For M2s, we can't compare absolute extents because scale varies.
        // We compare Aspect Ratios of the Principal Extents.
        
        var pm4Extents = pm4Stats.PrincipalExtents.OrderByDescending(x => x).ToArray();
        var m2Extents = m2Stats.PrincipalExtents.OrderByDescending(x => x).ToArray();

        if (m2Extents[0] < 0.001f || pm4Extents[0] < 0.001f) return 0;

        // Normalize by the largest extent to get shape profile
        var pm4Ratios = new[] { 1.0f, pm4Extents[1] / pm4Extents[0], pm4Extents[2] / pm4Extents[0] };
        var m2Ratios = new[] { 1.0f, m2Extents[1] / m2Extents[0], m2Extents[2] / m2Extents[0] };

        // Compare ratios
        float diff1 = Math.Abs(pm4Ratios[1] - m2Ratios[1]); // Max possible diff is around 1.0
        float diff2 = Math.Abs(pm4Ratios[2] - m2Ratios[2]);
        
        float avgDiff = (diff1 + diff2) / 2;
        
        // Score: 0 diff -> 1.0 score. 0.5 diff -> 0 score.
        return Math.Max(0, 1 - avgDiff * 2);
    }

    /// <summary>
    /// Export reconstruction results to CSV for analysis.
    /// </summary>
    public void ExportToCsv(ReconstructionResult result, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence,tile_x,tile_y");

        foreach (var entry in result.ModfEntries)
        {
            // Use TileX/TileY from PM4 filename instead of calculating from position
            sw.WriteLine(string.Join(",",
                entry.Ck24,
                entry.WmoPath,
                entry.NameId,
                entry.UniqueId,
                entry.Position.X.ToString("F2"),
                entry.Position.Y.ToString("F2"),
                entry.Position.Z.ToString("F2"), // Z is height in WoW standard
                entry.Rotation.X.ToString("F2"),
                entry.Rotation.Y.ToString("F2"),
                entry.Rotation.Z.ToString("F2"),
                (entry.Scale / 1024f).ToString("F4"),
                entry.MatchConfidence.ToString("F3"),
                entry.TileX,
                entry.TileY));
        }

        Console.WriteLine($"[INFO] Exported {result.ModfEntries.Count} MODF entries to {outputPath}");
    }

    /// <summary>
    /// Export MDDF reconstruction results to CSV.
    /// </summary>
    public void ExportMddfToCsv(MddfReconstructionResult result, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("UniqueId,Ck24,M2Path,PositionX,PositionY,PositionZ,RotX,RotY,RotZ,Scale,Confidence");

        foreach (var entry in result.MddfEntries)
        {
            sw.WriteLine($"{entry.UniqueId},{entry.Ck24},{entry.M2Path},{entry.Position.X:F4},{entry.Position.Y:F4},{entry.Position.Z:F4}," +
                         $"{entry.Rotation.X:F4},{entry.Rotation.Y:F4},{entry.Rotation.Z:F4},{(entry.Scale / 1024.0f):F4}, {entry.MatchConfidence:F4}");
        }

        Console.WriteLine($"[INFO] Exported {result.MddfEntries.Count} MDDF placement entries to {outputPath}");
    }

    /// <summary>
    /// Export MWMO string table.
    /// </summary>
    public void ExportMwmoNames(ReconstructionResult result, string outputPath)
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
    /// Export MWMO string table (Alias for ExportMwmoNames to match CLI).
    /// </summary>
    public void ExportMwmo(ReconstructionResult result, string outputPath) => ExportMwmoNames(result, outputPath);

    /// <summary>
    /// Export all match candidates to CSV for detailed analysis.
    /// </summary>
    public void ExportCandidatesCsv(ReconstructionResult result, string outputPath)
    {
        using var sw = new StreamWriter(outputPath);
        sw.WriteLine("pm4_ck24,wmo_path,confidence,scale,rot_x,rot_y,rot_z,pos_x,pos_y,pos_z");

        foreach (var c in result.AllCandidates.OrderBy(c => c.Pm4Ck24).ThenByDescending(c => c.Confidence))
        {
            sw.WriteLine(string.Join(",",
                c.Pm4Ck24,
                c.WmoPath,
                c.Confidence.ToString("F3"),
                c.Scale.ToString("F3"),
                c.Rotation.X.ToString("F2"),
                c.Rotation.Y.ToString("F2"),
                c.Rotation.Z.ToString("F2"),
                c.Position.X.ToString("F2"),
                c.Position.Y.ToString("F2"),
                c.Position.Z.ToString("F2")
            ));
        }

        Console.WriteLine($"[INFO] Exported {result.AllCandidates.Count} match candidates to {outputPath}");
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
    /// Convert PM4/server-space coordinates into ADT placement coordinates.
    /// Based on ADT_v18 docs:
    /// x' (Placement X) = 32 * TILESIZE - x (World X?) -> Wait, mapping X to Y?
    /// Let's use the explicit axis mapping:
    /// Placement X (West/East) corresponds to World Y (West/East).
    /// Placement Z (North/South) corresponds to World X (North/South).
    /// Placement Y (Up) corresponds to World Z (Up).
    /// 
    /// Formula:
    /// Placement X = 32 * TILESIZE - World Y
    /// Placement Z = 32 * TILESIZE - World X
    /// Placement Y = World Z
    /// </summary>
    private static Vector3 ServerToAdtPosition(Vector3 server)
    {
        const float TileSize = 533.33333f;
        const float HalfMap = TileSize * 32f; // ~17066.666

        // Inputs (server) are World Coordinates: X (North), Y (West), Z (Up)
        // Outputs are Placement Coordinates: X (West), Y (Up), Z (North) ?
        
        // Doc: x' = 32*TILESIZE - x. 
        // If 'x' in doc refers to the coordinate along the corresponding axis:
        // Placement X corresponds to World Y axis. So P_X = 32*T - W_Y.
        // Placement Z corresponds to World X axis. So P_Z = 32*T - W_X.
        
        return new Vector3(
            HalfMap - server.Y, // Placement X
            server.Z,           // Placement Y (Up)
            HalfMap - server.X  // Placement Z
        );
    }

    public static (int X, int Y) GetTileForPosition(Vector3 pos)
    {
        const float TileSize = 533.33333f;
        // PM4 MSVT vertices are in WoW server/world coordinates:
        // - Center of map is (0,0)
        // - X increases going south, Y increases going west
        // - Tile (0,0) is at world coords (32*533, 32*533) = (17066, 17066)
        // - Tile (32,32) is at world coords (0,0) = center
        // - Tile (63,63) is at world coords (-31*533, -31*533)
        // 
        // WoW server coords to tile: tile = 32 - (coord / TileSize)
        int x = Math.Clamp((int)(32 - (pos.X / TileSize)), 0, 63);
        int y = Math.Clamp((int)(32 - (pos.Y / TileSize)), 0, 63);
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
