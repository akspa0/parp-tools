// WMO Matcher Service - Clean WMO matching implementation
// Matches PM4 WMO candidates to WMO library entries
// Part of the PM4 Clean Reimplementation

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;

namespace WoWRollback.PM4Module.Pipeline;

/// <summary>
/// Matches PM4 WMO candidates to WMO library entries using geometric fingerprinting.
/// Uses bounding box dimensions and dominant wall angles for rotation calculation.
/// Supports CK24 lookup table for pre-seeded high-confidence matches.
/// </summary>
public class WmoMatcherService
{
    private readonly List<WmoLibraryEntry> _library = new();
    private readonly Dictionary<uint, string> _ck24Lookup = new(); // CK24 -> WMO path lookup
    private readonly float _sizeTolerance;
    
    /// <summary>
    /// Create a WMO matcher with optional WMO library.
    /// </summary>
    /// <param name="libraryPath">Path to WMO library JSON file</param>
    /// <param name="sizeTolerance">Size matching tolerance (0.15 = 15%)</param>
    public WmoMatcherService(string? libraryPath = null, float sizeTolerance = 0.15f)
    {
        _sizeTolerance = sizeTolerance;
        
        if (!string.IsNullOrEmpty(libraryPath) && File.Exists(libraryPath))
        {
            LoadLibrary(libraryPath);
        }
    }
    
    #region Library Management
    
    /// <summary>
    /// Load WMO library from JSON file.
    /// </summary>
    public void LoadLibrary(string libraryPath)
    {
        try
        {
            var json = File.ReadAllText(libraryPath);
            var entries = JsonSerializer.Deserialize<List<WmoLibraryJsonEntry>>(json);
            
            if (entries != null)
            {
                _library.Clear();
                foreach (var entry in entries)
                {
                    if (entry.Stats == null) continue;
                    
                    _library.Add(new WmoLibraryEntry(
                        Path: entry.WmoPath ?? "",
                        BoundsMin: ToVector3(entry.Stats.BoundsMin),
                        BoundsMax: ToVector3(entry.Stats.BoundsMax),
                        DominantAngle: ComputeDominantAngleFromPrincipalAxes(entry.Stats),
                        SurfaceCount: 0, // Not in current JSON format
                        VertexCount: entry.Stats.VertexCount
                    ));
                }
                
                Console.WriteLine($"[INFO] Loaded {_library.Count} WMO entries from library");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to load WMO library: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Get number of entries in the library.
    /// </summary>
    public int LibraryCount => _library.Count;
    
    /// <summary>
    /// Get number of entries in the CK24 lookup table.
    /// </summary>
    public int Ck24LookupCount => _ck24Lookup.Count;
    
    /// <summary>
    /// Load CK24 -> WMO path mappings from a CSV file.
    /// CSV format: CK24,WMO_Name,TypeFlags,MatchCount,Confidence
    /// </summary>
    public void LoadCk24Lookup(string csvPath)
    {
        if (!File.Exists(csvPath))
        {
            Console.WriteLine($"[WARN] CK24 lookup file not found: {csvPath}");
            return;
        }
        
        try
        {
            var lines = File.ReadAllLines(csvPath);
            int loaded = 0;
            
            foreach (var line in lines.Skip(1)) // Skip header
            {
                var parts = line.Split(',');
                if (parts.Length >= 2)
                {
                    // Parse CK24 (format: 0x3F9D43)
                    var ck24Str = parts[0].Trim();
                    if (ck24Str.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
                    {
                        if (uint.TryParse(ck24Str.Substring(2), System.Globalization.NumberStyles.HexNumber, null, out uint ck24))
                        {
                            var wmoPath = parts[1].Trim();
                            if (!_ck24Lookup.ContainsKey(ck24))
                            {
                                _ck24Lookup[ck24] = wmoPath;
                                loaded++;
                            }
                        }
                    }
                }
            }
            
            Console.WriteLine($"[INFO] Loaded {loaded} CK24 -> WMO mappings from {Path.GetFileName(csvPath)}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to load CK24 lookup: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Try to get a WMO path from the CK24 lookup table.
    /// </summary>
    public bool TryGetCk24Match(uint ck24, out string? wmoPath)
    {
        return _ck24Lookup.TryGetValue(ck24, out wmoPath);
    }
    
    /// <summary>
    /// Build library by scanning a directory for WMO files.
    /// caches the result to wmo_library_cache.json in the directory.
    /// </summary>
    public void BuildLibraryFromDirectory(string rootDirectory)
    {
        Console.WriteLine($"[INFO] Building WMO library from {rootDirectory}...");
        var wmoFiles = Directory.GetFiles(rootDirectory, "*.wmo", SearchOption.AllDirectories);
        
        _library.Clear();
        int processed = 0;
        
        foreach (var file in wmoFiles)
        {
            // Skip _xyz.wmo group files
            if (FileIsGroupWmo(file)) continue;
            
            try 
            {
                var entry = ParseWmoHeader(file, rootDirectory);
                if (entry != null)
                {
                    _library.Add(entry);
                }
            }
            catch (Exception ex)
            {
                // Console.WriteLine($"[WARN] Failed to parse {Path.GetFileName(file)}: {ex.Message}");
            }
            
            processed++;
            if (processed % 1000 == 0) Console.WriteLine($"[INFO] Scanned {processed} files...");
        }
        
        Console.WriteLine($"[INFO] Built library with {_library.Count} entries.");
        
        // Save cache
        try
        {
            var cachePath = Path.Combine(rootDirectory, "wmo_library_cache.json");
            SaveLibrary(cachePath);
            Console.WriteLine($"[INFO] Saved cache to {cachePath}");
        }
        catch {}
    }

    private bool FileIsGroupWmo(string path)
    {
        var name = Path.GetFileNameWithoutExtension(path);
        // Standard check: ends with _\d{3}
        if (name.Length > 4 && name[^4] == '_' && char.IsDigit(name[^3]) && char.IsDigit(name[^2]) && char.IsDigit(name[^1]))
            return true;
        return false;
    }

    private WmoLibraryEntry? ParseWmoHeader(string filePath, string rootDir)
    {
        using var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);
        
        // Read chunks until MOHD
        while (fs.Position < fs.Length)
        {
            if (fs.Position + 8 > fs.Length) break;
            
            // Read FourCC inverted
            var fourCCBytes = br.ReadBytes(4);
            var fourCC = System.Text.Encoding.ASCII.GetString(fourCCBytes.Reverse().ToArray());
            var size = br.ReadUInt32();
            
            if (fourCC == "MOHD")
            {
                // Parse MOHD
                // 36 bytes skip to bounds
                // nTextures(4), nGroups(4), nPortals(4), nLights(4), nDoodadNames(4), nDoodadDefs(4), nDoodadSets(4), ambColor(4), wmoID(4) = 36 bytes
                fs.Seek(36, SeekOrigin.Current);
                
                // Read Bounds (Min X, Z, Y, Max X, Z, Y) - 6 floats
                var minX = br.ReadSingle();
                var minZ = br.ReadSingle();
                var minY = br.ReadSingle();
                var maxX = br.ReadSingle();
                var maxZ = br.ReadSingle();
                var maxY = br.ReadSingle();
                
                // Convert Coordinate Space
                // WMO File: X, Z(Up), Y(Depth)? (Based on Pm4WmoWriter)
                // We want: X, Y(Depth), Z(Up) for our system to match PM4
                // Let's assume standard WMO is Z-up.
                // Our system uses Z-up.
                // Wait, Pm4WmoWriter wrote: bw.Write(b.Min.X); bw.Write(b.Min.Z); bw.Write(b.Min.Y);
                // Where b.Min was (X, Y=Depth, Z=Height).
                // So File has (X, Height, Depth).
                // So when reading:
                // Read 1: X
                // Read 2: Height (Z)
                // Read 3: Depth (Y)
                
                var boundsMin = new Vector3(minX, minY, minZ); // Map file Z->Y(Height->Depth?? No.)
                // File: X, Z_Up, Y_Depth
                // Vector3: X, Y_Depth, Z_Up
                // So: V.X = F.X, V.Y = F.Y_Depth (3rd), V.Z = F.Z_Up (2nd)
                boundsMin = new Vector3(minX, minY, minZ);
                var boundsMax = new Vector3(maxX, maxY, maxZ);
                
                var relPath = Path.GetRelativePath(rootDir, filePath).Replace("\\", "/");
                
                return new WmoLibraryEntry(
                    Path: relPath,
                    BoundsMin: boundsMin,
                    BoundsMax: boundsMax,
                    DominantAngle: 0f, // Cannot easily calc
                    SurfaceCount: 0,
                    VertexCount: 0
                );
            }
            else
            {
                fs.Seek(size, SeekOrigin.Current);
            }
        }
        return null;
    }
    
    private void SaveLibrary(string path)
    {
        var options = new JsonSerializerOptions { WriteIndented = true };
        
        // Convert to cache JSON format
        var cacheEntries = _library.Select(e => new WmoLibraryJsonEntry
        {
            WmoPath = e.Path,
            Stats = new WmoStatsJson
            {
                VertexCount = e.VertexCount,
                BoundsMin = new Vector3Json { X = e.BoundsMin.X, Y = e.BoundsMin.Y, Z = e.BoundsMin.Z },
                BoundsMax = new Vector3Json { X = e.BoundsMax.X, Y = e.BoundsMax.Y, Z = e.BoundsMax.Z },
                PrincipalAxes = new[] { new Vector3Json { X = 1, Y = 0, Z = 0 } } // Dummy axis
            }
        }).ToList();
        
        var json = JsonSerializer.Serialize(cacheEntries, options);
        File.WriteAllText(path, json);
    }

    
    #endregion
    
    #region Matching
    
    /// <summary>
    /// Find the best matching WMO for a PM4 candidate.
    /// Returns null if no suitable match is found.
    /// </summary>
    public WmoMatch? FindBestMatch(Pm4WmoCandidate candidate)
    {
        if (_library.Count == 0)
        {
            Console.WriteLine("[WARN] WMO library is empty - cannot match");
            return null;
        }
        
        WmoLibraryEntry? bestEntry = null;
        float bestScore = 0f;
        float bestYaw = 0f;
        
        var candidateSize = candidate.Size;
        
        // If MPRL rotation is available, use it directly instead of trying cardinal rotations
        bool useMprlRotation = candidate.HasMprlRotation;
        float mprlYaw = candidate.MprlRotationDegrees ?? 0f;
        
        foreach (var entry in _library)
        {
            var entrySize = entry.Size;
            
            if (useMprlRotation)
            {
                // Use MPRL rotation directly - much more accurate!
                var rotatedSize = RotateSize(entrySize, mprlYaw);
                
                if (!IsSizeMatch(candidateSize, rotatedSize, _sizeTolerance))
                    continue;
                
                float score = CalculateSizeScore(candidateSize, rotatedSize);
                // Boost score for MPRL-based rotation (higher confidence)
                score *= 1.2f;
                
                if (score > bestScore)
                {
                    bestScore = score;
                    bestEntry = entry;
                    bestYaw = mprlYaw;
                }
            }
            else
            {
                // Fall back to cardinal rotations (0°, 90°, 180°, 270°)
                for (int rotIdx = 0; rotIdx < 4; rotIdx++)
                {
                    var yawDegrees = rotIdx * 90f;
                    var rotatedSize = RotateSize(entrySize, yawDegrees);
                    
                    if (!IsSizeMatch(candidateSize, rotatedSize, _sizeTolerance))
                        continue;
                    
                    float score = CalculateSizeScore(candidateSize, rotatedSize);
                    
                    // Boost score if dominant angles align
                    float angleDiff = NormalizeAngle(candidate.DominantAngle - entry.DominantAngle - yawDegrees);
                    if (Math.Abs(angleDiff) < 15f || Math.Abs(angleDiff - 90f) < 15f || 
                        Math.Abs(angleDiff - 180f) < 15f || Math.Abs(angleDiff + 90f) < 15f)
                    {
                        score *= 1.1f;
                    }
                    
                    // Boost score if vertex counts are similar (using DebugGeometry count)
                    if (candidate.DebugGeometry != null && entry.VertexCount > 0)
                    {
                        float vCountRatio = (float)candidate.DebugGeometry.Count / entry.VertexCount;
                        if (vCountRatio > 0.8f && vCountRatio < 1.2f)
                        {
                            score *= 1.15f;
                        }
                    }
                    
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestEntry = entry;
                        bestYaw = yawDegrees;
                    }
                }
            }
        }
        
        if (bestEntry == null || bestScore < 0.5f)
            return null;
        
        // Calculate final placement - use MPRL position if available
        var position = candidate.MprlPosition.HasValue 
            ? candidate.MprlPosition.Value 
            : CalculatePosition(candidate, bestEntry, bestYaw);
        var rotation = new Vector3(0f, bestYaw, 0f); // Pitch=0, Yaw=rotation, Roll=0
        
        return new WmoMatch(
            WmoPath: bestEntry.Path,
            Position: position,
            Rotation: rotation,
            Scale: 1.0f, // WMOs cannot be scaled
            ConfidenceScore: Math.Min(1.0f, bestScore),
            SourceCandidate: candidate,
            MatchedEntry: bestEntry
        );
    }
    
    /// <summary>
    /// Find matches for all candidates from a list.
    /// Uses CK24 lookup table first, then confirmed matches, then geometric matching.
    /// </summary>
    public IEnumerable<WmoMatch> FindAllMatches(IEnumerable<Pm4WmoCandidate> candidates)
    {
        int matched = 0;
        int matchedFromLookup = 0;
        int matchedFromConfirmed = 0;
        int matchedFromGeometric = 0;
        int total = 0;
        
        // Dictionary to track matched instances by CK24 (Object ID)
        // If we match one instance with high confidence, we can reuse that match for others of same CK24
        var confirmedMatches = new Dictionary<uint, WmoLibraryEntry>();
        
        // Pre-seed confirmedMatches from CK24 lookup table
        foreach (var kvp in _ck24Lookup)
        {
            var entry = _library.FirstOrDefault(e => 
                e.Path.Equals(kvp.Value, StringComparison.OrdinalIgnoreCase) ||
                e.Path.EndsWith(Path.GetFileName(kvp.Value), StringComparison.OrdinalIgnoreCase));
            if (entry != null && !confirmedMatches.ContainsKey(kvp.Key))
            {
                confirmedMatches[kvp.Key] = entry;
            }
        }
        
        if (_ck24Lookup.Count > 0)
        {
            Console.WriteLine($"[INFO] Pre-seeded {confirmedMatches.Count} matches from CK24 lookup table");
        }
        
        var candidatesList = candidates.ToList();
        
        foreach (var candidate in candidatesList)
        {
            total++;
            
            WmoMatch? match = null;
            bool fromLookup = false;
            bool fromConfirmed = false;
            
            // Check if we already have a confirmed match for this CK24 (from lookup or previous match)
            if (confirmedMatches.TryGetValue(candidate.CK24, out var confirmedEntry))
            {
                // We have a confirmed match type!
                // But we still need to calculate Position/Rotation specific to this instance
                // Use the confirmed entry to calculate placement
                
                // For rotation, if we don't have MPRL, we might need to re-scan for best yaw
                // But typically instances share orientation logic or have MPRL.
                // Let's do a quick best-fit rotation for this specific instance using the confirmed entry
                
                float bestYaw = 0f;
                float bestScore = 0f;
                bool useMprl = candidate.HasMprlRotation;
                float mprlYaw = candidate.MprlRotationDegrees ?? 0f;
                
                if (useMprl)
                {
                    bestYaw = mprlYaw;
                    bestScore = 1.0f; 
                }
                else
                {
                    // Scan cardinal rotations for this instance against the confirmed WMO
                    for (int rotIdx = 0; rotIdx < 4; rotIdx++)
                    {
                        var yaw = rotIdx * 90f;
                        var rotatedSize = RotateSize(confirmedEntry.Size, yaw);
                        if (IsSizeMatch(candidate.Size, rotatedSize, _sizeTolerance))
                        {
                            float score = CalculateSizeScore(candidate.Size, rotatedSize);
                            if (score > bestScore)
                            {
                                bestScore = score;
                                bestYaw = yaw;
                            }
                        }
                    }
                }
                
                var position = candidate.MprlPosition ?? CalculatePosition(candidate, confirmedEntry, bestYaw);
                
                match = new WmoMatch(
                    WmoPath: confirmedEntry.Path,
                    Position: position,
                    Rotation: new Vector3(0f, bestYaw, 0f),
                    Scale: 1.0f,
                    ConfidenceScore: _ck24Lookup.ContainsKey(candidate.CK24) ? 0.98f : 0.95f, // Higher confidence for lookup-based
                    SourceCandidate: candidate,
                    MatchedEntry: confirmedEntry
                );
                
                fromConfirmed = true;
                if (_ck24Lookup.ContainsKey(candidate.CK24)) fromLookup = true;
            }
            else
            {
                // No confirmed match yet, perform full search
                match = FindBestMatch(candidate);
                
                // If we found a high confidence match, store it for propagation
                if (match != null && match.ConfidenceScore > 0.85f)
                {
                    if (!confirmedMatches.ContainsKey(candidate.CK24))
                    {
                        confirmedMatches[candidate.CK24] = match.MatchedEntry;
                    }
                }
            }

            if (match != null)
            {
                matched++;
                if (fromLookup) matchedFromLookup++;
                else if (fromConfirmed) matchedFromConfirmed++;
                else matchedFromGeometric++;
                yield return match;
            }
        }
        
        Console.WriteLine($"[INFO] Matched {matched}/{total} candidates ({100f * matched / total:F1}%)");
        if (matched > 0)
        {
            Console.WriteLine($"       - From CK24 lookup: {matchedFromLookup}");
            Console.WriteLine($"       - From propagation: {matchedFromConfirmed}");
            Console.WriteLine($"       - From geometric:   {matchedFromGeometric}");
        }
    }
    
    #endregion
    
    #region Calculation Helpers
    
    /// <summary>
    /// Check if two sizes match within tolerance, allowing for 90° rotations.
    /// </summary>
    private static bool IsSizeMatch(Vector3 size1, Vector3 size2, float tolerance)
    {
        // Compare sorted dimensions (allows for axis swaps)
        var s1 = new[] { size1.X, size1.Y, size1.Z }.OrderByDescending(x => x).ToArray();
        var s2 = new[] { size2.X, size2.Y, size2.Z }.OrderByDescending(x => x).ToArray();
        
        for (int i = 0; i < 3; i++)
        {
            float diff = Math.Abs(s1[i] - s2[i]) / Math.Max(s1[i], 1f);
            if (diff > tolerance)
                return false;
        }
        
        return true;
    }
    
    /// <summary>
    /// Calculate a score (0-1) based on how well two sizes match.
    /// </summary>
    private static float CalculateSizeScore(Vector3 size1, Vector3 size2)
    {
        var s1 = new[] { size1.X, size1.Y, size1.Z }.OrderByDescending(x => x).ToArray();
        var s2 = new[] { size2.X, size2.Y, size2.Z }.OrderByDescending(x => x).ToArray();
        
        float totalDiff = 0f;
        for (int i = 0; i < 3; i++)
        {
            float maxVal = Math.Max(s1[i], s2[i]);
            if (maxVal > 0.001f)
                totalDiff += Math.Abs(s1[i] - s2[i]) / maxVal;
        }
        
        // Convert difference to score (0 diff = 1.0 score, 0.3 total diff = 0.7 score, etc.)
        return Math.Max(0f, 1f - totalDiff);
    }
    
    /// <summary>
    /// Rotate a size vector around the Y axis by the given angle in degrees.
    /// </summary>
    private static Vector3 RotateSize(Vector3 size, float yawDegrees)
    {
        // For cardinal rotations, just swap X and Z axes
        int rotation = (int)(yawDegrees / 90) % 4;
        return rotation switch
        {
            0 => size,
            1 => new Vector3(size.Z, size.Y, size.X), // 90°
            2 => size, // 180° - no size change
            3 => new Vector3(size.Z, size.Y, size.X), // 270°
            _ => size
        };
    }
    
    /// <summary>
    /// Normalize an angle to the range [-180, 180].
    /// </summary>
    private static float NormalizeAngle(float angle)
    {
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }
    
    /// <summary>
    /// Calculate the world position for WMO placement.
    /// Uses bounding box center alignment.
    /// </summary>
    private static Vector3 CalculatePosition(Pm4WmoCandidate candidate, WmoLibraryEntry wmo, float yawDegrees)
    {
        // PM4 center (world coordinates)
        var pm4Center = candidate.Center;
        
        // WMO center (model coordinates, needs rotation)
        var wmoCenter = wmo.Center;
        
        // Rotate WMO center by yaw
        float yawRad = yawDegrees * MathF.PI / 180f;
        var rotatedCenter = new Vector3(
            wmoCenter.X * MathF.Cos(yawRad) - wmoCenter.Z * MathF.Sin(yawRad),
            wmoCenter.Y,
            wmoCenter.X * MathF.Sin(yawRad) + wmoCenter.Z * MathF.Cos(yawRad)
        );
        
        // Translation = PM4 center - rotated WMO center
        return pm4Center - rotatedCenter;
    }
    
    /// <summary>
    /// Compute dominant angle from principal axes in library entry.
    /// </summary>
    private static float ComputeDominantAngleFromPrincipalAxes(WmoStatsJson stats)
    {
        if (stats.PrincipalAxes == null || stats.PrincipalAxes.Length == 0)
            return 0f;
        
        // Use the first principal axis (longest extent)
        var axis = stats.PrincipalAxes[0];
        float angle = MathF.Atan2(axis.Y, axis.X) * 180f / MathF.PI;
        
        // Normalize to 0-180 (walls are bidirectional)
        if (angle < 0) angle += 180f;
        if (angle >= 180f) angle -= 180f;
        
        return angle;
    }
    
    private static Vector3 ToVector3(Vector3Json? v)
    {
        if (v == null) return Vector3.Zero;
        return new Vector3(v.X, v.Y, v.Z);
    }
    
    #endregion
    
    #region JSON Deserialization Models
    
    private record WmoLibraryJsonEntry
    {
        public string? WmoPath { get; init; }
        public WmoStatsJson? Stats { get; init; }
    }
    
    private record WmoStatsJson
    {
        public int VertexCount { get; init; }
        public int FaceCount { get; init; }
        public Vector3Json? BoundsMin { get; init; }
        public Vector3Json? BoundsMax { get; init; }
        public Vector3Json? Centroid { get; init; }
        public Vector3Json? Dimensions { get; init; }
        public Vector3Json[]? PrincipalAxes { get; init; }
        public float[]? PrincipalExtents { get; init; }
    }
    
    private record Vector3Json
    {
        public float X { get; init; }
        public float Y { get; init; }
        public float Z { get; init; }
    }
    
    #endregion
}
