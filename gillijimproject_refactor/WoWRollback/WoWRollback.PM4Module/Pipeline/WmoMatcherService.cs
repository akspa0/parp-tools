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
/// </summary>
public class WmoMatcherService
{
    private readonly List<WmoLibraryEntry> _library = new();
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
    /// </summary>
    public IEnumerable<WmoMatch> FindAllMatches(IEnumerable<Pm4WmoCandidate> candidates)
    {
        int matched = 0;
        int total = 0;
        
        foreach (var candidate in candidates)
        {
            total++;
            var match = FindBestMatch(candidate);
            if (match != null)
            {
                matched++;
                yield return match;
            }
        }
        
        Console.WriteLine($"[INFO] Matched {matched}/{total} candidates ({100f * matched / total:F1}%)");
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
