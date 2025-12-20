// PM4 Object Extractor - Clean PM4 parsing service
// Extracts WMO and M2 candidates from PM4 pathfinding files
// Part of the PM4 Clean Reimplementation

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

namespace WoWRollback.PM4Module.Pipeline;

/// <summary>
/// Extracts object candidates from PM4 pathfinding files.
/// Groups surfaces by CK24 for WMO candidates and clusters M2 geometry.
/// </summary>
public class Pm4ObjectExtractor
{
    #region Public API
    
    /// <summary>
    /// Extract all WMO candidates from a single PM4 file.
    /// Groups surfaces by CK24 (24-bit object key from MSUR.PackedParams).
    /// </summary>
    public IEnumerable<Pm4WmoCandidate> ExtractWmoCandidates(string pm4Path)
    {
        var candidates = new List<Pm4WmoCandidate>();
        
        // Parse tile coordinates from filename (e.g., development_29_39.pm4)
        var (tileX, tileY) = ParseTileCoordinates(pm4Path);
        if (tileX < 0 || tileY < 0)
        {
            Console.WriteLine($"[WARN] Could not parse tile coords from: {Path.GetFileName(pm4Path)}");
            yield break;
        }
        
        // Parse PM4 file
        byte[] pm4Data;
        PM4File pm4;
        try
        {
            pm4Data = File.ReadAllBytes(pm4Path);
            pm4 = new PM4File(pm4Data);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to parse PM4 {Path.GetFileName(pm4Path)}: {ex.Message}");
            yield break;
        }
        
        if (pm4.Surfaces.Count == 0 || pm4.MeshVertices.Count == 0)
            yield break;
        
        // Group surfaces by CK24 (excluding M2 buckets and zero CK24)
        var wmoSurfaces = pm4.Surfaces
            .Where(s => !s.IsM2Bucket && s.CK24 != 0)
            .GroupBy(s => s.CK24)
            .ToList();
        
        foreach (var group in wmoSurfaces)
        {
            var candidate = ExtractCandidateFromGroup(pm4, group, tileX, tileY);
            if (candidate != null)
                yield return candidate;
        }
    }
    
    /// <summary>
    /// Extract all WMO candidates from all PM4 files in a directory.
    /// </summary>
    public IEnumerable<Pm4WmoCandidate> ExtractAllWmoCandidates(string pm4Directory)
    {
        if (!Directory.Exists(pm4Directory))
        {
            Console.WriteLine($"[WARN] PM4 directory not found: {pm4Directory}");
            yield break;
        }
        
        var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
        Console.WriteLine($"[INFO] Found {pm4Files.Length} PM4 files to process...");
        
        int totalCandidates = 0;
        foreach (var pm4Path in pm4Files)
        {
            foreach (var candidate in ExtractWmoCandidates(pm4Path))
            {
                totalCandidates++;
                yield return candidate;
            }
        }
        
        Console.WriteLine($"[INFO] Extracted {totalCandidates} WMO candidates from {pm4Files.Length} files");
    }
    
    /// <summary>
    /// Extract M2 candidates from a single PM4 file using MSCN data.
    /// Note: M2 extraction is more complex and may require clustering.
    /// </summary>
    public IEnumerable<Pm4M2Candidate> ExtractM2Candidates(string pm4Path)
    {
        // Parse tile coordinates
        var (tileX, tileY) = ParseTileCoordinates(pm4Path);
        if (tileX < 0 || tileY < 0)
            yield break;
        
        // Parse PM4 file
        byte[] pm4Data;
        PM4File pm4;
        try
        {
            pm4Data = File.ReadAllBytes(pm4Path);
            pm4 = new PM4File(pm4Data);
        }
        catch
        {
            yield break;
        }
        
        // Extract from MSCN (exterior vertices) - these are potential M2 positions
        int index = 0;
        foreach (var vertex in pm4.ExteriorVertices)
        {
            // Transform MSCN coordinates: swap Y and X, negate X
            var position = new Vector3(vertex.Y, -vertex.X, vertex.Z);
            
            yield return new Pm4M2Candidate(
                Position: position,
                TileX: tileX,
                TileY: tileY,
                MscnIndex: index++
            );
        }
    }
    
    #endregion
    
    #region Internal Helpers
    
    /// <summary>
    /// Parse tile X,Y coordinates from PM4 filename.
    /// Expects format: mapname_XX_YY.pm4
    /// </summary>
    private static (int tileX, int tileY) ParseTileCoordinates(string pm4Path)
    {
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        var match = Regex.Match(baseName, @"(\d+)_(\d+)$");
        if (!match.Success)
            return (-1, -1);
        
        return (int.Parse(match.Groups[1].Value), int.Parse(match.Groups[2].Value));
    }
    
    /// <summary>
    /// Extract a WMO candidate from a group of surfaces with the same CK24.
    /// </summary>
    private Pm4WmoCandidate? ExtractCandidateFromGroup(PM4File pm4, IGrouping<uint, MsurEntry> group, int tileX, int tileY)
    {
        uint ck24 = group.Key;
        
        // Collect all vertices for this CK24 group
        var vertices = new List<Vector3>();
        foreach (var surface in group)
        {
            for (int i = 0; i < surface.IndexCount && surface.MsviFirstIndex + i < pm4.MeshIndices.Count; i++)
            {
                uint vertIdx = pm4.MeshIndices[(int)surface.MsviFirstIndex + i];
                if (vertIdx < pm4.MeshVertices.Count)
                {
                    // WMOs require (Y, X, Z) swap to match correctly
                    var v = pm4.MeshVertices[(int)vertIdx];
                    vertices.Add(new Vector3(v.Y, v.X, v.Z));
                }
            }
        }
        
        // Skip tiny objects
        if (vertices.Count < 10)
            return null;
        
        // Optionally enhance with MSCN vertices (exterior scene nodes)
        if (pm4.ExteriorVertices.Count > 0)
        {
            vertices = EnhanceWithMscnVertices(vertices, pm4.ExteriorVertices);
        }
        
        // Compute bounding box
        var boundsMin = new Vector3(float.MaxValue);
        var boundsMax = new Vector3(float.MinValue);
        foreach (var v in vertices)
        {
            boundsMin = Vector3.Min(boundsMin, v);
            boundsMax = Vector3.Max(boundsMax, v);
        }
        
        // Compute dominant wall angle for rotation matching
        float dominantAngle = ComputeDominantWallAngle(group.ToList(), pm4);
        
        // Extract type flags from CK24 byte structure
        byte typeFlags = (byte)((ck24 >> 16) & 0xFF);
        
        // Find nearby MPRL entry for rotation/position data
        // MPRL is stored as (Y, Z, X) - convert to (X, Y, Z) to match MSVT
        float? mprlRotation = null;
        Vector3? mprlPosition = null;
        
        if (pm4.PositionRefs.Count > 0)
        {
            var nearbyMprl = pm4.PositionRefs
                .Where(p => p.Unknown0x16 == 0) // Non-terminator entries only
                .Select(p => new {
                    Entry = p,
                    // YZX → XYZ conversion
                    X = p.PositionZ,  // stored Z is real X
                    Y = p.PositionX,  // stored X is real Y
                    Z = p.PositionY   // stored Y is real Z
                })
                .Where(p => 
                    p.X >= boundsMin.X - 50 && p.X <= boundsMax.X + 50 &&
                    p.Y >= boundsMin.Y - 50 && p.Y <= boundsMax.Y + 50)
                .OrderBy(p => Vector3.Distance(
                    new Vector3(p.X, p.Y, p.Z), 
                    (boundsMin + boundsMax) / 2f)) // Distance from centroid
                .FirstOrDefault();
            
            if (nearbyMprl != null)
            {
                // Extract rotation (0-65535 → 0-360°)
                mprlRotation = 360.0f * nearbyMprl.Entry.Unknown0x04 / 65536.0f;
                mprlPosition = new Vector3(nearbyMprl.X, nearbyMprl.Y, nearbyMprl.Z);
            }
        }
        
        return new Pm4WmoCandidate(
            CK24: ck24,
            TileX: tileX,
            TileY: tileY,
            BoundsMin: boundsMin,
            BoundsMax: boundsMax,
            DominantAngle: dominantAngle,
            SurfaceCount: group.Count(),
            VertexCount: vertices.Count,
            TypeFlags: typeFlags,
            MprlRotationDegrees: mprlRotation,
            MprlPosition: mprlPosition
        );
    }
    
    /// <summary>
    /// Enhance vertex list with MSCN (exterior vertices) that fall within the bounding box.
    /// This improves matching accuracy by including additional geometry detail.
    /// </summary>
    private List<Vector3> EnhanceWithMscnVertices(List<Vector3> vertices, IReadOnlyList<Vector3> mscnVertices)
    {
        // Compute current bounding box with margin
        var boundsMin = new Vector3(float.MaxValue);
        var boundsMax = new Vector3(float.MinValue);
        foreach (var v in vertices)
        {
            boundsMin = Vector3.Min(boundsMin, v);
            boundsMax = Vector3.Max(boundsMax, v);
        }
        
        const float MscnMargin = 7.0f;
        boundsMin -= new Vector3(MscnMargin);
        boundsMax += new Vector3(MscnMargin);
        
        // Add MSCN vertices that fall within extended bounds
        var enhanced = new List<Vector3>(vertices);
        foreach (var mscnVert in mscnVertices)
        {
            // Transform MSCN: swap Y to X, negate X to Y
            var transformed = new Vector3(mscnVert.Y, -mscnVert.X, mscnVert.Z);
            if (transformed.X >= boundsMin.X && transformed.X <= boundsMax.X &&
                transformed.Y >= boundsMin.Y && transformed.Y <= boundsMax.Y &&
                transformed.Z >= boundsMin.Z && transformed.Z <= boundsMax.Z)
            {
                enhanced.Add(transformed);
            }
        }
        
        return enhanced;
    }
    
    /// <summary>
    /// Compute the dominant wall angle from vertical surfaces.
    /// Uses surface area histogram to find the most common wall orientation.
    /// </summary>
    private float ComputeDominantWallAngle(List<MsurEntry> surfaces, PM4File pm4)
    {
        // Build histogram of wall angles (5-degree bins)
        var angleHistogram = new Dictionary<int, float>(); // bin -> area
        
        foreach (var surface in surfaces)
        {
            // Check if this is a vertical wall (normal mostly horizontal)
            var normal = new Vector3(surface.NormalX, surface.NormalY, surface.NormalZ);
            float horizontalMagnitude = MathF.Sqrt(normal.X * normal.X + normal.Y * normal.Y);
            
            // Skip non-walls (mostly horizontal or upward-facing)
            if (horizontalMagnitude < 0.5f)
                continue;
            
            // Compute angle of the wall's horizontal normal
            float angleRad = MathF.Atan2(normal.Y, normal.X);
            float angleDeg = angleRad * 180f / MathF.PI;
            
            // Normalize to 0-180 range (walls have two-sided normals)
            if (angleDeg < 0) angleDeg += 180f;
            if (angleDeg >= 180f) angleDeg -= 180f;
            
            // Bin the angle (5-degree bins)
            int bin = (int)(angleDeg / 5) * 5;
            
            // Estimate surface area (use index count as proxy)
            float area = surface.IndexCount / 3f; // triangles
            
            if (!angleHistogram.ContainsKey(bin))
                angleHistogram[bin] = 0;
            angleHistogram[bin] += area;
        }
        
        if (angleHistogram.Count == 0)
            return 0f;
        
        // Find the bin with maximum area
        var maxBin = angleHistogram.OrderByDescending(kvp => kvp.Value).First();
        return maxBin.Key;
    }
    
    #endregion
}
