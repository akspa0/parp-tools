// PM4 Object Extractor - Clean PM4 parsing service
// Extracts WMO and M2 candidates from PM4 pathfinding files
// Part of the PM4 Clean Reimplementation

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;

using WoWRollback.PM4Module.Decoding;

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
        // Parse tile coordinates
        var (tileX, tileY) = ParseTileCoordinates(pm4Path);
        if (tileX < 0 || tileY < 0)
        {
            Console.WriteLine($"[WARN] Could not parse tile coords from: {Path.GetFileName(pm4Path)}");
            yield break;
        }

        // Decode PM4
        byte[] data;
        try
        {
            data = File.ReadAllBytes(pm4Path);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to read PM4 {Path.GetFileName(pm4Path)}: {ex.Message}");
            yield break;
        }

        var pm4 = Pm4Decoder.Decode(data);
        if (pm4.Surfaces.Count == 0 || pm4.MeshVertices.Count == 0)
            yield break;

        // Build Candidates using the new Builder
        var candidates = Pm4ObjectBuilder.BuildCandidates(pm4, tileX, tileY);
        
        foreach (var candidate in candidates)
        {
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
    
    // Legacy helpers removed - logic moved to Pm4ObjectBuilder
    
    #endregion
}
