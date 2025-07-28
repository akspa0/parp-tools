namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.Linq;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Smart grouping service for PM4 objects based on legacy research and MSLK pattern analysis.
/// Uses Unknown0x00/Unknown0x01 patterns and tile coordinates for proper object grouping.
/// </summary>
internal static class Pm4SmartGrouper
{
    /// <summary>
    /// Groups PM4 scene objects using the proven legacy grouping logic.
    /// Combines Unknown0x00 (primary type) with Unknown0x01 (subtype) for logical object groups.
    /// </summary>
    public static Dictionary<string, List<(int A, int B, int C)>> GroupByMslkPatterns(Pm4Scene scene)
    {
        var groups = new Dictionary<string, List<(int A, int B, int C)>>();
        
        // Group by Unknown0x00 + Unknown0x01 pattern (primary grouping strategy)
        var linkGroups = scene.Links
            .Where(link => link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0) // Only entries with geometry
            .GroupBy(link => new { 
                Type = link.Flags_0x00, 
                Subtype = link.Type_0x01,
                TileCoords = GetTileCoordinates(link)
            })
            .ToList();
        
        foreach (var group in linkGroups)
        {
            var key = $"Type{group.Key.Type:X2}_Sub{group.Key.Subtype:X2}_{group.Key.TileCoords}";
            var faces = new List<(int A, int B, int C)>();
            
            foreach (var link in group)
            {
                // Extract faces for this link's geometry
                var linkFaces = ExtractFacesForLink(scene, link);
                faces.AddRange(linkFaces);
            }
            
            if (faces.Count > 0)
            {
                groups[key] = faces;
            }
        }
        
        return groups;
    }
    
    /// <summary>
    /// Groups PM4 objects by ReferenceIndex (Unknown0x10) for cross-linking analysis.
    /// This matches the legacy "by_reference_index" grouping strategy.
    /// </summary>
    public static Dictionary<string, List<(int A, int B, int C)>> GroupByReferenceIndex(Pm4Scene scene)
    {
        var groups = new Dictionary<string, List<(int A, int B, int C)>>();
        
        var linkGroups = scene.Links
            .Where(link => link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
            .GroupBy(link => link.SurfaceRefIndex)
            .ToList();
        
        foreach (var group in linkGroups)
        {
            var key = $"Ref{group.Key:X4}";
            var faces = new List<(int A, int B, int C)>();
            
            foreach (var link in group)
            {
                var linkFaces = ExtractFacesForLink(scene, link);
                faces.AddRange(linkFaces);
            }
            
            if (faces.Count > 0)
            {
                groups[key] = faces;
            }
        }
        
        return groups;
    }
    
    /// <summary>
    /// Groups PM4 objects by ParentIndex (Unknown0x04) for hierarchical analysis.
    /// This matches the legacy "by_parent_index" grouping strategy.
    /// </summary>
    public static Dictionary<string, List<(int A, int B, int C)>> GroupByParentIndex(Pm4Scene scene)
    {
        var groups = new Dictionary<string, List<(int A, int B, int C)>>();
        
        var linkGroups = scene.Links
            .Where(link => link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
            .GroupBy(link => link.ParentId)
            .ToList();
        
        foreach (var group in linkGroups)
        {
            var key = $"Parent{group.Key:X8}";
            var faces = new List<(int A, int B, int C)>();
            
            foreach (var link in group)
            {
                var linkFaces = ExtractFacesForLink(scene, link);
                faces.AddRange(linkFaces);
            }
            
            if (faces.Count > 0)
            {
                groups[key] = faces;
            }
        }
        
        return groups;
    }
    
    private static string GetTileCoordinates(ParpToolbox.Formats.P4.Chunks.Common.MslkEntry link)
    {
        if (link.TryDecodeTileCoordinates(out int tileX, out int tileY))
        {
            return $"T{tileY:D2}_{tileX:D2}";
        }
        return "T??_??";
    }
    
    private static List<(int A, int B, int C)> ExtractFacesForLink(Pm4Scene scene, ParpToolbox.Formats.P4.Chunks.Common.MslkEntry link)
    {
        var faces = new List<(int A, int B, int C)>();
        
        // Extract triangles from the MSPI indices for this link
        int startIndex = link.MspiFirstIndex;
        int count = link.MspiIndexCount;
        
        if (startIndex >= 0 && count > 0 && startIndex + count <= scene.Indices.Count)
        {
            // Convert indices to triangles (groups of 3)
            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 < count)
                {
                    int idxA = scene.Indices[startIndex + i];
                    int idxB = scene.Indices[startIndex + i + 1];
                    int idxC = scene.Indices[startIndex + i + 2];
                    
                    // Validate indices are within vertex bounds
                    if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                    {
                        faces.Add((idxA, idxB, idxC));
                    }
                }
            }
        }
        
        return faces;
    }
}
