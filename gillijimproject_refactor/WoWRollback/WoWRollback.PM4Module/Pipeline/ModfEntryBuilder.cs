// MODF Entry Builder - Converts WMO matches to ADT MODF entries
// Part of the PM4 Clean Reimplementation

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WoWRollback.PM4Module.Pipeline;

/// <summary>
/// Builds MODF (WMO placement) entries from WMO matches.
/// Handles coordinate transforms, unique ID assignment, and CSV export.
/// </summary>
public class ModfEntryBuilder
{
    private readonly Dictionary<string, int> _wmoNameToIndex = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> _wmoNames = new();
    private uint _nextUniqueId;
    
    /// <summary>
    /// Create a new MODF entry builder.
    /// </summary>
    /// <param name="startingUniqueId">Starting unique ID for entries</param>
    public ModfEntryBuilder(uint startingUniqueId = 1)
    {
        _nextUniqueId = startingUniqueId;
    }
    
    #region Entry Building
    
    /// <summary>
    /// Create a MODF entry from a WMO match.
    /// Handles WMO name indexing and unique ID assignment.
    /// </summary>
    public ModfEntry CreateEntry(WmoMatch match)
    {
        // Get or create WMO name index
        int nameIndex = GetOrAddWmoName(match.WmoPath);
        
        // Assign unique ID
        uint uniqueId = _nextUniqueId++;
        
        // Calculate bounds in world space
        // Apply rotation to WMO bounds and translate
        var (boundsMin, boundsMax) = CalculateWorldBounds(match);
        
        // Convert position to ADT format (XZY swap)
        // ADT MODF stores position as: X, Z, Y (Y and Z swapped)
        var adtPosition = new Vector3(
            match.Position.X,
            match.Position.Z,  // Z goes to Y slot
            match.Position.Y   // Y goes to Z slot
        );
        
        // Rotation stays as XYZ (no swap for rotation)
        var adtRotation = match.Rotation;
        
        // Convert bounds to ADT format (XZY swap)
        var adtBoundsMin = new Vector3(boundsMin.X, boundsMin.Z, boundsMin.Y);
        var adtBoundsMax = new Vector3(boundsMax.X, boundsMax.Z, boundsMax.Y);
        
        return new ModfEntry(
            NameIndex: nameIndex,
            UniqueId: uniqueId,
            Position: adtPosition,
            Rotation: adtRotation,
            BoundsMin: adtBoundsMin,
            BoundsMax: adtBoundsMax,
            Flags: 0,
            DoodadSet: 0,
            NameSet: 0,
            Scale: 0  // WMOs don't scale in 3.3.5
        );
    }
    
    /// <summary>
    /// Create MODF entries for all matches.
    /// </summary>
    public List<ModfEntry> CreateEntries(IEnumerable<WmoMatch> matches)
    {
        var entries = new List<ModfEntry>();
        
        foreach (var match in matches)
        {
            entries.Add(CreateEntry(match));
        }
        
        Console.WriteLine($"[INFO] Created {entries.Count} MODF entries with {_wmoNames.Count} unique WMOs");
        return entries;
    }
    
    /// <summary>
    /// Get entries for a specific tile.
    /// </summary>
    public List<ModfEntry> CreateEntriesForTile(IEnumerable<WmoMatch> matches, int tileX, int tileY)
    {
        var tileMatches = matches.Where(m => 
            m.SourceCandidate.TileX == tileX && 
            m.SourceCandidate.TileY == tileY);
        
        return CreateEntries(tileMatches);
    }
    
    #endregion
    
    #region WMO Name Management
    
    /// <summary>
    /// Get the index for a WMO path, adding it to the list if new.
    /// </summary>
    public int GetOrAddWmoName(string wmoPath)
    {
        // Normalize path separators
        var normalizedPath = wmoPath.Replace('/', '\\');
        
        if (_wmoNameToIndex.TryGetValue(normalizedPath, out int index))
            return index;
        
        index = _wmoNames.Count;
        _wmoNames.Add(normalizedPath);
        _wmoNameToIndex[normalizedPath] = index;
        return index;
    }
    
    /// <summary>
    /// Get all WMO names in order (for MWMO chunk).
    /// </summary>
    public IReadOnlyList<string> GetWmoNames() => _wmoNames;
    
    /// <summary>
    /// Get the current next unique ID (for tracking).
    /// </summary>
    public uint GetNextUniqueId() => _nextUniqueId;
    
    /// <summary>
    /// Set the next unique ID (for coordinating across tiles).
    /// </summary>
    public void SetNextUniqueId(uint id) => _nextUniqueId = id;
    
    #endregion
    
    #region Export
    
    /// <summary>
    /// Export MODF entries to CSV for debugging.
    /// </summary>
    public void ExportToCsv(IEnumerable<ModfEntry> entries, string path)
    {
        using var sw = new StreamWriter(path);
        sw.WriteLine("name_index,unique_id,wmo_path,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,bounds_min_x,bounds_min_y,bounds_min_z,bounds_max_x,bounds_max_y,bounds_max_z,flags,doodad_set,name_set,scale");
        
        foreach (var entry in entries)
        {
            string wmoPath = entry.NameIndex < _wmoNames.Count ? _wmoNames[entry.NameIndex] : "UNKNOWN";
            
            sw.WriteLine(string.Join(",",
                entry.NameIndex,
                entry.UniqueId,
                wmoPath,
                entry.Position.X.ToString("F4"),
                entry.Position.Y.ToString("F4"),
                entry.Position.Z.ToString("F4"),
                entry.Rotation.X.ToString("F2"),
                entry.Rotation.Y.ToString("F2"),
                entry.Rotation.Z.ToString("F2"),
                entry.BoundsMin.X.ToString("F2"),
                entry.BoundsMin.Y.ToString("F2"),
                entry.BoundsMin.Z.ToString("F2"),
                entry.BoundsMax.X.ToString("F2"),
                entry.BoundsMax.Y.ToString("F2"),
                entry.BoundsMax.Z.ToString("F2"),
                entry.Flags,
                entry.DoodadSet,
                entry.NameSet,
                entry.Scale
            ));
        }
        
        Console.WriteLine($"[INFO] Exported {entries.Count()} MODF entries to {path}");
    }
    
    /// <summary>
    /// Export WMO names to CSV (for MWMO chunk).
    /// </summary>
    public void ExportWmoNamesToCsv(string path)
    {
        using var sw = new StreamWriter(path);
        sw.WriteLine("index,wmo_path");
        
        for (int i = 0; i < _wmoNames.Count; i++)
        {
            sw.WriteLine($"{i},{_wmoNames[i]}");
        }
        
        Console.WriteLine($"[INFO] Exported {_wmoNames.Count} WMO names to {path}");
    }
    
    #endregion
    
    #region Helpers
    
    /// <summary>
    /// Calculate world-space bounding box for a WMO match.
    /// </summary>
    private static (Vector3 min, Vector3 max) CalculateWorldBounds(WmoMatch match)
    {
        var wmo = match.MatchedEntry;
        var yawRad = match.Rotation.Y * MathF.PI / 180f;
        
        // Get WMO local bounds
        var localMin = wmo.BoundsMin;
        var localMax = wmo.BoundsMax;
        
        // Get all 8 corners of the bounding box
        var corners = new[]
        {
            new Vector3(localMin.X, localMin.Y, localMin.Z),
            new Vector3(localMax.X, localMin.Y, localMin.Z),
            new Vector3(localMin.X, localMax.Y, localMin.Z),
            new Vector3(localMax.X, localMax.Y, localMin.Z),
            new Vector3(localMin.X, localMin.Y, localMax.Z),
            new Vector3(localMax.X, localMin.Y, localMax.Z),
            new Vector3(localMin.X, localMax.Y, localMax.Z),
            new Vector3(localMax.X, localMax.Y, localMax.Z),
        };
        
        // Rotate corners by yaw and translate
        var worldMin = new Vector3(float.MaxValue);
        var worldMax = new Vector3(float.MinValue);
        
        float cos = MathF.Cos(yawRad);
        float sin = MathF.Sin(yawRad);
        
        foreach (var corner in corners)
        {
            // Rotate around Y axis (yaw)
            var rotated = new Vector3(
                corner.X * cos - corner.Z * sin,
                corner.Y,
                corner.X * sin + corner.Z * cos
            );
            
            // Translate
            var world = rotated + match.Position;
            
            worldMin = Vector3.Min(worldMin, world);
            worldMax = Vector3.Max(worldMax, world);
        }
        
        return (worldMin, worldMax);
    }
    
    #endregion
}
