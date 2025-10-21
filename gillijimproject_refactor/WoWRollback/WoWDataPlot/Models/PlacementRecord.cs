namespace WoWDataPlot.Models;

/// <summary>
/// Represents a single object placement (M2 or WMO) extracted from WDT/ADT files.
/// Used for plotting UniqueID distribution and analyzing development timeline.
/// </summary>
public record PlacementRecord
{
    /// <summary>World X coordinate</summary>
    public float X { get; init; }
    
    /// <summary>World Y coordinate</summary>
    public float Y { get; init; }
    
    /// <summary>World Z coordinate (height)</summary>
    public float Z { get; init; }
    
    /// <summary>UniqueID from placement data</summary>
    public uint UniqueId { get; init; }
    
    /// <summary>Object type: M2 or WMO</summary>
    public string Type { get; init; } = string.Empty;
    
    /// <summary>Model/WMO name if available</summary>
    public string Name { get; init; } = string.Empty;
    
    /// <summary>AreaID from containing MCNK chunk</summary>
    public uint AreaId { get; init; }
    
    /// <summary>Tile X index (0-63)</summary>
    public int TileX { get; init; }
    
    /// <summary>Tile Y index (0-63)</summary>
    public int TileY { get; init; }
    
    /// <summary>File format version (Alpha, LK, etc.)</summary>
    public string Version { get; init; } = string.Empty;
}
