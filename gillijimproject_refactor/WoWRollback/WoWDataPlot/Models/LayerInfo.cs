namespace WoWDataPlot.Models;

/// <summary>
/// Represents a "layer" of content based on UniqueID ranges.
/// Allows filtering content by development timeline.
/// </summary>
public record LayerInfo
{
    /// <summary>Layer name (e.g., "Alpha 0.5.3", "Alpha 0.6.0", "Experimental")</summary>
    public string Name { get; init; } = string.Empty;
    
    /// <summary>Minimum UniqueID in this layer (inclusive)</summary>
    public uint MinUniqueId { get; init; }
    
    /// <summary>Maximum UniqueID in this layer (inclusive)</summary>
    public uint MaxUniqueId { get; init; }
    
    /// <summary>Number of placements in this layer</summary>
    public int PlacementCount { get; init; }
    
    /// <summary>Color code for visualization (hex RGB)</summary>
    public string Color { get; init; } = "#0000FF";
}

/// <summary>
/// Represents layer analysis for a single ADT tile.
/// </summary>
public record TileLayerInfo
{
    /// <summary>Tile X coordinate (0-63)</summary>
    public int TileX { get; init; }
    
    /// <summary>Tile Y coordinate (0-63)</summary>
    public int TileY { get; init; }
    
    /// <summary>Total placements in this tile</summary>
    public int TotalPlacements { get; init; }
    
    /// <summary>Layers present in this tile</summary>
    public List<LayerInfo> Layers { get; init; } = new();
    
    /// <summary>Path to layer visualization image (relative)</summary>
    public string? ImagePath { get; init; }
}

/// <summary>
/// Complete layer analysis for entire WDT.
/// </summary>
public record WdtLayerAnalysis
{
    /// <summary>WDT file name</summary>
    public string WdtName { get; init; } = string.Empty;
    
    /// <summary>Total placements across all tiles</summary>
    public int TotalPlacements { get; init; }
    
    /// <summary>Global UniqueID range</summary>
    public uint MinUniqueId { get; init; }
    public uint MaxUniqueId { get; init; }
    
    /// <summary>Defined layers for this WDT</summary>
    public List<LayerInfo> GlobalLayers { get; init; } = new();
    
    /// <summary>Per-tile layer breakdown</summary>
    public List<TileLayerInfo> Tiles { get; init; } = new();
    
    /// <summary>Analysis timestamp</summary>
    public DateTime AnalyzedAt { get; init; } = DateTime.UtcNow;
}
