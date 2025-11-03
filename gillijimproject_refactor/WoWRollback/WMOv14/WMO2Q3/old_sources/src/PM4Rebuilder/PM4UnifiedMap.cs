using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;

namespace PM4Rebuilder;

/// <summary>
/// Unified PM4 map containing all data from multiple PM4 tiles with cross-tile references resolved.
/// This is the core data structure that enables proper building assembly from distributed PM4 data.
/// </summary>
public class PM4UnifiedMap
{
    /// <summary>
    /// Global vertex pool from all MSVT chunks across all tiles.
    /// Indices are remapped to be global rather than per-tile.
    /// </summary>
    public List<Vector3> GlobalMSVTVertices { get; set; } = new();

    /// <summary>
    /// Global vertex pool from all MSPV chunks across all tiles.
    /// Indices are remapped to be global rather than per-tile.
    /// </summary>
    public List<Vector3> GlobalMSPVVertices { get; set; } = new();

    /// <summary>
    /// Global index pool from all MSVI chunks across all tiles.
    /// All indices corrected to reference global vertex pools.
    /// </summary>
    public List<uint> GlobalMSVIIndices { get; set; } = new();

    /// <summary>
    /// Global index pool from all MSPI chunks across all tiles.
    /// All indices corrected to reference global vertex pools.
    /// </summary>
    public List<uint> GlobalMSPIIndices { get; set; } = new();

    /// <summary>
    /// All MSLK linkage entries from all tiles.
    /// Contains the critical parent-child relationships for building assembly.
    /// </summary>
    public List<MslkEntry> AllMslkLinks { get; set; } = new();

    /// <summary>
    /// All MPRL placement entries from all tiles.
    /// Contains building positions and object IDs.
    /// </summary>
    public List<MprlChunk.Entry> AllMprlPlacements { get; set; } = new();

    /// <summary>
    /// All MSUR surface entries from all tiles.
    /// Contains geometry surface definitions and grouping keys.
    /// </summary>
    public List<MsurChunk.Entry> AllMsurSurfaces { get; set; } = new();

    /// <summary>
    /// Vertex offset information for each tile to enable cross-tile reference resolution.
    /// Maps (tileX, tileY) -> vertex offset information.
    /// </summary>
    public Dictionary<(int tileX, int tileY), PM4TileVertexOffsets> TileVertexOffsets { get; set; } = new();

    /// <summary>
    /// Metadata about each loaded tile for reference and debugging.
    /// Maps (tileX, tileY) -> tile metadata.
    /// </summary>
    public Dictionary<(int tileX, int tileY), PM4TileMetadata> TileData { get; set; } = new();

    /// <summary>
    /// Get total vertex count across both pools.
    /// </summary>
    public int TotalVertexCount => GlobalMSVTVertices.Count + GlobalMSPVVertices.Count;

    /// <summary>
    /// Get total index count across both pools.
    /// </summary>
    public int TotalIndexCount => GlobalMSVIIndices.Count + GlobalMSPIIndices.Count;

    /// <summary>
    /// Get total linkage entry count.
    /// </summary>
    public int TotalLinkageCount => AllMslkLinks.Count + AllMprlPlacements.Count + AllMsurSurfaces.Count;

    /// <summary>
    /// Get bounds of the entire unified map.
    /// </summary>
    public (Vector3 min, Vector3 max) GetMapBounds()
    {
        if (TotalVertexCount == 0)
            return (Vector3.Zero, Vector3.Zero);

        var allVertices = new List<Vector3>();
        allVertices.AddRange(GlobalMSVTVertices);
        allVertices.AddRange(GlobalMSPVVertices);

        var min = new Vector3(
            allVertices.Min(v => v.X),
            allVertices.Min(v => v.Y),
            allVertices.Min(v => v.Z)
        );

        var max = new Vector3(
            allVertices.Max(v => v.X),
            allVertices.Max(v => v.Y),
            allVertices.Max(v => v.Z)
        );

        return (min, max);
    }

    /// <summary>
    /// Get summary statistics for debugging and validation.
    /// </summary>
    public PM4UnifiedMapSummary GetSummary()
    {
        var bounds = GetMapBounds();
        
        return new PM4UnifiedMapSummary
        {
            TileCount = TileData.Count,
            TotalMSVTVertices = GlobalMSVTVertices.Count,
            TotalMSPVVertices = GlobalMSPVVertices.Count,
            TotalMSVIIndices = GlobalMSVIIndices.Count,
            TotalMSPIIndices = GlobalMSPIIndices.Count,
            TotalMslkLinks = AllMslkLinks.Count,
            TotalMprlPlacements = AllMprlPlacements.Count,
            TotalMsurSurfaces = AllMsurSurfaces.Count,
            MapBoundsMin = bounds.min,
            MapBoundsMax = bounds.max,
            MapSize = bounds.max - bounds.min
        };
    }
}

/// <summary>
/// Vertex offset information for a specific tile within the unified map.
/// Used for cross-tile reference resolution and index remapping.
/// </summary>
public class PM4TileVertexOffsets
{
    /// <summary>
    /// Starting index of this tile's MSVT vertices in the global MSVT vertex pool.
    /// </summary>
    public int MSVTStartIndex { get; set; }

    /// <summary>
    /// Number of MSVT vertices contributed by this tile.
    /// </summary>
    public int MSVTCount { get; set; }

    /// <summary>
    /// Starting index of this tile's MSPV vertices in the global MSPV vertex pool.
    /// </summary>
    public int MSPVStartIndex { get; set; }

    /// <summary>
    /// Number of MSPV vertices contributed by this tile.
    /// </summary>
    public int MSPVCount { get; set; }

    /// <summary>
    /// Check if a vertex index is within this tile's bounds for the specified pool.
    /// </summary>
    public bool IsVertexIndexInBounds(uint index, bool isMSVTPool)
    {
        var maxCount = isMSVTPool ? MSVTCount : MSPVCount;
        return index < maxCount;
    }

    /// <summary>
    /// Convert a local tile vertex index to global unified map index.
    /// </summary>
    public uint GetGlobalVertexIndex(uint localIndex, bool isMSVTPool)
    {
        var baseOffset = isMSVTPool ? MSVTStartIndex : MSPVStartIndex;
        return (uint)(baseOffset + localIndex);
    }
}

/// <summary>
/// Metadata about a loaded PM4 tile.
/// </summary>
public class PM4TileMetadata
{
    /// <summary>
    /// Tile X coordinate.
    /// </summary>
    public int TileX { get; set; }

    /// <summary>
    /// Tile Y coordinate.
    /// </summary>
    public int TileY { get; set; }

    /// <summary>
    /// File path of the original PM4 file.
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Total vertex count contributed by this tile.
    /// </summary>
    public int VertexCount { get; set; }

    /// <summary>
    /// Total MSLK link count in this tile.
    /// </summary>
    public int LinkCount { get; set; }

    /// <summary>
    /// Total MSUR surface count in this tile.
    /// </summary>
    public int SurfaceCount { get; set; }

    /// <summary>
    /// Whether this tile was successfully loaded.
    /// </summary>
    public bool IsLoaded { get; set; } = true;

    /// <summary>
    /// Any error message encountered during loading.
    /// </summary>
    public string? ErrorMessage { get; set; }
}

/// <summary>
/// Raw tile data loaded from a single PM4 file before unification.
/// </summary>
public class PM4TileData
{
    public int TileX { get; set; }
    public int TileY { get; set; }
    public string FilePath { get; set; } = string.Empty;
    public Pm4Scene? Scene { get; set; }

    // Raw chunk data extracted from this tile
    public List<Vector3> MSVTVertices { get; set; } = new();
    public List<Vector3> MSPVVertices { get; set; } = new();
    public List<uint> MSVIIndices { get; set; } = new();
    public List<uint> MSPIIndices { get; set; } = new();
    public List<MslkEntry> MslkLinks { get; set; } = new();
    public List<MprlChunk.Entry> MprlPlacements { get; set; } = new();
    public List<MsurChunk.Entry> MsurSurfaces { get; set; } = new();

    /// <summary>
    /// Starting index of this tile's first MSUR surface in the unified global surface list.
    /// </summary>
    public int SurfaceOffset { get; set; }
}

/// <summary>
/// Summary statistics for a unified PM4 map.
/// </summary>
public class PM4UnifiedMapSummary
{
    public int TileCount { get; set; }
    public int TotalMSVTVertices { get; set; }
    public int TotalMSPVVertices { get; set; }
    public int TotalMSVIIndices { get; set; }
    public int TotalMSPIIndices { get; set; }
    public int TotalMslkLinks { get; set; }
    public int TotalMprlPlacements { get; set; }
    public int TotalMsurSurfaces { get; set; }
    public Vector3 MapBoundsMin { get; set; }
    public Vector3 MapBoundsMax { get; set; }
    public Vector3 MapSize { get; set; }

    public int TotalVertices => TotalMSVTVertices + TotalMSPVVertices;
    public int TotalIndices => TotalMSVIIndices + TotalMSPIIndices;
    public int TotalLinkageEntries => TotalMslkLinks + TotalMprlPlacements + TotalMsurSurfaces;

    public override string ToString()
    {
        return $"PM4 Unified Map: {TileCount} tiles, {TotalVertices:N0} vertices, {TotalIndices:N0} indices, {TotalLinkageEntries:N0} linkage entries";
    }
}
