using System.Collections.Generic;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds AreaID overlay with boundary detection
/// </summary>
public static class AreaIdOverlayBuilder
{
    public static object Build(List<McnkTerrainEntry> chunks, string version, AreaTableLookup areaLookup)
    {
        // Create grid for boundary detection
        var grid = new Dictionary<(int row, int col), int>();
        foreach (var chunk in chunks)
        {
            grid[(chunk.ChunkRow, chunk.ChunkCol)] = chunk.AreaId;
        }

        // Build chunk list with area names
        var areaChunks = chunks
            .Select(c => new
            {
                row = c.ChunkRow,
                col = c.ChunkCol,
                area_id = c.AreaId,
                area_name = areaLookup.GetName(c.AreaId, preferAlpha: false) // Use LK names (3.3.5 AreaTable)
            })
            .ToList();

        // Detect boundaries
        var boundaries = new List<AreaBoundary>();
        foreach (var chunk in chunks)
        {
            var currentArea = chunk.AreaId;
            var row = chunk.ChunkRow;
            var col = chunk.ChunkCol;

            // Check north neighbor
            if (grid.TryGetValue((row - 1, col), out var northArea) && northArea != currentArea)
            {
                boundaries.Add(new AreaBoundary(
                    FromArea: currentArea,
                    FromName: areaLookup.GetName(currentArea, preferAlpha: false),
                    ToArea: northArea,
                    ToName: areaLookup.GetName(northArea, preferAlpha: false),
                    ChunkRow: row,
                    ChunkCol: col,
                    Edge: "north"
                ));
            }

            // Check east neighbor
            if (grid.TryGetValue((row, col + 1), out var eastArea) && eastArea != currentArea)
            {
                boundaries.Add(new AreaBoundary(
                    FromArea: currentArea,
                    FromName: areaLookup.GetName(currentArea, preferAlpha: false),
                    ToArea: eastArea,
                    ToName: areaLookup.GetName(eastArea, preferAlpha: false),
                    ChunkRow: row,
                    ChunkCol: col,
                    Edge: "east"
                ));
            }

            // Check south neighbor
            if (grid.TryGetValue((row + 1, col), out var southArea) && southArea != currentArea)
            {
                boundaries.Add(new AreaBoundary(
                    FromArea: currentArea,
                    FromName: areaLookup.GetName(currentArea, preferAlpha: false),
                    ToArea: southArea,
                    ToName: areaLookup.GetName(southArea, preferAlpha: false),
                    ChunkRow: row,
                    ChunkCol: col,
                    Edge: "south"
                ));
            }

            // Check west neighbor
            if (grid.TryGetValue((row, col - 1), out var westArea) && westArea != currentArea)
            {
                boundaries.Add(new AreaBoundary(
                    FromArea: currentArea,
                    FromName: areaLookup.GetName(currentArea, preferAlpha: false),
                    ToArea: westArea,
                    ToName: areaLookup.GetName(westArea, preferAlpha: false),
                    ChunkRow: row,
                    ChunkCol: col,
                    Edge: "west"
                ));
            }
        }

        var boundaryData = boundaries.Select(b => new
        {
            from_area = b.FromArea,
            from_name = b.FromName,
            to_area = b.ToArea,
            to_name = b.ToName,
            chunk_row = b.ChunkRow,
            chunk_col = b.ChunkCol,
            edge = b.Edge
        }).ToList();

        return new
        {
            version,
            chunks = areaChunks,
            boundaries = boundaryData
        };
    }
}
