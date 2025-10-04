using System.Collections.Generic;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds terrain properties overlay (impassible, vertex colors, etc.)
/// </summary>
public static class TerrainPropertiesOverlayBuilder
{
    public static object Build(List<McnkTerrainEntry> chunks, string version)
    {
        var impassible = chunks
            .Where(c => c.Impassible)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol })
            .ToList();

        var vertexColored = chunks
            .Where(c => c.HasMccv)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol })
            .ToList();

        var multiLayer = chunks
            .Where(c => c.NumLayers > 1)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol, layers = c.NumLayers })
            .ToList();

        return new
        {
            version,
            impassible,
            vertex_colored = vertexColored,
            multi_layer = multiLayer
        };
    }
}
