using System.Collections.Generic;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds liquids overlay (river, ocean, magma, slime)
/// </summary>
public static class LiquidsOverlayBuilder
{
    public static object Build(List<McnkTerrainEntry> chunks, string version)
    {
        var river = chunks
            .Where(c => c.LqRiver)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol })
            .ToList();

        var ocean = chunks
            .Where(c => c.LqOcean)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol })
            .ToList();

        var magma = chunks
            .Where(c => c.LqMagma)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol })
            .ToList();

        var slime = chunks
            .Where(c => c.LqSlime)
            .Select(c => new { row = c.ChunkRow, col = c.ChunkCol })
            .ToList();

        return new
        {
            version,
            river,
            ocean,
            magma,
            slime
        };
    }
}
