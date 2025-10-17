using System.Collections.Generic;

namespace WoWRollback.LkToAlphaModule.Models;

/// <summary>
/// Aggregates all data required to construct a Lich King ADT.
/// </summary>
public sealed class LkAdtSource
{
    public required string MapName { get; init; }
    public required int TileX { get; init; }
    public required int TileY { get; init; }

    /// <summary>
    /// MCNK sources in row-major 16x16 order.
    /// </summary>
    public List<LkMcnkSource> Mcnks { get; } = new();
}
