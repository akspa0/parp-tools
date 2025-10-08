using System;
using System.Collections.Generic;

namespace WoWRollback.Core.Services.Config;

/// <summary>
/// Top-level area override configuration, keyed by map name, with optional per-version sections.
/// </summary>
public sealed class AreaOverrideConfig
{
    public Dictionary<string, MapAreaOverride> Maps { get; init; } = new(StringComparer.OrdinalIgnoreCase);
}

/// <summary>
/// Overrides for a single map. Supports default overrides plus version-specific blocks.
/// </summary>
public sealed class MapAreaOverride
{
    public Dictionary<string, VersionAreaOverride> Versions { get; init; } = new(StringComparer.OrdinalIgnoreCase);

    public List<AreaOverrideEntry> Overrides { get; init; } = new();
}

/// <summary>
/// Overrides scoped to a source version (e.g., 0.5.3).
/// </summary>
public sealed class VersionAreaOverride
{
    public List<AreaOverrideEntry> Overrides { get; init; } = new();
}

/// <summary>
/// Maps an Alpha area number (zone<<16 | sub) to a target LK AreaID.
/// </summary>
public sealed class AreaOverrideEntry
{
    /// <summary>
    /// Alpha area value (zone<<16 | sub). Accepts either integer or hex when parsed from JSON.
    /// </summary>
    public int AlphaArea { get; init; }

    /// <summary>
    /// Desired LK AreaTable ID to write for this chunk.
    /// </summary>
    public int TargetAreaId { get; init; }

    /// <summary>
    /// Optional note for diagnostics/logging.
    /// </summary>
    public string? Note { get; init; }
}
