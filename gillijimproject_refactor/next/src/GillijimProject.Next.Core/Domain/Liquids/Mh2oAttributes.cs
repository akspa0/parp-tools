using System;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// MH2O per-chunk attributes. 8x8 bitmasks for fishable and deep.
/// </summary>
public sealed record Mh2oAttributes(
    /// <summary>8x8 bitmask; 1 = fishable/visible. Bit layout follows ADT v18 wiki (row-major 8x8).</summary>
    ulong FishableMask,
    /// <summary>8x8 bitmask; 1 = deep (often treated as fatigue).</summary>
    ulong DeepMask
);
