using System;

namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// MCLQ per-tile auxiliary flags. The lower 4 bits of the raw byte store the liquid type
/// (see <see cref="MclqLiquidType"/>). These flags represent the upper bits.
/// </summary>
[Flags]
public enum MclqTileFlags : byte
{
    None = 0,

    /// <summary>
    /// Mask for the type bits in the original byte (0..3). Use <see cref="MclqLiquidType"/> instead.
    /// </summary>
    TypeMask = 0x0F,

    /// <summary>Unknown semantics (bit 0x10). Preserved when round-tripping.</summary>
    Unknown10 = 0x10,

    /// <summary>Unknown semantics (bit 0x20). Preserved when round-tripping.</summary>
    Unknown20 = 0x20,

    /// <summary>
    /// Not low depth / forced-swim (bit 0x40).
    /// </summary>
    ForcedSwim = 0x40,

    /// <summary>
    /// Fatigue area (bit 0x80).
    /// </summary>
    Fatigue = 0x80,
}
