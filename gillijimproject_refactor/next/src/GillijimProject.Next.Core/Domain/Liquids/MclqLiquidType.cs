namespace GillijimProject.Next.Core.Domain.Liquids;

/// <summary>
/// MCLQ per-tile liquid type (lower 4 bits of the tile flags).
/// </summary>
public enum MclqLiquidType : byte
{
    None = 0,
    Ocean = 1,
    Slime = 3,
    River = 4,
    Magma = 6,
}
