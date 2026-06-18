using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public sealed record Pd4MsurEntry(
    byte Flags,
    byte IndexCount,
    byte Unknown02,
    byte Padding,
    Vector3 Normal,
    float Height,
    uint FirstIndex,
    uint RefIndex,
    uint Zero)
{
    public override string ToString() =>
        $"Flags=0x{Flags:X2} IndexCount={IndexCount} Unknown=0x{Unknown02:X2} " +
        $"Normal=({Normal.X:F4},{Normal.Y:F4},{Normal.Z:F4}) Height={Height:F4} " +
        $"FirstIndex={FirstIndex} RefIndex={RefIndex} Zero={Zero}";
}
