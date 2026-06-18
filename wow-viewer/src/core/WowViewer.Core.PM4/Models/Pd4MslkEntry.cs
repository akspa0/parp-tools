namespace WowViewer.Core.PM4.Models;

public sealed record Pd4MslkEntry(
    byte Flags,
    byte Unknown01,
    ushort Unknown02,
    uint Index04,
    uint Field08,
    uint Field0C,
    uint Field10,
    uint Field14)
{
    public override string ToString() =>
        $"Flags=0x{Flags:X2} Unk1=0x{Unknown01:X2} Unk2=0x{Unknown02:X4} " +
        $"Idx04={Index04} F08={Field08} F0C={Field0C} F10={Field10} F14={Field14}";
}
