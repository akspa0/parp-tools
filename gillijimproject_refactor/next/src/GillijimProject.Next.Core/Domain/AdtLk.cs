namespace GillijimProject.Next.Core.Domain;

/// <summary>
/// Lich King (v18) ADT model placeholder for downstream writing via Warcraft.NET.
/// </summary>
/// <remarks>
/// Invariants: FourCC forward in-memory; MFBO/MTXF included when present; MCLQ written last within MCNK; omit MH2O when empty.
/// </remarks>
public sealed record AdtLk(string Name);
