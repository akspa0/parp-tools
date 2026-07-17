namespace WowViewer.Core.Maps;

/// <summary>
/// Canonical decoder for MCNK chunk flag bits that indicate liquid type.
/// All wow-viewer code paths that map MCNK flags to <see cref="AdtLiquidBasicType"/>
/// MUST call through this helper. No code in the project should read raw bit
/// fields from MCNK flags to derive a liquid type.
/// </summary>
/// <remarks>
/// Bit assignments (canonical, per wowdev.wiki MCLQ tile flags and the 3.3.5
/// Ghidra evidence in <c>specs/040-mh2o-mclq-liquid-type-determination/research.md</c>):
/// <list type="bullet">
///   <item><description><c>0x04</c> = River (rendered as water by the wow-viewer renderer)</description></item>
///   <item><description><c>0x08</c> = Ocean</description></item>
///   <item><description><c>0x10</c> = Magma</description></item>
///   <item><description><c>0x20</c> = Slime</description></item>
/// </list>
/// Precedence (when multiple bits are set, which is malformed data):
/// Magma (0x10) is checked before Slime (0x20). This matches
/// <c>LiquidConverter.GetLiquidTypeFromMcnkFlags</c> in
/// <c>WowViewer.Core.IO/Liquids/LiquidConverter.cs</c>, the only correct
/// function that returns the canonical <c>MclqLiquidType</c> enum and feeds
/// the bidirection conversion path <c>LiquidConverter.MclqToMh2o</c>.
/// <para>
/// <c>AlphaLiquidTypeCodec.ResolveBasicType</c> delegates to this decoder so
/// Alpha reader, tensor, and conversion paths share the same raw MCLQ mapping.
/// The 0x30 case (both Magma and Slime bits set) is treated as Magma.
/// </para>
/// </remarks>
public static class McnkFlagDecoder
{
    private const uint McnkRiverBit = 0x04u;
    private const uint McnkOceanBit = 0x08u;
    private const uint McnkMagmaBit = 0x10u;
    private const uint McnkSlimeBit = 0x20u;

    // These are raw MCLQ lower-nibble values, not AdtLiquidBasicType values.
    // In particular, 0x04 means River (blue water), not Slime.
    private const byte MclqTileNibbleOcean = 0x01;
    private const byte MclqTileNibbleSlime = 0x03;
    private const byte MclqTileNibbleRiver = 0x04;
    private const byte MclqTileNibbleMagma = 0x06;
    private const byte MclqTileNibbleDontRender = 0x0F;

    /// <summary>
    /// Decode MCNK chunk flags into an <see cref="AdtLiquidBasicType"/>.
    /// Falls through bit checks in this order: Magma (0x10), Slime (0x20),
    /// Ocean (0x08), River (0x04). Default is <see cref="AdtLiquidBasicType.Water"/>.
    /// </summary>
    /// <param name="mcnkFlags">Raw MCNK flags (uint). Bit 2-5 are the liquid bits.</param>
    /// <returns>Resolved <see cref="AdtLiquidBasicType"/>.</returns>
    public static AdtLiquidBasicType Decode(uint mcnkFlags)
    {
        if ((mcnkFlags & McnkMagmaBit) != 0)
            return AdtLiquidBasicType.Magma;

        if ((mcnkFlags & McnkSlimeBit) != 0)
            return AdtLiquidBasicType.Slime;

        if ((mcnkFlags & McnkOceanBit) != 0)
            return AdtLiquidBasicType.Ocean;

        if ((mcnkFlags & McnkRiverBit) != 0)
            return AdtLiquidBasicType.Water;

        return AdtLiquidBasicType.Water;
    }

    /// <summary>
    /// Decode liquid type using the per-tile MCLQ nibble as primary source and
    /// MCNK flags as fallback. Matches the intent of
    /// <c>AlphaLiquidTypeCodec.ResolveBasicType</c> in
    /// <c>WowViewer.Core.IO/Maps/AlphaLiquidTypeCodec.cs</c>: when the per-tile
    /// nibble is non-zero and not <c>0x0F</c> (DontRender), the nibble wins.
    /// </summary>
    /// <param name="mcnkFlags">Raw MCNK flags (uint). Used when the nibble is 0 or DontRender.</param>
    /// <param name="mclqTileNibble">Lower 4 bits of an MCLQ tile flag byte. Raw MCLQ uses
    /// 0x01=Ocean, 0x03=Slime, 0x04=River/Water, and 0x06=Magma.</param>
    /// <returns>Resolved <see cref="AdtLiquidBasicType"/>.</returns>
    public static AdtLiquidBasicType DecodeWithMclqTileNibble(uint mcnkFlags, byte mclqTileNibble)
    {
        if (mclqTileNibble != 0 && mclqTileNibble != MclqTileNibbleDontRender)
        {
            return mclqTileNibble switch
            {
                MclqTileNibbleOcean => AdtLiquidBasicType.Ocean,
                MclqTileNibbleMagma => AdtLiquidBasicType.Magma,
                MclqTileNibbleSlime => AdtLiquidBasicType.Slime,
                MclqTileNibbleRiver => AdtLiquidBasicType.Water,
                _ => AdtLiquidBasicType.Water,
            };
        }

        return Decode(mcnkFlags);
    }
}
