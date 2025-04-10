namespace WoWToolbox.Core.Legacy.Liquid
{
    /// <summary>
    /// Types of liquid that can be defined in MCLQ chunks
    /// </summary>
    public enum LiquidType : ushort
    {
        /// <summary>
        /// Standard water
        /// </summary>
        Water = 0,

        /// <summary>
        /// Ocean water
        /// </summary>
        Ocean = 1,

        /// <summary>
        /// Lava/magma
        /// </summary>
        Magma = 2,

        /// <summary>
        /// Slime/ooze
        /// </summary>
        Slime = 3,

        /// <summary>
        /// WMO-specific liquid
        /// </summary>
        WMO = 4
    }
} 