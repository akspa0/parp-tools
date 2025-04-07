using System;

namespace WoWToolbox.Core.Legacy.Liquid
{
    /// <summary>
    /// Flags used in the MCLQ chunk to define liquid properties
    /// </summary>
    [Flags]
    public enum MCLQFlags : byte
    {
        /// <summary>
        /// No flags set
        /// </summary>
        None = 0,

        /// <summary>
        /// This chunk has liquid
        /// </summary>
        HasLiquid = 0x01,

        /// <summary>
        /// Liquid is hidden
        /// </summary>
        Hidden = 0x02,

        /// <summary>
        /// Liquid has alpha map
        /// </summary>
        HasAlpha = 0x04,

        /// <summary>
        /// Liquid is fishable
        /// </summary>
        Fishable = 0x08,

        /// <summary>
        /// Shared with adjacent chunks
        /// </summary>
        Shared = 0x10
    }
} 