using System;

namespace WoWToolbox.Core.v2.Utilities
{
    /// <summary>
    /// Helper for decoding the LinkId value found in MSLK entries (commonly 0xFFFFYYXX).
    /// </summary>
    public static class LinkIdDecoder
    {
        /// <summary>
        /// Attempts to decode a raw 32-bit linkId (usually from MSLK Unknown_0x0C field) into tile coordinates.
        /// </summary>
        /// <param name="linkId">Raw 32-bit value.</param>
        /// <param name="tileX">Returns tile X if pattern recognised, otherwise null.</param>
        /// <param name="tileY">Returns tile Y if pattern recognised, otherwise null.</param>
        /// <returns>True if pattern matched (upper 16 bits == 0xFFFF and lower 16 bits decoded), else false.</returns>
        public static bool TryDecode(uint linkId, out int tileX, out int tileY)
        {
            // default
            tileX = tileY = 0;
            ushort high = (ushort)(linkId >> 16);
            ushort low = (ushort)(linkId & 0xFFFF);
            if (high != 0xFFFF) return false; // unknown schema

            // low word stored as YYXX (little-endian). Split bytes.
            byte yy = (byte)(low >> 8);
            byte xx = (byte)(low & 0xFF);
            tileX = xx;
            tileY = yy;
            return true;
        }
    }
}
