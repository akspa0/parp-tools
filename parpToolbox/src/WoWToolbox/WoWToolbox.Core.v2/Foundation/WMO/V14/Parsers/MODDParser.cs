using System;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    /// <summary>
    /// Stub parser for the MODD (doodad set) chunk in v14 WMO groups.
    /// Only advances the read pointer; will be fully implemented later.
    /// </summary>
    internal static class MODDParser
    {
        public static void Parse(ReadOnlySpan<byte> payload)
        {
            // TODO: implement actual doodad parsing once structure is known
        }
    }
}
