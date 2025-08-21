using System;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    /// <summary>
    /// Stub parser for the MOLT (light data) chunk in v14 WMO groups.
    /// For the current scaffolding phase we only advance the stream; structure will be fleshed out later.
    /// </summary>
    internal static class MOLTParser
    {
        public static void Parse(ReadOnlySpan<byte> payload)
        {
            // TODO: implement actual light parsing once structure is verified
        }
    }
}
