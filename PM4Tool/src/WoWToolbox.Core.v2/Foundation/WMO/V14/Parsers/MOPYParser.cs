using System;
using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    internal static class MOPYParser
    {
        // For initial scaffolding we just read 1 byte per face (render flag) and skip material id.
        public static List<byte> Parse(ReadOnlySpan<byte> payload)
        {
            var list = new List<byte>(payload.Length);
            foreach (byte b in payload)
                list.Add(b);
            return list;
        }
    }
}
