using System;
using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    internal static class MOPYParser
    {
        // Each entry is 2 bytes: 1 byte render flag, 1 byte material id.
        public static void Parse(ReadOnlySpan<byte> payload, List<byte> flags, List<ushort> materialIds)
        {
            int count = payload.Length / 2;
            flags.Capacity = count;
            materialIds.Capacity = count;
            for (int i = 0; i < count; i++)
            {
                flags.Add(payload[i*2]);
                materialIds.Add(payload[i*2+1]);
            }
        }

        // Back-compat overload returning only render flags
        public static List<byte> Parse(ReadOnlySpan<byte> payload)
        {
            var flags = new List<byte>();
            Parse(payload, flags, new List<ushort>());
            return flags;
        }
    }
}
