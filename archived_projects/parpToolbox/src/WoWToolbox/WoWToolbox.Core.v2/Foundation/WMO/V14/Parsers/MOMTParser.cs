using System;
using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    /// <summary>
    /// Very small subset parser for the MOMT material entries.
    /// Each entry is 64 bytes; offset 0 contains a uint32 index into the MOTX texture list.
    /// Returns the texture indices in order of materials.
    /// </summary>
    internal static class MOMTParser
    {
        public static List<uint> Parse(ReadOnlySpan<byte> payload)
        {
            const int ENTRY_SIZE = 64;
            var list = new List<uint>(payload.Length / ENTRY_SIZE);
            for (int offset = 0; offset + ENTRY_SIZE <= payload.Length; offset += ENTRY_SIZE)
            {
                uint texIdx = BitConverter.ToUInt32(payload[offset..(offset + 4)]);
                list.Add(texIdx);
            }
            return list;
        }
    }
}
