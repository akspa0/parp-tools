using System;
using System.Collections.Generic;

namespace WoWToolbox.Core.v2.Foundation.WMO.V17.Parsers
{
    /// <summary>
    /// Parses MOPY chunk from WMO v17 group files. Each entry is 2 bytes:
    /// byte 0 = material ID (index into MOMT list)
    /// byte 1 = flags (bitmask: 0x04 = collision-only, etc.)
    /// </summary>
    public static class MOPYParser
    {
        public readonly record struct MopyEntry(byte MaterialId, byte Flags);

        public static List<MopyEntry> Parse(byte[] data)
        {
            var list = new List<MopyEntry>(data.Length / 2);
            for (int i = 0; i + 1 < data.Length; i += 2)
            {
                byte mat = data[i];
                byte flags = data[i + 1];
                list.Add(new MopyEntry(mat, flags));
            }
            return list;
        }
    }
}
