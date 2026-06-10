using System;
using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    /// <summary>
    /// Parses the MOTV chunk (texture coordinates). Each entry is two little-endian floats (u,v).
    /// </summary>
    internal static class MOTVParser
    {
        public static List<Vector2> Parse(ReadOnlySpan<byte> payload)
        {
            var uvs = new List<Vector2>(payload.Length / 8);
            for (int offset = 0; offset + 8 <= payload.Length; offset += 8)
            {
                float u = BitConverter.ToSingle(payload[offset..(offset + 4)]);
                float v = BitConverter.ToSingle(payload[(offset + 4)..(offset + 8)]);
                uvs.Add(new Vector2(u, v));
            }
            return uvs;
        }
    }
}
