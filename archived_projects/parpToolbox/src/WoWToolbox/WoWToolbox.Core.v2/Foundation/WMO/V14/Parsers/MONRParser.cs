using System;
using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    /// <summary>
    /// Parses the MONR chunk which contains one <see cref="Vector3"/> normal per vertex (little-endian floats).
    /// </summary>
    internal static class MONRParser
    {
        public static List<Vector3> Parse(ReadOnlySpan<byte> payload)
        {
            var normals = new List<Vector3>(payload.Length / 12);
            for (int offset = 0; offset + 12 <= payload.Length; offset += 12)
            {
                float x = BitConverter.ToSingle(payload[offset..(offset + 4)]);
                float y = BitConverter.ToSingle(payload[(offset + 4)..(offset + 8)]);
                float z = BitConverter.ToSingle(payload[(offset + 8)..(offset + 12)]);
                normals.Add(new Vector3(x, y, z));
            }
            return normals;
        }
    }
}
