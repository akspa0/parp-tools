using System;
using System.Collections.Generic;
using System.Buffers.Binary;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    internal static class MOVIParser
    {
        /// <summary>
        /// Parses MOVI payload – a sequence of uint16 vertex indices. Every consecutive triple forms a triangle.
        /// </summary>
        public static List<(ushort A, ushort B, ushort C)> Parse(ReadOnlySpan<byte> payload)
        {
            int indexCount = payload.Length / 2; // each index is uint16 little-endian
            if (indexCount % 3 != 0)
                indexCount -= indexCount % 3; // ignore trailing garbage if present

            var tris = new List<(ushort, ushort, ushort)>(indexCount / 3);
            for (int i = 0; i < indexCount; i += 3)
            {
                // Using BinaryPrimitives for little-endian safe read
                ushort a = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice(i * 2, 2));
                ushort b = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice((i + 1) * 2, 2));
                ushort c = BinaryPrimitives.ReadUInt16LittleEndian(payload.Slice((i + 2) * 2, 2));
                tris.Add((a, b, c));
            }
            return tris;
        }
    }
}
