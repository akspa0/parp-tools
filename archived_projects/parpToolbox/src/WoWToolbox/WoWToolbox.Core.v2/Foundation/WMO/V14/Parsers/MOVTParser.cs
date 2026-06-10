using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    internal static class MOVTParser
    {
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        private struct Vec3f
        {
            public float X;
            public float Y;
            public float Z;
        }

        public static List<Vector3> Parse(ReadOnlySpan<byte> payload)
        {
            int count = payload.Length / 12; // 3 floats
            var list = new List<Vector3>(count);
            for (int i = 0; i < count; i++)
            {
                var span = payload.Slice(i * 12, 12);
                var vec = MemoryMarshal.Cast<byte, Vec3f>(span)[0];
                list.Add(new Vector3(vec.X, vec.Y, vec.Z));
            }
            return list;
        }
    }
}
