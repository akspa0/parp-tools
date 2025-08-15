using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Models
{
    /// <summary>
    /// Represents the fixed-size MOHD header chunk found at the start of every v14 WMO.
    /// Field list is taken from multiple open-source projects (wow.export, mirrormachine).
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MOHDHeader
    {
        public uint TextureCount;
        public uint GroupCount;
        public uint PortalCount;
        public uint LightCount;
        public uint DoodadNameCount;
        public uint DoodadDefCount;
        public uint DoodadSetCount;
        public uint? Unknown; // placeholder for alignment, varies across refs

        public static MOHDHeader FromSpan(ReadOnlySpan<byte> span)
        {
            if (span.Length < 28) // 7 * 4 bytes (we ignore Unknown for now)
                throw new ArgumentException("MOHD chunk too small", nameof(span));

            return new MOHDHeader
            {
                TextureCount = BinaryPrimitives.ReadUInt32LittleEndian(span[0..4]),
                GroupCount = BinaryPrimitives.ReadUInt32LittleEndian(span[4..8]),
                PortalCount = BinaryPrimitives.ReadUInt32LittleEndian(span[8..12]),
                LightCount = BinaryPrimitives.ReadUInt32LittleEndian(span[12..16]),
                DoodadNameCount = BinaryPrimitives.ReadUInt32LittleEndian(span[16..20]),
                DoodadDefCount = BinaryPrimitives.ReadUInt32LittleEndian(span[20..24]),
                DoodadSetCount = BinaryPrimitives.ReadUInt32LittleEndian(span[24..28]),
                Unknown = span.Length >= 32 ? BinaryPrimitives.ReadUInt32LittleEndian(span[28..32]) : null
            };
        }
    }
}
