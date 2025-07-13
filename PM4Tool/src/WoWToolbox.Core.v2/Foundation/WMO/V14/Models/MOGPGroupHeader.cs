using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Models
{
    /// <summary>
    /// Represents the 80-byte MOGP group header (v17) inside a v14 WMO file.
    /// Only fields essential for geometry extraction are included for now.
    /// Layout reference: wow.export & mirrormachine.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MOGPGroupHeader
    {
        public uint NameOffset;
        public uint Description;
        public uint Flags;
        public ushort BoundingBoxMinX;
        public ushort BoundingBoxMinY;
        public ushort BoundingBoxMinZ;
        public ushort BoundingBoxMaxX;
        public ushort BoundingBoxMaxY;
        public ushort BoundingBoxMaxZ;
        public int LiquidType;
        public int WmoID;
        public uint DuplicateNameOffset;
        public uint Unknown1;
        public uint Unknown2;
        public uint Unknown3;

        // Post-v14 geometry offsets (present in v17)
        public uint FirstVertex;
        public uint VertexCount;
        public uint FirstIndex;
        public uint IndexCount;

        public static MOGPGroupHeader FromSpan(ReadOnlySpan<byte> span)
        {
            if (span.Length < 80)
                throw new ArgumentException("MOGP header requires 64 bytes", nameof(span));

            return new MOGPGroupHeader
            {
                NameOffset = BinaryPrimitives.ReadUInt32LittleEndian(span[0..4]),
                Description = BinaryPrimitives.ReadUInt32LittleEndian(span[4..8]),
                Flags = BinaryPrimitives.ReadUInt32LittleEndian(span[8..12]),
                BoundingBoxMinX = BinaryPrimitives.ReadUInt16LittleEndian(span[12..14]),
                BoundingBoxMinY = BinaryPrimitives.ReadUInt16LittleEndian(span[14..16]),
                BoundingBoxMinZ = BinaryPrimitives.ReadUInt16LittleEndian(span[16..18]),
                BoundingBoxMaxX = BinaryPrimitives.ReadUInt16LittleEndian(span[18..20]),
                BoundingBoxMaxY = BinaryPrimitives.ReadUInt16LittleEndian(span[20..22]),
                BoundingBoxMaxZ = BinaryPrimitives.ReadUInt16LittleEndian(span[22..24]),
                LiquidType = BinaryPrimitives.ReadInt32LittleEndian(span[24..28]),
                WmoID = BinaryPrimitives.ReadInt32LittleEndian(span[28..32]),
                DuplicateNameOffset = BinaryPrimitives.ReadUInt32LittleEndian(span[32..36]),
                Unknown1 = BinaryPrimitives.ReadUInt32LittleEndian(span[36..40]),
                Unknown2 = BinaryPrimitives.ReadUInt32LittleEndian(span[40..44]),
                Unknown3 = BinaryPrimitives.ReadUInt32LittleEndian(span[44..48]),
                FirstVertex = BinaryPrimitives.ReadUInt32LittleEndian(span[64..68]),
                VertexCount = BinaryPrimitives.ReadUInt32LittleEndian(span[68..72]),
                FirstIndex = BinaryPrimitives.ReadUInt32LittleEndian(span[72..76]),
                IndexCount = BinaryPrimitives.ReadUInt32LittleEndian(span[76..80])
            };
        }
    }
}
