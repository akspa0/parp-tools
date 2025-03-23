using System;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Map Object Header chunk. Contains global information about the WMO.
    /// </summary>
    public class MOHD : IChunk
    {
        public uint TextureCount { get; set; }
        public uint GroupCount { get; set; }
        public uint PortalCount { get; set; }
        public uint LightCount { get; set; }
        public uint DoodadNameCount { get; set; }
        public uint DoodadDefinitionCount { get; set; }
        public uint DoodadSetCount { get; set; }
        public ColorBGRA AmbientColor { get; set; }
        public uint WmoId { get; set; }
        public BoundingBox BoundingBox { get; set; }
        public WmoFlags Flags { get; set; }
        public ushort LodLevels { get; set; }

        // v14 specific fields
        public bool IsV14 { get; private set; }
        public byte[] InMemoryPadding { get; private set; }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;

            TextureCount = reader.ReadUInt32();
            GroupCount = reader.ReadUInt32();
            PortalCount = reader.ReadUInt32();
            LightCount = reader.ReadUInt32();
            DoodadNameCount = reader.ReadUInt32();
            DoodadDefinitionCount = reader.ReadUInt32();
            DoodadSetCount = reader.ReadUInt32();
            AmbientColor = new ColorBGRA(reader.ReadUInt32());
            WmoId = reader.ReadUInt32();

            BoundingBox = new BoundingBox
            {
                Min = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Max = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle())
            };

            // Check version to determine remaining structure
            var chunkSize = reader.BaseStream.Length - startPos;
            IsV14 = chunkSize == 0x40; // v14 has 64 bytes total

            if (IsV14)
            {
                InMemoryPadding = reader.ReadBytes(8); // v14 has 8 bytes of padding
                Flags = WmoFlags.None; // v14 doesn't have flags
                LodLevels = 0; // v14 doesn't have LOD
            }
            else
            {
                Flags = (WmoFlags)reader.ReadUInt16();
                LodLevels = reader.ReadUInt16();
            }

            ValidateData();
        }

        public void Write(BinaryWriter writer)
        {
            writer.Write(TextureCount);
            writer.Write(GroupCount);
            writer.Write(PortalCount);
            writer.Write(LightCount);
            writer.Write(DoodadNameCount);
            writer.Write(DoodadDefinitionCount);
            writer.Write(DoodadSetCount);
            writer.Write(AmbientColor.ToUInt32());
            writer.Write(WmoId);

            writer.Write(BoundingBox.Min.X);
            writer.Write(BoundingBox.Min.Y);
            writer.Write(BoundingBox.Min.Z);
            writer.Write(BoundingBox.Max.X);
            writer.Write(BoundingBox.Max.Y);
            writer.Write(BoundingBox.Max.Z);

            if (IsV14)
            {
                writer.Write(InMemoryPadding ?? new byte[8]);
            }
            else
            {
                writer.Write((ushort)Flags);
                writer.Write(LodLevels);
            }
        }

        private void ValidateData()
        {
            if (GroupCount == 0 || GroupCount > 512)
                throw new InvalidDataException($"Invalid group count: {GroupCount}. Must be between 1 and 512.");

            if (BoundingBox.Min.X > BoundingBox.Max.X ||
                BoundingBox.Min.Y > BoundingBox.Max.Y ||
                BoundingBox.Min.Z > BoundingBox.Max.Z)
                throw new InvalidDataException("Invalid bounding box: min values must be less than max values.");
        }
    }

    [Flags]
    public enum WmoFlags : ushort
    {
        None = 0x0000,
        DoNotAttenuateVerticesBasedOnDistanceToPortal = 0x0001,
        UseUnifiedRenderPath = 0x0002,
        UseLiquidTypeDBC = 0x0004,
        DoNotFixVertexColorAlpha = 0x0008,
        HasLod = 0x0010,
        DefaultMaxLod = 0x0020
    }
} 