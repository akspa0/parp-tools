using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Interfaces;
using ArcaneFileParser.Core.Structures;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOGP chunk - WMO Group Header
    /// Contains information about a WMO group and acts as a container for group chunks
    /// </summary>
    public class MOGP : ContainerChunkBase
    {
        public override string ChunkId => "MOGP";

        public uint GroupNameOffset { get; set; }
        public uint DescriptiveGroupNameOffset { get; set; }
        public uint Flags { get; set; }
        public CAaBox BoundingBox { get; set; }
        public uint PortalStart { get; set; } // 32-bit in v14, converted from 16-bit in v17
        public uint PortalCount { get; set; } // 32-bit in v14, converted from 16-bit in v17
        public ushort TransBatchCount { get; set; }
        public ushort IntBatchCount { get; set; }
        public ushort ExtBatchCount { get; set; }
        public ushort BatchTypeD { get; set; }
        public byte[] FogIndices { get; set; }
        public uint GroupLiquid { get; set; }
        public uint UniqueId { get; set; }
        public uint Flags2 { get; set; }
        public short ParentOrFirstChildSplitGroupIndex { get; set; }
        public short NextSplitChildGroupIndex { get; set; }

        // v14-specific batch data
        public List<SMOGxBatch> InteriorBatches { get; private set; }
        public List<SMOGxBatch> ExteriorBatches { get; private set; }

        private readonly uint version;

        public MOGP(uint version) : base()
        {
            this.version = version;
            BoundingBox = new CAaBox();
            FogIndices = new byte[4];
            ParentOrFirstChildSplitGroupIndex = -1;
            NextSplitChildGroupIndex = -1;
            InteriorBatches = new List<SMOGxBatch>();
            ExteriorBatches = new List<SMOGxBatch>();
        }

        public override void Read(BinaryReader reader, uint size)
        {
            long startPosition = reader.BaseStream.Position;

            GroupNameOffset = reader.ReadUInt32();
            DescriptiveGroupNameOffset = reader.ReadUInt32();
            Flags = reader.ReadUInt32();
            BoundingBox.Read(reader);

            // Handle version-specific portal data
            if (version == 14)
            {
                PortalStart = reader.ReadUInt32();
                PortalCount = reader.ReadUInt32();
            }
            else
            {
                PortalStart = reader.ReadUInt16();
                PortalCount = reader.ReadUInt16();
            }

            TransBatchCount = reader.ReadUInt16();
            IntBatchCount = reader.ReadUInt16();
            ExtBatchCount = reader.ReadUInt16();
            BatchTypeD = reader.ReadUInt16();
            FogIndices = reader.ReadBytes(4);
            GroupLiquid = reader.ReadUInt32();
            UniqueId = reader.ReadUInt32();

            // Read remaining data if we're not at the end
            if (reader.BaseStream.Position < startPosition + size)
            {
                Flags2 = reader.ReadUInt32();
                ParentOrFirstChildSplitGroupIndex = reader.ReadInt16();
                NextSplitChildGroupIndex = reader.ReadInt16();
            }

            // Handle v14-specific batch data
            if (version == 14)
            {
                // Read interior batches
                InteriorBatches.Clear();
                for (int i = 0; i < 4; i++) // v14 always has 4 interior batches
                {
                    var batch = new SMOGxBatch();
                    batch.Read(reader);
                    InteriorBatches.Add(batch);
                }

                // Read exterior batches
                ExteriorBatches.Clear();
                for (int i = 0; i < 4; i++) // v14 always has 4 exterior batches
                {
                    var batch = new SMOGxBatch();
                    batch.Read(reader);
                    ExteriorBatches.Add(batch);
                }
            }

            // Read sub-chunks until we reach the end of the MOGP chunk
            long endPosition = startPosition + size;
            while (reader.BaseStream.Position < endPosition)
            {
                IChunk chunk = ChunkFactory.CreateChunk(reader);
                if (chunk != null)
                {
                    SubChunks[chunk.ChunkId] = chunk;
                }
            }
        }

        public void ConvertToV17()
        {
            if (version != 14)
                return;

            // Convert 32-bit portal indices to 16-bit
            if (PortalStart > ushort.MaxValue || PortalCount > ushort.MaxValue)
            {
                throw new InvalidDataException("Portal indices too large for v17 format");
            }

            // Convert batch data from v14 to v17 format
            // In v17, this data is stored in the MOBA chunk instead
            var moba = new MOBA();
            
            // Convert interior batches
            foreach (var batch in InteriorBatches)
            {
                if (batch.Count > 0)
                {
                    moba.AddBatch(new MOBABatch
                    {
                        StartVertex = (uint)batch.StartVertex,
                        VertexCount = (uint)batch.Count,
                        MinIndex = (uint)batch.MinIndex,
                        MaxIndex = (uint)batch.MaxIndex,
                        MaterialId = (byte)batch.MaterialId,
                        Flags = 0, // Interior batch
                    });
                }
            }

            // Convert exterior batches
            foreach (var batch in ExteriorBatches)
            {
                if (batch.Count > 0)
                {
                    moba.AddBatch(new MOBABatch
                    {
                        StartVertex = (uint)batch.StartVertex,
                        VertexCount = (uint)batch.Count,
                        MinIndex = (uint)batch.MinIndex,
                        MaxIndex = (uint)batch.MaxIndex,
                        MaterialId = (byte)batch.MaterialId,
                        Flags = 1, // Exterior batch
                    });
                }
            }

            if (moba.Batches.Count > 0)
            {
                SubChunks["MOBA"] = moba;
            }

            // Clear v14 batch data
            InteriorBatches.Clear();
            ExteriorBatches.Clear();
        }
    }
} 