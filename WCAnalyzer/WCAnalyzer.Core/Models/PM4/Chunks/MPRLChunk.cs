using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// MPRL chunk - Contains position data for server-side terrain collision mesh and navigation.
    /// </summary>
    public class MPRLChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MPRL";

        /// <summary>
        /// Gets the metadata about this chunk
        /// </summary>
        public string Description => "Position Data - Contains vertex positions for server-side collision meshes";

        /// <summary>
        /// Gets the data entries.
        /// </summary>
        public List<ServerPositionData> Entries { get; private set; } = new List<ServerPositionData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MPRLChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MPRLChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Entries.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // The MPRL chunk contains entries with the following structure:
                // struct { 
                //   uint16_t _0x00; // Always 0 in version_??.u 
                //   int16_t _0x02;  // Always -1 in version_??.u 
                //   uint16_t _0x04; 
                //   uint16_t _0x06; 
                //   C3Vectori position; // 12 bytes (3 * 4 bytes)
                //   int16_t _0x14; 
                //   uint16_t _0x16; 
                // } mprl[];

                // Each entry is 24 bytes (0x18)
                int entrySize = 24;
                int entryCount = Data.Length / entrySize;
                
                for (int i = 0; i < entryCount; i++)
                {
                    var entry = new ServerPositionData
                    {
                        Index = i,
                        Value0x00 = reader.ReadUInt16(), // uint16_t _0x00
                        Value0x02 = reader.ReadInt16(),  // int16_t _0x02
                        Value0x04 = reader.ReadUInt16(), // uint16_t _0x04
                        Value0x06 = reader.ReadUInt16(), // uint16_t _0x06
                        
                        // C3Vectori position
                        PositionX = reader.ReadSingle(),
                        PositionY = reader.ReadSingle(),
                        PositionZ = reader.ReadSingle(),
                        
                        Value0x14 = reader.ReadInt16(),  // int16_t _0x14
                        Value0x16 = reader.ReadUInt16()  // uint16_t _0x16
                    };
                    
                    Entries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents a server-side position data entry.
        /// </summary>
        public class ServerPositionData
        {
            /// <summary>
            /// Gets or sets the sequential index of this entry in the chunk
            /// </summary>
            public int Index { get; set; }
            
            /// <summary>
            /// First value (uint16_t _0x00) - Always 0 in documented version
            /// </summary>
            public ushort Value0x00 { get; set; }
            
            /// <summary>
            /// Second value (int16_t _0x02) - Always -1 in documented version
            /// </summary>
            public short Value0x02 { get; set; }
            
            /// <summary>
            /// Third value (uint16_t _0x04)
            /// </summary>
            public ushort Value0x04 { get; set; }
            
            /// <summary>
            /// Fourth value (uint16_t _0x06)
            /// </summary>
            public ushort Value0x06 { get; set; }
            
            /// <summary>
            /// X component of the position vector
            /// </summary>
            public float PositionX { get; set; }
            
            /// <summary>
            /// Y component of the position vector
            /// </summary>
            public float PositionY { get; set; }
            
            /// <summary>
            /// Z component of the position vector
            /// </summary>
            public float PositionZ { get; set; }
            
            /// <summary>
            /// Value at offset 0x14 (int16_t)
            /// </summary>
            public short Value0x14 { get; set; }
            
            /// <summary>
            /// Value at offset 0x16 (uint16_t)
            /// </summary>
            public ushort Value0x16 { get; set; }
            
            /// <summary>
            /// Returns a string representation of this object.
            /// </summary>
            public override string ToString()
            {
                return $"Position: ({PositionX:F2}, {PositionY:F2}, {PositionZ:F2})";
            }
        }
    }
} 