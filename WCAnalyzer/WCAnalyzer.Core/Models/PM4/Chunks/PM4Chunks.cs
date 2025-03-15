using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Warcraft.NET.Files.Interfaces;

namespace WCAnalyzer.Core.Models.PM4.Chunks
{
    /// <summary>
    /// Base class for PM4 chunks containing common functionality.
    /// </summary>
    public abstract class PM4Chunk : IIFFChunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public abstract string Signature { get; }

        /// <summary>
        /// Gets the raw binary data of the chunk.
        /// </summary>
        public byte[] Data { get; protected set; }

        /// <summary>
        /// Gets the size of the chunk data.
        /// </summary>
        public uint Size => (uint)(Data?.Length ?? 0);

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4Chunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        protected PM4Chunk(byte[] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            ReadData();
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected abstract void ReadData();

        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        /// <returns>The chunk's signature.</returns>
        public string GetSignature() => Signature;

        /// <summary>
        /// Loads the chunk data from a byte array.
        /// </summary>
        /// <param name="data">The binary data to load from.</param>
        public void LoadBinaryData(byte[] data)
        {
            Data = data ?? throw new ArgumentNullException(nameof(data));
            ReadData();
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        /// <param name="size">The size of the chunk data.</param>
        public void ReadData(BinaryReader reader, uint size)
        {
            if (reader == null)
                throw new ArgumentNullException(nameof(reader));

            Data = reader.ReadBytes((int)size);
            ReadData();
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public void WriteData(BinaryWriter writer)
        {
            if (writer == null)
                throw new ArgumentNullException(nameof(writer));

            writer.Write(Data);
        }
    }

    /// <summary>
    /// MSHD chunk - Contains shadow data.
    /// </summary>
    public class MSHDChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSHD";

        /// <summary>
        /// Gets the shadow data entries.
        /// </summary>
        public List<ShadowEntry> ShadowEntries { get; private set; } = new List<ShadowEntry>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSHDChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSHDChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            ShadowEntries.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each entry is typically a fixed size structure
                while (ms.Position < ms.Length)
                {
                    var entry = new ShadowEntry
                    {
                        Value1 = reader.ReadUInt32(),
                        Value2 = reader.ReadUInt32(),
                        Value3 = reader.ReadUInt32(),
                        Value4 = reader.ReadUInt32()
                    };
                    ShadowEntries.Add(entry);
                }
            }
        }

        /// <summary>
        /// Represents a shadow data entry in the MSHD chunk.
        /// </summary>
        public class ShadowEntry
        {
            /// <summary>
            /// Gets or sets the first value.
            /// </summary>
            public uint Value1 { get; set; }

            /// <summary>
            /// Gets or sets the second value.
            /// </summary>
            public uint Value2 { get; set; }

            /// <summary>
            /// Gets or sets the third value.
            /// </summary>
            public uint Value3 { get; set; }

            /// <summary>
            /// Gets or sets the fourth value.
            /// </summary>
            public uint Value4 { get; set; }
        }
    }

    /// <summary>
    /// MSPV chunk - Contains vertex positions.
    /// </summary>
    public class MSPVChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSPV";

        /// <summary>
        /// Gets the vertex positions.
        /// </summary>
        public List<Vector3> Vertices { get; private set; } = new List<Vector3>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSPVChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSPVChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Vertices.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each vertex is 12 bytes (3 floats for X, Y, Z)
                while (ms.Position < ms.Length)
                {
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    Vertices.Add(new Vector3(x, y, z));
                }
            }
        }
    }

    /// <summary>
    /// MSPI chunk - Contains vertex indices.
    /// </summary>
    public class MSPIChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSPI";

        /// <summary>
        /// Gets the vertex indices.
        /// </summary>
        public List<uint> Indices { get; private set; } = new List<uint>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSPIChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSPIChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Indices.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each index is a uint (4 bytes)
                while (ms.Position < ms.Length)
                {
                    Indices.Add(reader.ReadUInt32());
                }
            }
        }
    }

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
                // Based on the observed pattern, the MPRL chunk contains alternating entries
                // of two types: command records and position records.
                // Each entry is 12 bytes (3 floats or equivalent)

                int entryCount = Data.Length / 12;
                
                // Flag used to track the alternating pattern
                bool isEvenEntry = true;
                int index = 0;
                
                while (ms.Position < ms.Length)
                {
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    
                    // Determine entry type based on the values and pattern
                    // Control records typically have NaN in X and a specific flag pattern
                    bool isControlRecord = float.IsNaN(x);
                    
                    var entry = new ServerPositionData
                    {
                        Index = index++,
                        Value1 = x,
                        Value2 = y,
                        Value3 = z,
                        IsControlRecord = isControlRecord,
                        
                        // For position records, use the values directly
                        CoordinateX = isControlRecord ? 0 : x,
                        CoordinateY = y,
                        CoordinateZ = isControlRecord ? 0 : z,
                        
                        // For control records, interpret the Z value as a flag/command
                        CommandValue = isControlRecord ? BitConverter.ToInt32(BitConverter.GetBytes(z), 0) : 0
                    };
                    
                    Entries.Add(entry);
                    isEvenEntry = !isEvenEntry;
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
            /// First raw value
            /// </summary>
            public float Value1 { get; set; }
            
            /// <summary>
            /// Second raw value
            /// </summary>
            public float Value2 { get; set; }
            
            /// <summary>
            /// Third raw value
            /// </summary>
            public float Value3 { get; set; }
            
            /// <summary>
            /// Indicates if this entry is a control/command record (rather than a position)
            /// </summary>
            public bool IsControlRecord { get; set; }
            
            /// <summary>
            /// X coordinate - only valid for position records
            /// </summary>
            public float CoordinateX { get; set; }
            
            /// <summary>
            /// Y coordinate
            /// </summary>
            public float CoordinateY { get; set; }
            
            /// <summary>
            /// Z coordinate - only valid for position records
            /// </summary>
            public float CoordinateZ { get; set; }
            
            /// <summary>
            /// Command/flag value - only valid for control records
            /// </summary>
            public int CommandValue { get; set; }
            
            /// <summary>
            /// Returns a string representation of this object.
            /// </summary>
            public override string ToString()
            {
                return IsControlRecord 
                    ? $"Control Record: Command=0x{CommandValue:X8}, Y={CoordinateY:F2}"
                    : $"Position: ({CoordinateX:F2}, {CoordinateY:F2}, {CoordinateZ:F2})";
            }
        }
    }
} 