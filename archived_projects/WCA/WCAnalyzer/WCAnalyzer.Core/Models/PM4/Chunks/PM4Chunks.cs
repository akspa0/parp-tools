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
        public string Description => "Position Data - Contains vertex positions and special entries for server-side collision meshes";

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
                // The MPRL chunk contains alternating entries of two types:
                // 1. Position records with actual XYZ coordinates
                // 2. Special records with a value and Y coordinate
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
                    // Special records typically have NaN in X and a specific value pattern
                    bool isSpecialEntry = float.IsNaN(x);
                    
                    var entry = new ServerPositionData
                    {
                        Index = index++,
                        Value1 = x,
                        Value2 = y,
                        Value3 = z,
                        IsSpecialEntry = isSpecialEntry,
                        
                        // For position records, use the values directly
                        CoordinateX = isSpecialEntry ? 0 : x,
                        CoordinateY = y,
                        CoordinateZ = isSpecialEntry ? 0 : z,
                        
                        // For special records, interpret the Z value as a special value
                        SpecialValue = isSpecialEntry ? BitConverter.ToInt32(BitConverter.GetBytes(z), 0) : 0
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
            /// Indicates if this entry is a special record (rather than a position)
            /// </summary>
            public bool IsSpecialEntry { get; set; }
            
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
            /// Special value - only valid for special records
            /// </summary>
            public int SpecialValue { get; set; }
            
            /// <summary>
            /// Returns a string representation of this object.
            /// </summary>
            public override string ToString()
            {
                return IsSpecialEntry 
                    ? $"Special Record: Value=0x{SpecialValue:X8}, Y={CoordinateY:F2}"
                    : $"Position: ({CoordinateX:F2}, {CoordinateY:F2}, {CoordinateZ:F2})";
            }
        }
    }

    /// <summary>
    /// MSCN chunk - Contains normal vector data for meshes.
    /// </summary>
    public class MSCNChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSCN";

        /// <summary>
        /// Gets the normal vectors.
        /// </summary>
        public List<Vector3> Normals { get; private set; } = new List<Vector3>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSCNChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSCNChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Normals.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                // Each normal is 12 bytes (3 floats for X, Y, Z)
                while (ms.Position < ms.Length)
                {
                    float x = reader.ReadSingle();
                    float y = reader.ReadSingle();
                    float z = reader.ReadSingle();
                    Normals.Add(new Vector3(x, y, z));
                }
            }
        }
    }

    /// <summary>
    /// MSLK chunk - Contains links data between vertices.
    /// </summary>
    public class MSLKChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSLK";

        /// <summary>
        /// Gets the links data.
        /// </summary>
        public List<LinkData> Links { get; private set; } = new List<LinkData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSLKChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSLKChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Links.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    var link = new LinkData
                    {
                        SourceIndex = reader.ReadUInt32(),
                        TargetIndex = reader.ReadUInt32()
                    };
                    Links.Add(link);
                }
            }
        }

        /// <summary>
        /// Represents a link between two vertices.
        /// </summary>
        public class LinkData
        {
            /// <summary>
            /// Gets or sets the source vertex index.
            /// </summary>
            public uint SourceIndex { get; set; }

            /// <summary>
            /// Gets or sets the target vertex index.
            /// </summary>
            public uint TargetIndex { get; set; }
        }
    }

    /// <summary>
    /// MSVI chunk - Contains vertex information data.
    /// </summary>
    public class MSVIChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSVI";

        /// <summary>
        /// Gets the vertex information entries.
        /// </summary>
        public List<VertexInfo> VertexInfos { get; private set; } = new List<VertexInfo>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSVIChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSVIChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            VertexInfos.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    var info = new VertexInfo
                    {
                        Value1 = reader.ReadUInt32(),
                        Value2 = reader.ReadUInt32()
                    };
                    VertexInfos.Add(info);
                }
            }
        }

        /// <summary>
        /// Represents vertex information data.
        /// </summary>
        public class VertexInfo
        {
            /// <summary>
            /// Gets or sets the first value.
            /// </summary>
            public uint Value1 { get; set; }

            /// <summary>
            /// Gets or sets the second value.
            /// </summary>
            public uint Value2 { get; set; }
        }
    }

    /// <summary>
    /// MSVT chunk - Contains vertex data.
    /// </summary>
    public class MSVTChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSVT";

        /// <summary>
        /// Gets the vertex data entries.
        /// </summary>
        public List<VertexData> Vertices { get; private set; } = new List<VertexData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSVTChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSVTChunk(byte[] data) : base(data)
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
                while (ms.Position < ms.Length)
                {
                    var vertex = new VertexData
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle(),
                        Flag1 = reader.ReadUInt32(),
                        Flag2 = reader.ReadUInt32()
                    };
                    Vertices.Add(vertex);
                }
            }
        }

        /// <summary>
        /// Represents vertex data.
        /// </summary>
        public class VertexData
        {
            /// <summary>
            /// Gets or sets the X coordinate.
            /// </summary>
            public float X { get; set; }

            /// <summary>
            /// Gets or sets the Y coordinate.
            /// </summary>
            public float Y { get; set; }

            /// <summary>
            /// Gets or sets the Z coordinate.
            /// </summary>
            public float Z { get; set; }

            /// <summary>
            /// Gets or sets the first flag.
            /// </summary>
            public uint Flag1 { get; set; }

            /// <summary>
            /// Gets or sets the second flag.
            /// </summary>
            public uint Flag2 { get; set; }
        }
    }

    /// <summary>
    /// MSUR chunk - Contains surface data.
    /// </summary>
    public class MSURChunk : PM4Chunk
    {
        /// <summary>
        /// Gets the chunk signature.
        /// </summary>
        public override string Signature => "MSUR";

        /// <summary>
        /// Gets the surface data entries.
        /// </summary>
        public List<SurfaceData> Surfaces { get; private set; } = new List<SurfaceData>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSURChunk"/> class.
        /// </summary>
        /// <param name="data">The chunk data.</param>
        public MSURChunk(byte[] data) : base(data)
        {
        }

        /// <summary>
        /// Reads and parses the chunk data.
        /// </summary>
        protected override void ReadData()
        {
            Surfaces.Clear();

            using (var ms = new MemoryStream(Data))
            using (var reader = new BinaryReader(ms))
            {
                while (ms.Position < ms.Length)
                {
                    var surface = new SurfaceData
                    {
                        Index1 = reader.ReadUInt32(),
                        Index2 = reader.ReadUInt32(),
                        Index3 = reader.ReadUInt32(),
                        Flags = reader.ReadUInt32()
                    };
                    Surfaces.Add(surface);
                }
            }
        }

        /// <summary>
        /// Represents surface data.
        /// </summary>
        public class SurfaceData
        {
            /// <summary>
            /// Gets or sets the first vertex index.
            /// </summary>
            public uint Index1 { get; set; }

            /// <summary>
            /// Gets or sets the second vertex index.
            /// </summary>
            public uint Index2 { get; set; }

            /// <summary>
            /// Gets or sets the third vertex index.
            /// </summary>
            public uint Index3 { get; set; }

            /// <summary>
            /// Gets or sets the flags.
            /// </summary>
            public uint Flags { get; set; }
        }
    }
} 