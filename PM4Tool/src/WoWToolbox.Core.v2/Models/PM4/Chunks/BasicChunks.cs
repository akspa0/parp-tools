using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using Warcraft.NET.Files.Interfaces;

namespace WoWToolbox.Core.v2.Models.PM4.Chunks
{
    #region Basic Data Types

    /// <summary>3D vector structure used by PM4 chunks</summary>
    public struct C3Vector
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public C3Vector(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static implicit operator Vector3(C3Vector vector) => new Vector3(vector.X, vector.Y, vector.Z);
        public static implicit operator C3Vector(Vector3 vector) => new C3Vector(vector.X, vector.Y, vector.Z);
    }

    /// <summary>MSVT vertex structure</summary>
    public struct MsvtVertex
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }
    }

    /// <summary>MPRL entry structure</summary>
    public struct MprlEntry
    {
        public C3Vector Position { get; set; }
        public uint Unknown { get; set; }
    }

    #endregion

    #region MVER Chunk

    /// <summary>Version chunk</summary>
    public class MVER : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MVER";
        public uint Version { get; set; }

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br) => Version = br.ReadUInt32();
        public byte[] Serialize(long offset = 0) => BitConverter.GetBytes(Version);
        public string GetSignature() => Signature;
        public uint GetSize() => sizeof(uint);
    }

    #endregion

    #region MSHD Chunk

    /// <summary>Header chunk</summary>
    public class MSHDChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSHD";
        public byte[] HeaderData { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => HeaderData = inData;
        public void Load(BinaryReader br) => HeaderData = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => HeaderData;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)HeaderData.Length;
    }

    #endregion

    #region MSPV Chunk

    /// <summary>Structural vertices chunk</summary>
    public class MSPVChunk : IIFFChunk, IBinarySerializable, IDisposable
    {
        public const string Signature = "MSPV";
        private List<C3Vector>? _vertices;

        public List<C3Vector> Vertices => _vertices ??= new List<C3Vector>();
        public int VertexCount => _vertices?.Count ?? 0;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            var count = (int)(br.BaseStream.Length - br.BaseStream.Position) / 12; // 3 floats
            _vertices = new List<C3Vector>(count);
            
            for (int i = 0; i < count; i++)
            {
                _vertices.Add(new C3Vector(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            if (_vertices == null) return Array.Empty<byte>();
            
            using var ms = new MemoryStream(_vertices.Count * 12);
            using var bw = new BinaryWriter(ms);
            
            foreach (var vertex in _vertices)
            {
                bw.Write(vertex.X);
                bw.Write(vertex.Y);
                bw.Write(vertex.Z);
            }
            
            return ms.ToArray();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => (uint)(VertexCount * 12);
        public void Dispose() { _vertices?.Clear(); _vertices = null; }
    }

    #endregion

    #region MSPI Chunk

    /// <summary>Structural vertex indices chunk</summary>
    public class MSPIChunk : IIFFChunk, IBinarySerializable, IDisposable
    {
        public const string Signature = "MSPI";
        private List<uint>? _indices;

        public List<uint> Indices => _indices ??= new List<uint>();
        public int IndexCount => _indices?.Count ?? 0;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            var count = (int)(br.BaseStream.Length - br.BaseStream.Position) / 4; // uint32
            _indices = new List<uint>(count);
            
            for (int i = 0; i < count; i++)
            {
                _indices.Add(br.ReadUInt32());
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            if (_indices == null) return Array.Empty<byte>();
            
            using var ms = new MemoryStream(_indices.Count * 4);
            using var bw = new BinaryWriter(ms);
            
            foreach (var index in _indices)
            {
                bw.Write(index);
            }
            
            return ms.ToArray();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => (uint)(IndexCount * 4);
        public void Dispose() { _indices?.Clear(); _indices = null; }
    }

    #endregion

    #region MSVT Chunk

    /// <summary>Render vertices chunk</summary>
    public class MSVTChunk : IIFFChunk, IBinarySerializable, IDisposable
    {
        public const string Signature = "MSVT";
        private List<MsvtVertex>? _vertices;

        public List<MsvtVertex> Vertices => _vertices ??= new List<MsvtVertex>();
        public int VertexCount => _vertices?.Count ?? 0;

        /// <summary>High-performance read-only access to vertices using spans</summary>
        public ReadOnlySpan<MsvtVertex> VerticesSpan => _vertices?.ToArray() ?? Array.Empty<MsvtVertex>();

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            var count = (int)(br.BaseStream.Length - br.BaseStream.Position) / 12; // 3 floats
            _vertices = new List<MsvtVertex>(count);
            
            for (int i = 0; i < count; i++)
            {
                _vertices.Add(new MsvtVertex 
                { 
                    X = br.ReadSingle(), 
                    Y = br.ReadSingle(), 
                    Z = br.ReadSingle() 
                });
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            if (_vertices == null) return Array.Empty<byte>();
            
            using var ms = new MemoryStream(_vertices.Count * 12);
            using var bw = new BinaryWriter(ms);
            
            foreach (var vertex in _vertices)
            {
                bw.Write(vertex.X);
                bw.Write(vertex.Y);
                bw.Write(vertex.Z);
            }
            
            return ms.ToArray();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => (uint)(VertexCount * 12);
        public void Dispose() { _vertices?.Clear(); _vertices = null; }
    }

    #endregion

    #region MSVI Chunk

    /// <summary>Render vertex indices chunk</summary>
    public class MSVIChunk : IIFFChunk, IBinarySerializable, IDisposable
    {
        public const string Signature = "MSVI";
        private List<uint>? _indices;

        public List<uint> Indices => _indices ??= new List<uint>();
        public int IndexCount => _indices?.Count ?? 0;

        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            var count = (int)(br.BaseStream.Length - br.BaseStream.Position) / 4; // uint32
            _indices = new List<uint>(count);
            
            for (int i = 0; i < count; i++)
            {
                _indices.Add(br.ReadUInt32());
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            if (_indices == null) return Array.Empty<byte>();
            
            using var ms = new MemoryStream(_indices.Count * 4);
            using var bw = new BinaryWriter(ms);
            
            foreach (var index in _indices)
            {
                bw.Write(index);
            }
            
            return ms.ToArray();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => (uint)(IndexCount * 4);
        public void Dispose() { _indices?.Clear(); _indices = null; }
    }

    #endregion

    #region Simple Chunks (Placeholder Implementations)

    /// <summary>Collision boundaries chunk (simplified)</summary>
    public class MSCNChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSCN";
        public byte[] Data { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => Data = inData;
        public void Load(BinaryReader br) => Data = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => Data;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)Data.Length;
    }

    /// <summary>Surface normals chunk (simplified)</summary>
    public class MSRNChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MSRN";
        public byte[] Data { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => Data = inData;
        public void Load(BinaryReader br) => Data = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => Data;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)Data.Length;
    }

    /// <summary>Position data chunk (simplified)</summary>
    public class MPRLChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MPRL";
        public byte[] Data { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => Data = inData;
        public void Load(BinaryReader br) => Data = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => Data;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)Data.Length;
    }

    /// <summary>Reference data chunk (simplified)</summary>
    public class MPRRChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MPRR";
        public byte[] Data { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => Data = inData;
        public void Load(BinaryReader br) => Data = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => Data;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)Data.Length;
    }

    /// <summary>Building header chunk (simplified)</summary>
    public class MDBHChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MDBH";
        public byte[] Data { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => Data = inData;
        public void Load(BinaryReader br) => Data = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => Data;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)Data.Length;
    }

    #endregion

    #region MDOS Chunk - Object Data (COMPLETE IMPLEMENTATION)

    /// <summary>
    /// Represents an entry in the MDOS chunk.
    /// Structure based on documentation at wowdev.wiki/PM4.md (MDOS section)
    /// </summary>
    public class MdosEntry
    {
        public uint m_destructible_building_index { get; set; }
        public uint destruction_state { get; set; }

        public const int Size = 8; // Bytes (uint32 + uint32)

        public void Load(BinaryReader br)
        {
            m_destructible_building_index = br.ReadUInt32();
            destruction_state = br.ReadUInt32();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(m_destructible_building_index);
            bw.Write(destruction_state);
        }

        public override string ToString()
        {
            return $"MDOS Entry [Index: {m_destructible_building_index}, State: {destruction_state}]";
        }
    }

    /// <summary>Object data chunk - CRITICAL for building extraction</summary>
    public class MDOSChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MDOS";
        public List<MdosEntry> Entries { get; private set; } = new List<MdosEntry>();

        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null) throw new ArgumentNullException(nameof(inData));

            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long endPosition = br.BaseStream.Length;
            long size = endPosition - startPosition;

            if (size < 0) throw new InvalidDataException("Stream size is negative.");
            if (size % MdosEntry.Size != 0)
            {
                Console.WriteLine($"Warning: MDOS chunk size {size} is not a multiple of entry size {MdosEntry.Size}. Possible padding or corruption.");
                size -= (size % MdosEntry.Size);
            }

            int entryCount = (int)(size / MdosEntry.Size);
            Entries = new List<MdosEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                var entry = new MdosEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var entry in Entries)
            {
                entry.Write(bw);
            }

            return ms.ToArray();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => (uint)(Entries.Count * MdosEntry.Size);
    }

    #endregion

    #region MDSF Chunk - Structure Data (COMPLETE IMPLEMENTATION)

    /// <summary>
    /// Represents an entry in the MDSF chunk.
    /// Structure based on documentation at wowdev.wiki/PM4.md (MDSF section)
    /// </summary>
    public class MdsfEntry
    {
        public uint msur_index { get; set; }
        public uint mdos_index { get; set; }

        public const int Size = 8; // Bytes (uint32 + uint32)

        public void Load(BinaryReader br)
        {
            msur_index = br.ReadUInt32();
            mdos_index = br.ReadUInt32();
        }

        public void Write(BinaryWriter bw)
        {
            bw.Write(msur_index);
            bw.Write(mdos_index);
        }

        public override string ToString()
        {
            return $"MDSF Entry [MSUR Index: {msur_index}, MDOS Index: {mdos_index}]";
        }
    }

    /// <summary>Structure data chunk - CRITICAL for building extraction</summary>
    public class MDSFChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MDSF";
        public List<MdsfEntry> Entries { get; private set; } = new List<MdsfEntry>();

        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null) throw new ArgumentNullException(nameof(inData));

            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long endPosition = br.BaseStream.Length;
            long size = endPosition - startPosition;

            if (size < 0) throw new InvalidDataException("Stream size is negative.");

            if (size % MdsfEntry.Size != 0)
            {
                Console.WriteLine($"Warning: MDSF chunk size {size} is not a multiple of the documented entry size {MdsfEntry.Size}. Possible padding or corruption.");
                size -= (size % MdsfEntry.Size);
            }

            int entryCount = (int)(size / MdsfEntry.Size);
            Entries = new List<MdsfEntry>(entryCount);

            for (int i = 0; i < entryCount; i++)
            {
                if (br.BaseStream.Position + MdsfEntry.Size > br.BaseStream.Length)
                {
                    Console.WriteLine($"Warning: MDSF chunk unexpected end of stream at entry {i}. Read {Entries.Count} entries out of expected {entryCount}.");
                    break;
                }
                var entry = new MdsfEntry();
                entry.Load(br);
                Entries.Add(entry);
            }
        }

        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var entry in Entries)
            {
                entry.Write(bw);
            }

            return ms.ToArray();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => (uint)(Entries.Count * MdsfEntry.Size);
    }

    #endregion
} 