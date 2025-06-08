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
        public const int ExpectedSize = 32;
        private byte[] _headerData = new byte[ExpectedSize];
        public byte[] HeaderData
        {
            get => _headerData;
            set
            {
                if (value == null || value.Length != ExpectedSize)
                    throw new ArgumentException($"MSHD HeaderData must be exactly {ExpectedSize} bytes.");
                _headerData = value;
            }
        }

        // Explicit field accessors (uint32, little-endian)
        public uint Unknown_0x00
        {
            get => BitConverter.ToUInt32(_headerData, 0);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 0);
        }
        public uint Unknown_0x04
        {
            get => BitConverter.ToUInt32(_headerData, 4);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 4);
        }
        public uint Unknown_0x08
        {
            get => BitConverter.ToUInt32(_headerData, 8);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 8);
        }
        public uint Unknown_0x0C
        {
            get => BitConverter.ToUInt32(_headerData, 12);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 12);
        }
        public uint Unknown_0x10
        {
            get => BitConverter.ToUInt32(_headerData, 16);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 16);
        }
        public uint Unknown_0x14
        {
            get => BitConverter.ToUInt32(_headerData, 20);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 20);
        }
        public uint Unknown_0x18
        {
            get => BitConverter.ToUInt32(_headerData, 24);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 24);
        }
        public uint Unknown_0x1C
        {
            get => BitConverter.ToUInt32(_headerData, 28);
            set => BitConverter.GetBytes(value).CopyTo(_headerData, 28);
        }

        public void LoadBinaryData(byte[] inData)
        {
            if (inData == null || inData.Length != ExpectedSize)
                throw new ArgumentException($"MSHD chunk must be exactly {ExpectedSize} bytes.");
            inData.CopyTo(_headerData, 0);
        }

        public void Load(BinaryReader br)
        {
            var data = br.ReadBytes(ExpectedSize);
            if (data.Length != ExpectedSize)
                throw new InvalidDataException($"MSHD chunk read {data.Length} bytes, expected {ExpectedSize}.");
            data.CopyTo(_headerData, 0);
        }

        public byte[] Serialize(long offset = 0)
        {
            // Always return a copy to prevent external mutation
            return (byte[])_headerData.Clone();
        }

        public string GetSignature() => Signature;
        public uint GetSize() => ExpectedSize;

        public override string ToString()
        {
            return $"MSHD: [00={Unknown_0x00:X8}, 04={Unknown_0x04:X8}, 08={Unknown_0x08:X8}, 0C={Unknown_0x0C:X8}, " +
                   $"10={Unknown_0x10:X8}, 14={Unknown_0x14:X8}, 18={Unknown_0x18:X8}, 1C={Unknown_0x1C:X8}]";
        }
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