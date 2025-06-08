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
        public byte[] HeaderData { get; set; } = Array.Empty<byte>();

        public void LoadBinaryData(byte[] inData) => HeaderData = inData;
        public void Load(BinaryReader br) => HeaderData = br.ReadBytes((int)(br.BaseStream.Length - br.BaseStream.Position));
        public byte[] Serialize(long offset = 0) => HeaderData;
        public string GetSignature() => Signature;
        public uint GetSize() => (uint)HeaderData.Length;
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