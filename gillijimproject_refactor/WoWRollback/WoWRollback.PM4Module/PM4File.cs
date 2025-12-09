using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Minimal PM4 file parser for extracting pathfinding data.
/// Based on PM4 specification and old_sources/WoWToolbox.Core.v2.
/// </summary>
public class PM4File
{
    public uint Version { get; private set; }
    public PM4Header Header { get; private set; } = new();
    public List<MslkEntry> LinkEntries { get; } = new();
    public List<Vector3> PathVertices { get; } = new();      // MSPV
    public List<uint> PathIndices { get; } = new();          // MSPI
    public List<Vector3> MeshVertices { get; } = new();      // MSVT
    public List<uint> MeshIndices { get; } = new();          // MSVI
    public List<MsurEntry> Surfaces { get; } = new();        // MSUR
    public List<Vector3> ExteriorVertices { get; } = new();  // MSCN

    public static PM4File FromFile(string path)
    {
        return new PM4File(File.ReadAllBytes(path));
    }

    public PM4File(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        Parse(br);
    }

    private void Parse(BinaryReader br)
    {
        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            var sigBytes = br.ReadBytes(4);
            Array.Reverse(sigBytes); // Chunk IDs are reversed on disk
            string sig = Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = br.BaseStream.Position;

            switch (sig)
            {
                case "MVER":
                    Version = br.ReadUInt32();
                    break;
                case "MSHD":
                    Header = ReadHeader(br);
                    break;
                case "MSLK":
                    ReadMslk(br, size);
                    break;
                case "MSPV":
                    ReadVectors(br, size, PathVertices);
                    break;
                case "MSPI":
                    ReadUints(br, size, PathIndices);
                    break;
                case "MSVT":
                    ReadVectors(br, size, MeshVertices);
                    break;
                case "MSVI":
                    ReadUints(br, size, MeshIndices);
                    break;
                case "MSUR":
                    ReadMsur(br, size);
                    break;
                case "MSCN":
                    ReadVectors(br, size, ExteriorVertices);
                    break;
                default:
                    // Skip unknown chunks
                    break;
            }

            br.BaseStream.Position = dataStart + size;
        }
    }

    private PM4Header ReadHeader(BinaryReader br)
    {
        return new PM4Header
        {
            Unk00 = br.ReadUInt32(),
            Unk04 = br.ReadUInt32(),
            Unk08 = br.ReadUInt32(),
            Unk0C = br.ReadUInt32(),
            Unk10 = br.ReadUInt32(),
            Unk14 = br.ReadUInt32(),
            Unk18 = br.ReadUInt32(),
            Unk1C = br.ReadUInt32()
        };
    }

    private void ReadMslk(BinaryReader br, uint size)
    {
        int count = (int)(size / 20);
        for (int i = 0; i < count; i++)
        {
            LinkEntries.Add(new MslkEntry
            {
                TypeFlags = br.ReadByte(),
                Subtype = br.ReadByte(),
                Padding = br.ReadUInt16(),
                GroupObjectId = br.ReadUInt32(),
                MspiFirstIndex = br.ReadInt32(),
                MspiIndexCount = br.ReadByte(),
                LinkIdBytes = br.ReadBytes(3),
                RefIndex = br.ReadUInt16(),
                SystemFlag = br.ReadUInt16()
            });
        }
    }

    private void ReadMsur(BinaryReader br, uint size)
    {
        int count = (int)(size / 32);
        for (int i = 0; i < count; i++)
        {
            Surfaces.Add(new MsurEntry
            {
                GroupKey = br.ReadByte(),
                IndexCount = br.ReadByte(),
                AttributeMask = br.ReadByte(),
                Padding = br.ReadByte(),
                NormalX = br.ReadSingle(),
                NormalY = br.ReadSingle(),
                NormalZ = br.ReadSingle(),
                Height = br.ReadSingle(),
                MsviFirstIndex = br.ReadUInt32(),
                MdosIndex = br.ReadUInt32(),
                PackedParams = br.ReadUInt32()
            });
        }
    }

    private void ReadVectors(BinaryReader br, uint size, List<Vector3> list)
    {
        int count = (int)(size / 12);
        for (int i = 0; i < count; i++)
        {
            list.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
        }
    }

    private void ReadUints(BinaryReader br, uint size, List<uint> list)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
        {
            list.Add(br.ReadUInt32());
        }
    }

    public override string ToString()
    {
        return $"PM4 v{Version}: {Surfaces.Count} surfaces, {MeshVertices.Count} verts, {ExteriorVertices.Count} MSCN verts, {LinkEntries.Count} links";
    }
}

public class PM4Header
{
    public uint Unk00, Unk04, Unk08, Unk0C, Unk10, Unk14, Unk18, Unk1C;
}

public class MslkEntry
{
    public byte TypeFlags { get; set; }
    public byte Subtype { get; set; }
    public ushort Padding { get; set; }
    public uint GroupObjectId { get; set; }
    public int MspiFirstIndex { get; set; }
    public byte MspiIndexCount { get; set; }
    public byte[] LinkIdBytes { get; set; } = new byte[3];
    public ushort RefIndex { get; set; }
    public ushort SystemFlag { get; set; }

    public uint LinkId => (uint)((LinkIdBytes[2] << 16) | (LinkIdBytes[1] << 8) | LinkIdBytes[0]);
    public bool HasGeometry => MspiFirstIndex >= 0;
}

public class MsurEntry
{
    public byte GroupKey { get; set; }         // 0 = M2 props (non-walkable)
    public byte IndexCount { get; set; }
    public byte AttributeMask { get; set; }    // bit7 = liquid?
    public byte Padding { get; set; }
    public float NormalX { get; set; }
    public float NormalY { get; set; }
    public float NormalZ { get; set; }
    public float Height { get; set; }
    public uint MsviFirstIndex { get; set; }
    public uint MdosIndex { get; set; }
    public uint PackedParams { get; set; }

    public Vector3 Normal => new(NormalX, NormalY, NormalZ);
    public bool IsM2Bucket => GroupKey == 0;
    public bool IsLiquidCandidate => (AttributeMask & 0x80) != 0;

    /// <summary>CK24 for grouping: (PackedParams & 0xFFFFFF00) >> 8</summary>
    public uint CK24 => (PackedParams & 0xFFFFFF00) >> 8;
}
