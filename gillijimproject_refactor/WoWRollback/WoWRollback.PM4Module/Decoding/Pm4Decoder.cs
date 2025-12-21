using System.Text;
using System.Numerics;

namespace WoWRollback.PM4Module.Decoding;

public class Pm4Decoder
{
    public static Pm4FileStructure Decode(byte[] data)
    {
        var links = new List<MslkChunk>();
        var pathVerts = new List<Vector3>();
        var pathIndices = new List<uint>();
        var meshVerts = new List<Vector3>();
        var meshIndices = new List<uint>();
        var surfaces = new List<MsurChunk>();
        var sceneNodes = new List<Vector3>();
        var posRefs = new List<MprlChunk>();
        var mprrEntries = new List<MprrChunk>();
        MshdChunk? header = null;
        uint version = 0;

        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);

        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            var sigBytes = br.ReadBytes(4);
            Array.Reverse(sigBytes); // Format uses reversed CC on disk
            string sig = Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = br.BaseStream.Position;

            switch (sig)
            {
                case "MVER":
                    version = br.ReadUInt32();
                    break;

                case "MSHD":
                    header = ReadMshd(br);
                    break;

                case "MSLK":
                    ReadMslk(br, size, links);
                    break;

                case "MSPV":
                    ReadVectors(br, size, pathVerts);
                    break;

                case "MSPI":
                    ReadUints(br, size, pathIndices);
                    break;

                case "MSVT":
                    ReadVectors(br, size, meshVerts);
                    break;

                case "MSVI":
                    ReadUints(br, size, meshIndices);
                    break;

                case "MSUR":
                    ReadMsur(br, size, surfaces);
                    break;

                case "MSCN":
                    ReadVectors(br, size, sceneNodes);
                    break;

                case "MPRL":
                    ReadMprl(br, size, posRefs);
                    break;

                case "MPRR":
                    ReadMprr(br, size, mprrEntries);
                    break;
            }

            br.BaseStream.Position = dataStart + size;
        }

        return new Pm4FileStructure(
            version,
            header,
            links,
            pathIndices,
            pathVerts,
            meshIndices,
            meshVerts,
            surfaces,
            sceneNodes,
            posRefs,
            mprrEntries
        );
    }

    private static MshdChunk ReadMshd(BinaryReader br)
    {
        // MSHD is 32 bytes
        return new MshdChunk(
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32()
        );
    }

    private static void ReadMslk(BinaryReader br, uint size, List<MslkChunk> list)
    {
        int count = (int)(size / 20);
        for (int i = 0; i < count; i++)
        {
            byte type = br.ReadByte();
            byte subtype = br.ReadByte();
            ushort pad = br.ReadUInt16();
            uint groupId = br.ReadUInt32();
            
            // MSPI first index is 24-bit
            byte[] b = br.ReadBytes(3);
            int mspiFirst = b[0] | (b[1] << 8) | (b[2] << 16);
            if ((mspiFirst & 0x800000) != 0) mspiFirst |= unchecked((int)0xFF000000); // Sign extend

            byte mspiCount = br.ReadByte();
            uint linkId = br.ReadUInt32();
            ushort refIdx = br.ReadUInt16();
            ushort sysFlag = br.ReadUInt16();

            list.Add(new MslkChunk(type, subtype, pad, groupId, mspiFirst, mspiCount, linkId, refIdx, sysFlag));
        }
    }

    private static void ReadMsur(BinaryReader br, uint size, List<MsurChunk> list)
    {
        int count = (int)(size / 32);
        for (int i = 0; i < count; i++)
        {
            byte grpKey = br.ReadByte();
            byte idxCount = br.ReadByte();
            byte attr = br.ReadByte();
            byte pad = br.ReadByte();
            Vector3 normal = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            float height = br.ReadSingle();
            uint msviFirst = br.ReadUInt32();
            uint mdosIdx = br.ReadUInt32(); // Confirmed link to MSCN
            uint packedParams = br.ReadUInt32();

            list.Add(new MsurChunk(grpKey, idxCount, attr, pad, normal, height, msviFirst, mdosIdx, packedParams));
        }
    }

    private static void ReadMprl(BinaryReader br, uint size, List<MprlChunk> list)
    {
        int count = (int)(size / 24);
        for (int i = 0; i < count; i++)
        {
            // Skip index logic, just read fields
            ushort u00 = br.ReadUInt16();
            short u02 = br.ReadInt16();
            ushort rot = br.ReadUInt16(); // Field04 = Rotation
            ushort u06 = br.ReadUInt16();
            Vector3 pos = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            short floor = br.ReadInt16();
            ushort type = br.ReadUInt16();

            list.Add(new MprlChunk(u00, u02, rot, u06, pos, floor, type));
        }
    }

    private static void ReadMprr(BinaryReader br, uint size, List<MprrChunk> list)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
        {
            list.Add(new MprrChunk(br.ReadUInt16(), br.ReadUInt16()));
        }
    }

    private static void ReadVectors(BinaryReader br, uint size, List<Vector3> list)
    {
        int count = (int)(size / 12);
        for (int i = 0; i < count; i++)
        {
            list.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
        }
    }

    private static void ReadUints(BinaryReader br, uint size, List<uint> list)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
        {
            list.Add(br.ReadUInt32());
        }
    }
}
