using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Formats.PM4;

/// <summary>
/// Canonical PM4 decoder that preserves the richer rollback chunk contract in core.
/// </summary>
public static class Pm4Decoder
{
    public static Pm4FileStructure Decode(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        return Decode(br);
    }

    private static Pm4FileStructure Decode(BinaryReader br)
    {
        var links = new List<MslkChunk>();
        var pathVertices = new List<Vector3>();
        var pathIndices = new List<uint>();
        var meshVertices = new List<Vector3>();
        var meshIndices = new List<uint>();
        var surfaces = new List<MsurChunk>();
        var sceneNodes = new List<Vector3>();
        var positionRefs = new List<MprlChunk>();
        var graphEntries = new List<MprrChunk>();
        var chunkSizes = new Dictionary<string, uint>(StringComparer.Ordinal);
        var unparsedChunks = new List<string>();

        MshdChunk? header = null;
        uint version = 0;

        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            byte[] sigBytes = br.ReadBytes(4);
            Array.Reverse(sigBytes);
            string sig = Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = br.BaseStream.Position;

            chunkSizes[sig] = size;

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
                    ReadVectors(br, size, pathVertices);
                    break;

                case "MSPI":
                    ReadUints(br, size, pathIndices);
                    break;

                case "MSVT":
                    ReadVectors(br, size, meshVertices);
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
                    ReadMprl(br, size, positionRefs);
                    break;

                case "MPRR":
                    ReadMprr(br, size, graphEntries);
                    break;

                default:
                    unparsedChunks.Add($"{sig}:{size}");
                    break;
            }

            br.BaseStream.Position = dataStart + size;
        }

        return new Pm4FileStructure(
            version,
            header,
            links,
            pathIndices,
            pathVertices,
            meshIndices,
            meshVertices,
            surfaces,
            sceneNodes,
            positionRefs,
            graphEntries,
            chunkSizes,
            unparsedChunks);
    }

    private static MshdChunk ReadMshd(BinaryReader br)
    {
        return new MshdChunk(
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32(),
            br.ReadUInt32());
    }

    private static void ReadMslk(BinaryReader br, uint size, List<MslkChunk> target)
    {
        int count = (int)(size / 20);
        for (int i = 0; i < count; i++)
        {
            byte typeFlags = br.ReadByte();
            byte subtype = br.ReadByte();
            ushort padding = br.ReadUInt16();
            uint groupObjectId = br.ReadUInt32();
            int mspiFirstIndex = ReadSignedInt24(br);
            byte mspiIndexCount = br.ReadByte();
            uint linkId = br.ReadUInt32();
            ushort refIndex = br.ReadUInt16();
            ushort systemFlag = br.ReadUInt16();

            target.Add(new MslkChunk(
                typeFlags,
                subtype,
                padding,
                groupObjectId,
                mspiFirstIndex,
                mspiIndexCount,
                linkId,
                refIndex,
                systemFlag));
        }
    }

    private static void ReadMsur(BinaryReader br, uint size, List<MsurChunk> target)
    {
        int count = (int)(size / 32);
        for (int i = 0; i < count; i++)
        {
            byte groupKey = br.ReadByte();
            byte indexCount = br.ReadByte();
            byte attributeMask = br.ReadByte();
            byte padding = br.ReadByte();
            Vector3 normal = new(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            float height = br.ReadSingle();
            uint msviFirstIndex = br.ReadUInt32();
            uint mdosIndex = br.ReadUInt32();
            uint packedParams = br.ReadUInt32();

            target.Add(new MsurChunk(
                groupKey,
                indexCount,
                attributeMask,
                padding,
                normal,
                height,
                msviFirstIndex,
                mdosIndex,
                packedParams));
        }
    }

    private static void ReadMprl(BinaryReader br, uint size, List<MprlChunk> target)
    {
        int count = (int)(size / 24);
        for (int i = 0; i < count; i++)
        {
            ushort unk00 = br.ReadUInt16();
            short unk02 = br.ReadInt16();
            ushort unk04 = br.ReadUInt16();
            ushort unk06 = br.ReadUInt16();
            Vector3 position = new(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            short unk14 = br.ReadInt16();
            ushort unk16 = br.ReadUInt16();

            target.Add(new MprlChunk(
                unk00,
                unk02,
                unk04,
                unk06,
                position,
                unk14,
                unk16));
        }
    }

    private static void ReadMprr(BinaryReader br, uint size, List<MprrChunk> target)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
            target.Add(new MprrChunk(br.ReadUInt16(), br.ReadUInt16()));
    }

    private static void ReadVectors(BinaryReader br, uint size, List<Vector3> target)
    {
        int count = (int)(size / 12);
        for (int i = 0; i < count; i++)
            target.Add(new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()));
    }

    private static void ReadUints(BinaryReader br, uint size, List<uint> target)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
            target.Add(br.ReadUInt32());
    }

    private static int ReadSignedInt24(BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(3);
        int value = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16);
        if ((value & 0x800000) != 0)
            value |= unchecked((int)0xFF000000);

        return value;
    }
}