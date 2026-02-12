using System.Numerics;

namespace WoWMapConverter.Core.Formats.Classic;

/// <summary>
/// ADT v18 format (Classic through WotLK, monolithic file).
/// Supports MH2O liquid chunk (WotLK+).
/// </summary>
public class AdtV18
{
    public uint Version { get; private set; } = 18;
    public MhdrData? Header { get; private set; }
    public McnkV18[] Chunks { get; } = new McnkV18[256];
    public List<string> Textures { get; } = new();
    public List<string> M2Names { get; } = new();
    public List<string> WmoNames { get; } = new();
    public List<MddfEntry> Doodads { get; } = new();
    public List<ModfEntry> WorldModels { get; } = new();
    public Mh2o? Liquid { get; private set; }

    public static AdtV18 Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return Load(reader);
    }

    public static AdtV18 Load(BinaryReader reader)
    {
        var adt = new AdtV18();
        int chunkIndex = 0;

        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            var chunkId = reader.ReadUInt32();
            var chunkSize = reader.ReadUInt32();
            var nextPos = reader.BaseStream.Position + chunkSize;

            switch (chunkId)
            {
                case 0x4D564552: // MVER
                    adt.Version = reader.ReadUInt32();
                    if (adt.Version != 18)
                        throw new InvalidDataException($"Expected ADT v18, got v{adt.Version}");
                    break;

                case 0x4D484452: // MHDR
                    adt.Header = ParseMhdr(reader);
                    break;

                case 0x4D544558: // MTEX
                    adt.Textures.AddRange(ReadStringBlock(reader, chunkSize));
                    break;

                case 0x4D4D4458: // MMDX
                    adt.M2Names.AddRange(ReadStringBlock(reader, chunkSize));
                    break;

                case 0x4D574D4F: // MWMO
                    adt.WmoNames.AddRange(ReadStringBlock(reader, chunkSize));
                    break;

                case 0x4D444446: // MDDF
                    ParseMddf(reader, chunkSize, adt.Doodads);
                    break;

                case 0x4D4F4446: // MODF
                    ParseModf(reader, chunkSize, adt.WorldModels);
                    break;

                case 0x4D48324F: // MH2O (WotLK+)
                    adt.Liquid = ParseMh2o(reader, chunkSize);
                    break;

                case 0x4D434E4B: // MCNK
                    if (chunkIndex < 256)
                        adt.Chunks[chunkIndex++] = ParseMcnk(reader, chunkSize);
                    break;
            }

            reader.BaseStream.Position = nextPos;
        }

        return adt;
    }

    private static MhdrData ParseMhdr(BinaryReader reader)
    {
        return new MhdrData
        {
            Flags = reader.ReadUInt32(),
            McnkOffset = reader.ReadUInt32(),
            MtexOffset = reader.ReadUInt32(),
            MmdxOffset = reader.ReadUInt32(),
            MmidOffset = reader.ReadUInt32(),
            MwmoOffset = reader.ReadUInt32(),
            MwidOffset = reader.ReadUInt32(),
            MddfOffset = reader.ReadUInt32(),
            ModfOffset = reader.ReadUInt32(),
            MfboOffset = reader.ReadUInt32(),
            Mh2oOffset = reader.ReadUInt32(),
            MtxfOffset = reader.ReadUInt32()
        };
    }

    public struct MhdrData
    {
        public uint Flags, McnkOffset, MtexOffset, MmdxOffset, MmidOffset;
        public uint MwmoOffset, MwidOffset, MddfOffset, ModfOffset;
        public uint MfboOffset, Mh2oOffset, MtxfOffset;
    }

    private static McnkV18 ParseMcnk(BinaryReader reader, uint chunkSize)
    {
        var startPos = reader.BaseStream.Position;
        var endPos = startPos + chunkSize;

        var mcnk = new McnkV18
        {
            Flags = reader.ReadUInt32(),
            IndexX = reader.ReadUInt32(),
            IndexY = reader.ReadUInt32(),
            NumLayers = reader.ReadUInt32(),
            NumDoodadRefs = reader.ReadUInt32(),
            HolesHighRes = reader.ReadBytes(8),
            LayerOffset = reader.ReadUInt32(),
            RefOffset = reader.ReadUInt32(),
            AlphaOffset = reader.ReadUInt32(),
            AlphaSize = reader.ReadUInt32(),
            ShadowOffset = reader.ReadUInt32(),
            ShadowSize = reader.ReadUInt32(),
            AreaId = reader.ReadUInt32(),
            NumMapObjRefs = reader.ReadUInt32(),
            HolesLowRes = reader.ReadUInt16(),
            Unknown1 = reader.ReadUInt16(),
            LowQualityTextureMap = reader.ReadBytes(16),
            NoEffectDoodad = reader.ReadUInt64(),
            SoundEmittersOffset = reader.ReadUInt32(),
            NumSoundEmitters = reader.ReadUInt32(),
            LiquidOffset = reader.ReadUInt32(),
            LiquidSize = reader.ReadUInt32(),
            Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
            VertexColorOffset = reader.ReadUInt32(),
            VertexLightingOffset = reader.ReadUInt32()
        };

        // Parse sub-chunks
        while (reader.BaseStream.Position < endPos - 8)
        {
            var subChunkId = reader.ReadUInt32();
            var subChunkSize = reader.ReadUInt32();
            var nextSubPos = reader.BaseStream.Position + subChunkSize;

            switch (subChunkId)
            {
                case 0x4D435654: // MCVT
                    mcnk.Heights = new float[145];
                    for (int i = 0; i < 145; i++)
                        mcnk.Heights[i] = reader.ReadSingle();
                    break;

                case 0x4D434E52: // MCNR
                    mcnk.Normals = new sbyte[145 * 3];
                    for (int i = 0; i < 145 * 3; i++)
                        mcnk.Normals[i] = reader.ReadSByte();
                    break;

                case 0x4D434C59: // MCLY
                    var layerCount = subChunkSize / 16;
                    mcnk.Layers = new MclyEntry[layerCount];
                    for (int i = 0; i < layerCount; i++)
                    {
                        mcnk.Layers[i] = new MclyEntry
                        {
                            TextureId = reader.ReadUInt32(),
                            Flags = reader.ReadUInt32(),
                            AlphaOffset = reader.ReadUInt32(),
                            EffectId = reader.ReadInt32()
                        };
                    }
                    break;

                case 0x4D43414C: // MCAL
                    mcnk.AlphaData = reader.ReadBytes((int)subChunkSize);
                    break;

                case 0x4D435348: // MCSH
                    mcnk.ShadowMap = reader.ReadBytes((int)subChunkSize);
                    break;

                case 0x4D434356: // MCCV (WotLK+)
                    mcnk.VertexColors = new uint[145];
                    for (int i = 0; i < 145; i++)
                        mcnk.VertexColors[i] = reader.ReadUInt32();
                    break;

                case 0x4D434C51: // MCLQ (pre-WotLK liquid)
                    mcnk.LegacyLiquid = reader.ReadBytes((int)subChunkSize);
                    break;
            }

            reader.BaseStream.Position = nextSubPos;
        }

        return mcnk;
    }

    private static void ParseMddf(BinaryReader reader, uint size, List<MddfEntry> list)
    {
        var count = size / 36;
        for (int i = 0; i < count; i++)
        {
            list.Add(new MddfEntry
            {
                NameId = reader.ReadUInt32(),
                UniqueId = reader.ReadUInt32(),
                Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Rotation = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Scale = reader.ReadUInt16(),
                Flags = reader.ReadUInt16()
            });
        }
    }

    private static void ParseModf(BinaryReader reader, uint size, List<ModfEntry> list)
    {
        var count = size / 64;
        for (int i = 0; i < count; i++)
        {
            list.Add(new ModfEntry
            {
                NameId = reader.ReadUInt32(),
                UniqueId = reader.ReadUInt32(),
                Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Rotation = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                LowerBounds = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                UpperBounds = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Flags = reader.ReadUInt16(),
                DoodadSet = reader.ReadUInt16(),
                NameSet = reader.ReadUInt16(),
                Scale = reader.ReadUInt16()
            });
        }
    }

    private static Mh2o? ParseMh2o(BinaryReader reader, uint size)
    {
        if (size == 0) return null;

        var mh2o = new Mh2o();
        var basePos = reader.BaseStream.Position;

        // Read 256 chunk headers
        for (int i = 0; i < 256; i++)
        {
            mh2o.ChunkHeaders[i] = new Mh2oChunkHeader
            {
                OffsetInstances = reader.ReadUInt32(),
                LayerCount = reader.ReadUInt32(),
                OffsetAttributes = reader.ReadUInt32()
            };
        }

        // Parse instance data for each chunk
        for (int i = 0; i < 256; i++)
        {
            var header = mh2o.ChunkHeaders[i];
            if (header.LayerCount == 0) continue;

            reader.BaseStream.Position = basePos + header.OffsetInstances;
            mh2o.Instances[i] = new Mh2oInstance[header.LayerCount];

            for (int j = 0; j < header.LayerCount; j++)
            {
                mh2o.Instances[i][j] = new Mh2oInstance
                {
                    LiquidType = reader.ReadUInt16(),
                    LiquidObject = reader.ReadUInt16(),
                    MinHeight = reader.ReadSingle(),
                    MaxHeight = reader.ReadSingle(),
                    XOffset = reader.ReadByte(),
                    YOffset = reader.ReadByte(),
                    Width = reader.ReadByte(),
                    Height = reader.ReadByte(),
                    OffsetExistsBitmap = reader.ReadUInt32(),
                    OffsetVertexData = reader.ReadUInt32()
                };
            }
        }

        return mh2o;
    }

    private static List<string> ReadStringBlock(BinaryReader reader, uint size)
    {
        var result = new List<string>();
        var endPos = reader.BaseStream.Position + size;
        var sb = new System.Text.StringBuilder();

        while (reader.BaseStream.Position < endPos)
        {
            var b = reader.ReadByte();
            if (b == 0)
            {
                if (sb.Length > 0)
                {
                    result.Add(sb.ToString());
                    sb.Clear();
                }
            }
            else
            {
                sb.Append((char)b);
            }
        }

        return result;
    }
}

public class McnkV18
{
    public uint Flags, IndexX, IndexY, NumLayers, NumDoodadRefs;
    public byte[] HolesHighRes = Array.Empty<byte>();
    public uint LayerOffset, RefOffset, AlphaOffset, AlphaSize;
    public uint ShadowOffset, ShadowSize, AreaId, NumMapObjRefs;
    public ushort HolesLowRes, Unknown1;
    public byte[] LowQualityTextureMap = Array.Empty<byte>();
    public ulong NoEffectDoodad;
    public uint SoundEmittersOffset, NumSoundEmitters, LiquidOffset, LiquidSize;
    public Vector3 Position;
    public uint VertexColorOffset, VertexLightingOffset;

    public float[]? Heights;
    public sbyte[]? Normals;
    public MclyEntry[]? Layers;
    public byte[]? AlphaData;
    public byte[]? ShadowMap;
    public uint[]? VertexColors;
    public byte[]? LegacyLiquid;
}

public struct MclyEntry
{
    public uint TextureId, Flags, AlphaOffset;
    public int EffectId;
}

public struct MddfEntry
{
    public uint NameId, UniqueId;
    public Vector3 Position, Rotation;
    public ushort Scale, Flags;
}

public struct ModfEntry
{
    public uint NameId, UniqueId;
    public Vector3 Position, Rotation, LowerBounds, UpperBounds;
    public ushort Flags, DoodadSet, NameSet, Scale;
}

public class Mh2o
{
    public Mh2oChunkHeader[] ChunkHeaders { get; } = new Mh2oChunkHeader[256];
    public Mh2oInstance[][] Instances { get; } = new Mh2oInstance[256][];
}

public struct Mh2oChunkHeader
{
    public uint OffsetInstances, LayerCount, OffsetAttributes;
}

public struct Mh2oInstance
{
    public ushort LiquidType, LiquidObject;
    public float MinHeight, MaxHeight;
    public byte XOffset, YOffset, Width, Height;
    public uint OffsetExistsBitmap, OffsetVertexData;
}
