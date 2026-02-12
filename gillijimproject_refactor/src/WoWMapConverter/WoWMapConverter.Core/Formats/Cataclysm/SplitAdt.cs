using System.Numerics;
using WoWMapConverter.Core.Formats.Classic;

namespace WoWMapConverter.Core.Formats.Cataclysm;

/// <summary>
/// Split ADT format (Cataclysm+).
/// ADTs are split into: root.adt, _tex0.adt, _obj0.adt, and optionally _lod.adt (Legion+).
/// </summary>
public class SplitAdt
{
    public AdtRoot Root { get; private set; } = new();
    public AdtTex Tex { get; private set; } = new();
    public AdtObj Obj { get; private set; } = new();
    public AdtLod? Lod { get; private set; }

    /// <summary>
    /// Load a split ADT from the base path (e.g., "MapName_X_Y" without extension).
    /// </summary>
    public static SplitAdt Load(string basePath)
    {
        var adt = new SplitAdt();

        var rootPath = basePath + ".adt";
        var texPath = basePath + "_tex0.adt";
        var objPath = basePath + "_obj0.adt";
        var lodPath = basePath + "_lod.adt";

        if (File.Exists(rootPath))
            adt.Root = AdtRoot.Load(rootPath);

        if (File.Exists(texPath))
            adt.Tex = AdtTex.Load(texPath);

        if (File.Exists(objPath))
            adt.Obj = AdtObj.Load(objPath);

        if (File.Exists(lodPath))
            adt.Lod = AdtLod.Load(lodPath);

        return adt;
    }

    /// <summary>
    /// Load from separate streams (for CASC/MPQ extraction).
    /// </summary>
    public static SplitAdt Load(Stream rootStream, Stream texStream, Stream objStream, Stream? lodStream = null)
    {
        var adt = new SplitAdt();

        using var rootReader = new BinaryReader(rootStream);
        adt.Root = AdtRoot.Load(rootReader);

        using var texReader = new BinaryReader(texStream);
        adt.Tex = AdtTex.Load(texReader);

        using var objReader = new BinaryReader(objStream);
        adt.Obj = AdtObj.Load(objReader);

        if (lodStream != null)
        {
            using var lodReader = new BinaryReader(lodStream);
            adt.Lod = AdtLod.Load(lodReader);
        }

        return adt;
    }
}

/// <summary>
/// Root ADT file - contains terrain geometry and liquid data.
/// </summary>
public class AdtRoot
{
    public uint Version { get; private set; } = 18;
    public RootMcnk[] Chunks { get; } = new RootMcnk[256];
    public Mh2o? Liquid { get; private set; }

    public static AdtRoot Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return Load(reader);
    }

    public static AdtRoot Load(BinaryReader reader)
    {
        var adt = new AdtRoot();
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
                    break;

                case 0x4D48324F: // MH2O
                    adt.Liquid = ParseMh2o(reader, chunkSize);
                    break;

                case 0x4D434E4B: // MCNK
                    if (chunkIndex < 256)
                        adt.Chunks[chunkIndex++] = ParseRootMcnk(reader, chunkSize);
                    break;
            }

            reader.BaseStream.Position = nextPos;
        }

        return adt;
    }

    private static RootMcnk ParseRootMcnk(BinaryReader reader, uint chunkSize)
    {
        var startPos = reader.BaseStream.Position;
        var endPos = startPos + chunkSize;

        var mcnk = new RootMcnk
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

                case 0x4D434356: // MCCV
                    mcnk.VertexColors = new uint[145];
                    for (int i = 0; i < 145; i++)
                        mcnk.VertexColors[i] = reader.ReadUInt32();
                    break;

                case 0x4D434242: // MCBB (MoP+)
                    var bbCount = subChunkSize / 20;
                    mcnk.BlendBatches = new BlendBatch[bbCount];
                    for (int i = 0; i < bbCount; i++)
                    {
                        mcnk.BlendBatches[i] = new BlendBatch
                        {
                            MbmhIndex = reader.ReadUInt32(),
                            IndexCount = reader.ReadUInt32(),
                            IndexFirst = reader.ReadUInt32(),
                            VertexCount = reader.ReadUInt32(),
                            VertexFirst = reader.ReadUInt32()
                        };
                    }
                    break;
            }

            reader.BaseStream.Position = nextSubPos;
        }

        return mcnk;
    }

    private static Mh2o? ParseMh2o(BinaryReader reader, uint size)
    {
        // Same as v18 MH2O parsing
        if (size == 0) return null;
        var mh2o = new Mh2o();
        var basePos = reader.BaseStream.Position;

        for (int i = 0; i < 256; i++)
        {
            mh2o.ChunkHeaders[i] = new Mh2oChunkHeader
            {
                OffsetInstances = reader.ReadUInt32(),
                LayerCount = reader.ReadUInt32(),
                OffsetAttributes = reader.ReadUInt32()
            };
        }

        return mh2o;
    }
}

public class RootMcnk
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
    public uint[]? VertexColors;
    public BlendBatch[]? BlendBatches;
}

public struct BlendBatch
{
    public uint MbmhIndex, IndexCount, IndexFirst, VertexCount, VertexFirst;
}

/// <summary>
/// Texture ADT file (_tex0.adt) - contains texture layers and alpha maps.
/// </summary>
public class AdtTex
{
    public uint Version { get; private set; } = 18;
    public List<string> Textures { get; } = new();
    public uint[]? DiffuseTextureIds { get; private set; }
    public uint[]? HeightTextureIds { get; private set; }
    public TexMcnk[] Chunks { get; } = new TexMcnk[256];

    public static AdtTex Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return Load(reader);
    }

    public static AdtTex Load(BinaryReader reader)
    {
        var adt = new AdtTex();
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
                    break;

                case 0x4D544558: // MTEX
                    adt.Textures.AddRange(ReadStringBlock(reader, chunkSize));
                    break;

                case 0x4D444944: // MDID (FileDataIDs for diffuse textures)
                    adt.DiffuseTextureIds = new uint[chunkSize / 4];
                    for (int i = 0; i < adt.DiffuseTextureIds.Length; i++)
                        adt.DiffuseTextureIds[i] = reader.ReadUInt32();
                    break;

                case 0x4D484944: // MHID (FileDataIDs for height textures)
                    adt.HeightTextureIds = new uint[chunkSize / 4];
                    for (int i = 0; i < adt.HeightTextureIds.Length; i++)
                        adt.HeightTextureIds[i] = reader.ReadUInt32();
                    break;

                case 0x4D434E4B: // MCNK (no header in tex file)
                    if (chunkIndex < 256)
                        adt.Chunks[chunkIndex++] = ParseTexMcnk(reader, chunkSize);
                    break;
            }

            reader.BaseStream.Position = nextPos;
        }

        return adt;
    }

    private static TexMcnk ParseTexMcnk(BinaryReader reader, uint chunkSize)
    {
        var mcnk = new TexMcnk();
        var endPos = reader.BaseStream.Position + chunkSize;

        while (reader.BaseStream.Position < endPos - 8)
        {
            var subChunkId = reader.ReadUInt32();
            var subChunkSize = reader.ReadUInt32();
            var nextSubPos = reader.BaseStream.Position + subChunkSize;

            switch (subChunkId)
            {
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
            }

            reader.BaseStream.Position = nextSubPos;
        }

        return mcnk;
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

public class TexMcnk
{
    public MclyEntry[]? Layers;
    public byte[]? AlphaData;
}

/// <summary>
/// Object ADT file (_obj0.adt) - contains model and WMO placements.
/// </summary>
public class AdtObj
{
    public uint Version { get; private set; } = 18;
    public List<string> M2Names { get; } = new();
    public List<string> WmoNames { get; } = new();
    public uint[]? M2Offsets { get; private set; }
    public uint[]? WmoOffsets { get; private set; }
    public List<MddfEntry> Doodads { get; } = new();
    public List<ModfEntry> WorldModels { get; } = new();

    public static AdtObj Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return Load(reader);
    }

    public static AdtObj Load(BinaryReader reader)
    {
        var adt = new AdtObj();

        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            var chunkId = reader.ReadUInt32();
            var chunkSize = reader.ReadUInt32();
            var nextPos = reader.BaseStream.Position + chunkSize;

            switch (chunkId)
            {
                case 0x4D564552: // MVER
                    adt.Version = reader.ReadUInt32();
                    break;

                case 0x4D4D4458: // MMDX
                    adt.M2Names.AddRange(ReadStringBlock(reader, chunkSize));
                    break;

                case 0x4D4D4944: // MMID
                    adt.M2Offsets = new uint[chunkSize / 4];
                    for (int i = 0; i < adt.M2Offsets.Length; i++)
                        adt.M2Offsets[i] = reader.ReadUInt32();
                    break;

                case 0x4D574D4F: // MWMO
                    adt.WmoNames.AddRange(ReadStringBlock(reader, chunkSize));
                    break;

                case 0x4D574944: // MWID
                    adt.WmoOffsets = new uint[chunkSize / 4];
                    for (int i = 0; i < adt.WmoOffsets.Length; i++)
                        adt.WmoOffsets[i] = reader.ReadUInt32();
                    break;

                case 0x4D444446: // MDDF
                    ParseMddf(reader, chunkSize, adt.Doodads);
                    break;

                case 0x4D4F4446: // MODF
                    ParseModf(reader, chunkSize, adt.WorldModels);
                    break;
            }

            reader.BaseStream.Position = nextPos;
        }

        return adt;
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

/// <summary>
/// LOD ADT file (_lod.adt) - Legion+ low-detail terrain for distant rendering.
/// </summary>
public class AdtLod
{
    public uint Version { get; private set; } = 18;
    // LOD-specific chunks (MLHD, MLVH, MLVI, MLLL, etc.)

    public static AdtLod Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return Load(reader);
    }

    public static AdtLod Load(BinaryReader reader)
    {
        var adt = new AdtLod();

        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            var chunkId = reader.ReadUInt32();
            var chunkSize = reader.ReadUInt32();
            var nextPos = reader.BaseStream.Position + chunkSize;

            switch (chunkId)
            {
                case 0x4D564552: // MVER
                    adt.Version = reader.ReadUInt32();
                    break;

                // TODO: Parse MLHD, MLVH, MLVI, MLLL, MLND, MLSI, etc.
            }

            reader.BaseStream.Position = nextPos;
        }

        return adt;
    }
}
