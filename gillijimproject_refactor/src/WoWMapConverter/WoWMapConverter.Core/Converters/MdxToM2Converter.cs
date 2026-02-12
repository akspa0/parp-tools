using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts Alpha MDX models to WotLK M2 format.
/// MDX is a WC3-like monolithic format, M2 uses chunked format with external .skin files.
/// </summary>
public class MdxToM2Converter
{
    private const uint M2_MAGIC = 0x3032444D; // "MD20"
    private const uint M2_VERSION = 264; // WotLK version

    /// <summary>
    /// Convert an MDX file to M2 format.
    /// </summary>
    public void Convert(string mdxPath, string m2OutputPath)
    {
        Console.WriteLine($"[INFO] Converting MDX → M2: {Path.GetFileName(mdxPath)}");

        using var mdxStream = File.OpenRead(mdxPath);
        using var reader = new BinaryReader(mdxStream);

        var mdxData = ParseMdx(reader);

        // Write M2
        using var m2Stream = File.Create(m2OutputPath);
        using var writer = new BinaryWriter(m2Stream);
        WriteM2(writer, mdxData);

        // Write .skin file
        var skinPath = Path.ChangeExtension(m2OutputPath, "00.skin");
        WriteSkin(skinPath, mdxData);

        Console.WriteLine($"[SUCCESS] Converted to M2: {m2OutputPath}");
    }

    /// <summary>
    /// Convert MDX bytes to M2 bytes in memory.
    /// </summary>
    public byte[] ConvertToBytes(byte[] mdxBytes)
    {
        using var mdxStream = new MemoryStream(mdxBytes);
        using var reader = new BinaryReader(mdxStream);
        using var m2Stream = new MemoryStream();
        using var writer = new BinaryWriter(m2Stream);

        var mdxData = ParseMdx(reader);
        WriteM2(writer, mdxData);

        return m2Stream.ToArray();
    }

    private MdxData ParseMdx(BinaryReader reader)
    {
        var data = new MdxData();

        // Check magic
        var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "MDLX")
            throw new InvalidDataException($"Invalid MDX magic: {magic}");

        // Parse chunks
        while (reader.BaseStream.Position < reader.BaseStream.Length - 8)
        {
            var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkId)
            {
                case "VERS":
                    data.Version = reader.ReadUInt32();
                    break;

                case "MODL":
                    data.Name = ReadFixedString(reader, 80);
                    data.AnimationFile = ReadFixedString(reader, 260);
                    data.BoundsRadius = reader.ReadSingle();
                    data.BoundsMin = ReadVector3(reader);
                    data.BoundsMax = ReadVector3(reader);
                    data.BlendTime = reader.ReadUInt32();
                    data.Flags = reader.ReadUInt32();
                    break;

                case "SEQS":
                    var seqCount = reader.ReadInt32();
                    data.Sequences = new MdxSequence[seqCount];
                    for (int i = 0; i < seqCount; i++)
                        data.Sequences[i] = ReadSequence(reader);
                    break;

                case "TEXS":
                    var texCount = (int)(chunkSize / 268);
                    data.Textures = new MdxTexture[texCount];
                    for (int i = 0; i < texCount; i++)
                        data.Textures[i] = ReadTexture(reader);
                    break;

                case "GEOS":
                    data.Geosets = ReadGeosets(reader, chunkEnd);
                    break;

                case "BONE":
                    var boneCount = reader.ReadInt32();
                    data.Bones = new MdxBone[boneCount];
                    for (int i = 0; i < boneCount; i++)
                        data.Bones[i] = ReadBone(reader);
                    break;

                case "PIVT":
                    var pivotCount = (int)(chunkSize / 12);
                    data.Pivots = new Vector3[pivotCount];
                    for (int i = 0; i < pivotCount; i++)
                        data.Pivots[i] = ReadVector3(reader);
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        return data;
    }

    private MdxSequence ReadSequence(BinaryReader reader)
    {
        return new MdxSequence
        {
            Name = ReadFixedString(reader, 80),
            IntervalStart = reader.ReadUInt32(),
            IntervalEnd = reader.ReadUInt32(),
            MoveSpeed = reader.ReadSingle(),
            Flags = reader.ReadUInt32(),
            Rarity = reader.ReadSingle(),
            SyncPoint = reader.ReadUInt32(),
            BoundsRadius = reader.ReadSingle(),
            BoundsMin = ReadVector3(reader),
            BoundsMax = ReadVector3(reader)
        };
    }

    private MdxTexture ReadTexture(BinaryReader reader)
    {
        return new MdxTexture
        {
            ReplaceableId = reader.ReadUInt32(),
            Filename = ReadFixedString(reader, 260),
            Flags = reader.ReadUInt32()
        };
    }

    private MdxGeoset[] ReadGeosets(BinaryReader reader, long chunkEnd)
    {
        var geosets = new List<MdxGeoset>();

        while (reader.BaseStream.Position < chunkEnd)
        {
            var geo = new MdxGeoset();
            var startPos = reader.BaseStream.Position;
            var size = reader.ReadUInt32();
            var geoEnd = startPos + size;

            // Parse sub-chunks
            while (reader.BaseStream.Position < geoEnd - 8)
            {
                var tag = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var count = reader.ReadInt32();

                switch (tag)
                {
                    case "VRTX":
                        geo.Vertices = new Vector3[count];
                        for (int i = 0; i < count; i++)
                            geo.Vertices[i] = ReadVector3(reader);
                        break;

                    case "NRMS":
                        geo.Normals = new Vector3[count];
                        for (int i = 0; i < count; i++)
                            geo.Normals[i] = ReadVector3(reader);
                        break;

                    case "UVAS":
                        var uvCount = geo.Vertices?.Length ?? count;
                        geo.UVs = new Vector2[uvCount];
                        for (int i = 0; i < uvCount; i++)
                            geo.UVs[i] = new Vector2(reader.ReadSingle(), reader.ReadSingle());
                        break;

                    case "PVTX":
                        geo.Indices = new ushort[count];
                        for (int i = 0; i < count; i++)
                            geo.Indices[i] = reader.ReadUInt16();
                        break;

                    case "GNDX":
                        geo.VertexGroups = reader.ReadBytes(count);
                        break;

                    default:
                        reader.BaseStream.Position += count * 4;
                        break;
                }
            }

            // Read remaining geoset data
            if (reader.BaseStream.Position < geoEnd)
            {
                geo.MaterialId = reader.ReadInt32();
                geo.SelectionGroup = reader.ReadInt32();
                geo.Unselectable = reader.ReadUInt32() == 1;
                geo.BoundsRadius = reader.ReadSingle();
                geo.BoundsMin = ReadVector3(reader);
                geo.BoundsMax = ReadVector3(reader);

                var seqBoundsCount = reader.ReadInt32();
                reader.BaseStream.Position += seqBoundsCount * 28;
            }

            reader.BaseStream.Position = geoEnd;
            geosets.Add(geo);
        }

        return geosets.ToArray();
    }

    private MdxBone ReadBone(BinaryReader reader)
    {
        var startPos = reader.BaseStream.Position;
        var size = reader.ReadUInt32();

        var bone = new MdxBone
        {
            Name = ReadFixedString(reader, 80),
            ObjectId = reader.ReadInt32(),
            ParentId = reader.ReadInt32(),
            Flags = reader.ReadUInt32()
        };

        reader.BaseStream.Position = startPos + size;
        return bone;
    }

    private void WriteM2(BinaryWriter writer, MdxData data)
    {
        // MD21 chunked format
        writer.Write(Encoding.ASCII.GetBytes("MD21"));
        var md21SizePos = writer.BaseStream.Position;
        writer.Write((uint)0);
        var md21DataStart = writer.BaseStream.Position;

        // Legacy M2 header
        writer.Write(M2_MAGIC);
        writer.Write(M2_VERSION);

        // Calculate data offset
        var dataOffset = 0x110;

        // Name
        var nameBytes = Encoding.UTF8.GetBytes(data.Name ?? "Converted");
        writer.Write((uint)nameBytes.Length + 1);
        writer.Write((uint)dataOffset);
        dataOffset += nameBytes.Length + 1;

        // Flags
        writer.Write(data.Flags);

        // Global sequences
        writer.Write((uint)0);
        writer.Write((uint)0);

        // Animations
        var animCount = data.Sequences?.Length ?? 0;
        writer.Write((uint)animCount);
        writer.Write((uint)(animCount > 0 ? dataOffset : 0));
        dataOffset += animCount * 64;

        // Animation lookups
        writer.Write((uint)animCount);
        writer.Write((uint)(animCount > 0 ? dataOffset : 0));
        dataOffset += animCount * 2;

        // Bones
        var boneCount = data.Bones?.Length ?? 0;
        writer.Write((uint)boneCount);
        writer.Write((uint)(boneCount > 0 ? dataOffset : 0));
        dataOffset += boneCount * 88;

        // Key bone lookups
        writer.Write((uint)0);
        writer.Write((uint)0);

        // Vertices
        var vertexCount = data.Geosets?.Sum(g => g.Vertices?.Length ?? 0) ?? 0;
        writer.Write((uint)vertexCount);
        writer.Write((uint)(vertexCount > 0 ? dataOffset : 0));
        dataOffset += vertexCount * 48;

        // Views (skins)
        writer.Write((uint)1);

        // Colors, textures, transparency, etc.
        for (int i = 0; i < 20; i++)
        {
            writer.Write((uint)0);
            writer.Write((uint)0);
        }

        // Bounding box
        WriteVector3(writer, data.BoundsMin);
        WriteVector3(writer, data.BoundsMax);
        writer.Write(data.BoundsRadius);
        WriteVector3(writer, data.BoundsMin);
        WriteVector3(writer, data.BoundsMax);
        writer.Write(data.BoundsRadius);

        // Pad to data offset
        var padding = dataOffset - (int)(writer.BaseStream.Position - md21DataStart);
        if (padding > 0)
            writer.Write(new byte[padding]);

        // Write name
        writer.Write(nameBytes);
        writer.Write((byte)0);

        // Write animations
        if (data.Sequences != null)
        {
            foreach (var seq in data.Sequences)
                WriteM2Animation(writer, seq);
            for (int i = 0; i < animCount; i++)
                writer.Write((ushort)i);
        }

        // Write bones
        if (data.Bones != null)
        {
            foreach (var bone in data.Bones)
                WriteM2Bone(writer, bone);
        }

        // Write vertices
        if (data.Geosets != null)
        {
            foreach (var geo in data.Geosets)
            {
                if (geo.Vertices == null) continue;
                for (int i = 0; i < geo.Vertices.Length; i++)
                    WriteM2Vertex(writer, geo, i);
            }
        }

        // Update MD21 size
        var md21End = writer.BaseStream.Position;
        writer.BaseStream.Position = md21SizePos;
        writer.Write((uint)(md21End - md21DataStart));
        writer.BaseStream.Position = md21End;

        // SFID chunk
        writer.Write(Encoding.ASCII.GetBytes("SFID"));
        writer.Write((uint)4);
        writer.Write((uint)0);
    }

    private void WriteM2Animation(BinaryWriter writer, MdxSequence seq)
    {
        writer.Write((ushort)0);
        writer.Write((ushort)0);
        writer.Write(seq.IntervalEnd - seq.IntervalStart);
        writer.Write(seq.MoveSpeed);
        writer.Write(seq.Flags);
        writer.Write((short)-1);
        writer.Write((ushort)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)0);
        WriteVector3(writer, seq.BoundsMin);
        WriteVector3(writer, seq.BoundsMax);
        writer.Write(seq.BoundsRadius);
        writer.Write((short)-1);
        writer.Write((ushort)0);
    }

    private void WriteM2Bone(BinaryWriter writer, MdxBone bone)
    {
        writer.Write(-1);
        writer.Write((uint)0);
        writer.Write((short)bone.ParentId);
        writer.Write((ushort)0);
        writer.Write((uint)0);
        writer.Write(new byte[48]); // Animation blocks
        writer.Write(0f); writer.Write(0f); writer.Write(0f); // Pivot
    }

    private void WriteM2Vertex(BinaryWriter writer, MdxGeoset geo, int index)
    {
        WriteVector3(writer, geo.Vertices[index]);
        writer.Write((byte)255); writer.Write((byte)0); writer.Write((byte)0); writer.Write((byte)0);
        writer.Write((byte)0); writer.Write((byte)0); writer.Write((byte)0); writer.Write((byte)0);
        
        var normal = geo.Normals != null && index < geo.Normals.Length 
            ? geo.Normals[index] : Vector3.UnitY;
        WriteVector3(writer, normal);

        var uv = geo.UVs != null && index < geo.UVs.Length 
            ? geo.UVs[index] : Vector2.Zero;
        writer.Write(uv.X);
        writer.Write(uv.Y);
        writer.Write(0f);
        writer.Write(0f);
    }

    private void WriteSkin(string path, MdxData data)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write(Encoding.ASCII.GetBytes("SKIN"));

        var totalIndices = data.Geosets?.Sum(g => g.Indices?.Length ?? 0) ?? 0;
        var submeshCount = data.Geosets?.Length ?? 0;

        writer.Write((uint)totalIndices);
        writer.Write((uint)0x10);
        writer.Write((uint)(totalIndices / 3));
        writer.Write((uint)(0x10 + totalIndices * 2));
        writer.Write((uint)0);
        writer.Write((uint)0);
        writer.Write((uint)submeshCount);
        writer.Write((uint)(0x10 + totalIndices * 4));
        writer.Write((uint)submeshCount);
        writer.Write((uint)(0x10 + totalIndices * 4 + submeshCount * 32));

        // Write indices
        if (data.Geosets != null)
        {
            foreach (var geo in data.Geosets)
                if (geo.Indices != null)
                    foreach (var idx in geo.Indices)
                        writer.Write(idx);

            foreach (var geo in data.Geosets)
                if (geo.Indices != null)
                    foreach (var idx in geo.Indices)
                        writer.Write(idx);
        }

        // Write submeshes
        var vertexOffset = 0;
        var indexOffset = 0;
        if (data.Geosets != null)
        {
            foreach (var geo in data.Geosets)
            {
                var vertCount = geo.Vertices?.Length ?? 0;
                var idxCount = geo.Indices?.Length ?? 0;

                writer.Write((ushort)0);
                writer.Write((ushort)vertexOffset);
                writer.Write((ushort)vertCount);
                writer.Write((ushort)indexOffset);
                writer.Write((ushort)(idxCount / 3));
                writer.Write((ushort)0);
                writer.Write((ushort)0);
                writer.Write((ushort)0);
                writer.Write((ushort)0);
                WriteVector3(writer, geo.BoundsMin);
                writer.Write(geo.BoundsRadius);

                vertexOffset += vertCount;
                indexOffset += idxCount / 3;
            }
        }

        // Write batches
        for (int i = 0; i < submeshCount; i++)
        {
            writer.Write((byte)0);
            writer.Write((byte)0);
            writer.Write((ushort)0);
            writer.Write((ushort)i);
            writer.Write((ushort)i);
            writer.Write((short)-1);
            writer.Write((ushort)0);
            writer.Write((ushort)0);
            writer.Write((ushort)0);
            writer.Write((ushort)0);
            writer.Write((ushort)0);
            writer.Write((short)-1);
            writer.Write((ushort)0);
        }
    }

    private string ReadFixedString(BinaryReader reader, int length)
    {
        var bytes = reader.ReadBytes(length);
        var end = Array.IndexOf(bytes, (byte)0);
        if (end < 0) end = length;
        return Encoding.UTF8.GetString(bytes, 0, end);
    }

    private Vector3 ReadVector3(BinaryReader reader)
        => new(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

    private void WriteVector3(BinaryWriter writer, Vector3 v)
    {
        writer.Write(v.X);
        writer.Write(v.Y);
        writer.Write(v.Z);
    }

    #region Data Structures

    public class MdxData
    {
        public uint Version, BlendTime, Flags;
        public string Name = "", AnimationFile = "";
        public float BoundsRadius;
        public Vector3 BoundsMin, BoundsMax;
        public MdxSequence[]? Sequences;
        public MdxTexture[]? Textures;
        public MdxGeoset[]? Geosets;
        public MdxBone[]? Bones;
        public Vector3[]? Pivots;
    }

    public struct MdxSequence
    {
        public string Name;
        public uint IntervalStart, IntervalEnd, Flags, SyncPoint;
        public float MoveSpeed, Rarity, BoundsRadius;
        public Vector3 BoundsMin, BoundsMax;
    }

    public struct MdxTexture
    {
        public uint ReplaceableId, Flags;
        public string Filename;
    }

    public class MdxGeoset
    {
        public Vector3[]? Vertices, Normals;
        public Vector2[]? UVs;
        public ushort[]? Indices;
        public byte[]? VertexGroups;
        public int MaterialId, SelectionGroup;
        public bool Unselectable;
        public float BoundsRadius;
        public Vector3 BoundsMin, BoundsMax;
    }

    public struct MdxBone
    {
        public string Name;
        public int ObjectId, ParentId;
        public uint Flags;
    }

    #endregion
}
