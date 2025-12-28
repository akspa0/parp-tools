using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts M2 (WotLK) models to MDX (Alpha) format.
/// M2 is chunked format with external .skin files, MDX is WC3-like monolithic.
/// </summary>
public class M2ToMdxConverter
{
    /// <summary>
    /// Convert an M2 file to MDX format.
    /// </summary>
    public void Convert(string m2Path, string mdxOutputPath)
    {
        Console.WriteLine($"[INFO] Converting M2 → MDX: {Path.GetFileName(m2Path)}");

        using var m2Stream = File.OpenRead(m2Path);
        using var reader = new BinaryReader(m2Stream);

        var m2Data = ParseM2(reader);

        // Try to load .skin file
        var skinPath = Path.ChangeExtension(m2Path, "00.skin");
        if (File.Exists(skinPath))
        {
            using var skinStream = File.OpenRead(skinPath);
            using var skinReader = new BinaryReader(skinStream);
            ParseSkin(skinReader, m2Data);
        }

        // Write MDX
        using var mdxStream = File.Create(mdxOutputPath);
        using var writer = new BinaryWriter(mdxStream);
        WriteMdx(writer, m2Data);

        Console.WriteLine($"[SUCCESS] Converted to MDX: {mdxOutputPath}");
    }

    /// <summary>
    /// Convert M2 bytes to MDX bytes in memory.
    /// </summary>
    public byte[] ConvertToBytes(byte[] m2Bytes, byte[]? skinBytes = null)
    {
        using var m2Stream = new MemoryStream(m2Bytes);
        using var reader = new BinaryReader(m2Stream);
        
        var m2Data = ParseM2(reader);

        if (skinBytes != null)
        {
            using var skinStream = new MemoryStream(skinBytes);
            using var skinReader = new BinaryReader(skinStream);
            ParseSkin(skinReader, m2Data);
        }

        using var mdxStream = new MemoryStream();
        using var writer = new BinaryWriter(mdxStream);
        WriteMdx(writer, m2Data);

        return mdxStream.ToArray();
    }

    private M2Data ParseM2(BinaryReader reader)
    {
        var data = new M2Data();

        // Check magic
        var magic = reader.ReadUInt32();
        if (magic != 0x3032444D) // "MD20"
            throw new InvalidDataException($"Invalid M2 magic: 0x{magic:X8}");

        data.Version = reader.ReadUInt32();

        // Name
        var nameLen = reader.ReadUInt32();
        var nameOfs = reader.ReadUInt32();
        if (nameLen > 0 && nameOfs > 0)
        {
            long pos = reader.BaseStream.Position;
            reader.BaseStream.Position = nameOfs;
            data.Name = ReadNullTermString(reader, (int)nameLen);
            reader.BaseStream.Position = pos;
        }

        data.GlobalFlags = reader.ReadUInt32();

        // Global sequences
        var globalSeqCount = reader.ReadUInt32();
        var globalSeqOfs = reader.ReadUInt32();

        // Animations
        var animCount = reader.ReadUInt32();
        var animOfs = reader.ReadUInt32();

        // Animation lookups
        var animLookupCount = reader.ReadUInt32();
        var animLookupOfs = reader.ReadUInt32();

        // Bones
        var boneCount = reader.ReadUInt32();
        var boneOfs = reader.ReadUInt32();

        // Key bone lookups
        var keyBoneLookupCount = reader.ReadUInt32();
        var keyBoneLookupOfs = reader.ReadUInt32();

        // Vertices
        var vertexCount = reader.ReadUInt32();
        var vertexOfs = reader.ReadUInt32();

        // Views (skins)
        data.ViewCount = reader.ReadUInt32();

        // Colors
        var colorCount = reader.ReadUInt32();
        var colorOfs = reader.ReadUInt32();

        // Textures
        var textureCount = reader.ReadUInt32();
        var textureOfs = reader.ReadUInt32();

        // Transparency
        var transparencyCount = reader.ReadUInt32();
        var transparencyOfs = reader.ReadUInt32();

        // Texture animations
        var texAnimCount = reader.ReadUInt32();
        var texAnimOfs = reader.ReadUInt32();

        // Texture replacements
        var texReplaceCount = reader.ReadUInt32();
        var texReplaceOfs = reader.ReadUInt32();

        // Render flags
        var renderFlagCount = reader.ReadUInt32();
        var renderFlagOfs = reader.ReadUInt32();

        // Bone lookups
        var boneLookupCount = reader.ReadUInt32();
        var boneLookupOfs = reader.ReadUInt32();

        // Texture lookups
        var texLookupCount = reader.ReadUInt32();
        var texLookupOfs = reader.ReadUInt32();

        // Texture unit lookups
        var texUnitLookupCount = reader.ReadUInt32();
        var texUnitLookupOfs = reader.ReadUInt32();

        // Transparency lookups
        var transLookupCount = reader.ReadUInt32();
        var transLookupOfs = reader.ReadUInt32();

        // Texture anim lookups
        var texAnimLookupCount = reader.ReadUInt32();
        var texAnimLookupOfs = reader.ReadUInt32();

        // Bounding box
        data.BoundingBox = new Vector3[2];
        data.BoundingBox[0] = ReadVector3(reader);
        data.BoundingBox[1] = ReadVector3(reader);
        data.BoundingSphereRadius = reader.ReadSingle();

        // Collision box
        data.CollisionBox = new Vector3[2];
        data.CollisionBox[0] = ReadVector3(reader);
        data.CollisionBox[1] = ReadVector3(reader);
        data.CollisionSphereRadius = reader.ReadSingle();

        // Read actual data arrays
        if (vertexCount > 0 && vertexOfs > 0)
        {
            reader.BaseStream.Position = vertexOfs;
            data.Vertices = new M2Vertex[vertexCount];
            for (int i = 0; i < vertexCount; i++)
            {
                data.Vertices[i] = new M2Vertex
                {
                    Position = ReadVector3(reader),
                    BoneWeights = reader.ReadBytes(4),
                    BoneIndices = reader.ReadBytes(4),
                    Normal = ReadVector3(reader),
                    TexCoords = new Vector2[2]
                };
                data.Vertices[i].TexCoords[0] = ReadVector2(reader);
                data.Vertices[i].TexCoords[1] = ReadVector2(reader);
            }
        }

        if (textureCount > 0 && textureOfs > 0)
        {
            reader.BaseStream.Position = textureOfs;
            data.Textures = new M2Texture[textureCount];
            for (int i = 0; i < textureCount; i++)
            {
                data.Textures[i] = new M2Texture
                {
                    Type = reader.ReadUInt32(),
                    Flags = reader.ReadUInt32()
                };
                var texNameLen = reader.ReadUInt32();
                var texNameOfs = reader.ReadUInt32();
                if (texNameLen > 0 && texNameOfs > 0)
                {
                    long pos = reader.BaseStream.Position;
                    reader.BaseStream.Position = texNameOfs;
                    data.Textures[i].Filename = ReadNullTermString(reader, (int)texNameLen);
                    reader.BaseStream.Position = pos;
                }
            }
        }

        if (animCount > 0 && animOfs > 0)
        {
            reader.BaseStream.Position = animOfs;
            data.Animations = new M2Animation[animCount];
            for (int i = 0; i < animCount; i++)
            {
                data.Animations[i] = new M2Animation
                {
                    AnimationId = reader.ReadUInt16(),
                    SubAnimationId = reader.ReadUInt16(),
                    Length = reader.ReadUInt32(),
                    MoveSpeed = reader.ReadSingle(),
                    Flags = reader.ReadUInt32(),
                    Probability = reader.ReadInt16(),
                    Padding = reader.ReadUInt16(),
                    MinExtent = ReadVector3(reader),
                    MaxExtent = ReadVector3(reader),
                    BoundRadius = reader.ReadSingle(),
                    NextAnimation = reader.ReadInt16(),
                    AliasNext = reader.ReadUInt16()
                };
                // Skip replay fields for WotLK
                reader.ReadUInt32(); // d1
                reader.ReadUInt32(); // d2
            }
        }

        if (boneCount > 0 && boneOfs > 0)
        {
            reader.BaseStream.Position = boneOfs;
            data.Bones = new M2Bone[boneCount];
            for (int i = 0; i < boneCount; i++)
            {
                data.Bones[i] = new M2Bone
                {
                    KeyBoneId = reader.ReadInt32(),
                    Flags = reader.ReadUInt32(),
                    ParentBone = reader.ReadInt16(),
                    SubmeshId = reader.ReadUInt16()
                };
                reader.ReadUInt32(); // uDistToFurthDesc
                reader.ReadUInt32(); // uZRatioOfChain
                // Skip animation blocks for now
                reader.BaseStream.Position += 40; // translation
                reader.BaseStream.Position += 40; // rotation
                reader.BaseStream.Position += 40; // scaling
                data.Bones[i].Pivot = ReadVector3(reader);
            }
        }

        return data;
    }

    private void ParseSkin(BinaryReader reader, M2Data data)
    {
        // SKIN magic
        var magic = reader.ReadUInt32();
        if (magic != 0x4E494B53) // "SKIN"
            return;

        var indexCount = reader.ReadUInt32();
        var indexOfs = reader.ReadUInt32();
        var triangleCount = reader.ReadUInt32();
        var triangleOfs = reader.ReadUInt32();
        var propCount = reader.ReadUInt32();
        var propOfs = reader.ReadUInt32();
        var submeshCount = reader.ReadUInt32();
        var submeshOfs = reader.ReadUInt32();
        var texUnitCount = reader.ReadUInt32();
        var texUnitOfs = reader.ReadUInt32();
        data.SkinBoneCount = reader.ReadUInt32();

        // Read indices
        if (indexCount > 0 && indexOfs > 0)
        {
            reader.BaseStream.Position = indexOfs;
            data.SkinIndices = new ushort[indexCount];
            for (int i = 0; i < indexCount; i++)
                data.SkinIndices[i] = reader.ReadUInt16();
        }

        // Read triangles
        if (triangleCount > 0 && triangleOfs > 0)
        {
            reader.BaseStream.Position = triangleOfs;
            data.SkinTriangles = new ushort[triangleCount];
            for (int i = 0; i < triangleCount; i++)
                data.SkinTriangles[i] = reader.ReadUInt16();
        }

        // Read submeshes
        if (submeshCount > 0 && submeshOfs > 0)
        {
            reader.BaseStream.Position = submeshOfs;
            data.Submeshes = new M2Submesh[submeshCount];
            for (int i = 0; i < submeshCount; i++)
            {
                data.Submeshes[i] = new M2Submesh
                {
                    Id = reader.ReadUInt16(),
                    Level = reader.ReadUInt16(),
                    StartVertex = reader.ReadUInt16(),
                    VertexCount = reader.ReadUInt16(),
                    StartTriangle = reader.ReadUInt16(),
                    TriangleCount = reader.ReadUInt16(),
                    BoneCount = reader.ReadUInt16(),
                    StartBone = reader.ReadUInt16(),
                    BoneInfluences = reader.ReadUInt16(),
                    RootBone = reader.ReadUInt16(),
                    CenterMass = ReadVector3(reader),
                    CenterBoundingBox = ReadVector3(reader),
                    Radius = reader.ReadSingle()
                };
            }
        }

        // Read texture units
        if (texUnitCount > 0 && texUnitOfs > 0)
        {
            reader.BaseStream.Position = texUnitOfs;
            data.TextureUnits = new M2TextureUnit[texUnitCount];
            for (int i = 0; i < texUnitCount; i++)
            {
                data.TextureUnits[i] = new M2TextureUnit
                {
                    Flags = reader.ReadUInt16(),
                    Priority = reader.ReadUInt16(),
                    ShaderId = reader.ReadUInt16(),
                    SkinSectionIndex = reader.ReadUInt16(),
                    GeosetIndex = reader.ReadUInt16(),
                    ColorIndex = reader.ReadInt16(),
                    MaterialIndex = reader.ReadUInt16(),
                    MaterialLayer = reader.ReadUInt16(),
                    TextureCount = reader.ReadUInt16(),
                    TextureLookup = reader.ReadUInt16(),
                    TextureUnitLookup = reader.ReadUInt16(),
                    TransparencyLookup = reader.ReadUInt16(),
                    TextureAnimLookup = reader.ReadUInt16()
                };
            }
        }
    }

    private void WriteMdx(BinaryWriter writer, M2Data data)
    {
        // MDLX magic
        writer.Write(Encoding.ASCII.GetBytes("MDLX"));

        // VERS chunk
        WriteChunk(writer, "VERS", w => w.Write((uint)800)); // MDX version

        // MODL chunk (model info)
        WriteChunk(writer, "MODL", w =>
        {
            WriteFixedString(w, data.Name ?? "Converted", 80);
            WriteFixedString(w, "", 260); // Animation file
            w.Write(data.BoundingSphereRadius);
            WriteVector3(w, data.BoundingBox?[0] ?? Vector3.Zero);
            WriteVector3(w, data.BoundingBox?[1] ?? Vector3.Zero);
            w.Write(150u); // Blend time
            w.Write(0u); // Flags
        });

        // SEQS chunk (sequences/animations)
        if (data.Animations?.Length > 0)
        {
            WriteChunk(writer, "SEQS", w =>
            {
                w.Write(data.Animations.Length);
                foreach (var anim in data.Animations)
                {
                    WriteFixedString(w, $"Anim{anim.AnimationId}", 80);
                    w.Write(0u); // Start
                    w.Write(anim.Length); // End
                    w.Write(anim.MoveSpeed);
                    w.Write(0u); // Flags
                    w.Write(0f); // Rarity
                    w.Write(0u); // SyncPoint
                    WriteVector3(w, anim.MinExtent);
                    WriteVector3(w, anim.MaxExtent);
                    w.Write(anim.BoundRadius);
                }
            });
        }

        // TEXS chunk (textures)
        if (data.Textures?.Length > 0)
        {
            WriteChunk(writer, "TEXS", w =>
            {
                foreach (var tex in data.Textures)
                {
                    w.Write(0u); // ReplaceableId
                    WriteFixedString(w, tex.Filename ?? "", 260);
                    w.Write(0u); // Flags
                }
            });
        }

        // GEOS chunk (geosets/geometry)
        if (data.Vertices?.Length > 0 && data.SkinTriangles?.Length > 0)
        {
            WriteChunk(writer, "GEOS", w =>
            {
                // Write as single geoset for simplicity
                var geosetSize = CalculateGeosetSize(data);
                w.Write(geosetSize);

                // VRTX (vertices)
                w.Write(Encoding.ASCII.GetBytes("VRTX"));
                w.Write(data.Vertices.Length);
                foreach (var v in data.Vertices)
                    WriteVector3(w, v.Position);

                // NRMS (normals)
                w.Write(Encoding.ASCII.GetBytes("NRMS"));
                w.Write(data.Vertices.Length);
                foreach (var v in data.Vertices)
                    WriteVector3(w, v.Normal);

                // PTYP (primitive types)
                w.Write(Encoding.ASCII.GetBytes("PTYP"));
                w.Write(1);
                w.Write(4u); // Triangles

                // PCNT (primitive counts)
                w.Write(Encoding.ASCII.GetBytes("PCNT"));
                w.Write(1);
                w.Write(data.SkinTriangles.Length);

                // PVTX (primitive vertices/indices)
                w.Write(Encoding.ASCII.GetBytes("PVTX"));
                w.Write(data.SkinTriangles.Length);
                foreach (var idx in data.SkinTriangles)
                    w.Write(idx);

                // GNDX (group indices)
                w.Write(Encoding.ASCII.GetBytes("GNDX"));
                w.Write(data.Vertices.Length);
                for (int i = 0; i < data.Vertices.Length; i++)
                    w.Write((byte)0);

                // MTGC (matrix group counts)
                w.Write(Encoding.ASCII.GetBytes("MTGC"));
                w.Write(1);
                w.Write(1u);

                // MATS (matrices)
                w.Write(Encoding.ASCII.GetBytes("MATS"));
                w.Write(1);
                w.Write(0u); // Identity matrix index

                // Material ID
                w.Write(0u);

                // Selection group
                w.Write(0u);

                // Selectable
                w.Write(0u);

                // Bounds
                w.Write(data.BoundingSphereRadius);
                WriteVector3(w, data.BoundingBox?[0] ?? Vector3.Zero);
                WriteVector3(w, data.BoundingBox?[1] ?? Vector3.Zero);

                // Extents count
                w.Write(0u);

                // UVAS (UV animation sets)
                w.Write(Encoding.ASCII.GetBytes("UVAS"));
                w.Write(1);

                // UVBS (UV coordinates)
                w.Write(Encoding.ASCII.GetBytes("UVBS"));
                w.Write(data.Vertices.Length);
                foreach (var v in data.Vertices)
                    WriteVector2(w, v.TexCoords?[0] ?? Vector2.Zero);
            });
        }

        // BONE chunk
        if (data.Bones?.Length > 0)
        {
            WriteChunk(writer, "BONE", w =>
            {
                foreach (var bone in data.Bones)
                {
                    // Simplified bone structure
                    w.Write(0u); // Node size placeholder
                    WriteFixedString(w, $"Bone{bone.KeyBoneId}", 80);
                    w.Write(0u); // ObjectId
                    w.Write(bone.ParentBone);
                    w.Write(0u); // Flags
                    // Skip animation data for now
                    w.Write(0u); // GeosetId
                    w.Write(0u); // GeosetAnimId
                }
            });
        }

        // PIVT chunk (pivot points)
        if (data.Bones?.Length > 0)
        {
            WriteChunk(writer, "PIVT", w =>
            {
                foreach (var bone in data.Bones)
                    WriteVector3(w, bone.Pivot);
            });
        }
    }

    private uint CalculateGeosetSize(M2Data data)
    {
        uint size = 0;
        size += 4 + 4 + (uint)(data.Vertices?.Length ?? 0) * 12; // VRTX
        size += 4 + 4 + (uint)(data.Vertices?.Length ?? 0) * 12; // NRMS
        size += 4 + 4 + 4; // PTYP
        size += 4 + 4 + 4; // PCNT
        size += 4 + 4 + (uint)(data.SkinTriangles?.Length ?? 0) * 2; // PVTX
        size += 4 + 4 + (uint)(data.Vertices?.Length ?? 0); // GNDX
        size += 4 + 4 + 4; // MTGC
        size += 4 + 4 + 4; // MATS
        size += 4 + 4 + 4; // Material, selection, selectable
        size += 4 + 12 + 12; // Bounds
        size += 4; // Extents count
        size += 4 + 4; // UVAS
        size += 4 + 4 + (uint)(data.Vertices?.Length ?? 0) * 8; // UVBS
        return size;
    }

    #region Helpers

    private static Vector3 ReadVector3(BinaryReader r) =>
        new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

    private static Vector2 ReadVector2(BinaryReader r) =>
        new(r.ReadSingle(), r.ReadSingle());

    private static string ReadNullTermString(BinaryReader r, int maxLen)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < maxLen; i++)
        {
            byte b = r.ReadByte();
            if (b == 0) break;
            sb.Append((char)b);
        }
        return sb.ToString();
    }

    private static void WriteChunk(BinaryWriter w, string id, Action<BinaryWriter> writeData)
    {
        w.Write(Encoding.ASCII.GetBytes(id));
        
        using var dataStream = new MemoryStream();
        using var dataWriter = new BinaryWriter(dataStream);
        writeData(dataWriter);
        
        var data = dataStream.ToArray();
        w.Write((uint)data.Length);
        w.Write(data);
    }

    private static void WriteVector3(BinaryWriter w, Vector3 v)
    {
        w.Write(v.X);
        w.Write(v.Y);
        w.Write(v.Z);
    }

    private static void WriteVector2(BinaryWriter w, Vector2 v)
    {
        w.Write(v.X);
        w.Write(v.Y);
    }

    private static void WriteFixedString(BinaryWriter w, string s, int len)
    {
        var bytes = new byte[len];
        var srcBytes = Encoding.ASCII.GetBytes(s ?? "");
        Buffer.BlockCopy(srcBytes, 0, bytes, 0, Math.Min(srcBytes.Length, len - 1));
        w.Write(bytes);
    }

    #endregion

    #region Data Classes

    private class M2Data
    {
        public uint Version;
        public string? Name;
        public uint GlobalFlags;
        public uint ViewCount;
        public Vector3[]? BoundingBox;
        public float BoundingSphereRadius;
        public Vector3[]? CollisionBox;
        public float CollisionSphereRadius;

        public M2Vertex[]? Vertices;
        public M2Texture[]? Textures;
        public M2Animation[]? Animations;
        public M2Bone[]? Bones;

        // From skin file
        public ushort[]? SkinIndices;
        public ushort[]? SkinTriangles;
        public M2Submesh[]? Submeshes;
        public M2TextureUnit[]? TextureUnits;
        public uint SkinBoneCount;
    }

    private class M2Vertex
    {
        public Vector3 Position;
        public byte[]? BoneWeights;
        public byte[]? BoneIndices;
        public Vector3 Normal;
        public Vector2[]? TexCoords;
    }

    private class M2Texture
    {
        public uint Type;
        public uint Flags;
        public string? Filename;
    }

    private class M2Animation
    {
        public ushort AnimationId;
        public ushort SubAnimationId;
        public uint Length;
        public float MoveSpeed;
        public uint Flags;
        public short Probability;
        public ushort Padding;
        public Vector3 MinExtent;
        public Vector3 MaxExtent;
        public float BoundRadius;
        public short NextAnimation;
        public ushort AliasNext;
    }

    private class M2Bone
    {
        public int KeyBoneId;
        public uint Flags;
        public short ParentBone;
        public ushort SubmeshId;
        public Vector3 Pivot;
    }

    private class M2Submesh
    {
        public ushort Id;
        public ushort Level;
        public ushort StartVertex;
        public ushort VertexCount;
        public ushort StartTriangle;
        public ushort TriangleCount;
        public ushort BoneCount;
        public ushort StartBone;
        public ushort BoneInfluences;
        public ushort RootBone;
        public Vector3 CenterMass;
        public Vector3 CenterBoundingBox;
        public float Radius;
    }

    private class M2TextureUnit
    {
        public ushort Flags;
        public ushort Priority;
        public ushort ShaderId;
        public ushort SkinSectionIndex;
        public ushort GeosetIndex;
        public short ColorIndex;
        public ushort MaterialIndex;
        public ushort MaterialLayer;
        public ushort TextureCount;
        public ushort TextureLookup;
        public ushort TextureUnitLookup;
        public ushort TransparencyLookup;
        public ushort TextureAnimLookup;
    }

    #endregion
}
