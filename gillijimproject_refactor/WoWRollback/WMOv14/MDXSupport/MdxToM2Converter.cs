using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WoWRollback.MDXSupport
{
    /// <summary>
    /// Converts Alpha MDX models to modern M2 format.
    /// This allows Alpha assets to be used in modern WoW clients.
    /// 
    /// MDX Format (Alpha):
    /// - Monolithic format with MDLX magic
    /// - All data in single file
    /// - WC3-like animation system
    /// 
    /// M2 Format (Modern):
    /// - Chunked format with MD21
    /// - External .skin files for geometry
    /// - Complex bone/animation system
    /// </summary>
    public class MdxToM2Converter
    {
        private const uint M2_MAGIC = 0x3032444D; // "MD20"
        private const uint M2_VERSION = 274; // WotLK version

        /// <summary>
        /// Convert an MDX file to M2 format.
        /// </summary>
        public void Convert(string mdxPath, string m2OutputPath)
        {
            Console.WriteLine($"[INFO] Converting MDX → M2: {Path.GetFileName(mdxPath)}");

            using var mdxStream = File.OpenRead(mdxPath);
            using var reader = new BinaryReader(mdxStream);

            // Parse MDX
            var mdxData = ParseMdx(reader);

            // Write M2
            using var m2Stream = File.Create(m2OutputPath);
            using var writer = new BinaryWriter(m2Stream);

            WriteM2(writer, mdxData);

            // Write .skin file
            var skinPath = Path.ChangeExtension(m2OutputPath, ".skin");
            WriteSkin(skinPath, mdxData);

            Console.WriteLine($"[SUCCESS] Converted to M2: {m2OutputPath}");
        }

        /// <summary>
        /// Convert MDX bytes to M2 bytes in memory.
        /// </summary>
        public byte[] ConvertToM2(byte[] mdxData)
        {
            using var mdxStream = new MemoryStream(mdxData);
            using var reader = new BinaryReader(mdxStream);
            using var m2Stream = new MemoryStream();
            using var writer = new BinaryWriter(m2Stream);

            var parsed = ParseMdx(reader);
            WriteM2(writer, parsed);

            return m2Stream.ToArray();
        }

        #region MDX Parsing

        private MdxData ParseMdx(BinaryReader reader)
        {
            var data = new MdxData();

            // Check magic
            var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
            if (magic != "MDLX")
            {
                throw new InvalidDataException($"Invalid MDX magic: {magic}");
            }

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
                        data.Name = Encoding.UTF8.GetString(reader.ReadBytes(80)).TrimEnd('\0');
                        data.AnimationFile = Encoding.UTF8.GetString(reader.ReadBytes(260)).TrimEnd('\0');
                        data.BoundsRadius = reader.ReadSingle();
                        data.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                        data.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                        data.BlendTime = reader.ReadUInt32();
                        data.Flags = reader.ReadUInt32();
                        break;

                    case "SEQS":
                        var seqCount = reader.ReadInt32();
                        data.Sequences = new MdxSequence[seqCount];
                        for (int i = 0; i < seqCount; i++)
                        {
                            data.Sequences[i] = ReadSequence(reader);
                        }
                        break;

                    case "TEXS":
                        var texCount = (int)(chunkSize / 268);
                        data.Textures = new MdxTexture[texCount];
                        for (int i = 0; i < texCount; i++)
                        {
                            data.Textures[i] = ReadTexture(reader);
                        }
                        break;

                    case "MTLS":
                        var matCount = reader.ReadInt32();
                        reader.ReadInt32(); // unused
                        data.Materials = new MdxMaterial[matCount];
                        for (int i = 0; i < matCount; i++)
                        {
                            data.Materials[i] = ReadMaterial(reader);
                        }
                        break;

                    case "GEOS":
                        data.Geosets = ReadGeosets(reader, chunkEnd);
                        break;

                    case "BONE":
                        var boneCount = reader.ReadInt32();
                        data.Bones = new MdxBone[boneCount];
                        for (int i = 0; i < boneCount; i++)
                        {
                            data.Bones[i] = ReadBone(reader);
                        }
                        break;

                    case "PIVT":
                        var pivotCount = (int)(chunkSize / 12);
                        data.Pivots = new Vector3[pivotCount];
                        for (int i = 0; i < pivotCount; i++)
                        {
                            data.Pivots[i] = new Vector3(
                                reader.ReadSingle(),
                                reader.ReadSingle(),
                                reader.ReadSingle());
                        }
                        break;

                    default:
                        // Skip unknown chunks
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
                Name = Encoding.UTF8.GetString(reader.ReadBytes(80)).TrimEnd('\0'),
                IntervalStart = reader.ReadUInt32(),
                IntervalEnd = reader.ReadUInt32(),
                MoveSpeed = reader.ReadSingle(),
                Flags = reader.ReadUInt32(),
                Rarity = reader.ReadSingle(),
                SyncPoint = reader.ReadUInt32(),
                BoundsRadius = reader.ReadSingle(),
                BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle())
            };
        }

        private MdxTexture ReadTexture(BinaryReader reader)
        {
            return new MdxTexture
            {
                ReplaceableId = reader.ReadUInt32(),
                Filename = Encoding.UTF8.GetString(reader.ReadBytes(260)).TrimEnd('\0'),
                Flags = reader.ReadUInt32()
            };
        }

        private MdxMaterial ReadMaterial(BinaryReader reader)
        {
            var mat = new MdxMaterial();
            var startPos = reader.BaseStream.Position;
            var size = reader.ReadUInt32();

            mat.PriorityPlane = reader.ReadUInt32();
            mat.Flags = reader.ReadUInt32();

            // Skip shader name
            reader.ReadBytes(80);

            // LAYS
            var laysTag = Encoding.ASCII.GetString(reader.ReadBytes(4));
            if (laysTag == "LAYS")
            {
                var layerCount = reader.ReadUInt32();
                mat.Layers = new MdxMaterialLayer[layerCount];

                for (int i = 0; i < layerCount; i++)
                {
                    var layerSize = reader.ReadUInt32();
                    var layerStart = reader.BaseStream.Position;

                    mat.Layers[i] = new MdxMaterialLayer
                    {
                        BlendMode = reader.ReadUInt32(),
                        ShadingFlags = reader.ReadUInt32(),
                        TextureId = reader.ReadUInt32(),
                        TextureAnimId = reader.ReadUInt32(),
                        CoordId = reader.ReadUInt32(),
                        Alpha = reader.ReadSingle()
                    };

                    reader.BaseStream.Position = layerStart + layerSize - 4;
                }
            }

            reader.BaseStream.Position = startPos + size;
            return mat;
        }

        private MdxGeoset[] ReadGeosets(BinaryReader reader, long chunkEnd)
        {
            var geosets = new List<MdxGeoset>();

            while (reader.BaseStream.Position < chunkEnd)
            {
                var geo = new MdxGeoset();
                var startPos = reader.BaseStream.Position;
                var size = reader.ReadUInt32();

                // Parse sub-chunks
                while (reader.BaseStream.Position < startPos + size)
                {
                    var tag = Encoding.ASCII.GetString(reader.ReadBytes(4));
                    var count = reader.ReadInt32();

                    switch (tag)
                    {
                        case "VRTX":
                            geo.Vertices = new Vector3[count];
                            for (int i = 0; i < count; i++)
                            {
                                geo.Vertices[i] = new Vector3(
                                    reader.ReadSingle(),
                                    reader.ReadSingle(),
                                    reader.ReadSingle());
                            }
                            break;

                        case "NRMS":
                            geo.Normals = new Vector3[count];
                            for (int i = 0; i < count; i++)
                            {
                                geo.Normals[i] = new Vector3(
                                    reader.ReadSingle(),
                                    reader.ReadSingle(),
                                    reader.ReadSingle());
                            }
                            break;

                        case "UVAS":
                            // UV set count, followed by UVs
                            var uvCount = geo.Normals?.Length ?? 0;
                            geo.UVs = new Vector2[uvCount];
                            for (int i = 0; i < uvCount; i++)
                            {
                                geo.UVs[i] = new Vector2(
                                    reader.ReadSingle(),
                                    reader.ReadSingle());
                            }
                            break;

                        case "PTYP":
                            geo.PrimitiveTypes = reader.ReadBytes(count);
                            break;

                        case "PCNT":
                            geo.PrimitiveCounts = new int[count];
                            for (int i = 0; i < count; i++)
                                geo.PrimitiveCounts[i] = reader.ReadInt32();
                            break;

                        case "PVTX":
                            geo.Indices = new ushort[count];
                            for (int i = 0; i < count; i++)
                                geo.Indices[i] = reader.ReadUInt16();
                            break;

                        case "GNDX":
                            geo.VertexGroups = reader.ReadBytes(count);
                            break;

                        case "MTGC":
                            geo.MatrixGroupCounts = new int[count];
                            for (int i = 0; i < count; i++)
                                geo.MatrixGroupCounts[i] = reader.ReadInt32();
                            break;

                        case "MATS":
                            geo.MatrixIndices = new int[count];
                            for (int i = 0; i < count; i++)
                                geo.MatrixIndices[i] = reader.ReadInt32();
                            break;

                        default:
                            // Skip unknown
                            reader.BaseStream.Position += count * 4; // Assume 4 bytes per element
                            break;
                    }
                }

                // Read remaining geoset data
                geo.MaterialId = reader.ReadInt32();
                geo.SelectionGroup = reader.ReadInt32();
                geo.Unselectable = reader.ReadUInt32() == 1;

                // Bounds
                geo.BoundsRadius = reader.ReadSingle();
                geo.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
                geo.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

                // Sequence bounds
                var seqBoundsCount = reader.ReadInt32();
                reader.BaseStream.Position += seqBoundsCount * 28; // Skip

                geosets.Add(geo);
            }

            return geosets.ToArray();
        }

        private MdxBone ReadBone(BinaryReader reader)
        {
            var bone = new MdxBone();
            var startPos = reader.BaseStream.Position;
            var size = reader.ReadUInt32();

            bone.Name = Encoding.UTF8.GetString(reader.ReadBytes(80)).TrimEnd('\0');
            bone.ObjectId = reader.ReadInt32();
            bone.ParentId = reader.ReadInt32();
            bone.Flags = reader.ReadUInt32();

            // Skip to end of bone data
            reader.BaseStream.Position = startPos + size;

            return bone;
        }

        #endregion

        #region M2 Writing

        private void WriteM2(BinaryWriter writer, MdxData mdxData)
        {
            // For modern clients, use chunked format
            // MD21 chunk contains the legacy M2 structure

            var md21Start = writer.BaseStream.Position;
            writer.Write(Encoding.ASCII.GetBytes("MD21"));
            var md21SizePos = writer.BaseStream.Position;
            writer.Write((uint)0); // Size placeholder

            var md21DataStart = writer.BaseStream.Position;

            // Write legacy M2 header
            WriteLegacyM2Header(writer, mdxData);

            // Update MD21 size
            var md21End = writer.BaseStream.Position;
            writer.BaseStream.Position = md21SizePos;
            writer.Write((uint)(md21End - md21DataStart));
            writer.BaseStream.Position = md21End;

            // SFID chunk (skin file IDs) - we'll use 0 for now
            writer.Write(Encoding.ASCII.GetBytes("SFID"));
            writer.Write((uint)4);
            writer.Write((uint)0);
        }

        private void WriteLegacyM2Header(BinaryWriter writer, MdxData mdxData)
        {
            var headerStart = writer.BaseStream.Position;

            // Magic and version
            writer.Write(M2_MAGIC);
            writer.Write(M2_VERSION);

            // We'll need to track offsets for various data sections
            var dataOffset = 0x130; // Approximate header size

            // Name
            var nameBytes = Encoding.UTF8.GetBytes(mdxData.Name ?? "Converted");
            writer.Write((uint)nameBytes.Length + 1);
            writer.Write((uint)dataOffset);
            dataOffset += nameBytes.Length + 1;

            // Flags
            writer.Write(mdxData.Flags);

            // Global sequences
            writer.Write((uint)0); // count
            writer.Write((uint)0); // offset

            // Animations
            var animCount = mdxData.Sequences?.Length ?? 0;
            writer.Write((uint)animCount);
            writer.Write((uint)(animCount > 0 ? dataOffset : 0));
            dataOffset += animCount * 64; // M2 animation entry size

            // Animation lookups
            writer.Write((uint)animCount);
            writer.Write((uint)(animCount > 0 ? dataOffset : 0));
            dataOffset += animCount * 2;

            // Bones
            var boneCount = mdxData.Bones?.Length ?? 0;
            writer.Write((uint)boneCount);
            writer.Write((uint)(boneCount > 0 ? dataOffset : 0));
            dataOffset += boneCount * 88; // M2 bone size

            // Key bone lookups
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Vertices
            var vertexCount = 0;
            if (mdxData.Geosets != null)
            {
                foreach (var geo in mdxData.Geosets)
                    vertexCount += geo.Vertices?.Length ?? 0;
            }
            writer.Write((uint)vertexCount);
            writer.Write((uint)(vertexCount > 0 ? dataOffset : 0));
            dataOffset += vertexCount * 48; // M2 vertex size

            // Views (skins)
            writer.Write((uint)1); // Always 1 view

            // Colors
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Textures
            var texCount = mdxData.Textures?.Length ?? 0;
            writer.Write((uint)texCount);
            writer.Write((uint)(texCount > 0 ? dataOffset : 0));
            dataOffset += texCount * 16;

            // Transparency
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Texture animations
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Texture replacements
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Materials
            var matCount = mdxData.Materials?.Length ?? 0;
            writer.Write((uint)matCount);
            writer.Write((uint)(matCount > 0 ? dataOffset : 0));
            dataOffset += matCount * 4;

            // Bone lookups
            writer.Write((uint)boneCount);
            writer.Write((uint)(boneCount > 0 ? dataOffset : 0));
            dataOffset += boneCount * 2;

            // Texture lookups
            writer.Write((uint)texCount);
            writer.Write((uint)(texCount > 0 ? dataOffset : 0));
            dataOffset += texCount * 2;

            // Texture units
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Transparency lookups
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Texture animation lookups
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Bounding box
            writer.Write(mdxData.BoundsMin.X);
            writer.Write(mdxData.BoundsMin.Y);
            writer.Write(mdxData.BoundsMin.Z);
            writer.Write(mdxData.BoundsMax.X);
            writer.Write(mdxData.BoundsMax.Y);
            writer.Write(mdxData.BoundsMax.Z);

            // Bounding sphere radius
            writer.Write(mdxData.BoundsRadius);

            // Collision box
            writer.Write(mdxData.BoundsMin.X);
            writer.Write(mdxData.BoundsMin.Y);
            writer.Write(mdxData.BoundsMin.Z);
            writer.Write(mdxData.BoundsMax.X);
            writer.Write(mdxData.BoundsMax.Y);
            writer.Write(mdxData.BoundsMax.Z);

            // Collision sphere radius
            writer.Write(mdxData.BoundsRadius);

            // Collision triangles
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Collision vertices
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Collision normals
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Attachments
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Attachment lookups
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Events
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Lights
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Cameras
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Camera lookups
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Ribbon emitters
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Particle emitters
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Pad to data offset
            var currentPos = writer.BaseStream.Position;
            var paddingNeeded = (headerStart + 0x130) - currentPos;
            if (paddingNeeded > 0)
                writer.Write(new byte[paddingNeeded]);

            // Write data sections
            // Name
            writer.Write(nameBytes);
            writer.Write((byte)0);

            // Animations
            if (mdxData.Sequences != null)
            {
                foreach (var seq in mdxData.Sequences)
                {
                    WriteM2Animation(writer, seq);
                }
            }

            // Animation lookups
            for (int i = 0; i < animCount; i++)
                writer.Write((ushort)i);

            // Bones
            if (mdxData.Bones != null)
            {
                foreach (var bone in mdxData.Bones)
                {
                    WriteM2Bone(writer, bone);
                }
            }

            // Vertices
            if (mdxData.Geosets != null)
            {
                foreach (var geo in mdxData.Geosets)
                {
                    if (geo.Vertices == null) continue;
                    for (int i = 0; i < geo.Vertices.Length; i++)
                    {
                        WriteM2Vertex(writer, geo, i);
                    }
                }
            }

            // Textures
            if (mdxData.Textures != null)
            {
                foreach (var tex in mdxData.Textures)
                {
                    WriteM2Texture(writer, tex);
                }
            }

            // Materials
            if (mdxData.Materials != null)
            {
                foreach (var mat in mdxData.Materials)
                {
                    writer.Write((ushort)0); // flags
                    writer.Write((ushort)0); // blend mode
                }
            }

            // Bone lookups
            for (int i = 0; i < boneCount; i++)
                writer.Write((ushort)i);

            // Texture lookups
            for (int i = 0; i < texCount; i++)
                writer.Write((ushort)i);
        }

        private void WriteM2Animation(BinaryWriter writer, MdxSequence seq)
        {
            writer.Write((ushort)0); // animation ID
            writer.Write((ushort)0); // sub-animation ID
            writer.Write(seq.IntervalEnd - seq.IntervalStart); // duration
            writer.Write(seq.MoveSpeed);
            writer.Write(seq.Flags);
            writer.Write((short)-1); // probability
            writer.Write((ushort)0); // unused
            writer.Write((uint)0); // minimum repetitions
            writer.Write((uint)0); // maximum repetitions
            writer.Write((uint)0); // blend time
            writer.Write(seq.BoundsMin.X);
            writer.Write(seq.BoundsMin.Y);
            writer.Write(seq.BoundsMin.Z);
            writer.Write(seq.BoundsMax.X);
            writer.Write(seq.BoundsMax.Y);
            writer.Write(seq.BoundsMax.Z);
            writer.Write(seq.BoundsRadius);
            writer.Write((short)-1); // next animation
            writer.Write((ushort)0); // alias next
        }

        private void WriteM2Bone(BinaryWriter writer, MdxBone bone)
        {
            writer.Write(-1); // key bone ID
            writer.Write((uint)0); // flags
            writer.Write((short)bone.ParentId);
            writer.Write((ushort)0); // submesh ID
            writer.Write((uint)0); // bone name CRC
            // Translation animation block
            writer.Write((uint)0); writer.Write((uint)0); writer.Write((uint)0); writer.Write((uint)0);
            // Rotation animation block
            writer.Write((uint)0); writer.Write((uint)0); writer.Write((uint)0); writer.Write((uint)0);
            // Scale animation block
            writer.Write((uint)0); writer.Write((uint)0); writer.Write((uint)0); writer.Write((uint)0);
            // Pivot
            writer.Write(0f); writer.Write(0f); writer.Write(0f);
        }

        private void WriteM2Vertex(BinaryWriter writer, MdxGeoset geo, int index)
        {
            var pos = geo.Vertices[index];
            writer.Write(pos.X);
            writer.Write(pos.Y);
            writer.Write(pos.Z);

            // Bone weights (4 bytes)
            writer.Write((byte)255);
            writer.Write((byte)0);
            writer.Write((byte)0);
            writer.Write((byte)0);

            // Bone indices (4 bytes)
            writer.Write((byte)0);
            writer.Write((byte)0);
            writer.Write((byte)0);
            writer.Write((byte)0);

            // Normal
            var normal = geo.Normals != null && index < geo.Normals.Length
                ? geo.Normals[index]
                : Vector3.UnitY;
            writer.Write(normal.X);
            writer.Write(normal.Y);
            writer.Write(normal.Z);

            // UV1
            var uv = geo.UVs != null && index < geo.UVs.Length
                ? geo.UVs[index]
                : Vector2.Zero;
            writer.Write(uv.X);
            writer.Write(uv.Y);

            // UV2
            writer.Write(0f);
            writer.Write(0f);
        }

        private void WriteM2Texture(BinaryWriter writer, MdxTexture tex)
        {
            writer.Write(tex.ReplaceableId);
            writer.Write(tex.Flags);
            writer.Write((uint)0); // name length
            writer.Write((uint)0); // name offset
        }

        private void WriteSkin(string skinPath, MdxData mdxData)
        {
            // Write a minimal .skin file
            using var stream = File.Create(skinPath);
            using var writer = new BinaryWriter(stream);

            // SKIN magic
            writer.Write(Encoding.ASCII.GetBytes("SKIN"));

            // Indices
            var totalIndices = 0;
            if (mdxData.Geosets != null)
            {
                foreach (var geo in mdxData.Geosets)
                    totalIndices += geo.Indices?.Length ?? 0;
            }

            writer.Write((uint)totalIndices);
            writer.Write((uint)0x10); // offset

            // Triangles
            writer.Write((uint)(totalIndices / 3));
            writer.Write((uint)(0x10 + totalIndices * 2));

            // Bones
            writer.Write((uint)0);
            writer.Write((uint)0);

            // Submeshes
            var submeshCount = mdxData.Geosets?.Length ?? 0;
            writer.Write((uint)submeshCount);
            writer.Write((uint)(0x10 + totalIndices * 2 + totalIndices * 2));

            // Batches
            writer.Write((uint)submeshCount);
            writer.Write((uint)(0x10 + totalIndices * 2 + totalIndices * 2 + submeshCount * 32));

            // Write indices
            if (mdxData.Geosets != null)
            {
                foreach (var geo in mdxData.Geosets)
                {
                    if (geo.Indices != null)
                    {
                        foreach (var idx in geo.Indices)
                            writer.Write(idx);
                    }
                }
            }

            // Write triangles (same as indices for now)
            if (mdxData.Geosets != null)
            {
                foreach (var geo in mdxData.Geosets)
                {
                    if (geo.Indices != null)
                    {
                        foreach (var idx in geo.Indices)
                            writer.Write(idx);
                    }
                }
            }

            // Write submeshes
            var vertexOffset = 0;
            var indexOffset = 0;
            if (mdxData.Geosets != null)
            {
                foreach (var geo in mdxData.Geosets)
                {
                    var vertCount = geo.Vertices?.Length ?? 0;
                    var idxCount = geo.Indices?.Length ?? 0;

                    writer.Write((ushort)0); // submesh ID
                    writer.Write((ushort)vertexOffset); // start vertex
                    writer.Write((ushort)vertCount); // vertex count
                    writer.Write((ushort)indexOffset); // start triangle
                    writer.Write((ushort)(idxCount / 3)); // triangle count
                    writer.Write((ushort)0); // bone count
                    writer.Write((ushort)0); // start bone
                    writer.Write((ushort)0); // unknown
                    writer.Write((ushort)0); // root bone
                    writer.Write(geo.BoundsMin.X);
                    writer.Write(geo.BoundsMin.Y);
                    writer.Write(geo.BoundsMin.Z);
                    writer.Write(geo.BoundsRadius);

                    vertexOffset += vertCount;
                    indexOffset += idxCount / 3;
                }
            }

            // Write batches
            if (mdxData.Geosets != null)
            {
                for (int i = 0; i < mdxData.Geosets.Length; i++)
                {
                    writer.Write((byte)0); // flags
                    writer.Write((byte)0); // priority plane
                    writer.Write((ushort)0); // shader ID
                    writer.Write((ushort)i); // submesh index
                    writer.Write((ushort)i); // submesh index 2
                    writer.Write((short)-1); // color index
                    writer.Write((ushort)0); // render flags
                    writer.Write((ushort)0); // layer
                    writer.Write((ushort)0); // op count
                    writer.Write((ushort)0); // texture
                    writer.Write((ushort)0); // tex unit 2
                    writer.Write((short)-1); // transparency
                    writer.Write((ushort)0); // texture anim
                }
            }
        }

        #endregion

        #region Data Structures

        public class MdxData
        {
            public uint Version;
            public string Name;
            public string AnimationFile;
            public float BoundsRadius;
            public Vector3 BoundsMin;
            public Vector3 BoundsMax;
            public uint BlendTime;
            public uint Flags;

            public MdxSequence[] Sequences;
            public MdxTexture[] Textures;
            public MdxMaterial[] Materials;
            public MdxGeoset[] Geosets;
            public MdxBone[] Bones;
            public Vector3[] Pivots;
        }

        public struct MdxSequence
        {
            public string Name;
            public uint IntervalStart;
            public uint IntervalEnd;
            public float MoveSpeed;
            public uint Flags;
            public float Rarity;
            public uint SyncPoint;
            public float BoundsRadius;
            public Vector3 BoundsMin;
            public Vector3 BoundsMax;
        }

        public struct MdxTexture
        {
            public uint ReplaceableId;
            public string Filename;
            public uint Flags;
        }

        public class MdxMaterial
        {
            public uint PriorityPlane;
            public uint Flags;
            public MdxMaterialLayer[] Layers;
        }

        public struct MdxMaterialLayer
        {
            public uint BlendMode;
            public uint ShadingFlags;
            public uint TextureId;
            public uint TextureAnimId;
            public uint CoordId;
            public float Alpha;
        }

        public class MdxGeoset
        {
            public Vector3[] Vertices;
            public Vector3[] Normals;
            public Vector2[] UVs;
            public byte[] PrimitiveTypes;
            public int[] PrimitiveCounts;
            public ushort[] Indices;
            public byte[] VertexGroups;
            public int[] MatrixGroupCounts;
            public int[] MatrixIndices;
            public int MaterialId;
            public int SelectionGroup;
            public bool Unselectable;
            public float BoundsRadius;
            public Vector3 BoundsMin;
            public Vector3 BoundsMax;
        }

        public struct MdxBone
        {
            public string Name;
            public int ObjectId;
            public int ParentId;
            public uint Flags;
        }

        #endregion
    }
}
