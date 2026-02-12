using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WoWRollback.MDXSupport
{
    /// <summary>
    /// Converts modern M2 models to Alpha MDX format.
    /// MDX is the model format used in WoW Alpha 0.5.3 (similar to Warcraft 3).
    /// 
    /// M2 Format (Modern):
    /// - Chunked format with MD21, SFID, AFID, etc.
    /// - External .skin, .anim, .bone files
    /// - Complex bone/animation system
    /// 
    /// MDX Format (Alpha):
    /// - Monolithic format with MDLX magic
    /// - All data in single file
    /// - Simpler animation system (WC3-like)
    /// </summary>
    public class M2ToMdxConverter
    {
        private const uint MDX_MAGIC = 0x584C444D; // "MDLX"
        private const uint MDX_VERSION = 800; // Alpha version

        /// <summary>
        /// Convert an M2 file to MDX format.
        /// </summary>
        public void Convert(string m2Path, string mdxOutputPath)
        {
            Console.WriteLine($"[INFO] Converting M2 → MDX: {Path.GetFileName(m2Path)}");

            using var m2Stream = File.OpenRead(m2Path);
            using var reader = new BinaryReader(m2Stream);

            // Parse M2
            var m2Data = ParseM2(reader);

            // Write MDX
            using var mdxStream = File.Create(mdxOutputPath);
            using var writer = new BinaryWriter(mdxStream);

            WriteMdx(writer, m2Data);

            Console.WriteLine($"[SUCCESS] Converted to MDX: {mdxOutputPath}");
        }

        /// <summary>
        /// Convert M2 bytes to MDX bytes in memory.
        /// </summary>
        public byte[] ConvertToMdx(byte[] m2Data)
        {
            using var m2Stream = new MemoryStream(m2Data);
            using var reader = new BinaryReader(m2Stream);
            using var mdxStream = new MemoryStream();
            using var writer = new BinaryWriter(mdxStream);

            var parsed = ParseM2(reader);
            WriteMdx(writer, parsed);

            return mdxStream.ToArray();
        }

        #region M2 Parsing

        private M2Data ParseM2(BinaryReader reader)
        {
            var data = new M2Data();

            // Check for chunked format (modern) vs legacy
            var magic = reader.ReadUInt32();
            reader.BaseStream.Position = 0;

            if (magic == 0x3132444D) // "MD21"
            {
                ParseChunkedM2(reader, data);
            }
            else if (magic == 0x3032444D) // "MD20"
            {
                ParseLegacyM2(reader, data);
            }
            else
            {
                throw new InvalidDataException($"Unknown M2 magic: 0x{magic:X8}");
            }

            return data;
        }

        private void ParseChunkedM2(BinaryReader reader, M2Data data)
        {
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                var chunkId = reader.ReadUInt32();
                var chunkSize = reader.ReadUInt32();
                var chunkEnd = reader.BaseStream.Position + chunkSize;

                switch (chunkId)
                {
                    case 0x3132444D: // "MD21"
                        ParseMD21(reader, data, chunkSize);
                        break;

                    case 0x44494653: // "SFID" - Skin file IDs
                        data.SkinFileIds = new uint[chunkSize / 4];
                        for (int i = 0; i < data.SkinFileIds.Length; i++)
                            data.SkinFileIds[i] = reader.ReadUInt32();
                        break;

                    case 0x44494641: // "AFID" - Animation file IDs
                        // Skip for now - animations need special handling
                        break;

                    case 0x44495854: // "TXID" - Texture file IDs
                        data.TextureFileIds = new uint[chunkSize / 4];
                        for (int i = 0; i < data.TextureFileIds.Length; i++)
                            data.TextureFileIds[i] = reader.ReadUInt32();
                        break;

                    default:
                        // Skip unknown chunks
                        break;
                }

                reader.BaseStream.Position = chunkEnd;
            }
        }

        private void ParseMD21(BinaryReader reader, M2Data data, uint chunkSize)
        {
            var startPos = reader.BaseStream.Position;

            // MD21 contains the legacy M2 header structure
            var magic = reader.ReadUInt32(); // "MD20"
            var version = reader.ReadUInt32();
            data.Version = version;

            // Name
            var nameLength = reader.ReadUInt32();
            var nameOffset = reader.ReadUInt32();
            if (nameLength > 0 && nameOffset > 0)
            {
                var savedPos = reader.BaseStream.Position;
                reader.BaseStream.Position = startPos + nameOffset;
                data.Name = Encoding.UTF8.GetString(reader.ReadBytes((int)nameLength)).TrimEnd('\0');
                reader.BaseStream.Position = savedPos;
            }

            // Flags
            data.Flags = reader.ReadUInt32();

            // Global sequences
            var globalSeqCount = reader.ReadUInt32();
            var globalSeqOffset = reader.ReadUInt32();

            // Animations
            var animCount = reader.ReadUInt32();
            var animOffset = reader.ReadUInt32();
            data.AnimationCount = (int)animCount;

            // Animation lookups
            reader.ReadUInt32(); // count
            reader.ReadUInt32(); // offset

            // Bones
            var boneCount = reader.ReadUInt32();
            var boneOffset = reader.ReadUInt32();
            data.BoneCount = (int)boneCount;

            // Key bone lookups
            reader.ReadUInt32();
            reader.ReadUInt32();

            // Vertices
            var vertexCount = reader.ReadUInt32();
            var vertexOffset = reader.ReadUInt32();

            // Views (skins)
            var viewCount = reader.ReadUInt32();

            // Skip to vertices and read them
            if (vertexCount > 0 && vertexOffset > 0)
            {
                var savedPos = reader.BaseStream.Position;
                reader.BaseStream.Position = startPos + vertexOffset;

                data.Vertices = new M2Vertex[vertexCount];
                for (int i = 0; i < vertexCount; i++)
                {
                    data.Vertices[i] = new M2Vertex
                    {
                        Position = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                        BoneWeights = reader.ReadBytes(4),
                        BoneIndices = reader.ReadBytes(4),
                        Normal = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                        TexCoord1 = new Vector2(reader.ReadSingle(), reader.ReadSingle()),
                        TexCoord2 = new Vector2(reader.ReadSingle(), reader.ReadSingle())
                    };
                }

                reader.BaseStream.Position = savedPos;
            }

            // Continue parsing header for textures, etc.
            // Skip to textures section
            reader.BaseStream.Position = startPos + 80; // Approximate offset to texture info

            var texCount = reader.ReadUInt32();
            var texOffset = reader.ReadUInt32();

            if (texCount > 0 && texOffset > 0)
            {
                var savedPos = reader.BaseStream.Position;
                reader.BaseStream.Position = startPos + texOffset;

                data.Textures = new M2Texture[texCount];
                for (int i = 0; i < texCount; i++)
                {
                    data.Textures[i] = new M2Texture
                    {
                        Type = reader.ReadUInt32(),
                        Flags = reader.ReadUInt32(),
                        NameLength = reader.ReadUInt32(),
                        NameOffset = reader.ReadUInt32()
                    };
                }

                reader.BaseStream.Position = savedPos;
            }
        }

        private void ParseLegacyM2(BinaryReader reader, M2Data data)
        {
            // Legacy M2 format (pre-Legion)
            // Similar to MD21 content but not chunked
            var magic = reader.ReadUInt32();
            data.Version = reader.ReadUInt32();

            // Parse similar to MD21
            var nameLength = reader.ReadUInt32();
            var nameOffset = reader.ReadUInt32();
            if (nameLength > 0 && nameOffset > 0)
            {
                var savedPos = reader.BaseStream.Position;
                reader.BaseStream.Position = nameOffset;
                data.Name = Encoding.UTF8.GetString(reader.ReadBytes((int)nameLength)).TrimEnd('\0');
                reader.BaseStream.Position = savedPos;
            }

            data.Flags = reader.ReadUInt32();

            // Continue parsing...
            // (Similar structure to MD21)
        }

        #endregion

        #region MDX Writing

        private void WriteMdx(BinaryWriter writer, M2Data m2Data)
        {
            // MDX Magic
            writer.Write(Encoding.ASCII.GetBytes("MDLX"));

            // VERS chunk
            WriteChunk(writer, "VERS", w => w.Write(MDX_VERSION));

            // MODL chunk (model info)
            WriteChunk(writer, "MODL", w =>
            {
                // Name (80 bytes, null-padded)
                var nameBytes = Encoding.UTF8.GetBytes(m2Data.Name ?? "Converted");
                w.Write(nameBytes);
                w.Write(new byte[80 - nameBytes.Length]);

                // Animation file name (260 bytes)
                w.Write(new byte[260]);

                // Bounds (CExtent - 28 bytes)
                WriteBounds(w, m2Data);

                // Blend time
                w.Write((uint)150);

                // Flags
                w.Write((uint)0);
            });

            // SEQS chunk (sequences/animations)
            if (m2Data.AnimationCount > 0)
            {
                WriteChunk(writer, "SEQS", w =>
                {
                    w.Write(m2Data.AnimationCount);
                    for (int i = 0; i < m2Data.AnimationCount; i++)
                    {
                        WriteSequence(w, i);
                    }
                });
            }

            // TEXS chunk (textures)
            if (m2Data.Textures != null && m2Data.Textures.Length > 0)
            {
                WriteChunk(writer, "TEXS", w =>
                {
                    foreach (var tex in m2Data.Textures)
                    {
                        WriteTexture(w, tex);
                    }
                });
            }

            // MTLS chunk (materials)
            WriteChunk(writer, "MTLS", w =>
            {
                w.Write(1); // count
                w.Write(0); // unused
                WriteMaterial(w);
            });

            // GEOS chunk (geosets/geometry)
            if (m2Data.Vertices != null && m2Data.Vertices.Length > 0)
            {
                WriteChunk(writer, "GEOS", w =>
                {
                    WriteGeoset(w, m2Data);
                });
            }

            // BONE chunk (bones)
            if (m2Data.BoneCount > 0)
            {
                WriteChunk(writer, "BONE", w =>
                {
                    w.Write(m2Data.BoneCount);
                    for (int i = 0; i < m2Data.BoneCount; i++)
                    {
                        WriteBone(w, i);
                    }
                });
            }

            // PIVT chunk (pivot points)
            if (m2Data.BoneCount > 0)
            {
                WriteChunk(writer, "PIVT", w =>
                {
                    for (int i = 0; i < m2Data.BoneCount; i++)
                    {
                        w.Write(0f); // X
                        w.Write(0f); // Y
                        w.Write(0f); // Z
                    }
                });
            }
        }

        private void WriteChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeContent)
        {
            writer.Write(Encoding.ASCII.GetBytes(chunkId));

            using var contentStream = new MemoryStream();
            using var contentWriter = new BinaryWriter(contentStream);
            writeContent(contentWriter);

            var content = contentStream.ToArray();
            writer.Write((uint)content.Length);
            writer.Write(content);
        }

        private void WriteBounds(BinaryWriter writer, M2Data data)
        {
            // Calculate bounds from vertices
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);

            if (data.Vertices != null)
            {
                foreach (var v in data.Vertices)
                {
                    min = Vector3.Min(min, v.Position);
                    max = Vector3.Max(max, v.Position);
                }
            }
            else
            {
                min = Vector3.Zero;
                max = Vector3.One;
            }

            // CExtent format
            var radius = Vector3.Distance(min, max) / 2;
            writer.Write(radius);
            writer.Write(min.X); writer.Write(min.Y); writer.Write(min.Z);
            writer.Write(max.X); writer.Write(max.Y); writer.Write(max.Z);
        }

        private void WriteSequence(BinaryWriter writer, int index)
        {
            // Sequence name (80 bytes)
            var name = $"Anim{index:D2}";
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes);
            writer.Write(new byte[80 - nameBytes.Length]);

            // Interval
            writer.Write((uint)0);    // start
            writer.Write((uint)1000); // end

            // Move speed
            writer.Write(0f);

            // Flags
            writer.Write((uint)0);

            // Rarity
            writer.Write(0f);

            // Sync point
            writer.Write((uint)0);

            // Bounds
            writer.Write(0f); // radius
            writer.Write(0f); writer.Write(0f); writer.Write(0f); // min
            writer.Write(0f); writer.Write(0f); writer.Write(0f); // max
        }

        private void WriteTexture(BinaryWriter writer, M2Texture tex)
        {
            // Replaceable ID
            writer.Write((uint)0);

            // Filename (260 bytes)
            var filename = "Textures\\White.blp";
            var nameBytes = Encoding.UTF8.GetBytes(filename);
            writer.Write(nameBytes);
            writer.Write(new byte[260 - nameBytes.Length]);

            // Flags
            writer.Write((uint)0);
        }

        private void WriteMaterial(BinaryWriter writer)
        {
            // Material size (inclusive)
            var startPos = writer.BaseStream.Position;
            writer.Write((uint)0); // Placeholder

            // Priority plane
            writer.Write((uint)0);

            // Flags
            writer.Write((uint)0);

            // Shader name (80 bytes)
            writer.Write(new byte[80]);

            // LAYS header
            writer.Write(Encoding.ASCII.GetBytes("LAYS"));
            writer.Write((uint)1); // Layer count

            // Layer
            var layerStart = writer.BaseStream.Position;
            writer.Write((uint)0); // Size placeholder

            writer.Write((uint)0); // Blend mode
            writer.Write((uint)0); // Shading flags
            writer.Write((uint)0); // Texture ID
            writer.Write((uint)0); // Texture anim ID
            writer.Write((uint)0); // Coord ID
            writer.Write(1f);      // Alpha

            // Update layer size
            var layerEnd = writer.BaseStream.Position;
            writer.BaseStream.Position = layerStart;
            writer.Write((uint)(layerEnd - layerStart));
            writer.BaseStream.Position = layerEnd;

            // Update material size
            var endPos = writer.BaseStream.Position;
            writer.BaseStream.Position = startPos;
            writer.Write((uint)(endPos - startPos));
            writer.BaseStream.Position = endPos;
        }

        private void WriteGeoset(BinaryWriter writer, M2Data data)
        {
            var startPos = writer.BaseStream.Position;
            writer.Write((uint)0); // Size placeholder

            // VRTX (vertices)
            writer.Write(Encoding.ASCII.GetBytes("VRTX"));
            writer.Write(data.Vertices.Length);
            foreach (var v in data.Vertices)
            {
                writer.Write(v.Position.X);
                writer.Write(v.Position.Y);
                writer.Write(v.Position.Z);
            }

            // NRMS (normals)
            writer.Write(Encoding.ASCII.GetBytes("NRMS"));
            writer.Write(data.Vertices.Length);
            foreach (var v in data.Vertices)
            {
                writer.Write(v.Normal.X);
                writer.Write(v.Normal.Y);
                writer.Write(v.Normal.Z);
            }

            // UVAS (UV count)
            writer.Write(Encoding.ASCII.GetBytes("UVAS"));
            writer.Write(1); // 1 UV set

            // UVBS (UVs)
            foreach (var v in data.Vertices)
            {
                writer.Write(v.TexCoord1.X);
                writer.Write(v.TexCoord1.Y);
            }

            // Material ID
            writer.Write(0);

            // Selection group
            writer.Write(0);

            // Unselectable
            writer.Write((uint)0);

            // Bounds
            WriteBounds(writer, data);

            // Sequence bounds count
            writer.Write(0);

            // Update size
            var endPos = writer.BaseStream.Position;
            writer.BaseStream.Position = startPos;
            writer.Write((uint)(endPos - startPos));
            writer.BaseStream.Position = endPos;
        }

        private void WriteBone(BinaryWriter writer, int index)
        {
            // GenObject header
            var startPos = writer.BaseStream.Position;
            writer.Write((uint)0); // Size placeholder

            // Name (80 bytes)
            var name = $"Bone{index:D2}";
            var nameBytes = Encoding.UTF8.GetBytes(name);
            writer.Write(nameBytes);
            writer.Write(new byte[80 - nameBytes.Length]);

            // Object ID
            writer.Write(index);

            // Parent ID
            writer.Write(-1);

            // Flags
            writer.Write((uint)0);

            // Geoset ID
            writer.Write(-1);

            // Geoset anim ID
            writer.Write(-1);

            // Update size
            var endPos = writer.BaseStream.Position;
            writer.BaseStream.Position = startPos;
            writer.Write((uint)(endPos - startPos));
            writer.BaseStream.Position = endPos;
        }

        #endregion

        #region Data Structures

        public class M2Data
        {
            public uint Version;
            public string Name;
            public uint Flags;
            public int AnimationCount;
            public int BoneCount;
            public M2Vertex[] Vertices;
            public M2Texture[] Textures;
            public uint[] SkinFileIds;
            public uint[] TextureFileIds;
        }

        public struct M2Vertex
        {
            public Vector3 Position;
            public byte[] BoneWeights;
            public byte[] BoneIndices;
            public Vector3 Normal;
            public Vector2 TexCoord1;
            public Vector2 TexCoord2;
        }

        public struct M2Texture
        {
            public uint Type;
            public uint Flags;
            public uint NameLength;
            public uint NameOffset;
        }

        #endregion
    }
}
