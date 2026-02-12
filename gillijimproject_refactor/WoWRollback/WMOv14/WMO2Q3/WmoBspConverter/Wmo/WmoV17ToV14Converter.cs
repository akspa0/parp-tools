using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Converts WMO v17 format (WotLK+) to v14 format (Alpha 0.5.3 compatible).
    /// This enables modern WMO assets to be used in the Alpha client.
    /// </summary>
    public class WmoV17ToV14Converter
    {
        // v14 WMO format differences from v17:
        // - MOHD: Different header layout (no LOD, different flags)
        // - MOMT: Simpler material format (40 bytes vs 64 bytes)
        // - MOBA: Different batch format (no bounding box data)
        // - No MOPV/MOPR (portal vertices/references) - portals not used in v14
        // - No MOLT v2 lighting - simpler light format
        // - MOGP header is smaller

        /// <summary>
        /// Convert a v17 WMO file to v14 format.
        /// </summary>
        public void ConvertAndWrite(string inputPath, string outputPath)
        {
            Console.WriteLine($"[INFO] Converting WMO v17 → v14: {Path.GetFileName(inputPath)}");

            using var inputStream = File.OpenRead(inputPath);
            using var reader = new BinaryReader(inputStream);

            // Parse v17 WMO
            var v17Data = ParseV17Root(reader);

            // Create output directory
            var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            Directory.CreateDirectory(outputDir);

            // Write v14 root file
            WriteV14Root(v17Data, outputPath);

            // Convert and write group files
            var baseName = Path.GetFileNameWithoutExtension(inputPath);
            var inputDir = Path.GetDirectoryName(inputPath) ?? ".";

            for (int i = 0; i < v17Data.GroupCount; i++)
            {
                var groupInputPath = Path.Combine(inputDir, $"{baseName}_{i:D3}.wmo");
                var groupOutputPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(outputPath)}_{i:D3}.wmo");

                if (File.Exists(groupInputPath))
                {
                    ConvertGroupFile(groupInputPath, groupOutputPath, v17Data);
                }
            }

            Console.WriteLine($"[SUCCESS] Converted to v14: {outputPath}");
        }

        /// <summary>
        /// Convert v17 WMO bytes to v14 format in memory.
        /// </summary>
        public byte[] ConvertToV14(byte[] v17Data)
        {
            using var inputStream = new MemoryStream(v17Data);
            using var reader = new BinaryReader(inputStream);
            using var outputStream = new MemoryStream();
            using var writer = new BinaryWriter(outputStream);

            var parsed = ParseV17Root(reader);
            WriteV14RootToStream(parsed, writer);

            return outputStream.ToArray();
        }

        #region V17 Parsing

        private WmoV17Data ParseV17Root(BinaryReader reader)
        {
            var data = new WmoV17Data();

            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var chunkSize = reader.ReadUInt32();
                var chunkEnd = reader.BaseStream.Position + chunkSize;

                switch (chunkId)
                {
                    case "MVER":
                        data.Version = reader.ReadUInt32();
                        break;

                    case "MOHD":
                        data.MaterialCount = reader.ReadUInt32();
                        data.GroupCount = reader.ReadUInt32();
                        data.PortalCount = reader.ReadUInt32();
                        data.LightCount = reader.ReadUInt32();
                        data.ModelCount = reader.ReadUInt32();
                        data.DoodadCount = reader.ReadUInt32();
                        data.SetCount = reader.ReadUInt32();
                        data.AmbientColor = reader.ReadUInt32();
                        data.AreaTableId = reader.ReadUInt32();
                        data.BoundingBox = ReadBoundingBox(reader);
                        data.Flags = reader.ReadUInt16();
                        data.LodCount = reader.ReadUInt16();
                        break;

                    case "MOTX":
                        data.TextureData = reader.ReadBytes((int)chunkSize);
                        break;

                    case "MOMT":
                        data.Materials = new List<WmoMaterialV17>();
                        int matCount = (int)(chunkSize / 64); // v17 materials are 64 bytes
                        for (int i = 0; i < matCount; i++)
                        {
                            data.Materials.Add(ReadMaterialV17(reader));
                        }
                        break;

                    case "MOGN":
                        data.GroupNameData = reader.ReadBytes((int)chunkSize);
                        break;

                    case "MOGI":
                        data.GroupInfos = new List<WmoGroupInfoV17>();
                        int groupCount = (int)(chunkSize / 32);
                        for (int i = 0; i < groupCount; i++)
                        {
                            data.GroupInfos.Add(ReadGroupInfoV17(reader));
                        }
                        break;

                    case "MODS":
                        data.DoodadSetData = reader.ReadBytes((int)chunkSize);
                        break;

                    case "MODN":
                        data.DoodadNameData = reader.ReadBytes((int)chunkSize);
                        break;

                    case "MODD":
                        data.DoodadDefData = reader.ReadBytes((int)chunkSize);
                        break;

                    default:
                        // Skip unknown chunks
                        break;
                }

                reader.BaseStream.Position = chunkEnd;
            }

            return data;
        }

        private WmoMaterialV17 ReadMaterialV17(BinaryReader reader)
        {
            return new WmoMaterialV17
            {
                Flags = reader.ReadUInt32(),
                Shader = reader.ReadUInt32(),
                BlendMode = reader.ReadUInt32(),
                Texture1Offset = reader.ReadUInt32(),
                SidnColor = reader.ReadUInt32(),
                FrameSidnColor = reader.ReadUInt32(),
                Texture2Offset = reader.ReadUInt32(),
                DiffColor = reader.ReadUInt32(),
                GroundType = reader.ReadUInt32(),
                Texture3Offset = reader.ReadUInt32(),
                Color3 = reader.ReadUInt32(),
                Flags3 = reader.ReadUInt32(),
                RuntimeData = reader.ReadBytes(16)
            };
        }

        private WmoGroupInfoV17 ReadGroupInfoV17(BinaryReader reader)
        {
            return new WmoGroupInfoV17
            {
                Flags = reader.ReadUInt32(),
                BoundingBox = ReadBoundingBox(reader),
                NameOffset = reader.ReadInt32()
            };
        }

        private BoundingBox ReadBoundingBox(BinaryReader reader)
        {
            return new BoundingBox
            {
                Min = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                Max = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle())
            };
        }

        #endregion

        #region V14 Writing

        private void WriteV14Root(WmoV17Data v17Data, string outputPath)
        {
            using var stream = File.Create(outputPath);
            using var writer = new BinaryWriter(stream);
            WriteV14RootToStream(v17Data, writer);
        }

        private void WriteV14RootToStream(WmoV17Data v17Data, BinaryWriter writer)
        {
            // MVER - version 14
            WriteChunk(writer, "MVER", w => w.Write((uint)14));

            // MOHD - v14 header (simpler than v17)
            WriteChunk(writer, "MOHD", w =>
            {
                w.Write(v17Data.MaterialCount);
                w.Write(v17Data.GroupCount);
                w.Write((uint)0); // nPortals - not used in v14
                w.Write(v17Data.LightCount);
                w.Write(v17Data.ModelCount);
                w.Write(v17Data.DoodadCount);
                w.Write(v17Data.SetCount);
                w.Write(v17Data.AmbientColor);
                w.Write((uint)0); // areaTableID - not used in v14
                WriteVector3(w, v17Data.BoundingBox.Min);
                WriteVector3(w, v17Data.BoundingBox.Max);
                w.Write((ushort)(v17Data.Flags & 0xFF)); // Strip v17-only flags
            });

            // MOTX - textures (unchanged)
            if (v17Data.TextureData != null && v17Data.TextureData.Length > 0)
            {
                WriteChunk(writer, "MOTX", w => w.Write(v17Data.TextureData));
            }

            // MOMT - materials (downgrade from 64 to 40 bytes)
            if (v17Data.Materials != null && v17Data.Materials.Count > 0)
            {
                WriteChunk(writer, "MOMT", w =>
                {
                    foreach (var mat in v17Data.Materials)
                    {
                        WriteMaterialV14(w, mat);
                    }
                });
            }

            // MOGN - group names (unchanged)
            if (v17Data.GroupNameData != null && v17Data.GroupNameData.Length > 0)
            {
                WriteChunk(writer, "MOGN", w => w.Write(v17Data.GroupNameData));
            }

            // MOGI - group info (downgrade from 32 to 28 bytes - no nameOffset in v14)
            if (v17Data.GroupInfos != null && v17Data.GroupInfos.Count > 0)
            {
                WriteChunk(writer, "MOGI", w =>
                {
                    foreach (var info in v17Data.GroupInfos)
                    {
                        w.Write(info.Flags);
                        WriteVector3(w, info.BoundingBox.Min);
                        WriteVector3(w, info.BoundingBox.Max);
                        w.Write(info.NameOffset); // v14 has nameOffset (32 bytes total)
                    }
                });
            }

            // MODS - doodad sets (unchanged)
            if (v17Data.DoodadSetData != null && v17Data.DoodadSetData.Length > 0)
            {
                WriteChunk(writer, "MODS", w => w.Write(v17Data.DoodadSetData));
            }

            // MODN - doodad names (unchanged)
            if (v17Data.DoodadNameData != null && v17Data.DoodadNameData.Length > 0)
            {
                WriteChunk(writer, "MODN", w => w.Write(v17Data.DoodadNameData));
            }

            // MODD - doodad definitions (unchanged)
            if (v17Data.DoodadDefData != null && v17Data.DoodadDefData.Length > 0)
            {
                WriteChunk(writer, "MODD", w => w.Write(v17Data.DoodadDefData));
            }
        }

        private void WriteMaterialV14(BinaryWriter writer, WmoMaterialV17 mat)
        {
            // v14 material is 44 bytes in our parser (starts with version)
            writer.Write((uint)1); // Version (V14)
            writer.Write(mat.Flags);
            writer.Write(mat.Shader);
            writer.Write(mat.BlendMode);
            writer.Write(mat.Texture1Offset);
            writer.Write(mat.SidnColor);
            writer.Write(mat.FrameSidnColor);
            writer.Write(mat.Texture2Offset);
            writer.Write(mat.DiffColor);
            writer.Write(mat.GroundType);
            writer.Write(mat.Texture3Offset);
        }

        private void ConvertGroupFile(string inputPath, string outputPath, WmoV17Data rootData)
        {
            using var inputStream = File.OpenRead(inputPath);
            using var reader = new BinaryReader(inputStream);
            using var outputStream = File.Create(outputPath);
            using var writer = new BinaryWriter(outputStream);

            // MVER
            WriteChunk(writer, "MVER", w => w.Write((uint)14));

            // Parse and convert MOGP
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var chunkSize = reader.ReadUInt32();
                var chunkStart = reader.BaseStream.Position;

                if (chunkId == "MVER")
                {
                    reader.ReadUInt32(); // Skip version
                }
                else if (chunkId == "MOGP")
                {
                    ConvertMOGP(reader, writer, chunkSize);
                }

                reader.BaseStream.Position = chunkStart + chunkSize;
            }
        }

        private void ConvertMOGP(BinaryReader reader, BinaryWriter writer, uint chunkSize)
        {
            var mogpStart = writer.BaseStream.Position;
            // PGOM reversed
            writer.Write(new byte[] { (byte)'P', (byte)'G', (byte)'O', (byte)'M' });
            var sizePos = writer.BaseStream.Position;
            writer.Write((uint)0); // Placeholder

            var dataStart = writer.BaseStream.Position;

            // Read v17 MOGP header (68 bytes)
            var nameOffset = reader.ReadUInt32();
            var descOffset = reader.ReadUInt32();
            var flags = reader.ReadUInt32();
            var boundingBox = ReadBoundingBox(reader);
            var portalStart = reader.ReadUInt16();
            var portalCount = reader.ReadUInt16();
            var transBatchCount = reader.ReadUInt16();
            var intBatchCount = reader.ReadUInt16();
            var extBatchCount = reader.ReadUInt16();
            var padding = reader.ReadUInt16();
            var fogIndices = reader.ReadBytes(4);
            var liquidType = reader.ReadUInt32();
            var groupId = reader.ReadUInt32();
            var unused1 = reader.ReadUInt32();
            var unused2 = reader.ReadUInt32();

            // Write v14 MOGP header (smaller - 60 bytes)
            writer.Write(nameOffset);
            writer.Write(descOffset);
            writer.Write(flags);
            WriteVector3(writer, boundingBox.Min);
            WriteVector3(writer, boundingBox.Max);
            writer.Write((ushort)0); // portalStart - not used
            writer.Write((ushort)0); // portalCount - not used
            writer.Write(transBatchCount);
            writer.Write(intBatchCount);
            writer.Write(extBatchCount);
            writer.Write(padding);
            writer.Write(fogIndices);
            writer.Write(liquidType);
            writer.Write(groupId);
            writer.Write((uint)0); // Padding to match 64 bytes (Parser expects 0x40)

            // Copy subchunks, converting as needed
            var subchunkEnd = reader.BaseStream.Position + (chunkSize - 68);
            while (reader.BaseStream.Position < subchunkEnd)
            {
                var subId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                var subSize = reader.ReadUInt32();
                var subData = reader.ReadBytes((int)subSize);

                switch (subId)
                {
                    case "MOBA":
                        // Convert v17 MOBA (24 bytes per batch) to v14 (12 bytes per batch)
                        WriteSubChunk(writer, "MOBA", w =>
                        {
                            int batchCount = subData.Length / 24;
                            using var batchReader = new BinaryReader(new MemoryStream(subData));
                            for (int i = 0; i < batchCount; i++)
                            {
                                // Skip bounding box (12 bytes)
                                batchReader.ReadBytes(12);
                                // Read the rest
                                var startIndex = batchReader.ReadUInt32();
                                var count = batchReader.ReadUInt16();
                                var minIndex = batchReader.ReadUInt16();
                                var maxIndex = batchReader.ReadUInt16();
                                var materialId = batchReader.ReadByte();
                                var unknown = batchReader.ReadByte();

                                // Write v14 format (12 bytes)
                                w.Write(startIndex);
                                w.Write(count);
                                w.Write(minIndex);
                                w.Write(maxIndex);
                                w.Write(materialId);
                                w.Write(unknown);
                            }
                        });
                        break;

                    case "MOPY":
                    case "MOVI":
                    case "MOVT":
                    case "MONR":
                    case "MOTV":
                    case "MOCV":
                    case "MLIQ":
                        // These chunks are compatible, copy as-is
                        WriteSubChunk(writer, subId, w => w.Write(subData));
                        break;

                    // Skip v17-only chunks
                    case "MODR":
                    case "MOLR":
                    case "MOBN":
                    case "MOBR":
                        // Not present in v14
                        break;

                    default:
                        // Copy unknown chunks
                        WriteSubChunk(writer, subId, w => w.Write(subData));
                        break;
                }
            }

            // Update MOGP size
            var endPos = writer.BaseStream.Position;
            writer.BaseStream.Position = sizePos;
            writer.Write((uint)(endPos - dataStart));
            writer.BaseStream.Position = endPos;
        }

        #endregion

        #region Helpers

        private void WriteChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeContent)
        {
            var bytes = Encoding.ASCII.GetBytes(chunkId);
            Array.Reverse(bytes);
            writer.Write(bytes);
            var sizePos = writer.BaseStream.Position;
            writer.Write((uint)0);
            var dataStart = writer.BaseStream.Position;

            writeContent(writer);

            var endPos = writer.BaseStream.Position;
            writer.BaseStream.Position = sizePos;
            writer.Write((uint)(endPos - dataStart));
            writer.BaseStream.Position = endPos;
        }

        private void WriteSubChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeContent)
        {
            WriteChunk(writer, chunkId, writeContent);
        }

        private void WriteVector3(BinaryWriter writer, Vector3 v)
        {
            writer.Write(v.X);
            writer.Write(v.Y);
            writer.Write(v.Z);
        }

        #endregion

        #region Data Structures

        public class WmoV17Data
        {
            public uint Version;
            public uint MaterialCount;
            public uint GroupCount;
            public uint PortalCount;
            public uint LightCount;
            public uint ModelCount;
            public uint DoodadCount;
            public uint SetCount;
            public uint AmbientColor;
            public uint AreaTableId;
            public BoundingBox BoundingBox;
            public ushort Flags;
            public ushort LodCount;

            public byte[] TextureData;
            public List<WmoMaterialV17> Materials;
            public byte[] GroupNameData;
            public List<WmoGroupInfoV17> GroupInfos;
            public byte[] DoodadSetData;
            public byte[] DoodadNameData;
            public byte[] DoodadDefData;
        }

        public struct WmoMaterialV17
        {
            public uint Flags;
            public uint Shader;
            public uint BlendMode;
            public uint Texture1Offset;
            public uint SidnColor;
            public uint FrameSidnColor;
            public uint Texture2Offset;
            public uint DiffColor;
            public uint GroundType;
            public uint Texture3Offset;
            public uint Color3;
            public uint Flags3;
            public byte[] RuntimeData;
        }

        public struct WmoGroupInfoV17
        {
            public uint Flags;
            public BoundingBox BoundingBox;
            public int NameOffset;
        }

        public struct BoundingBox
        {
            public Vector3 Min;
            public Vector3 Max;
        }

        #endregion
    }
}
