using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts WMO v14 (Alpha) to WMO v17 (LK 3.3.5) format.
/// Handles monolithic Alpha WMO → split root + group files.
/// </summary>
public class WmoV14ToV17Converter
{
    /// <summary>
    /// Convert a v14 WMO file to v17 format.
    /// </summary>
    public void Convert(string v14Path, string outputPath)
    {
        Console.WriteLine($"[INFO] Converting WMO v14 → v17: {Path.GetFileName(v14Path)}");

        using var fs = File.OpenRead(v14Path);
        using var reader = new BinaryReader(fs);

        var wmoData = ParseWmoV14(reader);

        // Write root file
        WriteRootFile(wmoData, outputPath);

        // Write group files
        WriteGroupFiles(wmoData, outputPath);

        Console.WriteLine($"[SUCCESS] Converted to v17: {outputPath}");
    }

    private WmoV14Data ParseWmoV14(BinaryReader reader)
    {
        var data = new WmoV14Data();

        // Read MVER
        var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "REVM") // Reversed
            throw new InvalidDataException($"Expected MVER, got {magic}");
        
        var size = reader.ReadUInt32();
        data.Version = reader.ReadUInt32();
        
        if (data.Version != 14)
            throw new InvalidDataException($"Expected WMO v14, got v{data.Version}");

        // Read MOMO container
        magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (magic != "OMOM") // Reversed MOMO
            throw new InvalidDataException($"Expected MOMO container, got {magic}");
        
        var momoSize = reader.ReadUInt32();
        var momoEnd = reader.BaseStream.Position + momoSize;

        // Parse chunks within MOMO
        while (reader.BaseStream.Position < momoEnd)
        {
            var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkMagic)
            {
                case "MOHD":
                    ParseMohd(reader, data);
                    break;
                case "MOTX":
                    ParseMotx(reader, chunkSize, data);
                    break;
                case "MOMT":
                    ParseMomt(reader, chunkSize, data);
                    break;
                case "MOGN":
                    ParseMogn(reader, chunkSize, data);
                    break;
                case "MOGI":
                    ParseMogi(reader, chunkSize, data);
                    break;
                case "MOGP":
                    ParseMogp(reader, chunkSize, data);
                    break;
                default:
                    // Skip unknown chunks
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        return data;
    }

    private void ParseMohd(BinaryReader reader, WmoV14Data data)
    {
        data.MaterialCount = reader.ReadUInt32();
        data.GroupCount = reader.ReadUInt32();
        data.PortalCount = reader.ReadUInt32();
        data.LightCount = reader.ReadUInt32();
        data.DoodadNameCount = reader.ReadUInt32();
        data.DoodadDefCount = reader.ReadUInt32();
        data.DoodadSetCount = reader.ReadUInt32();
        data.AmbientColor = reader.ReadUInt32();
        data.WmoId = reader.ReadUInt32();
        data.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        data.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        data.Flags = reader.ReadUInt32();
    }

    private void ParseMotx(BinaryReader reader, uint size, WmoV14Data data)
    {
        var bytes = reader.ReadBytes((int)size);
        data.Textures = ParseStringTable(bytes);
    }

    private void ParseMomt(BinaryReader reader, uint size, WmoV14Data data)
    {
        // v14 MOMT is 44 bytes per material
        int count = (int)(size / 44);
        data.Materials = new List<WmoMaterial>(count);

        for (int i = 0; i < count; i++)
        {
            var mat = new WmoMaterial
            {
                Flags = reader.ReadUInt32(),
                Shader = reader.ReadUInt32(),
                BlendMode = reader.ReadUInt32(),
                Texture1Offset = reader.ReadUInt32(),
                EmissiveColor = reader.ReadUInt32(),
                FrameEmissiveColor = reader.ReadUInt32(),
                Texture2Offset = reader.ReadUInt32(),
                DiffuseColor = reader.ReadUInt32(),
                GroundType = reader.ReadUInt32(),
                Texture3Offset = reader.ReadUInt32(),
                Color2 = reader.ReadUInt32()
            };
            data.Materials.Add(mat);
        }
    }

    private void ParseMogn(BinaryReader reader, uint size, WmoV14Data data)
    {
        var bytes = reader.ReadBytes((int)size);
        data.GroupNames = ParseStringTable(bytes);
    }

    private void ParseMogi(BinaryReader reader, uint size, WmoV14Data data)
    {
        // v14 MOGI is 40 bytes per group
        int count = (int)(size / 40);
        data.GroupInfos = new List<WmoGroupInfo>(count);

        for (int i = 0; i < count; i++)
        {
            var info = new WmoGroupInfo
            {
                Flags = reader.ReadUInt32(),
                BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()),
                NameOffset = reader.ReadInt32()
            };
            // Skip remaining bytes (40 - 32 = 8 bytes)
            reader.ReadBytes(8);
            data.GroupInfos.Add(info);
        }
    }

    private void ParseMogp(BinaryReader reader, uint size, WmoV14Data data)
    {
        var group = new WmoGroupData();
        var startPos = reader.BaseStream.Position;
        var endPos = startPos + size;

        // MOGP header
        group.NameOffset = reader.ReadUInt32();
        group.DescriptiveNameOffset = reader.ReadUInt32();
        group.Flags = reader.ReadUInt32();
        group.BoundsMin = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        group.BoundsMax = new Vector3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        group.PortalStart = reader.ReadUInt16();
        group.PortalCount = reader.ReadUInt16();
        group.TransBatchCount = reader.ReadUInt16();
        group.IntBatchCount = reader.ReadUInt16();
        group.ExtBatchCount = reader.ReadUInt16();
        reader.ReadUInt16(); // padding

        // Parse sub-chunks
        while (reader.BaseStream.Position < endPos - 8)
        {
            var chunkMagic = new string(reader.ReadChars(4).Reverse().ToArray());
            var chunkSize = reader.ReadUInt32();
            var chunkEnd = reader.BaseStream.Position + chunkSize;

            switch (chunkMagic)
            {
                case "MOPY":
                    // v14 MOPY is 4 bytes per face
                    int faceCount = (int)(chunkSize / 4);
                    group.FaceMaterials = new List<byte>(faceCount);
                    for (int i = 0; i < faceCount; i++)
                    {
                        reader.ReadByte(); // flags
                        group.FaceMaterials.Add(reader.ReadByte()); // material ID
                        reader.ReadBytes(2); // padding
                    }
                    break;

                case "MOVT":
                    int vertCount = (int)(chunkSize / 12);
                    group.Vertices = new List<Vector3>(vertCount);
                    for (int i = 0; i < vertCount; i++)
                    {
                        group.Vertices.Add(new Vector3(
                            reader.ReadSingle(),
                            reader.ReadSingle(),
                            reader.ReadSingle()));
                    }
                    break;

                case "MOIN": // v14 uses MOIN, not MOVI
                    int idxCount = (int)(chunkSize / 2);
                    group.Indices = new List<ushort>(idxCount);
                    for (int i = 0; i < idxCount; i++)
                    {
                        group.Indices.Add(reader.ReadUInt16());
                    }
                    break;

                case "MOTV":
                    int uvCount = (int)(chunkSize / 8);
                    group.UVs = new List<Vector2>(uvCount);
                    for (int i = 0; i < uvCount; i++)
                    {
                        group.UVs.Add(new Vector2(
                            reader.ReadSingle(),
                            reader.ReadSingle()));
                    }
                    break;

                case "MOBA":
                    // v14 MOBA is 24 bytes per batch
                    int batchCount = (int)(chunkSize / 24);
                    group.Batches = new List<WmoBatch>(batchCount);
                    for (int i = 0; i < batchCount; i++)
                    {
                        var batch = new WmoBatch();
                        reader.ReadBytes(12); // bounding box stuff
                        batch.FirstFace = reader.ReadUInt32();
                        batch.NumFaces = reader.ReadUInt16();
                        batch.FirstVertex = reader.ReadUInt16();
                        batch.LastVertex = reader.ReadUInt16();
                        batch.Flags = reader.ReadByte();
                        batch.MaterialId = reader.ReadByte();
                        group.Batches.Add(batch);
                    }
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        data.Groups.Add(group);
    }

    private void WriteRootFile(WmoV14Data data, string outputPath)
    {
        var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
        Directory.CreateDirectory(outputDir);

        using var stream = File.Create(outputPath);
        using var writer = new BinaryWriter(stream);

        // MVER (version 17)
        WriteChunk(writer, "MVER", w => w.Write((uint)17));

        // MOHD
        WriteChunk(writer, "MOHD", w =>
        {
            w.Write(data.MaterialCount);
            w.Write((uint)data.Groups.Count);
            w.Write((uint)0); // portals
            w.Write((uint)0); // lights
            w.Write((uint)0); // models
            w.Write((uint)0); // doodads
            w.Write((uint)1); // doodad sets
            w.Write(data.AmbientColor);
            w.Write(data.WmoId);
            WriteVector3(w, data.BoundsMin);
            WriteVector3(w, data.BoundsMax);
            w.Write((ushort)data.Flags);
            w.Write((ushort)0); // LOD
        });

        // MOTX
        WriteChunk(writer, "MOTX", w =>
        {
            foreach (var tex in data.Textures)
            {
                var bytes = Encoding.UTF8.GetBytes(tex);
                w.Write(bytes);
                w.Write((byte)0);
            }
        });

        // MOMT (expand to v17 64-byte format)
        WriteChunk(writer, "MOMT", w =>
        {
            foreach (var mat in data.Materials)
            {
                w.Write(mat.Flags);
                w.Write(mat.Shader);
                w.Write(mat.BlendMode);
                w.Write(mat.Texture1Offset);
                w.Write(mat.EmissiveColor);
                w.Write(mat.FrameEmissiveColor);
                w.Write(mat.Texture2Offset);
                w.Write(mat.DiffuseColor);
                w.Write(mat.GroundType);
                w.Write(mat.Texture3Offset);
                w.Write(mat.Color2);
                w.Write((uint)0); // flags2
                w.Write(new byte[16]); // runtime data
            }
        });

        // MOGN
        WriteChunk(writer, "MOGN", w =>
        {
            foreach (var name in data.GroupNames)
            {
                var bytes = Encoding.UTF8.GetBytes(name);
                w.Write(bytes);
                w.Write((byte)0);
            }
        });

        // MOGI (v17 is 32 bytes)
        WriteChunk(writer, "MOGI", w =>
        {
            foreach (var info in data.GroupInfos)
            {
                w.Write(info.Flags);
                WriteVector3(w, info.BoundsMin);
                WriteVector3(w, info.BoundsMax);
                w.Write(info.NameOffset);
            }
        });

        // MODS (minimal doodad set)
        WriteChunk(writer, "MODS", w =>
        {
            var setName = Encoding.UTF8.GetBytes("Set_$DefaultGlobal");
            w.Write(setName);
            w.Write(new byte[20 - setName.Length]); // pad to 20
            w.Write((uint)0); // firstIndex
            w.Write((uint)0); // count
            w.Write((uint)0); // unused
        });

        Console.WriteLine($"[DEBUG] Wrote v17 root: {data.Groups.Count} groups, {data.Materials.Count} materials");
    }

    private void WriteGroupFiles(WmoV14Data data, string rootPath)
    {
        var baseName = Path.GetFileNameWithoutExtension(rootPath);
        var outputDir = Path.GetDirectoryName(rootPath) ?? ".";

        for (int i = 0; i < data.Groups.Count; i++)
        {
            var group = data.Groups[i];
            var groupPath = Path.Combine(outputDir, $"{baseName}_{i:D3}.wmo");
            WriteGroupFile(group, groupPath, i);
        }
    }

    private void WriteGroupFile(WmoGroupData group, string outputPath, int groupIndex)
    {
        using var stream = File.Create(outputPath);
        using var writer = new BinaryWriter(stream);

        // MVER
        WriteChunk(writer, "MVER", w => w.Write((uint)17));

        // MOGP (header + subchunks)
        var mogpStart = writer.BaseStream.Position;
        writer.Write(Encoding.ASCII.GetBytes("PGOM")); // Reversed
        var sizePos = writer.BaseStream.Position;
        writer.Write((uint)0); // Size placeholder

        var mogpDataStart = writer.BaseStream.Position;

        // MOGP header (68 bytes)
        writer.Write(group.NameOffset);
        writer.Write(group.DescriptiveNameOffset);
        writer.Write(group.Flags);
        WriteVector3(writer, group.BoundsMin);
        WriteVector3(writer, group.BoundsMax);
        writer.Write(group.PortalStart);
        writer.Write(group.PortalCount);
        writer.Write(group.TransBatchCount);
        writer.Write(group.IntBatchCount);
        writer.Write(group.ExtBatchCount);
        writer.Write((ushort)0); // padding
        writer.Write(new byte[4]); // fog indices
        writer.Write((uint)0); // liquid type
        writer.Write((uint)0); // group ID
        writer.Write((uint)0); // unused
        writer.Write((uint)0); // unused

        // MOPY (v17 is 2 bytes per face)
        WriteSubChunk(writer, "MOPY", w =>
        {
            foreach (var matId in group.FaceMaterials)
            {
                w.Write((byte)0); // flags
                w.Write(matId);
            }
        });

        // MOVI (indices)
        WriteSubChunk(writer, "MOVI", w =>
        {
            foreach (var idx in group.Indices)
                w.Write(idx);
        });

        // MOVT (vertices)
        WriteSubChunk(writer, "MOVT", w =>
        {
            foreach (var v in group.Vertices)
                WriteVector3(w, v);
        });

        // MONR (normals - generate)
        WriteSubChunk(writer, "MONR", w =>
        {
            var normals = GenerateNormals(group);
            foreach (var n in normals)
                WriteVector3(w, n);
        });

        // MOTV (UVs)
        WriteSubChunk(writer, "MOTV", w =>
        {
            foreach (var uv in group.UVs)
            {
                w.Write(uv.X);
                w.Write(uv.Y);
            }
        });

        // MOBA (batches - v17 is 24 bytes)
        WriteSubChunk(writer, "MOBA", w =>
        {
            foreach (var batch in group.Batches)
            {
                w.Write((ushort)0); w.Write((ushort)0); w.Write((ushort)0);
                w.Write((ushort)0); w.Write((ushort)0); w.Write((ushort)0);
                w.Write(batch.FirstFace);
                w.Write(batch.NumFaces);
                w.Write(batch.FirstVertex);
                w.Write(batch.LastVertex);
                w.Write(batch.Flags);
                w.Write(batch.MaterialId);
            }
        });

        // Update MOGP size
        var mogpEnd = writer.BaseStream.Position;
        writer.BaseStream.Position = sizePos;
        writer.Write((uint)(mogpEnd - mogpDataStart));
        writer.BaseStream.Position = mogpEnd;

        Console.WriteLine($"[DEBUG] Wrote group {groupIndex}: {group.Vertices.Count} verts, {group.Indices.Count} indices");
    }

    private List<string> ParseStringTable(byte[] data)
    {
        var result = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                if (i > start)
                {
                    var str = Encoding.UTF8.GetString(data, start, i - start);
                    if (!string.IsNullOrEmpty(str))
                        result.Add(str);
                }
                start = i + 1;
            }
        }
        return result;
    }

    private List<Vector3> GenerateNormals(WmoGroupData group)
    {
        var normals = new Vector3[group.Vertices.Count];
        for (int i = 0; i < normals.Length; i++)
            normals[i] = Vector3.Zero;

        for (int i = 0; i + 2 < group.Indices.Count; i += 3)
        {
            var i0 = group.Indices[i];
            var i1 = group.Indices[i + 1];
            var i2 = group.Indices[i + 2];

            if (i0 >= group.Vertices.Count || i1 >= group.Vertices.Count || i2 >= group.Vertices.Count)
                continue;

            var e1 = group.Vertices[i1] - group.Vertices[i0];
            var e2 = group.Vertices[i2] - group.Vertices[i0];
            var normal = Vector3.Normalize(Vector3.Cross(e1, e2));

            normals[i0] += normal;
            normals[i1] += normal;
            normals[i2] += normal;
        }

        return normals.Select(n => n.Length() > 0.001f ? Vector3.Normalize(n) : Vector3.UnitY).ToList();
    }

    private void WriteChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
    {
        var reversed = new string(chunkId.Reverse().ToArray());
        writer.Write(Encoding.ASCII.GetBytes(reversed));
        var sizePos = writer.BaseStream.Position;
        writer.Write((uint)0);
        var dataStart = writer.BaseStream.Position;
        writeData(writer);
        var dataEnd = writer.BaseStream.Position;
        writer.BaseStream.Position = sizePos;
        writer.Write((uint)(dataEnd - dataStart));
        writer.BaseStream.Position = dataEnd;
    }

    private void WriteSubChunk(BinaryWriter writer, string chunkId, Action<BinaryWriter> writeData)
        => WriteChunk(writer, chunkId, writeData);

    private void WriteVector3(BinaryWriter writer, Vector3 v)
    {
        writer.Write(v.X);
        writer.Write(v.Y);
        writer.Write(v.Z);
    }

    #region Data Structures

    public class WmoV14Data
    {
        public uint Version;
        public uint MaterialCount, GroupCount, PortalCount, LightCount;
        public uint DoodadNameCount, DoodadDefCount, DoodadSetCount;
        public uint AmbientColor, WmoId, Flags;
        public Vector3 BoundsMin, BoundsMax;
        public List<string> Textures = new();
        public List<WmoMaterial> Materials = new();
        public List<string> GroupNames = new();
        public List<WmoGroupInfo> GroupInfos = new();
        public List<WmoGroupData> Groups = new();
    }

    public struct WmoMaterial
    {
        public uint Flags, Shader, BlendMode;
        public uint Texture1Offset, EmissiveColor, FrameEmissiveColor;
        public uint Texture2Offset, DiffuseColor, GroundType;
        public uint Texture3Offset, Color2;
    }

    public struct WmoGroupInfo
    {
        public uint Flags;
        public Vector3 BoundsMin, BoundsMax;
        public int NameOffset;
    }

    public class WmoGroupData
    {
        public uint NameOffset, DescriptiveNameOffset, Flags;
        public Vector3 BoundsMin, BoundsMax;
        public ushort PortalStart, PortalCount;
        public ushort TransBatchCount, IntBatchCount, ExtBatchCount;
        public List<Vector3> Vertices = new();
        public List<ushort> Indices = new();
        public List<Vector2> UVs = new();
        public List<byte> FaceMaterials = new();
        public List<WmoBatch> Batches = new();
    }

    public struct WmoBatch
    {
        public uint FirstFace;
        public ushort NumFaces, FirstVertex, LastVertex;
        public byte Flags, MaterialId;
    }

    #endregion
}
