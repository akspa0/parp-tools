using System.Numerics;
using System.Text;

namespace WoWMapConverter.Core.Converters;

/// <summary>
/// Converts WMO v17 (LK 3.3.5) to WMO v14 (Alpha) format.
/// Handles split root + group files → monolithic Alpha WMO.
/// </summary>
public class WmoV17ToV14Converter
{
    private static readonly string[] RootRequiredChunkOrder090 =
    {
        "MVER", "MOHD", "MOTX", "MOMT", "MOGN", "MOGI", "MOSB", "MOPV", "MOPT", "MOPR",
        "MOVV", "MOVB", "MOLT", "MODS", "MODN", "MODD", "MFOG"
    };

    private static readonly string[] GroupRequiredChunkOrder090 =
    {
        "MOPY", "MOVI", "MOVT", "MONR", "MOTV", "MOBA"
    };

    /// <summary>
    /// Convert a v17 WMO (root + groups) to v14 monolithic format.
    /// </summary>
    public void Convert(string v17RootPath, string outputPath)
    {
        Console.WriteLine($"[INFO] Converting WMO v17 → v14: {Path.GetFileName(v17RootPath)}");

        // Parse root file
        var rootData = ParseWmoV17Root(v17RootPath);

        // Parse group files
        var groupsDir = Path.GetDirectoryName(v17RootPath) ?? ".";
        var baseName = Path.GetFileNameWithoutExtension(v17RootPath);
        
        for (int i = 0; i < rootData.GroupCount; i++)
        {
            var groupPath = Path.Combine(groupsDir, $"{baseName}_{i:D3}.wmo");
            if (File.Exists(groupPath))
            {
                var groupData = ParseWmoV17Group(groupPath);
                rootData.Groups.Add(groupData);
            }
        }

        // Write monolithic v14 WMO
        WriteWmoV14(rootData, outputPath);

        Console.WriteLine($"[SUCCESS] Converted to v14: {outputPath}");
    }

    /// <summary>
    /// Convert v17 WMO bytes to v14 bytes in memory.
    /// </summary>
    public byte[] ConvertToBytes(byte[] v17RootBytes, List<byte[]> groupBytes)
    {
        using var rootStream = new MemoryStream(v17RootBytes);
        using var reader = new BinaryReader(rootStream);
        
        var rootData = ParseWmoV17RootFromReader(reader);

        foreach (var gb in groupBytes)
        {
            using var groupStream = new MemoryStream(gb);
            using var groupReader = new BinaryReader(groupStream);
            rootData.Groups.Add(ParseWmoV17GroupFromReader(groupReader));
        }

        using var output = new MemoryStream();
        WriteWmoV14ToStream(rootData, output);
        return output.ToArray();
    }

    /// <summary>
    /// Parse v17 WMO root + group bytes directly into WmoV14Data model
    /// without the lossy binary serialization roundtrip.
    /// </summary>
    public WmoV14ToV17Converter.WmoV14Data ParseV17ToModel(byte[] rootBytes, List<byte[]> groupBytesList)
    {
        using var rootStream = new MemoryStream(rootBytes);
        using var reader = new BinaryReader(rootStream);
        var v17 = ParseWmoV17RootFromReader(reader);

        foreach (var gb in groupBytesList)
        {
            using var groupStream = new MemoryStream(gb);
            using var groupReader = new BinaryReader(groupStream);
            v17.Groups.Add(ParseWmoV17GroupFromReader(groupReader));
        }

        return ConvertV17DataToModel(v17);
    }

    /// <summary>
    /// Resolve a null-terminated string at a byte offset within a raw string table blob.
    /// Returns empty string if offset is out of range.
    /// </summary>
    private static string ResolveStringAtOffset(byte[] rawBlob, uint offset)
    {
        if (rawBlob == null || offset >= rawBlob.Length)
            return "";
        int end = Array.IndexOf(rawBlob, (byte)0, (int)offset);
        if (end < 0) end = rawBlob.Length;
        return Encoding.ASCII.GetString(rawBlob, (int)offset, end - (int)offset);
    }

    /// <summary>
    /// Convert parsed v17 data structures directly to WmoV14Data model.
    /// </summary>
    private static WmoV14ToV17Converter.WmoV14Data ConvertV17DataToModel(WmoV17Data v17)
    {
        var model = new WmoV14ToV17Converter.WmoV14Data
        {
            Version = v17.Version,
            MaterialCount = v17.MaterialCount,
            GroupCount = v17.GroupCount,
            PortalCount = v17.PortalCount,
            LightCount = v17.LightCount,
            DoodadNameCount = v17.DoodadNameCount,
            DoodadDefCount = v17.DoodadDefCount,
            DoodadSetCount = v17.DoodadSetCount,
            AmbientColor = v17.AmbientColor,
            WmoId = v17.WmoId,
            Flags = v17.Flags,
            BoundsMin = v17.BoundingBox1,
            BoundsMax = v17.BoundingBox2,
        };

        // Textures
        model.Textures.AddRange(v17.TextureNames);
        // Store raw MOTX for the v14 model (WmoRenderer uses it for offset resolution)
        model.MotxRaw = v17.TextureNamesRaw;

        // Materials — resolve texture names via byte offset into raw MOTX blob
        foreach (var m in v17.Materials)
        {
            string tex1 = ResolveStringAtOffset(v17.TextureNamesRaw, m.Texture1);
            string tex2 = ResolveStringAtOffset(v17.TextureNamesRaw, m.Texture2);
            string tex3 = ResolveStringAtOffset(v17.TextureNamesRaw, m.Texture3);
            model.Materials.Add(new WmoV14ToV17Converter.WmoMaterial
            {
                Flags = m.Flags,
                Shader = m.Shader,
                BlendMode = m.BlendMode,
                Texture1Offset = m.Texture1,
                EmissiveColor = m.SidnColor,
                FrameEmissiveColor = m.FrameSidnColor,
                Texture2Offset = m.Texture2,
                DiffuseColor = m.DiffColor,
                GroundType = m.GroundType,
                Texture3Offset = m.Texture3,
                Color2 = m.Color3,
                Texture1Name = tex1,
                Texture2Name = tex2,
                Texture3Name = tex3,
            });
        }

        // Group names and infos
        model.GroupNames.AddRange(v17.GroupNames);
        foreach (var gi in v17.GroupInfos)
        {
            model.GroupInfos.Add(new WmoV14ToV17Converter.WmoGroupInfo
            {
                Flags = gi.Flags,
                BoundsMin = gi.BoundingBox1,
                BoundsMax = gi.BoundingBox2,
                NameOffset = gi.NameOfs,
            });
        }

        // Portal vertices
        model.PortalVertices.AddRange(v17.PortalVertices);

        // Portal infos
        foreach (var pi in v17.PortalInfos)
        {
            model.Portals.Add(new WmoV14ToV17Converter.WmoPortal
            {
                StartVertex = pi.StartVertex,
                Count = pi.VertexCount,
                PlaneA = pi.Normal.X,
                PlaneB = pi.Normal.Y,
                PlaneC = pi.Normal.Z,
                PlaneD = pi.Distance,
            });
        }

        // Portal refs
        foreach (var pr in v17.PortalRefs)
        {
            model.PortalRefs.Add(new WmoV14ToV17Converter.WmoPortalRef
            {
                PortalIndex = pr.PortalIndex,
                GroupIndex = pr.GroupIndex,
                Side = pr.Side,
            });
        }

        // Lights
        foreach (var l in v17.Lights)
        {
            model.Lights.Add(new WmoV14ToV17Converter.WmoLight
            {
                Type = l.Type,
                UseAtten = l.UseAtten != 0,
                Color = l.Color,
                Position = l.Position,
                Intensity = l.Intensity,
                AttenStart = l.AttenStart,
                AttenEnd = l.AttenEnd,
            });
        }

        // Doodad sets
        foreach (var ds in v17.DoodadSets)
        {
            model.DoodadSets.Add(new WmoV14ToV17Converter.WmoDoodadSet
            {
                Name = ds.Name,
                StartIndex = ds.StartIndex,
                Count = ds.Count,
            });
        }

        // Doodad names → use raw MODN blob directly (preserves byte offsets)
        model.DoodadNamesRaw = v17.DoodadNamesRaw;

        // Doodad defs
        foreach (var dd in v17.DoodadDefs)
        {
            model.DoodadDefs.Add(new WmoV14ToV17Converter.WmoDoodadDef
            {
                NameIndex = dd.NameOfs,
                Position = dd.Position,
                Orientation = new Quaternion(dd.Rotation[0], dd.Rotation[1], dd.Rotation[2], dd.Rotation[3]),
                Scale = dd.Scale,
                Color = dd.Color,
            });
        }

        // Groups
        foreach (var g in v17.Groups)
        {
            var group = new WmoV14ToV17Converter.WmoGroupData
            {
                NameOffset = g.GroupNameOfs,
                DescriptiveNameOffset = g.DescriptiveNameOfs,
                Flags = g.Flags,
                BoundsMin = g.BoundingBox1,
                BoundsMax = g.BoundingBox2,
                PortalStart = g.PortalStart,
                PortalCount = g.PortalCount,
                TransBatchCount = g.TransBatchCount,
                IntBatchCount = g.IntBatchCount,
                ExtBatchCount = g.ExtBatchCount,
                GroupLiquid = g.GroupLiquid,
            };

            // Resolve group name from raw MOGN blob via byte offset
            string resolved = ResolveStringAtOffset(v17.GroupNamesRaw, g.GroupNameOfs);
            group.Name = !string.IsNullOrEmpty(resolved) ? resolved : $"group_{v17.Groups.IndexOf(g)}";

            if (g.Vertices != null)
                group.Vertices.AddRange(g.Vertices);
            if (g.Indices != null)
                group.Indices.AddRange(g.Indices);
            if (g.TexCoords != null)
                group.UVs.AddRange(g.TexCoords);
            if (g.Normals != null)
                group.Normals.AddRange(g.Normals);
            if (g.DoodadRefs != null)
                group.DoodadRefs.AddRange(g.DoodadRefs);
            if (g.VertexColors != null)
            {
                for (int i = 0; i + 3 < g.VertexColors.Length; i += 4)
                    group.VertexColors.Add(BitConverter.ToUInt32(g.VertexColors, i));
            }
            if (g.LiquidData != null)
                group.LiquidData = g.LiquidData;

            // Material info → FaceMaterials (1 byte per face: material ID)
            if (g.MaterialInfo != null)
            {
                for (int i = 0; i + 1 < g.MaterialInfo.Length; i += 2)
                    group.FaceMaterials.Add(g.MaterialInfo[i + 1]); // byte[1] = materialId
            }

            // Batches
            if (g.Batches != null)
            {
                foreach (var b in g.Batches)
                {
                    var bbRaw = new byte[12];
                    for (int i = 0; i < 6 && i < b.BoundingBox.Length; i++)
                    {
                        bbRaw[i * 2] = (byte)(b.BoundingBox[i] & 0xFF);
                        bbRaw[i * 2 + 1] = (byte)((b.BoundingBox[i] >> 8) & 0xFF);
                    }
                    group.Batches.Add(new WmoV14ToV17Converter.WmoBatch
                    {
                        BoundingBoxRaw = bbRaw,
                        FirstIndex = b.StartIndex,
                        IndexCount = (ushort)b.IndexCount,
                        FirstVertex = b.StartVertex,
                        LastVertex = b.EndVertex,
                        Flags = b.Flags,
                        MaterialId = b.MaterialId,
                    });
                }
            }

            model.Groups.Add(group);
        }

        return model;
    }

    private WmoV17Data ParseWmoV17Root(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return ParseWmoV17RootFromReader(reader);
    }

    private WmoV17Data ParseWmoV17RootFromReader(BinaryReader reader)
    {
        var data = new WmoV17Data();
        int expectedRootChunkIndex = 0;
        bool sawOptionalMcvp = false;

        while (reader.BaseStream.Position + 8 <= reader.BaseStream.Length)
        {
            var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var size = reader.ReadUInt32();
            long payloadStart = reader.BaseStream.Position;
            var chunkEnd = Math.Min(payloadStart + size, reader.BaseStream.Length);
            uint effectiveSize = (uint)Math.Max(0, chunkEnd - payloadStart);

            // Reverse for comparison (chunks stored reversed on disk)
            var chunkId = new string(magic.Reverse().ToArray());

            if (expectedRootChunkIndex < RootRequiredChunkOrder090.Length)
            {
                string expected = RootRequiredChunkOrder090[expectedRootChunkIndex];
                if (chunkId != expected)
                    throw new InvalidDataException(
                        $"WMO root chunk order mismatch at offset 0x{payloadStart - 8:X}: got '{chunkId}', expected '{expected}'.");
                expectedRootChunkIndex++;
            }
            else
            {
                if (chunkId == "MCVP" && !sawOptionalMcvp)
                {
                    sawOptionalMcvp = true;
                }
                else
                {
                    throw new InvalidDataException(
                        $"Unexpected trailing WMO root chunk '{chunkId}' at offset 0x{payloadStart - 8:X}.");
                }
            }

            switch (chunkId)
            {
                case "MVER":
                    if (effectiveSize < 4)
                        throw new InvalidDataException("Invalid WMO MVER chunk size: expected at least 4 bytes.");

                    data.Version = reader.ReadUInt32();
                    if (data.Version != 0x11)
                        throw new InvalidDataException($"Unsupported WMO version 0x{data.Version:X}; expected 0x11 for 0.9.x profile.");
                    break;
                case "MOHD":
                    ParseMohd(reader, effectiveSize, data);
                    break;
                case "MOTX":
                    {
                        long pos = reader.BaseStream.Position;
                        data.TextureNamesRaw = reader.ReadBytes((int)effectiveSize);
                        reader.BaseStream.Position = pos;
                        data.TextureNames = ReadStringTable(reader, effectiveSize);
                    }
                    break;
                case "MOMT":
                    data.Materials = ReadMaterials(reader, effectiveSize);
                    break;
                case "MOGN":
                    {
                        long pos = reader.BaseStream.Position;
                        data.GroupNamesRaw = reader.ReadBytes((int)effectiveSize);
                        reader.BaseStream.Position = pos;
                        data.GroupNames = ReadStringTable(reader, effectiveSize);
                    }
                    break;
                case "MOGI":
                    data.GroupInfos = ReadGroupInfos(reader, effectiveSize);
                    break;
                case "MOSB":
                    data.SkyboxName = ReadNullTermString(reader, effectiveSize);
                    break;
                case "MOPV":
                    data.PortalVertices = ReadVector3Array(reader, effectiveSize);
                    break;
                case "MOPT":
                    data.PortalInfos = ReadPortalInfos(reader, effectiveSize);
                    break;
                case "MOPR":
                    data.PortalRefs = ReadPortalRefs(reader, effectiveSize);
                    break;
                case "MOVV":
                    data.MovvRaw = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MOVB":
                    data.MovbRaw = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MOLT":
                    data.Lights = ReadLights(reader, effectiveSize);
                    break;
                case "MODS":
                    data.DoodadSets = ReadDoodadSets(reader, effectiveSize);
                    break;
                case "MODN":
                    {
                        long pos = reader.BaseStream.Position;
                        data.DoodadNamesRaw = reader.ReadBytes((int)effectiveSize);
                        reader.BaseStream.Position = pos;
                        data.DoodadNames = ReadStringTable(reader, effectiveSize);
                    }
                    break;
                case "MODD":
                    data.DoodadDefs = ReadDoodadDefs(reader, effectiveSize);
                    break;
                case "MFOG":
                    data.Fogs = ReadFogs(reader, effectiveSize);
                    break;
                case "MCVP":
                    data.McvpRaw = reader.ReadBytes((int)effectiveSize);
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        if (expectedRootChunkIndex != RootRequiredChunkOrder090.Length)
        {
            throw new InvalidDataException(
                $"WMO root terminated early: parsed {expectedRootChunkIndex}/{RootRequiredChunkOrder090.Length} required chunks.");
        }

        return data;
    }

    private WmoV17GroupData ParseWmoV17Group(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        return ParseWmoV17GroupFromReader(reader);
    }

    private WmoV17GroupData ParseWmoV17GroupFromReader(BinaryReader reader)
    {
        var data = new WmoV17GroupData();
        int expectedTopChunkIndex = 0;

        while (reader.BaseStream.Position + 8 <= reader.BaseStream.Length)
        {
            var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var size = reader.ReadUInt32();
            long payloadStart = reader.BaseStream.Position;
            var chunkEnd = Math.Min(payloadStart + size, reader.BaseStream.Length);
            uint effectiveSize = (uint)Math.Max(0, chunkEnd - payloadStart);
            var chunkId = new string(magic.Reverse().ToArray());

            if (expectedTopChunkIndex == 0)
            {
                if (chunkId != "MVER")
                    throw new InvalidDataException(
                        $"WMO group chunk order mismatch at offset 0x{payloadStart - 8:X}: expected 'MVER', got '{chunkId}'.");
                expectedTopChunkIndex = 1;
            }
            else if (expectedTopChunkIndex == 1)
            {
                if (chunkId != "MOGP")
                    throw new InvalidDataException(
                        $"WMO group chunk order mismatch at offset 0x{payloadStart - 8:X}: expected 'MOGP', got '{chunkId}'.");
                expectedTopChunkIndex = 2;
            }
            else
            {
                throw new InvalidDataException(
                    $"Unexpected trailing WMO group chunk '{chunkId}' at offset 0x{payloadStart - 8:X}.");
            }

            switch (chunkId)
            {
                case "MVER":
                    if (effectiveSize < 4)
                        throw new InvalidDataException("Invalid WMO group MVER chunk size: expected at least 4 bytes.");

                    data.Version = reader.ReadUInt32();
                    if (data.Version != 0x10 && data.Version != 0x11)
                        throw new InvalidDataException($"Unsupported WMO group version 0x{data.Version:X}; expected 0x10 or 0x11.");
                    break;
                case "MOGP":
                    ParseMogp(reader, effectiveSize, data);
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        if (expectedTopChunkIndex != 2)
            throw new InvalidDataException("WMO group terminated early: expected MVER followed by MOGP.");

        return data;
    }

    private void ParseMohd(BinaryReader reader, uint size, WmoV17Data data)
    {
        long remaining = Math.Min(size, reader.BaseStream.Length - reader.BaseStream.Position);
        if (remaining < 60) return; // Need at least 60 bytes for core MOHD fields
        data.MaterialCount = reader.ReadUInt32();
        data.GroupCount = reader.ReadUInt32();
        data.PortalCount = reader.ReadUInt32();
        data.LightCount = reader.ReadUInt32();
        data.DoodadNameCount = reader.ReadUInt32();
        data.DoodadDefCount = reader.ReadUInt32();
        data.DoodadSetCount = reader.ReadUInt32();
        data.AmbientColor = reader.ReadUInt32();
        data.WmoId = reader.ReadUInt32();
        data.BoundingBox1 = ReadVector3(reader);
        data.BoundingBox2 = ReadVector3(reader);
        if (remaining >= 64)
            data.Flags = reader.ReadUInt32();
    }

    private void ParseMogp(BinaryReader reader, uint totalSize, WmoV17GroupData data)
    {
        long mogpStart = reader.BaseStream.Position;
        long mogpEnd = Math.Min(mogpStart + totalSize, reader.BaseStream.Length);
        
        // MOGP header (68 bytes in v17)
        if (mogpEnd - mogpStart < 68) return;
        data.GroupNameOfs = reader.ReadUInt32();
        data.DescriptiveNameOfs = reader.ReadUInt32();
        data.Flags = reader.ReadUInt32();
        data.BoundingBox1 = ReadVector3(reader);
        data.BoundingBox2 = ReadVector3(reader);
        data.PortalStart = reader.ReadUInt16();
        data.PortalCount = reader.ReadUInt16();
        data.TransBatchCount = reader.ReadUInt16();
        data.IntBatchCount = reader.ReadUInt16();
        data.ExtBatchCount = reader.ReadUInt16();
        reader.ReadUInt16(); // padding
        data.FogIndices = new byte[4];
        reader.Read(data.FogIndices, 0, 4);
        data.GroupLiquid = reader.ReadUInt32();
        data.UniqueId = reader.ReadUInt32();
        data.Flags2 = reader.ReadUInt32();
        reader.ReadUInt32(); // unused

        // Parse subchunks within MOGP:
        // - required chain is strict and ordered
        // - optional flagged chunks are strict-per-token but order-tolerant
        int expectedRequiredSubchunkIndex = 0;
        var allowedOptionalSubchunks = BuildAllowedOptionalMogpSubchunks(data.Flags);
        var seenOptionalSubchunks = new HashSet<string>(StringComparer.Ordinal);
        while (reader.BaseStream.Position + 8 <= mogpEnd)
        {
            var magic = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var size = reader.ReadUInt32();
            long payloadStart = reader.BaseStream.Position;
            var chunkEnd = Math.Min(payloadStart + size, mogpEnd);
            uint effectiveSize = (uint)Math.Max(0, chunkEnd - payloadStart);
            var chunkId = new string(magic.Reverse().ToArray());

            if (expectedRequiredSubchunkIndex < GroupRequiredChunkOrder090.Length)
            {
                string expectedSubchunk = GroupRequiredChunkOrder090[expectedRequiredSubchunkIndex];
                if (chunkId != expectedSubchunk)
                {
                    throw new InvalidDataException(
                        $"MOGP required subchunk order mismatch at 0x{payloadStart - 8:X}: got '{chunkId}', expected '{expectedSubchunk}'. Flags=0x{data.Flags:X}.");
                }
                expectedRequiredSubchunkIndex++;
            }
            else
            {
                if (Array.IndexOf(GroupRequiredChunkOrder090, chunkId) >= 0)
                {
                    throw new InvalidDataException(
                        $"Unexpected repeated required MOGP subchunk '{chunkId}' at 0x{payloadStart - 8:X}. Flags=0x{data.Flags:X}.");
                }

                if (!allowedOptionalSubchunks.Contains(chunkId))
                {
                    throw new InvalidDataException(
                        $"Unexpected ungated MOGP optional subchunk '{chunkId}' at 0x{payloadStart - 8:X}. Flags=0x{data.Flags:X}.");
                }

                if (!seenOptionalSubchunks.Add(chunkId))
                {
                    throw new InvalidDataException(
                        $"Duplicate MOGP optional subchunk '{chunkId}' at 0x{payloadStart - 8:X}. Flags=0x{data.Flags:X}.");
                }

                ValidateOptionalMogpDependency(chunkId, seenOptionalSubchunks, payloadStart, data.Flags);
            }

            switch (chunkId)
            {
                case "MOPY":
                    data.MaterialInfo = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MOVI":
                    data.Indices = ReadUInt16Array(reader, effectiveSize);
                    break;
                case "MOVT":
                    data.Vertices = ReadVector3Array(reader, effectiveSize);
                    break;
                case "MONR":
                    data.Normals = ReadVector3Array(reader, effectiveSize);
                    break;
                case "MOTV":
                    data.TexCoords = ReadVector2Array(reader, effectiveSize);
                    break;
                case "MOBA":
                    data.Batches = ReadBatches(reader, effectiveSize);
                    break;
                case "MOLR":
                    data.LightRefs = ReadUInt16Array(reader, effectiveSize);
                    break;
                case "MPBV":
                    data.MpbvData = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MPBP":
                    data.MpbpData = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MPBI":
                    data.MpbiData = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MPBG":
                    data.MpbgData = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MODR":
                    data.DoodadRefs = ReadUInt16Array(reader, effectiveSize);
                    break;
                case "MOBN":
                    data.BspNodes = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MOBR":
                    data.BspFaceIndices = ReadUInt16Array(reader, effectiveSize);
                    break;
                case "MOCV":
                    data.VertexColors = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MLIQ":
                    data.LiquidData = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MORI":
                    data.MoriData = reader.ReadBytes((int)effectiveSize);
                    break;
                case "MORB":
                    data.MorbData = reader.ReadBytes((int)effectiveSize);
                    break;
            }

            reader.BaseStream.Position = chunkEnd;
        }

        if (expectedRequiredSubchunkIndex != GroupRequiredChunkOrder090.Length)
        {
            throw new InvalidDataException(
                $"MOGP missing required subchunks: parsed {expectedRequiredSubchunkIndex}/{GroupRequiredChunkOrder090.Length}. Flags=0x{data.Flags:X}.");
        }

        EnsureAllFlaggedOptionalMogpSubchunksPresent(seenOptionalSubchunks, data.Flags);
    }

    private static HashSet<string> BuildAllowedOptionalMogpSubchunks(uint flags)
    {
        var optional = new HashSet<string>(StringComparer.Ordinal);

        if ((flags & 0x0001) != 0)
        {
            optional.Add("MOBN");
            optional.Add("MOBR");
        }

        if ((flags & 0x0004) != 0)
            optional.Add("MOCV");
        if ((flags & 0x0200) != 0)
            optional.Add("MOLR");

        if ((flags & 0x0400) != 0)
        {
            optional.Add("MPBV");
            optional.Add("MPBP");
            optional.Add("MPBI");
            optional.Add("MPBG");
        }

        if ((flags & 0x0800) != 0)
            optional.Add("MODR");
        if ((flags & 0x1000) != 0)
            optional.Add("MLIQ");

        if ((flags & 0x20000) != 0)
        {
            optional.Add("MORI");
            optional.Add("MORB");
        }

        return optional;
    }

    private static void ValidateOptionalMogpDependency(string chunkId, HashSet<string> seenOptionalSubchunks, long payloadStart, uint flags)
    {
        if (chunkId == "MOBR" && !seenOptionalSubchunks.Contains("MOBN"))
        {
            throw new InvalidDataException(
                $"MOGP optional dependency mismatch at 0x{payloadStart - 8:X}: 'MOBR' requires prior 'MOBN'. Flags=0x{flags:X}.");
        }

        if (chunkId == "MPBP" && !seenOptionalSubchunks.Contains("MPBV"))
        {
            throw new InvalidDataException(
                $"MOGP optional dependency mismatch at 0x{payloadStart - 8:X}: 'MPBP' requires prior 'MPBV'. Flags=0x{flags:X}.");
        }

        if (chunkId == "MPBI" && !seenOptionalSubchunks.Contains("MPBP"))
        {
            throw new InvalidDataException(
                $"MOGP optional dependency mismatch at 0x{payloadStart - 8:X}: 'MPBI' requires prior 'MPBP'. Flags=0x{flags:X}.");
        }

        if (chunkId == "MPBG" && !seenOptionalSubchunks.Contains("MPBI"))
        {
            throw new InvalidDataException(
                $"MOGP optional dependency mismatch at 0x{payloadStart - 8:X}: 'MPBG' requires prior 'MPBI'. Flags=0x{flags:X}.");
        }

        if (chunkId == "MORB" && !seenOptionalSubchunks.Contains("MORI"))
        {
            throw new InvalidDataException(
                $"MOGP optional dependency mismatch at 0x{payloadStart - 8:X}: 'MORB' requires prior 'MORI'. Flags=0x{flags:X}.");
        }
    }

    private static void EnsureAllFlaggedOptionalMogpSubchunksPresent(HashSet<string> seenOptionalSubchunks, uint flags)
    {
        var expectedOptional = BuildAllowedOptionalMogpSubchunks(flags);
        if (expectedOptional.Count == 0)
            return;

        var missing = new List<string>();
        foreach (var token in expectedOptional)
        {
            if (!seenOptionalSubchunks.Contains(token))
                missing.Add(token);
        }

        if (missing.Count > 0)
        {
            throw new InvalidDataException(
                $"MOGP missing flagged optional subchunks ({string.Join(", ", missing)}). Flags=0x{flags:X}.");
        }
    }

    private void WriteWmoV14(WmoV17Data data, string outputPath)
    {
        using var fs = File.Create(outputPath);
        WriteWmoV14ToStream(data, fs);
    }

    private void WriteWmoV14ToStream(WmoV17Data data, Stream output)
    {
        using var writer = new BinaryWriter(output, Encoding.ASCII, leaveOpen: true);

        // MVER (version 14)
        WriteChunk(writer, "MVER", w => w.Write((uint)14));

        // MOMO container - contains all other chunks
        using var momoStream = new MemoryStream();
        using var momoWriter = new BinaryWriter(momoStream);

        // MOHD
        WriteChunk(momoWriter, "MOHD", w =>
        {
            w.Write(data.MaterialCount);
            w.Write((uint)data.Groups.Count);
            w.Write(data.PortalCount);
            w.Write(data.LightCount);
            w.Write(data.DoodadNameCount);
            w.Write(data.DoodadDefCount);
            w.Write(data.DoodadSetCount);
            w.Write(data.AmbientColor);
            w.Write(data.WmoId);
            WriteVector3(w, data.BoundingBox1);
            WriteVector3(w, data.BoundingBox2);
            // v14 MOHD is smaller - no flags field
        });

        // MOTX
        if (data.TextureNames.Count > 0)
            WriteChunk(momoWriter, "MOTX", w => WriteStringTable(w, data.TextureNames));

        // MOMT
        if (data.Materials.Count > 0)
            WriteChunk(momoWriter, "MOMT", w => WriteMaterials(w, data.Materials));

        // MOGN
        if (data.GroupNames.Count > 0)
            WriteChunk(momoWriter, "MOGN", w => WriteStringTable(w, data.GroupNames));

        // MOGI
        if (data.GroupInfos.Count > 0)
            WriteChunk(momoWriter, "MOGI", w => WriteGroupInfos(w, data.GroupInfos));

        // MOSB
        if (!string.IsNullOrEmpty(data.SkyboxName))
            WriteChunk(momoWriter, "MOSB", w => WriteNullTermString(w, data.SkyboxName));

        // MOPV
        if (data.PortalVertices.Length > 0)
            WriteChunk(momoWriter, "MOPV", w => WriteVector3Array(w, data.PortalVertices));

        // MOPT
        if (data.PortalInfos.Count > 0)
            WriteChunk(momoWriter, "MOPT", w => WritePortalInfos(w, data.PortalInfos));

        // MOPR
        if (data.PortalRefs.Count > 0)
            WriteChunk(momoWriter, "MOPR", w => WritePortalRefs(w, data.PortalRefs));

        // MOLT
        if (data.Lights.Count > 0)
            WriteChunk(momoWriter, "MOLT", w => WriteLights(w, data.Lights));

        // MODS
        if (data.DoodadSets.Count > 0)
            WriteChunk(momoWriter, "MODS", w => WriteDoodadSets(w, data.DoodadSets));

        // MODN
        if (data.DoodadNames.Count > 0)
            WriteChunk(momoWriter, "MODN", w => WriteStringTable(w, data.DoodadNames));

        // MODD
        if (data.DoodadDefs.Count > 0)
            WriteChunk(momoWriter, "MODD", w => WriteDoodadDefs(w, data.DoodadDefs));

        // MFOG
        if (data.Fogs.Count > 0)
            WriteChunk(momoWriter, "MFOG", w => WriteFogs(w, data.Fogs));

        // MOGP groups (embedded in MOMO for v14)
        foreach (var group in data.Groups)
        {
            WriteGroupChunk(momoWriter, group);
        }

        // Write MOMO container
        var momoData = momoStream.ToArray();
        WriteChunk(writer, "MOMO", w => w.Write(momoData));
    }

    private void WriteGroupChunk(BinaryWriter writer, WmoV17GroupData group)
    {
        WriteChunk(writer, "MOGP", w =>
        {
            // MOGP header (v14 format - 56 bytes)
            w.Write(group.GroupNameOfs);
            w.Write(group.DescriptiveNameOfs);
            w.Write(group.Flags);
            WriteVector3(w, group.BoundingBox1);
            WriteVector3(w, group.BoundingBox2);
            w.Write(group.PortalStart);
            w.Write(group.PortalCount);
            w.Write(group.TransBatchCount);
            w.Write(group.IntBatchCount);
            w.Write(group.ExtBatchCount);
            w.Write((ushort)0); // padding
            w.Write(group.FogIndices);
            w.Write(group.GroupLiquid);
            w.Write(group.UniqueId);

            // Subchunks
            if (group.MaterialInfo?.Length > 0)
                WriteSubChunk(w, "MOPY", group.MaterialInfo);
            if (group.Indices?.Length > 0)
                WriteSubChunk(w, "MOVI", sw => WriteUInt16Array(sw, group.Indices));
            if (group.Vertices?.Length > 0)
                WriteSubChunk(w, "MOVT", sw => WriteVector3Array(sw, group.Vertices));
            if (group.Normals?.Length > 0)
                WriteSubChunk(w, "MONR", sw => WriteVector3Array(sw, group.Normals));
            if (group.TexCoords?.Length > 0)
                WriteSubChunk(w, "MOTV", sw => WriteVector2Array(sw, group.TexCoords));
            if (group.Batches?.Length > 0)
                WriteSubChunk(w, "MOBA", sw => WriteBatches(sw, group.Batches));
            if (group.LightRefs?.Length > 0)
                WriteSubChunk(w, "MOLR", sw => WriteUInt16Array(sw, group.LightRefs));
            if (group.DoodadRefs?.Length > 0)
                WriteSubChunk(w, "MODR", sw => WriteUInt16Array(sw, group.DoodadRefs));
            if (group.BspNodes?.Length > 0)
                WriteSubChunk(w, "MOBN", group.BspNodes);
            if (group.BspFaceIndices?.Length > 0)
                WriteSubChunk(w, "MOBR", sw => WriteUInt16Array(sw, group.BspFaceIndices));
            if (group.VertexColors?.Length > 0)
                WriteSubChunk(w, "MOCV", group.VertexColors);
            if (group.LiquidData?.Length > 0)
                WriteSubChunk(w, "MLIQ", group.LiquidData);
        });
    }

    #region Read Helpers

    private static Vector3 ReadVector3(BinaryReader r) =>
        new(r.ReadSingle(), r.ReadSingle(), r.ReadSingle());

    private static Vector2 ReadVector2(BinaryReader r) =>
        new(r.ReadSingle(), r.ReadSingle());

    private static List<string> ReadStringTable(BinaryReader r, uint size)
    {
        var result = new List<string>();
        long end = r.BaseStream.Position + size;
        var sb = new StringBuilder();
        
        while (r.BaseStream.Position < end)
        {
            byte b = r.ReadByte();
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

    private static string ReadNullTermString(BinaryReader r, uint size)
    {
        var bytes = r.ReadBytes((int)size);
        int len = Array.IndexOf(bytes, (byte)0);
        if (len < 0) len = bytes.Length;
        return Encoding.ASCII.GetString(bytes, 0, len);
    }

    private static Vector3[] ReadVector3Array(BinaryReader r, uint size)
    {
        int count = (int)(size / 12);
        var result = new Vector3[count];
        for (int i = 0; i < count; i++)
            result[i] = ReadVector3(r);
        return result;
    }

    private static Vector2[] ReadVector2Array(BinaryReader r, uint size)
    {
        int count = (int)(size / 8);
        var result = new Vector2[count];
        for (int i = 0; i < count; i++)
            result[i] = ReadVector2(r);
        return result;
    }

    private static ushort[] ReadUInt16Array(BinaryReader r, uint size)
    {
        int count = (int)(size / 2);
        var result = new ushort[count];
        for (int i = 0; i < count; i++)
            result[i] = r.ReadUInt16();
        return result;
    }

    private static List<WmoMaterial> ReadMaterials(BinaryReader r, uint size)
    {
        int count = (int)(size / 64); // v17 material size
        var result = new List<WmoMaterial>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoMaterial
            {
                Flags = r.ReadUInt32(),
                Shader = r.ReadUInt32(),
                BlendMode = r.ReadUInt32(),
                Texture1 = r.ReadUInt32(),
                SidnColor = r.ReadUInt32(),
                FrameSidnColor = r.ReadUInt32(),
                Texture2 = r.ReadUInt32(),
                DiffColor = r.ReadUInt32(),
                GroundType = r.ReadUInt32(),
                Texture3 = r.ReadUInt32(),
                Color3 = r.ReadUInt32(),
                Flags3 = r.ReadUInt32(),
                RuntimeData = new uint[4]
            });
            for (int j = 0; j < 4; j++)
                result[i].RuntimeData[j] = r.ReadUInt32();
        }
        return result;
    }

    private static List<WmoGroupInfo> ReadGroupInfos(BinaryReader r, uint size)
    {
        int count = (int)(size / 32);
        var result = new List<WmoGroupInfo>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoGroupInfo
            {
                Flags = r.ReadUInt32(),
                BoundingBox1 = ReadVector3(r),
                BoundingBox2 = ReadVector3(r),
                NameOfs = r.ReadInt32()
            });
        }
        return result;
    }

    private static List<WmoPortalInfo> ReadPortalInfos(BinaryReader r, uint size)
    {
        int count = (int)(size / 20);
        var result = new List<WmoPortalInfo>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoPortalInfo
            {
                StartVertex = r.ReadUInt16(),
                VertexCount = r.ReadUInt16(),
                Normal = ReadVector3(r),
                Distance = r.ReadSingle()
            });
        }
        return result;
    }

    private static List<WmoPortalRef> ReadPortalRefs(BinaryReader r, uint size)
    {
        int count = (int)(size / 8);
        var result = new List<WmoPortalRef>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoPortalRef
            {
                PortalIndex = r.ReadUInt16(),
                GroupIndex = r.ReadUInt16(),
                Side = r.ReadInt16(),
                Padding = r.ReadUInt16()
            });
        }
        return result;
    }

    private static List<WmoLight> ReadLights(BinaryReader r, uint size)
    {
        int count = (int)(size / 48);
        var result = new List<WmoLight>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoLight
            {
                Type = r.ReadByte(),
                UseAtten = r.ReadByte(),
                Padding = r.ReadUInt16(),
                Color = r.ReadUInt32(),
                Position = ReadVector3(r),
                Intensity = r.ReadSingle(),
                AttenStart = r.ReadSingle(),
                AttenEnd = r.ReadSingle(),
                Unk = new float[4]
            });
            for (int j = 0; j < 4; j++)
                result[i].Unk[j] = r.ReadSingle();
        }
        return result;
    }

    private static List<WmoDoodadSet> ReadDoodadSets(BinaryReader r, uint size)
    {
        int count = (int)(size / 32);
        var result = new List<WmoDoodadSet>(count);
        for (int i = 0; i < count; i++)
        {
            var name = new byte[20];
            r.Read(name, 0, 20);
            result.Add(new WmoDoodadSet
            {
                Name = Encoding.ASCII.GetString(name).TrimEnd('\0'),
                StartIndex = r.ReadUInt32(),
                Count = r.ReadUInt32(),
                Padding = r.ReadUInt32()
            });
        }
        return result;
    }

    private static List<WmoDoodadDef> ReadDoodadDefs(BinaryReader r, uint size)
    {
        int count = (int)(size / 40);
        var result = new List<WmoDoodadDef>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoDoodadDef
            {
                NameOfs = r.ReadUInt32(),
                Position = ReadVector3(r),
                Rotation = new float[4]
            });
            for (int j = 0; j < 4; j++)
                result[i].Rotation[j] = r.ReadSingle();
            result[i].Scale = r.ReadSingle();
            result[i].Color = r.ReadUInt32();
        }
        return result;
    }

    private static List<WmoFog> ReadFogs(BinaryReader r, uint size)
    {
        int count = (int)(size / 48);
        var result = new List<WmoFog>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add(new WmoFog
            {
                Flags = r.ReadUInt32(),
                Position = ReadVector3(r),
                SmallRadius = r.ReadSingle(),
                LargeRadius = r.ReadSingle(),
                FogEnd = r.ReadSingle(),
                FogStartScalar = r.ReadSingle(),
                Color = r.ReadUInt32(),
                Unk = new float[2]
            });
            result[i].Unk[0] = r.ReadSingle();
            result[i].Unk[1] = r.ReadSingle();
        }
        return result;
    }

    private static WmoBatch[] ReadBatches(BinaryReader r, uint size)
    {
        int count = (int)(size / 24);
        var result = new WmoBatch[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = new WmoBatch
            {
                BoundingBox = new short[6]
            };
            for (int j = 0; j < 6; j++)
                result[i].BoundingBox[j] = r.ReadInt16();
            result[i].StartIndex = r.ReadUInt32();
            result[i].IndexCount = r.ReadUInt16();
            result[i].StartVertex = r.ReadUInt16();
            result[i].EndVertex = r.ReadUInt16();
            result[i].Flags = r.ReadByte();
            result[i].MaterialId = r.ReadByte();
        }
        return result;
    }

    #endregion

    #region Write Helpers

    private static void WriteChunk(BinaryWriter w, string id, Action<BinaryWriter> writeData)
    {
        var reversed = new string(id.Reverse().ToArray());
        w.Write(Encoding.ASCII.GetBytes(reversed));
        
        using var dataStream = new MemoryStream();
        using var dataWriter = new BinaryWriter(dataStream);
        writeData(dataWriter);
        
        var data = dataStream.ToArray();
        w.Write((uint)data.Length);
        w.Write(data);
    }

    private static void WriteSubChunk(BinaryWriter w, string id, byte[] data)
    {
        var reversed = new string(id.Reverse().ToArray());
        w.Write(Encoding.ASCII.GetBytes(reversed));
        w.Write((uint)data.Length);
        w.Write(data);
    }

    private static void WriteSubChunk(BinaryWriter w, string id, Action<BinaryWriter> writeData)
    {
        var reversed = new string(id.Reverse().ToArray());
        w.Write(Encoding.ASCII.GetBytes(reversed));
        
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

    private static void WriteStringTable(BinaryWriter w, List<string> strings)
    {
        foreach (var s in strings)
        {
            w.Write(Encoding.ASCII.GetBytes(s));
            w.Write((byte)0);
        }
    }

    private static void WriteNullTermString(BinaryWriter w, string s)
    {
        w.Write(Encoding.ASCII.GetBytes(s));
        w.Write((byte)0);
    }

    private static void WriteVector3Array(BinaryWriter w, Vector3[] arr)
    {
        foreach (var v in arr)
            WriteVector3(w, v);
    }

    private static void WriteVector2Array(BinaryWriter w, Vector2[] arr)
    {
        foreach (var v in arr)
            WriteVector2(w, v);
    }

    private static void WriteUInt16Array(BinaryWriter w, ushort[] arr)
    {
        foreach (var v in arr)
            w.Write(v);
    }

    private static void WriteMaterials(BinaryWriter w, List<WmoMaterial> materials)
    {
        foreach (var m in materials)
        {
            w.Write(m.Flags);
            w.Write(m.Shader);
            w.Write(m.BlendMode);
            w.Write(m.Texture1);
            w.Write(m.SidnColor);
            w.Write(m.FrameSidnColor);
            w.Write(m.Texture2);
            w.Write(m.DiffColor);
            w.Write(m.GroundType);
            w.Write(m.Texture3);
            w.Write(m.Color3);
            w.Write(m.Flags3);
            // v14 materials are 48 bytes (no runtime data)
        }
    }

    private static void WriteGroupInfos(BinaryWriter w, List<WmoGroupInfo> infos)
    {
        foreach (var g in infos)
        {
            w.Write(g.Flags);
            WriteVector3(w, g.BoundingBox1);
            WriteVector3(w, g.BoundingBox2);
            w.Write(g.NameOfs);
        }
    }

    private static void WritePortalInfos(BinaryWriter w, List<WmoPortalInfo> portals)
    {
        foreach (var p in portals)
        {
            w.Write(p.StartVertex);
            w.Write(p.VertexCount);
            WriteVector3(w, p.Normal);
            w.Write(p.Distance);
        }
    }

    private static void WritePortalRefs(BinaryWriter w, List<WmoPortalRef> refs)
    {
        foreach (var r in refs)
        {
            w.Write(r.PortalIndex);
            w.Write(r.GroupIndex);
            w.Write(r.Side);
            w.Write(r.Padding);
        }
    }

    private static void WriteLights(BinaryWriter w, List<WmoLight> lights)
    {
        foreach (var l in lights)
        {
            w.Write(l.Type);
            w.Write(l.UseAtten);
            w.Write(l.Padding);
            w.Write(l.Color);
            WriteVector3(w, l.Position);
            w.Write(l.Intensity);
            w.Write(l.AttenStart);
            w.Write(l.AttenEnd);
            foreach (var u in l.Unk)
                w.Write(u);
        }
    }

    private static void WriteDoodadSets(BinaryWriter w, List<WmoDoodadSet> sets)
    {
        foreach (var s in sets)
        {
            var nameBytes = new byte[20];
            var srcBytes = Encoding.ASCII.GetBytes(s.Name);
            Buffer.BlockCopy(srcBytes, 0, nameBytes, 0, Math.Min(srcBytes.Length, 20));
            w.Write(nameBytes);
            w.Write(s.StartIndex);
            w.Write(s.Count);
            w.Write(s.Padding);
        }
    }

    private static void WriteDoodadDefs(BinaryWriter w, List<WmoDoodadDef> defs)
    {
        foreach (var d in defs)
        {
            w.Write(d.NameOfs);
            WriteVector3(w, d.Position);
            foreach (var r in d.Rotation)
                w.Write(r);
            w.Write(d.Scale);
            w.Write(d.Color);
        }
    }

    private static void WriteFogs(BinaryWriter w, List<WmoFog> fogs)
    {
        foreach (var f in fogs)
        {
            w.Write(f.Flags);
            WriteVector3(w, f.Position);
            w.Write(f.SmallRadius);
            w.Write(f.LargeRadius);
            w.Write(f.FogEnd);
            w.Write(f.FogStartScalar);
            w.Write(f.Color);
            foreach (var u in f.Unk)
                w.Write(u);
        }
    }

    private static void WriteBatches(BinaryWriter w, WmoBatch[] batches)
    {
        foreach (var b in batches)
        {
            foreach (var bb in b.BoundingBox)
                w.Write(bb);
            w.Write(b.StartIndex);
            w.Write(b.IndexCount);
            w.Write(b.StartVertex);
            w.Write(b.EndVertex);
            w.Write(b.Flags);
            w.Write(b.MaterialId);
        }
    }

    #endregion

    #region Data Classes

    private class WmoV17Data
    {
        public uint Version;
        public uint MaterialCount;
        public uint GroupCount;
        public uint PortalCount;
        public uint LightCount;
        public uint DoodadNameCount;
        public uint DoodadDefCount;
        public uint DoodadSetCount;
        public uint AmbientColor;
        public uint WmoId;
        public Vector3 BoundingBox1;
        public Vector3 BoundingBox2;
        public uint Flags;

        public List<string> TextureNames = new();
        public byte[] TextureNamesRaw = Array.Empty<byte>(); // Raw MOTX blob for byte-offset resolution
        public List<WmoMaterial> Materials = new();
        public List<string> GroupNames = new();
        public byte[] GroupNamesRaw = Array.Empty<byte>(); // Raw MOGN blob for byte-offset resolution
        public List<WmoGroupInfo> GroupInfos = new();
        public string SkyboxName = "";
        public Vector3[] PortalVertices = Array.Empty<Vector3>();
        public List<WmoPortalInfo> PortalInfos = new();
        public List<WmoPortalRef> PortalRefs = new();
        public byte[] MovvRaw = Array.Empty<byte>();
        public byte[] MovbRaw = Array.Empty<byte>();
        public List<WmoLight> Lights = new();
        public List<WmoDoodadSet> DoodadSets = new();
        public List<string> DoodadNames = new();
        public byte[] DoodadNamesRaw = Array.Empty<byte>(); // Raw MODN blob for byte-offset resolution
        public List<WmoDoodadDef> DoodadDefs = new();
        public List<WmoFog> Fogs = new();
        public byte[] McvpRaw = Array.Empty<byte>();
        public List<WmoV17GroupData> Groups = new();
    }

    private class WmoV17GroupData
    {
        public uint Version;
        public uint GroupNameOfs;
        public uint DescriptiveNameOfs;
        public uint Flags;
        public Vector3 BoundingBox1;
        public Vector3 BoundingBox2;
        public ushort PortalStart;
        public ushort PortalCount;
        public ushort TransBatchCount;
        public ushort IntBatchCount;
        public ushort ExtBatchCount;
        public byte[] FogIndices = new byte[4];
        public uint GroupLiquid;
        public uint UniqueId;
        public uint Flags2;

        public byte[]? MaterialInfo;
        public ushort[]? Indices;
        public Vector3[]? Vertices;
        public Vector3[]? Normals;
        public Vector2[]? TexCoords;
        public WmoBatch[]? Batches;
        public ushort[]? LightRefs;
        public ushort[]? DoodadRefs;
        public byte[]? BspNodes;
        public ushort[]? BspFaceIndices;
        public byte[]? VertexColors;
        public byte[]? LiquidData;
        public byte[]? MpbvData;
        public byte[]? MpbpData;
        public byte[]? MpbiData;
        public byte[]? MpbgData;
        public byte[]? MoriData;
        public byte[]? MorbData;
    }

    private class WmoMaterial
    {
        public uint Flags;
        public uint Shader;
        public uint BlendMode;
        public uint Texture1;
        public uint SidnColor;
        public uint FrameSidnColor;
        public uint Texture2;
        public uint DiffColor;
        public uint GroundType;
        public uint Texture3;
        public uint Color3;
        public uint Flags3;
        public uint[] RuntimeData = new uint[4];
    }

    private class WmoGroupInfo
    {
        public uint Flags;
        public Vector3 BoundingBox1;
        public Vector3 BoundingBox2;
        public int NameOfs;
    }

    private class WmoPortalInfo
    {
        public ushort StartVertex;
        public ushort VertexCount;
        public Vector3 Normal;
        public float Distance;
    }

    private class WmoPortalRef
    {
        public ushort PortalIndex;
        public ushort GroupIndex;
        public short Side;
        public ushort Padding;
    }

    private class WmoLight
    {
        public byte Type;
        public byte UseAtten;
        public ushort Padding;
        public uint Color;
        public Vector3 Position;
        public float Intensity;
        public float AttenStart;
        public float AttenEnd;
        public float[] Unk = new float[4];
    }

    private class WmoDoodadSet
    {
        public string Name = "";
        public uint StartIndex;
        public uint Count;
        public uint Padding;
    }

    private class WmoDoodadDef
    {
        public uint NameOfs;
        public uint Flags;
        public Vector3 Position;
        public float[] Rotation = new float[4];
        public float Scale;
        public uint Color;
    }

    private class WmoFog
    {
        public uint Flags;
        public Vector3 Position;
        public float SmallRadius;
        public float LargeRadius;
        public float FogEnd;
        public float FogStartScalar;
        public uint Color;
        public float[] Unk = new float[2];
    }

    private class WmoBatch
    {
        public short[] BoundingBox = new short[6];
        public uint StartIndex;
        public ushort IndexCount;
        public ushort StartVertex;
        public ushort EndVertex;
        public byte Flags;
        public byte MaterialId;
    }

    #endregion
}
